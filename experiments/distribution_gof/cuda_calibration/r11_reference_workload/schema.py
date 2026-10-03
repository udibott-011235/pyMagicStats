"""Strict schema/integrity loading; incomplete artifacts are diagnostic only."""
from __future__ import annotations

import base64

from .builder import ordered_payload_digest
from .codec import (SamplePayload, canonical_json, hex_digest, integer, keys,
                    require, sha256, strict_json, text_identity)
from .contract import (ALPHA, B_R11, NB_RETRY_CAP, SCIENTIFIC_SHA, SCIENTIFIC_TREE,
                       SCHEMA_VERSION, TOTAL_RECORDS)
from .source import SourceSurface, projection

WORKLOAD_KEYS = (
    "schema_version", "source_kind", "alpha", "B_R11", "nb_retry_cap",
    "source_manifest_base64", "source_manifest_hash", "source_projection_hash",
    "source_projection_bytes", "scientific_sha", "scientific_tree",
    "builder_binding", "environment", "completion_status", "failure", "outers",
    "ordered_workload_payload_digest",
)
OUTER_KEYS = (
    "outer_identity", "cell_id", "raw_outer_index", "observed",
    "reference_fit_parameters", "attempts", "accepted", "attempt_count",
    "accepted_count", "retry_cap", "retry_cap_reached", "completion_status", "failure",
)
ATTEMPT_KEYS = (
    "raw_inner_index", "seed_identity", "sample_digest",
    "canonical_eligibility_status", "reason",
)
ACCEPTED_KEYS = (
    "accepted_ordinal", "identity", "cell_id", "raw_outer_index",
    "raw_inner_index", "seed_identity", "payload",
)


def validate_binding(binding, *, required=True):
    if binding is None:
        require(not required, "builder SHA/tree binding required before use")
        return
    keys(binding, ("sha", "tree"))
    hex_digest(binding["sha"], 40)
    hex_digest(binding["tree"], 40)


def validate_failure(value):
    if value is not None:
        keys(value, ("stage", "error_type", "message"))
        for field in value:
            require(type(value[field]) is str, "invalid failure state")


def _check_outer(outer, source, *, complete):
    keys(outer, OUTER_KEYS)
    for field in ("outer_identity", "cell_id", "raw_outer_index"):
        require(type(outer[field]) is type(source[field]) and outer[field] == source[field],
                "outer order/identity mismatch")
    family = source["cell_id"].split("|", 1)[0]
    cap = NB_RETRY_CAP if family == "negative_binomial" else B_R11
    require(type(outer["retry_cap"]) is int and outer["retry_cap"] == cap, "retry cap mismatch")
    require(outer["completion_status"] in ("COMPLETE", "FAILED", "INCOMPLETE"),
            "invalid outer completion status")
    validate_failure(outer["failure"])
    require(type(outer["attempts"]) is list and type(outer["accepted"]) is list,
            "invalid record arrays")
    require(integer(outer["attempt_count"]) == len(outer["attempts"]) <= cap,
            "attempt count mismatch")
    require(integer(outer["accepted_count"]) == len(outer["accepted"]) <= B_R11,
            "accepted count mismatch")
    require(type(outer["retry_cap_reached"]) is bool
            and outer["retry_cap_reached"] == (len(outer["attempts"]) == cap),
            "retry cap status mismatch")
    if complete or outer["completion_status"] == "COMPLETE":
        require(outer["completion_status"] == "COMPLETE" and outer["failure"] is None
                and len(outer["accepted"]) == B_R11, "incomplete outer")
    observed = outer["observed"]
    if observed is None:
        require(not complete and not outer["accepted"] and not outer["attempts"],
                "observed payload missing")
    else:
        keys(observed, ("identity", "seed_identity", "payload"))
        require(observed["identity"] == source["observed_identity"], "observed identity mismatch")
        integer(observed["seed_identity"])
        SamplePayload.from_serialized(observed["payload"])
    params = outer["reference_fit_parameters"]
    require(params is None or type(params) is dict, "invalid reference parameters")
    require(not complete or params is not None, "observed reference fit missing")
    eligible = []
    for raw, attempt in enumerate(outer["attempts"]):
        keys(attempt, ATTEMPT_KEYS)
        require(integer(attempt["raw_inner_index"]) == raw, "prospective attempt order mismatch")
        status = attempt["canonical_eligibility_status"]
        require(status in ("ELIGIBLE", "INELIGIBLE", "FAILED"), "invalid eligibility status")
        if status == "FAILED":
            require(not complete and raw == len(outer["attempts"]) - 1,
                    "failed attempt cannot enter complete workload")
            if attempt["seed_identity"] is not None:
                integer(attempt["seed_identity"])
            require(attempt["sample_digest"] is None
                    or hex_digest(attempt["sample_digest"]), "invalid failed sample digest")
        else:
            integer(attempt["seed_identity"])
            hex_digest(attempt["sample_digest"])
            if status == "INELIGIBLE":
                require(family == "negative_binomial" and type(attempt["reason"]) is str
                        and attempt["reason"].startswith("NB_NOT_ASSESSED:"),
                        "non-canonical ineligibility")
            else:
                require(attempt["reason"] is None, "eligible reason must be null")
                eligible.append(attempt)
    require(len(eligible) == len(outer["accepted"]), "eligibility/accepted accounting mismatch")
    identities = set()
    for ordinal, (record, attempt) in enumerate(zip(outer["accepted"], eligible)):
        keys(record, ACCEPTED_KEYS)
        require(integer(record["accepted_ordinal"]) == ordinal, "accepted ordinal mismatch")
        raw = integer(record["raw_inner_index"])
        require(raw == attempt["raw_inner_index"], "accepted order/raw index mismatch")
        identity = f"{source['outer_identity']}|raw_inner={raw}"
        require(record["identity"] not in identities, "duplicate accepted identity")
        identities.add(record["identity"])
        require(record["identity"] == identity and record["cell_id"] == source["cell_id"]
                and type(record["raw_outer_index"]) is int
                and record["raw_outer_index"] == source["raw_outer_index"], "accepted identity mismatch")
        require(integer(record["seed_identity"]) == attempt["seed_identity"], "accepted seed mismatch")
        payload = SamplePayload.from_serialized(record["payload"])
        require(payload.sample_digest == attempt["sample_digest"], "accepted sample digest mismatch")
    if complete:
        require(bool(outer["attempts"]) and outer["attempts"][-1]["canonical_eligibility_status"] == "ELIGIBLE",
                "attempts continued after completion")


def load_artifact(data: bytes, *, allow_synthetic=False, require_complete=True,
                  require_binding=True):
    """Integrity/schema loader, not a scientific acceptance certificate.

    Use oracle.load_for_use for the mandatory historical R4-prefix gate.
    Diagnostic loading requires require_complete=False and never permits use.
    """
    document = strict_json(data, canonical=True)
    keys(document, ("workload", "workload_digest"))
    workload = document["workload"]
    keys(workload, WORKLOAD_KEYS)
    hex_digest(document["workload_digest"])
    require(sha256(canonical_json(workload)) == document["workload_digest"], "workload digest mismatch")
    require(workload["schema_version"] == SCHEMA_VERSION, "schema version mismatch")
    require(type(workload["alpha"]) is float and workload["alpha"] == ALPHA, "alpha mismatch")
    require(type(workload["B_R11"]) is int and workload["B_R11"] == B_R11, "B_R11 mismatch")
    require(type(workload["nb_retry_cap"]) is int and workload["nb_retry_cap"] == NB_RETRY_CAP,
            "NB retry cap mismatch")
    require(workload["scientific_sha"] == SCIENTIFIC_SHA
            and workload["scientific_tree"] == SCIENTIFIC_TREE, "scientific identity mismatch")
    validate_binding(workload["builder_binding"], required=require_binding)
    keys(workload["environment"], ("python", "numpy", "scipy"))
    for value in workload["environment"].values():
        text_identity(value)
    kind = workload["source_kind"]
    require(kind == "FROZEN_R11" or (allow_synthetic and kind == "SYNTHETIC_TEST"),
            "synthetic workload prohibited for scientific use")
    if kind == "FROZEN_R11":
        require(all(value != "UNAVAILABLE" for value in workload["environment"].values()),
                "reference environment unavailable")
    encoded = workload["source_manifest_base64"]
    require(type(encoded) is str, "invalid source bytes")
    try:
        source_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        from .codec import ContractError
        raise ContractError("invalid source base64") from exc
    require(base64.b64encode(source_bytes).decode("ascii") == encoded, "non-canonical source base64")
    rows = SourceSurface(kind, source_bytes).rows()
    projected = canonical_json(projection(rows))
    require(workload["source_manifest_hash"] == sha256(source_bytes)
            and workload["source_projection_hash"] == sha256(projected)
            and type(workload["source_projection_bytes"]) is int
            and workload["source_projection_bytes"] == len(projected), "source binding mismatch")
    require(workload["completion_status"] in ("COMPLETE", "INCOMPLETE", "FAILED"),
            "invalid completion status")
    validate_failure(workload["failure"])
    complete = workload["completion_status"] == "COMPLETE"
    require(not require_complete or complete, "incomplete workload rejected")
    require(not complete or workload["failure"] is None, "complete workload has failure state")
    require(complete or workload["failure"] is not None, "missing incomplete failure state")
    require(type(workload["outers"]) is list and len(workload["outers"]) <= len(rows),
            "invalid outer array")
    require(not complete or len(workload["outers"]) == len(rows), "missing source outer")
    for outer, row in zip(workload["outers"], rows):
        _check_outer(outer, row, complete=complete)
    if complete and kind == "FROZEN_R11":
        require(sum(1 + outer["accepted_count"] for outer in workload["outers"]) == TOTAL_RECORDS,
                "total scientific record count mismatch")
    hex_digest(workload["ordered_workload_payload_digest"])
    require(workload["ordered_workload_payload_digest"] == ordered_payload_digest(workload["outers"]),
            "ordered workload payload digest mismatch")
    return workload
