"""Prospective CPU construction with internal canonical wiring, with no CLI."""
from __future__ import annotations

import base64
import importlib.metadata
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .codec import (ContractError, SamplePayload, canonical_json, integer,
                    require, sha256, strict_json)
from .contract import (ALPHA, B_R11, NB_RETRY_CAP, NAMESPACE, SCIENTIFIC_SHA,
                       SCIENTIFIC_TREE, SCHEMA_VERSION)
from .source import SourceSurface, projection


@dataclass(frozen=True)
class CPUAdapters:
    """Explicit fake bindings for SYNTHETIC_TEST only.

    observed(row, namespace) -> (NumPy array, canonical integer seed)
    reference_fit(family, array) -> existing CPU_REFERENCE fit dictionary
    derive_seed(namespace, cell_id, outer, purpose, inner) -> canonical int
    generate(row, fitted_parameters, seed) -> canonical NumPy array

    FROZEN_R11 obtains its concrete canonical wiring internally and rejects
    this injection surface, including caller-supplied canonical adapters.
    """
    observed: Callable
    reference_fit: Callable
    derive_seed: Callable
    generate: Callable
    canonical_error_type: type[Exception]
    engine: str = "CPU_REFERENCE"

    def verify(self):
        require(self.engine == "CPU_REFERENCE", "CPU reference adapters required")
        require(isinstance(self.canonical_error_type, type)
                and issubclass(self.canonical_error_type, Exception)
                and self.canonical_error_type not in (Exception, BaseException),
                "specific canonical error type required")
        require(self.canonical_error_type.__module__ != "builtins",
                "canonical error type must not be a generic built-in exception")
        for method in (self.observed, self.reference_fit, self.derive_seed, self.generate):
            require(callable(method), "missing CPU adapter")


def environment():
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "UNAVAILABLE"
    return {"python": platform.python_version(), "numpy": version("numpy"),
            "scipy": version("scipy")}


def ordered_payload_digest(outers):
    """Ordered, length-delimited metadata + exact raw bytes; observed first."""
    import hashlib
    digest = hashlib.sha256()
    for outer in outers:
        records = ([outer["observed"]] if outer["observed"] is not None else [])
        records += outer["accepted"]
        for record in records:
            payload = SamplePayload.from_serialized(record["payload"])
            meta = canonical_json({key: value for key, value in record.items() if key != "payload"})
            interpretation = canonical_json({"dtype_str": payload.dtype_str,
                                             "shape": list(payload.shape)})
            for part in (meta, interpretation, payload.raw_bytes):
                digest.update(len(part).to_bytes(8, "big"))
                digest.update(part)
    return digest.hexdigest()


class BuildFailure(ContractError):
    """Immutable available evidence from the single failed invocation."""
    def __init__(self, artifact_bytes: bytes):
        self.artifact_bytes = artifact_bytes
        super().__init__("reference construction failed; no automatic rerun/resume")


def artifact_document(workload):
    body = {**workload, "ordered_workload_payload_digest": ordered_payload_digest(workload["outers"])}
    return canonical_json({"workload": body, "workload_digest": sha256(canonical_json(body))})


class ReferenceWorkloadBuilder:
    """One-shot machinery. Real invocation requires separate human authority."""
    def __init__(self):
        self._consumed = False

    def build(self, source: SourceSurface, adapters: CPUAdapters | None = None, *, builder_binding=None):
        require(not self._consumed, "builder consumed; automatic rerun/resume prohibited")
        self._consumed = True
        require(type(source) is SourceSurface, "exact SourceSurface required")
        if source.kind == "FROZEN_R11":
            require(adapters is None, "FROZEN_R11 prohibits external adapter injection")
            from .canonical_adapter import CanonicalCPUAdapters
            adapters = CanonicalCPUAdapters()
        else:
            require(source.kind == "SYNTHETIC_TEST", "unknown source kind")
            require(type(adapters) is CPUAdapters, "SYNTHETIC_TEST requires explicit fake adapters")
        from .schema import validate_binding
        validate_binding(builder_binding, required=False)
        rows = source.rows()
        env = environment()
        if source.kind == "FROZEN_R11":
            require(env["scipy"] != "UNAVAILABLE", "SciPy version unavailable")
            from .binding import binding_from_git
            actual_binding = binding_from_git(Path(__file__).resolve().parents[4])
            require(builder_binding is None or builder_binding == actual_binding,
                    "builder binding does not match executing canonical builder")
            builder_binding = actual_binding
        adapters.verify()
        projected = canonical_json(projection(rows))
        workload = {
            "schema_version": SCHEMA_VERSION, "source_kind": source.kind,
            "alpha": ALPHA, "B_R11": B_R11, "nb_retry_cap": NB_RETRY_CAP,
            "source_manifest_base64": base64.b64encode(source.manifest_bytes).decode("ascii"),
            "source_manifest_hash": sha256(source.manifest_bytes),
            "source_projection_hash": sha256(projected), "source_projection_bytes": len(projected),
            "scientific_sha": SCIENTIFIC_SHA, "scientific_tree": SCIENTIFIC_TREE,
            "builder_binding": builder_binding, "environment": env,
            "completion_status": "INCOMPLETE", "failure": None, "outers": [],
        }
        stage = "observed"
        current = None
        try:
            for row in rows:
                family = row["cell_id"].split("|", 1)[0]
                require(family in ("negative_binomial", "gamma", "exponential"),
                        "unsupported canonical family")
                cap = NB_RETRY_CAP if family == "negative_binomial" else B_R11
                current = {"outer_identity": row["outer_identity"], "cell_id": row["cell_id"],
                           "raw_outer_index": row["raw_outer_index"], "observed": None,
                           "reference_fit_parameters": None, "attempts": [], "accepted": [],
                           "attempt_count": 0, "accepted_count": 0, "retry_cap": cap,
                           "retry_cap_reached": False, "completion_status": "INCOMPLETE",
                           "failure": None}
                workload["outers"].append(current)
                stage = "observed"
                observed, observed_seed = adapters.observed(row, NAMESPACE)
                integer(observed_seed)
                frozen = SamplePayload.from_sample(observed)
                current["observed"] = {"identity": row["observed_identity"],
                                       "seed_identity": observed_seed, "payload": frozen.serialize()}
                stage = "observed_reference_fit"
                fit = adapters.reference_fit(family, frozen.deserialize())
                require(type(fit) is dict and fit.get("engine") == "CPU_REFERENCE"
                        and type(fit.get("parameters")) is dict,
                        "non-CPU reference fit")
                parameters = strict_json(canonical_json(fit["parameters"]))
                current["reference_fit_parameters"] = parameters
                for raw in range(cap):
                    attempt = {"raw_inner_index": raw, "seed_identity": None,
                               "sample_digest": None, "canonical_eligibility_status": "FAILED",
                               "reason": None}
                    current["attempts"].append(attempt)
                    current["attempt_count"] += 1
                    current["retry_cap_reached"] = current["attempt_count"] == cap
                    stage = "derive_seed"
                    seed = adapters.derive_seed(NAMESPACE, row["cell_id"], row["raw_outer_index"],
                                                "inner_bootstrap", raw)
                    integer(seed)
                    attempt["seed_identity"] = seed
                    stage = "generate"
                    sample = adapters.generate(row, strict_json(canonical_json(parameters)), seed)
                    stage = "canonicalize"
                    payload = SamplePayload.from_sample(sample)
                    attempt["sample_digest"] = payload.sample_digest
                    stage = "bootstrap_reference_fit"
                    try:
                        result = adapters.reference_fit(family, payload.deserialize())
                        require(type(result) is dict and result.get("engine") == "CPU_REFERENCE"
                                and type(result.get("parameters")) is dict,
                                "non-CPU eligibility result")
                    except adapters.canonical_error_type as exc:
                        if family != "negative_binomial" or not str(exc).startswith("NB_NOT_ASSESSED:"):
                            raise
                        attempt["canonical_eligibility_status"] = "INELIGIBLE"
                        attempt["reason"] = str(exc)
                        continue
                    attempt["canonical_eligibility_status"] = "ELIGIBLE"
                    identity = f"{row['outer_identity']}|raw_inner={raw}"
                    current["accepted"].append({
                        "accepted_ordinal": len(current["accepted"]),
                        "identity": identity, "cell_id": row["cell_id"],
                        "raw_outer_index": row["raw_outer_index"], "raw_inner_index": raw,
                        "seed_identity": seed, "payload": payload.serialize(),
                    })
                    current["accepted_count"] += 1
                    if current["accepted_count"] == B_R11:
                        break
                stage = "retry_cap"
                require(current["accepted_count"] == B_R11, "NB retry cap exhaustion")
                current["completion_status"] = "COMPLETE"
            workload["completion_status"] = "COMPLETE"
            encoded = artifact_document(workload)
            # Validate the complete software schema before returning frozen bytes.
            from .schema import load_artifact
            load_artifact(encoded, allow_synthetic=source.kind == "SYNTHETIC_TEST",
                          require_binding=False)
            return encoded
        except Exception as exc:
            failure = {"stage": stage, "error_type": type(exc).__name__, "message": str(exc)}
            workload["completion_status"] = "FAILED"
            workload["failure"] = failure
            if current is not None:
                current["completion_status"] = "FAILED"
                current["failure"] = failure
                if current["attempts"] and current["attempts"][-1]["canonical_eligibility_status"] == "FAILED":
                    current["attempts"][-1]["reason"] = str(exc)
            raise BuildFailure(artifact_document(workload)) from exc
