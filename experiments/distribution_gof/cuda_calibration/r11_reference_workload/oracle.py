"""Historical provenance gates only; no statistical comparison harness."""
from __future__ import annotations

import io
import tarfile

from .codec import (ContractError, integer, require, sha256, strict_json)
from .contract import (ARCHIVE_NAME, ARTIFACT_HASHES, CROSSINGS_NAME, PREFIX_COUNT,
                       WORKLOAD_NAME)
from .schema import load_artifact
from .source import SourceSurface


class OracleError(ContractError):
    """Unverified history blocks acceptance and downstream numerical use."""
    R4_RUNTIME_ORACLE_VERIFIED = "NO"
    R11_REFERENCE_WORKLOAD_ACCEPTED = "NO"
    GPU_EXECUTION = "PROHIBITED"


def verify_artifact_digest(name, digest):
    require(name in ARTIFACT_HASHES and digest == ARTIFACT_HASHES[name],
            "R4 artifact/digest association mismatch")


def verify_artifact(name, data):
    require(type(data) is bytes, "historical oracle unavailable")
    verify_artifact_digest(name, sha256(data))


def runtime_records(archive_bytes, *, crossings_bytes=None):
    """Verify archive before reading one regular member; never extract to disk."""
    try:
        verify_artifact(ARCHIVE_NAME, archive_bytes)
        if crossings_bytes is not None:
            verify_artifact(CROSSINGS_NAME, crossings_bytes)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            candidates = [member for member in archive.getmembers()
                          if member.name.rsplit("/", 1)[-1] == WORKLOAD_NAME]
            require(len(candidates) == 1 and candidates[0].isfile(),
                    "missing/ambiguous/nonregular historical workload member")
            stream = archive.extractfile(candidates[0])
            require(stream is not None, "historical oracle unavailable")
            with stream:
                data = stream.read()
        verify_artifact(WORKLOAD_NAME, data)
        lines = data.splitlines()
        require(bool(lines) and all(lines), "malformed historical JSONL")
        return [strict_json(line) for line in lines]
    except Exception as exc:
        raise OracleError(str(exc)) from exc


def compare_prefix(workload, source_rows, historical_records):
    """Pure identity/order/seed/digest comparison, also usable with fixed mocks.

    Passing this helper alone does not verify the runtime archive or accept
    a scientific workload. load_for_use composes every mandatory gate.
    """
    try:
        require(len(workload["outers"]) == len(source_rows), "prefix outer count mismatch")
        require(len(historical_records) == len(source_rows) * (PREFIX_COUNT + 1),
                "historical record count mismatch")
        cursor = 0
        seen = set()
        for outer, source in zip(workload["outers"], source_rows):
            require(outer["outer_identity"] == source["outer_identity"], "prefix outer order mismatch")
            observed = outer["observed"]
            require(observed is not None and observed["identity"] == source["observed_identity"],
                    "observed identity mismatch")
            require(len(outer["accepted"]) >= PREFIX_COUNT, "insufficient eligible prefix")
            selected = [observed] + outer["accepted"][:PREFIX_COUNT]
            for index, actual in enumerate(selected):
                history = historical_records[cursor]
                cursor += 1
                expected = source["observed_identity"] if index == 0 else source["bootstrap_identities"][index - 1]["identity"]
                raw = None if index == 0 else source["bootstrap_identities"][index - 1]["raw_inner_index"]
                require(history["identity"] not in seen, "duplicate historical identity")
                seen.add(history["identity"])
                require(actual["identity"] == history["identity"] == expected, "prefix identity mismatch")
                require(history["cell_id"] == source["cell_id"]
                        and type(history["raw_outer_index"]) is int
                        and history["raw_outer_index"] == source["raw_outer_index"],
                        "historical outer association mismatch")
                require(history["record_type"] == ("observed" if index == 0 else "bootstrap"),
                        "historical record type mismatch")
                require(history["raw_inner_index"] is None if raw is None
                        else type(history["raw_inner_index"]) is int and history["raw_inner_index"] == raw,
                        "historical raw inner order mismatch")
                if raw is not None:
                    require(type(actual["raw_inner_index"]) is int and actual["raw_inner_index"] == raw,
                            "prefix raw inner order mismatch")
                require(integer(actual["seed_identity"]) == integer(history["seed_identity"]),
                        "prefix seed mismatch")
                require(actual["payload"]["sample_digest"] == history["sample_digest"],
                        "prefix sample digest mismatch")
        return True
    except Exception as exc:
        raise OracleError(str(exc)) from exc


def load_for_use(artifact_bytes, manifest_bytes, archive_bytes, *, crossings_bytes=None):
    """Verify complete frozen payloads and R4 provenance before exposing records.

    No CPU/CUDA evaluation or scientific acceptance claim is performed here.
    An absent/mismatched oracle raises OracleError, never an identity-only pass.
    """
    try:
        source = SourceSurface.frozen(manifest_bytes)
        workload = load_artifact(artifact_bytes)
        require(workload["source_manifest_hash"] == sha256(manifest_bytes),
                "historical manifest association mismatch")
        history = runtime_records(archive_bytes, crossings_bytes=crossings_bytes)
        compare_prefix(workload, source.rows(), history)
        return workload
    except Exception as exc:
        raise OracleError(str(exc)) from exc
