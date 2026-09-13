"""Atomic, integrity-checked CP05-B artifacts and resumable checkpoints."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import numpy as np
import scipy

from .accounting import InnerAttempt, OuterResult, ReasonCode
from .manifest import ExperimentManifest, OUTPUT_SCHEMA_VERSION, canonical_json
from .runner import RunOutput, _assign_eligible_indices, owned_raw_indices, run_outer_unit
from .seed_derivation import SEED_DERIVATION_VERSION


CHECKPOINT_SCHEMA_VERSION = "cp05-b-checkpoint-v1"
ARTIFACT_SCHEMA_VERSION = "cp05-b-artifacts-v1"


class ArtifactIntegrityError(RuntimeError):
    reason_code = ReasonCode.ARTIFACT_INTEGRITY_FAILURE.value


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, encoded)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_payload(record: OuterResult) -> dict[str, Any]:
    return record.to_dict()


def _verify_record_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = dict(payload)
    recorded = candidate.pop("output_digest", None)
    actual = hashlib.sha256(canonical_json(candidate).encode("utf-8")).hexdigest()
    if recorded != actual:
        raise ArtifactIntegrityError("outer record output digest mismatch")
    return candidate


def _checkpoint_record_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _verify_record_digest(payload)
    aliases = {
        "float64_value": "statistic_value",
        "absolute_error": "oracle_abs_error",
        "relative_scaled_error": "oracle_relative_scaled_error",
        "remainder_bound": "tail_remainder_bound",
    }
    for alias, source in aliases.items():
        if candidate.get(alias) != candidate.get(source):
            raise ArtifactIntegrityError("outer record oracle field aliases mismatch")
        candidate.pop(alias, None)
    return candidate


def capture_environment(manifest: ExperimentManifest) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "repository": manifest.source_repository,
        "source_sha": manifest.source_sha,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    content = b"".join(
        canonical_json(dict(row)).encode("utf-8") + b"\n" for row in rows
    )
    _atomic_bytes(path, content)


def _write_parquet(path: Path, rows: list[Mapping[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Parquet output requires pandas and a Parquet engine") from exc
    temporary = _temporary_path(path)
    try:
        try:
            pd.DataFrame(rows).to_parquet(temporary, index=False)
        except ImportError as exc:
            raise RuntimeError("Parquet output requires pyarrow or fastparquet") from exc
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_run_artifacts(
    directory: str | Path,
    manifest: ExperimentManifest,
    output: RunOutput,
    *,
    table_format: str = "parquet",
    command: str,
    elapsed_seconds: float,
) -> dict[str, str]:
    """Materialize the canonical artifact surface with deterministic digests."""

    if table_format not in {"parquet", "jsonl"}:
        raise ValueError("table_format must be parquet or jsonl")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    inner_rows = [row.to_dict() for row in output.inner_accounting]
    environment = capture_environment(manifest)
    environment_digest = hashlib.sha256(
        canonical_json(environment).encode("utf-8")
    ).hexdigest()
    outer_rows = []
    for row in output.outer_results:
        record = _record_payload(row)
        record.pop("output_digest")
        record.update(
            source_sha=manifest.source_sha,
            config_digest=manifest.digest,
            environment_digest=environment_digest,
            phase=manifest.phase,
            null_type=manifest.null_type,
            family_id=manifest.family,
            canonical_parameters=dict(manifest.canonical_parameters),
            n=manifest.n,
            statistic_id=manifest.statistic,
            alpha=manifest.alpha,
            B=manifest.B,
            seed_derivation_version=SEED_DERIVATION_VERSION,
        )
        record["output_digest"] = hashlib.sha256(
            canonical_json(record).encode("utf-8")
        ).hexdigest()
        outer_rows.append(record)

    write_json_atomic(target / "manifest.json", manifest.to_dict())
    write_json_atomic(target / "environment.json", environment)
    outer_name = f"outer_results.{table_format}"
    inner_name = f"inner_accounting.{table_format}"
    if table_format == "parquet":
        _write_parquet(target / outer_name, outer_rows)
        _write_parquet(target / inner_name, inner_rows)
    else:
        _write_jsonl(target / outer_name, outer_rows)
        _write_jsonl(target / inner_name, inner_rows)

    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for result in output.outer_results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        reason_counts[result.reason_code] = reason_counts.get(result.reason_code, 0) + 1
    write_json_atomic(
        target / "cell_summary.json",
        {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "claim_status": "NON_CLAIMING",
            "calibration_executed": False,
            "canonical_cell_id": manifest.canonical_cell_id,
            "raw_outer_units": len(output.outer_results),
            "status_counts": dict(sorted(status_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
    )
    write_json_atomic(
        target / "run_metadata.json",
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "claim_status": "NON_CLAIMING",
            "normalized_command": " ".join(command.split()),
            "normalized_config": manifest.to_dict(),
            "manifest_digest": manifest.digest,
            "environment_digest": environment_digest,
            "output_schema_version": manifest.output_schema_version,
            "table_format": table_format,
            "timings": {"elapsed_seconds": float(elapsed_seconds)},
        },
    )
    artifact_names = (
        "manifest.json",
        "environment.json",
        outer_name,
        inner_name,
        "cell_summary.json",
        "run_metadata.json",
    )
    digests = {name: _sha256_file(target / name) for name in artifact_names}
    write_json_atomic(
        target / "digests.json",
        {"schema_version": ARTIFACT_SCHEMA_VERSION, "files": digests},
    )
    return digests


def validate_artifact_digests(directory: str | Path) -> dict[str, str]:
    target = Path(directory)
    try:
        payload = json.loads((target / "digests.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("digests.json is missing or invalid") from exc
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactIntegrityError("artifact digest schema mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ArtifactIntegrityError("artifact digest inventory is missing")
    for name, expected in files.items():
        path = target / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ArtifactIntegrityError(f"artifact digest mismatch: {name}")
    return dict(files)


def validate_auditable_artifacts(directory: str | Path) -> None:
    """Fail closed when integrity or mandatory provenance is absent."""

    target = Path(directory)
    validate_artifact_digests(target)
    try:
        manifest_payload = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        environment = json.loads((target / "environment.json").read_text(encoding="utf-8"))
        metadata = json.loads((target / "run_metadata.json").read_text(encoding="utf-8"))
        manifest = ExperimentManifest.from_dict(manifest_payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError("critical provenance is missing or invalid") from exc
    environment_required = {
        "schema_version",
        "repository",
        "source_sha",
        "python_version",
        "numpy_version",
        "scipy_version",
        "platform",
        "implementation",
    }
    metadata_required = {
        "schema_version",
        "claim_status",
        "normalized_command",
        "normalized_config",
        "manifest_digest",
        "environment_digest",
        "output_schema_version",
        "table_format",
        "timings",
    }
    if not environment_required.issubset(environment) or not metadata_required.issubset(metadata):
        raise ArtifactIntegrityError("critical provenance fields are absent")
    if (
        environment["repository"] != manifest.source_repository
        or environment["source_sha"] != manifest.source_sha
        or metadata["manifest_digest"] != manifest.digest
        or metadata["claim_status"] != "NON_CLAIMING"
        or metadata["normalized_config"] != manifest.to_dict()
    ):
        raise ArtifactIntegrityError("critical provenance identity mismatch")
    expected_environment_digest = hashlib.sha256(
        canonical_json(environment).encode("utf-8")
    ).hexdigest()
    if metadata["environment_digest"] != expected_environment_digest:
        raise ArtifactIntegrityError("environment provenance digest mismatch")
    table_format = metadata.get("table_format")
    required_files = {
        "manifest.json",
        "environment.json",
        f"outer_results.{table_format}",
        f"inner_accounting.{table_format}",
        "cell_summary.json",
        "run_metadata.json",
    }
    digest_files = json.loads((target / "digests.json").read_text(encoding="utf-8"))["files"]
    if not required_files.issubset(digest_files):
        raise ArtifactIntegrityError("canonical artifact digest inventory is incomplete")
    timings = metadata.get("timings")
    if (
        not metadata.get("normalized_command")
        or not isinstance(timings, dict)
        or not isinstance(timings.get("elapsed_seconds"), (int, float))
        or not np.isfinite(timings["elapsed_seconds"])
        or timings["elapsed_seconds"] < 0
    ):
        raise ArtifactIntegrityError("critical command or timing provenance is invalid")
    if table_format == "jsonl":
        try:
            for line in (target / "outer_results.jsonl").read_text(encoding="utf-8").splitlines():
                _verify_record_digest(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("outer result records are invalid") from exc


def write_checkpoint(
    path: str | Path,
    manifest: ExperimentManifest,
    output: RunOutput,
) -> None:
    body = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "output_schema_version": manifest.output_schema_version,
        "manifest_digest": manifest.digest,
        "source_sha": manifest.source_sha,
        "completed_raw_indices": sorted(
            result.raw_outer_index for result in output.outer_results
        ),
        "outer_results": [_record_payload(result) for result in output.outer_results],
        "inner_accounting": [row.to_dict() for row in output.inner_accounting],
    }
    body["checkpoint_digest"] = hashlib.sha256(
        canonical_json(body).encode("utf-8")
    ).hexdigest()
    write_json_atomic(Path(path), body)


def read_checkpoint(path: str | Path, manifest: ExperimentManifest) -> RunOutput:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("checkpoint is missing or invalid") from exc
    recorded = payload.pop("checkpoint_digest", None)
    actual = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    if recorded != actual:
        raise ArtifactIntegrityError("checkpoint digest mismatch")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactIntegrityError("checkpoint schema mismatch")
    if payload.get("output_schema_version") != manifest.output_schema_version:
        raise ArtifactIntegrityError("checkpoint output schema mismatch")
    if payload.get("manifest_digest") != manifest.digest:
        raise ArtifactIntegrityError("checkpoint manifest identity mismatch")
    if payload.get("source_sha") != manifest.source_sha:
        raise ArtifactIntegrityError("checkpoint source SHA mismatch")

    outer_payloads = payload.get("outer_results")
    inner_payloads = payload.get("inner_accounting")
    completed = payload.get("completed_raw_indices")
    if not isinstance(outer_payloads, list) or not isinstance(inner_payloads, list):
        raise ArtifactIntegrityError("checkpoint record collections are invalid")
    outer_data = [_checkpoint_record_data(item) for item in outer_payloads]
    allowed_fields = {field.name for field in fields(OuterResult)}
    if any(set(item) != allowed_fields for item in outer_data):
        raise ArtifactIntegrityError("checkpoint outer schema mismatch")
    try:
        outer = tuple(OuterResult(**item) for item in outer_data)
        inner = tuple(InnerAttempt(**item) for item in inner_payloads)
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError("checkpoint record schema mismatch") from exc
    raw_indices = [result.raw_outer_index for result in outer]
    if (
        completed != sorted(raw_indices)
        or len(raw_indices) != len(set(raw_indices))
        or not set(raw_indices).issubset(owned_raw_indices(manifest))
        or any(item.raw_outer_index not in set(raw_indices) for item in inner)
    ):
        raise ArtifactIntegrityError("completed raw indices are inconsistent")
    for result in outer:
        attempts = sorted(
            item.raw_inner_index
            for item in inner
            if item.raw_outer_index == result.raw_outer_index
        )
        if attempts != list(range(result.inner_attempts)):
            raise ArtifactIntegrityError("inner attempt indices are inconsistent")
    return RunOutput(
        outer_results=_assign_eligible_indices(outer),
        inner_accounting=tuple(
            sorted(inner, key=lambda item: (item.raw_outer_index, item.raw_inner_index))
        ),
    )


def resume_manifest(
    manifest: ExperimentManifest,
    checkpoint_path: str | Path,
    *,
    execution_order: Iterable[int] | None = None,
    max_new_units: int | None = None,
) -> tuple[RunOutput, bool]:
    """Continue a run and atomically checkpoint after every complete outer unit."""

    path = Path(checkpoint_path)
    if path.exists():
        existing = read_checkpoint(path, manifest)
    else:
        existing = RunOutput(outer_results=(), inner_accounting=())
    completed = {result.raw_outer_index for result in existing.outer_results}
    remaining = [index for index in owned_raw_indices(manifest) if index not in completed]
    if execution_order is not None:
        requested = tuple(execution_order)
        if len(requested) != len(set(requested)) or set(requested) != set(remaining):
            raise ValueError("resume execution_order must contain every remaining index once")
        remaining = list(requested)
    if max_new_units is not None:
        if type(max_new_units) is not int or max_new_units < 0:
            raise ValueError("max_new_units must be a non-negative Python int")
        remaining = remaining[:max_new_units]

    outer = list(existing.outer_results)
    inner = list(existing.inner_accounting)
    for raw_outer_index in remaining:
        result, attempts = run_outer_unit(manifest, raw_outer_index)
        outer.append(result)
        inner.extend(attempts)
        output = RunOutput(
            outer_results=_assign_eligible_indices(outer),
            inner_accounting=tuple(
                sorted(inner, key=lambda item: (item.raw_outer_index, item.raw_inner_index))
            ),
        )
        write_checkpoint(path, manifest, output)
    final = RunOutput(
        outer_results=_assign_eligible_indices(outer),
        inner_accounting=tuple(
            sorted(inner, key=lambda item: (item.raw_outer_index, item.raw_inner_index))
        ),
    )
    is_complete = len(final.outer_results) == len(owned_raw_indices(manifest))
    return final, is_complete
