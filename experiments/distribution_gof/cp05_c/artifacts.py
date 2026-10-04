"""Auditable Parquet bundles reusing CP05-B codecs and atomic writers."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
import json
from pathlib import Path

from ..accounting import OuterResult, InnerAttempt
from ..artifacts import (ArtifactIntegrityError, capture_environment, _write_parquet,
                         read_parquet_rows, write_json_atomic, _sha256_file, _verify_record_digest)
from ..seed_derivation import SEED_DERIVATION_VERSION
from .manifest import DevelopmentManifest, OUTPUT_SCHEMA, BLOCKED_STAGES
from .checkpoint import _digest, validate_unit, validate_shadow_state

ARTIFACT_SCHEMA = "cp05-c-null-artifacts-v1"
REQUIRED_FILES = ("manifest.json", "environment.json", "outer_results.parquet",
                  "inner_accounting.parquet", "cell_summary.json", "run_metadata.json", "shadow_oracle.json")


def write_bundle(directory, manifest, output, cell_summary, shadow, *, command, timings=()):
    directory = Path(directory)
    environment = capture_environment(manifest)
    environment["schema_version"] = ARTIFACT_SCHEMA
    environment_digest = _digest(environment)
    records = []
    for result in output.outer_results:
        row = result.to_dict()
        row.pop("output_digest")
        row.update(source_sha=manifest.source_sha, config_digest=manifest.digest,
                   environment_digest=environment_digest, seed_derivation_version=SEED_DERIVATION_VERSION)
        row["output_digest"] = _digest(row)
        records.append(row)
    write_json_atomic(directory / "manifest.json", manifest.to_dict())
    write_json_atomic(directory / "environment.json", environment)
    _write_parquet(directory / "outer_results.parquet", records)
    _write_parquet(directory / "inner_accounting.parquet", [a.to_dict() for a in output.inner_accounting])
    write_json_atomic(directory / "cell_summary.json", cell_summary)
    write_json_atomic(directory / "shadow_oracle.json", {
        "schema_version": ARTIFACT_SCHEMA, "canonical_cell_id": manifest.canonical_cell_id,
        "CANONICAL_ORACLE": "CP04 Decimal Negative Binomial fitter",
        "PRODUCTION_BACKEND": "NO", "FLAT_OBJECTIVE_WAIVER_USED": "NO", **shadow})
    write_json_atomic(directory / "run_metadata.json", {
        "schema_version": ARTIFACT_SCHEMA, "manifest_digest": manifest.digest,
        "environment_digest": environment_digest, "normalized_config": manifest.to_dict(),
        "normalized_command": " ".join(command.split()), "output_schema_version": OUTPUT_SCHEMA,
        "table_format": "parquet", "software_fixture": cell_summary["software_fixture"],
        "primary_outer_wall_seconds": list(timings),
        "timings": {k: cell_summary[k] for k in ("total_wall_seconds", "primary_wall_seconds", "shadow_wall_seconds")},
        "execution_class": "research-software-fixture" if cell_summary["software_fixture"] else "research-development-null",
        "wall_timing_boundary": "cumulative invocation wall through calibration/checkpoint preparation; final bundle serialization excluded",
        **BLOCKED_STAGES})
    write_json_atomic(directory / "digests.json", {
        "schema_version": ARTIFACT_SCHEMA,
        "files": {name: _sha256_file(directory / name) for name in REQUIRED_FILES}})


def validate_bundle(directory):
    """Verify inventory, bytes, identities, seeds, denominators and shadow gates."""
    directory = Path(directory)
    def require(condition, message):
        if not condition:
            raise ArtifactIntegrityError(message)
    try:
        digests = json.loads((directory / "digests.json").read_text(encoding="utf-8"))
        require(digests["schema_version"] == ARTIFACT_SCHEMA
                and set(digests["files"]) == set(REQUIRED_FILES), "incomplete artifact inventory")
        for name in REQUIRED_FILES:
            require(_sha256_file(directory / name) == digests["files"][name], "artifact digest mismatch: " + name)
        def read(name):
            return json.loads((directory / name).read_text(encoding="utf-8"))
        manifest = DevelopmentManifest.from_dict(read("manifest.json"))
        environment, metadata = read("environment.json"), read("run_metadata.json")
        summary, shadow = read("cell_summary.json"), read("shadow_oracle.json")
        require(environment["schema_version"] == metadata["schema_version"] == ARTIFACT_SCHEMA,
                "provenance schema mismatch")
        require(environment["repository"] == manifest.source_repository
                and environment["source_sha"] == manifest.source_sha
                and metadata["manifest_digest"] == manifest.digest
                and metadata["environment_digest"] == _digest(environment)
                and metadata["normalized_config"] == manifest.to_dict()
                and metadata["normalized_command"] and metadata["table_format"] == "parquet"
                and metadata["output_schema_version"] == OUTPUT_SCHEMA,
                "critical provenance mismatch")
        for payload in (summary, metadata):
            require(all(payload[k] == v for k, v in BLOCKED_STAGES.items()), "blocked stage flag changed")
        require(shadow["schema_version"] == ARTIFACT_SCHEMA
                and shadow["canonical_cell_id"] == manifest.canonical_cell_id
                and shadow["FLAT_OBJECTIVE_WAIVER_USED"] == "NO", "shadow identity/waiver mismatch")
        grouped = defaultdict(list)
        inner = [InnerAttempt(**a) for a in read_parquet_rows(directory / "inner_accounting.parquet")]
        for attempt in inner:
            grouped[attempt.raw_outer_index].append(attempt)
        rows, eligible = [], 0
        field_names = {f.name for f in fields(OuterResult)}
        for index, payload in enumerate(read_parquet_rows(directory / "outer_results.parquet")):
            _verify_record_digest(payload)
            require(payload["source_sha"] == manifest.source_sha and payload["config_digest"] == manifest.digest
                    and payload["environment_digest"] == _digest(environment)
                    and payload["seed_derivation_version"] == SEED_DERIVATION_VERSION,
                    "outer provenance mismatch")
            for alias, source in (("float64_value", "statistic_value"), ("absolute_error", "oracle_abs_error"),
                                  ("relative_scaled_error", "oracle_relative_scaled_error"),
                                  ("remainder_bound", "tail_remainder_bound")):
                require(payload[alias] == payload[source], "outer aliases mismatch")
            row = OuterResult(**{k: payload[k] for k in field_names})
            require(not any(r.status == "FAILED" for r in rows), "execution after failed unit")
            validate_unit(manifest, row, grouped.pop(index, []), index, eligible)
            eligible += row.status == "ASSESSED"
            rows.append(row)
        require(not grouped, "orphan inner accounting")
        require(summary["canonical_cell_id"] == manifest.canonical_cell_id and summary["schema_version"] == OUTPUT_SCHEMA,
                "summary identity mismatch")
        require(type(summary["software_fixture"]) is bool
                and metadata["software_fixture"] is summary["software_fixture"], "fixture provenance mismatch")
        target = summary["eligible_outer_target"]
        require(type(target) is int and 1 <= target <= (1999 if summary["software_fixture"] else 2000),
                "invalid target")
        require(summary["software_fixture"] or target == manifest.R_C, "scientific target changed")
        require(len(rows) <= target * (100 if manifest.nb_composite else 1) and eligible <= target,
                "outer cap/target exceeded")
        require(summary["cell_status"] in ("COMPLETE", "PAUSED", "FAILED", "FAILED_SHADOW_ORACLE"),
                "invalid terminal cell status")
        require(summary["cell_status"] != "COMPLETE" or eligible == target, "incomplete cell marked complete")
        state = {"status": summary["cell_status"], "target": target,
                 "execution_wall_seconds": summary["total_wall_seconds"],
                 "software_fixture": summary["software_fixture"], "failure_reason": summary["failure_reason"],
                 "cell_execution_started": summary["CELL_EXECUTION_STARTED"] == "YES", "shadow": shadow}
        validate_shadow_state(manifest, state, len(rows))
        timings = metadata["primary_outer_wall_seconds"]
        require(len(timings) == len(rows) and all(type(t) in (float, int) and 0 <= t < float("inf") for t in timings),
                "invalid per-outer timings")
        from .runner import summary as calculate_summary
        expected = calculate_summary(manifest, state, rows, inner, timings, summary["peak_sampled_rss_bytes"])
        require(summary == expected, "cell summary does not match outer/shadow accounting")
        require(metadata["timings"] == {k: summary[k] for k in metadata["timings"]}, "timing provenance mismatch")
        return manifest, summary
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError("bundle is missing or invalid") from exc
