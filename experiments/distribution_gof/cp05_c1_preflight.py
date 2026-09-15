"""Execute the frozen CP05-C1 performance preflight without inferential claims.

This module orchestrates the integrated CP05-B harness.  It intentionally does
not implement statistics, fitting, random-number generation, retry semantics,
or calibration logic.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pyarrow
import scipy

from .accounting import AssessmentStatus, ReasonCode
from .artifacts import (
    read_checkpoint,
    validate_auditable_artifacts,
    write_checkpoint,
    write_json_atomic,
    write_run_artifacts,
)
from .manifest import (
    ALPHA,
    DEVELOPMENT_NAMESPACE,
    MANIFEST_SCHEMA_VERSION,
    NON_CALIBRATION_PHASE,
    OUTPUT_SCHEMA_VERSION,
    SOURCE_REPOSITORY,
    ExperimentManifest,
)
from .runner import RunOutput, merge_outputs, run_outer_unit
from .resource_measurement import measure_peak_process_memory


WORK_ITEM = "CP05-C1"
BASELINE_SHA = "6fe0bcf0a71170dd29c2605fc6d6e3feeffe7bce"
AUTHORIZED_BRANCH = "experiments/cp05-c-performance-preflight"
R_PREFLIGHT = 20
OUTER_ATTEMPT_MULTIPLIER = 100
R_C = 2_000
STATISTICS = ("AD", "CVM")
B_VALUES = (199, 999)
RESULT_SCHEMA_VERSION = "cp05-c1-performance-preflight-v1"


@dataclass(frozen=True, slots=True)
class PreflightCell:
    cell_id: str
    family: str
    parameters: dict[str, str]
    n: int = 20


CELLS = (
    PreflightCell("gamma_shape_0p25_scale_1", "gamma", {"shape": "0.25", "scale": "1"}),
    PreflightCell("exponential_scale_1", "exponential", {"scale": "1"}),
    PreflightCell("negative_binomial_r_0p25_p_0p1", "negative_binomial", {"r": "0.25", "p": "0.1"}),
    PreflightCell("negative_binomial_r_20_p_0p9", "negative_binomial", {"r": "20", "p": "0.9"}),
)


def _configuration_id(cell: PreflightCell, statistic: str, B: int) -> str:
    return f"{cell.cell_id}__{statistic.lower()}__b{B}"


def _configurations() -> tuple[tuple[str, PreflightCell, str, int], ...]:
    return tuple(
        (_configuration_id(cell, statistic, B), cell, statistic, B)
        for cell in CELLS
        for statistic in STATISTICS
        for B in B_VALUES
    )


class GitEnvironmentError(RuntimeError):
    """Raised when Git cannot provide the evidence required for identity checks."""


class SourceIdentityError(RuntimeError):
    """Raised when repository evidence does not match the authorized work line."""


def _resolve_git_executable() -> str:
    executable = shutil.which("git")
    if executable is None:
        raise GitEnvironmentError(
            "BLOCKED_ENVIRONMENT: git executable was not found via shutil.which('git')"
        )
    return executable


def _git_result(
    executable: str,
    repository: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [executable, *args],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except OSError as exc:
        raise GitEnvironmentError(
            f"BLOCKED_ENVIRONMENT: unable to execute git: {exc}"
        ) from exc


def _git(
    executable: str,
    repository: Path,
    *args: str,
) -> str:
    completed = _git_result(executable, repository, *args)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise GitEnvironmentError(
            f"BLOCKED_ENVIRONMENT: git {' '.join(args)} failed "
            f"with exit {completed.returncode}: {detail}"
        )
    return completed.stdout.strip()


def _is_ancestor(
    executable: str,
    repository: Path,
    ancestor: str,
    descendant: str,
) -> bool:
    completed = _git_result(
        executable,
        repository,
        "merge-base",
        "--is-ancestor",
        ancestor,
        descendant,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
    raise GitEnvironmentError(
        "BLOCKED_ENVIRONMENT: git merge-base --is-ancestor failed "
        f"with exit {completed.returncode}: {detail}"
    )


def _optional_ref_sha(
    executable: str,
    repository: Path,
    ref: str,
) -> str | None:
    probe = _git_result(executable, repository, "show-ref", "--verify", "--quiet", ref)
    if probe.returncode == 1:
        return None
    if probe.returncode != 0:
        detail = probe.stderr.strip() or probe.stdout.strip() or "no diagnostic"
        raise GitEnvironmentError(
            f"BLOCKED_ENVIRONMENT: unable to inspect authorized ref {ref}: {detail}"
        )
    return _git(executable, repository, "show-ref", "--verify", "--hash", ref)


def _verify_source_identity(
    repository: Path | str = Path("."),
    *,
    baseline_sha: str = BASELINE_SHA,
    authorized_branch: str = AUTHORIZED_BRANCH,
    git_executable: str | None = None,
) -> dict[str, Any]:
    repository = Path(repository)
    executable = git_executable or _resolve_git_executable()
    actual_head_sha = _git(executable, repository, "rev-parse", "--verify", "HEAD^{commit}")
    branch = _git(executable, repository, "branch", "--show-current")

    baseline_object = _git_result(
        executable,
        repository,
        "cat-file",
        "-e",
        f"{baseline_sha}^{{commit}}",
    )
    if baseline_object.returncode != 0:
        raise SourceIdentityError(
            f"BLOCKED_IDENTITY: baseline commit is unavailable: {baseline_sha}"
        )
    if not _is_ancestor(executable, repository, baseline_sha, actual_head_sha):
        raise SourceIdentityError(
            "BLOCKED_IDENTITY: baseline is not an ancestor of actual HEAD: "
            f"baseline={baseline_sha} head={actual_head_sha}"
        )
    merge_base = _git(executable, repository, "merge-base", baseline_sha, actual_head_sha)
    if merge_base != baseline_sha:
        raise SourceIdentityError(
            "BLOCKED_IDENTITY: unexpected divergence from baseline: "
            f"merge_base={merge_base} baseline={baseline_sha}"
        )

    authorized_refs: list[dict[str, str]] = []
    if branch:
        if branch != authorized_branch:
            raise SourceIdentityError(
                "BLOCKED_IDENTITY: active branch is not the authorized work line: "
                f"actual={branch} authorized={authorized_branch}"
            )
        ref = f"refs/heads/{authorized_branch}"
        ref_sha = _optional_ref_sha(executable, repository, ref)
        if ref_sha != actual_head_sha:
            raise SourceIdentityError(
                "BLOCKED_IDENTITY: authorized local branch does not resolve to actual HEAD: "
                f"ref={ref_sha or 'UNAVAILABLE'} head={actual_head_sha}"
            )
        authorized_refs.append({"ref": ref, "sha": ref_sha})
        checkout_mode = "AUTHORIZED_BRANCH"
        identity_limitation = None
    else:
        checkout_mode = "DETACHED_HEAD"
        identity_limitation = None
        for ref in (
            f"refs/heads/{authorized_branch}",
            f"refs/remotes/origin/{authorized_branch}",
        ):
            ref_sha = _optional_ref_sha(executable, repository, ref)
            if ref_sha is not None:
                authorized_refs.append({"ref": ref, "sha": ref_sha})
        if not authorized_refs:
            identity_limitation = "AUTHORIZED_BRANCH_REF_UNAVAILABLE"
            raise SourceIdentityError(
                "BLOCKED_IDENTITY: detached HEAD cannot be tied to the authorized work line; "
                "limitation=AUTHORIZED_BRANCH_REF_UNAVAILABLE"
            )
        reachable_refs = [
            item
            for item in authorized_refs
            if _is_ancestor(executable, repository, actual_head_sha, item["sha"])
        ]
        if not reachable_refs:
            raise SourceIdentityError(
                "BLOCKED_IDENTITY: detached HEAD is not reachable from an authorized branch ref: "
                f"head={actual_head_sha}"
            )
        authorized_refs = reachable_refs

    # The authorized candidate is derived from the verified checkout rather than
    # frozen in source, so evidence commits on this branch remain reproducible.
    authorized_candidate_sha = actual_head_sha
    return {
        "source_identity_status": "PASS",
        "baseline_sha": baseline_sha,
        "baseline_main_sha": baseline_sha,
        "authorized_candidate_sha": authorized_candidate_sha,
        "execution_source_sha": actual_head_sha,
        "actual_head_sha": actual_head_sha,
        "source_sha": actual_head_sha,
        "authorized_branch": authorized_branch,
        "branch": branch or "DETACHED",
        "branch_or_detached_state": checkout_mode,
        "checkout_mode": checkout_mode,
        "baseline_ancestor_of_head": True,
        "merge_base": merge_base,
        "baseline_merge_base": merge_base,
        "authorized_refs": authorized_refs,
        "identity_limitation": identity_limitation,
    }


def _environment_requirements() -> dict[str, str]:
    return {
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "pyarrow": pyarrow.__version__,
        "python": platform.python_version(),
        "scipy": scipy.__version__,
    }


def _manifest(
    cell: PreflightCell,
    statistic: str,
    B: int,
    execution_source_sha: str,
) -> ExperimentManifest:
    raw_cap = (
        OUTER_ATTEMPT_MULTIPLIER * R_PREFLIGHT
        if cell.family == "negative_binomial"
        else R_PREFLIGHT
    )
    return ExperimentManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        phase=NON_CALIBRATION_PHASE,
        source_repository=SOURCE_REPOSITORY,
        source_sha=execution_source_sha,
        namespace_id=DEVELOPMENT_NAMESPACE,
        null_type="composite",
        family=cell.family,
        statistic=statistic,
        n=cell.n,
        canonical_parameters=cell.parameters,
        B=B,
        alpha=ALPHA,
        raw_outer_range=(0, raw_cap),
        shard_spec={"shard_id": 0, "shard_count": 1},
        batch_spec={"batch_size": 1},
        worker_spec={"workers": 1},
        environment_requirements=_environment_requirements(),
        output_schema_version=OUTPUT_SCHEMA_VERSION,
    )


def _load_timings(path: Path, manifest: ExperimentManifest, completed: set[int]) -> dict[int, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise RuntimeError("timing checkpoint schema mismatch")
    if payload.get("manifest_digest") != manifest.digest:
        raise RuntimeError("timing checkpoint manifest mismatch")
    timings = {int(key): float(value) for key, value in payload.get("seconds", {}).items()}
    timings = {key: value for key, value in timings.items() if key in completed}
    if set(timings) != completed:
        raise RuntimeError("timing checkpoint is incomplete for completed outer units")
    return timings


def _write_timings(path: Path, manifest: ExperimentManifest, timings: dict[int, float]) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "manifest_digest": manifest.digest,
            "clock": "time.perf_counter",
            "seconds": {str(key): timings[key] for key in sorted(timings)},
        },
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile, method="linear"))


def _run_configuration(
    output_root: Path,
    configuration_id: str,
    cell: PreflightCell,
    statistic: str,
    B: int,
    execution_source_sha: str,
) -> dict[str, Any]:
    manifest = _manifest(cell, statistic, B, execution_source_sha)
    target = output_root / "configurations" / configuration_id
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target / "checkpoint.json"
    timings_path = target / "assessment_timings.json"

    if checkpoint_path.exists():
        existing = read_checkpoint(checkpoint_path, manifest)
    else:
        existing = RunOutput(outer_results=(), inner_accounting=())
    completed = {item.raw_outer_index for item in existing.outer_results}
    timings = _load_timings(timings_path, manifest, completed)
    outer = list(existing.outer_results)
    inner = list(existing.inner_accounting)
    assessed = sum(item.status == AssessmentStatus.ASSESSED.value for item in outer)
    next_index = max(completed, default=-1) + 1
    raw_cap = manifest.raw_outer_range[1]
    unexpected_terminal: dict[str, Any] | None = None

    while assessed < R_PREFLIGHT and next_index < raw_cap:
        started = time.perf_counter()
        result, attempts = run_outer_unit(manifest, next_index)
        elapsed = time.perf_counter() - started
        timings[next_index] = elapsed
        outer.append(result)
        inner.extend(attempts)
        output = merge_outputs([RunOutput(tuple(outer), tuple(inner))])
        _write_timings(timings_path, manifest, timings)
        write_checkpoint(checkpoint_path, manifest, output)
        assessed = sum(
            item.status == AssessmentStatus.ASSESSED.value for item in output.outer_results
        )
        permitted_outer_ineligibility = (
            cell.family == "negative_binomial"
            and result.status == AssessmentStatus.NOT_ASSESSED.value
            and result.reason_code == ReasonCode.MATHEMATICAL_INELIGIBILITY.value
            and result.observed_mle_eligible is False
        )
        if result.status != AssessmentStatus.ASSESSED.value and not permitted_outer_ineligibility:
            unexpected_terminal = {
                "raw_outer_index": result.raw_outer_index,
                "status": result.status,
                "reason_code": result.reason_code,
                "failure_class": result.failure_class,
            }
            break
        print(
            json.dumps(
                {
                    "configuration": configuration_id,
                    "raw_outer_completed": len(output.outer_results),
                    "assessed": assessed,
                    "target": R_PREFLIGHT,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        outer = list(output.outer_results)
        inner = list(output.inner_accounting)
        next_index += 1

    output = merge_outputs([RunOutput(tuple(outer), tuple(inner))])
    status_counts = Counter(item.status for item in output.outer_results)
    reason_counts = Counter(item.reason_code for item in output.outer_results)
    eligible_outer = sum(item.observed_mle_eligible is True for item in output.outer_results)
    assessed_durations = [
        timings[item.raw_outer_index]
        for item in output.outer_results
        if item.status == AssessmentStatus.ASSESSED.value
    ]
    nb_outer_histogram = Counter(
        item.outer_ineligibility_reason
        for item in output.outer_results
        if item.outer_ineligibility_reason is not None
    )
    nb_inner_ineligible = [item for item in output.inner_accounting if not item.eligible]
    nb_inner_histogram = Counter(item.reason_code for item in nb_inner_ineligible)
    completed_outer = len(output.outer_results)
    assessed = status_counts[AssessmentStatus.ASSESSED.value]
    complete = assessed == R_PREFLIGHT and unexpected_terminal is None
    failure = unexpected_terminal
    if not complete and failure is None:
        failure = {
            "status": "NOT_ASSESSED",
            "reason_code": "OUTER_ATTEMPT_CAP_EXHAUSTED",
            "raw_outer_attempt_cap": raw_cap,
        }

    elapsed_total = float(sum(timings.values()))
    command = (
        f"{sys.executable} -m experiments.distribution_gof.cp05_c1_preflight "
        f"--single {configuration_id} --output {output_root}"
    )
    write_run_artifacts(
        target,
        manifest,
        output,
        table_format="parquet",
        command=command,
        elapsed_seconds=elapsed_total,
    )
    validate_auditable_artifacts(target)

    memory = measure_peak_process_memory()
    summary: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "work_item": WORK_ITEM,
        "claim_status": "NON_CALIBRATION_NON_CLAIMING",
        "configuration_id": configuration_id,
        "family": cell.family,
        "parameters": cell.parameters,
        "n": cell.n,
        "statistic": statistic,
        "B": B,
        "R_PREFLIGHT": R_PREFLIGHT,
        "completed_outer": completed_outer,
        "eligible_outer": eligible_outer,
        "assessed_outer": assessed,
        "not_assessed": status_counts[AssessmentStatus.NOT_ASSESSED.value],
        "failed": status_counts[AssessmentStatus.FAILED.value],
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "wall_time_total_seconds": elapsed_total,
        "wall_time_per_assessment_median_seconds": _percentile(assessed_durations, 50),
        "wall_time_per_assessment_p95_seconds": _percentile(assessed_durations, 95),
        "peak_memory": memory.to_dict(),
        "peak_memory_bytes": memory.value,
        "peak_memory_mib": memory.value_mib,
        "NB_raw_outer_attempts": completed_outer if cell.family == "negative_binomial" else None,
        "NB_eligibility_rate": (
            eligible_outer / completed_outer
            if cell.family == "negative_binomial" and completed_outer
            else None
        ),
        "NB_ineligibility_reason_histogram": (
            dict(sorted(nb_outer_histogram.items()))
            if cell.family == "negative_binomial"
            else None
        ),
        "NB_inner_retry_burden": (
            {
                "raw_inner_attempts": len(output.inner_accounting),
                "eligible_inner": sum(item.eligible for item in output.inner_accounting),
                "ineligible_inner": len(nb_inner_ineligible),
                "ineligibility_reason_histogram": dict(sorted(nb_inner_histogram.items())),
                "extra_attempts_per_assessed_outer": (
                    len(nb_inner_ineligible) / assessed if assessed else None
                ),
            }
            if cell.family == "negative_binomial"
            else None
        ),
        "manifest_digest": manifest.digest,
        "artifact_validation": "PASS",
        "configuration_status": "PASS" if complete else "FAILED",
        "failure": failure,
    }
    write_json_atomic(target / "configuration_summary.json", summary)
    return summary


def _full_matrix_projection(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in summaries:
        by_key.setdefault((row["family"], row["statistic"], row["B"]), []).append(row)

    parameter_counts = {"gamma": 5, "exponential": 1, "negative_binomial": 12}
    sample_sizes = (20, 50, 100, 250)
    null_types = ("simple", "composite")
    compute_median = 0.0
    compute_p95 = 0.0
    assessment_units = 0
    projected_raw_outer_attempts = 0.0
    for family, parameter_count in parameter_counts.items():
        for statistic in STATISTICS:
            for B in B_VALUES:
                reference = by_key[(family, statistic, B)]
                median = max(float(row["wall_time_per_assessment_median_seconds"]) for row in reference)
                p95 = max(float(row["wall_time_per_assessment_p95_seconds"]) for row in reference)
                nb_rate = (
                    min(float(row["NB_eligibility_rate"]) for row in reference)
                    if family == "negative_binomial"
                    else 1.0
                )
                for n in sample_sizes:
                    scale = n / 20.0
                    for null_type in null_types:
                        multiplier = 1.0 / nb_rate if family == "negative_binomial" and null_type == "composite" else 1.0
                        units = parameter_count * R_C
                        assessment_units += units
                        projected_raw_outer_attempts += units * multiplier
                        compute_median += units * median * scale * multiplier
                        compute_p95 += units * p95 * scale * multiplier

    assumed_workers = 1
    detected_logical_cpus = os.cpu_count() or 1
    max_peak = max(int(row["peak_memory_bytes"]) for row in summaries)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "claim_status": "NON_CALIBRATION_NON_CLAIMING",
        "guarantee": False,
        "full_matrix_primary_candidate_configurations": 576,
        "full_matrix_eligible_assessment_units": assessment_units,
        "projected_raw_outer_attempts": projected_raw_outer_attempts,
        "PROJECTED_CP05_C_FULL_MATRIX_COMPUTE": {
            "median_basis_seconds": compute_median,
            "p95_planning_seconds": compute_p95,
        },
        "PROJECTED_CP05_C_FULL_MATRIX_WALL_TIME": {
            "assumed_parallel_workers": assumed_workers,
            "median_basis_seconds": compute_median / assumed_workers,
            "p95_planning_seconds": compute_p95 / assumed_workers,
            "idealized_p95_seconds_at_detected_logical_cpus": compute_p95 / detected_logical_cpus,
            "detected_logical_cpus": detected_logical_cpus,
        },
        "PROJECTED_PEAK_MEMORY": {
            "assumed_parallel_workers": assumed_workers,
            "per_worker_peak_bytes": max_peak,
            "aggregate_peak_bytes": max_peak * assumed_workers,
        },
        "assumptions": [
            "Only primary candidates AD and CVM are included; KS and Pearson are excluded.",
            "R_C is 2000 eligible outer assessments per mandatory development cell.",
            "Both simple-null and composite-null cells and B in {199,999} are included.",
            "Observed n=20 time scales linearly with n for n in {20,50,100,250}.",
            "The slower observed preflight parameter cell is used within each family/statistic/B stratum.",
            "Simple-null cells are conservatively priced at the observed composite-null rate.",
            "Composite Negative Binomial raw attempts use the lowest observed eligibility rate in the two preregistered corner cells.",
            "The primary wall-time projection assumes one worker, matching the measured topology.",
            "The detected-CPU figure assumes perfect parallel efficiency and is sensitivity information only.",
            "Artifact I/O and scheduler contention beyond measured per-assessment time are not separately modeled.",
            "This projection is not a guarantee and is not calibration evidence.",
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_root_digests(output_root: Path) -> None:
    files = {
        path.relative_to(output_root).as_posix(): _sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "digests.json"
    }
    write_json_atomic(
        output_root / "digests.json",
        {"schema_version": RESULT_SCHEMA_VERSION, "files": files},
    )


def _plan_payload(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "work_item": WORK_ITEM,
        "purpose": "NON_INFERENTIAL_PERFORMANCE_PREFLIGHT",
        "claim_status": "NON_CALIBRATION_NON_CLAIMING",
        "source_repository": SOURCE_REPOSITORY,
        "source_sha": identity["actual_head_sha"],
        "execution_source_sha": identity["execution_source_sha"],
        "actual_head_sha": identity["actual_head_sha"],
        "authorized_candidate_sha": identity["authorized_candidate_sha"],
        "baseline_sha": identity["baseline_sha"],
        "baseline_main_sha": identity["baseline_main_sha"],
        "authorized_branch": identity["authorized_branch"],
        "branch": identity["branch"],
        "branch_or_detached_state": identity["branch_or_detached_state"],
        "checkout_mode": identity["checkout_mode"],
        "merge_base": identity["merge_base"],
        "namespace_id": DEVELOPMENT_NAMESPACE,
        "secret_namespace_used": False,
        "null_type": "composite",
        "R_PREFLIGHT": R_PREFLIGHT,
        "B_values": list(B_VALUES),
        "statistics": list(STATISTICS),
        "cells": [
            {
                "cell_id": cell.cell_id,
                "family": cell.family,
                "parameters": cell.parameters,
                "n": cell.n,
            }
            for cell in CELLS
        ],
        "configurations": [configuration_id for configuration_id, *_ in _configurations()],
        "full_matrix_executed": False,
        "calibration_claim_made": False,
        "power_analysis_executed": False,
        "method_selected": False,
        "cp05_d_executed": False,
        "holdout_accessed": False,
    }


def _provenance_payload(
    identity: dict[str, Any],
    configuration_count: int,
    *,
    executable: str,
    command: str,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "repository": SOURCE_REPOSITORY,
        **identity,
        "environment": _environment_requirements(),
        "executable": executable,
        "command": command,
        "configuration_count": configuration_count,
    }


def _run_all(output_root: Path, identity: dict[str, Any]) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / "preflight_manifest.json", _plan_payload(identity))
    failures: list[dict[str, Any]] = []
    for configuration_id, _, _, _ in _configurations():
        command = [
            sys.executable,
            "-m",
            "experiments.distribution_gof.cp05_c1_preflight",
            "--single",
            configuration_id,
            "--output",
            str(output_root),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failures.append(
                {"configuration_id": configuration_id, "return_code": completed.returncode}
            )
            break

    summary_paths = sorted((output_root / "configurations").glob("*/configuration_summary.json"))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    failed_configurations = [row for row in summaries if row["configuration_status"] != "PASS"]
    if failures or failed_configurations or len(summaries) != len(_configurations()):
        write_json_atomic(
            output_root / "preflight_failure.json",
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "work_item": WORK_ITEM,
                "status": "BLOCKED_SOFTWARE",
                "subprocess_failures": failures,
                "failed_configurations": failed_configurations,
                "completed_configurations": len(summaries),
                "baseline_sha": identity["baseline_sha"],
                "baseline_main_sha": identity["baseline_main_sha"],
                "execution_source_sha": identity["execution_source_sha"],
                "actual_head_sha": identity["actual_head_sha"],
            },
        )
        _write_root_digests(output_root)
        return 1

    projection = _full_matrix_projection(summaries)
    write_json_atomic(output_root / "cost_projection.json", projection)
    write_json_atomic(
        output_root / "resource_metrics.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "configurations": summaries,
            "projected_peak_memory": projection["PROJECTED_PEAK_MEMORY"],
        },
    )
    cell_status = {
        cell.cell_id: "PASS"
        if all(
            row["configuration_status"] == "PASS"
            for row in summaries
            if row["configuration_id"].startswith(cell.cell_id + "__")
        )
        else "FAILED"
        for cell in CELLS
    }
    write_json_atomic(
        output_root / "preflight_summary.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "work_item": WORK_ITEM,
            "status": "COMPLETE",
            "claim_status": "NON_CALIBRATION_NON_CLAIMING",
            "source_sha": identity["actual_head_sha"],
            "execution_source_sha": identity["execution_source_sha"],
            "actual_head_sha": identity["actual_head_sha"],
            "baseline_sha": identity["baseline_sha"],
            "baseline_main_sha": identity["baseline_main_sha"],
            "R_PREFLIGHT": R_PREFLIGHT,
            "cells_executed": len(CELLS),
            "configurations_executed": len(summaries),
            "cell_status": cell_status,
            "B199_completed": all(row["configuration_status"] == "PASS" for row in summaries if row["B"] == 199),
            "B999_completed": all(row["configuration_status"] == "PASS" for row in summaries if row["B"] == 999),
            "failures": [],
            "resource_assessment": "PASS",
            "full_matrix_executed": False,
            "calibration_claim_made": False,
            "power_analysis_executed": False,
            "method_selected": False,
            "cp05_d_executed": False,
            "holdout_accessed": False,
        },
    )
    write_json_atomic(
        output_root / "provenance.json",
        _provenance_payload(
            identity,
            len(summaries),
            executable=sys.executable,
            command=" ".join(sys.argv),
        ),
    )
    _write_root_digests(output_root)
    return 0


def _find_configuration(configuration_id: str) -> tuple[PreflightCell, str, int]:
    for candidate_id, cell, statistic, B in _configurations():
        if candidate_id == configuration_id:
            return cell, statistic, B
    raise ValueError(f"unknown configuration: {configuration_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/distribution_gof/cp05_c1_results"),
    )
    parser.add_argument("--single")
    args = parser.parse_args()
    identity = _verify_source_identity()
    if args.single:
        cell, statistic, B = _find_configuration(args.single)
        summary = _run_configuration(
            args.output,
            args.single,
            cell,
            statistic,
            B,
            identity["actual_head_sha"],
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0 if summary["configuration_status"] == "PASS" else 1
    return _run_all(args.output, identity)


if __name__ == "__main__":
    raise SystemExit(main())
