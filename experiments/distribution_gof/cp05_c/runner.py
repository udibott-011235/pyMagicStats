"""Cell orchestration around unmodified CP05-B outer/bootstrap execution."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
from pyMagicStat.distributions.families import FitIdentifiabilityError, NoFiniteMLEError

from ..runner import RunnerHooks, RunOutput, run_outer_unit
from ..cuda_calibration.nb_fitting_perf01.engines import RSSPeak
from .manifest import OUTPUT_SCHEMA, WILSON_UPPER_LIMIT, BLOCKED_STAGES
from .fitting import fit_hook, ENGINE
from .shadow import ShadowOracle, periodic_due
from .checkpoint import load_state, save_state, save_segment, validate_unit


def wilson95(x, n):
    if type(x) is not int or type(n) is not int or not 0 <= x <= n:
        raise ValueError("Wilson requires integer 0 <= x <= n")
    if n == 0:
        return None
    # Two-sided 95% normal quantile, with no rounding before the risk gate.
    z = 1.959963984540054
    rate = x / n
    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    radius = z * ((rate * (1 - rate) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _execute_outer(manifest, index, hooks, eligible_index):
    mathematics = []
    fitter = (hooks.fit if hooks and hooks.fit is not None else
              fit_hook(manifest) if manifest.nb_composite else None)
    def fit(family, sample):
        try:
            result = fitter(family, sample) if fitter else family.fit(sample)
            mathematics.append(None)
            return result
        except (FitIdentifiabilityError, NoFiniteMLEError) as exc:
            mathematics.append(type(exc).__name__)
            raise
    base_hooks = hooks or RunnerHooks()
    effective = replace(base_hooks, fit=fit)
    result, attempts = run_outer_unit(manifest, index, hooks=effective)
    result = replace(result, schema_version=OUTPUT_SCHEMA,
                     inner_attempts=len(attempts), raw_inner_indices=list(range(len(attempts))),
                     inner_eligible=sum(a.eligible for a in attempts),
                     inner_ineligible=sum(a.reason_code == "MATHEMATICAL_INELIGIBILITY" for a in attempts),
                     inner_ineligibility_reason_counts=dict(Counter(v for v in mathematics[1:] if v)))
    if result.status == "NOT_ASSESSED" and not (
            manifest.nb_composite and result.reason_code == "MATHEMATICAL_INELIGIBILITY"
            and result.observed_mle_eligible is False):
        result = replace(result, status="FAILED")
    result = replace(result, eligible_outer_index=eligible_index if result.status == "ASSESSED" else None)
    if manifest.nb_composite and result.observed_fit_provenance is not None:
        result.observed_fit_provenance.update(ENGINE=ENGINE, PRODUCTION_BACKEND="NO")
    validate_unit(manifest, result, attempts, index, eligible_index)
    return result, attempts


def summary(manifest, state, rows, inner, timings, peak_rss=0):
    assessed = [r for r in rows if r.status == "ASSESSED"]
    R, x = len(assessed), sum(r.reject is True for r in assessed)
    interval = wilson95(x, R)
    applicable = sum(r.observed_mle_eligible is True for r in rows)
    applicability = wilson95(applicable, len(rows))
    ineligible = Counter(r.outer_ineligibility_reason for r in rows if r.status == "NOT_ASSESSED")
    inner_histogram = Counter()
    for row in rows:
        inner_histogram.update(row.inner_ineligibility_reason_counts)
    assessed_times = [t for r, t in zip(rows, timings) if r.status == "ASSESSED"]
    completed = state["status"] == "COMPLETE"
    shadow = state["shadow"]
    return {"schema_version": OUTPUT_SCHEMA, "canonical_cell_id": manifest.canonical_cell_id,
            "cell_status": state["status"], "CELL_EXECUTION_STARTED": "YES" if state["cell_execution_started"] else "NO",
            "software_fixture": state["software_fixture"], "eligible_outer_target": state["target"],
            "primary": manifest.primary, "promotion_eligible": manifest.primary and not state["software_fixture"],
            "raw_outer_attempts": len(rows), "eligible_outer_count": R, "rejection_count": x,
            "rejection_rate": x / R if R else None,
            "Wilson95_lower": interval[0] if interval else None,
            "Wilson95_upper": interval[1] if interval else None,
            "CELL_ACCEPTED": bool(completed and interval and interval[1] <= WILSON_UPPER_LIMIT),
            "outer_ineligibility_reason_counts": dict(ineligible),
            "applicability_rate": applicable / len(rows) if rows else None,
            "applicability_Wilson95": applicability,
            "numerical_failure_count": sum(r.reason_code in (
                "FIT_NUMERICAL_FAILURE", "FIT_BACKEND_FAILURE", "GENERATION_FAILURE", "STATISTIC_FAILURE",
                "TAIL_CERTIFICATION_FAILURE", "ORACLE_MISMATCH") for r in rows),
            "failure_reason_counts": dict(Counter(r.reason_code for r in rows if r.status == "FAILED")),
            "failure_reason": state["failure_reason"],
            "retry_burden": {"extra_raw_outers": len(rows) - R,
                             "extra_inner_attempts": sum(not a.eligible for a in inner)},
            "inner_ineligibility_histogram": dict(inner_histogram), "inner_attempts": len(inner),
            "inner_fit_attempts": sum(a.fit_calls for a in inner),
            "fast_fit_count": sum(r.observed_fit_calls + r.replicate_fit_calls for r in rows) if manifest.nb_composite else 0,
            "canonical_shadow_fit_count": shadow["canonical_shadow_fit_count"],
            "shadow_wall_seconds": shadow["shadow_wall_seconds"],
            "SHADOW_GATE": ("PASS" if shadow["entry_gate"] == shadow["periodic_gate"] == "PASS"
                            else "NOT_APPLICABLE" if not manifest.nb_composite else "FAIL"),
            "primary_wall_seconds": sum(timings),
            "total_wall_seconds": state["execution_wall_seconds"],
            "non_primary_overhead_wall_seconds": max(0.0, state["execution_wall_seconds"] - sum(timings) - shadow["shadow_wall_seconds"]),
            "median_wall_per_assessed_outer": float(np.median(assessed_times)) if assessed_times else None,
            "p95_wall_per_assessed_outer": float(np.percentile(assessed_times, 95)) if assessed_times else None,
            "peak_sampled_rss_bytes": peak_rss,
            "rss_sampling_method": "process RSS every 2ms including baseline; shared across cell workers",
            **BLOCKED_STAGES}


def run_cell(manifest, directory, *, fixture_target=None, max_new_units=None,
             hooks=None, shadow_factory=ShadowOracle, command="python -m experiments.distribution_gof.cp05_c run"):
    """Sequential raw prefix per cell; parallelism is exclusively between cells.

    Explicit short software fixtures cannot count toward scientific aggregation.
    A failed unit or shadow gate is terminal and is never replaced on resume.
    """
    invocation_start = time.perf_counter()
    fixture = fixture_target is not None
    target = manifest.R_C if not fixture else fixture_target
    if type(target) is not int or not 1 <= target <= (manifest.R_C - 1 if fixture else manifest.R_C):
        raise ValueError("invalid explicit software fixture target")
    if not fixture and (hooks is not None or shadow_factory is not ShadowOracle):
        raise ValueError("custom hooks/oracles are allowed only in software fixtures")
    if not fixture:
        from .source_identity import repository_identity
        repository_identity(manifest.source_sha)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units < 0):
        raise ValueError("max_new_units must be a nonnegative int")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # File creation is exclusive: two schedulers must never share one cell.
    lock = directory / ".execution.lock"
    try:
        descriptor = lock.open("x")
    except FileExistsError as exc:
        raise RuntimeError("cell is already locked; verify no active writer before removing its lock") from exc
    try:
        from .artifacts import write_bundle, validate_bundle
        if (directory / "digests.json").exists():
            validate_bundle(directory)
        state, rows, inner, timings = load_state(directory, manifest, target, fixture)
        previous_wall = state["execution_wall_seconds"]
        def checkpoint():
            state["execution_wall_seconds"] = previous_wall + time.perf_counter() - invocation_start
            save_state(directory, state)
        peak = 0
        with RSSPeak() as rss:
            if state["status"] not in ("COMPLETE", "FAILED", "FAILED_SHADOW_ORACLE"):
                state["status"] = "RUNNING"
                oracle = shadow_factory(manifest) if manifest.nb_composite else None
                if oracle is not None and state["shadow"]["entry_gate"] == "NOT_APPLICABLE":
                    state["shadow"] = oracle.entry()
                    if state["shadow"]["entry_gate"] != "PASS":
                        state.update(status="FAILED_SHADOW_ORACLE", failure_reason=state["shadow"]["failure_reason"])
                    checkpoint()
                eligible = sum(r.status == "ASSESSED" for r in rows)
                cap = target * (100 if manifest.nb_composite else 1)
                new_units = 0
                while state["status"] == "RUNNING" and eligible < target and len(rows) < cap:
                    if max_new_units is not None and new_units >= max_new_units:
                        state["status"] = "PAUSED"
                        break
                    index = len(rows)
                    if oracle is not None and periodic_due(index, state["shadow"]["checks"]):
                        row, seconds, error = oracle.periodic(index)
                        shadow = state["shadow"]
                        shadow["shadow_wall_seconds"] += seconds
                        if row is not None:
                            shadow["checks"].append(row)
                            shadow["canonical_shadow_fit_count"] += 1
                        if error or row is None or not row["passed"]:
                            shadow["periodic_gate"] = "FAIL"
                            shadow["failure_reason"] = error or row["failure_reason"]
                            shadow["failure_identity"] = {"role": "periodic_observed", "raw_outer_index": index,
                                                          "raw_inner_index": None}
                            state.update(status="FAILED_SHADOW_ORACLE", failure_reason=shadow["failure_reason"])
                        checkpoint()
                        if state["status"] != "RUNNING":
                            break
                    state["cell_execution_started"] = True
                    start = time.perf_counter()
                    result, attempts = _execute_outer(manifest, index, hooks, eligible)
                    elapsed = time.perf_counter() - start
                    state["segments"].append(save_segment(directory, result, attempts, elapsed))
                    rows.append(result)
                    inner.extend(attempts)
                    timings.append(elapsed)
                    eligible += result.status == "ASSESSED"
                    new_units += 1
                    if result.status == "FAILED":
                        state.update(status="FAILED", failure_reason=result.reason_code)
                    elif eligible == target:
                        state["status"] = "COMPLETE"
                    elif len(rows) == cap:
                        state.update(status="FAILED", failure_reason="OUTER_RETRY_CAP_EXHAUSTED")
                    checkpoint()
            peak = max(rss.peak, state.get("peak_sampled_rss_bytes", 0))
        state["peak_sampled_rss_bytes"] = max(peak, rss.peak)
        checkpoint()
        output = RunOutput(tuple(rows), tuple(inner))
        cell_summary = summary(manifest, state, rows, inner, timings, state["peak_sampled_rss_bytes"])
        write_bundle(directory, manifest, output, cell_summary, state["shadow"], command=command, timings=timings)
        return cell_summary
    finally:
        descriptor.close()
        lock.unlink(missing_ok=True)


def run_cells(manifests, root, *, workers=1, batch_size=1, **options):
    manifests = tuple(manifests)
    if type(workers) is not int or workers < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("workers/batch_size must be positive ints")
    identities = [m.canonical_cell_id for m in manifests]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate cell scheduling")
    manifests = tuple(replace(m, workers=workers, batch_size=batch_size) for m in manifests)
    def execute(manifest):
        return run_cell(manifest, Path(root) / manifest.directory_name, **options)
    outputs = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(manifests), batch_size):
            outputs.extend(executor.map(execute, manifests[offset:offset + batch_size]))
    return sorted(outputs, key=lambda s: s["canonical_cell_id"])
