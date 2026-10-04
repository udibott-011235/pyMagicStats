"""Frozen-payload, single-pass PERF-01 orchestration. Research only."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np

from ..equivalence_preregistration import NB_TRANSFORM_ATOL, NB_OBJECTIVE_RTOL, logit, nb_objective_agreement
from ..r11_decision_equivalence.preflight import git, ordered_records, verify_workload
from ..r11_reference_workload.codec import SamplePayload, canonical_json, require, sha256
from ..r11_reference_workload.contract import SOURCE_PATH
from .contract import (BASE_SHA, BASE_TREE, B_R11, BUNDLE_NAMES, FAST_CPU_ENGINE,
                       ORDER, PACKAGE_PATH, REFERENCE_WORKLOAD_SHA256, SOURCE_OUTERS,
                       TEST_PATH, TOTAL_RECORDS)
from .engines import CudaEngine, canonical_cpu, compatible_batches, measure
from .fast_cpu import failed, fit_negative_binomial


@dataclass(frozen=True)
class Record:
    descriptor: dict
    payload: SamplePayload
    sample: np.ndarray


def repository_identity(repository):
    """Exact starting commit plus additive, isolated PERF-01 changes only."""
    repository = Path(repository).resolve()
    require(Path(git(repository, "rev-parse", "--show-toplevel")).resolve() == repository,
            "repository root required")
    require(git(repository, "rev-parse", BASE_SHA + "^{tree}") == BASE_TREE, "base tree mismatch")
    git(repository, "merge-base", "--is-ancestor", BASE_SHA, "HEAD")
    require(not git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
            "PERF-01 requires a clean committed checkout")
    changes = git(repository, "diff", "--no-renames", "--name-status", BASE_SHA, "HEAD")
    for line in changes.splitlines():
        status, path = line.split("\t")
        require(status == "A" and (path.startswith(PACKAGE_PATH + "/") or path == TEST_PATH),
                "change outside additive PERF-01 scope: " + path)
    return {"BASE_SHA": BASE_SHA, "BASE_TREE": BASE_TREE,
            "CANDIDATE_SHA": git(repository, "rev-parse", "HEAD"),
            "CANDIDATE_TREE": git(repository, "rev-parse", "HEAD^{tree}"),
            "PRODUCTION_FILES_MODIFIED": "NO"}


def records_from_verified_workload(workload):
    """Stored accepted records only, with mandatory counts/order/payload identity."""
    require(type(workload["B_R11"]) is int and workload["B_R11"] == B_R11,
            "B_R11 count mismatch")
    require(len(workload["outers"]) == SOURCE_OUTERS, "source outer count mismatch")
    records, identities = [], set()
    for outer in workload["outers"]:
        require(outer["observed"] is not None and len(outer["accepted"]) == B_R11,
                "incomplete R11 outer")
    for outer, descriptor, serialized in ordered_records(workload):
        require(outer["cell_id"].split("|", 1)[0] == "negative_binomial", "non-NB record")
        require(descriptor["identity"] not in identities, "duplicate sample identity")
        identities.add(descriptor["identity"])
        payload = SamplePayload.from_serialized(serialized)
        sample = payload.deserialize()
        require(len(payload.shape) == 1 and payload.shape[0] > 0, "NB vector required")
        records.append(Record(descriptor, payload, sample))
    require(len(records) == TOTAL_RECORDS, "exact record count required")
    return tuple(records)


def load_records(repository, workload_path, archive_path, crossings_path):
    data = Path(workload_path).read_bytes()
    workload = verify_workload(data, (Path(repository) / SOURCE_PATH).read_bytes(),
                               Path(archive_path).read_bytes(), Path(crossings_path).read_bytes())
    return records_from_verified_workload(workload)


def validate_samples(records):
    for record in records:
        payload = record.payload
        require(record.sample.dtype.str == payload.dtype_str and record.sample.shape == payload.shape
                and sha256(record.sample.tobytes(order="C")) == payload.sample_digest,
                "sample identity drift")


def evidence(record, outcome):
    if type(outcome) is not dict:
        outcome = failed("malformed fit result: " + repr(outcome))
    return {**record.descriptor, "sample_identity": record.descriptor["identity"],
            "result": outcome}


def compare_record(reference, candidate):
    """Strict accepted FIT gates. No flat-objective exception without downstream."""
    a, b = reference["result"], candidate["result"]
    identity = all(key in reference and key in candidate and
                   type(reference[key]) is type(candidate[key]) and reference[key] == candidate[key] for key in
                   ("identity", "sample_identity", "sample_digest", "payload_dtype", "payload_shape",
                    "seed_identity", "cell_id", "raw_outer_index", "raw_inner_index", "accepted_ordinal"))
    classification_match = (a.get("classification") == b.get("classification") and a.get("classification") in
                            ("ELIGIBLE", "ALL_ZERO_NON_IDENTIFYING", "VARIANCE_NOT_GREATER_THAN_MEAN"))
    convergence_match = (type(a.get("converged")) is bool and type(b.get("converged")) is bool
                         and a["converged"] == b["converged"])
    successful_fits = (a.get("classification") == b.get("classification") == "ELIGIBLE"
                       and a.get("converged") is True and b.get("converged") is True)
    parameter_gate = log_likelihood_gate = False
    reason = None
    try:
        ap, bp = a["parameters"], b["parameters"]
        values = [ap["r"], ap["p"], bp["r"], bp["p"], a["log_likelihood"], b["log_likelihood"]]
        require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "nonfinite fit evidence")
        parameter_gate = (abs(math.log(ap["r"]) - math.log(bp["r"])) <= NB_TRANSFORM_ATOL
                          and abs(logit(ap["p"]) - logit(bp["p"])) <= NB_TRANSFORM_ATOL)
        log_likelihood_gate = nb_objective_agreement(a["log_likelihood"], b["log_likelihood"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        reason = str(exc)
    gates = {"sample_identity_match": identity, "classification_match": classification_match,
             "convergence_match": convergence_match, "parameter_gate": parameter_gate,
             "log_likelihood_gate": log_likelihood_gate, "successful_fits": successful_fits}
    ok = all(gates.values()) and a.get("failure_reason") is None and b.get("failure_reason") is None
    if not ok:
        reason = "; ".join(filter(None, (reason, a.get("failure_reason"), b.get("failure_reason"),
                                        ",".join(key for key, passed in gates.items() if not passed))))
    return {**{k: v for k, v in reference.items() if k != "result"}, **gates,
            "scientifically_equivalent": ok, "failure_reason": reason,
            "candidate_sample_identity": candidate.get("sample_identity"),
            "candidate_sample_digest": candidate.get("sample_digest"),
            "canonical_result": a, "candidate_result": b}


def qualify(reference, candidate):
    require(len(reference) == len(candidate) == TOTAL_RECORDS, "qualification requires all 2400 records")
    require(len({r["identity"] for r in reference}) == TOTAL_RECORDS
            and len({r["identity"] for r in candidate}) == TOTAL_RECORDS,
            "qualification requires unique record identities")
    rows = [compare_record(a, b) for a, b in zip(reference, candidate)]
    return {"FAST_CPU_SCIENTIFIC_ELIGIBLE": all(row["scientifically_equivalent"] for row in rows),
            "records": TOTAL_RECORDS, "failures": sum(not row["scientifically_equivalent"] for row in rows),
            "NB_TRANSFORM_ATOL": NB_TRANSFORM_ATOL, "NB_OBJECTIVE_RTOL": NB_OBJECTIVE_RTOL,
            "flat_objective_exception_used": False, "rows": rows}


def _timing(engine, batch_size, metrics, outcomes, *, complete, reason=None, stages=None, batches=()):
    count, wall = len(outcomes), metrics.get("wall_seconds")
    usable = complete and count == TOTAL_RECORDS and wall is not None and wall > 0
    return {"engine": engine, "batch_size": batch_size, "records": count,
            "requested_records": TOTAL_RECORDS, **metrics,
            "total_host_wall": wall, "records_per_second": count / wall if usable else None,
            "milliseconds_per_record": 1000 * wall / count if usable else None,
            "gpu_device_seconds": None, "gpu_memory_peak_bytes": None,
            "host_to_device": None, "device_compute": None, "device_to_host": None,
            **(stages or {}), "configuration_complete": bool(complete and count == TOTAL_RECORDS),
            "failure_reason": reason, "actual_batches": list(batches),
            "convergence_failures": sum(row["result"].get("converged") is not True for row in outcomes) if outcomes else None,
            "classification_mismatches": None, "numerical_gate_failures": None}


def run_cpu(records, engine, fit):
    validate_samples(records)
    # Warm-up uses one stored payload, never enters timing or outcome counts.
    fit(records[0].sample)
    outcomes = []
    def operation():
        for record in records:
            try:
                outcome = fit(record.sample)
            except Exception as exc:
                outcome = failed(type(exc).__name__ + ": " + str(exc))
            outcomes.append(evidence(record, outcome))
    _, metrics, error = measure(operation)
    validate_samples(records)
    return _timing(engine, 1, metrics, outcomes, complete=error is None,
                   reason=str(error) if error else None), outcomes


def run_cuda(records, batch_size, cuda):
    validate_samples(records)
    batches = tuple(compatible_batches(records, batch_size))
    cuda.fit_batch(batches[0][:min(32, len(batches[0]))])
    cuda.reset_memory_accounting()  # warm-up reservations are not measurement peaks
    outcomes, batch_details = [], []
    stages = dict.fromkeys(("host_to_device", "device_compute", "device_to_host", "gpu_device_seconds"), 0.0)
    stages["gpu_memory_peak_bytes"] = cuda.memory_peak
    def operation():
        for batch in batches:
            batch_details.append({"records": len(batch), "shape": list(batch[0].payload.shape),
                                  "dtype": batch[0].payload.dtype_str,
                                  "first_identity": batch[0].descriptor["identity"],
                                  "last_identity": batch[-1].descriptor["identity"]})
            try:
                results, measured = cuda.fit_batch(batch)
                require(len(results) == len(batch), "CUDA result count mismatch")
            except Exception as exc:
                outcomes.extend(evidence(record, failed(type(exc).__name__ + ": " + str(exc))) for record in batch)
                # No smaller-batch retry, no new pass. Remaining records unattempted.
                raise
            outcomes.extend(evidence(record, result) for record, result in zip(batch, results))
            for key in ("host_to_device", "device_compute", "device_to_host", "gpu_device_seconds"):
                stages[key] += measured[key]
            stages["gpu_memory_peak_bytes"] = max(stages["gpu_memory_peak_bytes"], measured["gpu_memory_peak_bytes"])
    _, metrics, error = measure(operation, synchronize=cuda.synchronize)
    validate_samples(records)
    # Partial stage totals must never look like complete phase measurements.
    if error is not None:
        stages = {key: None for key in stages}
    stages.update(gpu_device_seconds_method="sum of CUDA events across synchronous pipeline stages; includes submission gaps",
                  gpu_memory_peak_method="peak default CuPy pool reserved bytes after stages; excludes non-pool allocations")
    timing = _timing("CUDA", batch_size, metrics, outcomes, complete=error is None,
                     reason=str(error) if error else None, stages=stages, batches=batch_details)
    timing["batch_size_limitation"] = ("contiguous shape/dtype runs and final tails; no padding/reordering"
                                       if isinstance(batch_size, int) and any(len(b) < batch_size for b in batches) else None)
    return timing, outcomes


def unavailable(engine, batch_size, reason):
    metrics = {"wall_seconds": None, "cpu_process_seconds": None, "peak_rss_bytes": None}
    return _timing(engine, batch_size, metrics, [], complete=False, reason=reason), []


def apply_counts(timing, reference, outcomes):
    rows = [compare_record(a, b) for a, b in zip(reference, outcomes)]
    timing["classification_mismatches"] = sum(not r["classification_match"] for r in rows) if rows else None
    timing["numerical_gate_failures"] = sum(not (r["parameter_gate"] and r["log_likelihood_gate"]) for r in rows) if rows else None
    timing["scientific_failures"] = sum(not r["scientifically_equivalent"] for r in rows) if rows else None
    timing["scientific_eligible"] = (timing["configuration_complete"] and len(rows) == TOTAL_RECORDS
                                      and timing["scientific_failures"] == 0)
    return rows


def summary(identity, timings, eligibility):
    by_config = {(t["engine"], t["batch_size"]): t for t in timings}
    a, b = by_config[ORDER[0]], by_config[ORDER[1]]
    valid_cuda = [t for t in timings if t["engine"] == "CUDA" and t.get("scientific_eligible")
                  and t["configuration_complete"] and (t["wall_seconds"] or 0) > 0]
    best = min(valid_cuda, key=lambda t: t["wall_seconds"]) if valid_cuda else None
    eligible = eligibility["FAST_CPU_SCIENTIFIC_ELIGIBLE"]
    def ratio(x, y):
        return x / y if x is not None and y is not None and x > 0 and y > 0 else None
    result = {**identity, "WORKLOAD_SHA256": REFERENCE_WORKLOAD_SHA256, "RECORD_COUNT": TOTAL_RECORDS,
              "CANONICAL_CPU_WALL_SECONDS": a["wall_seconds"], "CANONICAL_CPU_RECORDS_PER_SECOND": a["records_per_second"],
              "FAST_CPU_SCIENTIFIC_ELIGIBLE": eligible, "FAST_CPU_WALL_SECONDS": b["wall_seconds"],
              "FAST_CPU_RECORDS_PER_SECOND": b["records_per_second"], "FAST_CPU_TIMINGS_DIAGNOSTIC_ONLY": not eligible,
              "CUDA_BEST_BATCH": best["batch_size"] if best else None,
              "CUDA_BEST_WALL_SECONDS": best["wall_seconds"] if best else None,
              "CUDA_BEST_RECORDS_PER_SECOND": best["records_per_second"] if best else None,
              "SPEEDUP_FAST_CPU_VS_CANONICAL": ratio(a["wall_seconds"], b["wall_seconds"]) if eligible else None,
              "SPEEDUP_CUDA_VS_CANONICAL": ratio(a["wall_seconds"], best["wall_seconds"]) if best else None,
              "SPEEDUP_CUDA_VS_FAST_CPU": ratio(b["wall_seconds"], best["wall_seconds"]) if eligible and best else None,
              "SCIENTIFIC_FAILURES": {f"{t['engine']}:{t['batch_size']}": t.get("scientific_failures") for t in timings},
              "BENCHMARK_COMPLETE": all(by_config.get(key, {}).get("configuration_complete", False) for key in ORDER),
              "RANKING_METRIC": "TOTAL HOST WALL TIME", "ARCHITECTURAL_DECISION": None}
    for size in (1, 32, 128, 512, "full-compatible-batch"):
        suffix = str(size).upper().replace("-", "_")
        result[f"CUDA_BATCH_{suffix}_WALL_SECONDS"] = by_config[("CUDA", size)]["wall_seconds"]
    result["SPEEDUP_C1_VS_A"] = ratio(a["wall_seconds"], by_config[("CUDA", 1)]["wall_seconds"]) if by_config[("CUDA", 1)].get("scientific_eligible") else None
    result["diagnostic_raw_speedup_fast_cpu_vs_canonical"] = ratio(a["wall_seconds"], b["wall_seconds"])
    return result


def environment():
    import scipy
    import psutil
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "psutil": psutil.__version__, "platform": platform.platform(),
            "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False), "ram_bytes": psutil.virtual_memory().total,
            "thread_settings": {key: os.environ.get(key) for key in
                                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def json_safe(value):
    """Preserve failed nonfinite results explicitly without invalid JSON literals."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_bundle(output, documents):
    digests = {}
    for name in BUNDLE_NAMES:
        data = canonical_json(json_safe(documents[name]))
        with (Path(output) / name).open("xb") as stream:
            stream.write(data)
        digests[name] = sha256(data)
    with (Path(output) / "digests.json").open("xb") as stream:
        stream.write(canonical_json({"algorithm": "sha256", "files": digests}))


def execute(repository, workload_path, archive_path, crossings_path, output, *, cuda_requested=False):
    """Explicit real execution entry point. Tests use only mocked/literal data."""
    identity = repository_identity(repository)
    records = load_records(repository, workload_path, archive_path, crossings_path)
    from ..r11_decision_equivalence.runtime import verify_loaded_modules
    verify_loaded_modules(Path(repository).resolve())
    validate_samples(records)
    require(len(records) == TOTAL_RECORDS, "exact record count required")
    output = Path(output).resolve()
    require(not output.is_relative_to(Path(repository).resolve()), "bundle must be outside committed checkout")
    output.mkdir(parents=True, exist_ok=False)  # refuse campaign repetition/overwrite
    env = environment()
    timings, all_outcomes = [], []
    for engine, factory in (("CANONICAL_CPU", canonical_cpu), (FAST_CPU_ENGINE, lambda: fit_negative_binomial)):
        try:
            timing, outcomes = run_cpu(records, engine, factory())
        except Exception as exc:
            timing, outcomes = unavailable(engine, 1, type(exc).__name__ + ": " + str(exc))
        timings.append(timing)
        all_outcomes.append(outcomes)
    reference, fast = all_outcomes
    eligibility = qualify(reference, fast) if len(reference) == len(fast) == TOTAL_RECORDS else {
        "FAST_CPU_SCIENTIFIC_ELIGIBLE": False, "records": len(fast), "failures": None,
        "failure_reason": "incomplete canonical/fast CPU evidence", "rows": []}
    apply_counts(timings[0], reference, reference)
    apply_counts(timings[1], reference, fast)
    cuda, cuda_error = None, "CUDA execution not requested"
    if cuda_requested:
        try:
            cuda = CudaEngine()
            env["cuda"] = cuda.environment()
        except Exception as exc:
            cuda = None
            cuda_error = type(exc).__name__ + ": " + str(exc)
    cuda_gate_rows = {}
    for _, batch in ORDER[2:]:
        if cuda is None:
            timing, outcomes = unavailable("CUDA", batch, cuda_error)
        else:
            try:
                timing, outcomes = run_cuda(records, batch, cuda)
            except Exception as exc:
                timing, outcomes = unavailable("CUDA", batch, type(exc).__name__ + ": " + str(exc))
        cuda_gate_rows[str(batch)] = apply_counts(timing, reference, outcomes)
        timings.append(timing)
        all_outcomes.append(outcomes)
    # Freeze implementation and all input payloads throughout this invocation.
    provenance_error = None
    try:
        require(repository_identity(repository) == identity, "implementation changed during execution")
        verify_loaded_modules(Path(repository).resolve())
        validate_samples(records)
    except Exception as exc:
        provenance_error = type(exc).__name__ + ": " + str(exc)
        eligibility["FAST_CPU_SCIENTIFIC_ELIGIBLE"] = False
        eligibility["provenance_failure"] = provenance_error
        for timing in timings:
            timing["scientific_eligible"] = False
    result = summary(identity, timings, eligibility)
    result["provenance_failure"] = provenance_error
    if provenance_error:
        result["BENCHMARK_COMPLETE"] = False
    manifest = {**identity, "schema_version": "cp05-c2d-perf01-nb-fitting-v1",
                "WORKLOAD_SHA256": REFERENCE_WORKLOAD_SHA256, "RECORD_COUNT": TOTAL_RECORDS,
                "SOURCE_OUTERS": SOURCE_OUTERS, "B_R11": B_R11, "PRODUCTION_BACKEND": "NO",
                "order": [{"engine": engine, "batch_size": batch} for engine, batch in ORDER],
                "warmup_cpu_records": 1, "warmup_cuda_records": "first min(32, first compatible batch)",
                "measurement_passes_per_configuration": 1, "cuda_requested": cuda_requested,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "FIT only; strict accepted parameter/objective gates; no downstream exception"}
    write_bundle(output, {"manifest.json": manifest, "scientific_eligibility.json": eligibility,
                         "timings.json": timings, "environment.json": env, "summary.json": result,
                         "records_summary.json": {"records": [r.descriptor for r in records],
                                                  "configurations": [{"engine": t["engine"], "batch_size": t["batch_size"],
                                                                      "outcomes": rows} for t, rows in zip(timings, all_outcomes)],
                                                  "cuda_fit_gates": cuda_gate_rows}})
    return result
