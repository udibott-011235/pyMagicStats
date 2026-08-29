"""Small stage-level CPU versus optional GPU/hybrid benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Sequence

import pandas as pd

from .backends import resolve_backend
from .metrics import REPLICATE_COLUMNS, MethodExecutor, evaluate_batch
from .scenarios import select_cells
from .seeds import derive_seed
from .storage import write_frame_atomic, write_json_atomic


def benchmark_backend(
    backend_name: str,
    *,
    scenario_id: str,
    n: int,
    replicates: int,
    master_seed: int,
    workers: int,
) -> dict[str, object]:
    cells = select_cells((scenario_id,), (n,))
    cell = cells[0]
    backend = resolve_backend(backend_name)
    replicate_ids = tuple(range(replicates))
    seeds = tuple(
        derive_seed(master_seed, scenario_id, 0, replicate_id)
        for replicate_id in replicate_ids
    )
    timings: dict[str, float] = {}
    started_total = time.perf_counter()
    started = time.perf_counter()
    native = backend.generate_native(cell.scenario, n, seeds)
    timings["sample_generation_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    diagnostics = backend.diagnostics(native)
    timings["diagnostics_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    samples = backend.to_cpu(native)
    timings["device_to_host_seconds"] = time.perf_counter() - started
    with MethodExecutor(workers) as executor:
        records, method_timings = evaluate_batch(
            samples,
            diagnostics,
            cell.scenario,
            replicate_ids,
            seeds,
            shard_id=0,
            num_shards=1,
            alpha=0.05,
            confidence_level=0.95,
            generation_backend=native.engine,
            executor=executor,
        )
    timings.update(method_timings)
    frame = pd.DataFrame.from_records(records, columns=REPLICATE_COLUMNS)
    with tempfile.TemporaryDirectory(prefix="el-vs-t-benchmark-") as temporary:
        started = time.perf_counter()
        write_frame_atomic(
            Path(temporary) / "benchmark.csv.gz", frame, "csv.gz", None
        )
        timings["serialization_seconds"] = time.perf_counter() - started
    total = time.perf_counter() - started_total
    return {
        "status": "measured",
        "requested_backend": backend_name,
        "generation_backend": native.engine,
        "backend_metadata": backend.info.to_dict(),
        "scenario_id": scenario_id,
        "n": n,
        "replicates": replicates,
        "workers": workers,
        "stage_timings_seconds": timings,
        "end_to_end_seconds": total,
        "finite_simulation_inputs": bool(pd.notna(frame["sample_mean"]).all()),
        "valid_schema": list(frame.columns) == list(REPLICATE_COLUMNS),
    }


def run_benchmark(
    *,
    scenario_id: str = "normal",
    n: int = 30,
    replicates: int = 20,
    master_seed: int = 20260829,
    workers: int = 1,
) -> dict[str, object]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    results = [
        benchmark_backend(
            "cpu",
            scenario_id=scenario_id,
            n=n,
            replicates=replicates,
            master_seed=master_seed,
            workers=workers,
        )
    ]
    try:
        gpu = benchmark_backend(
            "gpu",
            scenario_id=scenario_id,
            n=n,
            replicates=replicates,
            master_seed=master_seed,
            workers=workers,
        )
    except RuntimeError as error:
        gpu = {
            "status": "unavailable",
            "requested_backend": "gpu",
            "reason": str(error),
        }
    results.append(gpu)
    comparison: dict[str, object] = {
        "claim": "No acceleration claim is made unless both paths were measured.",
    }
    if gpu["status"] == "measured":
        cpu_seconds = float(results[0]["end_to_end_seconds"])
        gpu_seconds = float(gpu["end_to_end_seconds"])
        comparison.update(
            {
                "cpu_over_gpu_end_to_end_ratio": cpu_seconds / gpu_seconds,
                "measured_faster_path": "gpu-hybrid" if gpu_seconds < cpu_seconds else "cpu",
                "claim": "Measured on this workload only; not a scientific-result comparison.",
            }
        )
    return {
        "schema_version": "el-vs-t-benchmark-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "comparison": comparison,
        "network_used": False,
        "llm_or_external_api_used": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark stage timings for CPU and GPU/hybrid when available."
    )
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--master-seed", type=int, default=20260829)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_benchmark(
        scenario_id=arguments.scenario,
        n=arguments.sample_size,
        replicates=arguments.replicates,
        master_seed=arguments.master_seed,
        workers=arguments.workers,
    )
    if arguments.output is not None:
        write_json_atomic(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
