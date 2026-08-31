"""Command-line checkpoints for the CP-06 proportion-CI calibration."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import tempfile
import time

import mpmath
import numpy
import pandas
import pyarrow
import scipy
import statsmodels

from experiments.proportion_ci_calibration.harness import (
    ALPHAS,
    CANDIDATE_SHA,
    CP04_DOCUMENT_SHA,
    EXPERIMENT_VERSION,
    METHODS,
    STRESS_N,
    calibrate_n,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPOSITORY_ROOT / "experiments" / "results"
CACHE_ROOT = (
    Path(tempfile.gettempdir())
    / "pymagicstats_cp06_cache"
    / CANDIDATE_SHA
)


@dataclass(frozen=True)
class CheckpointSpec:
    n_values: tuple[int, ...]
    linear_step: float | None
    expected_widths: bool
    oracle: bool
    include_base_grid: bool
    resume_sources: tuple[str, ...] = ()


def checkpoint_spec(name: str) -> CheckpointSpec:
    """Return the frozen CP-04 domain assigned to one deterministic checkpoint."""

    if name == "A":
        return CheckpointSpec((1, 2, 5, 10, 20, 50), 0.01, True, True, True)
    if name == "B":
        return CheckpointSpec(tuple(range(1, 201)), 0.0001, True, True, True)
    if name == "C":
        return CheckpointSpec(
            tuple(range(1, 5_001)),
            0.0001,
            True,
            True,
            True,
            ("B",),
        )
    if name == "D":
        return CheckpointSpec(STRESS_N, 0.0001, True, True, True)
    if name == "E":
        return CheckpointSpec(
            tuple(range(1, 5_001)) + STRESS_N,
            None,
            False,
            False,
            False,
        )
    if name == "SMOKE":
        return CheckpointSpec((1, 2, 5, 10, 97, 101), 0.01, True, True, True)
    raise ValueError(f"unknown checkpoint: {name}")


def _merge_frames(parts: list[dict[str, object]], key: str) -> pandas.DataFrame:
    return pandas.DataFrame([row for part in parts for row in part.get(key, [])])


def _write_checkpoint(
    name: str,
    parts: list[dict[str, object]],
    elapsed: float,
    *,
    workers: int,
    batch_size: int,
    spec: CheckpointSpec,
) -> None:
    prefix = f"proportion_ci_cp06_{name.lower()}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    mapping = {
        "summaries": "coverage_summary",
        "event_regimes": "event_regimes",
        "worst_cases": "worst_cases",
        "invariants": "invariants",
        "nesting": "nesting",
        "oracles": "oracles",
        "high_precision_triggers": "high_precision_triggers",
    }
    hashes: dict[str, str] = {}
    for key, suffix in mapping.items():
        path = RESULTS / f"{prefix}_{suffix}.parquet"
        _merge_frames(parts, key).to_parquet(path, index=False)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    inventory = pandas.DataFrame(
        [
            {
                "n": part["n"],
                "interval_rows": part["interval_rows"],
                "sha256": part["interval_hash"],
            }
            for part in parts
        ]
    )
    inventory_path = RESULTS / f"{prefix}_interval_inventory.parquet"
    inventory.to_parquet(inventory_path, index=False)
    hashes[inventory_path.name] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    metadata = {
        "checkpoint": name,
        "candidate_sha": CANDIDATE_SHA,
        "cp04_document_sha": CP04_DOCUMENT_SHA,
        "experiment_version": EXPERIMENT_VERSION,
        "n_values": [part["n"] for part in parts],
        "alphas": list(ALPHAS),
        "methods": list(METHODS),
        "elapsed_seconds": elapsed,
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "mpmath": mpmath.__version__,
            "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "backend": "CPU/SciPy float64",
        "workers": workers,
        "batch_size": batch_size,
        "expected_widths": spec.expected_widths,
        "oracle": spec.oracle,
        "include_base_grid": spec.include_base_grid,
        "linear_step": spec.linear_step,
        "hashes": hashes,
    }
    metadata_path = RESULTS / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _calibrate_one(
    arguments: tuple[int, float | None, int, bool, bool, bool],
) -> dict[str, object]:
    n, linear_step, batch_size, expected_widths, oracle, include_base_grid = arguments
    return calibrate_n(
        n,
        linear_step=linear_step,
        expected_widths=expected_widths,
        oracle=oracle,
        batch_size=batch_size,
        include_base_grid=include_base_grid,
    )


def _save_cache(path: Path, part: dict[str, object]) -> None:
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("wb") as stream:
        pickle.dump(part, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


def run_checkpoint(name: str, *, workers: int, batch_size: int) -> None:
    spec = checkpoint_spec(name)
    n_values = spec.n_values

    started = time.perf_counter()
    checkpoint_cache = CACHE_ROOT / name.lower()
    checkpoint_cache.mkdir(parents=True, exist_ok=True)
    parts_by_n: dict[int, dict[str, object]] = {}
    missing: list[int] = []
    for n in n_values:
        cache_path = checkpoint_cache / f"n_{n:07d}.pickle"
        source_path = cache_path
        if not source_path.exists():
            for source in spec.resume_sources:
                candidate = CACHE_ROOT / source.lower() / cache_path.name
                if candidate.exists():
                    source_path = candidate
                    break
        if source_path.exists():
            with source_path.open("rb") as stream:
                parts_by_n[n] = pickle.load(stream)
            if source_path != cache_path:
                _save_cache(cache_path, parts_by_n[n])
            print(
                f"CP06-{name} n={n} resumed from {source_path.parent.name}",
                flush=True,
            )
        else:
            missing.append(n)

    arguments = [
        (
            n,
            spec.linear_step,
            batch_size,
            spec.expected_widths,
            spec.oracle,
            spec.include_base_grid,
        )
        for n in missing
    ]
    if workers == 1:
        computed = map(_calibrate_one, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        computed = executor.map(_calibrate_one, arguments, chunksize=1)
    try:
        for n, part in zip(missing, computed):
            parts_by_n[n] = part
            _save_cache(checkpoint_cache / f"n_{n:07d}.pickle", part)
            print(f"CP06-{name} n={n} complete", flush=True)
    finally:
        if workers != 1:
            executor.shutdown()

    parts = [parts_by_n[n] for n in n_values]
    elapsed = time.perf_counter() - started
    _write_checkpoint(
        name,
        parts,
        elapsed,
        workers=workers,
        batch_size=batch_size,
        spec=spec,
    )
    print(f"CP06-{name} complete in {elapsed:.3f}s", flush=True)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", choices=("A", "B", "C", "D", "E", "SMOKE"))
    parser.add_argument("--workers", type=_positive_integer, default=1)
    parser.add_argument("--batch-size", type=_positive_integer, default=256)
    args = parser.parse_args(argv)
    run_checkpoint(
        args.checkpoint,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
