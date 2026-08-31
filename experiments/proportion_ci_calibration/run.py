"""Command-line checkpoints for the CP-06 proportion-CI calibration."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
    calibrate_n,
    write_calibration_result,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPOSITORY_ROOT / "experiments" / "results"
CACHE_ROOT = (
    Path(tempfile.gettempdir())
    / "pymagicstats_cp06_cache"
    / CANDIDATE_SHA
)


def _merge_frames(parts: list[dict[str, object]], key: str) -> pandas.DataFrame:
    return pandas.DataFrame([row for part in parts for row in part.get(key, [])])


def _write_checkpoint(
    name: str,
    parts: list[dict[str, object]],
    elapsed: float,
    *,
    workers: int,
    batch_size: int,
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
        "hashes": hashes,
    }
    metadata_path = RESULTS / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _calibrate_one(arguments: tuple[int, float | None, int]) -> dict[str, object]:
    n, linear_step, batch_size = arguments
    return calibrate_n(
        n,
        linear_step=linear_step,
        expected_widths=True,
        oracle=True,
        batch_size=batch_size,
    )


def _save_cache(path: Path, part: dict[str, object]) -> None:
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("wb") as stream:
        pickle.dump(part, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


def run_checkpoint(name: str, *, workers: int, batch_size: int) -> None:
    if name == "A":
        n_values = (1, 2, 5, 10, 20, 50)
        linear_step = 0.01
    elif name == "B":
        n_values = tuple(range(1, 201))
        linear_step = 0.0001
    elif name == "SMOKE":
        n_values = (1, 2, 5, 10, 97, 101)
        linear_step = 0.01
    else:
        raise ValueError("unknown checkpoint")

    started = time.perf_counter()
    checkpoint_cache = CACHE_ROOT / name.lower()
    checkpoint_cache.mkdir(parents=True, exist_ok=True)
    parts_by_n: dict[int, dict[str, object]] = {}
    missing: list[int] = []
    for n in n_values:
        cache_path = checkpoint_cache / f"n_{n:07d}.pickle"
        if cache_path.exists():
            with cache_path.open("rb") as stream:
                parts_by_n[n] = pickle.load(stream)
            print(f"CP06-{name} n={n} resumed", flush=True)
        else:
            missing.append(n)

    arguments = [(n, linear_step, batch_size) for n in missing]
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
    )
    print(f"CP06-{name} complete in {elapsed:.3f}s", flush=True)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", choices=("A", "B", "SMOKE"))
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
