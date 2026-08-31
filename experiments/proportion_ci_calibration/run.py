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
    DEFAULT_ENDPOINT_CACHE_ROOT,
    EXPERIMENT_VERSION,
    HARNESS_SCHEMA_VERSION,
    METHODS,
    STRESS_N,
    EndpointGridCache,
    calibrate_n,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPOSITORY_ROOT / "experiments" / "results"
CACHE_ROOT = (
    Path(tempfile.gettempdir())
    / "pymagicstats_cp06_cache"
    / CANDIDATE_SHA
    / HARNESS_SCHEMA_VERSION
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


def checkpoint_spec_hash(name: str, spec: CheckpointSpec | None = None) -> str:
    selected = checkpoint_spec(name) if spec is None else spec
    payload = {
        "checkpoint": name,
        "candidate_sha": CANDIDATE_SHA,
        "harness_schema_version": HARNESS_SCHEMA_VERSION,
        "n_values": selected.n_values,
        "linear_step": selected.linear_step,
        "expected_widths": selected.expected_widths,
        "oracle": selected.oracle,
        "include_base_grid": selected.include_base_grid,
        "alphas": ALPHAS,
        "methods": METHODS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def shard_semantic_hash(n: int, spec: CheckpointSpec) -> str:
    payload = {
        "candidate_sha": CANDIDATE_SHA,
        "harness_schema_version": HARNESS_SCHEMA_VERSION,
        "n": n,
        "linear_step": spec.linear_step,
        "expected_widths": spec.expected_widths,
        "oracle": spec.oracle,
        "include_base_grid": spec.include_base_grid,
        "alphas": ALPHAS,
        "methods": METHODS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_cache_provenance(
    checkpoint: str,
    n: int,
    spec: CheckpointSpec | None = None,
) -> dict[str, object]:
    selected = checkpoint_spec(checkpoint) if spec is None else spec
    return {
        "candidate_sha": CANDIDATE_SHA,
        "harness_schema_version": HARNESS_SCHEMA_VERSION,
        "checkpoint": checkpoint,
        "checkpoint_spec_sha256": checkpoint_spec_hash(checkpoint, selected),
        "shard_semantic_sha256": shard_semantic_hash(n, selected),
        "n": n,
    }


def checkpoint_cache_dir(checkpoint: str, spec: CheckpointSpec | None = None) -> Path:
    selected = checkpoint_spec(checkpoint) if spec is None else spec
    return CACHE_ROOT / checkpoint.lower() / checkpoint_spec_hash(checkpoint, selected)


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
        "wald_pathology_summaries": "wald_pathology_summary",
        "wald_pathology_worst_cases": "wald_pathology_worst_cases",
    }
    if name == "E":
        mapping["adversarial_minima"] = "adversarial_minima"
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
        "harness_schema_version": HARNESS_SCHEMA_VERSION,
        "checkpoint_spec_sha256": checkpoint_spec_hash(name, spec),
        "hashes": hashes,
    }
    metadata_path = RESULTS / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _calibrate_one(
    arguments: tuple[int, float | None, int, bool, bool, bool, str],
) -> dict[str, object]:
    (
        n,
        linear_step,
        batch_size,
        expected_widths,
        oracle,
        include_base_grid,
        endpoint_cache_root,
    ) = arguments
    return calibrate_n(
        n,
        linear_step=linear_step,
        expected_widths=expected_widths,
        oracle=oracle,
        batch_size=batch_size,
        include_base_grid=include_base_grid,
        endpoint_cache=EndpointGridCache(Path(endpoint_cache_root)),
    )


def save_shard_cache(
    path: Path,
    part: dict[str, object],
    provenance: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result_bytes = pickle.dumps(part, protocol=pickle.HIGHEST_PROTOCOL)
    payload = {
        "provenance": {
            **provenance,
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "interval_sha256": part["interval_hash"],
        },
        "result_bytes": result_bytes,
    }
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


def load_shard_cache(
    path: Path,
    expected: dict[str, object],
    *,
    allow_cross_checkpoint: bool = False,
) -> dict[str, object] | None:
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        provenance = payload["provenance"]
        result_bytes = payload["result_bytes"]
        required = (
            "candidate_sha",
            "harness_schema_version",
            "shard_semantic_sha256",
            "n",
        )
        if any(provenance.get(key) != expected.get(key) for key in required):
            return None
        if not allow_cross_checkpoint and (
            provenance.get("checkpoint") != expected.get("checkpoint")
            or provenance.get("checkpoint_spec_sha256")
            != expected.get("checkpoint_spec_sha256")
        ):
            return None
        if hashlib.sha256(result_bytes).hexdigest() != provenance.get("result_sha256"):
            return None
        part = pickle.loads(result_bytes)
        required_result_keys = {
            "n",
            "interval_hash",
            "summaries",
            "worst_cases",
            "adversarial_minima",
            "wald_pathology_summaries",
            "wald_pathology_worst_cases",
        }
        if not required_result_keys <= set(part):
            return None
        if part["n"] != expected["n"]:
            return None
        if part["interval_hash"] != provenance.get("interval_sha256"):
            return None
        return part
    except (OSError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
        return None


def run_checkpoint(name: str, *, workers: int, batch_size: int) -> None:
    spec = checkpoint_spec(name)
    n_values = spec.n_values

    started = time.perf_counter()
    checkpoint_cache = checkpoint_cache_dir(name, spec)
    checkpoint_cache.mkdir(parents=True, exist_ok=True)
    parts_by_n: dict[int, dict[str, object]] = {}
    missing: list[int] = []
    for n in n_values:
        cache_path = checkpoint_cache / f"n_{n:07d}.pickle"
        expected_provenance = build_cache_provenance(name, n, spec)
        source_path: Path | None = cache_path if cache_path.exists() else None
        source_checkpoint = name
        if source_path is None:
            for source in spec.resume_sources:
                candidate = checkpoint_cache_dir(source) / cache_path.name
                if candidate.exists():
                    source_path = candidate
                    source_checkpoint = source
                    break
        part = (
            None
            if source_path is None
            else load_shard_cache(
                source_path,
                expected_provenance,
                allow_cross_checkpoint=source_checkpoint != name,
            )
        )
        if part is not None:
            parts_by_n[n] = part
            if source_path != cache_path:
                save_shard_cache(cache_path, part, expected_provenance)
            print(
                f"CP06-{name} n={n} resumed from {source_checkpoint}",
                flush=True,
            )
        else:
            if source_path is not None:
                print(
                    f"CP06-{name} n={n} rejected incompatible cache from "
                    f"{source_checkpoint}",
                    flush=True,
                )
            missing.append(n)

    arguments = [
        (
            n,
            spec.linear_step,
            batch_size,
            spec.expected_widths,
            spec.oracle,
            spec.include_base_grid,
            str(DEFAULT_ENDPOINT_CACHE_ROOT),
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
            save_shard_cache(
                checkpoint_cache / f"n_{n:07d}.pickle",
                part,
                build_cache_provenance(name, n, spec),
            )
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
