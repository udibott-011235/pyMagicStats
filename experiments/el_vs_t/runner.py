"""Resumable, sharded runner for the paired EL-versus-t calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Iterable, Mapping

import pandas as pd
import scipy

from pyMagicStat.assumptions.models import Estimand, InferenceDesign
from pyMagicStat.inference.capabilities import capability_for

from .backends import SampleBackend, resolve_backend
from .metrics import (
    EL_METHOD_VERSION,
    REPLICATE_COLUMNS,
    REPLICATE_SCHEMA_VERSION,
    T_METHOD_VERSION,
    MethodExecutor,
    evaluate_batch,
)
from .scenarios import (
    ExperimentCell,
    active_holdout_policy,
    registry_digest,
    select_cells,
)
from .seeds import (
    SEED_DERIVATION_SCHEME,
    derive_seed,
    owned_replicate_ids,
    replicate_blocks,
)
from .storage import (
    FORMAT_EXTENSIONS,
    read_json,
    resolve_storage_format,
    sha256_file,
    write_frame_atomic,
    write_json_atomic,
)


RUN_SCHEMA_VERSION = "el-vs-t-run-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    output: Path
    replicates_per_cell: int
    master_seed: int = 20260829
    backend: str = "auto"
    workers: int = 1
    batch_size: int = 2048
    shard_id: int = 0
    num_shards: int = 1
    alpha: float = 0.05
    confidence_level: float = 0.95
    storage_format: str = "auto"
    force: bool = False
    scenario_ids: tuple[str, ...] | None = None
    sample_sizes: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RunOutcome:
    computed_blocks: int
    skipped_blocks: int
    rows_computed: int
    cells_completed: int
    output: str
    shard_id: int
    run_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_head() -> tuple[str, bool]:
    executable = shutil.which("git")
    if executable is None and platform.system() == "Windows":
        candidate = Path(r"C:\Program Files\Git\cmd\git.exe")
        executable = str(candidate) if candidate.is_file() else None
    if executable is None:
        return "unavailable", True
    try:
        head = subprocess.run(
            [executable, "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [executable, "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return head, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def _method_metadata() -> dict[str, object]:
    el_path = REPO_ROOT / "pyMagicStat" / "inference" / "empirical_likelihood.py"
    el_capability = capability_for(
        "empirical_likelihood", InferenceDesign.ONE_SAMPLE, Estimand.MEAN
    )
    if el_capability is None:
        raise RuntimeError("empirical_likelihood capability is not registered")
    if el_capability.calibrated or el_capability.automatic_selection_allowed:
        raise RuntimeError("calibration harness refuses an automatically enabled EL capability")
    return {
        "student_t": {
            "version": T_METHOD_VERSION,
            "scipy_version": scipy.__version__,
            "alternative": "two-sided",
        },
        "empirical_likelihood": {
            "version": EL_METHOD_VERSION,
            "source_sha256": sha256_file(el_path),
            "capability": el_capability.to_dict(),
            "bartlett_correction": False,
        },
    }


def _validate_config(config: RunConfig) -> None:
    if config.replicates_per_cell < 1:
        raise ValueError("replicates_per_cell must be positive")
    if config.master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    if config.workers < 1 or config.batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    if config.num_shards < 1 or not 0 <= config.shard_id < config.num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    if not 0.0 < config.alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if not 0.0 < config.confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")


def build_run_manifest(
    config: RunConfig,
    cells: tuple[ExperimentCell, ...],
    storage_format: str,
    parquet_backend: str | None,
) -> dict[str, object]:
    repository_sha, repository_dirty = _git_head()
    scientific_config = {
        "schema_version": RUN_SCHEMA_VERSION,
        "replicate_schema_version": REPLICATE_SCHEMA_VERSION,
        "repository_sha": repository_sha,
        "alpha": config.alpha,
        "confidence_level": config.confidence_level,
        "master_seed": config.master_seed,
        "seed_derivation_scheme": SEED_DERIVATION_SCHEME,
        "replicates_per_cell": config.replicates_per_cell,
        "replicate_id_semantics": (
            "global IDs 0..replicates_per_cell-1; shard owns IDs where "
            "replicate_id % num_shards == shard_id"
        ),
        "num_shards": config.num_shards,
        "scenario_registry_digest": registry_digest(cells),
        "cells": [cell.to_metadata() for cell in cells],
        "method_versions": _method_metadata(),
        "holdout_used": False,
        "holdout_exclusion_policy": active_holdout_policy(),
        "storage_format": storage_format,
    }
    return {
        **scientific_config,
        "run_id": _stable_digest(scientific_config),
        "created_at_utc": _utc_now(),
        "repository_dirty_at_start": repository_dirty,
        "parquet_engine_at_manifest_creation": parquet_backend,
        "network_required": False,
        "llm_or_external_api_required": False,
    }


def _ensure_root_manifest(path: Path, expected: Mapping[str, object]) -> dict[str, object]:
    if path.exists():
        existing = read_json(path)
        if existing.get("run_id") != expected.get("run_id"):
            raise ValueError("existing run_manifest.json is incompatible with this run")
        return existing
    write_json_atomic(path, expected)
    actual = read_json(path)
    if actual.get("run_id") != expected.get("run_id"):
        raise ValueError("concurrent run created an incompatible root manifest")
    return actual


def _shard_config_payload(
    config: RunConfig,
    backend: SampleBackend,
    run_manifest: Mapping[str, object],
    parquet_backend: str | None,
) -> dict[str, object]:
    immutable = {
        "run_id": run_manifest["run_id"],
        "shard_id": config.shard_id,
        "batch_size": config.batch_size,
        "backend": backend.info.to_dict(),
        "storage_format": run_manifest["storage_format"],
        "repository_sha": run_manifest["repository_sha"],
        "alpha": run_manifest["alpha"],
        "confidence_level": run_manifest["confidence_level"],
        "scenario_registry_digest": run_manifest["scenario_registry_digest"],
        "method_versions_digest": _stable_digest(run_manifest["method_versions"]),
        "num_shards": run_manifest["num_shards"],
        "pandas_version": pd.__version__,
        "parquet_engine_used": parquet_backend,
    }
    return {
        **immutable,
        "shard_config_digest": _stable_digest(immutable),
        "workers": config.workers,
        "status": "running",
        "started_at_utc": _utc_now(),
        "holdout_used": False,
        "holdout_exclusion_policy_version": active_holdout_policy()["policy_version"],
    }


def _ensure_shard_manifest(
    path: Path, expected: Mapping[str, object]
) -> dict[str, object]:
    if path.exists():
        existing = read_json(path)
        if existing.get("shard_config_digest") != expected.get("shard_config_digest"):
            raise ValueError(f"existing shard metadata is incompatible: {path}")
        existing["status"] = "running"
        existing["resumed_at_utc"] = _utc_now()
        write_json_atomic(path, existing)
        return existing
    write_json_atomic(path, expected)
    return dict(expected)


def _block_is_complete(
    marker_path: Path,
    data_path: Path,
    expected_ids: tuple[int, ...],
) -> bool:
    if not marker_path.exists():
        return False
    marker = read_json(marker_path)
    expected = {
        "rows": len(expected_ids),
        "first_replicate_id": expected_ids[0],
        "last_replicate_id": expected_ids[-1],
        "data_file": data_path.name,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"incompatible completion marker: {marker_path}")
    if not data_path.is_file():
        raise ValueError(f"completion marker exists but data file is missing: {data_path}")
    if marker.get("sha256") != sha256_file(data_path):
        raise ValueError(f"completed block checksum mismatch: {data_path}")
    return True


def run_shard(config: RunConfig) -> RunOutcome:
    """Run or resume one shard, checkpointing each bounded replicate block."""

    _validate_config(config)
    cells = select_cells(config.scenario_ids, config.sample_sizes)
    storage_format, parquet_backend = resolve_storage_format(config.storage_format)
    backend = resolve_backend(config.backend)
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_manifest = build_run_manifest(config, cells, storage_format, parquet_backend)
    run_manifest = _ensure_root_manifest(output / "run_manifest.json", expected_manifest)

    shard_dir = output / "shards" / f"shard_{config.shard_id:03d}"
    shard_manifest_path = shard_dir / "shard_manifest.json"
    shard_manifest = _ensure_shard_manifest(
        shard_manifest_path,
        _shard_config_payload(config, backend, run_manifest, parquet_backend),
    )
    extension = FORMAT_EXTENSIONS[storage_format]
    owned_ids = owned_replicate_ids(
        config.replicates_per_cell, config.shard_id, config.num_shards
    )

    computed_blocks = 0
    skipped_blocks = 0
    rows_computed = 0
    cells_completed = 0
    aggregate_timings = {
        "sample_generation_seconds": 0.0,
        "diagnostics_seconds": 0.0,
        "device_to_host_seconds": 0.0,
        "student_t_seconds": 0.0,
        "el_test_seconds": 0.0,
        "el_ci_seconds": 0.0,
        "serialization_seconds": 0.0,
    }

    with MethodExecutor(config.workers) as executor:
        for cell_index, cell in enumerate(cells, start=1):
            cell_dir = shard_dir / cell.scenario.name / f"n_{cell.n:05d}"
            expected_blocks = (len(owned_ids) + config.batch_size - 1) // config.batch_size
            for block_index, replicate_ids in enumerate(
                replicate_blocks(owned_ids, config.batch_size)
            ):
                data_path = cell_dir / f"block_{block_index:06d}.{extension}"
                marker_path = cell_dir / f"block_{block_index:06d}.complete.json"
                if not config.force and _block_is_complete(
                    marker_path, data_path, replicate_ids
                ):
                    skipped_blocks += 1
                    continue

                seeds = tuple(
                    derive_seed(
                        config.master_seed,
                        cell.scenario.name,
                        config.shard_id,
                        replicate_id,
                    )
                    for replicate_id in replicate_ids
                )
                stage: dict[str, float] = {}
                started = time.perf_counter()
                native = backend.generate_native(cell.scenario, cell.n, seeds)
                stage["sample_generation_seconds"] = time.perf_counter() - started
                started = time.perf_counter()
                diagnostics = backend.diagnostics(native)
                stage["diagnostics_seconds"] = time.perf_counter() - started
                started = time.perf_counter()
                samples = backend.to_cpu(native)
                stage["device_to_host_seconds"] = time.perf_counter() - started
                records, method_timings = evaluate_batch(
                    samples,
                    diagnostics,
                    cell.scenario,
                    replicate_ids,
                    seeds,
                    shard_id=config.shard_id,
                    num_shards=config.num_shards,
                    alpha=config.alpha,
                    confidence_level=config.confidence_level,
                    generation_backend=native.engine,
                    executor=executor,
                )
                stage.update(method_timings)
                frame = pd.DataFrame.from_records(records, columns=REPLICATE_COLUMNS)
                started = time.perf_counter()
                write_frame_atomic(data_path, frame, storage_format, parquet_backend)
                stage["serialization_seconds"] = time.perf_counter() - started
                marker = {
                    "status": "complete",
                    "run_id": run_manifest["run_id"],
                    "scenario_id": cell.scenario.name,
                    "n": cell.n,
                    "shard_id": config.shard_id,
                    "block_index": block_index,
                    "rows": len(replicate_ids),
                    "first_replicate_id": replicate_ids[0],
                    "last_replicate_id": replicate_ids[-1],
                    "replicate_id_step": config.num_shards,
                    "data_file": data_path.name,
                    "sha256": sha256_file(data_path),
                    "generation_backend": native.engine,
                    "stage_timings_seconds": stage,
                    "estimated_sample_batch_bytes": len(replicate_ids) * cell.n * 8,
                    "holdout_used": False,
                    "holdout_exclusion_policy_version": active_holdout_policy()["policy_version"],
                    "completed_at_utc": _utc_now(),
                }
                write_json_atomic(marker_path, marker)
                computed_blocks += 1
                rows_computed += len(replicate_ids)
                for key in aggregate_timings:
                    aggregate_timings[key] += stage[key]
            write_json_atomic(
                cell_dir / "_SUCCESS.json",
                {
                    "status": "complete",
                    "run_id": run_manifest["run_id"],
                    "scenario_id": cell.scenario.name,
                    "n": cell.n,
                    "shard_id": config.shard_id,
                    "expected_rows": len(owned_ids),
                    "expected_blocks": expected_blocks,
                    "holdout_used": False,
                    "holdout_exclusion_policy_version": active_holdout_policy()["policy_version"],
                    "completed_at_utc": _utc_now(),
                },
            )
            cells_completed += 1
            print(
                f"[{cell_index}/{len(cells)}] shard={config.shard_id} "
                f"{cell.scenario.name} n={cell.n} complete",
                flush=True,
            )

    shard_manifest.update(
        {
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "computed_blocks_last_invocation": computed_blocks,
            "skipped_blocks_last_invocation": skipped_blocks,
            "rows_computed_last_invocation": rows_computed,
            "stage_timings_seconds_last_invocation": aggregate_timings,
            "holdout_used": False,
        }
    )
    write_json_atomic(shard_manifest_path, shard_manifest)
    return RunOutcome(
        computed_blocks=computed_blocks,
        skipped_blocks=skipped_blocks,
        rows_computed=rows_computed,
        cells_completed=cells_completed,
        output=str(output),
        shard_id=config.shard_id,
        run_id=str(run_manifest["run_id"]),
    )
