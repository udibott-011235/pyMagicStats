"""CP05-C2A CUDA/RAPIDS experimental engine (DEC-014, DEC-015).

This deliberately small module keeps the mathematical operations visible for
line-by-line audit.  CuPy is optional at import time; a NumPy backend supports
only deterministic development fixtures.  A real calibration run requires a
CUDA runtime and the separately preregistered CP05-C2B equivalence gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # Quantum target: RAPIDS/CuPy.  Import must remain safe on non-GPU hosts.
    import cupy as cp
except ImportError:  # pragma: no cover - exercised by the CPU-only import contract
    cp = None


ENGINE_ID = "CUDA_RAPIDS_EXPERIMENTAL"
ARTIFACT_SCHEMA_VERSION = "cp05-c2a-cuda-artifacts-v1"
ALPHA = 0.05
SUPPORTED_FAMILIES = frozenset({"gamma", "exponential", "negative_binomial"})
SUPPORTED_STATISTICS = frozenset({"AD", "CVM"})


class EngineContractError(ValueError):
    """Fail closed when a request would violate the frozen experimental scope."""


def gpu_runtime_available() -> bool:
    """True only when CuPy can enumerate an actual CUDA device."""

    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _xp(require_gpu: bool):
    if require_gpu:
        if not gpu_runtime_available():
            raise EngineContractError("CUDA runtime is unavailable; no GPU run was started")
        return cp
    return cp if gpu_runtime_available() else np


def derive_seed(namespace: str, canonical_cell_id: str, raw_outer_index: int,
                purpose: str, raw_inner_index: int | None = None) -> int:
    """Frozen SHA-256/NUL seed identity; backend generation is separate."""

    if not all(isinstance(value, str) and value and "\0" not in value
               for value in (namespace, canonical_cell_id, purpose)):
        raise EngineContractError("namespace, cell id and purpose must be nonempty NUL-free strings")
    if type(raw_outer_index) is not int or raw_outer_index < 0:
        raise EngineContractError("raw_outer_index must be a non-negative Python int")
    if raw_inner_index is not None and (type(raw_inner_index) is not int or raw_inner_index < 0):
        raise EngineContractError("raw_inner_index must be None or a non-negative Python int")
    fields = (namespace, canonical_cell_id, str(raw_outer_index), purpose,
              "" if raw_inner_index is None else str(raw_inner_index))
    return int.from_bytes(hashlib.sha256("\0".join(fields).encode("utf-8")).digest()[:16], "big")


def continuous_cvm(pit_sorted, xp=np) -> Any:
    """DEC-014: W²=1/(12n)+Σ[u_i-(2i-1)/(2n)]²; no clipping."""

    u = xp.asarray(pit_sorted, dtype=xp.float64)
    if u.ndim < 1 or u.shape[-1] == 0:
        raise EngineContractError("PIT input requires a nonempty final dimension")
    n = u.shape[-1]
    ranks = (2 * xp.arange(1, n + 1) - 1) / (2 * n)
    return 1 / (12 * n) + xp.sum((u - ranks) ** 2, axis=-1)


def continuous_ad(logcdf_sorted, logsf_reverse, xp=np) -> Any:
    """DEC-014 AD using stable logcdf/logsf; deliberately no silent clipping."""

    left = xp.asarray(logcdf_sorted, dtype=xp.float64)
    right = xp.asarray(logsf_reverse, dtype=xp.float64)
    if left.shape != right.shape or left.ndim < 1 or left.shape[-1] == 0:
        raise EngineContractError("AD log probability arrays require equal nonempty final dimensions")
    n = left.shape[-1]
    if bool(xp.any(~xp.isfinite(left))) or bool(xp.any(~xp.isfinite(right))):
        raise EngineContractError("AD requires finite stable logcdf/logsf values")
    weights = 2 * xp.arange(1, n + 1) - 1
    return -n - xp.sum(weights * (left + right), axis=-1) / n


def nb_discrete_cvm(pmf, cdf, sample, xp=np) -> Any:
    """DEC-014: W_d²=(1/n)Σ(S_j-nH_j)² p_j over certified supplied support."""

    p = xp.asarray(pmf, dtype=xp.float64)
    h = xp.asarray(cdf, dtype=xp.float64)
    values = xp.asarray(sample, dtype=xp.int64)
    if p.ndim != 1 or h.shape != p.shape or values.ndim != 1 or values.size == 0:
        raise EngineContractError("NB statistic requires one-dimensional support and sample")
    support = xp.arange(p.size)
    s = xp.searchsorted(xp.sort(values), support, side="right")
    z = s - values.size * h
    return xp.sum(z ** 2 * p) / values.size


def nb_discrete_ad(pmf, cdf, sample, xp=np) -> Any:
    """DEC-014: A_d²=(1/n)Σ Z_j²p_j/[H_j(1-H_j)], no approximate substitute."""

    p = xp.asarray(pmf, dtype=xp.float64)
    h = xp.asarray(cdf, dtype=xp.float64)
    if bool(xp.any(h <= 0)) or bool(xp.any(h >= 1)):
        raise EngineContractError("NB AD support must exclude uncertified 0/1 CDF endpoints")
    values = xp.asarray(sample, dtype=xp.int64)
    support = xp.arange(p.size)
    s = xp.searchsorted(xp.sort(values), support, side="right")
    z = s - values.size * h
    return xp.sum(z ** 2 * p / (h * (1 - h))) / values.size


def nb_eligibility(sample) -> tuple[bool, str | None]:
    """DEC-014 NB MLE eligibility; mathematical ineligibility is not failure."""

    values = np.asarray(sample, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise EngineContractError("NB sample must be a finite non-negative vector")
    if np.all(values == 0):
        return False, "ALL_ZERO_NON_IDENTIFYING"
    if np.var(values, ddof=1) <= np.mean(values):
        return False, "VARIANCE_NOT_GREATER_THAN_MEAN"
    return True, None


def mc_pvalue(observed: float, bootstrap_statistics: Iterable[float]) -> tuple[int, float]:
    """DEC-014 plus-one p=(b+1)/(B+1), with ties counted by >=."""

    values = np.asarray(tuple(bootstrap_statistics), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(observed) or not np.all(np.isfinite(values)):
        raise EngineContractError("Monte Carlo p-value requires finite observed and nonempty finite replicates")
    b = int(np.sum(values >= observed))
    return b, (b + 1) / (values.size + 1)


@dataclass(frozen=True)
class EngineRequest:
    family: str
    parameters: dict[str, float]
    n: int
    statistic: str
    null_type: str
    B: int
    R: int
    outer_batch_size: int
    bootstrap_batch_size: int
    seed_namespace: str

    def __post_init__(self):
        if self.family not in SUPPORTED_FAMILIES or self.statistic not in SUPPORTED_STATISTICS:
            raise EngineContractError("family/statistic is outside CP05-C2A scope")
        if self.null_type != "composite":
            raise EngineContractError("CP05-C2A supports composite null only")
        if min(self.n, self.B, self.R, self.outer_batch_size, self.bootstrap_batch_size) < 1:
            raise EngineContractError("sizes must be positive")


def artifact_metadata(request: EngineRequest) -> dict[str, Any]:
    return {"schema_version": ARTIFACT_SCHEMA_VERSION, "engine": ENGINE_ID,
            "production_engine": False, "equivalence_gate_passed": False,
            "calibration_claim": False, "request": asdict(request)}


def write_artifact_bundle(output: Path, request: EngineRequest, rows: list[dict[str, Any]]) -> None:
    """Write the six-artifact experimental bundle; parquet needs pyarrow or cuDF."""

    output.mkdir(parents=True, exist_ok=False)
    meta = artifact_metadata(request)
    payloads = {"manifest.json": meta, "accounting.json": {"rows": len(rows)},
                "environment.json": {"gpu_runtime_available": gpu_runtime_available(), "engine": ENGINE_ID},
                "summary.json": {"status": "EXPERIMENTAL_NON_CALIBRATION", "rows": len(rows)}}
    for name, payload in payloads.items():
        (output / name).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(output / "results.parquet", index=False)
    except Exception as exc:
        raise EngineContractError("results.parquet requires an installed parquet backend") from exc
    digests = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    (output / "digests.json").write_text(json.dumps(digests, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _parse_parameters(value: str) -> dict[str, float]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("--parameters must be JSON") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--parameters must be a JSON object")
    return {str(key): float(item) for key, item in parsed.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CP05-C2A experimental CUDA/RAPIDS prototype")
    parser.add_argument("--family", choices=sorted(SUPPORTED_FAMILIES), default="exponential")
    parser.add_argument("--parameters", type=_parse_parameters, default={"scale": 1.0})
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--statistic", choices=sorted(SUPPORTED_STATISTICS), default="CVM")
    parser.add_argument("--null-type", default="composite")
    parser.add_argument("--B", type=int, default=9)
    parser.add_argument("--R", type=int, default=1)
    parser.add_argument("--outer-batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-batch-size", type=int, default=3)
    parser.add_argument("--seed-namespace", default="CP05-C2A-development")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args(argv)
    request = EngineRequest(args.family, args.parameters, args.n, args.statistic, args.null_type,
                            args.B, args.R, args.outer_batch_size, args.bootstrap_batch_size,
                            args.seed_namespace)
    if not args.allow_cpu_smoke and not gpu_runtime_available():
        raise SystemExit("CUDA runtime unavailable; use --allow-cpu-smoke only for tiny development fixtures")
    # Execution is intentionally not implemented in C2A: CP05-C2B must preregister
    # reference↔CUDA equivalence before any calibration or substantial GPU campaign.
    raise SystemExit("C2A is a prototype only; no calibration execution is authorized")


if __name__ == "__main__":
    main()
