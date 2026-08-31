"""Deterministic 80-digit audit for CP-06 high-precision triggers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import warnings

import mpmath as mp
import numpy as np
import pandas as pd

from experiments.proportion_ci_calibration.harness import (
    CANDIDATE_SHA,
    CP04_DOCUMENT_SHA,
    EXPERIMENT_VERSION,
    jeffreys_interval_grid,
    production_interval_grid,
)
from pyMagicStat.inference import PopulationProportionCI


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPOSITORY_ROOT / "experiments" / "results"
QUEUE_COLUMNS = (
    "audit_kind",
    "reason",
    "checkpoint",
    "n",
    "alpha",
    "method",
    "p",
    "x",
    "first_x",
    "last_x",
    "coverage",
    "oracle",
)


def _mp(value: float | int) -> mp.mpf:
    if isinstance(value, (int, np.integer)):
        return mp.mpf(int(value))
    return mp.mpf(repr(float(value)))


def high_precision_binomial_range(
    n: int,
    p: float,
    first: int,
    last: int,
    *,
    digits: int = 80,
) -> mp.mpf:
    """Return P(first <= X <= last) without float64 tail subtraction."""

    if digits < 80:
        raise ValueError("CP-04 requires at least 80 decimal digits")
    first = max(0, int(first))
    last = min(int(n), int(last))
    if first > last:
        return mp.mpf("0")
    with mp.workdps(digits):
        probability = _mp(p)
        if probability <= 0:
            return mp.mpf(1 if first == 0 else 0)
        if probability >= 1:
            return mp.mpf(1 if last == n else 0)
        if first == 0 and last == n:
            return mp.mpf(1)
        if first == 0:
            return mp.betainc(
                n - last,
                last + 1,
                0,
                1 - probability,
                regularized=True,
            )
        if last == n:
            return mp.betainc(
                first,
                n - first + 1,
                0,
                probability,
                regularized=True,
            )
        cdf_high = mp.betainc(
            n - last,
            last + 1,
            0,
            1 - probability,
            regularized=True,
        )
        cdf_low = mp.betainc(
            n - first + 1,
            first,
            0,
            1 - probability,
            regularized=True,
        )
        return cdf_high - cdf_low


def _beta_quantile(probability: mp.mpf, a: mp.mpf, b: mp.mpf) -> mp.mpf:
    if probability <= 0:
        return mp.mpf(0)
    if probability >= 1:
        return mp.mpf(1)
    lower = mp.mpf(0)
    upper = mp.mpf(1)
    iterations = int(math.ceil(mp.mp.dps * math.log2(10))) + 8
    for _ in range(iterations):
        midpoint = (lower + upper) / 2
        value = mp.betainc(a, b, 0, midpoint, regularized=True)
        if value < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def high_precision_interval(
    method: str,
    n: int,
    x: int,
    alpha: float,
    *,
    digits: int = 80,
) -> tuple[mp.mpf, mp.mpf]:
    """Independent high-precision endpoint oracle for all CP-06 methods."""

    if digits < 80:
        raise ValueError("CP-04 requires at least 80 decimal digits")
    with mp.workdps(digits):
        a = _mp(alpha)
        nn = mp.mpf(n)
        xx = mp.mpf(x)
        p_hat = xx / nn
        if method == "wilson":
            z_value = mp.sqrt(2) * mp.erfinv(1 - a)
            z_squared = z_value**2
            denominator = 1 + z_squared / nn
            center = (p_hat + z_squared / (2 * nn)) / denominator
            half_width = (
                z_value
                * mp.sqrt(p_hat * (1 - p_hat) / nn + z_squared / (4 * nn**2))
                / denominator
            )
            return center - half_width, center + half_width
        if method == "wald":
            z_value = mp.sqrt(2) * mp.erfinv(1 - a)
            half_width = z_value * mp.sqrt(p_hat * (1 - p_hat) / nn)
            return p_hat - half_width, p_hat + half_width
        if method == "clopper_pearson":
            lower = (
                mp.mpf(0)
                if x == 0
                else _beta_quantile(a / 2, xx, nn - xx + 1)
            )
            upper = (
                mp.mpf(1)
                if x == n
                else _beta_quantile(1 - a / 2, xx + 1, nn - xx)
            )
            return lower, upper
        if method == "jeffreys":
            return (
                _beta_quantile(a / 2, xx + mp.mpf("0.5"), nn - xx + mp.mpf("0.5")),
                _beta_quantile(
                    1 - a / 2,
                    xx + mp.mpf("0.5"),
                    nn - xx + mp.mpf("0.5"),
                ),
            )
        raise ValueError(f"unknown method: {method}")


def _float64_interval(method: str, n: int, x: int, alpha: float) -> tuple[float, float]:
    if method == "jeffreys":
        grid = jeffreys_interval_grid(n, alpha)
        return float(grid.lower[x]), float(grid.upper[x])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = PopulationProportionCI.from_counts(
            x,
            n,
            alpha=alpha,
            method=method,
            independence="assumed",
        ).calculate_interval()
    return float(result["lb"]), float(result["ub"])


def _audit_one(arguments: tuple[dict[str, object], int]) -> dict[str, object]:
    row, digits = arguments
    common = {
        "audit_kind": row["audit_kind"],
        "reason": row["reason"],
        "checkpoint": row["checkpoint"],
        "n": int(row["n"]),
        "alpha": float(row["alpha"]),
        "method": row["method"],
        "digits": digits,
    }
    if row["audit_kind"] == "coverage":
        value = high_precision_binomial_range(
            int(row["n"]),
            float(row["p"]),
            int(row["first_x"]),
            int(row["last_x"]),
            digits=digits,
        )
        float64_value = float(row["coverage"])
        with mp.workdps(digits):
            decimal_value = mp.nstr(value, digits)
            absolute_error = float(abs(value - _mp(float64_value)))
            undercoverage = float(
                max(mp.mpf(0), 1 - _mp(row["alpha"]) - value)
            )
        return {
            **common,
            "p": float(row["p"]),
            "first_x": int(row["first_x"]),
            "last_x": int(row["last_x"]),
            "float64_coverage": float64_value,
            "high_precision_coverage": decimal_value,
            "high_precision_coverage_float": float(value),
            "absolute_error": absolute_error,
            "high_precision_undercoverage": undercoverage,
            "status": "resolved",
        }
    if row["audit_kind"] == "endpoint":
        x = int(row["x"])
        lower, upper = high_precision_interval(
            str(row["method"]),
            int(row["n"]),
            x,
            float(row["alpha"]),
            digits=digits,
        )
        float_lower, float_upper = _float64_interval(
            str(row["method"]),
            int(row["n"]),
            x,
            float(row["alpha"]),
        )
        with mp.workdps(digits):
            decimal_lower = mp.nstr(lower, digits)
            decimal_upper = mp.nstr(upper, digits)
            lower_error = float(abs(lower - _mp(float_lower)))
            upper_error = float(abs(upper - _mp(float_upper)))
        return {
            **common,
            "x": x,
            "float64_lower": float_lower,
            "float64_upper": float_upper,
            "high_precision_lower": decimal_lower,
            "high_precision_upper": decimal_upper,
            "lower_error": lower_error,
            "upper_error": upper_error,
            "status": "resolved",
        }
    raise ValueError(f"unknown audit kind: {row['audit_kind']}")


def _frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required checkpoint artifact not found: {path}")
    return pd.read_parquet(path)


def _grid_for(
    cache: dict[tuple[int, float, str], object],
    n: int,
    alpha: float,
    method: str,
):
    identity = (n, alpha, method)
    if identity not in cache:
        cache[identity] = (
            jeffreys_interval_grid(n, alpha)
            if method == "jeffreys"
            else production_interval_grid(n, alpha, method)
        )
    return cache[identity]


def _endpoint_queue_row(
    *,
    checkpoint: str,
    reason: str,
    n: int,
    alpha: float,
    method: str,
    x: int,
) -> dict[str, object]:
    return {
        "audit_kind": "endpoint",
        "reason": reason,
        "checkpoint": checkpoint,
        "n": n,
        "alpha": alpha,
        "method": method,
        "p": np.nan,
        "x": x,
        "first_x": np.nan,
        "last_x": np.nan,
        "coverage": np.nan,
        "oracle": "high_precision_structural_oracle",
    }


def _structural_queue_rows(
    checkpoint: str,
    prefix: str,
    results_dir: Path,
) -> list[dict[str, object]]:
    """Localize every preregistered structural trigger to exact x values."""

    rows: list[dict[str, object]] = []
    cache: dict[tuple[int, float, str], object] = {}
    invariants = _frame(results_dir / f"{prefix}_invariants.parquet")
    for finding in invariants.to_dict("records"):
        n = int(finding["n"])
        alpha = float(finding["alpha"])
        method = str(finding["method"])
        tolerance = 1e-12 if n <= 5_000 else 1e-10
        requires_localization = (
            int(finding["nan_count"]) > 0
            or float(finding["max_complement_error"]) > tolerance
            or (
                method in {"wilson", "clopper_pearson"}
                and (
                    int(finding["bounds_failures"]) > 0
                    or int(finding["lower_monotonic_failures"]) > 0
                    or int(finding["upper_monotonic_failures"]) > 0
                )
            )
        )
        if not requires_localization:
            continue
        grid = _grid_for(cache, n, alpha, method)
        checks = {
            "nan_or_inf": np.flatnonzero(
                ~np.isfinite(grid.lower) | ~np.isfinite(grid.upper)
            ),
            "complement_symmetry": np.flatnonzero(
                (np.abs(grid.lower - (1.0 - grid.upper[::-1])) > tolerance)
                | (np.abs(grid.upper - (1.0 - grid.lower[::-1])) > tolerance)
            ),
        }
        if method in {"wilson", "clopper_pearson"}:
            checks["bounds"] = np.flatnonzero(
                (grid.lower < -tolerance) | (grid.upper > 1.0 + tolerance)
            )
            lower_pairs = np.flatnonzero(np.diff(grid.lower) < -tolerance)
            upper_pairs = np.flatnonzero(np.diff(grid.upper) < -tolerance)
            checks["endpoint_monotonicity"] = np.unique(
                np.concatenate(
                    (lower_pairs, lower_pairs + 1, upper_pairs, upper_pairs + 1)
                )
            )
        for reason, indices in checks.items():
            for x in indices:
                rows.append(
                    _endpoint_queue_row(
                        checkpoint=checkpoint,
                        reason=reason,
                        n=n,
                        alpha=alpha,
                        method=method,
                        x=int(x),
                    )
                )

    nesting = _frame(results_dir / f"{prefix}_nesting.parquet")
    for finding in nesting.to_dict("records"):
        if not (
            int(finding["lower_nesting_failures"]) > 0
            or int(finding["upper_nesting_failures"]) > 0
        ):
            continue
        n = int(finding["n"])
        method = str(finding["method"])
        alpha_wider = float(finding["alpha_wider"])
        alpha_narrower = float(finding["alpha_narrower"])
        tolerance = 1e-12 if n <= 5_000 else 1e-10
        wide = _grid_for(cache, n, alpha_wider, method)
        narrow = _grid_for(cache, n, alpha_narrower, method)
        indices = np.flatnonzero(
            (wide.lower > narrow.lower + tolerance)
            | (wide.upper < narrow.upper - tolerance)
        )
        for x in indices:
            rows.extend(
                (
                    _endpoint_queue_row(
                        checkpoint=checkpoint,
                        reason="nesting_wider",
                        n=n,
                        alpha=alpha_wider,
                        method=method,
                        x=int(x),
                    ),
                    _endpoint_queue_row(
                        checkpoint=checkpoint,
                        reason="nesting_narrower",
                        n=n,
                        alpha=alpha_narrower,
                        method=method,
                        x=int(x),
                    ),
                )
            )
    return rows


def build_high_precision_queue(
    checkpoints: tuple[str, ...],
    *,
    results_dir: Path = RESULTS,
) -> pd.DataFrame:
    """Collect coverage and oracle triggers without changing their thresholds."""

    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        prefix = f"proportion_ci_cp06_{checkpoint.lower()}"
        triggers = _frame(results_dir / f"{prefix}_high_precision_triggers.parquet")
        for row in triggers.to_dict("records"):
            rows.append(
                {
                    **row,
                    "audit_kind": "coverage",
                    "reason": row.get("trigger", "coverage_trigger"),
                    "checkpoint": checkpoint,
                    "x": np.nan,
                    "oracle": "",
                }
            )
            for boundary_name, boundary_x in (
                ("acceptance_first_endpoint", int(row["first_x"])),
                ("acceptance_last_endpoint", int(row["last_x"])),
            ):
                if 0 <= boundary_x <= int(row["n"]):
                    rows.append(
                        _endpoint_queue_row(
                            checkpoint=checkpoint,
                            reason=boundary_name,
                            n=int(row["n"]),
                            alpha=float(row["alpha"]),
                            method=str(row["method"]),
                            x=boundary_x,
                        )
                    )

        oracle_path = results_dir / f"{prefix}_oracles.parquet"
        oracles = _frame(oracle_path)
        if not oracles.empty:
            maximum_error = oracles[["lower_error", "upper_error"]].max(axis=1)
            tolerance = np.where(oracles["n"] <= 5_000, 1e-12, 1e-10)
            discrepant = oracles[
                oracles["gate_applicable"].astype(bool) & (maximum_error > tolerance)
            ]
            for row in discrepant.to_dict("records"):
                rows.append(
                    {
                        **row,
                        "audit_kind": "endpoint",
                        "reason": "oracle_discrepancy",
                        "checkpoint": checkpoint,
                        "p": np.nan,
                        "first_x": np.nan,
                        "last_x": np.nan,
                        "coverage": np.nan,
                    }
                )

        rows.extend(_structural_queue_rows(checkpoint, prefix, results_dir))

    queue = pd.DataFrame(rows).reindex(columns=QUEUE_COLUMNS)
    if queue.empty:
        return queue
    identity = [
        "audit_kind",
        "reason",
        "n",
        "alpha",
        "method",
        "p",
        "x",
        "first_x",
        "last_x",
    ]
    return queue.drop_duplicates(subset=identity).sort_values(
        identity,
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def run_audit(
    checkpoints: tuple[str, ...],
    *,
    workers: int,
    digits: int,
    results_dir: Path = RESULTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if digits < 80:
        raise ValueError("CP-04 requires at least 80 decimal digits")
    queue = build_high_precision_queue(checkpoints, results_dir=results_dir)
    arguments = [(row, digits) for row in queue.to_dict("records")]
    if workers == 1:
        audit_rows = list(map(_audit_one, arguments))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            audit_rows = list(executor.map(_audit_one, arguments, chunksize=1))
    return queue, pd.DataFrame(audit_rows)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=("C", "D", "E"))
    parser.add_argument("--workers", type=_positive_integer, default=1)
    parser.add_argument("--digits", type=_positive_integer, default=80)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    queue, audit = run_audit(
        tuple(args.checkpoints),
        workers=args.workers,
        digits=args.digits,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    queue_path = RESULTS / "proportion_ci_cp06_f_high_precision_queue.parquet"
    audit_path = RESULTS / "proportion_ci_cp06_f_high_precision_audit.parquet"
    queue.to_parquet(queue_path, index=False)
    audit.to_parquet(audit_path, index=False)
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (queue_path, audit_path)
    }
    metadata = {
        "checkpoint": "F",
        "candidate_sha": CANDIDATE_SHA,
        "cp04_document_sha": CP04_DOCUMENT_SHA,
        "experiment_version": EXPERIMENT_VERSION,
        "source_checkpoints": args.checkpoints,
        "digits": args.digits,
        "workers": args.workers,
        "queue_rows": len(queue),
        "audit_rows": len(audit),
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "mpmath": mp.__version__,
        "hashes": hashes,
    }
    metadata_path = RESULTS / "proportion_ci_cp06_f_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"CP06-F audited {len(audit)} cells at {args.digits} digits", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
