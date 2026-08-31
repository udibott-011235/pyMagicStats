"""Deterministic CP-06 calibration primitives.

Production endpoints are obtained only through
``PopulationProportionCI.from_counts``.  Independent formulas in this module
are restricted to reference oracles and the non-production Jeffreys comparator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, gammaln
from statsmodels.stats.proportion import proportion_confint

from pyMagicStat.inference import PopulationProportionCI


CANDIDATE_SHA = "2df5b90a5395163e723f9c52aafbb91fdce96d43"
CP04_DOCUMENT_SHA = "63eaaed6842e2f82473bfa857524645123f95218"
EXPERIMENT_VERSION = "proportion-ci-cp06-v1"
ALPHAS = (0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.200)
PRODUCTION_METHODS = ("wilson", "clopper_pearson", "wald")
METHODS = PRODUCTION_METHODS + ("jeffreys",)
STRESS_N = (
    7_500,
    10_000,
    15_000,
    20_000,
    30_000,
    50_000,
    75_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
)
FIXED_ANCHORS = np.asarray(
    (
        0.0,
        1e-12,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        2.5e-4,
        5e-4,
        1e-3,
        2.5e-3,
        5e-3,
        0.01,
        0.025,
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ),
    dtype=np.float64,
)
EVENT_LAMBDA_ANCHORS = np.asarray(
    (
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        5.0,
        7.5,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        50.0,
        75.0,
        100.0,
    ),
    dtype=np.float64,
)
EVENT_LAMBDAS = np.unique(
    np.concatenate((np.logspace(-6.0, 2.0, 801), EVENT_LAMBDA_ANCHORS))
)
EVENT_REGIME_LABELS = (
    "<0.5",
    "0.5-1",
    "1-2",
    "2-5",
    "5-10",
    "10-20",
    "20-30",
    ">=30",
)
ORIGIN_LABELS = {
    0: "grid",
    1: "boundary",
    2: "nextafter",
    3: "midpoint",
    4: "stationary",
}


@dataclass(frozen=True)
class IntervalGrid:
    lower: np.ndarray
    upper: np.ndarray
    method: str
    interval_kind: str
    source: str

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower


def production_interval_grid(n: int, alpha: float, method: str) -> IntervalGrid:
    """Evaluate every discrete outcome through the frozen production API."""

    if method not in PRODUCTION_METHODS:
        raise ValueError(f"not a production method: {method}")
    lower = np.empty(n + 1, dtype=np.float64)
    upper = np.empty(n + 1, dtype=np.float64)
    interval_kind = ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for x in range(n + 1):
            result = PopulationProportionCI.from_counts(
                x,
                n,
                alpha=alpha,
                method=method,
                independence="assumed",
            ).calculate_interval()
            if result["calibration_status"] != "not_calibrated":
                raise RuntimeError("CP-06 must not promote production calibration metadata")
            lower[x] = result["lb"]
            upper[x] = result["ub"]
            interval_kind = result["interval_kind"]
    return IntervalGrid(lower, upper, method, interval_kind, "production_api")


def jeffreys_interval_grid(n: int, alpha: float) -> IntervalGrid:
    """Experimental Beta(1/2,1/2) equal-tail credible interval comparator."""

    x = np.arange(n + 1, dtype=np.float64)
    lower = stats.beta.ppf(alpha / 2.0, x + 0.5, n - x + 0.5)
    upper = stats.beta.ppf(1.0 - alpha / 2.0, x + 0.5, n - x + 0.5)
    return IntervalGrid(
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        "jeffreys",
        "bayesian_comparator",
        "experimental_scipy_beta",
    )


def all_interval_grids(n: int) -> dict[tuple[float, str], IntervalGrid]:
    grids: dict[tuple[float, str], IntervalGrid] = {}
    for alpha in ALPHAS:
        for method in PRODUCTION_METHODS:
            grids[(alpha, method)] = production_interval_grid(n, alpha, method)
        grids[(alpha, "jeffreys")] = jeffreys_interval_grid(n, alpha)
    return grids


def base_probability_grid(
    n: int,
    *,
    linear_step: float | None,
) -> np.ndarray:
    """Construct anchors, optional linear grid, and event-scale points."""

    pieces = [FIXED_ANCHORS, 1.0 - FIXED_ANCHORS]
    if linear_step is not None:
        count = int(round(1.0 / linear_step))
        pieces.append(np.arange(1, count, dtype=np.float64) * linear_step)
    event = EVENT_LAMBDAS / float(n)
    event = event[event <= 0.5]
    pieces.extend((event, 1.0 - event, np.asarray((0.0, 1.0))))
    return np.unique(np.clip(np.concatenate(pieces), 0.0, 1.0))


def _acceptance_range(
    lower: np.ndarray,
    upper: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.searchsorted(upper, p, side="left").astype(np.int64)
    last = (np.searchsorted(lower, p, side="right") - 1).astype(np.int64)
    return first, last


def coverage_from_intervals(
    n: int,
    p: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable binomial coverage for monotone contiguous acceptance sets."""

    probabilities = np.asarray(p, dtype=np.float64)
    first, last = _acceptance_range(lower, upper, probabilities)
    coverage = np.zeros(probabilities.shape, dtype=np.float64)
    valid = (first <= last) & (first <= n) & (last >= 0)
    first = np.clip(first, 0, n + 1)
    last = np.clip(last, -1, n)

    left_full = valid & (first == 0)
    right_full = valid & (last == n) & (first > 0)
    interior = valid & (first > 0) & (last < n)
    coverage[left_full] = stats.binom.cdf(
        last[left_full], n, probabilities[left_full]
    )
    coverage[right_full] = stats.binom.sf(
        first[right_full] - 1, n, probabilities[right_full]
    )
    if np.any(interior):
        cdf_high = stats.binom.cdf(
            last[interior], n, probabilities[interior]
        )
        cdf_low = stats.binom.cdf(
            first[interior] - 1, n, probabilities[interior]
        )
        coverage[interior] = cdf_high - cdf_low
    return coverage, first, last


def _stationary_probability(n: int, first: int, last: int) -> float | None:
    """Analytic stationary point for P(first <= Bin(n,p) <= last)."""

    if first <= 0 or last >= n or first > last:
        return None
    exponent = last - first + 1
    log_left = gammaln(n) - gammaln(first) - gammaln(n - first + 1)
    log_right = gammaln(n) - gammaln(last + 1) - gammaln(n - last)
    return float(expit((log_left - log_right) / exponent))


def induced_probability_grid(
    n: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Endpoint, nextafter, midpoint, and analytic stationary candidates."""

    endpoints = np.unique(
        np.clip(
            np.concatenate((lower[np.isfinite(lower)], upper[np.isfinite(upper)])),
            0.0,
            1.0,
        )
    )
    interior = endpoints[(endpoints > 0.0) & (endpoints < 1.0)]
    next_points = np.unique(
        np.concatenate(
            (
                np.nextafter(interior, 0.0),
                np.nextafter(interior, 1.0),
            )
        )
    )
    boundaries = np.unique(np.concatenate((np.asarray((0.0, 1.0)), endpoints)))
    midpoints = boundaries[:-1] + (boundaries[1:] - boundaries[:-1]) / 2.0

    stationary: list[float] = []
    first, last = _acceptance_range(lower, upper, midpoints)
    for left, right, a, b in zip(
        boundaries[:-1],
        boundaries[1:],
        first,
        last,
    ):
        candidate = _stationary_probability(n, int(a), int(b))
        if candidate is not None and left < candidate < right:
            stationary.append(candidate)

    pieces = [boundaries, next_points, midpoints]
    origins = [
        np.full(boundaries.size, 1, dtype=np.int8),
        np.full(next_points.size, 2, dtype=np.int8),
        np.full(midpoints.size, 3, dtype=np.int8),
    ]
    if stationary:
        stationary_array = np.unique(np.asarray(stationary, dtype=np.float64))
        pieces.append(stationary_array)
        origins.append(np.full(stationary_array.size, 4, dtype=np.int8))

    points = np.concatenate(pieces)
    origin = np.concatenate(origins)
    order = np.argsort(points, kind="stable")
    points = points[order]
    origin = origin[order]
    unique, index = np.unique(points, return_index=True)
    return unique, origin[index]


def probability_grid_with_origins(
    n: int,
    interval: IntervalGrid,
    *,
    linear_step: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    base = base_probability_grid(n, linear_step=linear_step)
    induced, induced_origin = induced_probability_grid(
        n, interval.lower, interval.upper
    )
    points = np.concatenate((induced, base))
    origins = np.concatenate(
        (induced_origin, np.zeros(base.size, dtype=np.int8))
    )
    order = np.argsort(points, kind="stable")
    points = points[order]
    origins = origins[order]
    unique, index = np.unique(points, return_index=True)
    return unique, origins[index]


def expected_width_matrix(
    n: int,
    p: np.ndarray,
    widths: np.ndarray,
    *,
    tail_mass_bound: float = 1e-14,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate binomial expected widths with an explicit omitted-tail bound."""

    probabilities = np.asarray(p, dtype=np.float64)
    width_matrix = np.asarray(widths, dtype=np.float64)
    if width_matrix.ndim == 1:
        width_matrix = width_matrix[:, None]
    if width_matrix.shape[0] != n + 1:
        raise ValueError("width rows must correspond to x=0..n")

    expected = np.empty((probabilities.size, width_matrix.shape[1]), dtype=np.float64)
    retained_mass = np.empty(probabilities.size, dtype=np.float64)
    for start in range(0, probabilities.size, batch_size):
        stop = min(start + batch_size, probabilities.size)
        current = probabilities[start:stop]
        at_zero = current == 0.0
        at_one = current == 1.0
        interior = ~(at_zero | at_one)
        block_expected = np.empty((current.size, width_matrix.shape[1]))
        block_mass = np.ones(current.size, dtype=np.float64)
        block_expected[at_zero] = width_matrix[0]
        block_expected[at_one] = width_matrix[n]
        if np.any(interior):
            active = current[interior]
            deviation = math.sqrt(
                0.5 * n * math.log(2.0 / tail_mass_bound)
            )
            means = n * active
            lower_q = np.maximum(0, np.floor(means - deviation)).astype(int)
            upper_q = np.minimum(n, np.ceil(means + deviation)).astype(int)
            spans = upper_q - lower_q + 1
            max_span = int(np.max(spans))
            offsets = np.arange(max_span, dtype=np.int64)
            outcomes = lower_q[:, None] + offsets[None, :]
            mask = outcomes <= upper_q[:, None]
            clipped = np.clip(outcomes, 0, n)
            pmf = stats.binom.pmf(clipped, n, active[:, None])
            pmf[~mask] = 0.0
            selected_widths = width_matrix[clipped]
            values = np.einsum("ij,ijk->ik", pmf, selected_widths, optimize=True)
            block_expected[interior] = values
            block_mass[interior] = np.sum(pmf, axis=1)
        expected[start:stop] = block_expected
        retained_mass[start:stop] = block_mass
    return expected, retained_mass


def undercoverage_tier(deficit: float) -> str:
    if deficit <= 0.005:
        return "nominal_like"
    if deficit <= 0.015:
        return "mild_shortfall"
    if deficit <= 0.030:
        return "material_shortfall"
    if deficit <= 0.050:
        return "severe_shortfall"
    return "critical_shortfall"


def event_regime(n: int, p: np.ndarray) -> np.ndarray:
    rate = n * np.minimum(p, 1.0 - p)
    bins = np.asarray((0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0))
    return np.searchsorted(bins, rate, side="right")


def invariant_row(n: int, alpha: float, interval: IntervalGrid) -> dict[str, object]:
    lower = interval.lower
    upper = interval.upper
    complement_lower_error = np.max(np.abs(lower - (1.0 - upper[::-1])))
    complement_upper_error = np.max(np.abs(upper - (1.0 - lower[::-1])))
    return {
        "n": n,
        "alpha": alpha,
        "method": interval.method,
        "source": interval.source,
        "interval_kind": interval.interval_kind,
        "lower_monotonic_failures": int(np.sum(np.diff(lower) < -5e-15)),
        "upper_monotonic_failures": int(np.sum(np.diff(upper) < -5e-15)),
        "bounds_failures": int(
            np.sum((lower < -5e-15) | (upper > 1.0 + 5e-15))
        ),
        "max_complement_error": float(
            max(complement_lower_error, complement_upper_error)
        ),
        "nan_count": int(np.sum(~np.isfinite(lower)) + np.sum(~np.isfinite(upper))),
    }


def nesting_rows(
    n: int,
    grids: dict[tuple[float, str], IntervalGrid],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for tighter, narrower in zip(ALPHAS[:-1], ALPHAS[1:]):
            wide = grids[(tighter, method)]
            narrow = grids[(narrower, method)]
            rows.append(
                {
                    "n": n,
                    "method": method,
                    "alpha_wider": tighter,
                    "alpha_narrower": narrower,
                    "lower_nesting_failures": int(
                        np.sum(wide.lower > narrow.lower + 5e-15)
                    ),
                    "upper_nesting_failures": int(
                        np.sum(wide.upper < narrow.upper - 5e-15)
                    ),
                }
            )
    return rows


def oracle_errors(
    n: int,
    alpha: float,
    grids: dict[tuple[float, str], IntervalGrid],
) -> list[dict[str, object]]:
    points = sorted({0, 1 if n >= 1 else 0, n // 4, n // 2, (3 * n) // 4, n})
    rows: list[dict[str, object]] = []
    for x in points:
        confidence = 1.0 - alpha
        scipy_wilson = stats.binomtest(x, n).proportion_ci(
            confidence_level=confidence,
            method="wilson",
        )
        scipy_cp = stats.binomtest(x, n).proportion_ci(
            confidence_level=confidence,
            method="exact",
        )
        oracle_map = {
            "wilson": (scipy_wilson.low, scipy_wilson.high),
            "clopper_pearson": (scipy_cp.low, scipy_cp.high),
        }
        for method in ("wilson", "clopper_pearson"):
            interval = grids[(alpha, method)]
            oracle = oracle_map[method]
            rows.append(
                {
                    "n": n,
                    "x": x,
                    "alpha": alpha,
                    "method": method,
                    "oracle": "scipy_binomtest",
                    "gate_applicable": True,
                    "oracle_note": "independent SciPy proportion interval",
                    "lower_error": float(abs(interval.lower[x] - oracle[0])),
                    "upper_error": float(abs(interval.upper[x] - oracle[1])),
                }
            )

        p_hat = x / n
        z_value = stats.norm.ppf(1.0 - alpha / 2.0)
        wald_half_width = z_value * math.sqrt(p_hat * (1.0 - p_hat) / n)
        wald_oracle = (p_hat - wald_half_width, p_hat + wald_half_width)
        wald = grids[(alpha, "wald")]
        rows.append(
            {
                "n": n,
                "x": x,
                "alpha": alpha,
                "method": "wald",
                "oracle": "independent_unclipped_formula",
                "gate_applicable": True,
                "oracle_note": "CP-03 legacy Wald formula without clipping",
                "lower_error": float(abs(wald.lower[x] - wald_oracle[0])),
                "upper_error": float(abs(wald.upper[x] - wald_oracle[1])),
            }
        )

        for method, statsmodels_method in (
            ("wilson", "wilson"),
            ("clopper_pearson", "beta"),
            ("wald", "normal"),
            ("jeffreys", "jeffreys"),
        ):
            oracle = proportion_confint(x, n, alpha=alpha, method=statsmodels_method)
            interval = grids[(alpha, method)]
            rows.append(
                {
                    "n": n,
                    "x": x,
                    "alpha": alpha,
                    "method": method,
                    "oracle": f"statsmodels_{statsmodels_method}",
                    "gate_applicable": method != "wald",
                    "oracle_note": (
                        "statsmodels normal clips to [0,1], unlike the approved legacy Wald contract"
                        if method == "wald"
                        else "independent statsmodels reference"
                    ),
                    "lower_error": float(abs(interval.lower[x] - oracle[0])),
                    "upper_error": float(abs(interval.upper[x] - oracle[1])),
                }
            )
    return rows


def _hash_interval_grids(
    n: int,
    grids: dict[tuple[float, str], IntervalGrid],
) -> str:
    digest = hashlib.sha256()
    digest.update(CANDIDATE_SHA.encode("ascii"))
    digest.update(np.asarray((n,), dtype="<i8").tobytes())
    for alpha in ALPHAS:
        for method in METHODS:
            interval = grids[(alpha, method)]
            digest.update(np.asarray((alpha,), dtype="<f8").tobytes())
            digest.update(method.encode("ascii"))
            digest.update(np.asarray(interval.lower, dtype="<f8").tobytes())
            digest.update(np.asarray(interval.upper, dtype="<f8").tobytes())
    return digest.hexdigest()


def calibrate_n(
    n: int,
    *,
    linear_step: float | None,
    expected_widths: bool = True,
    oracle: bool = True,
    batch_size: int = 256,
) -> dict[str, object]:
    """Calibrate one n shard and return compact reproducible summaries."""

    grids = all_interval_grids(n)
    summary_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    worst_rows: list[dict[str, object]] = []
    invariant_rows = [
        invariant_row(n, alpha, grids[(alpha, method)])
        for alpha in ALPHAS
        for method in METHODS
    ]
    nesting = nesting_rows(n, grids)
    oracle_rows = (
        [row for alpha in ALPHAS for row in oracle_errors(n, alpha, grids)]
        if oracle
        else []
    )
    high_precision_triggers: list[dict[str, object]] = []

    for alpha in ALPHAS:
        width_matrix = np.column_stack(
            [grids[(alpha, method)].width for method in METHODS]
        )
        for method_index, method in enumerate(METHODS):
            interval = grids[(alpha, method)]
            p, origin_codes = probability_grid_with_origins(
                n,
                interval,
                linear_step=linear_step,
            )
            coverage, first, last = coverage_from_intervals(
                n, p, interval.lower, interval.upper
            )
            if expected_widths:
                width_values, retained_mass = expected_width_matrix(
                    n, p, width_matrix, batch_size=batch_size
                )
                method_width = width_values[:, method_index]
                ratio_wilson = np.divide(
                    method_width,
                    width_values[:, 0],
                    out=np.full(method_width.shape, np.nan),
                    where=width_values[:, 0] != 0.0,
                )
                ratio_cp = np.divide(
                    method_width,
                    width_values[:, 1],
                    out=np.full(method_width.shape, np.nan),
                    where=width_values[:, 1] != 0.0,
                )
                minimum_retained_mass = float(np.min(retained_mass))
            else:
                method_width = np.full(p.shape, np.nan)
                ratio_wilson = np.full(p.shape, np.nan)
                ratio_cp = np.full(p.shape, np.nan)
                minimum_retained_mass = math.nan

            nominal = 1.0 - alpha
            deficit = np.maximum(0.0, nominal - coverage)
            excess = np.maximum(0.0, coverage - nominal)
            worst_index = int(np.argmin(coverage))
            worst_deficit = float(deficit[worst_index])
            common = {
                "n": n,
                "alpha": alpha,
                "method": method,
                "interval_kind": interval.interval_kind,
                "cell_count": int(p.size),
            }
            summary_rows.append(
                {
                    **common,
                    "coverage_min": float(coverage[worst_index]),
                    "coverage_max": float(np.max(coverage)),
                    "coverage_mean": float(np.mean(coverage)),
                    "max_undercoverage": worst_deficit,
                    "max_excess_coverage": float(np.max(excess)),
                    "undercoverage_tier": undercoverage_tier(worst_deficit),
                    "worst_p": float(p[worst_index]),
                    "worst_origin": ORIGIN_LABELS[int(origin_codes[worst_index])],
                    "expected_width_min": float(np.nanmin(method_width)) if expected_widths else math.nan,
                    "expected_width_max": float(np.nanmax(method_width)) if expected_widths else math.nan,
                    "expected_width_mean": float(np.nanmean(method_width)) if expected_widths else math.nan,
                    "mean_width_ratio_vs_wilson": float(np.nanmean(ratio_wilson)) if expected_widths else math.nan,
                    "mean_width_ratio_vs_clopper_pearson": float(np.nanmean(ratio_cp)) if expected_widths else math.nan,
                    "minimum_retained_pmf_mass": minimum_retained_mass,
                }
            )
            worst_rows.append(
                {
                    **common,
                    "p": float(p[worst_index]),
                    "coverage": float(coverage[worst_index]),
                    "nominal": nominal,
                    "undercoverage": worst_deficit,
                    "tier": undercoverage_tier(worst_deficit),
                    "first_x": int(first[worst_index]),
                    "last_x": int(last[worst_index]),
                    "origin": ORIGIN_LABELS[int(origin_codes[worst_index])],
                }
            )
            if method == "clopper_pearson" and coverage[worst_index] < nominal - 1e-12:
                high_precision_triggers.append({**worst_rows[-1], "trigger": "cp_undercoverage"})
            if method in {"wilson", "jeffreys"} and (
                worst_deficit > 0.030
                or abs(p[worst_index] - interval.lower[int(first[worst_index])]) < 1e-10
            ):
                high_precision_triggers.append(
                    {**worst_rows[-1], "trigger": "material_minimum_or_endpoint"}
                )

            regimes = event_regime(n, p)
            for regime_code, regime_label in enumerate(EVENT_REGIME_LABELS):
                selected = regimes == regime_code
                if not np.any(selected):
                    continue
                event_rows.append(
                    {
                        **common,
                        "event_regime": regime_label,
                        "cells": int(np.sum(selected)),
                        "coverage_min": float(np.min(coverage[selected])),
                        "coverage_mean": float(np.mean(coverage[selected])),
                        "max_undercoverage": float(np.max(deficit[selected])),
                        "expected_width_mean": float(np.nanmean(method_width[selected])) if expected_widths else math.nan,
                    }
                )

    return {
        "n": n,
        "interval_hash": _hash_interval_grids(n, grids),
        "interval_rows": len(ALPHAS) * len(METHODS) * (n + 1),
        "summaries": summary_rows,
        "event_regimes": event_rows,
        "worst_cases": worst_rows,
        "invariants": invariant_rows,
        "nesting": nesting,
        "oracles": oracle_rows,
        "high_precision_triggers": high_precision_triggers,
    }


def write_calibration_result(result: dict[str, object], output_dir: Path, stem: str) -> None:
    """Persist one or more n-shards as compact checkpoint artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for key in (
        "summaries",
        "event_regimes",
        "worst_cases",
        "invariants",
        "nesting",
        "oracles",
        "high_precision_triggers",
    ):
        rows = result.get(key, [])
        frame = pd.DataFrame(rows)
        frame.to_parquet(output_dir / f"{stem}_{key}.parquet", index=False)
    inventory = pd.DataFrame(
        [
            {
                "n": result["n"],
                "interval_rows": result["interval_rows"],
                "sha256": result["interval_hash"],
                "candidate_sha": CANDIDATE_SHA,
                "experiment_version": EXPERIMENT_VERSION,
            }
        ]
    )
    inventory.to_parquet(output_dir / f"{stem}_interval_inventory.parquet", index=False)
