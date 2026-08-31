"""Deterministic CP-06 calibration primitives.

Production endpoints are obtained only through
``PopulationProportionCI.from_counts``.  Independent formulas in this module
are restricted to reference oracles and the non-production Jeffreys comparator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit, gammaln, logsumexp
from statsmodels.stats.proportion import proportion_confint

from pyMagicStat.inference import PopulationProportionCI


CANDIDATE_SHA = "2df5b90a5395163e723f9c52aafbb91fdce96d43"
CP04_DOCUMENT_SHA = "63eaaed6842e2f82473bfa857524645123f95218"
EXPERIMENT_VERSION = "proportion-ci-cp06-v2"
HARNESS_SCHEMA_VERSION = "cp06-harness-schema-v2"
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
    5: "optimizer",
}
DEFAULT_ENDPOINT_CACHE_ROOT = (
    Path(tempfile.gettempdir())
    / "pymagicstats_cp06_endpoint_cache"
    / CANDIDATE_SHA
    / HARNESS_SCHEMA_VERSION
)


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


@dataclass(frozen=True)
class CoverageEvaluation:
    coverage: np.ndarray
    first: np.ndarray
    last: np.ndarray
    run_count: np.ndarray
    acceptance_kind: str
    endpoint_monotone: bool


def _endpoint_payload_hash(
    metadata: dict[str, object],
    lower: np.ndarray,
    upper: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(np.asarray(lower, dtype="<f8").tobytes())
    digest.update(np.asarray(upper, dtype="<f8").tobytes())
    return digest.hexdigest()


class EndpointGridCache:
    """Validated persistent cache for endpoint grids produced by the frozen API."""

    def __init__(self, root: Path = DEFAULT_ENDPOINT_CACHE_ROOT):
        self.root = Path(root)

    def path_for(self, n: int, alpha: float, method: str) -> Path:
        identity = hashlib.sha256(
            f"{n}|{float(alpha).hex()}|{method}".encode("ascii")
        ).hexdigest()[:20]
        return self.root / f"n_{n:07d}" / f"{method}_{identity}.npz"

    @staticmethod
    def _identity(n: int, alpha: float, method: str) -> dict[str, object]:
        return {
            "candidate_sha": CANDIDATE_SHA,
            "harness_schema_version": HARNESS_SCHEMA_VERSION,
            "n": int(n),
            "alpha_hex": float(alpha).hex(),
            "method": method,
        }

    def load(self, n: int, alpha: float, method: str) -> IntervalGrid | None:
        path = self.path_for(n, alpha, method)
        if not path.exists():
            return None
        identity = self._identity(n, alpha, method)
        try:
            with np.load(path, allow_pickle=False) as payload:
                lower = np.asarray(payload["lower"], dtype=np.float64)
                upper = np.asarray(payload["upper"], dtype=np.float64)
                metadata = json.loads(str(payload["metadata"].item()))
            endpoint_hash = metadata.pop("endpoint_sha256")
            if any(metadata.get(key) != value for key, value in identity.items()):
                return None
            if lower.shape != (n + 1,) or upper.shape != (n + 1,):
                return None
            if endpoint_hash != _endpoint_payload_hash(metadata, lower, upper):
                return None
            return IntervalGrid(
                lower,
                upper,
                method,
                str(metadata["interval_kind"]),
                str(metadata["source"]),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def store(self, n: int, alpha: float, interval: IntervalGrid) -> Path:
        path = self.path_for(n, alpha, interval.method)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self._identity(n, alpha, interval.method),
            "interval_kind": interval.interval_kind,
            "source": interval.source,
        }
        metadata["endpoint_sha256"] = _endpoint_payload_hash(
            metadata,
            interval.lower,
            interval.upper,
        )
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                lower=np.asarray(interval.lower, dtype=np.float64),
                upper=np.asarray(interval.upper, dtype=np.float64),
                metadata=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
            )
        os.replace(temporary, path)
        return path

    def get_or_create(
        self,
        n: int,
        alpha: float,
        method: str,
        factory,
    ) -> IntervalGrid:
        cached = self.load(n, alpha, method)
        if cached is not None:
            return cached
        interval = factory()
        if interval.method != method:
            raise ValueError("endpoint cache factory returned the wrong method")
        self.store(n, alpha, interval)
        validated = self.load(n, alpha, method)
        if validated is None:
            raise RuntimeError("new endpoint cache payload failed validation")
        return validated


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


def all_interval_grids(
    n: int,
    endpoint_cache: EndpointGridCache | None = None,
) -> dict[tuple[float, str], IntervalGrid]:
    grids: dict[tuple[float, str], IntervalGrid] = {}
    for alpha in ALPHAS:
        for method in PRODUCTION_METHODS:
            factory = lambda alpha=alpha, method=method: production_interval_grid(
                n, alpha, method
            )
            grids[(alpha, method)] = (
                factory()
                if endpoint_cache is None
                else endpoint_cache.get_or_create(n, alpha, method, factory)
            )
        jeffreys_factory = lambda alpha=alpha: jeffreys_interval_grid(n, alpha)
        grids[(alpha, "jeffreys")] = (
            jeffreys_factory()
            if endpoint_cache is None
            else endpoint_cache.get_or_create(
                n,
                alpha,
                "jeffreys",
                jeffreys_factory,
            )
        )
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


def endpoints_are_monotone(lower: np.ndarray, upper: np.ndarray) -> bool:
    """Return whether searchsorted is structurally valid for both endpoints."""

    return bool(np.all(np.diff(lower) >= 0.0) and np.all(np.diff(upper) >= 0.0))


def acceptance_runs(
    lower: np.ndarray,
    upper: np.ndarray,
    p: float,
) -> list[tuple[int, int]]:
    """Canonical inclusive runs reconstructing A(p)."""

    return outcome_runs((lower <= p) & (p <= upper))


def outcome_runs(selected_outcomes: np.ndarray) -> list[tuple[int, int]]:
    """Canonical inclusive runs reconstructing a boolean outcome mask."""

    indices = np.flatnonzero(np.asarray(selected_outcomes, dtype=bool))
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate((indices[:1], indices[breaks + 1]))
    stops = np.concatenate((indices[breaks], indices[-1:]))
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _contiguous_coverage(
    n: int,
    probabilities: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> CoverageEvaluation:
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
        cdf_high = stats.binom.cdf(last[interior], n, probabilities[interior])
        cdf_low = stats.binom.cdf(
            first[interior] - 1, n, probabilities[interior]
        )
        coverage[interior] = cdf_high - cdf_low
    return CoverageEvaluation(
        coverage,
        first,
        last,
        valid.astype(np.int32),
        "monotone_contiguous",
        True,
    )


class _FenwickTree:
    def __init__(self, size: int):
        self.size = size
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int, delta: int) -> None:
        position = index + 1
        while position <= self.size:
            self.tree[position] += delta
            position += position & -position

    def total(self) -> int:
        position = self.size
        result = 0
        while position:
            result += int(self.tree[position])
            position -= position & -position
        return result

    def find_by_order(self, order: int) -> int:
        """Return the zero-based index of the requested one-valued order."""

        if order < 0 or order >= self.total():
            raise IndexError("Fenwick order outside active range")
        position = 0
        accumulated = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            candidate = position + bit
            if (
                candidate <= self.size
                and accumulated + int(self.tree[candidate]) <= order
            ):
                position = candidate
                accumulated += int(self.tree[candidate])
            bit >>= 1
        return position


class _AcceptanceRunIndex:
    """Dynamic exact run representation for an endpoint sweep over p."""

    def __init__(self, size: int):
        self.active = np.zeros(size, dtype=bool)
        self.starts = np.zeros(size, dtype=bool)
        self.ends = np.zeros(size, dtype=bool)
        self.start_tree = _FenwickTree(size)
        self.end_tree = _FenwickTree(size)

    def _start_value(self, index: int) -> bool:
        return bool(
            self.active[index] and (index == 0 or not self.active[index - 1])
        )

    def _end_value(self, index: int) -> bool:
        return bool(
            self.active[index]
            and (index == self.active.size - 1 or not self.active[index + 1])
        )

    @staticmethod
    def _update_flag(
        flags: np.ndarray,
        tree: _FenwickTree,
        index: int,
        value: bool,
    ) -> None:
        if bool(flags[index]) != value:
            tree.add(index, 1 if value else -1)
            flags[index] = value

    def set_active(self, index: int, value: bool) -> None:
        if bool(self.active[index]) == value:
            return
        self.active[index] = value
        for candidate in (index, index + 1):
            if 0 <= candidate < self.active.size:
                self._update_flag(
                    self.starts,
                    self.start_tree,
                    candidate,
                    self._start_value(candidate),
                )
        for candidate in (index - 1, index):
            if 0 <= candidate < self.active.size:
                self._update_flag(
                    self.ends,
                    self.end_tree,
                    candidate,
                    self._end_value(candidate),
                )

    def runs(self) -> list[tuple[int, int]]:
        count = self.start_tree.total()
        if count != self.end_tree.total():
            raise RuntimeError("acceptance run index lost boundary parity")
        return [
            (
                self.start_tree.find_by_order(order),
                self.end_tree.find_by_order(order),
            )
            for order in range(count)
        ]


def _explicit_coverage(
    n: int,
    probabilities: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    probability_batch_size: int,
    outcome_batch_size: int,
) -> CoverageEvaluation:
    coverage = np.zeros(probabilities.size, dtype=np.float64)
    first = np.full(probabilities.size, n + 1, dtype=np.int64)
    last = np.full(probabilities.size, -1, dtype=np.int64)
    run_count = np.zeros(probabilities.size, dtype=np.int32)

    del probability_batch_size, outcome_batch_size  # sweep output is batch invariant
    finite = np.isfinite(lower) & np.isfinite(upper)
    valid_indices = np.flatnonzero(finite)
    lower_order = valid_indices[np.argsort(lower[valid_indices], kind="stable")]
    upper_order = valid_indices[np.argsort(upper[valid_indices], kind="stable")]
    query_order = np.argsort(probabilities, kind="stable")
    run_index = _AcceptanceRunIndex(n + 1)
    lower_cursor = 0
    upper_cursor = 0
    current_runs: list[tuple[int, int]] = []
    group_indices: list[int] = []

    def flush_group() -> None:
        if not group_indices:
            return
        indices = np.asarray(group_indices, dtype=np.int64)
        coverage[indices] = _coverage_for_runs_array(
            n,
            probabilities[indices],
            current_runs,
        )
        if current_runs:
            first[indices] = current_runs[0][0]
            last[indices] = current_runs[-1][1]
        run_count[indices] = len(current_runs)

    for query_index in query_order:
        probability = float(probabilities[query_index])
        changed = False
        while (
            lower_cursor < lower_order.size
            and lower[lower_order[lower_cursor]] <= probability
        ):
            run_index.set_active(int(lower_order[lower_cursor]), True)
            lower_cursor += 1
            changed = True
        while (
            upper_cursor < upper_order.size
            and upper[upper_order[upper_cursor]] < probability
        ):
            run_index.set_active(int(upper_order[upper_cursor]), False)
            upper_cursor += 1
            changed = True
        if changed:
            flush_group()
            group_indices.clear()
            current_runs = run_index.runs()
        group_indices.append(int(query_index))
    flush_group()
    return CoverageEvaluation(
        coverage,
        first,
        last,
        run_count,
        "explicit_nonmonotone_endpoints",
        False,
    )


def evaluate_coverage(
    n: int,
    p: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    probability_batch_size: int = 64,
    outcome_batch_size: int = 4096,
) -> CoverageEvaluation:
    probabilities = np.asarray(p, dtype=np.float64)
    if endpoints_are_monotone(lower, upper):
        return _contiguous_coverage(n, probabilities, lower, upper)
    return _explicit_coverage(
        n,
        probabilities,
        lower,
        upper,
        probability_batch_size=probability_batch_size,
        outcome_batch_size=outcome_batch_size,
    )


def coverage_from_intervals(
    n: int,
    p: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route monotone endpoints fast and enumerate nonmonotone A(p) explicitly."""

    result = evaluate_coverage(n, p, lower, upper)
    return result.coverage, result.first, result.last


def _stationary_probability(n: int, first: int, last: int) -> float | None:
    """Analytic stationary point for P(first <= Bin(n,p) <= last)."""

    if first <= 0 or last >= n or first > last:
        return None
    exponent = last - first + 1
    log_left = gammaln(n) - gammaln(first) - gammaln(n - first + 1)
    log_right = gammaln(n) - gammaln(last + 1) - gammaln(n - last)
    return float(expit((log_left - log_right) / exponent))


def _coverage_for_runs_array(
    n: int,
    probabilities: np.ndarray,
    runs: list[tuple[int, int]],
) -> np.ndarray:
    active = np.asarray(probabilities, dtype=np.float64)
    total = np.zeros(active.size, dtype=np.float64)
    for first, last in runs:
        if first == 0:
            mass = stats.binom.cdf(last, n, active)
        elif last == n:
            mass = stats.binom.sf(first - 1, n, active)
        else:
            cdf_mass = stats.binom.cdf(last, n, active) - stats.binom.cdf(
                first - 1, n, active
            )
            sf_mass = stats.binom.sf(first - 1, n, active) - stats.binom.sf(
                last, n, active
            )
            mass = np.maximum(np.maximum(cdf_mass, sf_mass), 0.0)
        total += np.asarray(mass, dtype=np.float64)
    return np.clip(total, 0.0, 1.0)


def _coverage_for_runs(n: int, probability: float, runs: list[tuple[int, int]]) -> float:
    return float(
        _coverage_for_runs_array(
            n,
            np.asarray((probability,), dtype=np.float64),
            runs,
        )[0]
    )


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
    optimized: list[float] = []
    if endpoints_are_monotone(lower, upper):
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
    else:
        for left, right, midpoint in zip(
            boundaries[:-1],
            boundaries[1:],
            midpoints,
        ):
            runs = acceptance_runs(lower, upper, float(midpoint))
            if len(runs) == 1:
                candidate = _stationary_probability(n, runs[0][0], runs[0][1])
                if candidate is not None and left < candidate < right:
                    stationary.append(candidate)
            elif len(runs) > 1 and right - left > 1e-12:
                bounded_left = float(np.nextafter(left, right))
                bounded_right = float(np.nextafter(right, left))
                result = optimize.minimize_scalar(
                    lambda probability: _coverage_for_runs(n, probability, runs),
                    bounds=(bounded_left, bounded_right),
                    method="bounded",
                    options={"xatol": 5e-13, "maxiter": 256},
                )
                if result.success and bounded_left <= result.x <= bounded_right:
                    optimized.append(float(result.x))

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
    if optimized:
        optimized_array = np.unique(np.asarray(optimized, dtype=np.float64))
        pieces.append(optimized_array)
        origins.append(np.full(optimized_array.size, 5, dtype=np.int8))

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
    include_base_grid: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    induced, induced_origin = induced_probability_grid(
        n, interval.lower, interval.upper
    )
    if not include_base_grid:
        return induced, induced_origin
    base = base_probability_grid(n, linear_step=linear_step)
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


def outcome_set_probability(
    n: int,
    p: np.ndarray,
    selected_outcomes: np.ndarray,
    *,
    probability_batch_size: int = 64,
    outcome_batch_size: int = 4096,
) -> np.ndarray:
    """Stable, bounded probability mass of a fixed explicit outcome set."""

    probabilities = np.asarray(p, dtype=np.float64)
    selected = np.asarray(selected_outcomes, dtype=bool)
    if selected.shape != (n + 1,):
        raise ValueError("selected_outcomes must correspond to x=0..n")
    result = np.zeros(probabilities.size, dtype=np.float64)
    for p_start in range(0, probabilities.size, probability_batch_size):
        p_stop = min(p_start + probability_batch_size, probabilities.size)
        active = probabilities[p_start:p_stop]
        log_total = np.full(active.size, -np.inf, dtype=np.float64)
        for x_start in range(0, n + 1, outcome_batch_size):
            x_stop = min(x_start + outcome_batch_size, n + 1)
            selected_chunk = selected[x_start:x_stop]
            if not np.any(selected_chunk):
                continue
            outcomes = np.arange(x_start, x_stop, dtype=np.int64)[selected_chunk]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_pmf = stats.binom.logpmf(outcomes[None, :], n, active[:, None])
            log_total = np.logaddexp(log_total, logsumexp(log_pmf, axis=1))
        result[p_start:p_stop] = np.exp(log_total)
    return result


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
    include_base_grid: bool = True,
    endpoint_cache: EndpointGridCache | None = None,
) -> dict[str, object]:
    """Calibrate one n shard and return compact reproducible summaries."""

    grids = all_interval_grids(n, endpoint_cache=endpoint_cache)
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
    adversarial_minima: list[dict[str, object]] = []
    wald_pathology_summaries: list[dict[str, object]] = []
    wald_pathology_worst_cases: list[dict[str, object]] = []

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
                include_base_grid=include_base_grid,
            )
            coverage_evaluation = evaluate_coverage(
                n,
                p,
                interval.lower,
                interval.upper,
                probability_batch_size=batch_size,
            )
            coverage = coverage_evaluation.coverage
            first = coverage_evaluation.first
            last = coverage_evaluation.last
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
            worst_p = float(p[worst_index])
            worst_runs = acceptance_runs(interval.lower, interval.upper, worst_p)
            runs_json = json.dumps(worst_runs, separators=(",", ":"))
            worst_origin = ORIGIN_LABELS[int(origin_codes[worst_index])]
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
                    "worst_p": worst_p,
                    "worst_origin": worst_origin,
                    "acceptance_kind": coverage_evaluation.acceptance_kind,
                    "endpoint_monotone": coverage_evaluation.endpoint_monotone,
                    "worst_acceptance_run_count": int(
                        coverage_evaluation.run_count[worst_index]
                    ),
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
                    "p": worst_p,
                    "coverage": float(coverage[worst_index]),
                    "nominal": nominal,
                    "undercoverage": worst_deficit,
                    "tier": undercoverage_tier(worst_deficit),
                    "first_x": int(first[worst_index]),
                    "last_x": int(last[worst_index]),
                    "origin": worst_origin,
                    "acceptance_kind": coverage_evaluation.acceptance_kind,
                    "acceptance_run_count": int(
                        coverage_evaluation.run_count[worst_index]
                    ),
                    "acceptance_runs": runs_json,
                }
            )
            if not include_base_grid:
                if coverage_evaluation.endpoint_monotone:
                    search_method = {
                        "boundary": "endpoint_partition_boundary",
                        "nextafter": "endpoint_partition_nextafter",
                        "midpoint": "endpoint_partition_midpoint",
                        "stationary": "analytic_stationary_candidate",
                        "grid": "preregistered_grid",
                    }[worst_origin]
                    optimizer_status = "not_required_partition_complete"
                else:
                    search_method = {
                        "optimizer": "explicit_acceptance_set_bounded_optimizer",
                        "stationary": "explicit_acceptance_set_analytic_stationary",
                    }.get(worst_origin, "explicit_acceptance_set_enumeration")
                    optimizer_status = (
                        "converged_xatol_5e-13"
                        if worst_origin == "optimizer"
                        else "not_selected_or_not_required"
                    )
                adversarial_minima.append(
                    {
                        "n": n,
                        "alpha": alpha,
                        "method": method,
                        "p": worst_p,
                        "coverage": float(coverage[worst_index]),
                        "nominal": nominal,
                        "deficit": worst_deficit,
                        "acceptance_kind": coverage_evaluation.acceptance_kind,
                        "first_x": int(first[worst_index]),
                        "last_x": int(last[worst_index]),
                        "acceptance_run_count": int(
                            coverage_evaluation.run_count[worst_index]
                        ),
                        "acceptance_runs": runs_json,
                        "acceptance_representation": "inclusive_integer_runs_json",
                        "origin": worst_origin,
                        "search_method": search_method,
                        "optimizer_status": optimizer_status,
                    }
                )

            if method == "wald":
                outside = (interval.lower < 0.0) | (interval.upper > 1.0)
                degenerate = interval.upper == interval.lower
                p_outside = outcome_set_probability(
                    n,
                    p,
                    outside,
                    probability_batch_size=batch_size,
                )
                p_degenerate = outcome_set_probability(
                    n,
                    p,
                    degenerate,
                    probability_batch_size=batch_size,
                )
                outside_index = int(np.argmax(p_outside))
                degenerate_index = int(np.argmax(p_degenerate))
                wald_pathology_summaries.append(
                    {
                        "n": n,
                        "alpha": alpha,
                        "evaluated_p_count": int(p.size),
                        "p_outside_max": float(p_outside[outside_index]),
                        "p_outside_mean": float(np.mean(p_outside)),
                        "p_outside_worst_p": float(p[outside_index]),
                        "p_degenerate_max": float(p_degenerate[degenerate_index]),
                        "p_degenerate_mean": float(np.mean(p_degenerate)),
                        "p_degenerate_worst_p": float(p[degenerate_index]),
                        "coverage_worst_p": worst_p,
                        "p_outside_at_coverage_worst": float(p_outside[worst_index]),
                        "p_degenerate_at_coverage_worst": float(
                            p_degenerate[worst_index]
                        ),
                    }
                )
                for metric, values, index in (
                    ("p_outside", p_outside, outside_index),
                    ("p_degenerate", p_degenerate, degenerate_index),
                ):
                    wald_pathology_worst_cases.append(
                        {
                            "n": n,
                            "alpha": alpha,
                            "p": float(p[index]),
                            "metric": metric,
                            "probability": float(values[index]),
                            "origin": ORIGIN_LABELS[int(origin_codes[index])],
                            "outcome_runs": json.dumps(
                                outcome_runs(
                                    outside if metric == "p_outside" else degenerate
                                ),
                                separators=(",", ":"),
                            ),
                        }
                    )
            if method == "clopper_pearson" and coverage[worst_index] < nominal - 1e-12:
                high_precision_triggers.append({**worst_rows[-1], "trigger": "cp_undercoverage"})
            first_worst = int(first[worst_index])
            near_endpoint = (
                0 <= first_worst <= n
                and abs(worst_p - interval.lower[first_worst]) < 1e-10
            )
            if method in {"wilson", "jeffreys"} and (
                worst_deficit > 0.030 or near_endpoint
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
        "adversarial_minima": adversarial_minima,
        "wald_pathology_summaries": wald_pathology_summaries,
        "wald_pathology_worst_cases": wald_pathology_worst_cases,
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
        "adversarial_minima",
        "wald_pathology_summaries",
        "wald_pathology_worst_cases",
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
