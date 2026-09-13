"""Independent, memory-bounded acceptance localization for CP06-H.

The localization formulas are independent of the production endpoint
implementation.  For production methods the resulting transition and its two
neighbors are then checked against the public count API.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import warnings

import mpmath as mp
import numpy as np
from scipy import stats

from experiments.proportion_ci_calibration.high_precision import (
    classify_coverage_verdict,
    high_precision_binomial_range,
    high_precision_interval,
)
from pyMagicStat.inference import PopulationProportionCI


PRODUCTION_METHODS = ("wilson", "clopper_pearson", "wald")


class AcceptanceLocalizationError(RuntimeError):
    """Raised when independent localization cannot be validated."""


@dataclass(frozen=True)
class AcceptanceLocalization:
    first_x: int
    last_x: int
    acceptance_kind: str
    localization_method: str
    boundary_validation: str

    @property
    def is_empty(self) -> bool:
        return self.first_x > self.last_x

    @property
    def runs(self) -> list[tuple[int, int]]:
        return [] if self.is_empty else [(self.first_x, self.last_x)]


def _production_interval(method: str, n: int, x: int, alpha: float) -> tuple[float, float]:
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


def _jeffreys_interval(n: int, x: int, alpha: float) -> tuple[float, float]:
    return (
        float(stats.beta.ppf(alpha / 2.0, x + 0.5, n - x + 0.5)),
        float(stats.beta.ppf(1.0 - alpha / 2.0, x + 0.5, n - x + 0.5)),
    )


def interval_for_cell(method: str, n: int, x: int, alpha: float) -> tuple[float, float]:
    if method == "jeffreys":
        return _jeffreys_interval(n, x, alpha)
    if method not in PRODUCTION_METHODS:
        raise ValueError(f"unknown method: {method}")
    return _production_interval(method, n, x, alpha)


def _first_true(low: int, high: int, predicate) -> int:
    """Return the first true index for a false-then-true predicate."""

    answer = high + 1
    while low <= high:
        middle = (low + high) // 2
        if predicate(middle):
            answer = middle
            high = middle - 1
        else:
            low = middle + 1
    return answer


def _last_true(low: int, high: int, predicate) -> int:
    """Return the last true index for a true-then-false predicate."""

    answer = low - 1
    while low <= high:
        middle = (low + high) // 2
        if predicate(middle):
            answer = middle
            low = middle + 1
        else:
            high = middle - 1
    return answer


def _wilson_bounds(n: int, alpha: float, p: float) -> tuple[int, int]:
    if p == 0.0:
        return 0, 0
    if p == 1.0:
        return n, n
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    radius = z * math.sqrt(n * p * (1.0 - p))
    lower = n * p - radius
    upper = n * p + radius
    return math.ceil(np.nextafter(lower, -math.inf)), math.floor(
        np.nextafter(upper, math.inf)
    )


def _clopper_pearson_bounds(n: int, alpha: float, p: float) -> tuple[int, int]:
    tail = alpha / 2.0
    first = _first_true(
        0,
        n,
        lambda x: bool(stats.binom.cdf(x, n, p) >= tail),
    )
    last = _last_true(
        0,
        n,
        lambda x: bool(stats.binom.sf(x - 1, n, p) >= tail),
    )
    return first, last


def _wald_bounds(n: int, alpha: float, p: float) -> tuple[int, int]:
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    discriminant = z2 * z2 + 4.0 * n * z2 * p * (1.0 - p)
    root = math.sqrt(max(0.0, discriminant))
    denominator = 2.0 * (n + z2)
    y_lower = (2.0 * n * p + z2 - root) / denominator
    y_upper = (2.0 * n * p + z2 + root) / denominator
    return math.ceil(np.nextafter(n * y_lower, -math.inf)), math.floor(
        np.nextafter(n * y_upper, math.inf)
    )


def _jeffreys_bounds(n: int, alpha: float, p: float) -> tuple[int, int]:
    first = _first_true(
        0,
        n,
        lambda x: _jeffreys_interval(n, x, alpha)[1] >= p,
    )
    last = _last_true(
        0,
        n,
        lambda x: _jeffreys_interval(n, x, alpha)[0] <= p,
    )
    return first, last


def _validate_and_reconcile(
    method: str,
    n: int,
    alpha: float,
    p: float,
    first: int,
    last: int,
) -> tuple[int, int, str]:
    """Reconcile at most two rounding cells, then prove both transitions."""

    if method not in PRODUCTION_METHODS:
        return first, last, json.dumps(
            {"source": "independent_beta_inversion", "production_api": "not_applicable"},
            sort_keys=True,
            separators=(",", ":"),
        )

    def contains(x: int) -> bool:
        if x < 0 or x > n:
            return False
        lower, upper = _production_interval(method, n, x, alpha)
        return lower <= p <= upper

    independent = (first, last)
    first = min(max(first, 0), n + 1)
    last = min(max(last, -1), n)
    for _ in range(3):
        if first > 0 and contains(first - 1):
            first -= 1
        elif first <= n and not contains(first):
            first += 1
        else:
            break
    for _ in range(3):
        if last < n and contains(last + 1):
            last += 1
        elif last >= 0 and not contains(last):
            last -= 1
        else:
            break

    checks = {
        "first_included": first <= n and contains(first),
        "before_first_excluded": first == 0 or not contains(first - 1),
        "last_included": last >= 0 and contains(last),
        "after_last_excluded": last == n or not contains(last + 1),
    }
    if first > last:
        checks = {
            "empty": True,
            "independent_empty": independent[0] > independent[1],
        }
    if not all(checks.values()) or abs(first - independent[0]) > 2 or abs(last - independent[1]) > 2:
        raise AcceptanceLocalizationError(
            f"{method} localization did not validate for n={n}, alpha={alpha}, p={p}: "
            f"independent={independent}, reconciled={(first, last)}, checks={checks}"
        )
    validation = {
        "source": "public PopulationProportionCI.from_counts API",
        "independent_bounds": list(independent),
        "validated_bounds": [first, last],
        "rounding_reconciliation": [first - independent[0], last - independent[1]],
        "checks": checks,
    }
    return first, last, json.dumps(validation, sort_keys=True, separators=(",", ":"))


def localize_acceptance(method: str, n: int, alpha: float, p: float) -> AcceptanceLocalization:
    if n < 1 or not 0.0 <= p <= 1.0:
        raise ValueError("n must be positive and p must be in [0,1]")
    if method == "wilson":
        first, last = _wilson_bounds(n, alpha, p)
        localization_method = "independent_score_inversion"
    elif method == "clopper_pearson":
        first, last = _clopper_pearson_bounds(n, alpha, p)
        localization_method = "independent_equal_tail_binomial_inversion"
    elif method == "wald":
        first, last = _wald_bounds(n, alpha, p)
        localization_method = "independent_unclipped_wald_quadratic"
    elif method == "jeffreys":
        first, last = _jeffreys_bounds(n, alpha, p)
        localization_method = "independent_monotone_beta_endpoint_inversion"
    else:
        raise ValueError(f"unknown method: {method}")
    first, last, validation = _validate_and_reconcile(
        method, n, alpha, p, first, last
    )
    return AcceptanceLocalization(
        first_x=first,
        last_x=last,
        acceptance_kind="contiguous_integer_range" if first <= last else "empty",
        localization_method=localization_method,
        boundary_validation=validation,
    )


def stable_binomial_coverage(n: int, p: float, first: int, last: int) -> float:
    if first > last:
        return 0.0
    first = max(0, int(first))
    last = min(int(n), int(last))
    if p == 0.0:
        return 1.0 if first <= 0 <= last else 0.0
    if p == 1.0:
        return 1.0 if first <= n <= last else 0.0
    mean = n * p
    if last < mean:
        value = stats.binom.cdf(last, n, p) - stats.binom.cdf(first - 1, n, p)
    elif first > mean:
        value = stats.binom.sf(first - 1, n, p) - stats.binom.sf(last, n, p)
    else:
        value = 1.0 - stats.binom.cdf(first - 1, n, p) - stats.binom.sf(last, n, p)
    return float(min(1.0, max(0.0, value)))


def endpoint_proximity_for_localization(
    method: str,
    n: int,
    alpha: float,
    p: float,
    localization: AcceptanceLocalization,
) -> dict[str, object]:
    candidates = sorted(
        {
            x
            for x in (
                localization.first_x - 1,
                localization.first_x,
                localization.last_x,
                localization.last_x + 1,
            )
            if 0 <= x <= n
        }
    )
    matches: list[dict[str, object]] = []
    for x in candidates:
        lower, upper = interval_for_cell(method, n, x, alpha)
        for kind, endpoint in (("lower", lower), ("upper", upper)):
            matches.append(
                {
                    "x": x,
                    "kind": kind,
                    "endpoint": endpoint,
                    "distance": abs(endpoint - p),
                    "side": "at" if p == endpoint else ("below" if p < endpoint else "above"),
                }
            )
    nearest = min(matches, key=lambda row: (row["distance"], row["x"], row["kind"]))
    return {
        "is_near": bool(nearest["distance"] < 1e-10),
        "nearest_distance": float(nearest["distance"]),
        "nearest": nearest,
        "examined": matches,
    }


def high_precision_recheck(
    method: str,
    n: int,
    alpha: float,
    p: float,
    coverage_float64: float,
    localization: AcceptanceLocalization,
    *,
    digits: int = 80,
) -> dict[str, object]:
    """Re-localize transition cells with HP endpoints and classify the result."""

    if digits < 80:
        raise ValueError("CP-04 requires at least 80 decimal digits")
    with mp.workdps(digits):
        p_hp = mp.mpf(str(float(p)))

        def hp_contains(x: int) -> bool:
            if x < 0 or x > n:
                return False
            lower, upper = high_precision_interval(method, n, x, alpha, digits=digits)
            return bool(lower <= p_hp <= upper)

        first = localization.first_x
        last = localization.last_x
        for _ in range(4):
            if first > 0 and hp_contains(first - 1):
                first -= 1
            elif first <= n and not hp_contains(first):
                first += 1
            else:
                break
        for _ in range(4):
            if last < n and hp_contains(last + 1):
                last += 1
            elif last >= 0 and not hp_contains(last):
                last -= 1
            else:
                break
        context_consistent = (
            first > last
            or (
                hp_contains(first)
                and hp_contains(last)
                and (first == 0 or not hp_contains(first - 1))
                and (last == n or not hp_contains(last + 1))
            )
        )
        hp_coverage = high_precision_binomial_range(
            n,
            p,
            first,
            last,
            digits=digits,
        )
        verdict = classify_coverage_verdict(
            method,
            n,
            alpha,
            coverage_float64,
            hp_coverage,
            acceptance_changed=(first, last)
            != (localization.first_x, localization.last_x),
            consistent_float_representation=context_consistent,
            digits=digits,
            audit_error=None if context_consistent else "HP transition could not be validated",
        )
        return {
            "method": method,
            "n": n,
            "alpha": alpha,
            "p": p,
            "digits": digits,
            "p_hp": mp.nstr(p_hp, digits),
            "acceptance_runs_float64": json.dumps(localization.runs, separators=(",", ":")),
            "acceptance_runs_hp": json.dumps(
                [] if first > last else [(first, last)], separators=(",", ":")
            ),
            "acceptance_changed": (first, last)
            != (localization.first_x, localization.last_x),
            "coverage_float64": coverage_float64,
            "coverage_hp": mp.nstr(hp_coverage, digits),
            "coverage_hp_float": float(hp_coverage),
            **verdict,
        }
