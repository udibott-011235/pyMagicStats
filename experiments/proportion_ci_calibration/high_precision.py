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
    HARNESS_SCHEMA_VERSION,
    EndpointGridCache,
    acceptance_runs,
    jeffreys_interval_grid,
    outcome_runs,
    production_interval_grid,
)
from experiments.proportion_ci_calibration.run import SHARD_SCHEMA_VERSION
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
    "interval_kind",
    "p",
    "x",
    "first_x",
    "last_x",
    "coverage",
    "oracle",
    "acceptance_kind",
    "acceptance_runs",
    "endpoint_relation",
    "endpoint_proximity",
    "complement_x",
    "x_left",
    "x_right",
    "endpoint_kind",
    "alpha_wider",
    "alpha_narrower",
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


def high_precision_binomial_runs(
    n: int,
    p: float,
    runs: list[tuple[int, int]],
    *,
    digits: int = 80,
) -> mp.mpf:
    """Audit an explicit, potentially noncontiguous acceptance representation."""

    if digits < 80:
        raise ValueError("CP-04 requires at least 80 decimal digits")
    with mp.workdps(digits):
        return mp.fsum(
            high_precision_binomial_range(
                n,
                p,
                first,
                last,
                digits=digits,
            )
            for first, last in runs
        )


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


def reconcile_boundary_acceptance(
    method: str,
    n: int,
    alpha: float,
    p: float,
    float_lower: np.ndarray,
    float_upper: np.ndarray,
    float_runs: list[tuple[int, int]],
    *,
    digits: int = 80,
    endpoint_provider=high_precision_interval,
    proximity_tolerance: float = 1e-10,
) -> dict[str, object]:
    """Rebuild A(p) with HP endpoints for every float64-near boundary."""

    if digits < 80:
        raise ValueError("CP-04 requires at least 80 decimal digits")
    float_mask = (float_lower <= p) & (p <= float_upper)
    reconstructed_float_runs = outcome_runs(float_mask)
    hp_mask = float_mask.copy()
    candidate_indices = np.flatnonzero(
        (np.abs(float_lower - p) < proximity_tolerance)
        | (np.abs(float_upper - p) < proximity_tolerance)
    )
    relation: list[dict[str, object]] = []
    with mp.workdps(digits):
        p_hp = _mp(p)
        for x in candidate_indices:
            try:
                lower_hp, upper_hp = endpoint_provider(
                    method,
                    n,
                    int(x),
                    alpha,
                    digits=digits,
                )
            except Exception as error:
                return {
                    "consistent_float_representation": reconstructed_float_runs
                    == float_runs,
                    "acceptance_changed": False,
                    "float_runs": reconstructed_float_runs,
                    "hp_runs": [],
                    "p_hp": mp.nstr(p_hp, digits),
                    "p_hp_representation": "decimal_round_trip_of_float64",
                    "endpoint_relation": relation,
                    "error": f"{type(error).__name__}: {error}",
                }
            float_included = bool(float_mask[x])
            hp_included = bool(lower_hp <= p_hp <= upper_hp)
            hp_mask[x] = hp_included
            relations = []
            for kind, float_endpoint, hp_endpoint in (
                ("lower", float_lower[x], lower_hp),
                ("upper", float_upper[x], upper_hp),
            ):
                distance = abs(float(float_endpoint) - p)
                if distance >= proximity_tolerance:
                    continue
                if p == float_endpoint:
                    side = "at"
                elif p < float_endpoint:
                    side = "below"
                else:
                    side = "above"
                relations.append(
                    {
                        "kind": kind,
                        "side_float64": side,
                        "float64_endpoint": float(float_endpoint),
                        "high_precision_endpoint": mp.nstr(hp_endpoint, digits),
                        "distance_float64": distance,
                    }
                )
            relation.append(
                {
                    "x": int(x),
                    "float64_included": float_included,
                    "high_precision_included": hp_included,
                    "relations": relations,
                }
            )
        hp_runs = outcome_runs(hp_mask)
        return {
            "consistent_float_representation": reconstructed_float_runs == float_runs,
            "acceptance_changed": hp_runs != reconstructed_float_runs,
            "float_runs": reconstructed_float_runs,
            "hp_runs": hp_runs,
            "p_hp": mp.nstr(p_hp, digits),
            "p_hp_representation": "decimal_round_trip_of_float64",
            "endpoint_relation": relation,
            "error": None,
        }


def classify_coverage_verdict(
    method: str,
    n: int,
    alpha: float,
    coverage_float64: float,
    coverage_hp: mp.mpf,
    *,
    acceptance_changed: bool,
    consistent_float_representation: bool,
    digits: int = 80,
    audit_error: str | None = None,
) -> dict[str, object]:
    """Apply explicit CP-04 interpretation rules, with HP governing."""

    if digits < 80:
        raise ValueError("CP-04 requires at least 80 decimal digits")
    with mp.workdps(digits):
        nominal_hp = 1 - _mp(alpha)
        hp_epsilon = mp.power(10, -(digits - 20))
        coverage_is_valid = mp.isfinite(coverage_hp) and -hp_epsilon <= coverage_hp <= 1 + hp_epsilon
        deficit_hp = max(mp.mpf(0), nominal_hp - coverage_hp)
        nominal_float64 = 1.0 - float(alpha)
        deficit_float64 = max(0.0, nominal_float64 - float(coverage_float64))
        tolerance = 1e-12 if n <= 5_000 else 1e-10
        if audit_error or not consistent_float_representation or not coverage_is_valid:
            return {
                "classification": "unresolved",
                "resolved": False,
                "notes": audit_error
                or "float64 acceptance representation is inconsistent with the endpoint grid",
                "deficit_float64": deficit_float64,
                "deficit_hp": mp.nstr(deficit_hp, digits),
            }
        if acceptance_changed:
            return {
                "classification": "float64_boundary_artifact",
                "resolved": True,
                "notes": "HP endpoints changed A(p); HP acceptance and coverage govern",
                "deficit_float64": deficit_float64,
                "deficit_hp": mp.nstr(deficit_hp, digits),
            }
        hp_shortfall = deficit_hp > hp_epsilon
        float_shortfall = deficit_float64 > tolerance
        if method == "clopper_pearson":
            if hp_shortfall:
                return {
                    "classification": "unresolved",
                    "resolved": False,
                    "notes": "Clopper-Pearson shortfall persists at HP and requires STOP review",
                    "deficit_float64": deficit_float64,
                    "deficit_hp": mp.nstr(deficit_hp, digits),
                }
            return {
                "classification": "confirmed_exact_coverage",
                "resolved": True,
                "notes": "HP confirms Clopper-Pearson coverage at or above nominal",
                "deficit_float64": deficit_float64,
                "deficit_hp": mp.nstr(deficit_hp, digits),
            }
        numerical_difference = abs(coverage_hp - _mp(coverage_float64))
        if numerical_difference > tolerance and float_shortfall == hp_shortfall:
            return {
                "classification": "numerical_difference_without_claim_change",
                "resolved": True,
                "notes": "HP changes the numeric value but not the coverage claim",
                "deficit_float64": deficit_float64,
                "deficit_hp": mp.nstr(deficit_hp, digits),
            }
        if hp_shortfall:
            return {
                "classification": "confirmed_statistical_shortfall",
                "resolved": True,
                "notes": "HP confirms the shortfall with unchanged acceptance semantics",
                "deficit_float64": deficit_float64,
                "deficit_hp": mp.nstr(deficit_hp, digits),
            }
        comparator_note = (
            "; Jeffreys remains a Bayesian comparator without an exact "
            "frequentist guarantee"
            if method == "jeffreys"
            else ""
        )
        return {
            "classification": "confirmed_no_shortfall_at_audited_cell",
            "resolved": True,
            "notes": "HP confirms no shortfall at this audited cell" + comparator_note,
            "deficit_float64": deficit_float64,
            "deficit_hp": mp.nstr(deficit_hp, digits),
        }


def classify_endpoint_verdict(
    reason: str,
    n: int,
    float_lower: float,
    float_upper: float,
    hp_lower: mp.mpf,
    hp_upper: mp.mpf,
    *,
    context_consistent: bool = True,
) -> dict[str, object]:
    """Classify endpoint/oracle rows without equating computation with resolution."""

    tolerance = 1e-12 if n <= 5_000 else 1e-10
    valid_hp = mp.isfinite(hp_lower) and mp.isfinite(hp_upper) and hp_lower <= hp_upper
    if not context_consistent or not valid_hp:
        return {
            "classification": "unresolved",
            "resolved": False,
            "notes": "endpoint context is inconsistent or HP endpoints are invalid",
        }
    lower_error = float(abs(hp_lower - _mp(float_lower)))
    upper_error = float(abs(hp_upper - _mp(float_upper)))
    if reason == "bounds":
        hp_in_bounds = hp_lower >= 0 and hp_upper <= 1
        float_outside = float_lower < 0 or float_upper > 1
        if hp_in_bounds and float_outside:
            return {
                "classification": "float64_boundary_artifact",
                "resolved": True,
                "notes": "HP endpoints restore the required bounds",
            }
        if not hp_in_bounds:
            return {
                "classification": "unresolved",
                "resolved": False,
                "notes": "endpoint remains outside bounds at HP",
            }
    if max(lower_error, upper_error) <= tolerance:
        return {
            "classification": "numerical_difference_without_claim_change",
            "resolved": True,
            "notes": "independent HP endpoint agrees within the preregistered tolerance",
        }
    return {
        "classification": "unresolved",
        "resolved": False,
        "notes": "endpoint discrepancy persists beyond the preregistered tolerance",
    }


def _json_number(value: object, digits: int) -> float | str:
    """Return a reconstructible JSON scalar without non-standard NaN tokens."""

    if isinstance(value, mp.mpf):
        return mp.nstr(value, digits)
    numeric = float(value)
    return numeric if math.isfinite(numeric) else repr(numeric)


def audit_structural_predicate(
    row: dict[str, object],
    *,
    digits: int = 80,
    endpoint_provider=high_precision_interval,
    float_endpoint_provider=_float64_interval,
) -> dict[str, object]:
    """Re-evaluate the exact paired predicate that caused a structural trigger."""

    if digits < 80:
        raise ValueError("CP-04 requires at least 80 decimal digits")
    reason = str(row["reason"])
    method = str(row["method"])
    n = int(row["n"])
    alpha = float(row["alpha"])
    records: list[dict[str, object]] = []

    def endpoint(label: str, x: int, selected_alpha: float):
        float_lower, float_upper = float_endpoint_provider(
            method, n, x, selected_alpha
        )
        hp_lower, hp_upper = endpoint_provider(
            method,
            n,
            x,
            selected_alpha,
            digits=digits,
        )
        records.append(
            {
                "label": label,
                "x": x,
                "alpha": selected_alpha,
                "float64_lower": _json_number(float_lower, digits),
                "float64_upper": _json_number(float_upper, digits),
                "hp_lower": _json_number(hp_lower, digits),
                "hp_upper": _json_number(hp_upper, digits),
            }
        )
        return float_lower, float_upper, hp_lower, hp_upper

    details: dict[str, object] = {}
    try:
        with mp.workdps(digits):
            hp_epsilon = mp.power(10, -(digits - 20))
            if reason == "complement_symmetry":
                x = int(row["x"])
                complement_x = int(row["complement_x"])
                if complement_x != n - x:
                    raise ValueError("complement_x must equal n - x")
                fl, fu, hl, hu = endpoint("x", x, alpha)
                cfl, cfu, chl, chu = endpoint(
                    "complement_x", complement_x, alpha
                )
                float_lower_error = abs(fl - (1.0 - cfu))
                float_upper_error = abs(fu - (1.0 - cfl))
                hp_lower_error = abs(hl - (1 - chu))
                hp_upper_error = abs(hu - (1 - chl))
                float_passes = float_lower_error == 0.0 and float_upper_error == 0.0
                hp_passes = (
                    mp.isfinite(hp_lower_error)
                    and mp.isfinite(hp_upper_error)
                    and max(hp_lower_error, hp_upper_error) <= hp_epsilon
                )
                details = {
                    "float64_lower_error": float_lower_error,
                    "float64_upper_error": float_upper_error,
                    "hp_lower_error": mp.nstr(hp_lower_error, digits),
                    "hp_upper_error": mp.nstr(hp_upper_error, digits),
                }
            elif reason == "endpoint_monotonicity":
                x_left = int(row["x_left"])
                x_right = int(row["x_right"])
                endpoint_kind = str(row["endpoint_kind"])
                if x_right != x_left + 1:
                    raise ValueError("monotonicity requires adjacent outcomes")
                if endpoint_kind not in {"lower", "upper"}:
                    raise ValueError("endpoint_kind must be lower or upper")
                left = endpoint("x_left", x_left, alpha)
                right = endpoint("x_right", x_right, alpha)
                offset = 0 if endpoint_kind == "lower" else 1
                float_left = left[offset]
                float_right = right[offset]
                hp_left = left[offset + 2]
                hp_right = right[offset + 2]
                float_passes = float_left <= float_right
                hp_passes = (
                    mp.isfinite(hp_left)
                    and mp.isfinite(hp_right)
                    and hp_left <= hp_right
                )
                details = {
                    "endpoint_kind": endpoint_kind,
                    "float64_left": _json_number(float_left, digits),
                    "float64_right": _json_number(float_right, digits),
                    "hp_left": _json_number(hp_left, digits),
                    "hp_right": _json_number(hp_right, digits),
                }
            elif reason == "nesting":
                x = int(row["x"])
                alpha_wider = float(row["alpha_wider"])
                alpha_narrower = float(row["alpha_narrower"])
                if not alpha_wider < alpha_narrower:
                    raise ValueError("nesting requires alpha_wider < alpha_narrower")
                wide = endpoint("alpha_wider", x, alpha_wider)
                narrow = endpoint("alpha_narrower", x, alpha_narrower)
                float_passes = wide[0] <= narrow[0] and wide[1] >= narrow[1]
                hp_passes = (
                    all(mp.isfinite(value) for value in (*wide[2:], *narrow[2:]))
                    and wide[2] <= narrow[2]
                    and wide[3] >= narrow[3]
                )
                details = {
                    "float64_lower_margin": _json_number(narrow[0] - wide[0], digits),
                    "float64_upper_margin": _json_number(wide[1] - narrow[1], digits),
                    "hp_lower_margin": _json_number(narrow[2] - wide[2], digits),
                    "hp_upper_margin": _json_number(wide[3] - narrow[3], digits),
                }
            elif reason in {"bounds", "nan_or_inf"}:
                x = int(row["x"])
                fl, fu, hl, hu = endpoint("x", x, alpha)
                float_finite_ordered = math.isfinite(fl) and math.isfinite(fu) and fl <= fu
                hp_finite_ordered = mp.isfinite(hl) and mp.isfinite(hu) and hl <= hu
                if reason == "bounds":
                    float_passes = float_finite_ordered and 0.0 <= fl and fu <= 1.0
                    hp_passes = hp_finite_ordered and 0 <= hl and hu <= 1
                else:
                    float_passes = float_finite_ordered
                    hp_passes = hp_finite_ordered
                details = {
                    "float64_valid": bool(float_passes),
                    "hp_valid": bool(hp_passes),
                }
            else:
                raise ValueError(f"unknown structural predicate: {reason}")
    except Exception as error:
        return {
            "predicate_evaluated": False,
            "predicate_float64": False,
            "predicate_hp": False,
            "paired_values": json.dumps(records, sort_keys=True, separators=(",", ":")),
            "predicate_details": json.dumps(details, sort_keys=True, separators=(",", ":")),
            "classification": "unresolved",
            "resolved": False,
            "notes": f"{type(error).__name__}: {error}",
            "status": "unresolved",
        }

    if hp_passes:
        classification = (
            "confirmed_structural_invariant"
            if float_passes
            else "float64_structural_artifact"
        )
        resolved = True
        notes = "HP evaluation satisfies the paired structural predicate"
    else:
        classification = "confirmed_structural_violation"
        resolved = False
        notes = "HP evaluation confirms that the paired structural predicate fails"
    return {
        "predicate_evaluated": True,
        "predicate_float64": bool(float_passes),
        "predicate_hp": bool(hp_passes),
        "paired_values": json.dumps(records, sort_keys=True, separators=(",", ":")),
        "predicate_details": json.dumps(details, sort_keys=True, separators=(",", ":")),
        "classification": classification,
        "resolved": resolved,
        "notes": notes,
        "status": "resolved" if resolved else "unresolved",
    }


def _audit_one(arguments: tuple[dict[str, object], int]) -> dict[str, object]:
    row, digits = arguments
    interval_kind = row.get("interval_kind")
    if not isinstance(interval_kind, str) or not interval_kind:
        interval_kind = (
            "bayesian_comparator" if row["method"] == "jeffreys" else "frequentist"
        )
    common = {
        "audit_kind": row["audit_kind"],
        "reason": row["reason"],
        "checkpoint": row["checkpoint"],
        "n": int(row["n"]),
        "alpha": float(row["alpha"]),
        "method": row["method"],
        "interval_kind": interval_kind,
        "digits": digits,
    }
    if row["audit_kind"] == "coverage":
        runs_value = row.get("acceptance_runs")
        if isinstance(runs_value, str) and runs_value:
            float_runs = [tuple(item) for item in json.loads(runs_value)]
        else:
            first = int(row["first_x"])
            last = int(row["last_x"])
            float_runs = [] if first > last else [(first, last)]
        grid = _grid_for(
            {},
            int(row["n"]),
            float(row["alpha"]),
            str(row["method"]),
        )
        reconciliation = reconcile_boundary_acceptance(
            str(row["method"]),
            int(row["n"]),
            float(row["alpha"]),
            float(row["p"]),
            grid.lower,
            grid.upper,
            float_runs,
            digits=digits,
        )
        if reconciliation["error"] is None:
            value = high_precision_binomial_runs(
                int(row["n"]),
                float(row["p"]),
                reconciliation["hp_runs"],
                digits=digits,
            )
        else:
            value = mp.mpf("nan")
        float64_value = float(row["coverage"])
        verdict = classify_coverage_verdict(
            str(row["method"]),
            int(row["n"]),
            float(row["alpha"]),
            float64_value,
            value,
            acceptance_changed=bool(reconciliation["acceptance_changed"]),
            consistent_float_representation=bool(
                reconciliation["consistent_float_representation"]
            ),
            digits=digits,
            audit_error=reconciliation["error"],
        )
        with mp.workdps(digits):
            decimal_value = mp.nstr(value, digits)
            absolute_error = float(abs(value - _mp(float64_value)))
        relation_json = json.dumps(
            reconciliation["endpoint_relation"],
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            **common,
            "p": float(row["p"]),
            "p_float64": float(row["p"]),
            "p_hp": reconciliation["p_hp"],
            "p_hp_representation": reconciliation["p_hp_representation"],
            "first_x": int(row["first_x"]),
            "last_x": int(row["last_x"]),
            "acceptance_runs_float64": json.dumps(
                reconciliation["float_runs"], separators=(",", ":")
            ),
            "acceptance_runs_hp": json.dumps(
                reconciliation["hp_runs"], separators=(",", ":")
            ),
            "acceptance_changed": reconciliation["acceptance_changed"],
            "endpoint_relation": relation_json,
            "float64_coverage": float64_value,
            "coverage_float64": float64_value,
            "high_precision_coverage": decimal_value,
            "coverage_hp": decimal_value,
            "high_precision_coverage_float": float(value),
            "coverage_hp_float": float(value),
            "absolute_error": absolute_error,
            "deficit_float64": verdict["deficit_float64"],
            "deficit_hp": verdict["deficit_hp"],
            "classification": verdict["classification"],
            "resolved": verdict["resolved"],
            "notes": verdict["notes"],
            "status": "resolved" if verdict["resolved"] else "unresolved",
        }
    if row["audit_kind"] == "structural":
        verdict = audit_structural_predicate(row, digits=digits)
        context = {
            key: row.get(key)
            for key in (
                "x",
                "complement_x",
                "x_left",
                "x_right",
                "endpoint_kind",
                "alpha_wider",
                "alpha_narrower",
            )
        }
        return {**common, **context, **verdict}
    if row["audit_kind"] == "endpoint":
        x = int(row["x"])
        try:
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
        except Exception as error:
            return {
                **common,
                "x": x,
                "classification": "unresolved",
                "resolved": False,
                "notes": f"{type(error).__name__}: {error}",
                "status": "unresolved",
            }
        with mp.workdps(digits):
            decimal_lower = mp.nstr(lower, digits)
            decimal_upper = mp.nstr(upper, digits)
            lower_error = float(abs(lower - _mp(float_lower)))
            upper_error = float(abs(upper - _mp(float_upper)))
        verdict = classify_endpoint_verdict(
            str(row["reason"]),
            int(row["n"]),
            float_lower,
            float_upper,
            lower,
            upper,
        )
        return {
            **common,
            "x": x,
            "float64_lower": float_lower,
            "float64_upper": float_upper,
            "high_precision_lower": decimal_lower,
            "high_precision_upper": decimal_upper,
            "lower_error": lower_error,
            "upper_error": upper_error,
            "endpoint_relation": str(row["reason"]),
            "classification": verdict["classification"],
            "resolved": verdict["resolved"],
            "notes": verdict["notes"],
            "status": "resolved" if verdict["resolved"] else "unresolved",
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
        endpoint_cache = EndpointGridCache()
        factory = (
            (lambda: jeffreys_interval_grid(n, alpha))
            if method == "jeffreys"
            else (lambda: production_interval_grid(n, alpha, method))
        )
        cache[identity] = endpoint_cache.get_or_create(
            n,
            alpha,
            method,
            factory,
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
        "interval_kind": (
            "bayesian_comparator" if method == "jeffreys" else "frequentist"
        ),
        "p": np.nan,
        "x": x,
        "first_x": np.nan,
        "last_x": np.nan,
        "coverage": np.nan,
        "oracle": "high_precision_structural_oracle",
    }


def _structural_queue_row(
    *,
    checkpoint: str,
    reason: str,
    n: int,
    alpha: float,
    method: str,
    x: int | None = None,
    complement_x: int | None = None,
    x_left: int | None = None,
    x_right: int | None = None,
    endpoint_kind: str | None = None,
    alpha_wider: float | None = None,
    alpha_narrower: float | None = None,
) -> dict[str, object]:
    """Carry all paired predicate context into the HP queue."""

    return {
        "audit_kind": "structural",
        "reason": reason,
        "checkpoint": checkpoint,
        "n": n,
        "alpha": alpha,
        "method": method,
        "interval_kind": (
            "bayesian_comparator" if method == "jeffreys" else "frequentist"
        ),
        "p": np.nan,
        "x": np.nan if x is None else x,
        "first_x": np.nan,
        "last_x": np.nan,
        "coverage": np.nan,
        "oracle": "high_precision_structural_predicate",
        "complement_x": np.nan if complement_x is None else complement_x,
        "x_left": np.nan if x_left is None else x_left,
        "x_right": np.nan if x_right is None else x_right,
        "endpoint_kind": "" if endpoint_kind is None else endpoint_kind,
        "alpha_wider": np.nan if alpha_wider is None else alpha_wider,
        "alpha_narrower": np.nan if alpha_narrower is None else alpha_narrower,
    }


def _structural_queue_rows(
    checkpoint: str,
    prefix: str,
    results_dir: Path,
) -> list[dict[str, object]]:
    """Localize triggers while preserving each paired predicate's full context."""

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

        for x in np.flatnonzero(~np.isfinite(grid.lower) | ~np.isfinite(grid.upper)):
            invalid_kinds = [
                kind
                for kind, value in (("lower", grid.lower[x]), ("upper", grid.upper[x]))
                if not np.isfinite(value)
            ]
            rows.append(
                _structural_queue_row(
                    checkpoint=checkpoint,
                    reason="nan_or_inf",
                    n=n,
                    alpha=alpha,
                    method=method,
                    x=int(x),
                    endpoint_kind="+".join(invalid_kinds),
                )
            )

        symmetry_indices = np.flatnonzero(
            (np.abs(grid.lower - (1.0 - grid.upper[::-1])) > tolerance)
            | (np.abs(grid.upper - (1.0 - grid.lower[::-1])) > tolerance)
        )
        symmetry_pairs = sorted(
            {(min(int(x), n - int(x)), max(int(x), n - int(x))) for x in symmetry_indices}
        )
        for x, complement_x in symmetry_pairs:
            rows.append(
                _structural_queue_row(
                    checkpoint=checkpoint,
                    reason="complement_symmetry",
                    n=n,
                    alpha=alpha,
                    method=method,
                    x=x,
                    complement_x=complement_x,
                )
            )

        if method in {"wilson", "clopper_pearson"}:
            for x in np.flatnonzero(
                (grid.lower < -5e-15) | (grid.upper > 1.0 + 5e-15)
            ):
                kinds = []
                if grid.lower[x] < -5e-15:
                    kinds.append("lower")
                if grid.upper[x] > 1.0 + 5e-15:
                    kinds.append("upper")
                rows.append(
                    _structural_queue_row(
                        checkpoint=checkpoint,
                        reason="bounds",
                        n=n,
                        alpha=alpha,
                        method=method,
                        x=int(x),
                        endpoint_kind="+".join(kinds),
                    )
                )
            for endpoint_kind, pairs in (
                ("lower", np.flatnonzero(np.diff(grid.lower) < -5e-15)),
                ("upper", np.flatnonzero(np.diff(grid.upper) < -5e-15)),
            ):
                for x_left in pairs:
                    rows.append(
                        _structural_queue_row(
                            checkpoint=checkpoint,
                            reason="endpoint_monotonicity",
                            n=n,
                            alpha=alpha,
                            method=method,
                            x=int(x_left),
                            x_left=int(x_left),
                            x_right=int(x_left + 1),
                            endpoint_kind=endpoint_kind,
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
        wide = _grid_for(cache, n, alpha_wider, method)
        narrow = _grid_for(cache, n, alpha_narrower, method)
        lower_failure = wide.lower > narrow.lower + 5e-15
        upper_failure = wide.upper < narrow.upper - 5e-15
        for x in np.flatnonzero(lower_failure | upper_failure):
            kinds = []
            if lower_failure[x]:
                kinds.append("lower")
            if upper_failure[x]:
                kinds.append("upper")
            rows.append(
                _structural_queue_row(
                    checkpoint=checkpoint,
                    reason="nesting",
                    n=n,
                    alpha=alpha_wider,
                    method=method,
                    x=int(x),
                    endpoint_kind="+".join(kinds),
                    alpha_wider=alpha_wider,
                    alpha_narrower=alpha_narrower,
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
        "complement_x",
        "x_left",
        "x_right",
        "endpoint_kind",
        "alpha_wider",
        "alpha_narrower",
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
        "harness_schema_version": HARNESS_SCHEMA_VERSION,
        "source_shard_schema_version": SHARD_SCHEMA_VERSION,
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
