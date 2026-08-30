"""Shared Pearson goodness-of-fit mechanics for discrete distributions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import scipy.stats as stats


MINIMUM_EXPECTED = 5.0


def point_cell(value: int, observed: int, expected: float) -> Dict[str, Any]:
    """Build one exhaustive, ordered support cell."""

    return {
        "lower": int(value),
        "upper": int(value),
        "label": str(int(value)),
        "observed": int(observed),
        "expected": float(expected),
    }


def upper_tail_cell(lower: int, observed: int, expected: float) -> Dict[str, Any]:
    """Build an explicit unbounded upper-tail cell."""

    return {
        "lower": int(lower),
        "upper": None,
        "label": f"{int(lower)}+",
        "observed": int(observed),
        "expected": float(expected),
    }


def unavailable_result(
    *,
    status: str,
    hypothesis: str,
    alpha: float,
    parameter_count_estimated: int,
    parameters: Mapping[str, Mapping[str, Any]],
    observed_total: int,
    reason: str,
    legacy_values: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the complete GOF contract when a p-value cannot be produced."""

    if status not in {"not_assessed", "error"}:
        raise ValueError("Unavailable GOF status must be 'not_assessed' or 'error'")
    result: Dict[str, Any] = {
        "status": status,
        "decision": None,
        "hypothesis": hypothesis,
        "alpha": float(alpha),
        "statistic": None,
        "chi2": None,
        "p_value": None,
        "df": None,
        "parameter_count_estimated": int(parameter_count_estimated),
        "parameters": {
            name: {key: _plain_value(item) for key, item in value.items()}
            for name, value in parameters.items()
        },
        "original_cells": [],
        "pooled_cells": [],
        "observed_total": int(observed_total),
        "expected_total": None,
        "minimum_expected": None,
        "reason": str(reason),
    }
    if legacy_values:
        result.update({key: _plain_value(value) for key, value in legacy_values.items()})
    return result


def pearson_gof_result(
    *,
    cells: Iterable[Mapping[str, Any]],
    hypothesis: str,
    alpha: float,
    parameter_count_estimated: int,
    parameters: Mapping[str, Mapping[str, Any]],
    legacy_values: Optional[Mapping[str, Any]] = None,
    minimum_expected: float = MINIMUM_EXPECTED,
) -> Dict[str, Any]:
    """Pool exhaustive tail cells and evaluate Pearson's chi-square statistic."""

    original_cells = [_copy_cell(cell) for cell in cells]
    result: Dict[str, Any] = {
        "status": "not_assessed",
        "decision": None,
        "hypothesis": hypothesis,
        "alpha": float(alpha),
        "statistic": None,
        "chi2": None,
        "p_value": None,
        "df": None,
        "parameter_count_estimated": int(parameter_count_estimated),
        "parameters": {
            name: {key: _plain_value(item) for key, item in value.items()}
            for name, value in parameters.items()
        },
        "original_cells": original_cells,
        "pooled_cells": [],
        "observed_total": int(sum(cell["observed"] for cell in original_cells)),
        "expected_total": float(sum(cell["expected"] for cell in original_cells)),
        "minimum_expected": None,
    }
    if legacy_values:
        result.update({key: _plain_value(value) for key, value in legacy_values.items()})

    invalid_reason = _validate_cells(original_cells)
    if invalid_reason is not None:
        result.update(status="error", reason=invalid_reason)
        return result

    pooled_cells = pool_adjacent_tails(
        original_cells,
        minimum_expected=float(minimum_expected),
    )
    result["pooled_cells"] = pooled_cells
    result["minimum_expected"] = (
        float(min(cell["expected"] for cell in pooled_cells))
        if pooled_cells
        else None
    )
    result["df"] = int(len(pooled_cells) - 1 - parameter_count_estimated)

    observed_total = float(result["observed_total"])
    expected_total = float(result["expected_total"])
    if not np.isclose(observed_total, expected_total, rtol=1e-12, atol=1e-9):
        result.update(
            status="error",
            reason=(
                "Observed and expected totals differ; the exhaustive-support "
                "invariant was not satisfied."
            ),
        )
        return result
    if not pooled_cells or result["minimum_expected"] < minimum_expected:
        result["reason"] = (
            "Adjacent tail pooling could not produce cells with expected "
            f"frequency at least {float(minimum_expected):g}."
        )
        return result
    if result["df"] <= 0:
        result["reason"] = (
            "Pearson chi-square goodness-of-fit requires positive degrees of "
            "freedom after accounting for estimated parameters."
        )
        return result

    observed = np.asarray([cell["observed"] for cell in pooled_cells], dtype=float)
    expected = np.asarray([cell["expected"] for cell in pooled_cells], dtype=float)
    statistic = float(np.sum(np.square(observed - expected) / expected))
    p_value = float(stats.chi2.sf(statistic, result["df"]))
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        result.update(
            status="error",
            reason="Pearson chi-square evaluation produced a non-finite result.",
        )
        return result

    result.update(
        status="ok",
        decision="reject" if p_value <= alpha else "fail_to_reject",
        statistic=statistic,
        chi2=statistic,
        p_value=p_value,
    )
    return result


def pool_adjacent_tails(
    cells: Iterable[Mapping[str, Any]],
    *,
    minimum_expected: float = MINIMUM_EXPECTED,
) -> List[Dict[str, Any]]:
    """Pool contiguous cells inward from both tails without dropping mass."""

    ordered = [_copy_cell(cell) for cell in cells]
    if not ordered:
        return []

    left_index = 0
    left_group: Optional[Dict[str, Any]] = None
    if ordered[0]["expected"] < minimum_expected:
        while left_index < len(ordered) and (
            left_group is None or left_group["expected"] < minimum_expected
        ):
            left_group = (
                ordered[left_index]
                if left_group is None
                else _merge_cells(left_group, ordered[left_index])
            )
            left_index += 1

    right_index = len(ordered) - 1
    right_group: Optional[Dict[str, Any]] = None
    if right_index >= left_index and ordered[right_index]["expected"] < minimum_expected:
        while right_index >= left_index and (
            right_group is None or right_group["expected"] < minimum_expected
        ):
            right_group = (
                ordered[right_index]
                if right_group is None
                else _merge_cells(ordered[right_index], right_group)
            )
            right_index -= 1

    pooled: List[Dict[str, Any]] = []
    if left_group is not None:
        pooled.append(left_group)
    pooled.extend(ordered[left_index : right_index + 1])
    if right_group is not None:
        pooled.append(right_group)
    if len(pooled) > 1 and pooled[0]["expected"] < minimum_expected:
        pooled[0:2] = [_merge_cells(pooled[0], pooled[1])]
    if len(pooled) > 1 and pooled[-1]["expected"] < minimum_expected:
        pooled[-2:] = [_merge_cells(pooled[-2], pooled[-1])]
    return pooled


def _copy_cell(cell: Mapping[str, Any]) -> Dict[str, Any]:
    upper = cell.get("upper")
    return {
        "lower": int(cell["lower"]),
        "upper": None if upper is None else int(upper),
        "label": str(cell["label"]),
        "observed": int(cell["observed"]),
        "expected": float(cell["expected"]),
    }


def _merge_cells(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    lower = int(left["lower"])
    upper = right.get("upper")
    upper = None if upper is None else int(upper)
    label = f"{lower}+" if upper is None else (str(lower) if lower == upper else f"{lower}-{upper}")
    return {
        "lower": lower,
        "upper": upper,
        "label": label,
        "observed": int(left["observed"] + right["observed"]),
        "expected": float(left["expected"] + right["expected"]),
    }


def _validate_cells(cells: List[Mapping[str, Any]]) -> Optional[str]:
    if not cells:
        return "The discrete support contains no cells."
    expected = np.asarray([cell["expected"] for cell in cells], dtype=float)
    observed = np.asarray([cell["observed"] for cell in cells], dtype=float)
    if not np.all(np.isfinite(expected)) or np.any(expected < 0.0):
        return "Expected frequencies must be finite and non-negative."
    if not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
        return "Observed frequencies must be finite and non-negative."
    for left, right in zip(cells, cells[1:]):
        if left["upper"] is None or int(right["lower"]) != int(left["upper"]) + 1:
            return "Discrete support cells must be exhaustive, ordered, and contiguous."
    return None


def _plain_value(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
