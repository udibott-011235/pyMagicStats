"""Frozen Monte Carlo decision rule for CP05."""

from __future__ import annotations

import math
from typing import Iterable


def monte_carlo_p_value(
    observed: float,
    replicates: Iterable[float],
    *,
    B: int,
) -> tuple[int, float]:
    values = tuple(float(value) for value in replicates)
    if len(values) != B:
        raise ValueError("exactly B eligible replicate statistics are required")
    if not math.isfinite(float(observed)) or any(not math.isfinite(value) for value in values):
        raise ValueError("Monte Carlo statistics must be finite")
    exceedances = sum(value >= observed for value in values)
    return exceedances, (exceedances + 1) / (B + 1)
