"""Shared sample-descriptive conventions used across pyMagicStat."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import scipy.stats as stats


def sample_source(data: Any) -> Any:
    """Return a canonical ndarray stored by a data container, when present."""

    candidate = getattr(data, "data", None)
    return candidate if isinstance(candidate, np.ndarray) else data


def univariate_sample(data: Any, *, label: str = "sample") -> np.ndarray:
    """Return a non-empty one-dimensional sample or raise a stable error."""

    array = np.asarray(sample_source(data))
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{label} must contain at least one observation")
    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"{label} must contain numeric data")
    return array


def sample_descriptives(data: Any) -> Dict[str, float | int]:
    """Return canonical one-sample descriptives without copying the dataset.

    The calculations treat all observations as one sample. Variance and
    standard deviation use ``ddof=1``; skewness and excess kurtosis use the
    bias-corrected SciPy estimators.
    """

    array = univariate_sample(data)
    n, skewness, excess_kurtosis = sample_shape_statistics(array)
    q1, q3 = np.percentile(array, [25, 75])

    return {
        "n": n,
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if n >= 2 else np.nan,
        "var": float(np.var(array, ddof=1)) if n >= 2 else np.nan,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "range": float(np.max(array) - np.min(array)),
    }


def sample_shape_statistics(data: Any) -> tuple[int, float, float]:
    """Return the canonical size, skewness and excess-kurtosis triple."""

    array = univariate_sample(data)
    n = int(array.size)
    return (
        n,
        float(stats.skew(array, bias=False)) if n >= 3 else np.nan,
        (
            float(stats.kurtosis(array, fisher=True, bias=False))
            if n >= 4
            else np.nan
        ),
    )
