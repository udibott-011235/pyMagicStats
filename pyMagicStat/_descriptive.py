"""Shared sample-descriptive conventions used across pyMagicStat."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import scipy.stats as stats


def sample_descriptives(data: Any) -> Dict[str, float | int]:
    """Return canonical one-sample descriptives without copying the dataset.

    The calculations treat all observations as one sample. Variance and
    standard deviation use ``ddof=1``; skewness and excess kurtosis use the
    bias-corrected SciPy estimators.
    """

    array = np.asarray(data)
    flat = np.ravel(array)
    n, skewness, excess_kurtosis = sample_shape_statistics(flat)
    q1, q3 = np.percentile(flat, [25, 75])

    return {
        "n": n,
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat, ddof=1)) if n >= 2 else np.nan,
        "var": float(np.var(flat, ddof=1)) if n >= 2 else np.nan,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "range": float(np.max(flat) - np.min(flat)),
    }


def sample_shape_statistics(data: Any) -> tuple[int, float, float]:
    """Return the canonical size, skewness and excess-kurtosis triple."""

    flat = np.ravel(np.asarray(data))
    n = int(flat.size)
    return (
        n,
        float(stats.skew(flat, bias=False)) if n >= 3 else np.nan,
        (
            float(stats.kurtosis(flat, fisher=True, bias=False))
            if n >= 4
            else np.nan
        ),
    )
