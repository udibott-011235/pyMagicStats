"""Frozen CP05 candidate statistics and numerical certification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from pyMagicStat.distributions._discrete_gof import (
    pearson_gof_result,
    point_cell,
    upper_tail_cell,
)

from .oracles.negative_binomial_high_precision import high_precision_nb_statistic


FLOAT_ORACLE_RTOL = 1e-11
TAIL_ABSOLUTE_TOLERANCE = 1e-14
TAIL_RELATIVE_TOLERANCE = 1e-12
MAX_SUPPORT_TERMS = 1_000_000


class StatisticError(RuntimeError):
    pass


class TailCertificationError(StatisticError):
    pass


class OracleMismatchError(StatisticError):
    pass


class ComparatorNotAssessable(StatisticError):
    pass


@dataclass(frozen=True, slots=True)
class StatisticEvaluation:
    value: float
    remainder_bound: float | None = None
    oracle_value: float | None = None
    absolute_error: float | None = None
    relative_scaled_error: float | None = None


def _continuous_sample(sample) -> np.ndarray:
    values = np.asarray(sample, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise StatisticError("statistic requires a nonempty finite one-dimensional sample")
    return np.sort(values)


def continuous_statistic(sample, bound, statistic: str) -> StatisticEvaluation:
    values = _continuous_sample(sample)
    n = values.size
    if statistic == "CVM":
        cdf = np.asarray(bound.cdf(values), dtype=np.float64)
        if not np.all(np.isfinite(cdf)):
            raise StatisticError("continuous CDF evaluation was nonfinite")
        ranks = (2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n)
        value = 1.0 / (12.0 * n) + math.fsum(float(item) for item in (cdf - ranks) ** 2)
    elif statistic == "AD":
        logcdf = np.asarray(bound.logcdf(values), dtype=np.float64)
        logsf_reverse = np.asarray(bound.logsf(values[::-1]), dtype=np.float64)
        if not np.all(np.isfinite(logcdf)) or not np.all(np.isfinite(logsf_reverse)):
            raise StatisticError("continuous AD log probability was nonfinite")
        weights = 2.0 * np.arange(1, n + 1) - 1.0
        value = -float(n) - math.fsum(
            float(weight * (left + right))
            for weight, left, right in zip(weights, logcdf, logsf_reverse)
        ) / n
    elif statistic == "KS":
        cdf = np.asarray(bound.cdf(values), dtype=np.float64)
        if not np.all(np.isfinite(cdf)):
            raise StatisticError("continuous CDF evaluation was nonfinite")
        lower = np.arange(n, dtype=np.float64) / n
        upper = np.arange(1, n + 1, dtype=np.float64) / n
        value = max(float(np.max(cdf - lower)), float(np.max(upper - cdf)))
    else:
        raise ValueError("unknown continuous statistic")
    if not math.isfinite(value) or value < 0.0:
        raise StatisticError("continuous statistic was invalid")
    return StatisticEvaluation(value=float(value))


def _discrete_sample(sample) -> np.ndarray:
    original = np.asarray(sample)
    if original.ndim != 1 or original.size == 0:
        raise StatisticError("statistic requires a nonempty one-dimensional sample")
    try:
        values = original.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StatisticError("discrete sample is not representable as int64") from exc
    if np.any(values < 0) or np.any(original != values):
        raise StatisticError("discrete sample must contain non-negative integers")
    return np.sort(values)


def stable_ad_upper_tail_term(bound, n: int, j: int) -> float:
    """Complete AD contribution for j >= max(sample), including 1/n."""

    logsf = float(bound.logsf(j))
    logpmf = float(bound.logpmf(j))
    logcdf = float(bound.logcdf(j))
    if not all(math.isfinite(value) for value in (logsf, logpmf, logcdf)):
        raise TailCertificationError("stable AD tail log probability was nonfinite")
    log_term = math.log(n) + logsf + logpmf - logcdf
    try:
        term = math.exp(log_term)
    except OverflowError as exc:
        raise StatisticError("stable AD upper-tail term overflowed") from exc
    if not math.isfinite(term) or term < 0:
        raise StatisticError("stable AD upper-tail term was invalid")
    return term


def _positive_float_bound(log_bound: float) -> float:
    if not math.isfinite(log_bound):
        raise TailCertificationError("tail remainder log bound was nonfinite")
    if log_bound < math.log(np.nextafter(0.0, 1.0)):
        return float(np.nextafter(0.0, 1.0))
    return math.exp(log_bound)


def _nb_support_sum(sample, bound, statistic: str) -> tuple[float, float]:
    values = _discrete_sample(sample)
    n = int(values.size)
    maximum = int(values[-1])
    partial_terms: list[float] = []
    for j in range(MAX_SUPPORT_TERMS):
        logpmf = float(bound.logpmf(j))
        logcdf = float(bound.logcdf(j))
        logsf = float(bound.logsf(j))
        if not all(math.isfinite(value) for value in (logpmf, logcdf, logsf)):
            raise TailCertificationError("NB support probability could not be certified")
        if j >= maximum:
            if statistic == "AD":
                term = stable_ad_upper_tail_term(bound, n, j)
                log_remainder = math.log(n) + 2.0 * logsf - logcdf
            else:
                term = math.exp(math.log(n) + 2.0 * logsf + logpmf)
                log_remainder = math.log(n) + 3.0 * logsf
        else:
            empirical_count = int(np.searchsorted(values, j, side="right"))
            cdf = math.exp(logcdf)
            z = empirical_count - n * cdf
            if z == 0.0:
                term = 0.0
            else:
                log_numerator = 2.0 * math.log(abs(z)) - math.log(n) + logpmf
                if statistic == "AD":
                    log_numerator -= logcdf + logsf
                term = math.exp(log_numerator)
            log_remainder = math.inf
        if not math.isfinite(term) or term < 0:
            raise StatisticError("NB support summand was invalid")
        partial_terms.append(term)
        partial = math.fsum(partial_terms)
        if j >= maximum:
            remainder = _positive_float_bound(log_remainder)
            required = max(TAIL_ABSOLUTE_TOLERANCE, TAIL_RELATIVE_TOLERANCE * abs(partial))
            if remainder <= required:
                return partial, remainder
    raise TailCertificationError("NB tail certification term budget exhausted")


def discrete_ks(sample, bound) -> StatisticEvaluation:
    values = _discrete_sample(sample)
    n = values.size
    maximum = int(values[-1])
    if maximum >= MAX_SUPPORT_TERMS:
        raise StatisticError("discrete KS support budget exhausted")
    support = np.arange(maximum + 1, dtype=np.int64)
    cdf = np.asarray(bound.cdf(support), dtype=np.float64)
    empirical = np.searchsorted(values, support, side="right") / n
    if not np.all(np.isfinite(cdf)):
        raise StatisticError("discrete KS CDF was nonfinite")
    return StatisticEvaluation(value=float(np.max(np.abs(empirical - cdf))))


def discrete_pearson(sample, bound, *, parameter_count_estimated: int) -> StatisticEvaluation:
    values = _discrete_sample(sample)
    n = values.size
    maximum = int(values[-1])
    if maximum >= MAX_SUPPORT_TERMS:
        raise ComparatorNotAssessable("Pearson support budget exhausted")
    counts = np.bincount(values, minlength=maximum + 1)
    cells = [
        point_cell(j, int(counts[j]), n * float(bound.pmf(j)))
        for j in range(maximum + 1)
    ]
    cells.append(upper_tail_cell(maximum + 1, 0, n * float(bound.sf(maximum))))
    result = pearson_gof_result(
        cells=cells,
        hypothesis="CP05 historical Pearson comparator",
        alpha=0.05,
        parameter_count_estimated=parameter_count_estimated,
        parameters={},
    )
    if result["status"] != "ok":
        raise ComparatorNotAssessable(str(result.get("reason", "Pearson unavailable")))
    return StatisticEvaluation(value=float(result["statistic"]))


def negative_binomial_statistic(
    sample,
    bound,
    statistic: str,
    *,
    parameter_count_estimated: int,
) -> StatisticEvaluation:
    if statistic == "KS":
        return discrete_ks(sample, bound)
    if statistic == "PEARSON":
        return discrete_pearson(
            sample, bound, parameter_count_estimated=parameter_count_estimated
        )
    if statistic not in {"AD", "CVM"}:
        raise ValueError("unknown negative binomial statistic")
    value, remainder = _nb_support_sum(sample, bound, statistic)
    parameters = bound.parameters
    try:
        oracle, _ = high_precision_nb_statistic(
            sample, r=parameters.r, p=parameters.p, statistic=statistic
        )
    except (ArithmeticError, ValueError) as exc:
        raise OracleMismatchError("independent NB oracle failed") from exc
    absolute_error = abs(value - oracle)
    scaled_error = absolute_error / max(1.0, abs(oracle))
    if absolute_error > FLOAT_ORACLE_RTOL * max(1.0, abs(oracle)):
        raise OracleMismatchError("float64 NB statistic disagrees with independent oracle")
    return StatisticEvaluation(
        value=value,
        remainder_bound=remainder,
        oracle_value=oracle,
        absolute_error=absolute_error,
        relative_scaled_error=scaled_error,
    )


def evaluate_statistic(
    sample,
    bound,
    family_id: str,
    statistic: str,
    *,
    parameter_count_estimated: int,
) -> StatisticEvaluation:
    if family_id == "negative_binomial":
        return negative_binomial_statistic(
            sample,
            bound,
            statistic,
            parameter_count_estimated=parameter_count_estimated,
        )
    return continuous_statistic(sample, bound, statistic)
