"""Backend-neutral paired method evaluation and replicate schema."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import math
import os
import time
from typing import Callable, Iterable, Sequence, TypeVar

import numpy as np
from scipy import stats

from pyMagicStat.inference.empirical_likelihood import (
    EmpiricalLikelihoodMeanCI,
    EmpiricalLikelihoodMeanResult,
    empirical_likelihood_mean_ci,
    empirical_likelihood_mean_test,
)

from experiments.adversarial_robustness_calibration import Scenario
from .seeds import SeedMaterial


REPLICATE_SCHEMA_VERSION = "el-vs-t-replicates-v1"
T_METHOD_VERSION = "two-sided-one-sample-student-t-scipy-v1"
EL_METHOD_VERSION = "uncorrected-owen-mean-el-production-engine-v1"
NUMERICAL_EXCEPTIONS = (FloatingPointError, OverflowError, RuntimeError, ValueError)


REPLICATE_COLUMNS = (
    "schema_version",
    "scenario_id",
    "family",
    "parameters_json",
    "n",
    "mu0",
    "shard_id",
    "num_shards",
    "replicate_id",
    "seed_identity",
    "generation_backend",
    "paired_sample_fingerprint",
    "sample_min",
    "sample_max",
    "mu0_hull_location",
    "sample_mean",
    "sample_variance",
    "sample_skewness",
    "sample_excess_kurtosis",
    "t_statistic",
    "t_p_value",
    "t_reject",
    "t_test_numerical_failure",
    "t_test_failure_reason",
    "t_ci_lower",
    "t_ci_upper",
    "t_ci_covers_mu0",
    "t_ci_width",
    "t_ci_numerical_failure",
    "el_statistic",
    "el_p_value",
    "el_reject",
    "el_lambda",
    "el_lambda_residual",
    "el_feasible",
    "el_regular",
    "el_boundary",
    "el_converged",
    "el_test_numerical_failure",
    "el_test_failure_reason",
    "el_ci_lower",
    "el_ci_upper",
    "el_ci_covers_mu0",
    "el_ci_width",
    "el_ci_feasible",
    "el_ci_regular",
    "el_ci_numerical_failure",
    "el_ci_failure_reason",
    "el_solver_failure",
)


@dataclass(frozen=True)
class TInference:
    statistic: float
    p_value: float
    reject: bool
    lower: float
    upper: float


T = TypeVar("T")
R = TypeVar("R")


class MethodExecutor:
    """One reusable process pool; BLAS limits are set before workers spawn."""

    def __init__(self, workers: int) -> None:
        if int(workers) < 1:
            raise ValueError("workers must be positive")
        self.workers = int(workers)
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> "MethodExecutor":
        if self.workers > 1:
            for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[variable] = "1"
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        return self

    def __exit__(self, *args: object) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)

    def map(self, function: Callable[[T], R], values: Iterable[T]) -> list[R]:
        if self._pool is None:
            return [function(value) for value in values]
        return list(self._pool.map(function, values, chunksize=1))


def _student_t_task(task: tuple[np.ndarray, float, float]) -> tuple[TInference | None, str]:
    sample, mu0, alpha = task
    try:
        n = int(sample.size)
        if n < 2:
            raise ValueError("Student t requires at least two observations")
        mean = float(np.mean(sample))
        variance = float(np.var(sample, ddof=1))
        standard_error = math.sqrt(variance / n)
        if not np.isfinite(standard_error) or standard_error <= 0.0:
            raise FloatingPointError("Student t standard error is not positive and finite")
        statistic = (mean - mu0) / standard_error
        p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
        critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        lower = mean - critical * standard_error
        upper = mean + critical * standard_error
        if not np.all(np.isfinite([statistic, p_value, lower, upper])):
            raise FloatingPointError("Student t produced non-finite output")
        return TInference(statistic, p_value, p_value < alpha, lower, upper), ""
    except NUMERICAL_EXCEPTIONS as error:
        return None, f"{type(error).__name__}: {error}"


def _el_test_task(
    task: tuple[np.ndarray, float],
) -> tuple[EmpiricalLikelihoodMeanResult | None, str]:
    sample, mu0 = task
    try:
        return empirical_likelihood_mean_test(sample, mu0), ""
    except NUMERICAL_EXCEPTIONS as error:
        return None, f"{type(error).__name__}: {error}"


def _el_ci_task(
    task: tuple[np.ndarray, float],
) -> tuple[EmpiricalLikelihoodMeanCI | None, str]:
    sample, confidence_level = task
    try:
        return empirical_likelihood_mean_ci(sample, confidence_level), ""
    except NUMERICAL_EXCEPTIONS as error:
        return None, f"{type(error).__name__}: {error}"


def evaluate_batch(
    samples: np.ndarray,
    diagnostics: np.ndarray,
    scenario: Scenario,
    replicate_ids: Sequence[int],
    seeds: Sequence[SeedMaterial],
    *,
    shard_id: int,
    num_shards: int,
    alpha: float,
    confidence_level: float,
    generation_backend: str,
    executor: MethodExecutor,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Apply both methods to each shared sample and return stage timings."""

    if samples.ndim != 2 or diagnostics.shape != (samples.shape[0], 4):
        raise ValueError("samples and diagnostics have incompatible batch shapes")
    if samples.shape[0] != len(replicate_ids) or len(replicate_ids) != len(seeds):
        raise ValueError("replicate metadata does not match batch length")
    if not 0.0 < alpha < 1.0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("alpha and confidence_level must be between zero and one")

    start = time.perf_counter()
    t_results = executor.map(
        _student_t_task,
        ((sample, scenario.population_mean, alpha) for sample in samples),
    )
    t_seconds = time.perf_counter() - start

    start = time.perf_counter()
    el_tests = executor.map(
        _el_test_task,
        ((sample, scenario.population_mean) for sample in samples),
    )
    el_test_seconds = time.perf_counter() - start

    start = time.perf_counter()
    el_intervals = executor.map(
        _el_ci_task,
        ((sample, confidence_level) for sample in samples),
    )
    el_ci_seconds = time.perf_counter() - start

    records: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        sample_min = float(np.min(sample))
        sample_max = float(np.max(sample))
        mu0 = float(scenario.population_mean)
        if mu0 < sample_min or mu0 > sample_max:
            hull_location = "outside"
        elif mu0 == sample_min or mu0 == sample_max:
            hull_location = "boundary"
        else:
            hull_location = "inside"

        t_result, t_error = t_results[index]
        el_test, el_test_error = el_tests[index]
        el_ci, el_ci_error = el_intervals[index]

        t_failed = t_result is None
        t_lower = math.nan if t_failed else t_result.lower
        t_upper = math.nan if t_failed else t_result.upper
        t_width = t_upper - t_lower if not t_failed else math.nan

        el_p_value = None if el_test is None else el_test.p_value
        el_test_solver_failure = bool(
            el_test is None or (el_test.feasible and el_test.regular and not el_test.converged)
        )
        el_test_available = el_p_value is not None and np.isfinite(el_p_value)
        el_lower = None if el_ci is None else el_ci.lower
        el_upper = None if el_ci is None else el_ci.upper
        el_ci_available = bool(
            el_ci is not None
            and el_ci.feasible
            and el_ci.regular
            and el_lower is not None
            and el_upper is not None
            and np.all(np.isfinite([el_lower, el_upper]))
        )
        el_ci_numerical_failure = not el_ci_available
        el_width = float(el_upper - el_lower) if el_ci_available else math.nan
        regular_continuous_sample = sample.size >= 2 and sample_min < sample_max
        el_solver_failure = bool(
            el_test_solver_failure or (el_ci_numerical_failure and regular_continuous_sample)
        )

        contiguous = np.ascontiguousarray(sample, dtype=np.float64)
        fingerprint = hashlib.blake2b(
            contiguous.view(np.uint8), digest_size=12, person=b"paired-sample"
        ).hexdigest()
        record = {
            "schema_version": REPLICATE_SCHEMA_VERSION,
            "scenario_id": scenario.name,
            "family": scenario.family,
            "parameters_json": scenario.parameters_json,
            "n": int(sample.size),
            "mu0": mu0,
            "shard_id": int(shard_id),
            "num_shards": int(num_shards),
            "replicate_id": int(replicate_ids[index]),
            "seed_identity": seeds[index].identity,
            "generation_backend": generation_backend,
            "paired_sample_fingerprint": fingerprint,
            "sample_min": sample_min,
            "sample_max": sample_max,
            "mu0_hull_location": hull_location,
            "sample_mean": float(diagnostics[index, 0]),
            "sample_variance": float(diagnostics[index, 1]),
            "sample_skewness": float(diagnostics[index, 2]),
            "sample_excess_kurtosis": float(diagnostics[index, 3]),
            "t_statistic": math.nan if t_failed else t_result.statistic,
            "t_p_value": math.nan if t_failed else t_result.p_value,
            "t_reject": math.nan if t_failed else int(t_result.reject),
            "t_test_numerical_failure": int(t_failed),
            "t_test_failure_reason": t_error,
            "t_ci_lower": t_lower,
            "t_ci_upper": t_upper,
            "t_ci_covers_mu0": math.nan if t_failed else int(t_lower <= mu0 <= t_upper),
            "t_ci_width": t_width,
            "t_ci_numerical_failure": int(t_failed),
            "el_statistic": math.nan if el_test is None else el_test.statistic,
            "el_p_value": math.nan if not el_test_available else float(el_p_value),
            "el_reject": math.nan if not el_test_available else int(float(el_p_value) < alpha),
            "el_lambda": math.nan if el_test is None or el_test.lambda_value is None else el_test.lambda_value,
            "el_lambda_residual": math.nan if el_test is None or el_test.lambda_residual is None else el_test.lambda_residual,
            "el_feasible": int(el_test.feasible) if el_test is not None else 0,
            "el_regular": int(el_test.regular) if el_test is not None else 0,
            "el_boundary": int(el_test.boundary) if el_test is not None else 0,
            "el_converged": int(el_test.converged) if el_test is not None else 0,
            "el_test_numerical_failure": int(el_test_solver_failure),
            "el_test_failure_reason": el_test_error or ("" if el_test is None else el_test.reason or ""),
            "el_ci_lower": float(el_lower) if el_ci_available else math.nan,
            "el_ci_upper": float(el_upper) if el_ci_available else math.nan,
            "el_ci_covers_mu0": int(float(el_lower) <= mu0 <= float(el_upper)) if el_ci_available else math.nan,
            "el_ci_width": el_width,
            "el_ci_feasible": int(el_ci.feasible) if el_ci is not None else 0,
            "el_ci_regular": int(el_ci.regular) if el_ci is not None else 0,
            "el_ci_numerical_failure": int(el_ci_numerical_failure),
            "el_ci_failure_reason": el_ci_error or ("" if el_ci is None else el_ci.reason or ""),
            "el_solver_failure": int(el_solver_failure),
        }
        records.append({column: record[column] for column in REPLICATE_COLUMNS})
    return records, {
        "student_t_seconds": t_seconds,
        "el_test_seconds": el_test_seconds,
        "el_ci_seconds": el_ci_seconds,
    }
