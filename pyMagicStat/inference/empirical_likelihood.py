"""Owen-style empirical likelihood for a one-sample arithmetic mean."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2

from pyMagicStat.inference.capabilities import InferenceGuarantee


EMPIRICAL_LIKELIHOOD_METHOD = "empirical_likelihood"
LAMBDA_RESIDUAL_TOLERANCE = 1e-10
CI_ENDPOINT_RESIDUAL_TOLERANCE = 1e-8
STATISTIC_ZERO_TOLERANCE = 1e-10
_ROOT_RTOL = 8.0 * np.finfo(float).eps
_ROOT_XTOL = 1e-14
_ROOT_MAXITER = 100


@dataclass(frozen=True)
class EmpiricalLikelihoodMeanResult:
    """Profile empirical-likelihood test result for one candidate mean.

    ``lambda_residual`` is the absolute mean of the dimensionless normalized
    estimating equation. Its scale therefore does not grow with sample size.
    """

    estimate: float
    null_value: float
    log_likelihood_ratio: float
    statistic: float
    df: int
    p_value: float | None
    lambda_value: float | None
    lambda_residual: float | None
    feasible: bool
    regular: bool
    converged: bool
    boundary: bool
    n: int
    guarantee: InferenceGuarantee
    method: str
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate,
            "null_value": self.null_value,
            "log_likelihood_ratio": self.log_likelihood_ratio,
            "statistic": self.statistic,
            "df": self.df,
            "p_value": self.p_value,
            "lambda_value": self.lambda_value,
            "lambda_residual": self.lambda_residual,
            "feasible": self.feasible,
            "regular": self.regular,
            "converged": self.converged,
            "boundary": self.boundary,
            "n": self.n,
            "guarantee": self.guarantee.value,
            "method": self.method,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EmpiricalLikelihoodMeanCI:
    """Profile empirical-likelihood confidence interval for the mean."""

    estimate: float
    lower: float | None
    upper: float | None
    confidence_level: float
    critical_value: float
    lower_statistic: float | None
    upper_statistic: float | None
    lower_endpoint_residual: float | None
    upper_endpoint_residual: float | None
    feasible: bool
    regular: bool
    n: int
    guarantee: InferenceGuarantee
    method: str
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "critical_value": self.critical_value,
            "lower_statistic": self.lower_statistic,
            "upper_statistic": self.upper_statistic,
            "lower_endpoint_residual": self.lower_endpoint_residual,
            "upper_endpoint_residual": self.upper_endpoint_residual,
            "feasible": self.feasible,
            "regular": self.regular,
            "n": self.n,
            "guarantee": self.guarantee.value,
            "method": self.method,
            "reason": self.reason,
        }


def empirical_likelihood_mean_test(
    data: Any,
    mu: float,
) -> EmpiricalLikelihoodMeanResult:
    """Evaluate the empirical likelihood ratio for a candidate population mean.

    The chi-square p-value is a first-order asymptotic calibration. Candidates
    outside the closed sample convex hull are returned as infeasible. Boundary
    and degenerate-support cases are represented explicitly without applying
    the regular chi-square approximation.
    """

    sample = _validate_sample(data)
    null_value = _validate_scalar(mu, "mu")
    return _evaluate_validated(sample, null_value)


def empirical_likelihood_mean_ci(
    data: Any,
    confidence_level: float = 0.95,
) -> EmpiricalLikelihoodMeanCI:
    """Construct a profile empirical-likelihood interval inside the sample hull."""

    sample = _validate_sample(data)
    confidence = _validate_confidence_level(confidence_level)
    estimate = _stable_mean(sample)
    n = int(sample.size)
    critical_value = float(chi2.ppf(confidence, df=1))
    sample_min = float(np.min(sample))
    sample_max = float(np.max(sample))

    if n < 2:
        return _failed_ci(
            estimate,
            confidence,
            critical_value,
            n,
            "At least two observations are required for regular empirical likelihood.",
        )
    if sample_min == sample_max:
        return _failed_ci(
            estimate,
            confidence,
            critical_value,
            n,
            "A constant sample has no regular profile empirical-likelihood interval.",
        )

    scale = float(np.max(np.abs(sample - estimate)))
    if not np.isfinite(scale) or scale <= 0.0:
        return _failed_ci(
            estimate,
            confidence,
            critical_value,
            n,
            "The centered sample cannot be scaled reliably for interval profiling.",
        )
    standardized = (sample - estimate) / scale
    lower_hull = float(np.min(standardized))
    upper_hull = float(np.max(standardized))

    def profile(candidate: float) -> float:
        result = _evaluate_validated(standardized, float(candidate))
        if not result.regular or not result.converged:
            raise FloatingPointError(
                "Profile evaluation did not produce a regular converged solution"
            )
        return result.statistic

    try:
        lower_edge = _finite_profile_edge(
            lower_hull,
            0.0,
            critical_value,
            profile,
        )
        upper_edge = _finite_profile_edge(
            upper_hull,
            0.0,
            critical_value,
            profile,
        )
        lower_root = float(
            brentq(
                lambda candidate: profile(candidate) - critical_value,
                lower_edge,
                0.0,
                xtol=_ROOT_XTOL,
                rtol=_ROOT_RTOL,
                maxiter=_ROOT_MAXITER,
            )
        )
        upper_root = float(
            brentq(
                lambda candidate: profile(candidate) - critical_value,
                0.0,
                upper_edge,
                xtol=_ROOT_XTOL,
                rtol=_ROOT_RTOL,
                maxiter=_ROOT_MAXITER,
            )
        )
        lower_statistic = float(profile(lower_root))
        upper_statistic = float(profile(upper_root))
    except (FloatingPointError, RuntimeError, ValueError) as error:
        return _failed_ci(
            estimate,
            confidence,
            critical_value,
            n,
            f"Profile interval construction failed: {error}",
        )

    lower_residual = abs(lower_statistic - critical_value)
    upper_residual = abs(upper_statistic - critical_value)
    if max(lower_residual, upper_residual) > CI_ENDPOINT_RESIDUAL_TOLERANCE:
        return _failed_ci(
            estimate,
            confidence,
            critical_value,
            n,
            "Profile roots did not meet the endpoint statistic tolerance.",
        )

    lower = float(np.clip(estimate + scale * lower_root, sample_min, estimate))
    upper = float(np.clip(estimate + scale * upper_root, estimate, sample_max))
    return EmpiricalLikelihoodMeanCI(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence,
        critical_value=critical_value,
        lower_statistic=lower_statistic,
        upper_statistic=upper_statistic,
        lower_endpoint_residual=lower_residual,
        upper_endpoint_residual=upper_residual,
        feasible=True,
        regular=True,
        n=n,
        guarantee=InferenceGuarantee.ASYMPTOTIC_MOMENT_BASED,
        method=EMPIRICAL_LIKELIHOOD_METHOD,
    )


def _evaluate_validated(
    sample: np.ndarray,
    null_value: float,
) -> EmpiricalLikelihoodMeanResult:
    n = int(sample.size)
    estimate = _stable_mean(sample)
    sample_min = float(np.min(sample))
    sample_max = float(np.max(sample))
    common = {
        "estimate": estimate,
        "null_value": null_value,
        "df": 1,
        "n": n,
        "guarantee": InferenceGuarantee.ASYMPTOTIC_MOMENT_BASED,
        "method": EMPIRICAL_LIKELIHOOD_METHOD,
    }

    if null_value < sample_min or null_value > sample_max:
        return EmpiricalLikelihoodMeanResult(
            log_likelihood_ratio=-np.inf,
            statistic=np.inf,
            p_value=None,
            lambda_value=None,
            lambda_residual=None,
            feasible=False,
            regular=False,
            converged=False,
            boundary=False,
            reason="The candidate mean lies outside the sample convex hull.",
            **common,
        )

    if n == 1:
        return EmpiricalLikelihoodMeanResult(
            log_likelihood_ratio=0.0,
            statistic=0.0,
            p_value=None,
            lambda_value=0.0,
            lambda_residual=0.0,
            feasible=True,
            regular=False,
            converged=True,
            boundary=True,
            reason=(
                "A single observation has degenerate support and cannot provide "
                "regular empirical-likelihood inference."
            ),
            **common,
        )

    constant = sample_min == sample_max
    if constant:
        return EmpiricalLikelihoodMeanResult(
            log_likelihood_ratio=0.0,
            statistic=0.0,
            p_value=None,
            lambda_value=0.0,
            lambda_residual=0.0,
            feasible=True,
            regular=False,
            converged=True,
            boundary=True,
            reason=(
                "The candidate equals the constant empirical support, but zero "
                "variance invalidates regular chi-square calibration."
            ),
            **common,
        )

    if null_value == sample_min or null_value == sample_max:
        lower_boundary = null_value == sample_min
        return EmpiricalLikelihoodMeanResult(
            log_likelihood_ratio=-np.inf,
            statistic=np.inf,
            p_value=None,
            lambda_value=np.inf if lower_boundary else -np.inf,
            lambda_residual=None,
            feasible=True,
            regular=False,
            converged=False,
            boundary=True,
            reason=(
                "The candidate is on the sample convex-hull boundary; empirical "
                "weights collapse and the finite-lambda regular solution does not exist."
            ),
            **common,
        )

    if null_value == estimate:
        return EmpiricalLikelihoodMeanResult(
            log_likelihood_ratio=0.0,
            statistic=0.0,
            p_value=1.0,
            lambda_value=0.0,
            lambda_residual=0.0,
            feasible=True,
            regular=n >= 2,
            converged=True,
            boundary=False,
            **common,
        )

    centered = sample - null_value
    scale = float(np.max(np.abs(centered)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise FloatingPointError("Candidate-centered observations cannot be scaled")
    normalized = centered / scale
    tau, residual = _solve_normalized_lambda(normalized)
    denominators = 1.0 + tau * normalized
    if np.any(denominators <= 0.0) or not np.all(np.isfinite(denominators)):
        raise FloatingPointError("Lambda solution left the admissible domain")

    statistic = float(
        2.0
        * np.sum(
            np.log1p(tau * normalized),
            dtype=np.longdouble,
        )
    )
    if statistic < 0.0:
        if statistic >= -STATISTIC_ZERO_TOLERANCE:
            statistic = 0.0
        else:
            raise FloatingPointError(
                "Empirical likelihood statistic is materially negative"
            )
    lambda_value = float(tau / scale)
    converged = bool(
        residual <= LAMBDA_RESIDUAL_TOLERANCE
        and np.isfinite(lambda_value)
    )
    return EmpiricalLikelihoodMeanResult(
        log_likelihood_ratio=-0.5 * statistic,
        statistic=statistic,
        p_value=float(chi2.sf(statistic, df=1)) if converged else None,
        lambda_value=lambda_value,
        lambda_residual=residual,
        feasible=True,
        regular=True,
        converged=converged,
        boundary=False,
        reason=(
            None
            if converged
            else "Lambda solution did not meet the declared numerical tolerance."
        ),
        **common,
    )


def _solve_normalized_lambda(normalized: np.ndarray) -> tuple[float, float]:
    """Solve the dual equation and return its sample-size-stable mean residual."""

    positive = normalized[normalized > 0.0]
    negative = normalized[normalized < 0.0]
    if positive.size == 0 or negative.size == 0:
        raise FloatingPointError("An interior mean requires support on both sides")

    lower_boundary = float(np.max(-1.0 / positive))
    upper_boundary = float(np.min(-1.0 / negative))

    def equation(tau: float) -> float:
        denominators = 1.0 + tau * normalized
        if np.any(denominators <= 0.0):
            return np.inf if tau < 0.0 else -np.inf
        value = np.mean(
            normalized / denominators,
            dtype=np.longdouble,
        )
        return float(value)

    at_zero = equation(0.0)
    if at_zero == 0.0:
        return 0.0, 0.0
    if at_zero > 0.0:
        endpoint = _finite_lambda_endpoint(
            upper_boundary,
            equation,
            expected_sign=-1,
        )
        root = brentq(
            equation,
            0.0,
            endpoint,
            xtol=_ROOT_XTOL,
            rtol=_ROOT_RTOL,
            maxiter=_ROOT_MAXITER,
        )
    else:
        endpoint = _finite_lambda_endpoint(
            lower_boundary,
            equation,
            expected_sign=1,
        )
        root = brentq(
            equation,
            endpoint,
            0.0,
            xtol=_ROOT_XTOL,
            rtol=_ROOT_RTOL,
            maxiter=_ROOT_MAXITER,
        )
    root = float(root)
    return root, abs(equation(root))


def _finite_lambda_endpoint(
    boundary: float,
    equation: Any,
    *,
    expected_sign: int,
) -> float:
    if not np.isfinite(boundary):
        raise FloatingPointError("The admissible lambda boundary is not finite")
    candidates = [float(np.nextafter(boundary, 0.0))]
    candidates.extend(
        float(boundary * (1.0 - margin))
        for margin in np.geomspace(1e-15, 1e-3, 13)
    )
    for candidate in candidates:
        value = equation(candidate)
        if np.isfinite(value) and np.sign(value) == expected_sign:
            return candidate
    raise FloatingPointError("Could not construct a finite lambda root bracket")


def _finite_profile_edge(
    boundary: float,
    center: float,
    critical_value: float,
    profile: Any,
) -> float:
    candidates = [float(np.nextafter(boundary, center))]
    candidates.extend(
        float(boundary + margin * (center - boundary))
        for margin in np.geomspace(1e-15, 1e-2, 14)
    )
    for candidate in candidates:
        try:
            statistic = profile(candidate)
        except (FloatingPointError, RuntimeError, ValueError):
            continue
        if np.isfinite(statistic) and statistic > critical_value:
            return candidate
    raise FloatingPointError("Could not construct a finite profile root bracket")


def _failed_ci(
    estimate: float,
    confidence_level: float,
    critical_value: float,
    n: int,
    reason: str,
) -> EmpiricalLikelihoodMeanCI:
    return EmpiricalLikelihoodMeanCI(
        estimate=estimate,
        lower=None,
        upper=None,
        confidence_level=confidence_level,
        critical_value=critical_value,
        lower_statistic=None,
        upper_statistic=None,
        lower_endpoint_residual=None,
        upper_endpoint_residual=None,
        feasible=False,
        regular=False,
        n=n,
        guarantee=InferenceGuarantee.ASYMPTOTIC_MOMENT_BASED,
        method=EMPIRICAL_LIKELIHOOD_METHOD,
        reason=reason,
    )


def _validate_sample(data: Any) -> np.ndarray:
    try:
        sample = np.array(data, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("data must be numeric") from error
    if sample.ndim != 1:
        raise ValueError("data must be a one-dimensional sample")
    if sample.size == 0:
        raise ValueError("data must contain at least one observation")
    if not np.all(np.isfinite(sample)):
        raise ValueError("data must contain only finite observations")
    return sample


def _validate_scalar(value: Any, name: str) -> float:
    candidate = np.asarray(value)
    if candidate.ndim != 0:
        raise ValueError(f"{name} must be a finite scalar")
    try:
        number = float(candidate)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite scalar")
    return number


def _validate_confidence_level(value: Any) -> float:
    confidence = _validate_scalar(value, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    return confidence


def _stable_mean(sample: np.ndarray) -> float:
    scale = float(np.max(np.abs(sample)))
    if scale == 0.0:
        return 0.0
    return float(scale * np.mean(sample / scale))
