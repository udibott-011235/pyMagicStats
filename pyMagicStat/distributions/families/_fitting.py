"""Private fixed-location MLEs; validation precedes the numerical boundary.

Negative Binomial uses Brent's bounded scalar solver in log(r). Decimal
evaluation of the digamma recurrence/asymptotic expansion avoids subtracting
nearly equal float64 values at large r. Work per distinct count is bounded,
independently of the count's magnitude; counts are never expanded into trials.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, DecimalException, localcontext
import math
from numbers import Integral, Real
import re
import warnings

import numpy as np
from scipy import optimize, stats

from ._core import (
    FitIdentifiabilityError, FitNumericalError, FitResult,
    FittedContinuousDistribution, FittedDiscreteDistribution, NoFiniteMLEError,
)


def _fit_input(data, *, discrete=False):
    """Inspect original elements before any numeric coercion or narrowing."""
    try:
        original = np.asarray(data, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("data must be a one-dimensional real sample") from exc
    if original.ndim != 1 or original.size == 0:
        raise ValueError("data must be nonempty and one-dimensional")
    values = []
    for value in original:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("data must contain only real numeric observations")
        if discrete and isinstance(value, Integral):
            canonical = int(value)
        else:
            try:
                canonical = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError("data must be safely representable") from exc
            if not math.isfinite(canonical):
                raise ValueError("data must contain only finite observations")
            if discrete:
                integer = int(value)
                if value != integer:
                    raise ValueError("data must contain integer-valued observations")
                canonical = integer
            elif (int(value) != canonical if isinstance(value, Integral) else value != canonical):
                raise ValueError("data must be safely representable as float64")
        if canonical < 0:
            raise ValueError("data is outside the nonnegative support")
        if discrete and canonical > 2**63 - 1:
            raise ValueError("data is outside int64 range")
        values.append(canonical)
    return np.asarray(values, dtype=np.int64 if discrete else np.float64)


def _warning_strings(records):
    normalized = []
    for record in records:
        message = str(record.message)
        message = re.sub(r"0x[0-9a-fA-F]+", "<address>", message)
        # Match absolute paths at a token boundary, not ratios such as x/y.
        # File suffixes delimit unquoted paths containing spaces. Consume
        # attached source coordinates as part of the path, before token-only
        # fallback handles extensionless paths.
        start = r"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\|/)"
        message = re.sub(
            start + r"[^\r\n'\"<>]*?\.[A-Za-z0-9_]+(?=:\d|\s|$|['\"<>])(?::\d+(?::\d+)?)?",
            "<path>", message,
        )
        message = re.sub(start + r"[^\s'\"<>]+", "<path>", message)
        message = re.sub(r"\bline\s+\d+(?::\d+)?", "line <number>", message, flags=re.IGNORECASE)
        normalized.append(f"{record.category.__name__}: {' '.join(message.split())}")
    return tuple(normalized)


def _result(bound, n, log_likelihood, estimated, estimator, solver, records):
    k = len(estimated)
    ll = float(log_likelihood)
    aic = float(2*k - 2*ll)
    bic = float(k*math.log(n) - 2*ll)
    if not all(math.isfinite(x) for x in (ll, aic, bic)):
        raise FitNumericalError("nonfinite fitted likelihood or information criteria")
    wrapper = FittedDiscreteDistribution if bound.kind.value == "discrete" else FittedContinuousDistribution
    return FitResult(
        fitted_distribution=wrapper(bound), estimation_method="maximum_likelihood",
        fixed_parameters=("loc",), estimated_parameters=estimated,
        n_observations=int(n), log_likelihood=ll, aic=aic, bic=bic,
        converged=True, warnings=_warning_strings(records),
        metadata={"estimator_id": estimator, "solver_id": solver},
    )


def _mean(data):
    # Scaling before summation avoids overflow without losing tiny homogeneous
    # samples to division by n before summation.
    maximum = float(np.max(data))
    return maximum * (math.fsum(float(x)/maximum for x in data)/len(data))


def fit_exponential(family, data):
    sample = _fit_input(data)
    if not np.any(sample > 0):
        raise NoFiniteMLEError("all-zero exponential sample has no finite MLE")
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        try:
            scale = _mean(sample)
            if not math.isfinite(scale) or scale <= 0:
                raise FitNumericalError("exponential scale is not representable")
            bound = family.bind(scale=scale)
            ll = -len(sample)*math.log(scale) - math.fsum(float(x)/scale for x in sample)
            return _result(bound, len(sample), ll, ("scale",),
                           "pymagicstats-exponential-closed-form-mle-v1", "closed_form", records)
        except (ValueError, OverflowError, FloatingPointError) as exc:
            raise FitNumericalError("exponential numerical fitting failure") from exc


def fit_gamma(family, data):
    sample = _fit_input(data)
    if np.any(sample == 0) or np.all(sample == sample[0]):
        raise NoFiniteMLEError("zero or constant gamma sample has no finite MLE")
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        try:
            result = stats.gamma.fit(sample, floc=0, method="MLE")
            if not isinstance(result, tuple) or len(result) != 3:
                raise FitNumericalError("malformed gamma backend fit")
            if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in result):
                raise FitNumericalError("nonnumeric gamma backend fit")
            shape, loc, scale = map(float, result)
            if not all(math.isfinite(x) for x in (shape, loc, scale)) or shape <= 0 or scale <= 0 or loc != 0:
                raise FitNumericalError("invalid gamma backend fit")
            bound = family.bind(shape=shape, scale=scale)
            ll = math.fsum(float(x) for x in bound.logpdf(sample))
            return _result(bound, len(sample), ll, ("shape", "scale"),
                           "scipy-gamma-fixed-loc-mle-v1", "scipy.stats.gamma.fit", records)
        except (ValueError, OverflowError, FloatingPointError) as exc:
            raise FitNumericalError("gamma numerical fitting failure") from exc


# Exact Bernoulli numbers B_2 ... B_32. At z >= 64 the first omitted
# digamma term is below 2e-44; recurrence takes at most 64 steps. Decimal
# precision also grows with r to retain the O(r^-2) profile cancellation.
_BERNOULLI = (
    (1, 6), (-1, 30), (1, 42), (-1, 30), (5, 66), (-691, 2730),
    (7, 6), (-3617, 510), (43867, 798), (-174611, 330), (854513, 138),
    (-236364091, 2730), (8553103, 6), (-23749461029, 870),
    (8615841276005, 14322), (-7709321041217, 510),
)
_LOG_SQRT_2PI = Decimal(
    "0.9189385332046727417803297364056176398613974736377834128171515404827656959272603976947432986359541976"
)
_BRACKET_STEPS = 1024
_SOLVER_ITERATIONS = 128
_NB_SOLVER = "scipy.optimize.brentq"


def _decimal_psi(z):
    correction = Decimal(0)
    for _ in range(64):
        if z >= 64:
            break
        correction -= 1/z
        z += 1
    value = z.ln() - 1/(2*z)
    for j, (a, b) in enumerate(_BERNOULLI, 1):
        value -= Decimal(a)/b/(2*j)/z**(2*j)
    return value + correction


def _decimal_loggamma(z):
    correction = Decimal(0)
    for _ in range(64):
        if z >= 64:
            break
        correction -= z.ln()
        z += 1
    value = (z-Decimal('0.5'))*z.ln()-z+_LOG_SQRT_2PI
    for j, (a, b) in enumerate(_BERNOULLI, 1):
        value += Decimal(a)/b/(2*j*(2*j-1))/z**(2*j-1)
    return value + correction


def _profile_score(log_r, counts, n, total):
    with localcontext() as context:
        context.prec = 80 + 3*max(0, int(log_r/math.log(10)))
        r = Decimal.from_float(math.exp(log_r))
        mean = Decimal(total)/n
        base = _decimal_psi(r)
        score = sum(Decimal(c)*(_decimal_psi(r+x)-base) for x, c in counts)/n - (1+mean/r).ln()
        # Positive rescaling retains the root/sign and avoids float underflow
        # for large roots. No float64 digamma subtraction is performed.
        scaled = score*r*r/(mean*mean) if r >= 1 else score*r
        value = float(scaled)
        if not math.isfinite(value):
            raise FitNumericalError("nonfinite negative binomial score")
        return value


def _nb_root(counts, n, total, excess):
    # MOM supplies only a starting point. Its estimate is never returned.
    start = math.log(total*total) - math.log(excess)
    low = high = start
    lower_limit = math.log(np.nextafter(0.0, 1.0))
    upper_limit = math.log(np.finfo(float).max) - 1e-12
    args = (counts, n, total)
    for _ in range(_BRACKET_STEPS):
        if low < lower_limit or high > upper_limit:
            raise FitNumericalError("negative binomial root is not safely bracketable in float64")
        left = _profile_score(low, *args)
        right = _profile_score(high, *args)
        if left > 0 and right < 0:
            break
        if left <= 0:
            low -= math.log(2)
        if right >= 0:
            high += math.log(2)
    else:
        raise FitNumericalError("negative binomial bracketing budget exhausted")
    root, status = optimize.brentq(
        _profile_score, low, high, args=args, xtol=1e-13, rtol=1e-14,
        maxiter=_SOLVER_ITERATIONS, full_output=True, disp=False,
    )
    if not status.converged or not math.isfinite(root) or not low <= root <= high:
        raise FitNumericalError("negative binomial root solver did not converge")
    residual = _profile_score(root, *args)
    left = _profile_score(root-1e-7, *args)
    right = _profile_score(root+1e-7, *args)
    if not left > 0 > right or abs(residual) > max(abs(left), abs(right))*1e-4:
        raise FitNumericalError("negative binomial score sign or residual validation failed")
    return math.exp(root)


def _nb_decimal_log_likelihood(counts, r, p):
    with localcontext() as context:
        context.prec = 80 + 3*max(0, Decimal.from_float(r).adjusted(), len(str(counts[-1][0])))
        rd, pd = Decimal.from_float(r), Decimal.from_float(p)
        base, logp, logq = _decimal_loggamma(rd), pd.ln(), (1-pd).ln()
        return sum(
            Decimal(c)*(_decimal_loggamma(rd+x)-base-_decimal_loggamma(Decimal(x)+1)
                        +rd*logp+Decimal(x)*logq)
            for x, c in counts
        )


def _nb_log_likelihood(counts, r, p):
    return float(_nb_decimal_log_likelihood(counts, r, p))


def _validate_nb_canonical_pair(counts, n, total, r, p):
    """Reject loss of optimality caused by float64 canonicalization.

    The continuous profile root has already passed its score checks. Inspect
    adjacent representable coordinates and the r implied by the rounded p
    and exact mean (plus its adjacent floats). These are diagnostic competitors,
    never replacement estimates. Use the complete high-precision likelihood;
    an improvement exceeding one output ULP from each likelihood is material.
    This is an output rounding budget, not a parameter validity threshold.
    """
    with localcontext() as context:
        context.prec = 80
        pd = Decimal.from_float(p)
        implied_r = float((Decimal(total)/n)*pd/(1-pd))
    competitors = {(math.nextafter(r, 0.0), p), (math.nextafter(r, math.inf), p),
                   (r, math.nextafter(p, 0.0)), (r, math.nextafter(p, 1.0))}
    if math.isfinite(implied_r) and implied_r > 0:
        competitors.update((rr, p) for rr in (
            implied_r, math.nextafter(implied_r, 0.0), math.nextafter(implied_r, math.inf),
        ))
    baseline = _nb_decimal_log_likelihood(counts, r, p)
    for rr, pp in sorted(competitors):
        if not math.isfinite(rr) or rr <= 0 or not 0 < pp < 1:
            continue
        alternative = _nb_decimal_log_likelihood(counts, rr, pp)
        with localcontext() as context:
            context.prec = 80
            budget = (Decimal.from_float(math.ulp(float(baseline)))
                      + Decimal.from_float(math.ulp(float(alternative))))
            if alternative - baseline > budget:
                raise FitNumericalError("canonical negative binomial pair loses MLE optimality")


def fit_negative_binomial(family, data):
    sample = _fit_input(data, discrete=True)
    counts = tuple(sorted(Counter(map(int, sample)).items()))
    n = len(sample)
    total = sum(x*c for x, c in counts)
    if total == 0:
        raise FitIdentifiabilityError("all-zero negative binomial data do not identify r")
    excess = n*sum(x*x*c for x, c in counts)-total*total-n*total
    if excess <= 0:
        raise NoFiniteMLEError("negative binomial requires strict population overdispersion")
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        try:
            r = _nb_root(counts, n, total, excess)
            with localcontext() as context:
                context.prec = 80
                rd = Decimal.from_float(r)
                p = float(rd/(rd+Decimal(total)/n))
            if not math.isfinite(r) or r <= 0 or not 0 < p < 1:
                raise FitNumericalError("negative binomial MLE cannot be represented canonically")
            bound = family.bind(r=r, p=p)
            ll = _nb_log_likelihood(counts, r, p)
            _validate_nb_canonical_pair(counts, n, total, r, p)
            return _result(bound, n, ll, ("r", "p"),
                           "pymagicstats-negative-binomial-profile-mle-v1", _NB_SOLVER, records)
        except (ValueError, OverflowError, FloatingPointError, DecimalException) as exc:
            raise FitNumericalError("negative binomial numerical fitting failure") from exc
