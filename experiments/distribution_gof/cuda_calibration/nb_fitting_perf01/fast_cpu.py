"""EXPERIMENTAL_FAST_CPU_FLOAT64. PRODUCTION_BACKEND=NO.

Independent CPU implementation of the CUDA candidate's stable integer
recurrences. No CUDA imports, canonical solver replacement or RNG calls.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import optimize, special

from .contract import FAST_CPU_ENGINE, PRODUCTION_BACKEND


def classification(sample):
    """Exact canonical population overdispersion using Python integer sums."""
    x = np.asarray(sample)
    if (x.ndim != 1 or not x.size or x.dtype.kind not in "iuf"
            or not np.all(np.isfinite(x)) or np.any(x < 0)
            or np.any(x != np.floor(x))):
        raise ValueError("NB nonempty nonnegative integral sample required")
    integers = [int(v) for v in x]
    if max(integers) > (1 << 63) - 1:
        raise ValueError("NB sample outside int64 range")
    n, total = len(integers), sum(integers)
    if total == 0:
        return "ALL_ZERO_NON_IDENTIFYING"
    if n * sum(v * v for v in integers) - total * total <= n * total:
        return "VARIANCE_NOT_GREATER_THAN_MEAN"
    return "ELIGIBLE"


def failed(reason, label="FAILED"):
    return {"classification": label, "converged": False, "parameters": {},
            "log_likelihood": None, "failure_reason": str(reason)}


def _small_ratio(z):
    polynomial = 1.0 / 13
    for denominator in range(12, 1, -1):
        polynomial = 1.0 / denominator - z * polynomial
    return polynomial


def fit_negative_binomial(sample):
    """Bounded log-r root with sign, residual and objective validation.

    Solver budgets and recurrence envelope match the current CUDA candidate.
    The strict transformed-parameter and objective gates are applied separately
    over every frozen record, never used to tune this solver during execution.
    """
    label = "FAILED"
    try:
        label = classification(sample)
        if label != "ELIGIBLE":
            return failed(label, label)
        x = np.asarray(sample, dtype=np.float64)
        maximum = int(np.max(x))
        if maximum > 1_000_000:
            return failed("NB score recurrence term budget exhausted", label)
        n = x.size
        # count(x > k)/n, equivalent to CUDA's aggregated recurrence, without
        # materializing an observations-by-support matrix on the CPU.
        counts = np.bincount(x.astype(np.int64), minlength=maximum + 1)
        survival = np.cumsum(counts[:0:-1], dtype=np.float64)[::-1] / n
        k = np.arange(maximum, dtype=np.float64)
        mean = float(np.mean(x))
        excess = float(np.mean((x - mean) ** 2)) - mean
        r0 = mean * mean / max(excess, np.finfo(np.float64).tiny)
        if not math.isfinite(r0) or r0 <= 0:
            r0 = 1.0

        def score(eta):
            r = math.exp(eta)
            z = mean / r
            ratio = _small_ratio(z) if z < 1e-3 else (z - math.log1p(z)) / (z * z)
            return mean * mean * ratio - float(np.sum(survival * (k / (1 + k / r))))

        def likelihood(eta):
            r = math.exp(eta)
            z = mean / r
            recurrence = float(np.sum(survival * np.log1p(k / r)))
            return n * (recurrence - log_factorial_mean + mean * math.log(mean)
                        - mean * (math.log1p(z) / z + math.log1p(z)))

        low = high = math.log(r0)
        lower = math.log(np.finfo(np.float64).tiny) + math.log(2)
        upper = math.log(np.finfo(np.float64).max) - math.log(2)
        for _ in range(1024):
            if not lower <= low <= high <= upper:
                return failed("NB root outside float64 bracket envelope", label)
            left, right = score(low), score(high)
            if not math.isfinite(left) or not math.isfinite(right):
                return failed("NB nonfinite bracket score", label)
            if left > 0 > right:
                break
            if left <= 0:
                low -= math.log(2)
            if right >= 0:
                high += math.log(2)
        else:
            return failed("NB bracketing budget exhausted", label)
        eta, status = optimize.brentq(score, low, high, xtol=1e-13, rtol=1e-14,
                                     maxiter=128, full_output=True, disp=False)
        residual = score(eta)
        left, right = score(eta - 1e-7), score(eta + 1e-7)
        checks = {
            "bracket_found": True,
            "root_inside_bracket": status.converged and low < eta < high,
            "root_sign_check": left > 0 > right,
            "root_residual_check": math.isfinite(residual) and
                abs(residual) <= max(abs(left), abs(right)) * 1e-4,
        }
        r = math.exp(eta)
        p = r / (r + mean)
        log_factorial_mean = float(np.mean(special.gammaln(x + 1)))
        ll = likelihood(eta)
        competitor = max(likelihood(math.log(r0)), likelihood(eta - 1e-5),
                         likelihood(eta + 1e-5))
        budget = 64 * np.finfo(np.float64).eps * max(1, abs(ll), abs(competitor))
        checks["objective_valid"] = math.isfinite(ll) and math.isfinite(competitor) and ll + budget >= competitor
        checks["finite_pair"] = math.isfinite(r) and r > 0 and math.isfinite(p) and 0 < p < 1
        ok = all(checks.values())
        return {"classification": label, "converged": ok, "parameters": {"r": r, "p": p},
                "log_likelihood": ll, "checks": checks, "iterations": status.iterations,
                "failure_reason": None if ok else "NB validation failed: " +
                    ",".join(name for name, passed in checks.items() if not passed),
                "engine": FAST_CPU_ENGINE, "PRODUCTION_BACKEND": PRODUCTION_BACKEND}
    except (ValueError, OverflowError, FloatingPointError, RuntimeError) as exc:
        return failed(type(exc).__name__ + ": " + str(exc), label)
