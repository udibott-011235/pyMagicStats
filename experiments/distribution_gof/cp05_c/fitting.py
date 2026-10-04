"""Research fit protocol adapter. No production FitResult provenance spoofing."""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np
import scipy

from pyMagicStat.distributions.families import (
    NegativeBinomialFamily, FittedDiscreteDistribution, FitIdentifiabilityError,
    NoFiniteMLEError, FitNumericalError,
)
from ..cuda_calibration.nb_fitting_perf01 import fast_cpu

ENGINE = "EXPERIMENTAL_FAST_CPU_FLOAT64"
ESTIMATOR_ID = "research-cp05-c-fastcpu-nb-profile-mle-v1"


@dataclass(frozen=True, slots=True)
class ResearchFitResult:
    fitted_distribution: FittedDiscreteDistribution
    n_observations: int
    log_likelihood: float
    estimation_method: str = "maximum_likelihood"
    fixed_parameters: tuple = ("loc",)
    estimated_parameters: tuple = ("r", "p")
    converged: bool = True
    warnings: tuple = ("PRODUCTION_BACKEND=NO", "ENGINE=" + ENGINE)

    @property
    def metadata(self):
        return MappingProxyType({"estimator_id": ESTIMATOR_ID,
                                 "solver_id": "research-perf01-fastcpu-brentq-v1",
                                 "ENGINE": ENGINE, "PRODUCTION_BACKEND": "NO"})

    @property
    def backend(self):
        return ENGINE

    @property
    def backend_version(self):
        return scipy.__version__

    @property
    def aic(self):
        return 4 - 2 * self.log_likelihood

    @property
    def bic(self):
        return 2 * math.log(self.n_observations) - 2 * self.log_likelihood


def operational_fit(family, sample, outcome):
    """Map PERF-01 output to CP04 eligibility/errors and fitted operations."""
    if type(family) is not NegativeBinomialFamily:
        raise ValueError("fast adapter only supports concrete NegativeBinomialFamily")
    if not isinstance(outcome, dict):
        raise FitNumericalError("malformed fast solver result")
    label = outcome.get("classification")
    if label in ("ALL_ZERO_NON_IDENTIFYING", "VARIANCE_NOT_GREATER_THAN_MEAN"):
        # Only the solver's exact classification tokens map to mathematics.
        if outcome.get("converged") is not False or outcome.get("failure_reason") != label:
            raise FitNumericalError("invalid fast mathematical-ineligibility evidence")
        error = FitIdentifiabilityError if label == "ALL_ZERO_NON_IDENTIFYING" else NoFiniteMLEError
        raise error(label)
    if (label != "ELIGIBLE" or outcome.get("converged") is not True
            or outcome.get("failure_reason") is not None):
        raise FitNumericalError(str(outcome.get("failure_reason") or "fast fit failed"))
    try:
        r, p = outcome["parameters"]["r"], outcome["parameters"]["p"]
        ll = outcome["log_likelihood"]
        if not all(type(v) in (float, int) and math.isfinite(v) for v in (r, p, ll)):
            raise ValueError("nonfinite fast fit")
        if r <= 0 or not 0 < p < 1:
            raise ValueError("invalid fast parameter pair")
        fitted = FittedDiscreteDistribution(family.bind(r=float(r), p=float(p)))
        result = ResearchFitResult(fitted, int(np.asarray(sample).size), float(ll))
        if result.n_observations < 1 or not all(math.isfinite(v) for v in (result.aic, result.bic)):
            raise ValueError("invalid fast likelihood criteria")
        return result
    except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        raise FitNumericalError("invalid fast solver evidence") from exc


def fast_fit(family, sample):
    if type(family) is not NegativeBinomialFamily:
        raise ValueError("fast adapter only supports concrete NegativeBinomialFamily")
    try:
        outcome = fast_cpu.fit_negative_binomial(sample)
    except Exception as exc:
        raise FitNumericalError("fast solver backend failure") from exc
    return operational_fit(family, sample, outcome)


def fit_hook(manifest):
    if not manifest.nb_composite:
        raise ValueError("fast fit hook is restricted to NB composite cells")
    return fast_fit
