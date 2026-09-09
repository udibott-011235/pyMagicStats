"""Public distribution containers and validators."""

from pyMagicStat.distributions.distributions import Distribution, NormalDistribution
from pyMagicStat.distributions.families import (
    FittedDistribution,
    FittedContinuousDistribution,
    FittedDiscreteDistribution,
    FitResult,
    DistributionFitError,
    FitIdentifiabilityError,
    NoFiniteMLEError,
    FitNumericalError,
    DiscreteDistributionFamily,
    DistributionSupport,
    ExponentialFamily,
    GammaFamily,
    NegativeBinomialFamily,
    NegativeBinomialParameters,
    ParameterizedDiscreteDistribution,
    SupportKind,
)

__all__ = [
    "FittedDistribution",
    "FittedContinuousDistribution",
    "FittedDiscreteDistribution",
    "FitResult",
    "DistributionFitError",
    "FitIdentifiabilityError",
    "NoFiniteMLEError",
    "FitNumericalError",
    "Distribution",
    "NormalDistribution",
    "GammaFamily",
    "ExponentialFamily",
    "SupportKind",
    "DistributionSupport",
    "DiscreteDistributionFamily",
    "ParameterizedDiscreteDistribution",
    "NegativeBinomialParameters",
    "NegativeBinomialFamily",
]
