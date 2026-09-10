"""Public probability-family contracts."""

from pyMagicStat.distributions.families._core import (
    FittedDistribution,
    FittedContinuousDistribution,
    FittedDiscreteDistribution,
    FitResult,
    DistributionFitError,
    FitIdentifiabilityError,
    NoFiniteMLEError,
    FitNumericalError,
    ContinuousDistributionFamily,
    DiscreteDistributionFamily,
    DistributionFamily,
    DistributionSupport,
    ParameterizedContinuousDistribution,
    ParameterizedDiscreteDistribution,
    ParameterizedDistribution,
    SupportKind,
)
from pyMagicStat.distributions.families.continuous import (
    ExponentialFamily,
    ExponentialParameters,
    GammaFamily,
    GammaParameters,
)
from pyMagicStat.distributions.families.discrete import (
    NegativeBinomialFamily,
    NegativeBinomialParameters,
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
    "ContinuousDistributionFamily",
    "DiscreteDistributionFamily",
    "DistributionFamily",
    "DistributionSupport",
    "ExponentialFamily",
    "ExponentialParameters",
    "GammaFamily",
    "GammaParameters",
    "NegativeBinomialFamily",
    "NegativeBinomialParameters",
    "ParameterizedContinuousDistribution",
    "ParameterizedDiscreteDistribution",
    "ParameterizedDistribution",
    "SupportKind",
]
