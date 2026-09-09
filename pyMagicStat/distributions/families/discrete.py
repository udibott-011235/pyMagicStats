"""Stateless Negative Binomial probability-family descriptor."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy import stats

from pyMagicStat.distributions.families._core import (
    FitResult,
    DiscreteDistributionFamily,
    DistributionSupport,
    ParameterizedDiscreteDistribution,
    SupportKind,
)


def _finite_parameter(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar")
    try:
        canonical = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as float64") from exc
    if not math.isfinite(canonical):
        raise ValueError(f"{name} must be finite")
    return canonical


def _positive_finite_parameter(value: Any, *, name: str) -> float:
    canonical = _finite_parameter(value, name=name)
    if canonical <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return canonical


def _probability_parameter(value: Any, *, name: str) -> float:
    canonical = _positive_finite_parameter(value, name=name)
    if canonical > 1.0:
        raise ValueError(f"{name} must not exceed one")
    return canonical


@dataclass(frozen=True, slots=True)
class NegativeBinomialParameters:
    """Canonical manually supplied Negative Binomial parameters."""

    r: float
    p: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "r",
            _positive_finite_parameter(self.r, name="r"),
        )
        object.__setattr__(
            self,
            "p",
            _probability_parameter(self.p, name="p"),
        )


_NONNEGATIVE_INTEGER_SUPPORT = DistributionSupport(
    lower=0.0,
    upper=math.inf,
    lower_closed=True,
    upper_closed=False,
    kind=SupportKind.DISCRETE,
)


def _immutable_specs(
    specs: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    return MappingProxyType(
        {name: MappingProxyType(dict(spec)) for name, spec in specs.items()}
    )


_NEGATIVE_BINOMIAL_PARAMETER_SPECS = _immutable_specs(
    {
        "r": {
            "canonical": True,
            "fixed": False,
            "constraint": "finite real > 0",
        },
        "p": {
            "canonical": True,
            "fixed": False,
            "constraint": "finite real in (0, 1]",
        },
        "loc": {"canonical": False, "fixed": True, "value": 0.0},
    }
)
_FIXED_LOC_ZERO: Mapping[str, float] = MappingProxyType({"loc": 0.0})


class NegativeBinomialFamily(DiscreteDistributionFamily):
    """Stateless Negative Binomial family using ``r,p`` with fixed ``loc=0``."""

    __slots__ = ()

    def fit(self, data: Any) -> FitResult:
        """Fit the frozen fixed-location maximum-likelihood model."""
        from ._fitting import fit_negative_binomial

        return fit_negative_binomial(self, data)

    @property
    def name(self) -> str:
        return "negative_binomial"

    @property
    def kind(self) -> SupportKind:
        return SupportKind.DISCRETE

    @property
    def support(self) -> DistributionSupport:
        return _NONNEGATIVE_INTEGER_SUPPORT

    @property
    def parameterization(self) -> str:
        return "r-p; loc=0 fixed"

    @property
    def parameter_specs(self) -> Mapping[str, Mapping[str, Any]]:
        return _NEGATIVE_BINOMIAL_PARAMETER_SPECS

    @property
    def fixed_parameters(self) -> Mapping[str, float]:
        return _FIXED_LOC_ZERO

    @property
    def backend_distribution(self) -> str:
        return "scipy.stats.nbinom"

    def validate_parameters(self, *, r: Any, p: Any) -> NegativeBinomialParameters:
        return NegativeBinomialParameters(r=r, p=p)

    def bind(self, *, r: Any, p: Any) -> ParameterizedDiscreteDistribution:
        return ParameterizedDiscreteDistribution(
            family=self,
            parameters=self.validate_parameters(r=r, p=p),
        )

    def _validate_parameter_object(self, parameters: Any) -> None:
        if not isinstance(parameters, NegativeBinomialParameters):
            raise TypeError("NegativeBinomialFamily requires NegativeBinomialParameters")

    def _backend(self, parameters: Any) -> Any:
        self._validate_parameter_object(parameters)
        return stats.nbinom(
            n=parameters.r,
            p=parameters.p,
            loc=0,
        )


__all__ = [
    "NegativeBinomialFamily",
    "NegativeBinomialParameters",
]
