"""Stateless Gamma and Exponential probability-family descriptors."""

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
    ContinuousDistributionFamily,
    DistributionSupport,
    ParameterizedContinuousDistribution,
    SupportKind,
)


def _positive_finite_parameter(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar")
    try:
        canonical = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as float64") from exc
    if not math.isfinite(canonical):
        raise ValueError(f"{name} must be finite")
    if canonical <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return canonical


@dataclass(frozen=True, slots=True)
class GammaParameters:
    """Canonical manually supplied Gamma parameters."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shape",
            _positive_finite_parameter(self.shape, name="shape"),
        )
        object.__setattr__(
            self,
            "scale",
            _positive_finite_parameter(self.scale, name="scale"),
        )


@dataclass(frozen=True, slots=True)
class ExponentialParameters:
    """Canonical manually supplied Exponential parameters."""

    scale: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scale",
            _positive_finite_parameter(self.scale, name="scale"),
        )


_NONNEGATIVE_CONTINUOUS_SUPPORT = DistributionSupport(
    lower=0.0,
    upper=math.inf,
    lower_closed=True,
    upper_closed=False,
    kind=SupportKind.CONTINUOUS,
)


def _immutable_specs(
    specs: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    return MappingProxyType(
        {name: MappingProxyType(dict(spec)) for name, spec in specs.items()}
    )


_GAMMA_PARAMETER_SPECS = _immutable_specs(
    {
        "shape": {
            "canonical": True,
            "fixed": False,
            "constraint": "finite real > 0",
        },
        "scale": {
            "canonical": True,
            "fixed": False,
            "constraint": "finite real > 0",
        },
        "loc": {"canonical": False, "fixed": True, "value": 0.0},
    }
)
_EXPONENTIAL_PARAMETER_SPECS = _immutable_specs(
    {
        "scale": {
            "canonical": True,
            "fixed": False,
            "constraint": "finite real > 0",
        },
        "loc": {"canonical": False, "fixed": True, "value": 0.0},
    }
)
_FIXED_LOC_ZERO: Mapping[str, float] = MappingProxyType({"loc": 0.0})


class GammaFamily(ContinuousDistributionFamily):
    """Stateless Gamma family using shape/scale with fixed ``loc=0``."""

    __slots__ = ()

    def fit(self, data: Any) -> FitResult:
        """Fit the frozen fixed-location maximum-likelihood model."""
        from ._fitting import fit_gamma

        return fit_gamma(self, data)

    @property
    def name(self) -> str:
        return "gamma"

    @property
    def kind(self) -> SupportKind:
        return SupportKind.CONTINUOUS

    @property
    def support(self) -> DistributionSupport:
        return _NONNEGATIVE_CONTINUOUS_SUPPORT

    @property
    def parameterization(self) -> str:
        return "shape-scale; loc=0 fixed"

    @property
    def parameter_specs(self) -> Mapping[str, Mapping[str, Any]]:
        return _GAMMA_PARAMETER_SPECS

    @property
    def fixed_parameters(self) -> Mapping[str, float]:
        return _FIXED_LOC_ZERO

    @property
    def backend_distribution(self) -> str:
        return "scipy.stats.gamma"

    def validate_parameters(self, *, shape: Any, scale: Any) -> GammaParameters:
        return GammaParameters(shape=shape, scale=scale)

    def bind(
        self,
        *,
        shape: Any,
        scale: Any,
    ) -> ParameterizedContinuousDistribution:
        return ParameterizedContinuousDistribution(
            family=self,
            parameters=self.validate_parameters(shape=shape, scale=scale),
        )

    def _validate_parameter_object(self, parameters: Any) -> None:
        if not isinstance(parameters, GammaParameters):
            raise TypeError("GammaFamily requires GammaParameters")

    def _backend(self, parameters: Any) -> Any:
        self._validate_parameter_object(parameters)
        return stats.gamma(
            a=parameters.shape,
            loc=0.0,
            scale=parameters.scale,
        )


class ExponentialFamily(ContinuousDistributionFamily):
    """Stateless Exponential family using scale with fixed ``loc=0``."""

    __slots__ = ()

    def fit(self, data: Any) -> FitResult:
        """Fit the frozen fixed-location maximum-likelihood model."""
        from ._fitting import fit_exponential

        return fit_exponential(self, data)

    @property
    def name(self) -> str:
        return "exponential"

    @property
    def kind(self) -> SupportKind:
        return SupportKind.CONTINUOUS

    @property
    def support(self) -> DistributionSupport:
        return _NONNEGATIVE_CONTINUOUS_SUPPORT

    @property
    def parameterization(self) -> str:
        return "scale; loc=0 fixed"

    @property
    def parameter_specs(self) -> Mapping[str, Mapping[str, Any]]:
        return _EXPONENTIAL_PARAMETER_SPECS

    @property
    def fixed_parameters(self) -> Mapping[str, float]:
        return _FIXED_LOC_ZERO

    @property
    def backend_distribution(self) -> str:
        return "scipy.stats.expon"

    def validate_parameters(self, *, scale: Any) -> ExponentialParameters:
        return ExponentialParameters(scale=scale)

    def bind(self, *, scale: Any) -> ParameterizedContinuousDistribution:
        return ParameterizedContinuousDistribution(
            family=self,
            parameters=self.validate_parameters(scale=scale),
        )

    def _validate_parameter_object(self, parameters: Any) -> None:
        if not isinstance(parameters, ExponentialParameters):
            raise TypeError("ExponentialFamily requires ExponentialParameters")

    def _backend(self, parameters: Any) -> Any:
        self._validate_parameter_object(parameters)
        return stats.expon(loc=0.0, scale=parameters.scale)


__all__ = [
    "ExponentialFamily",
    "ExponentialParameters",
    "GammaFamily",
    "GammaParameters",
]
