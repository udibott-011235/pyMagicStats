"""Shared contracts for parameterized probability distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np
import scipy


class SupportKind(str, Enum):
    """Mathematical domain kind for a probability distribution."""

    CONTINUOUS = "continuous"


def _canonical_bound(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar")
    try:
        canonical = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as float64") from exc
    if math.isnan(canonical):
        raise ValueError(f"{name} must not be NaN")
    return canonical


@dataclass(frozen=True, slots=True)
class DistributionSupport:
    """Immutable mathematical support independent of observed data."""

    lower: float
    upper: float
    lower_closed: bool
    upper_closed: bool
    kind: SupportKind

    def __post_init__(self) -> None:
        lower = _canonical_bound(self.lower, name="lower")
        upper = _canonical_bound(self.upper, name="upper")
        if lower > upper:
            raise ValueError("lower must not exceed upper")
        if not isinstance(self.lower_closed, (bool, np.bool_)):
            raise TypeError("lower_closed must be boolean")
        if not isinstance(self.upper_closed, (bool, np.bool_)):
            raise TypeError("upper_closed must be boolean")
        if math.isinf(lower) and bool(self.lower_closed):
            raise ValueError("an infinite lower bound cannot be closed")
        if math.isinf(upper) and bool(self.upper_closed):
            raise ValueError("an infinite upper bound cannot be closed")
        if not isinstance(self.kind, SupportKind):
            raise TypeError("kind must be a SupportKind")

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "lower_closed", bool(self.lower_closed))
        object.__setattr__(self, "upper_closed", bool(self.upper_closed))

    def contains(self, value: Any) -> bool | np.ndarray:
        """Return mathematical membership, elementwise for array-like input."""

        scalar_input = np.isscalar(value)
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return False if scalar_input else np.asarray(False)

        if (
            np.issubdtype(array.dtype, np.bool_)
            or not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)
        ):
            result = np.zeros(array.shape, dtype=bool)
            return bool(result.item()) if scalar_input else result

        try:
            numeric = np.asarray(array, dtype=np.float64)
        except (OverflowError, TypeError, ValueError):
            result = np.zeros(array.shape, dtype=bool)
            return bool(result.item()) if scalar_input else result
        result = np.isfinite(numeric)
        if self.lower_closed:
            result &= numeric >= self.lower
        else:
            result &= numeric > self.lower
        if self.upper_closed:
            result &= numeric <= self.upper
        else:
            result &= numeric < self.upper

        result = np.asarray(result, dtype=bool)
        return bool(result.item()) if scalar_input else result


def _query_input(value: Any, *, name: str) -> tuple[np.ndarray, bool]:
    """Validate and normalize one probability-operation input."""

    scalar_input = np.isscalar(value)
    if scalar_input and isinstance(value, Real) and not isinstance(
        value, (bool, np.bool_)
    ):
        try:
            numeric_scalar = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be representable as float64") from exc
        if not math.isfinite(numeric_scalar):
            raise ValueError(f"{name} must contain only finite values")
        return np.asarray(numeric_scalar, dtype=np.float64), True

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric scalar or array-like") from exc

    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must contain only real numeric values")

    try:
        numeric = np.asarray(array, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as float64") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values")
    return numeric, scalar_input


def _normalize_result(value: Any, *, scalar_input: bool) -> float | np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if scalar_input:
        return float(result.item())
    return result


def _validated_rng(rng: Any) -> np.random.Generator:
    if isinstance(rng, (bool, np.bool_)):
        raise TypeError("rng must be a non-negative integer seed or Generator")
    if isinstance(rng, Integral):
        seed = int(rng)
        if seed < 0:
            raise ValueError("rng integer seed must be non-negative")
        return np.random.default_rng(seed)
    if isinstance(rng, np.random.Generator):
        return rng
    raise TypeError("rng must be a non-negative integer seed or Generator")


def _validated_size(size: Any) -> None | int | tuple[int, ...]:
    if size is None:
        return None
    if isinstance(size, (bool, np.bool_)):
        raise TypeError("size dimensions must be non-negative integers")
    if isinstance(size, Integral):
        dimension = int(size)
        if dimension < 0:
            raise ValueError("size dimensions must be non-negative")
        return dimension
    if isinstance(size, tuple):
        dimensions: list[int] = []
        for dimension in size:
            if isinstance(dimension, (bool, np.bool_)) or not isinstance(
                dimension, Integral
            ):
                raise TypeError("size dimensions must be non-negative integers")
            canonical = int(dimension)
            if canonical < 0:
                raise ValueError("size dimensions must be non-negative")
            dimensions.append(canonical)
        return tuple(dimensions)
    raise TypeError("size must be None, an integer, or a tuple of integers")


class DistributionFamily(ABC):
    """Stateless descriptor of a probability-family definition."""

    __slots__ = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical family name."""

    @property
    @abstractmethod
    def kind(self) -> SupportKind:
        """Continuous or discrete family kind."""

    @property
    @abstractmethod
    def support(self) -> DistributionSupport:
        """Mathematical support independent of observations."""

    @property
    @abstractmethod
    def parameterization(self) -> str:
        """Canonical parameterization identifier."""

    @property
    @abstractmethod
    def parameter_specs(self) -> Mapping[str, Mapping[str, Any]]:
        """Immutable canonical and fixed-parameter metadata."""

    @property
    @abstractmethod
    def fixed_parameters(self) -> Mapping[str, float]:
        """Immutable fixed backend parameters."""

    @property
    def backend_library(self) -> str:
        return "scipy"

    @property
    @abstractmethod
    def backend_distribution(self) -> str:
        """Fully qualified backend distribution identity."""

    @property
    def backend_version(self) -> str:
        return scipy.__version__

    @abstractmethod
    def validate_parameters(self, **parameters: Any) -> Any:
        """Return the canonical immutable parameter object or fail closed."""

    @abstractmethod
    def bind(self, **parameters: Any) -> "ParameterizedDistribution":
        """Create one immutable manually parameterized distribution."""

    @abstractmethod
    def _validate_parameter_object(self, parameters: Any) -> None:
        """Validate direct construction of a bound object."""

    @abstractmethod
    def _backend(self, parameters: Any) -> Any:
        """Construct the SciPy backend lazily from canonical parameters."""


class ContinuousDistributionFamily(DistributionFamily, ABC):
    """Stateless descriptor for a continuous probability family."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class ParameterizedDistribution:
    """Immutable family plus one canonical parameter value object."""

    family: DistributionFamily
    parameters: Any

    def __post_init__(self) -> None:
        if not isinstance(self.family, DistributionFamily):
            raise TypeError("family must be a DistributionFamily")
        self.family._validate_parameter_object(self.parameters)

    @property
    def name(self) -> str:
        return self.family.name

    @property
    def kind(self) -> SupportKind:
        return self.family.kind

    @property
    def support(self) -> DistributionSupport:
        return self.family.support

    @property
    def parameterization(self) -> str:
        return self.family.parameterization

    @property
    def parameter_specs(self) -> Mapping[str, Mapping[str, Any]]:
        return self.family.parameter_specs

    @property
    def fixed_parameters(self) -> Mapping[str, float]:
        return self.family.fixed_parameters

    @property
    def backend_library(self) -> str:
        return self.family.backend_library

    @property
    def backend_distribution(self) -> str:
        return self.family.backend_distribution

    @property
    def backend_version(self) -> str:
        return self.family.backend_version

    def _evaluate(self, method: str, value: Any, *, name: str) -> float | np.ndarray:
        query, scalar_input = _query_input(value, name=name)
        result = getattr(self.family._backend(self.parameters), method)(query)
        return _normalize_result(result, scalar_input=scalar_input)

    def cdf(self, x: Any) -> float | np.ndarray:
        return self._evaluate("cdf", x, name="x")

    def logcdf(self, x: Any) -> float | np.ndarray:
        return self._evaluate("logcdf", x, name="x")

    def sf(self, x: Any) -> float | np.ndarray:
        return self._evaluate("sf", x, name="x")

    def logsf(self, x: Any) -> float | np.ndarray:
        return self._evaluate("logsf", x, name="x")

    def ppf(self, q: Any) -> float | np.ndarray:
        probabilities, scalar_input = _query_input(q, name="q")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("q must contain only values in [0, 1]")
        result = self.family._backend(self.parameters).ppf(probabilities)
        return _normalize_result(result, scalar_input=scalar_input)

    def rvs(
        self,
        size: None | int | tuple[int, ...] = None,
        *,
        rng: Any,
    ) -> float | np.ndarray:
        """Sample with an explicit seed or caller-owned Generator.

        A caller-owned ``Generator`` is used directly and therefore advances.
        Integer seeds create a local ``Generator`` and never touch NumPy's
        process-global random state.
        """

        validated_size = _validated_size(size)
        generator = _validated_rng(rng)
        result = self.family._backend(self.parameters).rvs(
            size=validated_size,
            random_state=generator,
        )
        return _normalize_result(result, scalar_input=validated_size is None)


@dataclass(frozen=True, slots=True)
class ParameterizedContinuousDistribution(ParameterizedDistribution):
    """Immutable parameterized distribution with continuous operations."""

    def __post_init__(self) -> None:
        ParameterizedDistribution.__post_init__(self)
        if not isinstance(self.family, ContinuousDistributionFamily):
            raise TypeError("family must be a ContinuousDistributionFamily")

    def pdf(self, x: Any) -> float | np.ndarray:
        return self._evaluate("pdf", x, name="x")

    def logpdf(self, x: Any) -> float | np.ndarray:
        return self._evaluate("logpdf", x, name="x")


__all__ = [
    "ContinuousDistributionFamily",
    "DistributionFamily",
    "DistributionSupport",
    "ParameterizedContinuousDistribution",
    "ParameterizedDistribution",
    "SupportKind",
]
