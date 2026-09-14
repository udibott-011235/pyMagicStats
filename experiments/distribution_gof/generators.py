"""Family construction and sample generation for the private harness."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np

from pyMagicStat.distributions.families import (
    ExponentialFamily,
    GammaFamily,
    NegativeBinomialFamily,
)


def family_for(family_id: str):
    classes = {
        "gamma": GammaFamily,
        "exponential": ExponentialFamily,
        "negative_binomial": NegativeBinomialFamily,
    }
    try:
        return classes[family_id]()
    except KeyError as exc:
        raise ValueError("unknown family") from exc


def numeric_parameters(parameters: dict[str, str] | Any) -> dict[str, float]:
    return {name: float(Decimal(value)) for name, value in dict(parameters).items()}


def bind_family(family_id: str, parameters: dict[str, str] | Any):
    return family_for(family_id).bind(**numeric_parameters(parameters))


def generate_sample(bound, n: int, rng: np.random.Generator) -> np.ndarray:
    try:
        sample = np.asarray(bound.rvs(size=n, rng=rng))
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise GenerationError("backend sample generation failed") from exc
    if sample.shape != (n,):
        raise GenerationError("backend returned a sample with the wrong shape")
    return sample


class GenerationError(RuntimeError):
    pass
