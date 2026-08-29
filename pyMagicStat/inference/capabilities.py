"""Typed inference guarantees and the central method-capability registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Tuple

from pyMagicStat.assumptions.models import Estimand, InferenceDesign


INFERENCE_ROUTING_VERSION = "inference-capability-routing-v1-candidate-2026-08"


class InferenceGuarantee(str, Enum):
    """Why an inference engine can justify its stated estimand."""

    EXACT_PARAMETRIC = "exact_parametric"
    ASYMPTOTIC_PARAMETRIC = "asymptotic_parametric"
    ASYMPTOTIC_MOMENT_BASED = "asymptotic_moment_based"
    HIGHER_ORDER_CORRECTED = "higher_order_corrected"
    RESAMPLING_BASED = "resampling_based"
    NOT_CALIBRATED = "not_calibrated"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class InferenceCapability:
    """Declarative relationship between a method and its inferential basis."""

    method: str
    estimand: Estimand
    design: InferenceDesign
    guarantee: InferenceGuarantee
    assumptions_required: Tuple[str, ...]
    calibrated: bool
    automatic_selection_allowed: bool
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "estimand": self.estimand.value,
            "design": self.design.value,
            "guarantee": self.guarantee.value,
            "assumptions_required": list(self.assumptions_required),
            "calibrated": self.calibrated,
            "automatic_selection_allowed": self.automatic_selection_allowed,
            "notes": list(self.notes),
        }


_ONE_SAMPLE_MEAN_CAPABILITIES = (
    InferenceCapability(
        method="one_sample_t",
        estimand=Estimand.MEAN,
        design=InferenceDesign.ONE_SAMPLE,
        guarantee=InferenceGuarantee.EXACT_PARAMETRIC,
        assumptions_required=(
            "structural_data_supported",
            "independence_supported",
            "external_gaussian_model",
        ),
        calibrated=True,
        automatic_selection_allowed=True,
        notes=(
            "Exact finite-sample inference under the externally supported Gaussian model.",
        ),
    ),
    InferenceCapability(
        method="empirical_likelihood",
        estimand=Estimand.MEAN,
        design=InferenceDesign.ONE_SAMPLE,
        guarantee=InferenceGuarantee.ASYMPTOTIC_MOMENT_BASED,
        assumptions_required=(
            "structural_data_supported",
            "independence_supported",
            "finite_mean",
            "empirical_likelihood_regularity",
        ),
        calibrated=False,
        automatic_selection_allowed=False,
        notes=(
            "Candidate registration only; the numerical engine is not implemented.",
        ),
    ),
    InferenceCapability(
        method="bartlett_empirical_likelihood",
        estimand=Estimand.MEAN,
        design=InferenceDesign.ONE_SAMPLE,
        guarantee=InferenceGuarantee.HIGHER_ORDER_CORRECTED,
        assumptions_required=(
            "structural_data_supported",
            "independence_supported",
            "finite_mean",
            "bartlett_regularity",
        ),
        calibrated=False,
        automatic_selection_allowed=False,
        notes=(
            "Candidate registration only; no Bartlett correction is implemented.",
        ),
    ),
    InferenceCapability(
        method="bootstrap_t",
        estimand=Estimand.MEAN,
        design=InferenceDesign.ONE_SAMPLE,
        guarantee=InferenceGuarantee.RESAMPLING_BASED,
        assumptions_required=(
            "structural_data_supported",
            "independence_supported",
            "finite_variance",
            "bootstrap_t_regularity",
        ),
        calibrated=False,
        automatic_selection_allowed=False,
        notes=(
            "Candidate registration only; no automatic bootstrap-t routing is calibrated.",
        ),
    ),
)


INFERENCE_CAPABILITY_REGISTRY: Mapping[
    tuple[InferenceDesign, Estimand],
    Tuple[InferenceCapability, ...],
] = MappingProxyType(
    {
        (InferenceDesign.ONE_SAMPLE, Estimand.MEAN): _ONE_SAMPLE_MEAN_CAPABILITIES,
    }
)


def capabilities_for(
    design: InferenceDesign,
    estimand: Estimand,
) -> Tuple[InferenceCapability, ...]:
    """Return registered capabilities without inferring evidence from data shape."""

    return INFERENCE_CAPABILITY_REGISTRY.get((design, estimand), ())


def capability_for(
    method: str,
    design: InferenceDesign,
    estimand: Estimand,
) -> InferenceCapability | None:
    """Return one registered method capability for an estimand/design pair."""

    return next(
        (
            capability
            for capability in capabilities_for(design, estimand)
            if capability.method == method
        ),
        None,
    )
