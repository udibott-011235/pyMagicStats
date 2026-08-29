from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from pyMagicStat.assumptions.models import (
    AssumptionReport,
    Estimand,
    InferenceDesign,
)
from pyMagicStat.assumptions.robustness import RobustnessResult
from pyMagicStat.assumptions.robustness_v3 import RobustnessResultV3
from pyMagicStat.inference.capabilities import (
    InferenceCapability,
    InferenceGuarantee,
)


@dataclass(frozen=True)
class MethodAlternative:
    method: str
    estimand: str
    note: str


class InferenceDecisionStatus(str, Enum):
    SELECTED = "selected"
    REVIEW_REQUIRED = "review_required"
    INSUFFICIENT = "insufficient"
    NOT_CALIBRATED = "not_calibrated"


@dataclass(frozen=True)
class InferenceDecision:
    selected_method: Optional[str]
    robustness: Union[RobustnessResult, RobustnessResultV3]
    report: AssumptionReport
    reasons: Tuple[str, ...] = ()
    alternatives: Tuple[MethodAlternative, ...] = field(default_factory=tuple)
    status: InferenceDecisionStatus = InferenceDecisionStatus.SELECTED
    guarantee: Optional[InferenceGuarantee] = None
    assumptions_used: Tuple[str, ...] = field(default_factory=tuple)
    capabilities: Tuple[InferenceCapability, ...] = field(default_factory=tuple)
    policy_version: Optional[str] = None
    routing_version: Optional[str] = None

    @property
    def estimand(self) -> Estimand:
        return self.report.estimand

    @property
    def design(self) -> InferenceDesign:
        return self.report.design

    @property
    def parametric_recommended(self) -> bool:
        return self.selected_method in {
            "one_sample_t",
            "paired_t",
            "student_t",
            "welch_t",
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_method": self.selected_method,
            "status": self.status.value,
            "estimand": self.estimand.value,
            "design": self.design.value,
            "guarantee": self.guarantee.value if self.guarantee else None,
            "assumptions_used": list(self.assumptions_used),
            "policy_version": self.policy_version,
            "routing_version": self.routing_version,
            "parametric_recommended": self.parametric_recommended,
            "robustness": self.robustness.to_dict(),
            "reasons": list(self.reasons),
            "alternatives": [asdict(item) for item in self.alternatives],
            "capabilities": [item.to_dict() for item in self.capabilities],
            "assumptions": self.report.to_dict(),
        }
