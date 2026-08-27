from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pyMagicStat.assumptions.models import AssumptionReport
from pyMagicStat.assumptions.robustness import RobustnessResult


@dataclass(frozen=True)
class MethodAlternative:
    method: str
    estimand: str
    note: str


class InferenceDecisionStatus(str, Enum):
    SELECTED = "selected"
    INSUFFICIENT = "insufficient"
    NOT_CALIBRATED = "not_calibrated"


@dataclass(frozen=True)
class InferenceDecision:
    selected_method: Optional[str]
    robustness: RobustnessResult
    report: AssumptionReport
    reasons: Tuple[str, ...] = ()
    alternatives: Tuple[MethodAlternative, ...] = field(default_factory=tuple)
    status: InferenceDecisionStatus = InferenceDecisionStatus.SELECTED

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
            "parametric_recommended": self.parametric_recommended,
            "robustness": self.robustness.to_dict(),
            "reasons": list(self.reasons),
            "alternatives": [asdict(item) for item in self.alternatives],
            "assumptions": self.report.to_dict(),
        }
