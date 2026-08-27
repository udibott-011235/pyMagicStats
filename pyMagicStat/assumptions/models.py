from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Tuple

import numpy as np


class AssessmentStatus(str, Enum):
    """Severity of one diagnostic assessment."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_ASSESSED = "not_assessed"


class InferenceDesign(str, Enum):
    ONE_SAMPLE = "one_sample"
    PAIRED = "paired"
    TWO_SAMPLE = "two_sample"
    ONE_WAY = "one_way"


class Estimand(str, Enum):
    MEAN = "mean"
    MEAN_DIFFERENCE = "mean_difference"
    PROPORTION = "proportion"
    VARIANCE = "variance"


@dataclass(frozen=True)
class Assessment:
    """Structured result from a single diagnostic."""

    name: str
    status: AssessmentStatus
    metrics: Mapping[str, Any] = field(default_factory=dict)
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["metrics"] = _json_ready(dict(self.metrics))
        return result


@dataclass(frozen=True)
class AssumptionReport:
    """All diagnostics relevant to one inferential design."""

    design: InferenceDesign
    estimand: Estimand
    assessments: Mapping[str, Assessment]

    @property
    def has_failures(self) -> bool:
        return any(item.status is AssessmentStatus.FAIL for item in self.assessments.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "design": self.design.value,
            "estimand": self.estimand.value,
            "has_failures": self.has_failures,
            "assessments": {
                name: assessment.to_dict()
                for name, assessment in self.assessments.items()
            },
        }


@dataclass(frozen=True)
class ValidationResult:
    """Normalized samples plus their diagnostic report."""

    samples: Tuple[np.ndarray, ...]
    relevant_samples: Tuple[np.ndarray, ...]
    report: AssumptionReport


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value
