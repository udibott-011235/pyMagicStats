from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np

from pyMagicStat.assumptions.models import (
    AssessmentStatus,
    AssumptionReport,
)


class RobustnessLevel(str, Enum):
    ACCEPTABLE = "acceptable"
    CAUTION = "caution"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class RobustnessResult:
    level: RobustnessLevel
    reasons: Tuple[str, ...]

    def to_dict(self):
        return {"level": self.level.value, "reasons": list(self.reasons)}


class SamplingRobustness:
    """Interpret diagnostics for mean-based asymptotic inference.

    These conservative rules are intentionally explicit. They are a versioned
    policy, not a claim that one sample-size threshold proves robustness.
    """

    def evaluate(self, report: AssumptionReport) -> RobustnessResult:
        if report.has_failures and any(
            name.startswith("data_quality")
            for name, item in report.assessments.items()
            if item.status is AssessmentStatus.FAIL
        ):
            return RobustnessResult(
                RobustnessLevel.INSUFFICIENT,
                ("Structural data requirements failed.",),
            )

        shapes = [
            item
            for name, item in report.assessments.items()
            if name.startswith("shape")
        ]
        outliers = [
            item
            for name, item in report.assessments.items()
            if name.startswith("outliers")
        ]
        reasons: List[str] = []
        independence_unknown = any(
            name.startswith("independence")
            and item.status is AssessmentStatus.NOT_ASSESSED
            for name, item in report.assessments.items()
        )
        if independence_unknown:
            reasons.append("Independence was not assessed from study-design metadata.")

        if not shapes:
            return RobustnessResult(
                RobustnessLevel.CAUTION,
                ("No shape assessment was available for the target estimator.",),
            )

        if any(item.status is AssessmentStatus.FAIL for item in shapes):
            return RobustnessResult(
                RobustnessLevel.INSUFFICIENT,
                ("Severe skewness or tail weight makes the asymptotic approximation unreliable.",),
            )

        min_n = min(int(item.metrics.get("n", 0)) for item in shapes)
        extreme_count = sum(int(item.metrics.get("count", 0)) for item in outliers)
        max_abs_skew = max(
            abs(float(item.metrics.get("skewness", np.inf))) for item in shapes
        )
        max_abs_kurtosis = max(
            abs(float(item.metrics.get("excess_kurtosis", np.inf))) for item in shapes
        )
        shape_warning = any(item.status is AssessmentStatus.WARN for item in shapes)

        if extreme_count:
            if shape_warning and min_n < 80:
                return RobustnessResult(
                    RobustnessLevel.INSUFFICIENT,
                    ("Extreme observations combined with shape departure can dominate the mean.",),
                )
            reasons.append("Extreme observations may remain influential for mean-based inference.")

        if extreme_count and not shape_warning:
            reasons.append("The overall shape remains compatible with t-based inference.")
            return RobustnessResult(RobustnessLevel.CAUTION, tuple(reasons))

        if not shape_warning and not extreme_count:
            reasons.append("The relevant observations are compatible with direct t-based inference.")
            level = RobustnessLevel.CAUTION if independence_unknown else RobustnessLevel.ACCEPTABLE
            return RobustnessResult(level, tuple(reasons))

        if min_n >= 80 and max_abs_skew <= 2.0 and max_abs_kurtosis <= 7.0:
            reasons.append("Large samples and bounded shape departure support an asymptotic approximation.")
            return RobustnessResult(
                RobustnessLevel.CAUTION
                if extreme_count or independence_unknown
                else RobustnessLevel.ACCEPTABLE,
                tuple(reasons),
            )

        if min_n >= 40 and max_abs_skew <= 1.0 and max_abs_kurtosis <= 3.0 and not extreme_count:
            reasons.append("Moderate shape departure is acceptable at the available sample size.")
            level = RobustnessLevel.CAUTION if independence_unknown else RobustnessLevel.ACCEPTABLE
            return RobustnessResult(level, tuple(reasons))

        reasons.append("The sample size does not offset the observed shape departure.")
        return RobustnessResult(RobustnessLevel.INSUFFICIENT, tuple(reasons))
