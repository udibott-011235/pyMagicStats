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

    These conservative rules are intentionally explicit. They are a calibrated
    policy, not a claim that one sample-size threshold proves robustness.

    ``ShapeAssessment`` status is descriptive input, never an independent veto.
    The policy uses sample size, skewness, tail weight and detected-outlier
    fraction together.  The thresholds are documented and reproducible in
    ``experiments/robustness_calibration.py``.
    """

    POLICY_VERSION = "mean-v2-2026-08"
    MODERATE_N = 40
    MODERATE_MAX_SKEW = 1.0
    MODERATE_MAX_KURTOSIS = 3.0
    MODERATE_MAX_OUTLIER_FRACTION = 0.025
    LARGE_N = 80
    LARGE_MAX_SKEW = 2.0
    LARGE_MAX_KURTOSIS = 7.0
    LARGE_MAX_OUTLIER_FRACTION = 0.05
    HEAVY_TAIL_N = 200
    HEAVY_TAIL_MAX_SKEW = 2.0
    HEAVY_TAIL_MAX_KURTOSIS = 25.0
    HEAVY_TAIL_MAX_OUTLIER_FRACTION = 0.05

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

        min_n = min(int(item.metrics.get("n", 0)) for item in shapes)
        extreme_count = sum(int(item.metrics.get("count", 0)) for item in outliers)
        max_outlier_fraction = max(
            (float(item.metrics.get("fraction", 0.0)) for item in outliers),
            default=0.0,
        )
        max_abs_skew = max(
            abs(float(item.metrics.get("skewness", np.inf))) for item in shapes
        )
        max_abs_kurtosis = max(
            abs(float(item.metrics.get("excess_kurtosis", np.inf))) for item in shapes
        )
        shape_departure = any(
            item.status in {AssessmentStatus.WARN, AssessmentStatus.FAIL}
            for item in shapes
        )

        if extreme_count:
            reasons.append("Extreme observations may remain influential for mean-based inference.")

        if extreme_count and not shape_departure:
            reasons.append("The overall shape remains compatible with t-based inference.")
            return RobustnessResult(RobustnessLevel.CAUTION, tuple(reasons))

        if not shape_departure and not extreme_count:
            reasons.append("The relevant observations are compatible with direct t-based inference.")
            level = RobustnessLevel.CAUTION if independence_unknown else RobustnessLevel.ACCEPTABLE
            return RobustnessResult(level, tuple(reasons))

        if (
            min_n >= self.LARGE_N
            and max_abs_skew <= self.LARGE_MAX_SKEW
            and max_abs_kurtosis <= self.LARGE_MAX_KURTOSIS
            and max_outlier_fraction <= self.LARGE_MAX_OUTLIER_FRACTION
        ):
            reasons.append("Large samples and bounded shape departure support an asymptotic approximation.")
            return RobustnessResult(
                RobustnessLevel.CAUTION
                if extreme_count or independence_unknown
                else RobustnessLevel.ACCEPTABLE,
                tuple(reasons),
            )

        if (
            min_n >= self.MODERATE_N
            and max_abs_skew <= self.MODERATE_MAX_SKEW
            and max_abs_kurtosis <= self.MODERATE_MAX_KURTOSIS
            and max_outlier_fraction <= self.MODERATE_MAX_OUTLIER_FRACTION
        ):
            reasons.append("Moderate shape departure is acceptable at the available sample size.")
            level = (
                RobustnessLevel.CAUTION
                if extreme_count or independence_unknown
                else RobustnessLevel.ACCEPTABLE
            )
            return RobustnessResult(level, tuple(reasons))

        if (
            min_n >= self.HEAVY_TAIL_N
            and max_abs_skew <= self.HEAVY_TAIL_MAX_SKEW
            and max_abs_kurtosis <= self.HEAVY_TAIL_MAX_KURTOSIS
            and max_outlier_fraction <= self.HEAVY_TAIL_MAX_OUTLIER_FRACTION
        ):
            reasons.append(
                "Very large samples support cautious mean inference for calibrated heavy-tail departures."
            )
            return RobustnessResult(RobustnessLevel.CAUTION, tuple(reasons))

        reasons.append("The sample size does not offset the observed shape departure.")
        return RobustnessResult(RobustnessLevel.INSUFFICIENT, tuple(reasons))
