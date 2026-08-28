"""Candidate robustness policy for independent one-way mean inference."""

from typing import List

import numpy as np

from pyMagicStat.assumptions.models import AssessmentStatus, AssumptionReport, InferenceDesign
from pyMagicStat.assumptions.robustness import RobustnessLevel, RobustnessResult


class OneWayRobustness:
    """Interpret one-way diagnostics without reusing one-sample thresholds.

    The constants remain a candidate policy until the versioned ANOVA
    calibration is complete.  Structural constraints are evaluated before any
    shape-based route.
    """

    POLICY_VERSION = "anova-v1-2026-08"
    MIN_GROUP_N = 8
    DIRECT_ACCEPTABLE_N = 15
    SMALL_MAX_SKEW = 1.25
    SMALL_MAX_KURTOSIS = 4.0
    SMALL_MAX_OUTLIER_FRACTION = 0.10
    MODERATE_GROUP_N = 25
    MODERATE_TOTAL_N = 75
    MODERATE_MAX_SKEW = 1.75
    MODERATE_MAX_KURTOSIS = 6.0
    MODERATE_MAX_OUTLIER_FRACTION = 0.08
    LARGE_GROUP_N = 50
    LARGE_TOTAL_N = 150
    LARGE_MAX_SKEW = 3.0
    LARGE_MAX_KURTOSIS = 15.0
    LARGE_MAX_OUTLIER_FRACTION = 0.10

    def evaluate(self, report: AssumptionReport) -> RobustnessResult:
        if report.design is not InferenceDesign.ONE_WAY:
            raise ValueError("OneWayRobustness requires an ONE_WAY assumption report")

        quality = [
            item
            for name, item in report.assessments.items()
            if name.startswith("data_quality_group_")
        ]
        if not quality or any(item.status is AssessmentStatus.FAIL for item in quality):
            return RobustnessResult(
                RobustnessLevel.INSUFFICIENT,
                ("Structural one-way data requirements failed.",),
            )

        group_shapes = [
            item
            for name, item in report.assessments.items()
            if name.startswith("shape_group_")
        ]
        group_outliers = [
            item
            for name, item in report.assessments.items()
            if name.startswith("outliers_group_")
        ]
        if not group_shapes or not group_outliers:
            return RobustnessResult(
                RobustnessLevel.INSUFFICIENT,
                ("Per-group shape and influence diagnostics are required.",),
            )

        min_group_n = min(int(item.metrics.get("n", 0)) for item in quality)
        total_n = sum(int(item.metrics.get("n", 0)) for item in quality)
        if min_group_n < self.MIN_GROUP_N:
            return RobustnessResult(
                RobustnessLevel.INSUFFICIENT,
                (
                    f"At least {self.MIN_GROUP_N} observations per group are required "
                    "by the candidate one-way policy.",
                ),
            )

        dominating_extreme = any(
            item.metrics.get("method") == "modified_z_score"
            and float(item.metrics.get("max_robust_score", 0.0)) >= 8.0
            for item in group_outliers
        )
        if dominating_extreme:
            return RobustnessResult(
                RobustnessLevel.INSUFFICIENT,
                (
                    "A within-group extreme observation exceeds the calibrated "
                    "influence guardrail.",
                ),
            )

        max_abs_skew = max(
            abs(float(item.metrics.get("skewness", np.inf)))
            for item in group_shapes
        )
        max_abs_kurtosis = max(
            abs(float(item.metrics.get("excess_kurtosis", np.inf)))
            for item in group_shapes
        )
        max_outlier_fraction = max(
            float(item.metrics.get("fraction", 0.0)) for item in group_outliers
        )
        extreme_count = sum(
            int(item.metrics.get("count", 0)) for item in group_outliers
        )
        residual_shape = report.assessments.get("shape_standardized_residuals")
        residual_outliers = report.assessments.get("outliers_standardized_residuals")
        independence_unknown = any(
            name.startswith("independence")
            and item.status is AssessmentStatus.NOT_ASSESSED
            for name, item in report.assessments.items()
        )
        reasons: List[str] = []
        if independence_unknown:
            reasons.append("Independence was not assessed from study-design metadata.")
        if extreme_count:
            reasons.append("Extreme observations may influence one or more group means.")

        directly_compatible = (
            all(item.status is AssessmentStatus.PASS for item in group_shapes)
            and residual_shape is not None
            and residual_shape.status is AssessmentStatus.PASS
            and extreme_count == 0
            and (
                residual_outliers is None
                or residual_outliers.status is AssessmentStatus.PASS
            )
        )
        if directly_compatible:
            reasons.append(
                "Within-group residual diagnostics support direct one-way mean inference."
            )
            if min_group_n >= self.DIRECT_ACCEPTABLE_N and not independence_unknown:
                return RobustnessResult(RobustnessLevel.ACCEPTABLE, tuple(reasons))
            return RobustnessResult(RobustnessLevel.CAUTION, tuple(reasons))

        residual_abs_skew = (
            abs(float(residual_shape.metrics.get("skewness", np.inf)))
            if residual_shape is not None
            else np.inf
        )
        residual_abs_kurtosis = (
            abs(float(residual_shape.metrics.get("excess_kurtosis", np.inf)))
            if residual_shape is not None
            else np.inf
        )
        max_abs_skew = max(max_abs_skew, residual_abs_skew)
        max_abs_kurtosis = max(max_abs_kurtosis, residual_abs_kurtosis)

        if (
            min_group_n >= self.LARGE_GROUP_N
            and total_n >= self.LARGE_TOTAL_N
            and max_abs_skew <= self.LARGE_MAX_SKEW
            and max_abs_kurtosis <= self.LARGE_MAX_KURTOSIS
            and max_outlier_fraction <= self.LARGE_MAX_OUTLIER_FRACTION
        ):
            reasons.append(
                "Large groups and bounded residual departure support cautious Welch inference."
            )
            return RobustnessResult(RobustnessLevel.CAUTION, tuple(reasons))

        if (
            min_group_n >= self.MODERATE_GROUP_N
            and total_n >= self.MODERATE_TOTAL_N
            and max_abs_skew <= self.MODERATE_MAX_SKEW
            and max_abs_kurtosis <= self.MODERATE_MAX_KURTOSIS
            and max_outlier_fraction <= self.MODERATE_MAX_OUTLIER_FRACTION
        ):
            reasons.append(
                "Moderate group sizes and bounded residual departure support cautious Welch inference."
            )
            return RobustnessResult(RobustnessLevel.CAUTION, tuple(reasons))

        if (
            max_abs_skew <= self.SMALL_MAX_SKEW
            and max_abs_kurtosis <= self.SMALL_MAX_KURTOSIS
            and max_outlier_fraction <= self.SMALL_MAX_OUTLIER_FRACTION
        ):
            reasons.append(
                "Small-group residual departure remains inside the candidate cautious route."
            )
            return RobustnessResult(RobustnessLevel.CAUTION, tuple(reasons))

        reasons.append(
            "The candidate one-way sample-size, residual-shape and influence constraints are not all satisfied."
        )
        return RobustnessResult(RobustnessLevel.INSUFFICIENT, tuple(reasons))


__all__ = ["OneWayRobustness"]
