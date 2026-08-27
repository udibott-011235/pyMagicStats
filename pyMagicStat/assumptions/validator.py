from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

from pyMagicStat.assumptions.assessments import (
    DataQualityAssessment,
    IndependenceAssessment,
    OutlierAssessment,
    ShapeAssessment,
    VarianceAssessment,
)
from pyMagicStat.assumptions.models import (
    Assessment,
    AssessmentStatus,
    AssumptionReport,
    Estimand,
    InferenceDesign,
    ValidationResult,
)


class InferenceValidator:
    """Build method-aware diagnostics from raw samples."""

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)
        self.data_quality = DataQualityAssessment()
        self.shape = ShapeAssessment(alpha=alpha)
        self.outliers = OutlierAssessment()
        self.variance = VarianceAssessment(alpha=alpha)
        self.independence = IndependenceAssessment()

    def validate_one_sample(
        self,
        data: Any,
        *,
        estimand: Estimand = Estimand.MEAN,
        independence: str = "unknown",
    ) -> ValidationResult:
        sample, quality = self.data_quality.normalize(data)
        self._raise_for_quality_failures((quality,))
        assessments: Dict[str, Assessment] = {
            "data_quality": quality,
            "shape": self.shape.assess(sample),
            "outliers": self.outliers.assess(sample),
            "independence": self.independence.assess(independence),
        }
        report = AssumptionReport(InferenceDesign.ONE_SAMPLE, estimand, assessments)
        return ValidationResult((sample,), (sample,), report)

    def validate_paired(
        self,
        data1: Any,
        data2: Any,
        *,
        independence: str = "unknown",
    ) -> ValidationResult:
        sample1, quality1 = self.data_quality.normalize(data1, "group_1")
        sample2, quality2 = self.data_quality.normalize(data2, "group_2")
        self._raise_for_quality_failures((quality1, quality2))
        if sample1.size != sample2.size:
            raise ValueError("Paired samples must have the same length")
        differences = sample1 - sample2
        difference_quality = self.data_quality.normalize(differences, "differences")[1]
        self._raise_for_quality_failures((difference_quality,))
        assessments: Dict[str, Assessment] = {
            "data_quality_group_1": quality1,
            "data_quality_group_2": quality2,
            "data_quality_differences": difference_quality,
            "shape_differences": self.shape.assess(differences, "differences"),
            "outliers_differences": self.outliers.assess(differences, "differences"),
            "independence_of_pairs": self.independence.assess(independence),
        }
        report = AssumptionReport(
            InferenceDesign.PAIRED,
            Estimand.MEAN_DIFFERENCE,
            assessments,
        )
        return ValidationResult((sample1, sample2), (differences,), report)

    def validate_two_sample(
        self,
        data1: Any,
        data2: Any,
        *,
        independence: str = "unknown",
    ) -> ValidationResult:
        sample1, quality1 = self.data_quality.normalize(data1, "group_1")
        sample2, quality2 = self.data_quality.normalize(data2, "group_2")
        self._raise_for_quality_failures((quality1, quality2))
        centered1 = sample1 - np.mean(sample1)
        centered2 = sample2 - np.mean(sample2)
        assessments: Dict[str, Assessment] = {
            "data_quality_group_1": quality1,
            "data_quality_group_2": quality2,
            "shape_group_1": self.shape.assess(centered1, "group_1"),
            "shape_group_2": self.shape.assess(centered2, "group_2"),
            "outliers_group_1": self.outliers.assess(centered1, "group_1"),
            "outliers_group_2": self.outliers.assess(centered2, "group_2"),
            "variance": self.variance.assess((sample1, sample2)),
            "independence": self.independence.assess(independence),
        }
        report = AssumptionReport(
            InferenceDesign.TWO_SAMPLE,
            Estimand.MEAN_DIFFERENCE,
            assessments,
        )
        return ValidationResult((sample1, sample2), (centered1, centered2), report)

    @staticmethod
    def _raise_for_quality_failures(assessments: Iterable[Assessment]) -> None:
        reasons = [
            reason
            for assessment in assessments
            if assessment.status is AssessmentStatus.FAIL
            for reason in assessment.reasons
        ]
        if reasons:
            raise ValueError("; ".join(reasons))
