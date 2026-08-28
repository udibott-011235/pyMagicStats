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

    def validate_one_way(
        self,
        *groups: Any,
        independence: str = "unknown",
    ) -> ValidationResult:
        if len(groups) < 3:
            raise ValueError("One-way inference requires at least three groups")

        normalized = []
        assessments: Dict[str, Assessment] = {}
        qualities = []
        for index, group in enumerate(groups, start=1):
            label = f"group_{index}"
            sample, quality = self.data_quality.normalize(group, label)
            normalized.append(sample)
            qualities.append(quality)
            assessments[f"data_quality_{label}"] = quality
        self._raise_for_quality_failures(qualities)

        centered = tuple(sample - np.mean(sample) for sample in normalized)
        for index, residuals in enumerate(centered, start=1):
            label = f"group_{index}"
            assessments[f"shape_{label}"] = self.shape.assess(residuals, label)
            assessments[f"outliers_{label}"] = self.outliers.assess(residuals, label)
        standardized_residuals = np.concatenate(
            [
                residuals / np.std(sample, ddof=1)
                for sample, residuals in zip(normalized, centered)
            ]
        )
        assessments["shape_standardized_residuals"] = self.shape.assess(
            standardized_residuals,
            "standardized_residuals",
        )
        assessments["outliers_standardized_residuals"] = self.outliers.assess(
            standardized_residuals,
            "standardized_residuals",
        )
        assessments["variance"] = self.variance.assess(tuple(normalized))
        assessments["independence"] = self.independence.assess(independence)

        report = AssumptionReport(
            InferenceDesign.ONE_WAY,
            Estimand.GROUP_MEAN_DIFFERENCES,
            assessments,
        )
        return ValidationResult(tuple(normalized), centered, report)

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
