"""Reusable diagnostics for statistical inference."""

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
from pyMagicStat.assumptions.robustness import (
    RobustnessLevel,
    RobustnessResult,
    SamplingRobustness,
)
from pyMagicStat.assumptions.anova_robustness import OneWayRobustness
from pyMagicStat.assumptions.validator import InferenceValidator
from pyMagicStat.assumptions.variance_policy import (
    PopulationNormality,
    VarianceInferenceLevel,
    VarianceInferencePolicy,
    VarianceInferenceResult,
)

__all__ = [
    "Assessment",
    "AssessmentStatus",
    "AssumptionReport",
    "DataQualityAssessment",
    "Estimand",
    "IndependenceAssessment",
    "InferenceDesign",
    "InferenceValidator",
    "OutlierAssessment",
    "OneWayRobustness",
    "PopulationNormality",
    "RobustnessLevel",
    "RobustnessResult",
    "SamplingRobustness",
    "ShapeAssessment",
    "ValidationResult",
    "VarianceAssessment",
    "VarianceInferenceLevel",
    "VarianceInferencePolicy",
    "VarianceInferenceResult",
]
