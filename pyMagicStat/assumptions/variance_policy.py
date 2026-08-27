"""Population-model policy for exact chi-square variance inference."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from pyMagicStat.assumptions.models import AssessmentStatus, AssumptionReport


class PopulationNormality(str, Enum):
    """User-supplied status of the population model, not a sample-test result."""

    ASSUMED = "assumed"
    UNKNOWN = "unknown"
    NOT_ASSUMED = "not_assumed"


class VarianceInferenceLevel(str, Enum):
    SUPPORTED = "supported"
    CAUTION = "caution"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class VarianceInferenceResult:
    level: VarianceInferenceLevel
    population_normality: PopulationNormality
    reasons: Tuple[str, ...]

    @property
    def chi_square_validated(self) -> bool:
        return self.level is not VarianceInferenceLevel.UNSUPPORTED

    def to_dict(self):
        return {
            "level": self.level.value,
            "population_normality": self.population_normality.value,
            "chi_square_validated": self.chi_square_validated,
            "reasons": list(self.reasons),
        }


class VarianceInferencePolicy:
    """Validate the population-normality contract of the exact interval.

    The chi-square pivot is exact only for an independent sample from a normal
    population.  A sample shape diagnostic can challenge that model, but cannot
    establish it, and increasing ``n`` does not invoke a CLT for this pivot.
    """

    def evaluate(
        self,
        report: AssumptionReport,
        population_normality: PopulationNormality | str,
    ) -> VarianceInferenceResult:
        try:
            normality = PopulationNormality(population_normality)
        except ValueError as exc:
            choices = ", ".join(item.value for item in PopulationNormality)
            raise ValueError(f"population_normality must be one of: {choices}") from exc

        if normality is PopulationNormality.UNKNOWN:
            return VarianceInferenceResult(
                VarianceInferenceLevel.UNSUPPORTED,
                normality,
                (
                    "Exact chi-square variance inference requires an explicit normal-population assumption.",
                    "Sample normality diagnostics cannot establish population normality.",
                ),
            )
        if normality is PopulationNormality.NOT_ASSUMED:
            return VarianceInferenceResult(
                VarianceInferenceLevel.UNSUPPORTED,
                normality,
                ("The stated population model does not support the exact chi-square pivot.",),
            )

        shape = report.assessments.get("shape")
        if shape is None:
            return VarianceInferenceResult(
                VarianceInferenceLevel.CAUTION,
                normality,
                ("Population normality is assumed, but no sample shape diagnostic is available.",),
            )
        if shape.status is AssessmentStatus.FAIL:
            return VarianceInferenceResult(
                VarianceInferenceLevel.UNSUPPORTED,
                normality,
                (
                    "The observed shape strongly conflicts with the assumed normal population.",
                    "Consider a bootstrap variance interval with an explicit variance estimand.",
                ),
            )
        if shape.status is AssessmentStatus.WARN:
            return VarianceInferenceResult(
                VarianceInferenceLevel.CAUTION,
                normality,
                ("Population normality is assumed, but the sample shape raises a diagnostic warning.",),
            )
        return VarianceInferenceResult(
            VarianceInferenceLevel.SUPPORTED,
            normality,
            ("Population normality is explicitly assumed and is not contradicted by the sample diagnostic.",),
        )
