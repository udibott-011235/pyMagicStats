from typing import Optional, Tuple

from pyMagicStat.assumptions.models import AssumptionReport, InferenceDesign
from pyMagicStat.assumptions.models import AssessmentStatus
from pyMagicStat.assumptions.anova_robustness import OneWayRobustness
from pyMagicStat.assumptions.robustness import (
    RobustnessLevel,
    RobustnessResult,
    SamplingRobustness,
)
from pyMagicStat.inference.decision import (
    InferenceDecision,
    InferenceDecisionStatus,
    MethodAlternative,
)


class MethodSelector:
    """Select an inferential procedure without changing observed data."""

    def __init__(
        self,
        robustness_policy: Optional[SamplingRobustness] = None,
        one_way_policy: Optional[OneWayRobustness] = None,
    ) -> None:
        self.robustness_policy = robustness_policy or SamplingRobustness()
        self.one_way_policy = one_way_policy or OneWayRobustness()

    def select(
        self,
        report: AssumptionReport,
        *,
        equal_var: Optional[bool] = None,
    ) -> InferenceDecision:
        if report.design is InferenceDesign.ONE_WAY:
            return self._select_one_way(report, equal_var=equal_var)

        robustness = self.robustness_policy.evaluate(report)
        alternatives = self._alternatives(report.design)
        reasons = list(robustness.reasons)

        if robustness.level is RobustnessLevel.INSUFFICIENT:
            reasons.append("A mean-preserving resampling or robust procedure should be considered.")
            return InferenceDecision(
                selected_method=None,
                robustness=robustness,
                report=report,
                reasons=tuple(reasons),
                alternatives=alternatives,
                status=InferenceDecisionStatus.INSUFFICIENT,
            )

        if report.design is InferenceDesign.ONE_SAMPLE:
            method = "one_sample_t"
        elif report.design is InferenceDesign.PAIRED:
            method = "paired_t"
        elif report.design is InferenceDesign.TWO_SAMPLE:
            if equal_var is True:
                method = "student_t"
                reasons.append("Equal-variance Student inference was explicitly requested.")
            else:
                method = "welch_t"
                reasons.append("Welch inference is the variance-robust default.")
        else:
            method = None

        return InferenceDecision(
            selected_method=method,
            robustness=robustness,
            report=report,
            reasons=tuple(reasons),
            alternatives=alternatives,
            status=InferenceDecisionStatus.SELECTED,
        )

    def _select_one_way(
        self,
        report: AssumptionReport,
        *,
        equal_var: Optional[bool],
    ) -> InferenceDecision:
        robustness = self.one_way_policy.evaluate(report)
        alternatives = self._alternatives(InferenceDesign.ONE_WAY)
        reasons = list(robustness.reasons)
        if robustness.level is RobustnessLevel.INSUFFICIENT:
            reasons.append(
                "One-way mean inference is outside the calibrated ANOVA policy."
            )
            return InferenceDecision(
                selected_method=None,
                robustness=robustness,
                report=report,
                reasons=tuple(reasons),
                alternatives=alternatives,
                status=InferenceDecisionStatus.INSUFFICIENT,
            )

        variance = report.assessments.get("variance")
        if variance is None:
            reason = "A variance assessment is required for one-way method selection."
            return InferenceDecision(
                selected_method=None,
                robustness=RobustnessResult(
                    RobustnessLevel.INSUFFICIENT,
                    tuple(reasons + [reason]),
                ),
                report=report,
                reasons=tuple(reasons + [reason]),
                alternatives=alternatives,
                status=InferenceDecisionStatus.INSUFFICIENT,
            )

        if equal_var is True:
            variance_ratio = float(variance.metrics.get("variance_ratio", float("inf")))
            common_variance_unsupported = (
                variance.status is AssessmentStatus.WARN
                or variance_ratio > 4.0
                or bool(variance.metrics.get("small_group_large_variance", False))
            )
            if common_variance_unsupported:
                reason = (
                    "Classical ANOVA was requested, but magnitude and robust variance "
                    "diagnostics do not support a common-variance model."
                )
                method_robustness = RobustnessResult(
                    RobustnessLevel.INSUFFICIENT,
                    tuple(reasons + [reason]),
                )
                return InferenceDecision(
                    selected_method=None,
                    robustness=method_robustness,
                    report=report,
                    reasons=tuple(reasons + [reason]),
                    alternatives=alternatives,
                    status=InferenceDecisionStatus.INSUFFICIENT,
                )
            method = "classical_anova"
            reasons.append(
                "Classical ANOVA was explicitly requested and the common-variance "
                "diagnostics are compatible."
            )
        else:
            method = "welch_anova"
            reasons.append(
                "Welch ANOVA is the calibrated variance-robust default."
                if equal_var is None
                else "Welch ANOVA was explicitly requested."
            )

        return InferenceDecision(
            selected_method=method,
            robustness=robustness,
            report=report,
            reasons=tuple(reasons),
            alternatives=alternatives,
            status=InferenceDecisionStatus.SELECTED,
        )

    @staticmethod
    def _alternatives(design: InferenceDesign) -> Tuple[MethodAlternative, ...]:
        if design is InferenceDesign.ONE_SAMPLE:
            return (
                MethodAlternative(
                    "bootstrap_bca_mean_ci",
                    "mean",
                    "Preserves inference about the population mean.",
                ),
                MethodAlternative(
                    "wilcoxon_signed_rank",
                    "symmetric_location",
                    "Targets a rank-based location hypothesis, not the arithmetic mean.",
                ),
            )
        if design is InferenceDesign.PAIRED:
            return (
                MethodAlternative(
                    "bootstrap_bca_mean_difference_ci",
                    "mean_difference",
                    "Resample complete pairs or their differences.",
                ),
                MethodAlternative(
                    "wilcoxon_signed_rank",
                    "symmetric_difference_location",
                    "Requires a different location interpretation.",
                ),
            )
        if design is InferenceDesign.TWO_SAMPLE:
            return (
                MethodAlternative(
                    "bootstrap_bca_mean_difference_ci",
                    "mean_difference",
                    "Resample independently within each group.",
                ),
                MethodAlternative(
                    "permutation_mean_difference",
                    "mean_difference_under_exchangeability",
                    "Requires exchangeability under the null hypothesis.",
                ),
                MethodAlternative(
                    "mann_whitney_u",
                    "probabilistic_ordering",
                    "Does not generally test equality of arithmetic means.",
                ),
            )
        return (
            MethodAlternative(
                "bootstrap_group_contrast",
                "specified_group_contrast",
                "The resampling scheme must match the study design.",
            ),
            MethodAlternative(
                "kruskal_wallis",
                "rank_distribution_difference",
                "Not a drop-in test of equality of group means.",
            ),
        )
