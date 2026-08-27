from typing import Optional, Tuple

from pyMagicStat.assumptions.models import AssumptionReport, InferenceDesign
from pyMagicStat.assumptions.robustness import (
    RobustnessLevel,
    SamplingRobustness,
)
from pyMagicStat.inference.decision import InferenceDecision, MethodAlternative


class MethodSelector:
    """Select an inferential procedure without changing observed data."""

    def __init__(self, robustness_policy: Optional[SamplingRobustness] = None) -> None:
        self.robustness_policy = robustness_policy or SamplingRobustness()

    def select(
        self,
        report: AssumptionReport,
        *,
        equal_var: Optional[bool] = None,
    ) -> InferenceDecision:
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
            )

        if report.design is InferenceDesign.ONE_SAMPLE:
            method = "one_sample_t"
        elif report.design is InferenceDesign.PAIRED:
            method = "paired_t"
        elif report.design in {InferenceDesign.TWO_SAMPLE, InferenceDesign.ONE_WAY}:
            if report.design is InferenceDesign.ONE_WAY:
                method = "welch_anova"
            elif equal_var is True:
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
