from typing import Optional, Tuple, Union

from pyMagicStat.assumptions.models import (
    AssessmentStatus,
    AssumptionReport,
    InferenceDesign,
)
from pyMagicStat.assumptions.robustness import (
    RobustnessLevel,
    RobustnessResult,
    SamplingRobustness,
)
from pyMagicStat.assumptions.robustness_v3 import (
    AssumptionProvenance,
    EmpiricalSupport,
    RobustnessResultV3,
    SamplingRobustnessV3,
)
from pyMagicStat.inference.capabilities import (
    INFERENCE_ROUTING_VERSION,
    InferenceCapability,
    InferenceGuarantee,
    capabilities_for,
    capability_for,
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
        robustness_policy: Optional[
            Union[SamplingRobustness, SamplingRobustnessV3]
        ] = None,
    ) -> None:
        self.robustness_policy = robustness_policy or SamplingRobustness()

    def select(
        self,
        report: AssumptionReport,
        *,
        equal_var: Optional[bool] = None,
    ) -> InferenceDecision:
        capabilities = capabilities_for(report.design, report.estimand)
        decision_metadata = {
            "capabilities": capabilities,
            "policy_version": getattr(
                self.robustness_policy,
                "POLICY_VERSION",
                None,
            ),
            "routing_version": INFERENCE_ROUTING_VERSION,
        }
        if report.design is InferenceDesign.ONE_WAY:
            reason = (
                "One-way inference is not calibrated or implemented in this release."
            )
            return InferenceDecision(
                selected_method=None,
                robustness=RobustnessResult(
                    RobustnessLevel.INSUFFICIENT,
                    (reason,),
                ),
                report=report,
                reasons=(reason,),
                alternatives=(),
                status=InferenceDecisionStatus.NOT_CALIBRATED,
                guarantee=InferenceGuarantee.NOT_CALIBRATED,
                **decision_metadata,
            )

        robustness = self.robustness_policy.evaluate(report)
        alternatives = self._alternatives(report.design)
        if isinstance(robustness, RobustnessResultV3):
            return self._select_v3(
                report,
                robustness,
                alternatives,
                capabilities,
                decision_metadata,
            )
        return self._select_v2(
            report,
            robustness,
            alternatives,
            equal_var,
            decision_metadata,
        )

    @staticmethod
    def _select_v2(
        report: AssumptionReport,
        robustness: RobustnessResult,
        alternatives: Tuple[MethodAlternative, ...],
        equal_var: Optional[bool],
        decision_metadata: dict[str, object],
    ) -> InferenceDecision:
        """Preserve the production v2 routing behavior exactly."""

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
                **decision_metadata,
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
            **decision_metadata,
        )

    @staticmethod
    def _select_v3(
        report: AssumptionReport,
        robustness: RobustnessResultV3,
        alternatives: Tuple[MethodAlternative, ...],
        capabilities: Tuple[InferenceCapability, ...],
        decision_metadata: dict[str, object],
    ) -> InferenceDecision:
        """Route v3 through explicit capabilities rather than shape fields."""

        reasons = list(robustness.reasons)
        if robustness.empirical_support is EmpiricalSupport.NOT_CALIBRATED:
            return InferenceDecision(
                selected_method=None,
                robustness=robustness,
                report=report,
                reasons=tuple(reasons),
                alternatives=alternatives,
                status=InferenceDecisionStatus.NOT_CALIBRATED,
                guarantee=InferenceGuarantee.NOT_CALIBRATED,
                **decision_metadata,
            )

        if robustness.level is RobustnessLevel.INSUFFICIENT:
            reasons.append(
                "A mean-preserving resampling or robust procedure should be considered."
            )
            return InferenceDecision(
                selected_method=None,
                robustness=robustness,
                report=report,
                reasons=tuple(reasons),
                alternatives=alternatives,
                status=InferenceDecisionStatus.INSUFFICIENT,
                guarantee=InferenceGuarantee.INSUFFICIENT,
                **decision_metadata,
            )

        if robustness.level is RobustnessLevel.CAUTION:
            reasons.append(
                "SamplingRobustnessV3 requires review before a method is selected."
            )
            return InferenceDecision(
                selected_method=None,
                robustness=robustness,
                report=report,
                reasons=tuple(reasons),
                alternatives=alternatives,
                status=InferenceDecisionStatus.REVIEW_REQUIRED,
                **decision_metadata,
            )

        supported = set(MethodSelector._supported_assumptions(report, robustness))
        one_sample_t = capability_for(
            "one_sample_t",
            report.design,
            report.estimand,
        )
        if (
            one_sample_t is not None
            and one_sample_t in capabilities
            and one_sample_t.calibrated
            and one_sample_t.automatic_selection_allowed
            and set(one_sample_t.assumptions_required) <= supported
        ):
            reasons.append(
                "The registered exact-parametric capability matches the available assumptions."
            )
            return InferenceDecision(
                selected_method=one_sample_t.method,
                robustness=robustness,
                report=report,
                reasons=tuple(reasons),
                alternatives=alternatives,
                status=InferenceDecisionStatus.SELECTED,
                guarantee=one_sample_t.guarantee,
                assumptions_used=one_sample_t.assumptions_required,
                **decision_metadata,
            )

        reasons.append(
            "No calibrated automatic inference capability matches the available guarantees."
        )
        return InferenceDecision(
            selected_method=None,
            robustness=robustness,
            report=report,
            reasons=tuple(reasons),
            alternatives=alternatives,
            status=InferenceDecisionStatus.REVIEW_REQUIRED,
            **decision_metadata,
        )

    @staticmethod
    def _supported_assumptions(
        report: AssumptionReport,
        robustness: RobustnessResultV3,
    ) -> Tuple[str, ...]:
        """Resolve structural/model evidence without inspecting sample shape."""

        supported: list[str] = []
        data_quality = [
            item
            for name, item in report.assessments.items()
            if name.startswith("data_quality")
        ]
        if data_quality and all(
            item.status is AssessmentStatus.PASS for item in data_quality
        ):
            supported.append("structural_data_supported")

        independence_supported = any(
            name.startswith("independence")
            and item.status is AssessmentStatus.PASS
            and item.metrics.get("independence") in {"assumed", "verified"}
            for name, item in report.assessments.items()
        )
        if independence_supported:
            supported.append("independence_supported")

        if robustness.model_support is AssumptionProvenance.EXTERNAL:
            supported.append("external_gaussian_model")
        return tuple(supported)

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
