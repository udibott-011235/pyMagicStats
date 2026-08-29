import numpy as np
import pytest

import pyMagicStat.inference.capabilities as capability_module
from pyMagicStat.assumptions import (
    Assessment,
    AssessmentStatus,
    AssumptionProvenance,
    AssumptionReport,
    Estimand,
    InferenceDesign,
    InferenceValidator,
    ProcessUncertainty,
    RobustnessLevel,
    SamplingRobustness,
    SamplingRobustnessV3,
)
from pyMagicStat.inference import (
    INFERENCE_ROUTING_VERSION,
    InferenceDecisionStatus,
    InferenceGuarantee,
    MethodSelector,
    capabilities_for,
)


def _v3_policy(provenance):
    return SamplingRobustnessV3(
        model_provenance=provenance,
        process_uncertainty=ProcessUncertainty.LOW,
    )


def _one_sample_report(*, independence="assumed"):
    return InferenceValidator().validate_one_sample(
        np.linspace(-1.0, 1.0, 101),
        independence=independence,
    ).report


def _manual_shape_report(*, skewness, kurtosis, exact_rejected=False):
    return AssumptionReport(
        design=InferenceDesign.ONE_SAMPLE,
        estimand=Estimand.MEAN,
        assessments={
            "data_quality": Assessment(
                "data_quality_sample",
                AssessmentStatus.PASS,
                {"n": 10_000},
            ),
            "shape": Assessment(
                "shape_sample",
                AssessmentStatus.PASS,
                {
                    "n": 10_000,
                    "skewness": skewness,
                    "excess_kurtosis": kurtosis,
                    "exact_normality_rejected": exact_rejected,
                },
            ),
            "outliers": Assessment(
                "outliers_sample",
                AssessmentStatus.PASS,
                {
                    "count": 0,
                    "fraction": 0.0,
                    "influence_ratio": 0.0,
                },
            ),
            "independence": Assessment(
                "independence",
                AssessmentStatus.PASS,
                {"independence": "assumed"},
            ),
        },
    )


def test_default_selector_still_uses_legacy_v2_policy():
    selector = MethodSelector()

    assert isinstance(selector.robustness_policy, SamplingRobustness)
    assert selector.robustness_policy.POLICY_VERSION == "mean-v2.1-2026-08"


def test_v2_one_sample_caution_preserves_automatic_selection():
    report = _one_sample_report(independence="unknown")

    decision = MethodSelector().select(report)

    assert decision.robustness.level is RobustnessLevel.CAUTION
    assert decision.selected_method == "one_sample_t"
    assert decision.status is InferenceDecisionStatus.SELECTED


def test_v3_caution_remains_review_required_without_a_method():
    decision = MethodSelector(SamplingRobustnessV3()).select(
        _one_sample_report()
    )

    assert decision.robustness.level is RobustnessLevel.CAUTION
    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.REVIEW_REQUIRED


@pytest.mark.parametrize(
    ("design", "estimand"),
    [
        (InferenceDesign.PAIRED, Estimand.MEAN_DIFFERENCE),
        (InferenceDesign.TWO_SAMPLE, Estimand.MEAN_DIFFERENCE),
    ],
)
def test_v3_out_of_domain_designs_remain_not_calibrated(design, estimand):
    source = _one_sample_report()
    report = AssumptionReport(
        design=design,
        estimand=estimand,
        assessments=source.assessments,
    )

    decision = MethodSelector(
        _v3_policy(AssumptionProvenance.EXTERNAL)
    ).select(report)

    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.NOT_CALIBRATED
    assert decision.guarantee is InferenceGuarantee.NOT_CALIBRATED


def test_missing_independence_cannot_select_a_v3_method():
    source = _one_sample_report()
    report = AssumptionReport(
        design=source.design,
        estimand=source.estimand,
        assessments={
            name: assessment
            for name, assessment in source.assessments.items()
            if not name.startswith("independence")
        },
    )

    decision = MethodSelector(
        _v3_policy(AssumptionProvenance.EXTERNAL)
    ).select(report)

    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.REVIEW_REQUIRED
    assert decision.guarantee is None


def test_external_gaussian_support_exposes_exact_parametric_t_capability():
    decision = MethodSelector(
        _v3_policy(AssumptionProvenance.EXTERNAL)
    ).select(_one_sample_report())
    serialized = decision.to_dict()

    assert decision.selected_method == "one_sample_t"
    assert decision.guarantee is InferenceGuarantee.EXACT_PARAMETRIC
    assert decision.assumptions_used == (
        "structural_data_supported",
        "independence_supported",
        "external_gaussian_model",
    )
    assert decision.estimand is Estimand.MEAN
    assert decision.design is InferenceDesign.ONE_SAMPLE
    assert decision.routing_version == INFERENCE_ROUTING_VERSION
    assert decision.policy_version == SamplingRobustnessV3.POLICY_VERSION
    assert serialized["estimand"] == "mean"
    assert serialized["design"] == "one_sample"
    assert serialized["guarantee"] == "exact_parametric"
    assert {
        item["method"] for item in serialized["capabilities"]
    } >= {
        "one_sample_t",
        "empirical_likelihood",
        "bartlett_empirical_likelihood",
        "bootstrap_t",
    }


def test_empirical_provenance_cannot_masquerade_as_exact_parametric():
    decision = MethodSelector(
        _v3_policy(AssumptionProvenance.EMPIRICAL)
    ).select(_one_sample_report())

    assert decision.robustness.level is RobustnessLevel.ACCEPTABLE
    assert decision.selected_method is None
    assert decision.guarantee is not InferenceGuarantee.EXACT_PARAMETRIC
    assert decision.status is InferenceDecisionStatus.REVIEW_REQUIRED


def test_unknown_provenance_cannot_masquerade_as_exact_parametric():
    decision = MethodSelector(SamplingRobustnessV3()).select(
        _one_sample_report()
    )

    assert decision.selected_method is None
    assert decision.guarantee is not InferenceGuarantee.EXACT_PARAMETRIC


def test_empirical_likelihood_is_registered_as_nonautomatic_candidate():
    capability = {
        item.method: item
        for item in capabilities_for(
            InferenceDesign.ONE_SAMPLE,
            Estimand.MEAN,
        )
    }["empirical_likelihood"]

    assert capability.guarantee is InferenceGuarantee.ASYMPTOTIC_MOMENT_BASED
    assert capability.calibrated is False
    assert capability.automatic_selection_allowed is False


def test_bartlett_empirical_likelihood_is_nonautomatic_candidate():
    capability = {
        item.method: item
        for item in capabilities_for(
            InferenceDesign.ONE_SAMPLE,
            Estimand.MEAN,
        )
    }["bartlett_empirical_likelihood"]

    assert capability.guarantee is InferenceGuarantee.HIGHER_ORDER_CORRECTED
    assert capability.calibrated is False
    assert capability.automatic_selection_allowed is False


def test_bootstrap_t_is_registered_but_never_an_automatic_fallback():
    capabilities = {
        item.method: item
        for item in capabilities_for(
            InferenceDesign.ONE_SAMPLE,
            Estimand.MEAN,
        )
    }
    decision = MethodSelector(
        _v3_policy(AssumptionProvenance.EMPIRICAL)
    ).select(_one_sample_report())

    assert capabilities["bootstrap_t"].guarantee is InferenceGuarantee.RESAMPLING_BASED
    assert capabilities["bootstrap_t"].calibrated is False
    assert capabilities["bootstrap_t"].automatic_selection_allowed is False
    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.REVIEW_REQUIRED


def test_shape_diagnostic_changes_do_not_define_method_identity_in_selector():
    selector = MethodSelector(
        _v3_policy(AssumptionProvenance.EXTERNAL)
    )
    baseline = selector.select(
        _manual_shape_report(
            skewness=0.0,
            kurtosis=0.0,
            exact_rejected=False,
        )
    )
    changed = selector.select(
        _manual_shape_report(
            skewness=0.5,
            kurtosis=0.5,
            exact_rejected=True,
        )
    )

    assert baseline.selected_method == changed.selected_method == "one_sample_t"
    assert baseline.guarantee is changed.guarantee is InferenceGuarantee.EXACT_PARAMETRIC


def test_capability_routing_defines_no_minimum_sample_size_constant():
    assert not hasattr(capability_module, "MIN_N")
    assert not hasattr(MethodSelector, "MIN_N")
