import numpy as np
import pandas as pd
import pytest

from experiments.sampling_robustness_v3_calibration import (
    V3_DEFAULT_PROFILE,
    V3_ORACLE_PROFILE,
    flag_operating_regions,
    run,
    summarize_false_safe_metrics,
)
from pyMagicStat.assumptions import (
    Assessment,
    AssessmentStatus,
    AssumptionProvenance,
    AssumptionReport,
    EmpiricalSupport,
    Estimand,
    InferenceDesign,
    InferenceValidator,
    InfluenceRisk,
    OutlierAssessment,
    ProcessUncertainty,
    RobustnessLevel,
    SamplingRobustness,
    SamplingRobustnessV3,
)
from pyMagicStat.inference import InferenceDecisionStatus, MethodSelector


def _policy(
    provenance=AssumptionProvenance.EXTERNAL,
    process=ProcessUncertainty.LOW,
):
    return SamplingRobustnessV3(
        model_provenance=provenance,
        process_uncertainty=process,
    )


def _report(
    *,
    n=100,
    skewness=0.0,
    kurtosis=0.0,
    exact_rejected=False,
    extreme_count=0,
    extreme_fraction=0.0,
    influence_ratio=0.0,
    independence=AssessmentStatus.PASS,
    independence_metric=None,
    quality=AssessmentStatus.PASS,
    design=InferenceDesign.ONE_SAMPLE,
    estimand=Estimand.MEAN,
):
    if abs(skewness) > 2.0 or abs(kurtosis) > 7.0:
        shape_status, magnitude = AssessmentStatus.FAIL, "severe"
    elif abs(skewness) > 1.0 or abs(kurtosis) > 3.0:
        shape_status, magnitude = AssessmentStatus.WARN, "moderate"
    else:
        shape_status, magnitude = AssessmentStatus.PASS, "mild"
    assessments = {
        "data_quality": Assessment(
            "data_quality_sample",
            quality,
            {"n": n},
        ),
        "shape": Assessment(
            "shape_sample",
            shape_status,
            {
                "n": n,
                "skewness": skewness,
                "excess_kurtosis": kurtosis,
                "departure_magnitude": magnitude,
                "exact_normality_rejected": exact_rejected,
                "shapiro_p_value": 0.001 if exact_rejected else 0.5,
            },
        ),
        "outliers": Assessment(
            "outliers_sample",
            AssessmentStatus.WARN if extreme_count else AssessmentStatus.PASS,
            {
                "count": extreme_count,
                "fraction": extreme_fraction,
                "influence_ratio": influence_ratio,
            },
        ),
    }
    if independence is not None:
        metric = independence_metric
        if metric is None:
            metric = (
                "unknown"
                if independence is AssessmentStatus.NOT_ASSESSED
                else "assumed"
            )
        assessments["independence"] = Assessment(
            "independence",
            independence,
            {"independence": metric},
        )
    return AssumptionReport(
        design=design,
        estimand=estimand,
        assessments=assessments,
    )


def test_external_model_support_allows_exact_small_normal_without_minimum_n():
    report = InferenceValidator().validate_one_sample(
        np.array([-1.0, 0.0, 1.0]),
        independence="assumed",
    ).report

    result = _policy().evaluate(report)

    assert result.level is RobustnessLevel.ACCEPTABLE
    assert result.model_support is AssumptionProvenance.EXTERNAL
    assert result.empirical_support is EmpiricalSupport.LIMITED
    assert result.diagnostics["n"] == 3


def test_unknown_small_sample_distinguishes_validity_from_limited_evidence():
    report = InferenceValidator().validate_one_sample(
        np.array([-1.0, 0.0, 1.0]),
        independence="assumed",
    ).report

    result = SamplingRobustnessV3().evaluate(report)

    assert result.level is RobustnessLevel.CAUTION
    assert result.model_support is AssumptionProvenance.UNKNOWN
    assert result.empirical_support is EmpiricalSupport.LIMITED
    assert any("provenance is unknown" in reason for reason in result.reasons)


def test_formal_normality_rejection_is_descriptive_and_not_a_veto():
    result = _policy().evaluate(_report(exact_rejected=True))

    assert result.level is RobustnessLevel.ACCEPTABLE
    assert result.diagnostics["exact_normality_rejected_descriptive_only"] is True
    assert "p_value" not in " ".join(result.diagnostics)


def test_symmetric_bimodal_and_student_t_are_not_automatic_failures():
    rng = np.random.default_rng(44)
    bimodal = np.concatenate((rng.normal(-2.0, 1.0, 150), rng.normal(2.0, 1.0, 150)))
    bimodal_report = InferenceValidator().validate_one_sample(
        bimodal,
        independence="assumed",
    ).report
    student_report = _report(n=20, skewness=0.5, kurtosis=1.5, exact_rejected=True)

    bimodal_result = _policy(AssumptionProvenance.EMPIRICAL).evaluate(bimodal_report)
    student_result = _policy(AssumptionProvenance.EMPIRICAL).evaluate(student_report)

    assert bimodal_report.assessments["shape"].metrics["exact_normality_rejected"] is True
    assert bimodal_result.level is not RobustnessLevel.INSUFFICIENT
    assert student_result.level is not RobustnessLevel.INSUFFICIENT


def test_strong_joint_shape_is_insufficient_while_process_risk_alone_is_caution():
    observable = _policy(
        AssumptionProvenance.EMPIRICAL,
        ProcessUncertainty.UNKNOWN,
    ).evaluate(_report(n=100, skewness=3.0, kurtosis=10.0))
    contextual = _policy(process=ProcessUncertainty.ELEVATED).evaluate(_report())

    assert observable.level is RobustnessLevel.INSUFFICIENT
    assert observable.empirical_support is EmpiricalSupport.ADVERSE
    assert contextual.level is RobustnessLevel.CAUTION


def test_large_normal_extremes_do_not_degrade_only_because_they_exist():
    result = _policy().evaluate(
        _report(
            n=10000,
            skewness=0.02,
            kurtosis=0.05,
            extreme_count=5,
            extreme_fraction=0.0005,
            influence_ratio=0.05,
        )
    )

    assert result.level is RobustnessLevel.ACCEPTABLE
    assert result.influence is InfluenceRisk.LOW
    assert result.diagnostics["extreme_count"] == 5
    assert result.diagnostics["influence_ratio"] == 0.05


def test_outlier_assessment_separates_extremeness_from_counterfactual_influence():
    data = np.array([-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0, 9.0])
    original = data.copy()

    result = OutlierAssessment().assess(data)

    assert result.metrics["count"] == 1
    assert result.metrics["fraction"] == 0.1
    assert result.metrics["influence_ratio"] > 0.0
    assert result.metrics["influence_is_counterfactual"] is True
    np.testing.assert_array_equal(data, original)


def test_elevated_influence_moves_to_caution_but_never_removes_data_or_forces_failure():
    result = _policy().evaluate(
        _report(extreme_count=1, extreme_fraction=0.01, influence_ratio=2.0)
    )

    assert result.level is RobustnessLevel.CAUTION
    assert result.influence is InfluenceRisk.ELEVATED
    assert any("does not authorize removing" in reason for reason in result.reasons)


def test_dense_perturbations_cannot_jump_directly_from_acceptable_to_insufficient():
    policy = _policy()
    order = {
        RobustnessLevel.ACCEPTABLE: 0,
        RobustnessLevel.CAUTION: 1,
        RobustnessLevel.INSUFFICIENT: 2,
    }
    levels = []
    scores = []
    for skewness in np.linspace(0.60, 2.00, 1401):
        result = policy.evaluate(_report(n=10000, skewness=float(skewness)))
        levels.append(order[result.level])
        scores.append(result.diagnostics["shape_risk_score"])

    assert np.all(np.diff(scores) >= 0.0)
    assert np.max(np.abs(np.diff(levels))) <= 1


def test_old_thresholds_no_longer_create_special_discontinuities():
    policy = _policy()
    shape_cases = (
        ("skewness", 1.0, {"kurtosis": 0.0}),
        ("skewness", 2.0, {"kurtosis": 5.0}),
        ("kurtosis", 3.0, {"skewness": 2.5}),
        ("kurtosis", 7.0, {"skewness": 2.5}),
    )
    for parameter, boundary, fixed in shape_cases:
        below_values = {parameter: boundary - 1e-9, **fixed}
        above_values = {parameter: boundary + 1e-9, **fixed}
        below = policy.evaluate(_report(n=10000, **below_values))
        above = policy.evaluate(_report(n=10000, **above_values))
        assert below.level is above.level
        assert abs(
            below.diagnostics["shape_risk_score"]
            - above.diagnostics["shape_risk_score"]
        ) < 1e-8

    for boundary in (0.025, 0.10):
        below = policy.evaluate(
            _report(
                n=10000,
                extreme_count=1,
                extreme_fraction=boundary - 1e-9,
                influence_ratio=0.0,
            )
        )
        above = policy.evaluate(
            _report(
                n=10000,
                extreme_count=1,
                extreme_fraction=boundary + 1e-9,
                influence_ratio=0.0,
            )
        )
        assert below.level is above.level is RobustnessLevel.ACCEPTABLE

    empirical_policy = _policy(AssumptionProvenance.EMPIRICAL)
    order = {
        RobustnessLevel.ACCEPTABLE: 0,
        RobustnessLevel.CAUTION: 1,
        RobustnessLevel.INSUFFICIENT: 2,
    }
    for old_n in (40, 80, 200):
        levels = [
            order[empirical_policy.evaluate(_report(n=n)).level]
            for n in range(old_n - 2, old_n + 3)
        ]
        assert max(np.abs(np.diff(levels)), default=0) <= 1


def test_structural_failure_is_insufficient():
    result = SamplingRobustnessV3().evaluate(_report(quality=AssessmentStatus.FAIL))

    assert result.level is RobustnessLevel.INSUFFICIENT


def test_v2_legacy_policy_and_method_selector_default_are_unchanged():
    legacy = SamplingRobustness()
    selector = MethodSelector()

    assert legacy.POLICY_VERSION == "mean-v2.1-2026-08"
    assert isinstance(selector.robustness_policy, SamplingRobustness)
    assert set(legacy.evaluate(_report()).to_dict()) == {"level", "reasons"}


def test_v2_caution_preserves_legacy_automatic_t_selection():
    decision = MethodSelector().select(
        _report(independence=AssessmentStatus.NOT_ASSESSED)
    )

    assert decision.robustness.level is RobustnessLevel.CAUTION
    assert decision.selected_method == "one_sample_t"
    assert decision.status is InferenceDecisionStatus.SELECTED


def test_method_selector_accepts_injected_v3_and_serializes_rich_result():
    selector = MethodSelector(_policy())
    decision = selector.select(_report())
    serialized = decision.to_dict()

    assert decision.selected_method == "one_sample_t"
    assert serialized["robustness"]["policy_version"] == "mean-v3-candidate-2026-08"
    assert serialized["robustness"]["model_support"] == "external"


def test_v3_caution_requires_review_and_does_not_select_a_method():
    decision = MethodSelector(SamplingRobustnessV3()).select(_report())

    assert decision.robustness.level is RobustnessLevel.CAUTION
    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.REVIEW_REQUIRED
    assert decision.alternatives


def test_v3_insufficient_does_not_select_a_method():
    decision = MethodSelector(
        _policy(AssumptionProvenance.EMPIRICAL)
    ).select(_report(n=100, skewness=3.0, kurtosis=10.0))

    assert decision.robustness.level is RobustnessLevel.INSUFFICIENT
    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.INSUFFICIENT


def test_v3_not_calibrated_estimand_does_not_select_a_method():
    decision = MethodSelector(_policy()).select(
        _report(estimand=Estimand.VARIANCE)
    )

    assert decision.robustness.empirical_support is EmpiricalSupport.NOT_CALIBRATED
    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.NOT_CALIBRATED


def test_v3_paired_design_is_not_calibrated_and_never_selects_paired_t():
    decision = MethodSelector(_policy()).select(
        _report(
            design=InferenceDesign.PAIRED,
            estimand=Estimand.MEAN_DIFFERENCE,
        )
    )

    assert decision.robustness.empirical_support is EmpiricalSupport.NOT_CALIBRATED
    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.NOT_CALIBRATED
    assert decision.selected_method != "paired_t"


@pytest.mark.parametrize("equal_var", [False, True])
def test_v3_two_sample_design_never_selects_welch_or_student_t(equal_var):
    decision = MethodSelector(_policy()).select(
        _report(
            design=InferenceDesign.TWO_SAMPLE,
            estimand=Estimand.MEAN_DIFFERENCE,
        ),
        equal_var=equal_var,
    )

    assert decision.robustness.empirical_support is EmpiricalSupport.NOT_CALIBRATED
    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.NOT_CALIBRATED
    assert decision.selected_method not in {"student_t", "welch_t"}


def test_v3_missing_independence_assessment_cannot_be_acceptable():
    result = _policy().evaluate(_report(independence=None))

    assert result.level is RobustnessLevel.CAUTION
    assert result.diagnostics["independence_unknown"] is True


def test_v3_not_assessed_independence_cannot_be_acceptable():
    result = _policy().evaluate(
        _report(independence=AssessmentStatus.NOT_ASSESSED)
    )

    assert result.level is RobustnessLevel.CAUTION
    assert result.diagnostics["independence_unknown"] is True


@pytest.mark.parametrize("metric", ["assumed", "verified"])
def test_v3_explicit_supported_independence_can_satisfy_component(metric):
    result = _policy().evaluate(
        _report(
            independence=AssessmentStatus.PASS,
            independence_metric=metric,
        )
    )

    assert result.level is RobustnessLevel.ACCEPTABLE
    assert result.diagnostics["independence_unknown"] is False


def test_v3_marks_non_one_sample_design_as_not_calibrated_caution():
    report = _report(
        design=InferenceDesign.TWO_SAMPLE,
        estimand=Estimand.MEAN_DIFFERENCE,
    )

    result = _policy().evaluate(report)

    assert result.level is RobustnessLevel.CAUTION
    assert result.empirical_support is EmpiricalSupport.NOT_CALIBRATED
    assert result.diagnostics["calibrated_domain"] is False


def _reporting_summary() -> pd.DataFrame:
    rows = []
    for policy, acceptable in (
        ("v2", 50),
        (V3_DEFAULT_PROFILE, 0),
        (V3_ORACLE_PROFILE, 80),
    ):
        rows.extend(
            [
                {
                    "policy": policy,
                    "level": "all",
                    "total_replications": 100,
                    "conditional_denominator": 100,
                    "type_i_error": 0.05,
                    "ci_coverage": 0.95,
                },
                {
                    "policy": policy,
                    "level": "acceptable",
                    "total_replications": 100,
                    "conditional_denominator": acceptable,
                    "type_i_error": 0.05 if acceptable else np.nan,
                    "ci_coverage": 0.95 if acceptable else np.nan,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_false_safe_reporting_marks_zero_acceptance_as_vacuous():
    summary = _reporting_summary()
    metrics = summarize_false_safe_metrics(summary)
    default = next(
        item for item in metrics if item["policy"] == V3_DEFAULT_PROFILE
    )
    flagged = flag_operating_regions(summary)
    default_flag = flagged[
        (flagged["policy"] == V3_DEFAULT_PROFILE)
        & (flagged["level"] == "acceptable")
    ].iloc[0]

    assert default["acceptable_denominator"] == 0
    assert default["overall_acceptable_rate"] == 0.0
    assert default["confirmatory_false_safe_regions"] is None
    assert default["false_safe_evaluation"] == "NOT EVALUABLE / VACUOUS"
    assert default["supports_zero_false_safe_claim"] is False
    assert default_flag["operating_region"] == "not_evaluable_vacuous"


def test_oracle_profile_is_labeled_and_excluded_from_headline_claims():
    oracle = next(
        item
        for item in summarize_false_safe_metrics(_reporting_summary())
        if item["policy"] == V3_ORACLE_PROFILE
    )

    assert V3_ORACLE_PROFILE == "v3_oracle_simulation_truth"
    assert oracle["profile_kind"] == "oracle_simulation_truth"
    assert oracle["headline_eligible"] is False
    assert oracle["supports_zero_false_safe_claim"] is False


def test_v3_calibration_runner_rejects_any_holdout_path(tmp_path):
    with pytest.raises(ValueError, match="Only experiments/results"):
        run(tmp_path / "holdout.csv.gz", tmp_path / "results")
