import numpy as np
import pytest

from experiments.sampling_robustness_v3_calibration import run
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
from pyMagicStat.inference import MethodSelector


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
    return AssumptionReport(
        design=design,
        estimand=estimand,
        assessments={
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
            "independence": Assessment(
                "independence",
                independence,
                {"independence": "unknown" if independence is AssessmentStatus.NOT_ASSESSED else "assumed"},
            ),
        },
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


def test_method_selector_accepts_injected_v3_and_serializes_rich_result():
    selector = MethodSelector(_policy())
    decision = selector.select(_report())
    serialized = decision.to_dict()

    assert decision.selected_method == "one_sample_t"
    assert serialized["robustness"]["policy_version"] == "mean-v3-candidate-2026-08"
    assert serialized["robustness"]["model_support"] == "external"


def test_v3_marks_non_one_sample_design_as_not_calibrated_caution():
    report = _report(
        design=InferenceDesign.TWO_SAMPLE,
        estimand=Estimand.MEAN_DIFFERENCE,
    )

    result = _policy().evaluate(report)

    assert result.level is RobustnessLevel.CAUTION
    assert result.empirical_support is EmpiricalSupport.NOT_CALIBRATED
    assert result.diagnostics["calibrated_domain"] is False


def test_v3_calibration_runner_rejects_any_holdout_path(tmp_path):
    with pytest.raises(ValueError, match="Only experiments/results"):
        run(tmp_path / "holdout.csv.gz", tmp_path / "results")
