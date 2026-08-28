import numpy as np

from pyMagicStat.assumptions import (
    Assessment,
    AssessmentStatus,
    AssumptionReport,
    InferenceValidator,
    RobustnessLevel,
    SamplingRobustness,
)
from pyMagicStat.inference import MethodSelector


def test_small_gaussian_sample_can_use_direct_t_inference():
    data = np.array([-1.3, -0.9, -0.4, -0.1, 0.0, 0.2, 0.5, 0.8, 1.2])
    report = InferenceValidator().validate_one_sample(
        data,
        independence="assumed",
    ).report

    decision = MethodSelector().select(report)

    assert decision.selected_method == "one_sample_t"
    assert decision.robustness.level is RobustnessLevel.ACCEPTABLE


def test_large_moderately_skewed_sample_can_use_asymptotic_inference():
    data = np.random.default_rng(21).gamma(shape=4.0, scale=1.0, size=120)
    report = InferenceValidator().validate_one_sample(data).report

    decision = MethodSelector().select(report)

    assert decision.selected_method == "one_sample_t"
    assert decision.robustness.level is RobustnessLevel.CAUTION


def test_heavy_skew_and_extreme_observations_are_not_approved_by_sample_size_alone():
    data = np.random.default_rng(7).lognormal(mean=0.0, sigma=1.8, size=100)
    report = InferenceValidator().validate_one_sample(data).report

    decision = MethodSelector().select(report)

    assert decision.selected_method is None
    assert decision.robustness.level is RobustnessLevel.INSUFFICIENT
    assert decision.alternatives[0].estimand == "mean"


def test_small_sample_shape_pass_cannot_bypass_extreme_outlier_constraints():
    data = np.array(
        [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 15.0]
    )
    report = InferenceValidator().validate_one_sample(
        data,
        independence="assumed",
    ).report

    assert report.assessments["outliers"].metrics["fraction"] == 1 / 12

    decision = MethodSelector().select(report)

    assert decision.selected_method is None
    assert decision.robustness.level is RobustnessLevel.INSUFFICIENT

    # Preserve the report's structural/outlier evidence while controlling the
    # shape status so this test catches the original early-return bypass across
    # SciPy versions with different diagnostic behaviour.
    assessments = dict(report.assessments)
    assessments["shape"] = Assessment(
        name="shape_sample",
        status=AssessmentStatus.PASS,
        metrics={
            **report.assessments["shape"].metrics,
            "skewness": 0.5,
            "excess_kurtosis": 0.5,
        },
        reasons=("Controlled compatible-shape diagnostic.",),
    )
    controlled_report = AssumptionReport(
        design=report.design,
        estimand=report.estimand,
        assessments=assessments,
    )

    controlled_result = SamplingRobustness().evaluate(controlled_report)
    assert controlled_result.level is RobustnessLevel.INSUFFICIENT


def test_shape_failure_is_not_an_independent_large_sample_veto():
    data = np.random.default_rng(1).standard_t(df=3, size=200)
    report = InferenceValidator().validate_one_sample(
        data,
        independence="assumed",
    ).report

    assert report.assessments["shape"].status.value == "fail"
    decision = MethodSelector().select(report)

    assert decision.selected_method == "one_sample_t"
    assert decision.robustness.level is RobustnessLevel.CAUTION
    assert any("heavy-tail" in reason for reason in decision.reasons)


def test_welch_is_the_default_for_two_independent_groups():
    rng = np.random.default_rng(42)
    report = InferenceValidator().validate_two_sample(
        rng.normal(size=60),
        rng.normal(loc=0.5, size=60),
    ).report

    decision = MethodSelector().select(report)

    assert decision.selected_method == "welch_t"
    assert any("variance-robust default" in reason for reason in decision.reasons)


def test_student_requires_an_explicit_request():
    rng = np.random.default_rng(12)
    report = InferenceValidator().validate_two_sample(
        rng.normal(size=50),
        rng.normal(size=50),
    ).report

    decision = MethodSelector().select(report, equal_var=True)

    assert decision.selected_method == "student_t"


def test_mann_whitney_is_labeled_with_its_distinct_estimand():
    rng = np.random.default_rng(9)
    report = InferenceValidator().validate_two_sample(
        rng.normal(size=40),
        rng.normal(size=40),
    ).report

    alternatives = MethodSelector().select(report).to_dict()["alternatives"]
    mann_whitney = next(item for item in alternatives if item["method"] == "mann_whitney_u")

    assert mann_whitney["estimand"] == "probabilistic_ordering"


def test_unknown_independence_is_reported_as_caution_not_inferred_from_values():
    data = np.array([-1.3, -0.9, -0.4, -0.1, 0.0, 0.2, 0.5, 0.8, 1.2])
    report = InferenceValidator().validate_one_sample(data).report

    decision = MethodSelector().select(report)

    assert decision.selected_method == "one_sample_t"
    assert decision.robustness.level is RobustnessLevel.CAUTION
    assert any("Independence was not assessed" in reason for reason in decision.reasons)


def test_one_way_selector_defaults_to_calibrated_welch_anova():
    rng = np.random.default_rng(31)
    report = InferenceValidator().validate_one_way(
        rng.normal(size=40),
        rng.normal(loc=1.0, size=45),
        rng.normal(loc=2.0, scale=2.0, size=50),
        independence="assumed",
    ).report

    decision = MethodSelector().select(report)

    serialized = decision.to_dict()

    assert decision.selected_method == "welch_anova"
    assert decision.parametric_recommended is True
    assert serialized["status"] == "selected"
    assert any("calibrated variance-robust default" in reason for reason in decision.reasons)


def test_one_way_classical_requires_explicit_request_and_variance_support():
    rng = np.random.default_rng(312)
    report = InferenceValidator().validate_one_way(
        rng.normal(size=50),
        rng.normal(loc=0.5, size=50),
        rng.normal(loc=1.0, size=50),
        independence="assumed",
    ).report

    decision = MethodSelector().select(report, equal_var=True)

    assert decision.selected_method == "classical_anova"
    assert any("explicitly requested" in reason for reason in decision.reasons)


def test_one_way_classical_is_denied_for_dangerous_size_variance_alignment():
    rng = np.random.default_rng(20260827)
    report = InferenceValidator().validate_one_way(
        rng.normal(scale=8.0, size=12),
        rng.normal(scale=3.0, size=30),
        rng.normal(scale=1.0, size=60),
        independence="assumed",
    ).report

    decision = MethodSelector().select(report, equal_var=True)

    assert decision.selected_method is None
    assert decision.status.value == "insufficient"
    assert any("common-variance" in reason for reason in decision.reasons)


def test_one_way_severe_skew_and_contamination_remain_insufficient():
    rng = np.random.default_rng(919)
    report = InferenceValidator().validate_one_way(
        *(rng.lognormal(sigma=1.5, size=60) for _ in range(3)),
        independence="assumed",
    ).report

    decision = MethodSelector().select(report)

    assert decision.selected_method is None
    assert decision.robustness.level is RobustnessLevel.INSUFFICIENT
