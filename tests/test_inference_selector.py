import numpy as np

from pyMagicStat.assumptions import InferenceValidator, RobustnessLevel
from pyMagicStat.inference import MethodSelector


def test_small_gaussian_sample_can_use_direct_t_inference():
    data = np.array([-1.3, -0.9, -0.4, -0.1, 0.0, 0.2, 0.5, 0.8, 1.2])
    report = InferenceValidator().validate_one_sample(data).report

    decision = MethodSelector().select(report)

    assert decision.selected_method == "one_sample_t"
    assert decision.robustness.level is RobustnessLevel.ACCEPTABLE


def test_large_moderately_skewed_sample_can_use_asymptotic_inference():
    data = np.random.default_rng(21).gamma(shape=4.0, scale=1.0, size=120)
    report = InferenceValidator().validate_one_sample(data).report

    decision = MethodSelector().select(report)

    assert decision.selected_method == "one_sample_t"
    assert decision.robustness.level in {
        RobustnessLevel.ACCEPTABLE,
        RobustnessLevel.CAUTION,
    }


def test_heavy_skew_and_extreme_observations_are_not_approved_by_sample_size_alone():
    data = np.random.default_rng(7).lognormal(mean=0.0, sigma=1.8, size=100)
    report = InferenceValidator().validate_one_sample(data).report

    decision = MethodSelector().select(report)

    assert decision.selected_method is None
    assert decision.robustness.level is RobustnessLevel.INSUFFICIENT
    assert decision.alternatives[0].estimand == "mean"


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
