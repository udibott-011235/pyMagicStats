import numpy as np
import pytest
from scipy import stats

from experiments.shape_contract_simulation import scenarios, simulate, summarize
from pyMagicStat.assumptions import (
    AssessmentStatus,
    InferenceValidator,
    RobustnessLevel,
    ShapeAssessment,
)
from pyMagicStat.distributions import Distribution, NormalDistribution
from pyMagicStat.inference import MethodSelector


def test_distribution_is_the_canonical_dataset_and_descriptive_source():
    data = np.array([1, 2, 4, 8, 16, 32], dtype=np.int64)
    distribution = Distribution(data)

    assert np.array_equal(distribution.data, np.asarray(data))
    assert distribution.n == data.size
    assert distribution.mean == pytest.approx(np.mean(data))
    assert distribution.median == pytest.approx(np.median(data))
    assert distribution.var == pytest.approx(np.var(data, ddof=1))
    assert distribution.std == pytest.approx(np.std(data, ddof=1))
    assert distribution.skewness == pytest.approx(stats.skew(data, bias=False))
    assert distribution.excess_kurtosis == pytest.approx(
        stats.kurtosis(data, fisher=True, bias=False)
    )
    assert distribution.iqr == pytest.approx(distribution.q3 - distribution.q1)
    assert distribution.range == pytest.approx(distribution.max - distribution.min)


def test_shape_assessment_matches_distribution_for_array_and_object_inputs():
    data = np.random.default_rng(123).standard_t(df=10, size=100)
    distribution = Distribution(data)

    from_array = ShapeAssessment().assess(distribution.data)
    from_distribution = ShapeAssessment().assess(distribution)

    for assessment in (from_array, from_distribution):
        assert assessment.metrics["skewness"] == pytest.approx(
            distribution.skewness
        )
        assert assessment.metrics["excess_kurtosis"] == pytest.approx(
            distribution.excess_kurtosis
        )


def test_generated_normal_sample_exposes_evidence_without_binary_assumption():
    data = np.random.default_rng(42).normal(size=750)
    shape = ShapeAssessment().assess(data)

    assert shape.metrics["departure_magnitude"] in {"mild", "moderate", "severe"}
    assert isinstance(shape.metrics["shapiro_rejects_exact_normality"], bool)
    assert isinstance(shape.metrics["dagostino_rejects_exact_normality"], bool)
    assert isinstance(shape.metrics["exact_normality_rejected"], bool)


def test_formal_rejection_with_mild_observed_departure_is_not_called_material():
    data = np.random.default_rng(526).normal(size=750)
    shape = ShapeAssessment().assess(data)

    assert shape.metrics["shapiro_rejects_exact_normality"] is True
    assert shape.metrics["dagostino_rejects_exact_normality"] is True
    assert shape.metrics["exact_normality_rejected"] is True
    assert shape.metrics["departure_magnitude"] == "mild"
    assert shape.status is AssessmentStatus.PASS
    assert "material departure" not in " ".join(shape.reasons).lower()


def test_formal_rejection_does_not_block_large_sample_t_inference():
    data = np.random.default_rng(526).normal(size=750)
    report = InferenceValidator().validate_one_sample(
        data,
        independence="assumed",
    ).report

    shape = report.assessments["shape"]
    decision = MethodSelector().select(report)

    assert shape.metrics["exact_normality_rejected"] is True
    assert shape.metrics["departure_magnitude"] == "mild"
    assert decision.robustness.level in {
        RobustnessLevel.ACCEPTABLE,
        RobustnessLevel.CAUTION,
    }
    assert decision.selected_method == "one_sample_t"


def test_severely_skewed_sample_remains_insufficient_for_mean_inference():
    data = np.random.default_rng(7).lognormal(sigma=1.25, size=30)
    report = InferenceValidator().validate_one_sample(
        data,
        independence="assumed",
    ).report

    shape = report.assessments["shape"]
    decision = MethodSelector().select(report)

    assert shape.metrics["departure_magnitude"] == "severe"
    assert shape.status is AssessmentStatus.FAIL
    assert decision.robustness.level is RobustnessLevel.INSUFFICIENT
    assert decision.selected_method is None


def test_unassessed_shape_cannot_enter_the_direct_compatibility_path():
    report = InferenceValidator().validate_one_sample(
        np.array([1.0, 2.0]),
        independence="assumed",
    ).report

    shape = report.assessments["shape"]
    decision = MethodSelector().select(report)

    assert shape.metrics["departure_magnitude"] == "not_assessed"
    assert shape.status is AssessmentStatus.NOT_ASSESSED
    assert decision.robustness.level is RobustnessLevel.INSUFFICIENT
    assert decision.selected_method is None


def test_normal_distribution_keeps_only_an_explicit_legacy_boolean():
    data = np.random.default_rng(526).normal(size=750)
    validator = NormalDistribution(data)
    result = validator.evaluate_normality()

    assert result["assessment"]["metrics"]["exact_normality_rejected"] is True
    assert validator.distribution.type["Normal"] is False
    assert validator.distribution.assessments["normality"].metrics[
        "departure_magnitude"
    ] == "mild"


def test_legacy_distribution_apis_are_explicitly_deprecated():
    distribution = Distribution(np.arange(5.0))

    with pytest.warns(DeprecationWarning, match="excess_kurtosis"):
        assert distribution.kurtosis == distribution.excess_kurtosis
    with pytest.warns(DeprecationWarning, match="structured assessment"):
        distribution.update_type("Example", True, "result", {"p_value": 0.5})


def test_small_shape_contract_simulation_is_reproducible_and_complete():
    assert {scenario.name for scenario in scenarios()} == {
        "normal",
        "student_t_df10",
        "student_t_df5",
        "lognormal_moderate",
        "lognormal_severe",
    }

    kwargs = {"replications": 2, "sample_sizes": (30, 100, 750), "seed": 99}
    first = simulate(**kwargs)
    second = simulate(**kwargs)

    assert first == second
    assert len(first) == 5 * 3 * 2
    assert {
        "skewness",
        "excess_kurtosis",
        "shapiro_rejects_exact_normality",
        "dagostino_rejects_exact_normality",
        "departure_magnitude",
        "shape_status",
        "robustness",
    } <= first[0].keys()
    assert len(summarize(first)) == 15
