import numpy as np
import pytest

from pyMagicStat.assumptions import (
    AssessmentStatus,
    InferenceDesign,
    InferenceValidator,
)


def test_one_sample_reports_structural_shape_and_independence():
    validator = InferenceValidator()
    result = validator.validate_one_sample([1.0, 2.0, 3.0, 4.0])

    assert result.report.design is InferenceDesign.ONE_SAMPLE
    assert result.report.assessments["data_quality"].status is AssessmentStatus.PASS
    assert result.report.assessments["independence"].status is AssessmentStatus.NOT_ASSESSED
    assert result.report.to_dict()["estimand"] == "mean"


@pytest.mark.parametrize(
    "data, message",
    [
        ([1.0, np.nan, 2.0], "NaN"),
        ([3.0, 3.0, 3.0], "zero variance"),
        ([[1.0, 2.0], [3.0, 4.0]], "one-dimensional"),
    ],
)
def test_structural_failures_are_rejected_before_distribution_tests(data, message):
    with pytest.raises(ValueError, match=message):
        InferenceValidator().validate_one_sample(data)


def test_float64_distinct_values_with_negligible_variance_are_rejected():
    adjacent = np.nextafter(1.0, 2.0)
    data = np.array([1.0, adjacent, 1.0], dtype=np.float64)

    assert np.unique(data).size == 2
    assert np.var(data, ddof=1) > 0.0

    with pytest.raises(ValueError, match="numerically negligible variance"):
        InferenceValidator().validate_one_sample(data)


def test_paired_validation_assesses_differences():
    before = np.array([10.0, 11.0, 13.0, 15.0, 18.0])
    after = np.array([9.0, 9.5, 12.0, 13.5, 16.0])
    result = InferenceValidator().validate_paired(before, after)

    np.testing.assert_allclose(result.relevant_samples[0], before - after)
    assert "shape_differences" in result.report.assessments
    assert "shape_group_1" not in result.report.assessments


def test_paired_validation_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        InferenceValidator().validate_paired([1, 2, 3], [1, 2])


def test_two_sample_reports_outliers_and_variance_without_selecting_a_test():
    group1 = np.array([0.0, 0.1, -0.1, 0.2, -0.2, 12.0])
    group2 = np.array([0.0, 2.0, -2.0, 4.0, -4.0, 6.0])
    result = InferenceValidator().validate_two_sample(group1, group2, independence="assumed")

    assert result.report.assessments["outliers_group_1"].metrics["count"] == 1
    assert result.report.assessments["variance"].status in {
        AssessmentStatus.PASS,
        AssessmentStatus.WARN,
    }
    assert result.report.assessments["independence"].status is AssessmentStatus.PASS


def test_assessment_is_deterministic():
    data = np.random.default_rng(42).exponential(size=50)
    validator = InferenceValidator()

    first = validator.validate_one_sample(data).report.to_dict()
    second = validator.validate_one_sample(data).report.to_dict()

    assert first == second


def test_one_way_validation_centralizes_residual_and_variance_diagnostics():
    rng = np.random.default_rng(17)
    result = InferenceValidator().validate_one_way(
        rng.normal(size=30),
        rng.normal(loc=1.0, scale=2.0, size=35),
        rng.normal(loc=2.0, scale=3.0, size=40),
    )

    assert result.report.design is InferenceDesign.ONE_WAY
    assert result.report.estimand.value == "group_mean_differences"
    assert len(result.relevant_samples) == 3
    assert "shape_group_3" in result.report.assessments
    assert "shape_standardized_residuals" in result.report.assessments
    assert "outliers_standardized_residuals" in result.report.assessments
    assert "variance" in result.report.assessments
    for residuals in result.relevant_samples:
        assert np.mean(residuals) == pytest.approx(0.0, abs=1e-12)

    metrics = result.report.assessments["variance"].metrics
    assert set(
        [
            "sizes",
            "variance_ratio",
            "brown_forsythe_p_value",
            "fligner_p_value",
            "bartlett_p_value",
            "size_variance_spearman",
            "small_group_large_variance",
        ]
    ).issubset(metrics)


def test_one_way_requires_at_least_three_independent_groups():
    with pytest.raises(ValueError, match="at least three groups"):
        InferenceValidator().validate_one_way([1.0, 2.0], [3.0, 4.0])


def test_one_way_shape_diagnostics_do_not_pool_different_group_means():
    offsets = (0.0, 100.0, 1_000.0)
    base = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    result = InferenceValidator().validate_one_way(
        *(base + offset for offset in offsets),
        independence="assumed",
    )

    for residuals in result.relevant_samples:
        np.testing.assert_allclose(residuals, base)
    standardized_shape = result.report.assessments["shape_standardized_residuals"]
    assert standardized_shape.status is AssessmentStatus.PASS


def test_one_way_variance_assessment_flags_small_group_large_variance_alignment():
    rng = np.random.default_rng(20260827)
    result = InferenceValidator().validate_one_way(
        rng.normal(scale=8.0, size=12),
        rng.normal(scale=3.0, size=30),
        rng.normal(scale=1.0, size=60),
        independence="assumed",
    )

    variance = result.report.assessments["variance"]
    assert variance.metrics["size_variance_spearman"] <= -0.5
    assert variance.metrics["small_group_large_variance"] is True
    assert variance.status is AssessmentStatus.WARN
