import json

import numpy as np
import pytest
from scipy.stats import t

from pyMagicStat.assumptions import (
    AssumptionProvenance,
    Estimand,
    InferenceDesign,
    InferenceValidator,
    ProcessUncertainty,
    SamplingRobustnessV3,
)
from pyMagicStat.inference import (
    CI_ENDPOINT_RESIDUAL_TOLERANCE,
    EMPIRICAL_LIKELIHOOD_METHOD,
    LAMBDA_RESIDUAL_TOLERANCE,
    InferenceDecisionStatus,
    InferenceGuarantee,
    MethodSelector,
    capabilities_for,
    empirical_likelihood_mean_ci,
    empirical_likelihood_mean_test,
)


@pytest.fixture
def sample():
    return np.array([1.0, 2.0, 3.0, 5.0, 8.0])


def test_sample_mean_has_zero_lambda_statistic_and_unit_p_value(sample):
    result = empirical_likelihood_mean_test(sample, np.mean(sample))

    assert result.lambda_value == pytest.approx(0.0, abs=1e-15)
    assert result.lambda_residual == pytest.approx(0.0, abs=1e-15)
    assert result.statistic == pytest.approx(0.0, abs=1e-15)
    assert result.log_likelihood_ratio == pytest.approx(0.0, abs=1e-15)
    assert result.p_value == pytest.approx(1.0, abs=1e-15)
    assert result.guarantee is InferenceGuarantee.ASYMPTOTIC_MOMENT_BASED


@pytest.mark.parametrize("mu", [1.1, 2.0, 3.0, 4.0, 6.0, 7.9])
def test_profile_statistic_is_nonnegative(sample, mu):
    result = empirical_likelihood_mean_test(sample, mu)

    assert result.feasible is True
    assert result.statistic >= 0.0


def test_profile_statistic_increases_locally_away_from_sample_mean(sample):
    estimate = float(np.mean(sample))
    lower_near = empirical_likelihood_mean_test(sample, estimate - 0.1).statistic
    lower_far = empirical_likelihood_mean_test(sample, estimate - 0.4).statistic
    upper_near = empirical_likelihood_mean_test(sample, estimate + 0.1).statistic
    upper_far = empirical_likelihood_mean_test(sample, estimate + 0.4).statistic

    assert 0.0 < lower_near < lower_far
    assert 0.0 < upper_near < upper_far


@pytest.mark.parametrize("mu", [0.99, 8.01])
def test_candidate_outside_sample_convex_hull_is_explicitly_infeasible(sample, mu):
    result = empirical_likelihood_mean_test(sample, mu)

    assert result.feasible is False
    assert result.regular is False
    assert result.statistic == np.inf
    assert result.p_value is None
    assert result.lambda_value is None
    assert "outside" in result.reason


@pytest.mark.parametrize(
    ("mu", "expected_lambda"),
    [(1.0, np.inf), (8.0, -np.inf)],
)
def test_convex_hull_boundary_is_feasible_but_nonregular(
    sample,
    mu,
    expected_lambda,
):
    result = empirical_likelihood_mean_test(sample, mu)

    assert result.feasible is True
    assert result.boundary is True
    assert result.regular is False
    assert result.statistic == np.inf
    assert result.lambda_value == expected_lambda
    assert result.p_value is None


def test_profile_ci_contains_estimate_stays_in_hull_and_hits_critical_value(sample):
    interval = empirical_likelihood_mean_ci(sample)

    assert interval.feasible is True
    assert interval.regular is True
    assert np.min(sample) <= interval.lower < interval.estimate
    assert interval.estimate < interval.upper <= np.max(sample)
    assert interval.lower_statistic == pytest.approx(
        interval.critical_value,
        abs=CI_ENDPOINT_RESIDUAL_TOLERANCE,
    )
    assert interval.upper_statistic == pytest.approx(
        interval.critical_value,
        abs=CI_ENDPOINT_RESIDUAL_TOLERANCE,
    )
    assert interval.lower_endpoint_residual <= CI_ENDPOINT_RESIDUAL_TOLERANCE
    assert interval.upper_endpoint_residual <= CI_ENDPOINT_RESIDUAL_TOLERANCE

    lower = empirical_likelihood_mean_test(sample, interval.lower)
    upper = empirical_likelihood_mean_test(sample, interval.upper)
    assert lower.statistic == pytest.approx(
        interval.critical_value,
        abs=CI_ENDPOINT_RESIDUAL_TOLERANCE,
    )
    assert upper.statistic == pytest.approx(
        interval.critical_value,
        abs=CI_ENDPOINT_RESIDUAL_TOLERANCE,
    )


def test_affine_translation_invariance(sample):
    mu = 3.0
    shift = 1234.5

    baseline = empirical_likelihood_mean_test(sample, mu)
    translated = empirical_likelihood_mean_test(sample + shift, mu + shift)

    assert translated.statistic == pytest.approx(
        baseline.statistic,
        rel=1e-12,
        abs=1e-12,
    )
    assert translated.lambda_value == pytest.approx(
        baseline.lambda_value,
        rel=1e-12,
        abs=1e-12,
    )


def test_positive_scale_invariance(sample):
    mu = 3.0
    factor = 17.25

    baseline = empirical_likelihood_mean_test(sample, mu)
    scaled = empirical_likelihood_mean_test(factor * sample, factor * mu)

    assert scaled.statistic == pytest.approx(
        baseline.statistic,
        rel=1e-12,
        abs=1e-12,
    )
    assert scaled.lambda_value == pytest.approx(
        baseline.lambda_value / factor,
        rel=1e-12,
        abs=1e-12,
    )


def test_permutation_invariance(sample):
    permutation = np.array([3, 0, 4, 1, 2])

    baseline = empirical_likelihood_mean_test(sample, 3.0)
    permuted = empirical_likelihood_mean_test(sample[permutation], 3.0)

    assert permuted.statistic == pytest.approx(baseline.statistic, abs=1e-14)
    assert permuted.lambda_value == pytest.approx(baseline.lambda_value, abs=1e-14)


def test_repeated_values_do_not_create_order_dependence():
    data = np.array([1.0, 1.0, 2.0, 2.0, 5.0, 9.0])

    forward = empirical_likelihood_mean_test(data, 3.0)
    reversed_result = empirical_likelihood_mean_test(data[::-1], 3.0)

    assert reversed_result.statistic == pytest.approx(forward.statistic, abs=1e-14)
    assert reversed_result.lambda_value == pytest.approx(
        forward.lambda_value,
        abs=1e-14,
    )


def test_lambda_equation_residual_matches_declared_tolerance(sample):
    mu = 3.0
    result = empirical_likelihood_mean_test(sample, mu)
    centered = sample - mu
    scale = np.max(np.abs(centered))
    normalized = centered / scale
    tau = result.lambda_value * scale
    residual = abs(np.sum(normalized / (1.0 + tau * normalized)))

    assert residual == pytest.approx(result.lambda_residual, abs=1e-15)
    assert residual <= LAMBDA_RESIDUAL_TOLERANCE
    assert result.converged is True


def test_lambda_solution_reconstructs_probability_constraints_and_ratio(sample):
    mu = 3.0
    result = empirical_likelihood_mean_test(sample, mu)
    centered = sample - mu
    denominators = 1.0 + result.lambda_value * centered
    weights = 1.0 / (sample.size * denominators)
    statistic = 2.0 * np.sum(np.log1p(result.lambda_value * centered))

    assert np.all(weights > 0.0)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-14)
    assert np.sum(weights * centered) == pytest.approx(0.0, abs=1e-14)
    assert statistic == pytest.approx(result.statistic, abs=1e-14)


@pytest.mark.parametrize(
    "bad_data",
    [
        np.array([1.0, np.nan, 2.0]),
        np.array([1.0, np.inf, 2.0]),
        np.array([1.0, -np.inf, 2.0]),
    ],
)
def test_nonfinite_data_are_rejected(bad_data):
    with pytest.raises(ValueError, match="finite"):
        empirical_likelihood_mean_test(bad_data, 1.0)


def test_two_dimensional_data_are_rejected():
    with pytest.raises(ValueError, match="one-dimensional"):
        empirical_likelihood_mean_test(np.ones((2, 2)), 1.0)


def test_empty_data_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        empirical_likelihood_mean_test([], 0.0)


def test_constant_sample_is_explicitly_nonregular():
    data = np.full(5, 4.25)

    supported = empirical_likelihood_mean_test(data, 4.25)
    outside = empirical_likelihood_mean_test(data, 4.0)
    interval = empirical_likelihood_mean_ci(data)

    assert supported.feasible is True
    assert supported.regular is False
    assert supported.statistic == 0.0
    assert supported.p_value is None
    assert "zero variance" in supported.reason
    assert outside.feasible is False
    assert interval.feasible is False
    assert interval.lower is interval.upper is None
    assert "constant sample" in interval.reason


def test_single_observation_is_explicitly_nonregular():
    result = empirical_likelihood_mean_test([7.0], 7.0)
    interval = empirical_likelihood_mean_ci([7.0])

    assert result.feasible is True
    assert result.regular is False
    assert result.p_value is None
    assert interval.feasible is False
    assert "At least two" in interval.reason


def test_engine_does_not_mutate_input(sample):
    original = sample.copy()

    empirical_likelihood_mean_test(sample, 3.0)
    empirical_likelihood_mean_ci(sample)

    np.testing.assert_array_equal(sample, original)


@pytest.mark.parametrize(
    ("data", "mu"),
    [
        (np.array([0.01, 0.02, 0.03, 0.1, 1.0, 10.0, 50.0]), 5.0),
        (np.array([1.0, 2.0, 3.0, 4.0, 1e12]), 1e11),
        (1.0 + np.array([-3.0, -1.0, 0.0, 2.0, 4.0]) * 1e-12, 1.0 + 1e-13),
        (np.array([-1e12, -1e-6, 0.0, 1e-6, 1e12]), 1e10),
        (np.array([1.0, 4.0]), 2.0),
    ],
)
def test_adversarial_magnitude_and_small_sample_cases_converge(data, mu):
    result = empirical_likelihood_mean_test(data, mu)

    assert result.feasible is True
    assert result.regular is True
    assert result.converged is True
    assert np.isfinite(result.statistic)
    assert result.statistic >= 0.0
    assert result.lambda_residual <= LAMBDA_RESIDUAL_TOLERANCE


@pytest.mark.parametrize("mu", [1e-12, 9.0 - 1e-12])
def test_candidates_extremely_close_to_hull_boundaries_are_stable(mu):
    data = np.array([0.0, 1.0, 2.0, 5.0, 9.0])

    result = empirical_likelihood_mean_test(data, mu)

    assert result.feasible is True
    assert result.regular is True
    assert result.converged is True
    assert np.isfinite(result.statistic)
    assert result.lambda_residual <= LAMBDA_RESIDUAL_TOLERANCE


def test_large_sample_profile_and_interval_are_numerically_regular():
    data = np.random.default_rng(20260829).lognormal(
        mean=0.0,
        sigma=0.8,
        size=5_000,
    )
    estimate = float(np.mean(data))

    result = empirical_likelihood_mean_test(data, estimate * 0.99)
    interval = empirical_likelihood_mean_ci(data)

    assert result.converged is True
    assert result.lambda_residual <= LAMBDA_RESIDUAL_TOLERANCE
    assert interval.feasible is True
    assert interval.lower < estimate < interval.upper


def test_gaussian_el_and_t_intervals_broadly_converge_as_sanity_check():
    data = np.random.default_rng(44).normal(
        loc=2.0,
        scale=3.0,
        size=2_000,
    )
    estimate = float(np.mean(data))
    standard_error = float(np.std(data, ddof=1) / np.sqrt(data.size))
    t_critical = float(t.ppf(0.975, df=data.size - 1))
    t_lower = estimate - t_critical * standard_error
    t_upper = estimate + t_critical * standard_error

    interval = empirical_likelihood_mean_ci(data)

    assert interval.feasible is True
    assert interval.lower == pytest.approx(t_lower, rel=0.03)
    assert interval.upper == pytest.approx(t_upper, rel=0.03)


def test_result_objects_are_json_serializable(sample):
    test_result = empirical_likelihood_mean_test(sample, 3.0)
    interval = empirical_likelihood_mean_ci(sample)

    json.dumps(test_result.to_dict())
    json.dumps(interval.to_dict())


def test_registered_capability_stays_nonautomatic_and_selector_does_not_fallback():
    capability = {
        item.method: item
        for item in capabilities_for(
            InferenceDesign.ONE_SAMPLE,
            Estimand.MEAN,
        )
    }[EMPIRICAL_LIKELIHOOD_METHOD]
    report = InferenceValidator().validate_one_sample(
        np.linspace(-1.0, 1.0, 101),
        independence="assumed",
    ).report
    decision = MethodSelector(
        SamplingRobustnessV3(
            model_provenance=AssumptionProvenance.EMPIRICAL,
            process_uncertainty=ProcessUncertainty.LOW,
        )
    ).select(report)

    assert capability.guarantee is InferenceGuarantee.ASYMPTOTIC_MOMENT_BASED
    assert capability.calibrated is False
    assert capability.automatic_selection_allowed is False
    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.REVIEW_REQUIRED
