import json
from copy import deepcopy

import numpy as np
import pytest
import scipy.stats as stats

from pyMagicStat.distributions.distributions import (
    BinomialDistribution,
    LognormalDistribution,
    PoissonDistribution,
)
from pyMagicStat.distributions._discrete_gof import (
    point_cell,
    pool_adjacent_tails,
    upper_tail_cell,
)


def _deterministic_sample(probabilities, size):
    probabilities = np.asarray(probabilities, dtype=float)
    raw_counts = probabilities * size
    counts = np.floor(raw_counts).astype(int)
    remainder = int(size - counts.sum())
    if remainder:
        fractional_order = np.argsort(-(raw_counts - counts), kind="stable")
        counts[fractional_order[:remainder]] += 1
    return np.repeat(np.arange(probabilities.size, dtype=int), counts)


def _assert_mass_and_contiguity(result, sample_size):
    assert result["observed_total"] == sample_size
    assert result["expected_total"] == pytest.approx(sample_size, abs=1e-9)
    assert sum(cell["observed"] for cell in result["original_cells"]) == sample_size
    assert sum(cell["expected"] for cell in result["original_cells"]) == pytest.approx(
        sample_size, abs=1e-9
    )
    assert sum(cell["observed"] for cell in result["pooled_cells"]) == sample_size
    assert sum(cell["expected"] for cell in result["pooled_cells"]) == pytest.approx(
        sample_size, abs=1e-9
    )
    for left, right in zip(result["pooled_cells"], result["pooled_cells"][1:]):
        assert left["upper"] is not None
        assert right["lower"] == left["upper"] + 1


def _assert_direct_pooling_invariants(original, pooled, minimum_expected=5.0):
    assert pooled
    assert all(cell["expected"] >= minimum_expected for cell in pooled)
    assert sum(cell["observed"] for cell in pooled) == sum(
        cell["observed"] for cell in original
    )
    assert sum(cell["expected"] for cell in pooled) == pytest.approx(
        sum(cell["expected"] for cell in original), abs=1e-12
    )
    assert pooled[0]["lower"] == original[0]["lower"]
    assert pooled[-1]["upper"] == original[-1]["upper"]
    for left, right in zip(pooled, pooled[1:]):
        assert left["upper"] is not None
        assert right["lower"] == left["upper"] + 1


def test_pooling_absorbs_insufficient_cell_into_sufficient_cumulative_tail():
    cells = [
        point_cell(0, 100, 100.0),
        point_cell(1, 2, 2.0),
        upper_tail_cell(2, 6, 6.0),
    ]

    pooled = pool_adjacent_tails(cells)

    assert [(cell["lower"], cell["upper"]) for cell in pooled] == [
        (0, 0),
        (1, None),
    ]
    assert [cell["observed"] for cell in pooled] == [100, 8]
    assert [cell["expected"] for cell in pooled] == [100.0, 8.0]
    _assert_direct_pooling_invariants(cells, pooled)


@pytest.mark.parametrize(
    "cells",
    [
        [
            point_cell(0, 2, 2.0),
            point_cell(1, 3, 3.0),
            point_cell(2, 20, 20.0),
            point_cell(3, 3, 3.0),
            upper_tail_cell(4, 2, 2.0),
        ],
        [
            point_cell(0, 2, 2.0),
            point_cell(1, 2, 2.0),
            point_cell(2, 2, 2.0),
            upper_tail_cell(3, 2, 2.0),
        ],
        [
            point_cell(0, 6, 6.0),
            point_cell(1, 1, 1.0),
            point_cell(2, 1, 1.0),
            point_cell(3, 1, 1.0),
            upper_tail_cell(4, 6, 6.0),
        ],
    ],
)
def test_pooling_preserves_partition_when_both_tails_meet_or_approach(cells):
    pooled = pool_adjacent_tails(cells)

    _assert_direct_pooling_invariants(cells, pooled)


def test_poisson_high_lambda_pools_tail_and_matches_manual_pearson_result():
    data = np.random.default_rng(123).poisson(80.0, size=5000)

    result = PoissonDistribution(data).evaluate_goodness_of_fit()

    observed = np.asarray(
        [cell["observed"] for cell in result["pooled_cells"]], dtype=float
    )
    expected = np.asarray(
        [cell["expected"] for cell in result["pooled_cells"]], dtype=float
    )
    manual_df = len(result["pooled_cells"]) - 2
    manual_statistic = float(np.sum(np.square(observed - expected) / expected))
    manual_p_value = float(stats.chi2.sf(manual_statistic, manual_df))

    assert result["status"] == "ok"
    assert result["minimum_expected"] >= 5.0
    assert result["observed_total"] == len(data)
    assert result["expected_total"] == pytest.approx(len(data), abs=1e-9)
    assert result["df"] == manual_df
    assert result["statistic"] == pytest.approx(manual_statistic)
    assert result["chi2"] == pytest.approx(manual_statistic)
    assert result["p_value"] == pytest.approx(manual_p_value)
    assert result["decision"] == (
        "reject" if manual_p_value <= result["alpha"] else "fail_to_reject"
    )
    _assert_mass_and_contiguity(result, len(data))


def test_lognormal_rejects_when_log_data_reject_exact_normality():
    data = np.exp(np.concatenate([np.zeros(39), np.array([8.0])]))

    validator = LognormalDistribution(data)
    result = validator.evaluate_normality()

    assert result["status"] == "reject"
    assert result["decision"] == "reject"
    assert result["evaluated_variable"] == "log(data)"
    assert result["assessment"]["metrics"]["exact_normality_rejected"] is True
    assert validator.distribution.assessments["lognormality"] is result
    assert validator.distribution.type["Lognormal"] is False


def test_lognormal_propagates_fail_to_reject_without_claiming_identity():
    probabilities = (np.arange(100, dtype=float) + 0.5) / 100.0
    data = np.exp(stats.norm.ppf(probabilities))

    validator = LognormalDistribution(data)
    result = validator.evaluate_normality()

    assert result["status"] == "fail_to_reject"
    assert result["decision"] == "fail_to_reject"
    assert result["assessment"]["metrics"]["exact_normality_rejected"] is False
    assert "does not demonstrate" in result["reason"]
    assert validator.distribution.type["Lognormal"] is True


def test_lognormal_propagates_not_assessed():
    validator = LognormalDistribution(np.array([1.0, np.e]))

    result = validator.evaluate_normality()

    assert result["status"] == "not_assessed"
    assert result["decision"] == "not_assessed"
    assert result["assessment"]["metrics"]["exact_normality_rejected"] is None
    assert validator.distribution.type["Lognormal"] is None


def test_poisson_preserves_mass_including_unobserved_upper_tail_and_df():
    data = np.random.default_rng(1107).poisson(4.0, size=2000)
    validator = PoissonDistribution(data)

    result = validator.evaluate_goodness_of_fit()

    assert result["status"] == "ok"
    assert result["decision"] in {"reject", "fail_to_reject"}
    assert result["parameter_count_estimated"] == 1
    assert result["parameters"]["lambda"]["source"] == "estimated_from_sample_mean"
    assert result["df"] == len(result["pooled_cells"]) - 2
    assert result["minimum_expected"] >= 5.0
    assert result["original_cells"][-1]["upper"] is None
    assert result["original_cells"][-1]["observed"] == 0
    assert result["original_cells"][-1]["expected"] > 0.0
    assert result["chi2"] == result["statistic"]
    assert result["lambda"] == result["parameters"]["lambda"]["value"]
    _assert_mass_and_contiguity(result, len(data))


def test_binomial_fixed_parameters_use_full_support_and_k_minus_one_df():
    n = 10
    p = 0.35
    data = _deterministic_sample(stats.binom.pmf(np.arange(n + 1), n, p), 5000)
    validator = BinomialDistribution(data)

    result = validator.evaluate_goodness_of_fit(n=n, p=p)

    assert result["status"] == "ok"
    assert result["parameter_count_estimated"] == 0
    assert result["df"] == len(result["pooled_cells"]) - 1
    assert result["parameters"] == {
        "n": {"value": n, "source": "provided"},
        "p": {"value": p, "source": "provided"},
    }
    assert [cell["lower"] for cell in result["original_cells"]] == list(range(n + 1))
    assert result["n"] == n
    assert result["p"] == p
    _assert_mass_and_contiguity(result, len(data))


def test_binomial_estimates_only_p_when_n_is_fixed_and_uses_k_minus_two_df():
    n = 12
    p = 0.3
    data = _deterministic_sample(stats.binom.pmf(np.arange(n + 1), n, p), 6000)
    validator = BinomialDistribution(data)

    result = validator.evaluate_goodness_of_fit(n=n)

    assert result["status"] == "ok"
    assert result["parameter_count_estimated"] == 1
    assert result["df"] == len(result["pooled_cells"]) - 2
    assert result["parameters"]["n"] == {"value": n, "source": "provided"}
    assert result["parameters"]["p"]["source"] == "estimated_from_sample_mean_given_n"
    assert result["p"] == pytest.approx(np.mean(data) / n)
    _assert_mass_and_contiguity(result, len(data))


def test_binomial_without_n_is_not_assessed_even_when_p_is_provided():
    data = np.array([0, 1, 1, 2, 2, 2], dtype=int)
    validator = BinomialDistribution(data)

    result = validator.evaluate_goodness_of_fit(p=0.4)

    assert result["status"] == "not_assessed"
    assert result["decision"] is None
    assert result["p_value"] is None
    assert result["df"] is None
    assert result["n"] is None
    assert result["p"] == 0.4
    assert "required structural parameter" in result["reason"]
    assert validator.distribution.type["Binomial"] is None


def test_insufficient_degrees_of_freedom_returns_not_assessed_without_p_value():
    data = np.array([0, 1, 0, 1], dtype=int)
    result = BinomialDistribution(data).evaluate_goodness_of_fit(n=1, p=0.5)

    assert result["status"] == "not_assessed"
    assert result["decision"] is None
    assert result["df"] == 0
    assert result["p_value"] is None
    assert result["chi2"] is None


@pytest.mark.parametrize(
    ("kwargs", "reason_fragment"),
    [
        ({"n": 0, "p": 0.5}, "positive integer"),
        ({"n": 3, "p": 1.0}, "strictly between"),
        ({"n": 3, "p": 0.0}, "strictly between"),
    ],
)
def test_binomial_invalid_parameters_are_explicitly_not_assessed(kwargs, reason_fragment):
    data = np.array([0, 1, 2, 1, 0], dtype=int)

    result = BinomialDistribution(data).evaluate_goodness_of_fit(**kwargs)

    assert result["status"] == "not_assessed"
    assert result["decision"] is None
    assert result["p_value"] is None
    assert reason_fragment in result["reason"]


def test_binomial_observations_outside_fixed_support_are_not_assessed():
    data = np.array([0, 1, 2, 3, 4], dtype=int)

    result = BinomialDistribution(data).evaluate_goodness_of_fit(n=3, p=0.5)

    assert result["status"] == "not_assessed"
    assert result["decision"] is None
    assert result["p_value"] is None
    assert "support [0, n]" in result["reason"]


def test_tail_anomaly_is_retained_and_rejected_after_contiguous_pooling():
    n = 10
    p = 0.2
    counts = np.bincount(
        _deterministic_sample(stats.binom.pmf(np.arange(n + 1), n, p), 2000),
        minlength=n + 1,
    )
    counts[2] -= 40
    counts[10] += 40
    data = np.repeat(np.arange(n + 1, dtype=int), counts)

    result = BinomialDistribution(data).evaluate_goodness_of_fit(n=n, p=p)

    assert result["status"] == "ok"
    assert result["decision"] == "reject"
    tail_group = next(cell for cell in result["pooled_cells"] if cell["upper"] == n)
    assert tail_group["observed"] > tail_group["expected"]
    _assert_mass_and_contiguity(result, len(data))


def test_fit_test_uses_structured_decision_instead_of_legacy_type(monkeypatch):
    validator = BinomialDistribution(np.array([0, 1, 1, 2, 2, 3], dtype=int))
    validator.distribution.type = {"Binomial": False}

    fail_to_reject = {"status": "ok", "decision": "fail_to_reject"}

    def structured_success(*args, **kwargs):
        validator.distribution.assessments["goodness_of_fit"] = fail_to_reject
        validator.distribution.type["Binomial"] = False
        validator.distribution.type["goodness_of_fit"] = {
            "status": "ok",
            "decision": "reject",
        }
        return fail_to_reject

    monkeypatch.setattr(validator, "evaluate_goodness_of_fit", structured_success)
    monkeypatch.setattr(validator, "evaluate_normal_approximation", lambda: True)

    result = validator.fit_test(n=3, p=0.5)
    assert result["approx_normal"] is True

    reject = {"status": "ok", "decision": "reject"}

    def structured_reject(*args, **kwargs):
        validator.distribution.assessments["goodness_of_fit"] = reject
        validator.distribution.type["Binomial"] = True
        validator.distribution.type["goodness_of_fit"] = {
            "status": "ok",
            "decision": "fail_to_reject",
        }
        return reject

    monkeypatch.setattr(validator, "evaluate_goodness_of_fit", structured_reject)
    with pytest.raises(ValueError, match="decision='reject'"):
        validator.fit_test(n=3, p=0.5)

    for blocked in (
        {"status": "not_assessed", "decision": None, "reason": "insufficient df"},
        {"status": "error", "decision": None, "reason": "solver failed"},
    ):
        def structured_block(*args, _blocked=blocked, **kwargs):
            validator.distribution.assessments["goodness_of_fit"] = _blocked
            validator.distribution.type["Binomial"] = True
            return _blocked

        monkeypatch.setattr(validator, "evaluate_goodness_of_fit", structured_block)
        with pytest.raises(ValueError, match=blocked["status"]):
            validator.fit_test(n=3, p=0.5)


def test_legacy_type_is_only_a_mirror_of_the_structured_gof_decision():
    n = 8
    p = 0.4
    data = _deterministic_sample(stats.binom.pmf(np.arange(n + 1), n, p), 4000)
    validator = BinomialDistribution(data)

    result = validator.evaluate_goodness_of_fit(n=n, p=p)

    canonical = validator.distribution.assessments["goodness_of_fit"]
    legacy = validator.distribution.type["goodness_of_fit"]
    assert canonical is result
    assert canonical is not legacy
    assert canonical == legacy
    expected_legacy = result["decision"] == "fail_to_reject"
    assert validator.distribution.type["Binomial"] is expected_legacy
    assert {"status", "decision", "chi2", "p_value", "n", "p"} <= result.keys()
    json.dumps(result, allow_nan=False)

    original = deepcopy(canonical)
    legacy["decision"] = (
        "reject" if canonical["decision"] != "reject" else "fail_to_reject"
    )
    legacy["parameters"]["n"]["value"] = 999
    legacy["pooled_cells"][0]["expected"] = -1.0
    assert canonical == original


def test_lognormal_legacy_result_is_deeply_isolated_from_canonical_assessment():
    probabilities = (np.arange(100, dtype=float) + 0.5) / 100.0
    validator = LognormalDistribution(np.exp(stats.norm.ppf(probabilities)))

    result = validator.evaluate_normality()

    canonical = validator.distribution.assessments["lognormality"]
    legacy = validator.distribution.type["normality_log_results"]
    assert canonical is result
    assert canonical is not legacy
    assert canonical == legacy

    original = deepcopy(canonical)
    legacy["decision"] = "reject"
    legacy["assessment"]["metrics"]["exact_normality_rejected"] = True
    legacy["shape"]["skewness"] = 999.0
    assert canonical == original


@pytest.mark.parametrize(("n", "p"), [(2, 0.1), (5, 0.5), (20, 0.9)])
def test_binomial_mass_invariants_across_support_shapes(n, p):
    data = np.random.default_rng(n).binomial(n, p, size=3000)

    result = BinomialDistribution(data).evaluate_goodness_of_fit(n=n, p=p)

    assert result["status"] in {"ok", "not_assessed"}
    _assert_mass_and_contiguity(result, len(data))
    if result["status"] == "ok":
        assert result["minimum_expected"] >= 5.0
        assert result["df"] > 0


@pytest.mark.parametrize("lambda_value", [0.2, 1.0, 4.0, 20.0])
def test_poisson_mass_invariants_across_tail_shapes(lambda_value):
    data = np.random.default_rng(int(lambda_value * 10) + 1).poisson(
        lambda_value, size=3000
    )

    result = PoissonDistribution(data).evaluate_goodness_of_fit()

    assert result["status"] in {"ok", "not_assessed"}
    _assert_mass_and_contiguity(result, len(data))
    assert result["original_cells"][-1]["upper"] is None
    if result["status"] == "ok":
        assert result["minimum_expected"] >= 5.0
        assert result["df"] > 0
