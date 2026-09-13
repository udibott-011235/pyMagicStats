from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.distribution_gof.bootstrap import monte_carlo_p_value
from experiments.distribution_gof.generators import bind_family
from experiments.distribution_gof.oracles.continuous_reference import (
    continuous_simple_null_reference,
)
from experiments.distribution_gof.oracles.negative_binomial_high_precision import (
    high_precision_nb_statistic,
)
from experiments.distribution_gof.statistics import (
    FLOAT_ORACLE_RTOL,
    TAIL_ABSOLUTE_TOLERANCE,
    TAIL_RELATIVE_TOLERANCE,
    OracleMismatchError,
    TailCertificationError,
    continuous_statistic,
    negative_binomial_statistic,
    stable_ad_upper_tail_term,
)


class UniformBound:
    @staticmethod
    def cdf(x):
        return np.asarray(x, dtype=float)

    @staticmethod
    def logcdf(x):
        return np.log(np.asarray(x, dtype=float))

    @staticmethod
    def logsf(x):
        return np.log1p(-np.asarray(x, dtype=float))


def test_mc_plus_one_boundaries_and_greater_equal_tie_rule():
    assert monte_carlo_p_value(2.0, [1.0] * 199, B=199) == (0, 1 / 200)
    assert monte_carlo_p_value(2.0, [2.0] * 199, B=199) == (199, 1.0)
    exceedances, value = monte_carlo_p_value(2.0, [1.0, 2.0, 3.0], B=3)
    assert exceedances == 2
    assert value == 0.75


def test_continuous_ad_cvm_ks_hand_fixtures():
    sample = np.array([0.2, 0.6, 0.9])
    n = len(sample)
    ranks = (2 * np.arange(1, n + 1) - 1) / (2 * n)
    expected_cvm = 1 / (12 * n) + np.sum((sample - ranks) ** 2)
    expected_ad = -n - np.sum(
        (2 * np.arange(1, n + 1) - 1)
        * (np.log(sample) + np.log1p(-sample[::-1]))
    ) / n
    expected_ks = max(
        np.max(sample - np.arange(n) / n),
        np.max(np.arange(1, n + 1) / n - sample),
    )
    assert continuous_statistic(sample, UniformBound(), "CVM").value == pytest.approx(expected_cvm)
    assert continuous_statistic(sample, UniformBound(), "AD").value == pytest.approx(expected_ad)
    assert continuous_statistic(sample, UniformBound(), "KS").value == pytest.approx(expected_ks)


@pytest.mark.parametrize("statistic", ["CVM", "KS"])
def test_continuous_simple_null_matches_external_scipy_reference(statistic):
    sample = np.array([0.05, 0.2, 0.7, 1.1, 2.4])
    bound = bind_family("exponential", {"scale": "1"})
    actual = continuous_statistic(sample, bound, statistic).value
    expected = continuous_simple_null_reference(sample, bound, statistic)
    assert actual == pytest.approx(expected, rel=0, abs=2e-15)


@pytest.mark.parametrize("statistic", ["AD", "CVM"])
def test_nb_support_sum_has_remainder_certificate_and_oracle_tolerance(statistic):
    sample = np.array([0, 0, 1, 2, 4, 7], dtype=np.int64)
    bound = bind_family("negative_binomial", {"r": "2.5", "p": "0.4"})
    result = negative_binomial_statistic(
        sample, bound, statistic, parameter_count_estimated=0
    )
    oracle, oracle_remainder = high_precision_nb_statistic(
        sample, r=2.5, p=0.4, statistic=statistic
    )
    assert result.oracle_value == oracle
    assert result.remainder_bound > 0
    assert result.remainder_bound <= max(
        TAIL_ABSOLUTE_TOLERANCE,
        TAIL_RELATIVE_TOLERANCE * abs(result.value),
    )
    assert result.absolute_error <= FLOAT_ORACLE_RTOL * max(1, abs(oracle))
    assert oracle_remainder <= 1e-28 * max(1, abs(oracle))


def test_nb_stable_ad_tail_includes_boundary_j_equals_sample_maximum():
    sample = np.array([0, 1, 2, 4])
    bound = bind_family("negative_binomial", {"r": "2", "p": "0.5"})
    j = int(sample.max())
    stable = stable_ad_upper_tail_term(bound, len(sample), j)
    direct_safe = len(sample) * float(bound.sf(j)) * float(bound.pmf(j)) / float(bound.cdf(j))
    assert stable == pytest.approx(direct_safe, rel=2e-15)


def test_nb_stable_ad_tail_uses_only_logsf_logpmf_logcdf():
    calls = []

    class InstrumentedBound:
        def logsf(self, j):
            calls.append(("logsf", j))
            return math.log(0.25)

        def logpmf(self, j):
            calls.append(("logpmf", j))
            return math.log(0.1)

        def logcdf(self, j):
            calls.append(("logcdf", j))
            return math.log(0.75)

        def cdf(self, j):
            raise AssertionError("stable branch must not call cdf")

        def sf(self, j):
            raise AssertionError("stable branch must not call sf")

    assert stable_ad_upper_tail_term(InstrumentedBound(), 4, 7) == pytest.approx(
        4 * 0.25 * 0.1 / 0.75
    )
    assert calls == [("logsf", 7), ("logpmf", 7), ("logcdf", 7)]


def test_uncertifiable_nb_tail_fails_explicitly():
    class BrokenBound:
        class Parameters:
            r = 2.0
            p = 0.5

        parameters = Parameters()

        @staticmethod
        def logpmf(j):
            return -1.0

        @staticmethod
        def logcdf(j):
            return -0.1

        @staticmethod
        def logsf(j):
            return -math.inf

    with pytest.raises(TailCertificationError):
        negative_binomial_statistic(
            [0, 1], BrokenBound(), "AD", parameter_count_estimated=0
        )


def test_nb_oracle_mismatch_fails_closed(monkeypatch):
    bound = bind_family("negative_binomial", {"r": "2", "p": "0.5"})
    monkeypatch.setattr(
        "experiments.distribution_gof.statistics.high_precision_nb_statistic",
        lambda *args, **kwargs: (1000.0, 0.0),
    )
    with pytest.raises(OracleMismatchError):
        negative_binomial_statistic(
            [0, 1, 2], bound, "CVM", parameter_count_estimated=0
        )
