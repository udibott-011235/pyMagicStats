import numpy as np
import pytest

from pyMagicStat.inference.non_parametric import (
    BootstrapCI,
    BootstrapMeanDifferenceCI,
    _numba_resample_variance,
)


def test_numba_variance_uses_one_resample_per_replication():
    replicates = _numba_resample_variance(
        np.array([0.0, 10.0]),
        n_resamples=500,
        seed=42,
    )

    assert set(np.unique(replicates)).issubset({0.0, 50.0})


def test_numba_variance_supports_empirical_ddof_zero_explicitly():
    replicates = _numba_resample_variance(
        np.array([0.0, 10.0]),
        n_resamples=500,
        seed=42,
        ddof=0,
    )

    assert set(np.unique(replicates)).issubset({0.0, 25.0})


@pytest.mark.parametrize("backend", ["scipy", "numba"])
def test_bootstrap_is_reproducible_with_an_explicit_seed(backend):
    data = np.array([1.0, 2.0, 2.5, 4.0, 8.0, 9.0])
    kwargs = {
        "data": data,
        "stat": "mean",
        "method": backend,
        "interval_method": "percentile",
        "n_resamples": 1000,
        "random_state": 123,
    }

    first = BootstrapCI(**kwargs).compute()
    second = BootstrapCI(**kwargs).compute()

    assert first == second


@pytest.mark.parametrize("backend", ["scipy", "numba"])
def test_repeated_compute_on_same_instance_is_reproducible(backend):
    data = np.array([1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9])
    interval = BootstrapCI(
        data,
        stat="mean",
        method=backend,
        interval_method="percentile",
        n_resamples=1000,
        random_state=42,
    )

    assert interval.compute() == interval.compute()


def test_bootstrap_does_not_advance_an_explicit_generator():
    data = np.arange(1.0, 9.0)
    rng = np.random.default_rng(42)
    expected_next = np.random.default_rng(42).random()

    interval = BootstrapCI(
        data,
        method="numba",
        interval_method="percentile",
        n_resamples=500,
        random_state=rng,
    )
    interval.compute()

    assert rng.random() == expected_next


def test_scipy_bootstrap_supports_bca_and_reports_the_estimand():
    data = np.arange(1.0, 11.0)

    result = BootstrapCI(
        data,
        stat="mean",
        interval_method="bca",
        n_resamples=1000,
        random_state=7,
    ).compute()

    assert result["lb"] < np.mean(data) < result["ub"]
    assert result["estimate"] == np.mean(data)
    assert result["interval_method"] == "bca"


@pytest.mark.parametrize("backend", ["scipy", "numba"])
def test_variance_bootstrap_defaults_to_sample_variance_ddof_one(backend):
    data = np.arange(1.0, 11.0)

    result = BootstrapCI(
        data,
        stat="variance",
        method=backend,
        interval_method="percentile",
        n_resamples=1000,
        random_state=7,
    ).compute()

    assert result["estimate"] == np.var(data, ddof=1)
    assert result["ddof"] == 1


def test_variance_bootstrap_can_target_empirical_second_moment():
    data = np.arange(1.0, 11.0)

    result = BootstrapCI(
        data,
        stat="variance",
        ddof=0,
        interval_method="percentile",
        n_resamples=1000,
        random_state=7,
    ).compute()

    assert result["estimate"] == np.var(data, ddof=0)
    assert result["ddof"] == 0


def test_variance_bootstrap_rejects_unsupported_ddof():
    with pytest.raises(ValueError, match="ddof"):
        BootstrapCI([1.0, 2.0, 3.0], stat="variance", ddof=2)


def test_numba_rejects_interval_algorithms_it_does_not_implement():
    with pytest.raises(ValueError, match="numba backend"):
        BootstrapCI(
            [1.0, 2.0, 3.0],
            method="numba",
            interval_method="bca",
        )


def test_independent_group_bootstrap_targets_mean_difference():
    group1 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    group2 = np.array([7.0, 8.0, 9.0, 10.0, 11.0])

    result = BootstrapMeanDifferenceCI(
        group1,
        group2,
        n_resamples=1500,
        random_state=99,
    ).compute()

    assert result["estimate"] == 3.0
    assert result["lb"] < 3.0 < result["ub"]
    assert result["stat"] == "mean_difference"
