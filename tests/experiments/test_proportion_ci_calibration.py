import warnings

import numpy as np
import pytest
from scipy import stats

from experiments.proportion_ci_calibration.harness import (
    ALPHAS,
    CANDIDATE_SHA,
    base_probability_grid,
    calibrate_n,
    coverage_from_intervals,
    expected_width_matrix,
    induced_probability_grid,
    jeffreys_interval_grid,
    production_interval_grid,
)
from pyMagicStat.inference import PopulationProportionCI


def test_harness_is_pinned_to_the_frozen_candidate():
    assert CANDIDATE_SHA == "2df5b90a5395163e723f9c52aafbb91fdce96d43"
    assert ALPHAS == (0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.200)


@pytest.mark.parametrize("method", ("wilson", "clopper_pearson", "wald"))
def test_production_grid_calls_the_public_count_api(method):
    grid = production_interval_grid(5, 0.05, method)
    for x in range(6):
        with pytest.warns(UserWarning) if method == "wald" else _no_warning():
            direct = PopulationProportionCI.from_counts(x, 5, method=method).calculate_interval()
        assert grid.lower[x] == direct["lb"]
        assert grid.upper[x] == direct["ub"]


class _no_warning:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_coverage_matches_brute_force_enumeration():
    n = 8
    p = np.asarray((0.01, 0.2, 0.5, 0.9))
    grid = production_interval_grid(n, 0.05, "wilson")
    observed, _, _ = coverage_from_intervals(n, p, grid.lower, grid.upper)
    expected = []
    for probability in p:
        mask = (grid.lower <= probability) & (probability <= grid.upper)
        expected.append(np.sum(stats.binom.pmf(np.arange(n + 1), n, probability)[mask]))
    np.testing.assert_allclose(observed, expected, atol=2e-15, rtol=0.0)


def test_expected_width_matches_full_pmf_sum():
    n = 12
    p = np.asarray((0.01, 0.25, 0.5, 0.99))
    grids = [
        production_interval_grid(n, 0.05, method)
        for method in ("wilson", "clopper_pearson")
    ]
    widths = np.column_stack([grid.width for grid in grids])
    observed, mass = expected_width_matrix(n, p, widths)
    brute = np.asarray(
        [
            stats.binom.pmf(np.arange(n + 1), n, probability) @ widths
            for probability in p
        ]
    )
    np.testing.assert_allclose(observed, brute, atol=2e-14, rtol=0.0)
    assert np.all(mass >= 1.0 - 1.1e-14)


def test_expected_width_uses_a_warning_free_tail_bound_at_extreme_probabilities():
    n = 200
    p = np.asarray(
        (
            np.nextafter(0.0, 1.0),
            1e-12,
            1.0 - 1e-12,
            np.nextafter(1.0, 0.0),
        )
    )
    grid = production_interval_grid(n, 0.05, "wilson")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        observed, mass = expected_width_matrix(n, p, grid.width)
    assert not any(item.category is RuntimeWarning for item in recorded)
    brute = np.asarray(
        [
            stats.binom.pmf(np.arange(n + 1), n, probability) @ grid.width
            for probability in p
        ]
    )[:, None]
    np.testing.assert_allclose(observed, brute, atol=2e-14, rtol=0.0)
    assert np.all(mass >= 1.0 - 1.1e-14)


@pytest.mark.parametrize("n", (1, 2, 5, 10, 97, 101))
def test_expected_width_support_matches_full_pmf_and_complement_symmetry(n):
    left = np.asarray(
        (
            np.nextafter(0.0, 1.0),
            1e-12,
            1e-8,
            1e-4,
            0.01,
            0.1,
            0.5,
        ),
        dtype=np.float64,
    )
    probabilities = np.unique(np.concatenate((left, 1.0 - left)))
    grids = [
        production_interval_grid(n, 0.05, method)
        for method in ("wilson", "clopper_pearson", "wald")
    ]
    grids.append(jeffreys_interval_grid(n, 0.05))
    widths = np.column_stack([grid.width for grid in grids])

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        observed, retained_mass = expected_width_matrix(
            n,
            probabilities,
            widths,
            batch_size=3,
        )
    assert not any(item.category is RuntimeWarning for item in recorded)

    outcomes = np.arange(n + 1)
    full_pmf = np.asarray(
        [stats.binom.pmf(outcomes, n, probability) for probability in probabilities]
    )
    expected = full_pmf @ widths
    np.testing.assert_allclose(observed, expected, atol=5e-14, rtol=0.0)
    assert np.all(retained_mass >= 1.0 - 1.1e-14)

    complement, _ = expected_width_matrix(
        n,
        1.0 - probabilities,
        widths,
        batch_size=64,
    )
    np.testing.assert_allclose(observed, complement, atol=5e-14, rtol=0.0)

    alternate_batch, _ = expected_width_matrix(
        n,
        probabilities,
        widths,
        batch_size=64,
    )
    np.testing.assert_allclose(observed, alternate_batch, atol=5e-15, rtol=0.0)


def test_base_grid_contains_all_linear_and_event_families():
    grid = base_probability_grid(20, linear_step=0.0001)
    assert {0.0, 1.0, 1e-12, 0.5, 0.0001, 0.9999} <= set(grid)
    assert np.any(np.isclose(grid, 1e-6 / 20, rtol=0.0, atol=0.0))


def test_induced_grid_contains_endpoints_neighbors_midpoints_and_stationary_points():
    grid = production_interval_grid(10, 0.05, "wilson")
    p, origins = induced_probability_grid(10, grid.lower, grid.upper)
    assert p[0] == 0.0 and p[-1] == 1.0
    assert {1, 2, 3} <= set(origins)
    assert np.all(np.diff(p) > 0.0)


def test_small_clopper_pearson_grid_never_undercovers_nominal():
    for n in (1, 2, 5, 10):
        grid = production_interval_grid(n, 0.05, "clopper_pearson")
        p = base_probability_grid(n, linear_step=0.01)
        coverage, _, _ = coverage_from_intervals(n, p, grid.lower, grid.upper)
        assert np.min(coverage) >= 0.95 - 1e-12


def test_one_n_shard_is_deterministic_and_preserves_not_calibrated_metadata():
    first = calibrate_n(3, linear_step=0.1, expected_widths=True, oracle=True)
    second = calibrate_n(3, linear_step=0.1, expected_widths=True, oracle=True)
    assert first["interval_hash"] == second["interval_hash"]
    assert first["summaries"] == second["summaries"]
    assert not first["high_precision_triggers"] or all(
        "trigger" in row for row in first["high_precision_triggers"]
    )
    wald_oracles = [
        row
        for row in first["oracles"]
        if row["method"] == "wald"
        and row["oracle"] == "independent_unclipped_formula"
    ]
    assert wald_oracles
    assert max(
        max(row["lower_error"], row["upper_error"])
        for row in wald_oracles
    ) <= 1e-15
