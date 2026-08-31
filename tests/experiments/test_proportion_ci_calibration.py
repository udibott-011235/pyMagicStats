import warnings

import mpmath as mp
import numpy as np
import pandas as pd
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
    probability_grid_with_origins,
    production_interval_grid,
)
from experiments.proportion_ci_calibration.high_precision import (
    build_high_precision_queue,
    high_precision_binomial_range,
    high_precision_interval,
    run_audit,
)
from experiments.proportion_ci_calibration.run import checkpoint_spec
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


def test_cp06_c_d_e_checkpoint_domains_are_frozen_without_execution():
    checkpoint_c = checkpoint_spec("C")
    checkpoint_d = checkpoint_spec("D")
    checkpoint_e = checkpoint_spec("E")

    assert checkpoint_c.n_values == tuple(range(1, 5_001))
    assert checkpoint_c.linear_step == 0.0001
    assert checkpoint_c.expected_widths and checkpoint_c.oracle
    assert checkpoint_c.resume_sources == ("B",)

    assert checkpoint_d.n_values == (
        7_500,
        10_000,
        15_000,
        20_000,
        30_000,
        50_000,
        75_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
    )
    assert checkpoint_d.linear_step == 0.0001

    assert checkpoint_e.n_values[:5] == (1, 2, 3, 4, 5)
    assert checkpoint_e.n_values[4_999] == 5_000
    assert checkpoint_e.n_values[5_000:] == checkpoint_d.n_values
    assert checkpoint_e.linear_step is None
    assert not checkpoint_e.expected_widths
    assert not checkpoint_e.oracle
    assert not checkpoint_e.include_base_grid


def test_adversarial_mode_contains_only_induced_partition_candidates():
    interval = production_interval_grid(12, 0.05, "wilson")
    expected_p, expected_origin = induced_probability_grid(
        12,
        interval.lower,
        interval.upper,
    )
    observed_p, observed_origin = probability_grid_with_origins(
        12,
        interval,
        linear_step=None,
        include_base_grid=False,
    )
    np.testing.assert_array_equal(observed_p, expected_p)
    np.testing.assert_array_equal(observed_origin, expected_origin)
    assert {1, 2, 3, 4} <= set(observed_origin)


@pytest.mark.parametrize(
    ("n", "probability", "first", "last"),
    (
        (5, 1e-12, 0, 1),
        (12, 0.31, 2, 7),
        (101, 0.99, 95, 101),
        (101, 1.0 - 1e-12, 99, 101),
    ),
)
def test_high_precision_binomial_range_matches_full_pmf(n, probability, first, last):
    observed = high_precision_binomial_range(
        n,
        probability,
        first,
        last,
        digits=80,
    )
    outcomes = np.arange(first, last + 1)
    expected = np.sum(stats.binom.pmf(outcomes, n, probability))
    assert abs(float(observed) - expected) <= 5e-15


@pytest.mark.parametrize("method", ("wilson", "clopper_pearson", "wald", "jeffreys"))
@pytest.mark.parametrize(("n", "x"), ((5, 0), (5, 2), (101, 99), (101, 101)))
def test_high_precision_endpoint_oracles_match_float64_references(method, n, x):
    lower, upper = high_precision_interval(method, n, x, 0.05, digits=80)
    if method == "jeffreys":
        reference = jeffreys_interval_grid(n, 0.05)
    else:
        reference = production_interval_grid(n, 0.05, method)
    assert abs(float(lower) - reference.lower[x]) <= 5e-14
    assert abs(float(upper) - reference.upper[x]) <= 5e-14
    assert mp.isfinite(lower) and mp.isfinite(upper)


def test_high_precision_refuses_less_than_preregistered_precision():
    with pytest.raises(ValueError, match="at least 80"):
        high_precision_binomial_range(5, 0.5, 0, 5, digits=79)
    with pytest.raises(ValueError, match="at least 80"):
        high_precision_interval("wilson", 5, 2, 0.05, digits=79)


def test_high_precision_queue_collects_coverage_and_oracle_triggers(tmp_path):
    prefix = tmp_path / "proportion_ci_cp06_x"
    pd.DataFrame(
        [
            {
                "n": 5,
                "alpha": 0.05,
                "method": "wilson",
                "p": 0.2,
                "first_x": 0,
                "last_x": 3,
                "coverage": 0.94,
                "trigger": "material_minimum_or_endpoint",
            }
        ]
    ).to_parquet(f"{prefix}_high_precision_triggers.parquet", index=False)
    pd.DataFrame(
        [
            {
                "n": 5,
                "x": 2,
                "alpha": 0.05,
                "method": "wilson",
                "oracle": "synthetic",
                "gate_applicable": True,
                "lower_error": 2e-12,
                "upper_error": 0.0,
            },
            {
                "n": 5,
                "x": 3,
                "alpha": 0.05,
                "method": "wilson",
                "oracle": "within_tolerance",
                "gate_applicable": True,
                "lower_error": 1e-15,
                "upper_error": 1e-15,
            },
        ]
    ).to_parquet(f"{prefix}_oracles.parquet", index=False)
    pd.DataFrame(
        [
            {
                "n": 5,
                "alpha": 0.05,
                "method": "wilson",
                "nan_count": 0,
                "max_complement_error": 0.0,
                "bounds_failures": 0,
                "lower_monotonic_failures": 0,
                "upper_monotonic_failures": 0,
            }
        ]
    ).to_parquet(f"{prefix}_invariants.parquet", index=False)
    pd.DataFrame(
        [
            {
                "n": 5,
                "method": "wilson",
                "alpha_wider": 0.025,
                "alpha_narrower": 0.05,
                "lower_nesting_failures": 0,
                "upper_nesting_failures": 0,
            }
        ]
    ).to_parquet(f"{prefix}_nesting.parquet", index=False)

    queue = build_high_precision_queue(("X",), results_dir=tmp_path)
    assert list(queue["audit_kind"]).count("coverage") == 1
    assert list(queue["audit_kind"]).count("endpoint") == 3
    assert set(queue["reason"]) == {
        "material_minimum_or_endpoint",
        "oracle_discrepancy",
        "acceptance_first_endpoint",
        "acceptance_last_endpoint",
    }

    persisted_queue, audit = run_audit(
        ("X",),
        workers=1,
        digits=80,
        results_dir=tmp_path,
    )
    assert len(persisted_queue) == len(queue) == len(audit)
    assert set(audit["status"]) == {"resolved"}
    assert set(audit["digits"]) == {80}
