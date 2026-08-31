import warnings
import json
import pickle

import mpmath as mp
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from experiments.proportion_ci_calibration.harness import (
    ALPHAS,
    CANDIDATE_SHA,
    ENDPOINT_CACHE_SCHEMA_VERSION,
    HARNESS_SCHEMA_VERSION,
    IntervalGrid,
    EndpointGridCache,
    acceptance_runs,
    base_probability_grid,
    calibrate_n,
    coverage_from_intervals,
    expected_width_matrix,
    endpoints_are_monotone,
    endpoint_proximity,
    evaluate_coverage,
    induced_probability_grid,
    jeffreys_interval_grid,
    outcome_set_probability,
    probability_grid_with_origins,
    production_interval_grid,
)
from experiments.proportion_ci_calibration.high_precision import (
    audit_structural_predicate,
    build_high_precision_queue,
    classify_coverage_verdict,
    classify_endpoint_verdict,
    high_precision_binomial_range,
    high_precision_interval,
    high_precision_binomial_runs,
    reconcile_boundary_acceptance,
    run_audit,
)
from experiments.proportion_ci_calibration import high_precision as hp_module
from experiments.proportion_ci_calibration import run as run_module
from experiments.proportion_ci_calibration.run import (
    SHARD_SCHEMA_VERSION,
    build_cache_provenance,
    checkpoint_spec,
    load_shard_cache,
    save_shard_cache,
    shard_semantic_hash,
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


def test_monotone_coverage_routes_to_contiguous_fast_path_and_matches_brute_force():
    n = 12
    probabilities = np.asarray((0.0, 0.01, 0.2, 0.5, 0.99, 1.0))
    grid = production_interval_grid(n, 0.05, "wilson")
    result = evaluate_coverage(n, probabilities, grid.lower, grid.upper)
    brute = np.asarray(
        [
            np.sum(
                stats.binom.pmf(np.arange(n + 1), n, probability)
                * ((grid.lower <= probability) & (probability <= grid.upper))
            )
            for probability in probabilities
        ]
    )
    assert endpoints_are_monotone(grid.lower, grid.upper)
    assert result.endpoint_monotone
    assert result.acceptance_kind == "monotone_contiguous"
    np.testing.assert_allclose(result.coverage, brute, atol=2e-15, rtol=0.0)


def test_wald_nonmonotone_coverage_uses_explicit_set_and_matches_brute_force():
    n = 5
    probabilities = np.asarray((0.0, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0))
    grid = production_interval_grid(n, 0.05, "wald")
    result = evaluate_coverage(
        n,
        probabilities,
        grid.lower,
        grid.upper,
        probability_batch_size=2,
        outcome_batch_size=2,
    )
    brute = np.asarray(
        [
            np.sum(
                stats.binom.pmf(np.arange(n + 1), n, probability)
                * ((grid.lower <= probability) & (probability <= grid.upper))
            )
            for probability in probabilities
        ]
    )
    assert not endpoints_are_monotone(grid.lower, grid.upper)
    assert not result.endpoint_monotone
    assert result.acceptance_kind == "explicit_nonmonotone_endpoints"
    np.testing.assert_allclose(result.coverage, brute, atol=2e-15, rtol=0.0)


def test_noncontiguous_acceptance_differs_from_forced_first_last_range():
    n = 3
    probability = np.asarray((0.3,))
    lower = np.asarray((0.0, 0.8, 0.2, 0.9))
    upper = np.asarray((0.4, 1.0, 0.6, 1.0))
    explicit = evaluate_coverage(
        n,
        probability,
        lower,
        upper,
        probability_batch_size=1,
        outcome_batch_size=2,
    )
    runs = acceptance_runs(lower, upper, probability[0])
    brute = stats.binom.pmf(0, n, probability[0]) + stats.binom.pmf(
        2, n, probability[0]
    )
    forced_contiguous = stats.binom.cdf(2, n, probability[0])
    assert runs == [(0, 0), (2, 2)]
    assert explicit.run_count[0] == 2
    assert explicit.coverage[0] == pytest.approx(brute, abs=2e-15)
    assert abs(explicit.coverage[0] - forced_contiguous) > 0.1


def test_noncontiguous_partition_adds_deterministic_bounded_optimizer_candidates():
    lower = np.asarray((0.0, 0.8, 0.2, 0.9))
    upper = np.asarray((0.4, 1.0, 0.6, 1.0))
    probabilities, origins = induced_probability_grid(3, lower, upper)
    assert 5 in origins
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_endpoint_proximity_detects_near_upper_missed_by_lower_first_check():
    lower = np.asarray((0.0, 0.1, 0.2))
    upper = np.asarray((0.4, 0.6, 0.8))
    probability = np.nextafter(0.6, 1.0)
    proximity = endpoint_proximity(probability, lower, upper)
    nearest = proximity["matches"][0]
    assert proximity["is_near"]
    assert nearest["relation"] == "nextafter_above"
    assert nearest["kinds"] == ["upper"]
    assert abs(probability - lower[2]) > 1e-10


def test_endpoint_proximity_classifies_lower_upper_complements_deterministically():
    lower = np.asarray((0.1, 0.4))
    upper = np.asarray((0.6, 0.9))
    below_lower = endpoint_proximity(np.nextafter(0.4, 0.0), lower, upper)
    above_upper = endpoint_proximity(
        1.0 - np.nextafter(0.4, 0.0),
        lower,
        upper,
    )
    assert below_lower["nearest_relation"] == "nextafter_below"
    assert below_lower["nearest_kinds"] == ["lower"]
    assert above_upper["nearest_relation"] == "nextafter_above"
    assert above_upper["nearest_kinds"] == ["upper"]


def test_explicit_coverage_is_worker_batch_invariant():
    n = 20
    grid = production_interval_grid(n, 0.01, "wald")
    probabilities = np.linspace(0.0, 1.0, 101)
    narrow = evaluate_coverage(
        n,
        probabilities,
        grid.lower,
        grid.upper,
        probability_batch_size=1,
        outcome_batch_size=3,
    )
    broad = evaluate_coverage(
        n,
        probabilities,
        grid.lower,
        grid.upper,
        probability_batch_size=64,
        outcome_batch_size=64,
    )
    np.testing.assert_allclose(narrow.coverage, broad.coverage, atol=2e-15, rtol=0.0)
    np.testing.assert_array_equal(narrow.first, broad.first)
    np.testing.assert_array_equal(narrow.last, broad.last)
    np.testing.assert_array_equal(narrow.run_count, broad.run_count)


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


def test_high_precision_explicit_runs_do_not_fill_noncontiguous_gaps():
    observed = high_precision_binomial_runs(
        3,
        0.3,
        [(0, 0), (2, 2)],
        digits=80,
    )
    expected = stats.binom.pmf(0, 3, 0.3) + stats.binom.pmf(2, 3, 0.3)
    assert abs(float(observed) - expected) <= 2e-15


def _synthetic_endpoint_provider(endpoints):
    def provider(method, n, x, alpha, *, digits):
        del method, n, alpha, digits
        return mp.mpf(endpoints[x][0]), mp.mpf(endpoints[x][1])

    return provider


def _mapped_float_provider(endpoints):
    def provider(method, n, x, alpha):
        del method, n
        lower, upper = endpoints[(float(alpha), x)]
        return float(lower), float(upper)

    return provider


def _mapped_hp_provider(endpoints):
    def provider(method, n, x, alpha, *, digits):
        del method, n, digits
        lower, upper = endpoints[(float(alpha), x)]
        return mp.mpf(lower), mp.mpf(upper)

    return provider


def test_hp_boundary_reconciliation_preserves_equal_acceptance_set():
    lower = np.asarray((0.0, 0.3, 0.7))
    upper = np.asarray((0.2, 0.8, 1.0))
    runs = acceptance_runs(lower, upper, 0.5)
    result = reconcile_boundary_acceptance(
        "wilson",
        2,
        0.05,
        0.5,
        lower,
        upper,
        runs,
        endpoint_provider=_synthetic_endpoint_provider(
            (("0", "0.2"), ("0.3", "0.8"), ("0.7", "1"))
        ),
    )
    assert result["consistent_float_representation"]
    assert not result["acceptance_changed"]
    assert result["float_runs"] == result["hp_runs"] == [(1, 1)]


def test_hp_boundary_reconciliation_changes_synthetic_float64_inclusion():
    lower = np.asarray((np.nextafter(0.0, 1.0), 0.6))
    upper = np.asarray((0.4, 1.0))
    result = reconcile_boundary_acceptance(
        "wilson",
        1,
        0.05,
        0.0,
        lower,
        upper,
        [],
        endpoint_provider=_synthetic_endpoint_provider(
            (("0", "0.4"), ("0.6", "1"))
        ),
    )
    assert result["consistent_float_representation"]
    assert result["acceptance_changed"]
    assert result["float_runs"] == []
    assert result["hp_runs"] == [(0, 0)]
    verdict = classify_coverage_verdict(
        "wilson",
        100,
        0.05,
        0.0,
        mp.mpf(1),
        acceptance_changed=True,
        consistent_float_representation=True,
    )
    assert verdict["classification"] == "float64_boundary_artifact"
    assert verdict["resolved"]


def test_hp_verdict_confirms_wilson_shortfall_and_cp_exact_coverage():
    wilson = classify_coverage_verdict(
        "wilson",
        100,
        0.05,
        0.90,
        mp.mpf("0.90"),
        acceptance_changed=False,
        consistent_float_representation=True,
    )
    exact = classify_coverage_verdict(
        "clopper_pearson",
        100,
        0.05,
        0.96,
        mp.mpf("0.96"),
        acceptance_changed=False,
        consistent_float_representation=True,
    )
    assert wilson["classification"] == "confirmed_statistical_shortfall"
    assert wilson["resolved"]
    assert exact["classification"] == "confirmed_exact_coverage"
    assert exact["resolved"]


def test_nonexact_methods_use_cell_specific_no_shortfall_classification():
    for method in ("wilson", "jeffreys"):
        verdict = classify_coverage_verdict(
            method,
            100,
            0.05,
            0.96,
            mp.mpf("0.96"),
            acceptance_changed=False,
            consistent_float_representation=True,
        )
        assert verdict["classification"] == "confirmed_no_shortfall_at_audited_cell"
        assert verdict["resolved"]
        if method == "jeffreys":
            assert "Bayesian comparator" in verdict["notes"]


STRUCTURAL_ROWS = {
    "complement_symmetry": {
        "reason": "complement_symmetry",
        "method": "wilson",
        "n": 1,
        "alpha": 0.05,
        "x": 0,
        "complement_x": 1,
    },
    "endpoint_monotonicity": {
        "reason": "endpoint_monotonicity",
        "method": "wilson",
        "n": 1,
        "alpha": 0.05,
        "x": 0,
        "x_left": 0,
        "x_right": 1,
        "endpoint_kind": "lower",
    },
    "nesting": {
        "reason": "nesting",
        "method": "wilson",
        "n": 1,
        "alpha": 0.025,
        "x": 0,
        "alpha_wider": 0.025,
        "alpha_narrower": 0.05,
    },
}

STRUCTURAL_FLOAT_ENDPOINTS = {
    "complement_symmetry": {
        (0.05, 0): ("0", "0.4"),
        (0.05, 1): ("0.7", "1"),
    },
    "endpoint_monotonicity": {
        (0.05, 0): ("0.3", "0.6"),
        (0.05, 1): ("0.2", "0.9"),
    },
    "nesting": {
        (0.025, 0): ("0.3", "0.7"),
        (0.05, 0): ("0.2", "0.8"),
    },
}

STRUCTURAL_HP_RESTORED = {
    "complement_symmetry": {
        (0.05, 0): ("0", "0.4"),
        (0.05, 1): ("0.6", "1"),
    },
    "endpoint_monotonicity": {
        (0.05, 0): ("0.1", "0.6"),
        (0.05, 1): ("0.2", "0.9"),
    },
    "nesting": {
        (0.025, 0): ("0.1", "0.9"),
        (0.05, 0): ("0.2", "0.8"),
    },
}


@pytest.mark.parametrize("reason", tuple(STRUCTURAL_ROWS))
def test_float64_structural_violation_disappears_in_hp(reason):
    result = audit_structural_predicate(
        STRUCTURAL_ROWS[reason],
        digits=80,
        float_endpoint_provider=_mapped_float_provider(
            STRUCTURAL_FLOAT_ENDPOINTS[reason]
        ),
        endpoint_provider=_mapped_hp_provider(STRUCTURAL_HP_RESTORED[reason]),
    )
    assert result["predicate_evaluated"]
    assert not result["predicate_float64"]
    assert result["predicate_hp"]
    assert result["classification"] == "float64_structural_artifact"
    assert result["resolved"]
    assert len(json.loads(result["paired_values"])) == 2


@pytest.mark.parametrize("reason", tuple(STRUCTURAL_ROWS))
def test_genuine_structural_violation_persists_in_hp_and_is_unresolved(reason):
    result = audit_structural_predicate(
        STRUCTURAL_ROWS[reason],
        digits=80,
        float_endpoint_provider=_mapped_float_provider(
            STRUCTURAL_FLOAT_ENDPOINTS[reason]
        ),
        endpoint_provider=_mapped_hp_provider(STRUCTURAL_FLOAT_ENDPOINTS[reason]),
    )
    assert result["predicate_evaluated"]
    assert not result["predicate_float64"]
    assert not result["predicate_hp"]
    assert result["classification"] == "confirmed_structural_violation"
    assert not result["resolved"]
    assert result["status"] == "unresolved"


@pytest.mark.parametrize(
    ("reason", "float_values", "restored_values", "persistent_values"),
    (
        ("bounds", ("-0.01", "0.8"), ("0", "0.8"), ("-0.01", "0.8")),
        ("nan_or_inf", ("nan", "0.8"), ("0", "0.8"), ("nan", "0.8")),
    ),
)
def test_bounds_and_nonfinite_predicates_require_an_hp_conclusion(
    reason, float_values, restored_values, persistent_values
):
    row = {
        "reason": reason,
        "method": "wilson",
        "n": 1,
        "alpha": 0.05,
        "x": 0,
    }
    float_mapping = {(0.05, 0): float_values}
    restored = audit_structural_predicate(
        row,
        float_endpoint_provider=_mapped_float_provider(float_mapping),
        endpoint_provider=_mapped_hp_provider({(0.05, 0): restored_values}),
    )
    persistent = audit_structural_predicate(
        row,
        float_endpoint_provider=_mapped_float_provider(float_mapping),
        endpoint_provider=_mapped_hp_provider({(0.05, 0): persistent_values}),
    )
    assert restored["classification"] == "float64_structural_artifact"
    assert restored["resolved"]
    assert persistent["classification"] == "confirmed_structural_violation"
    assert not persistent["resolved"]


def test_structural_predicate_with_inconsistent_pair_context_is_unresolved():
    row = {**STRUCTURAL_ROWS["complement_symmetry"], "complement_x": 0}
    result = audit_structural_predicate(
        row,
        float_endpoint_provider=_mapped_float_provider(
            STRUCTURAL_FLOAT_ENDPOINTS["complement_symmetry"]
        ),
        endpoint_provider=_mapped_hp_provider(
            STRUCTURAL_HP_RESTORED["complement_symmetry"]
        ),
    )
    assert not result["predicate_evaluated"]
    assert result["classification"] == "unresolved"
    assert not result["resolved"]


def test_hp_endpoint_verdict_resolves_numerical_oracle_without_claim_change():
    verdict = classify_endpoint_verdict(
        "oracle_discrepancy",
        100,
        0.2,
        0.8,
        mp.mpf("0.20000000000001"),
        mp.mpf("0.79999999999999"),
    )
    assert verdict["classification"] == "numerical_difference_without_claim_change"
    assert verdict["resolved"]


def test_hp_verdict_is_unresolved_for_inconsistent_float_acceptance():
    verdict = classify_coverage_verdict(
        "wilson",
        100,
        0.05,
        0.90,
        mp.mpf("0.90"),
        acceptance_changed=False,
        consistent_float_representation=False,
    )
    assert verdict["classification"] == "unresolved"
    assert not verdict["resolved"]


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
    with pytest.raises(ValueError, match="at least 80"):
        audit_structural_predicate(STRUCTURAL_ROWS["complement_symmetry"], digits=79)


def test_structural_queue_preserves_all_paired_predicate_context(tmp_path, monkeypatch):
    prefix = tmp_path / "proportion_ci_cp06_x"
    pd.DataFrame({"n": pd.Series(dtype="int64")}).to_parquet(
        f"{prefix}_high_precision_triggers.parquet", index=False
    )
    pd.DataFrame({"n": pd.Series(dtype="int64")}).to_parquet(
        f"{prefix}_oracles.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "n": 2,
                "alpha": 0.05,
                "method": "wilson",
                "nan_count": 0,
                "max_complement_error": 0.2,
                "bounds_failures": 0,
                "lower_monotonic_failures": 1,
                "upper_monotonic_failures": 0,
            }
        ]
    ).to_parquet(f"{prefix}_invariants.parquet", index=False)
    pd.DataFrame(
        [
            {
                "n": 2,
                "method": "wilson",
                "alpha_wider": 0.025,
                "alpha_narrower": 0.05,
                "lower_nesting_failures": 1,
                "upper_nesting_failures": 1,
            }
        ]
    ).to_parquet(f"{prefix}_nesting.parquet", index=False)
    grids = {
        0.05: IntervalGrid(
            np.asarray((0.0, 0.3, 0.2)),
            np.asarray((0.4, 0.8, 1.0)),
            "wilson",
            "frequentist",
            "synthetic",
        ),
        0.025: IntervalGrid(
            np.asarray((0.0, 0.4, 0.2)),
            np.asarray((0.4, 0.6, 1.0)),
            "wilson",
            "frequentist",
            "synthetic",
        ),
    }

    def fake_grid(cache, n, alpha, method):
        del cache, n, method
        return grids[float(alpha)]

    monkeypatch.setattr(hp_module, "_grid_for", fake_grid)
    queue = build_high_precision_queue(("X",), results_dir=tmp_path)
    symmetry = queue[queue["reason"] == "complement_symmetry"].iloc[0]
    monotonicity = queue[queue["reason"] == "endpoint_monotonicity"].iloc[0]
    nesting = queue[queue["reason"] == "nesting"].iloc[0]
    assert symmetry["audit_kind"] == "structural"
    assert (int(symmetry["x"]), int(symmetry["complement_x"])) == (0, 2)
    assert (int(monotonicity["x_left"]), int(monotonicity["x_right"])) == (1, 2)
    assert monotonicity["endpoint_kind"] == "lower"
    assert nesting["alpha_wider"] == 0.025
    assert nesting["alpha_narrower"] == 0.05
    assert int(nesting["x"]) == 1


def test_high_precision_queue_collects_coverage_and_oracle_triggers(tmp_path):
    prefix = tmp_path / "proportion_ci_cp06_x"
    trigger_grid = production_interval_grid(5, 0.05, "wilson")
    trigger_p = 0.2
    trigger_runs = acceptance_runs(
        trigger_grid.lower,
        trigger_grid.upper,
        trigger_p,
    )
    trigger_coverage = np.sum(
        stats.binom.pmf(np.arange(6), 5, trigger_p)
        * ((trigger_grid.lower <= trigger_p) & (trigger_p <= trigger_grid.upper))
    )
    pd.DataFrame(
        [
            {
                "n": 5,
                "alpha": 0.05,
                "method": "wilson",
                "p": trigger_p,
                "first_x": trigger_runs[0][0],
                "last_x": trigger_runs[-1][1],
                "coverage": trigger_coverage,
                "trigger": "material_minimum_or_endpoint",
                "acceptance_kind": "monotone_contiguous",
                "acceptance_runs": json.dumps(trigger_runs),
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
    coverage_audit = audit[audit["audit_kind"] == "coverage"].iloc[0]
    assert {
        "p_float64",
        "p_hp",
        "coverage_float64",
        "coverage_hp",
        "deficit_float64",
        "deficit_hp",
        "endpoint_relation",
        "classification",
        "resolved",
        "notes",
    } <= set(audit.columns)
    assert coverage_audit["resolved"]


def _minimal_cache_part(n):
    return {
        "n": n,
        "interval_hash": "a" * 64,
        "summaries": [],
        "worst_cases": [],
        "adversarial_minima": [],
        "wald_pathology_summaries": [],
        "wald_pathology_worst_cases": [],
    }


def test_shard_cache_rejects_legacy_and_incompatible_schema(tmp_path):
    path = tmp_path / "shard.pickle"
    with path.open("wb") as stream:
        pickle.dump(_minimal_cache_part(5), stream)
    expected = build_cache_provenance("B", 5)
    assert load_shard_cache(path, expected) is None

    save_shard_cache(path, _minimal_cache_part(5), expected)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    payload["provenance"]["shard_schema_version"] = "cp06-shard-schema-v2"
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    assert load_shard_cache(path, expected) is None

    save_shard_cache(path, _minimal_cache_part(5), expected)
    incompatible = {**expected, "harness_schema_version": "obsolete-schema"}
    assert load_shard_cache(path, incompatible) is None

    with path.open("rb") as stream:
        payload = pickle.load(stream)
    payload["result_bytes"] += b"corrupt"
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    assert load_shard_cache(path, expected) is None


def test_shard_cache_resumes_compatible_payload_and_proves_cross_checkpoint_semantics(
    tmp_path,
):
    path = tmp_path / "shard.pickle"
    part = _minimal_cache_part(5)
    provenance_b = build_cache_provenance("B", 5)
    provenance_c = build_cache_provenance("C", 5)
    assert provenance_b["shard_schema_version"] == SHARD_SCHEMA_VERSION
    assert SHARD_SCHEMA_VERSION == "cp06-shard-schema-v3"
    assert shard_semantic_hash(5, checkpoint_spec("B")) == shard_semantic_hash(
        5, checkpoint_spec("C")
    )
    save_shard_cache(path, part, provenance_b)
    assert load_shard_cache(path, provenance_b) == part
    assert load_shard_cache(path, provenance_c) is None
    assert (
        load_shard_cache(path, provenance_c, allow_cross_checkpoint=True) == part
    )


def test_endpoint_cache_validates_hash_provenance_and_reuses_api_grid(tmp_path):
    cache = EndpointGridCache(tmp_path)
    calls = []

    def factory():
        calls.append("called")
        return production_interval_grid(5, 0.05, "wilson")

    first = cache.get_or_create(5, 0.05, "wilson", factory)
    second = cache.get_or_create(
        5,
        0.05,
        "wilson",
        lambda: pytest.fail("compatible endpoint grid should resume"),
    )
    assert calls == ["called"]
    np.testing.assert_array_equal(first.lower, second.lower)
    np.testing.assert_array_equal(first.upper, second.upper)

    path = cache.path_for(5, 0.05, "wilson")
    with np.load(path, allow_pickle=False) as payload:
        lower = payload["lower"].copy()
        upper = payload["upper"].copy()
        metadata_text = str(payload["metadata"].item())
    metadata = json.loads(metadata_text)
    assert metadata["candidate_sha"] == CANDIDATE_SHA
    assert metadata["harness_schema_version"] == ENDPOINT_CACHE_SCHEMA_VERSION
    assert ENDPOINT_CACHE_SCHEMA_VERSION == "cp06-harness-schema-v2"
    assert HARNESS_SCHEMA_VERSION == "cp06-harness-schema-v3"
    assert len(metadata["endpoint_sha256"]) == 64

    lower[0] = np.nextafter(lower[0], 1.0)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            lower=lower,
            upper=upper,
            metadata=np.asarray(metadata_text),
        )
    assert cache.load(5, 0.05, "wilson") is None


def test_wald_pathology_probabilities_are_probability_weighted():
    n = 5
    probabilities = np.asarray((0.0, 0.01, 0.2, 0.5, 0.99, 1.0))
    grid = production_interval_grid(n, 0.05, "wald")
    outside = (grid.lower < 0.0) | (grid.upper > 1.0)
    degenerate = grid.upper == grid.lower
    observed_outside = outcome_set_probability(n, probabilities, outside)
    observed_degenerate = outcome_set_probability(n, probabilities, degenerate)
    outcomes = np.arange(n + 1)
    expected_outside = np.asarray(
        [stats.binom.pmf(outcomes, n, p)[outside].sum() for p in probabilities]
    )
    expected_degenerate = np.asarray(
        [stats.binom.pmf(outcomes, n, p)[degenerate].sum() for p in probabilities]
    )
    np.testing.assert_allclose(observed_outside, expected_outside, atol=2e-15, rtol=0.0)
    np.testing.assert_allclose(
        observed_degenerate,
        expected_degenerate,
        atol=2e-15,
        rtol=0.0,
    )
    assert not np.array_equal(outside.astype(float), observed_outside)
    assert observed_degenerate[0] == 1.0 and observed_degenerate[-1] == 1.0


def test_checkpoint_e_writes_dedicated_adversarial_schema(tmp_path, monkeypatch):
    part = calibrate_n(
        2,
        linear_step=None,
        expected_widths=False,
        oracle=False,
        include_base_grid=False,
        batch_size=3,
    )
    monkeypatch.setattr(run_module, "RESULTS", tmp_path)
    run_module._write_checkpoint(
        "E",
        [part],
        0.0,
        workers=1,
        batch_size=3,
        spec=checkpoint_spec("E"),
    )
    path = tmp_path / "proportion_ci_cp06_e_adversarial_minima.parquet"
    assert path.exists()
    frame = pd.read_parquet(path)
    required = {
        "n",
        "alpha",
        "method",
        "p",
        "coverage",
        "nominal",
        "deficit",
        "acceptance_kind",
        "first_x",
        "last_x",
        "acceptance_runs",
        "acceptance_representation",
        "origin",
        "search_method",
        "optimizer_status",
    }
    assert required <= set(frame.columns)
    assert len(frame) == len(ALPHAS) * 4
    assert not frame.duplicated(["method", "alpha", "n"]).any()
    wald = frame[frame["method"] == "wald"]
    for row in wald.to_dict("records"):
        grid = production_interval_grid(2, row["alpha"], "wald")
        expected_kind = (
            "monotone_contiguous"
            if endpoints_are_monotone(grid.lower, grid.upper)
            else "explicit_nonmonotone_endpoints"
        )
        assert row["acceptance_kind"] == expected_kind
        if not endpoints_are_monotone(grid.lower, grid.upper):
            assert row["search_method"].startswith("explicit_acceptance_set_")
    assert "explicit_nonmonotone_endpoints" in set(wald["acceptance_kind"])
