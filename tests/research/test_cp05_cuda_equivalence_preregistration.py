"""Deterministic C2B preregistration checks; no GPU workload is executed."""
import json
from pathlib import Path

import numpy as np
from experiments.distribution_gof.cuda_calibration.equivalence_preregistration import (
    ADVERSARIAL_FIXTURES, B_EQ, GENERATOR_SANITY_CASES, GENERATOR_SANITY_N, PRIMARY_CELL_COUNT, R_EQ,
    batch_agreement, categorical_agreement, fit_agreement, generator_sanity_check,
    global_equivalence_pass, initial_summary, mc_agreement, nb_flat_objective_diagnostic,
    nb_objective_agreement, nb_objective_tolerance, primary_fixture_matrix,
    seed_identity_agreement, statistic_agreement, theoretical_moments,
)

def test_frozen_primary_matrix_and_execution_sizes():
    cells = primary_fixture_matrix()
    assert len(cells) == PRIMARY_CELL_COUNT == 144
    assert {cell.n for cell in cells} == {20, 50, 100, 250}
    assert R_EQ == 8 and B_EQ == 15 and GENERATOR_SANITY_N == 1_000_000

def test_categorical_seed_mc_and_batch_rules_are_exact():
    assert categorical_agreement("ELIGIBLE", "ELIGIBLE")
    assert not categorical_agreement("ELIGIBLE", "FAILED")
    assert seed_identity_agreement((1, 2), (1, 2))
    assert not seed_identity_agreement((1, 2), (2, 1))
    assert mc_agreement(3, 3, False, False)
    assert not mc_agreement(3, 2, False, False)
    row = {"outer_indices": [0], "inner_indices": [0, 1], "classifications": ["ELIGIBLE"], "replicate_counts": [2], "retry_accounting": [0]}
    assert batch_agreement(row, dict(row))

def test_frozen_numeric_and_flat_nb_rules():
    assert statistic_agreement(1.0, 1.0 + 1.5e-11)
    assert not statistic_agreement(1.0, 1.0 + 3e-11)
    cpu, cuda = {"r": 2.0, "p": 0.4}, {"r": 2.2, "p": 0.45}
    ll = lambda parameters: -10.0
    result = fit_agreement("negative_binomial", cpu, cuda, cpu_log_likelihood=-10.0,
                           cuda_log_likelihood=-10.0, reference_log_likelihood=ll,
                           same_eligibility=True, downstream_passed=True)
    assert result[1] == "PARAMETERIZATION_DIFFERENCE_ON_FLAT_OBJECTIVE"

def test_artifact_summary_is_non_claiming_and_adversarial_set_is_frozen():
    summary = initial_summary()
    assert summary["equivalence_gate_passed"] is False
    assert summary["calibration_claim"] is False
    assert "mc_exact_tie" in ADVERSARIAL_FIXTURES
    assert len(GENERATOR_SANITY_CASES) == 7

def _fixtures():
    path = Path("experiments/distribution_gof/cuda_calibration/cp05_c2b_adversarial_fixtures.json")
    return json.loads(path.read_text(encoding="utf-8"))["fixtures"]

def test_nb_objective_uses_cpu_floor_and_boundary_cases():
    for ll_cpu in (0.0, 0.25, -12.0):
        tau = nb_objective_tolerance(ll_cpu)
        # Zero has an exactly representable boundary; other cases use the
        # adjacent in-bound float because addition itself rounds in float64.
        delta = tau if ll_cpu == 0.0 else tau * 0.99
        assert nb_objective_agreement(ll_cpu, ll_cpu + delta)
        assert nb_objective_agreement(ll_cpu, ll_cpu + tau * 0.999)
        assert not nb_objective_agreement(ll_cpu, ll_cpu + tau * 1.001)

def test_flat_objective_is_computed_on_nine_point_path_not_caller_boolean():
    cpu, cuda = {"r": 1.0, "p": 0.5}, {"r": 1.2, "p": 0.55}
    flat = nb_flat_objective_diagnostic(cpu, cuda, reference_log_likelihood=lambda _: -4.0)
    assert len(flat["t_grid"]) == len(flat["LL_PATH"]) == 9
    assert flat["FLAT_OBJECTIVE"] is True
    decision = fit_agreement("negative_binomial", cpu, cuda, cpu_log_likelihood=-4.0,
                             cuda_log_likelihood=-4.0, reference_log_likelihood=lambda _: -4.0,
                             same_eligibility=True, downstream_passed=False)
    assert decision[0] is False
    nonflat = nb_flat_objective_diagnostic(cpu, cuda, reference_log_likelihood=lambda p: -(p["r"] - 1.0) ** 2)
    assert nonflat["FLAT_OBJECTIVE"] is False

def test_materialized_adversarial_fixtures_and_nb_mc_boundaries():
    fixtures = _fixtures()
    assert set(ADVERSARIAL_FIXTURES) == set(fixtures)
    below = np.asarray(fixtures["nb_variance_just_below_mean"]["sample"], dtype=float)
    equal = np.asarray(fixtures["nb_variance_equal_mean"]["sample"], dtype=float)
    above = np.asarray(fixtures["nb_variance_just_above_mean"]["sample"], dtype=float)
    assert np.var(below, ddof=1) < np.mean(below)
    # Same rational quantity (1/20); float64 reduction is compared at roundoff.
    assert np.isclose(np.var(equal, ddof=1), np.mean(equal), rtol=0.0, atol=1e-15)
    assert np.var(above, ddof=1) > np.mean(above)
    cliff = fixtures["mc_near_comparison_cliff"]
    assert cliff["below"] < cliff["T_obs"] == cliff["equal"] < cliff["above"]
    assert mc_agreement(2, 2, False, False)

def test_theoretical_moments_and_generator_sanity_formula():
    gamma = theoretical_moments("gamma", {"shape": 2.0, "scale": 1.0})
    exponential = theoretical_moments("exponential", {"scale": 1.0})
    nb = theoretical_moments("negative_binomial", {"r": 1.0, "p": 0.5})
    assert gamma == (2.0, 2.0, 24.0)
    assert exponential == (1.0, 1.0, 9.0)
    assert nb == (1.0, 2.0, 38.0)
    r, p, q = 1.0, 0.5, 0.5
    assert (1 + 4 * q + q ** 2) / (r * q) == 6 / r + p ** 2 / (r * q)
    check = generator_sanity_check(1.0, 1.0, "exponential", {"scale": 1.0})
    assert check["passed"] is True and check["z_mean"] == check["z_variance"] == 0.0
    assert global_equivalence_pass((True, True)) is True
    assert global_equivalence_pass((True, False)) is False
