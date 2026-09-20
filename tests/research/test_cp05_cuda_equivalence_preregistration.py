"""Deterministic C2B preregistration checks; no GPU workload is executed."""
from experiments.distribution_gof.cuda_calibration.equivalence_preregistration import (
    ADVERSARIAL_FIXTURES, B_EQ, GENERATOR_SANITY_N, PRIMARY_CELL_COUNT, R_EQ,
    batch_agreement, categorical_agreement, fit_agreement, initial_summary,
    mc_agreement, primary_fixture_matrix, seed_identity_agreement, statistic_agreement,
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
    assert fit_agreement("negative_binomial", cpu, cuda, cpu_log_likelihood=-10.0, cuda_log_likelihood=-10.0, flat_objective=True, downstream_passed=True)[1] == "PARAMETERIZATION_DIFFERENCE_ON_FLAT_OBJECTIVE"

def test_artifact_summary_is_non_claiming_and_adversarial_set_is_frozen():
    summary = initial_summary()
    assert summary["equivalence_gate_passed"] is False
    assert summary["calibration_claim"] is False
    assert "mc_exact_tie" in ADVERSARIAL_FIXTURES
