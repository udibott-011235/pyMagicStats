import math

import numpy as np

from experiments.adversarial_robustness_calibration import (
    CalibrationCell,
    calibration_plan,
    map_threshold_cliffs,
    scenario_catalog,
    simulate_cell,
    summarize_cell,
    wilson_interval,
)


def test_scenario_catalog_covers_prespecified_families_and_contamination_levels():
    scenarios = scenario_catalog()
    families = {scenario.family for scenario in scenarios}
    contamination = [
        scenario
        for scenario in scenarios
        if scenario.family.startswith("normal_contamination")
    ]

    assert {
        "normal",
        "student_t",
        "lognormal",
        "normal_contamination_symmetric",
        "normal_contamination_asymmetric",
        "bimodal",
        "normal_mixture",
        "gamma",
    } <= families
    assert {scenario.parameters["epsilon"] for scenario in contamination} == {
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.10,
    }


def test_adaptive_plan_has_critical_cells_and_required_replication_counts():
    cells = {
        (cell.scenario.name, cell.n): cell
        for cell in calibration_plan(
            exploratory_replications=2,
            confirmatory_replications=10,
            minimum_confirmatory_replications=5,
        )
    }

    assert cells[("lognormal_sigma_0.25", 20)].replications == 10
    assert cells[("lognormal_sigma_0.50", 50)].replications == 10
    assert cells[("lognormal_sigma_1.00", 30)].replications == 10
    assert cells[("student_t_df_5", 20)].replications == 10
    assert cells[("bimodal_symmetric", 300)].replications == 10
    assert cells[("normal", 10000)].replications == 5
    assert cells[("normal", 3)].replications == 5


def test_cell_simulation_is_reproducible_and_records_required_metrics():
    scenario = next(item for item in scenario_catalog() if item.name == "normal")
    cell = CalibrationCell(scenario, 8, 4, "test")

    first = simulate_cell(cell, cell_seed=np.random.SeedSequence(1234))
    second = simulate_cell(cell, cell_seed=np.random.SeedSequence(1234))

    assert first == second
    assert len(first) == 4
    assert {
        "n",
        "distribution_family",
        "distribution_parameters",
        "seed",
        "sample_mean",
        "sample_std",
        "skewness",
        "excess_kurtosis",
        "shape_status",
        "departure_magnitude",
        "exact_normality_rejected",
        "outlier_status",
        "extreme_count",
        "extreme_fraction",
        "sampling_robustness_level",
        "selected_method",
        "t_statistic",
        "p_value",
        "reject_h0",
        "ci_lower",
        "ci_upper",
        "ci_contains_true_mean",
        "delta_mean_remove_extremes",
        "influence_ratio",
    } <= first[0].keys()


def test_summary_has_explicit_conditional_denominators_and_binomial_intervals():
    scenario = next(item for item in scenario_catalog() if item.name == "normal")
    records = simulate_cell(
        CalibrationCell(scenario, 10, 8, "test"),
        cell_seed=np.random.SeedSequence(5678),
    )
    summary = summarize_cell(records)

    assert [row["decision_scope"] for row in summary] == [
        "all",
        "acceptable",
        "caution",
        "insufficient",
    ]
    assert summary[0]["conditional_denominator"] == 8
    assert sum(row["conditional_denominator"] for row in summary[1:]) == 8
    for row in summary:
        if row["conditional_denominator"]:
            assert 0.0 <= row["type_i_ci95_lower"] <= row["type_i_ci95_upper"] <= 1.0
            assert 0.0 <= row["coverage_ci95_lower"] <= row["coverage_ci95_upper"] <= 1.0
        else:
            assert math.isnan(row["type_i_error"])


def test_wilson_interval_and_threshold_map_expose_policy_cliffs():
    lower, upper = wilson_interval(5, 10)
    transitions = [row for row in map_threshold_cliffs() if row["transition"]]

    assert 0.2 < lower < 0.5 < upper < 0.8
    assert {
        (row["transition_from"], row["transition_to"])
        for row in transitions
    } >= {
        ("acceptable", "insufficient"),
        ("insufficient", "caution"),
    }
