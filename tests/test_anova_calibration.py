from experiments.anova_calibration import SCENARIOS, run_matrix


def test_anova_calibration_matrix_covers_required_scenario_families():
    names = {scenario.name for scenario in SCENARIOS}
    assert {
        "normal_equal_balanced",
        "normal_equal_unbalanced",
        "normal_unequal_balanced",
        "normal_small_group_high_variance",
        "normal_large_group_high_variance",
        "gamma_moderate",
        "exponential_severe",
        "lognormal_severe",
        "student_t_df3",
        "laplace",
        "mixture_symmetric",
        "mixture_skewed",
        "outlier_contamination_5pct",
    }.issubset(names)


def test_anova_calibration_is_reproducible_and_records_decisions():
    kwargs = {
        "replications": 3,
        "nominal_sizes": [10],
        "effect_sizes": [0.0, 0.8],
        "seeds": [20260827],
        "scenarios": SCENARIOS[:2],
    }
    first = run_matrix(**kwargs)
    second = run_matrix(**kwargs)

    assert first == second
    assert len(first) == 4
    for row in first:
        assert row["policy_version"] == "anova-v1-2026-08"
        assert row["hypothesis"] in {"H0", "H1"}
        assert 0.0 <= row["welch_selection_rate"] <= 1.0
        assert 0.0 <= row["insufficient_rate"] <= 1.0
        assert row["welch_selection_rate"] + row["insufficient_rate"] == 1.0
