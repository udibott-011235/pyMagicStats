from experiments.robustness_calibration import scenarios, simulate, summarize


def test_calibration_matrix_contains_required_families_and_sample_sizes():
    families = {scenario.family for scenario in scenarios()}

    assert {
        "normal",
        "exponential",
        "gamma",
        "lognormal",
        "student_t",
        "laplace",
        "mixture",
        "contamination",
    } <= families


def test_calibration_is_reproducible_and_records_engine_diagnostics():
    selected = (scenarios()[0], scenarios()[1])
    kwargs = {
        "replications": 3,
        "sample_sizes": (10, 20),
        "seed": 1234,
        "selected_scenarios": selected,
    }

    first = simulate(**kwargs)
    second = simulate(**kwargs)

    assert first == second
    assert len(first) == 12
    assert {
        "skewness",
        "excess_kurtosis",
        "outlier_fraction",
        "shape_status",
        "robustness_decision",
        "ci_covered",
        "type_i_rejection",
    } <= first[0].keys()

    summary = summarize(first)
    assert len(summary) == 4
    assert all(0.0 <= row["ci_coverage"] <= 1.0 for row in summary)
    assert all(0.0 <= row["type_i_error"] <= 1.0 for row in summary)
