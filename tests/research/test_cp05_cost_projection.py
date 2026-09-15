from __future__ import annotations

from experiments.distribution_gof.cp05_c1_preflight import (
    B_VALUES,
    CELLS,
    COMPARATOR_CONFIGURATION_COUNT,
    ELIGIBLE_OUTER_TARGET,
    FULL_CONFIGURATION_COUNT,
    FULL_DEVELOPMENT_CELL_COUNT,
    PRIMARY_CONFIGURATION_COUNT,
    STATISTICS,
    _configurations,
    _full_matrix_projection,
)


def _summaries() -> list[dict]:
    rows = []
    for cell_index, cell in enumerate(CELLS, start=1):
        for statistic_index, statistic in enumerate(STATISTICS, start=1):
            for B in B_VALUES:
                cost = float(cell_index * 10 + statistic_index + B / 1000)
                is_nb = cell.family == "negative_binomial"
                rows.append(
                    {
                        "family": cell.family,
                        "statistic": statistic,
                        "B": B,
                        "wall_time_per_assessment_median_seconds": cost,
                        "wall_time_per_assessment_p95_seconds": cost * 2,
                        "peak_memory_bytes": cell_index * 1_000,
                        "NB_eligibility_rate": 0.5 if is_nb else None,
                        "NB_inner_retry_burden": (
                            {
                                "raw_inner_attempts": 250,
                                "eligible_inner": 200,
                            }
                            if is_nb
                            else None
                        ),
                    }
                )
    return rows


def test_frozen_preflight_remains_16_primary_configurations():
    configurations = _configurations()

    assert len(configurations) == 16
    assert {statistic for _, _, statistic, _ in configurations} == {"AD", "CVM"}
    assert not {"KS", "DISCRETE_KS", "PEARSON_HELPER"} & {
        statistic for _, _, statistic, _ in configurations
    }


def test_full_matrix_contract_counts_are_exact():
    assert PRIMARY_CONFIGURATION_COUNT == 576
    assert COMPARATOR_CONFIGURATION_COUNT == 480
    assert FULL_DEVELOPMENT_CELL_COUNT == 144
    assert FULL_CONFIGURATION_COUNT == 1_056
    assert ELIGIBLE_OUTER_TARGET == 2_112_000


def test_projection_includes_measured_and_proxy_cost_classes():
    projection = _full_matrix_projection(_summaries())

    measured = projection["cost_basis_breakdown"]["MEASURED_COST"]
    proxy = projection["cost_basis_breakdown"]["PROXY_PROJECTED_COST"]
    assert measured["configuration_count"] == 576
    assert proxy["configuration_count"] == 480
    assert measured["eligible_outer_assessments"] == 1_152_000
    assert proxy["eligible_outer_assessments"] == 960_000
    assert projection["COMPARATOR_TIMINGS_DIRECTLY_MEASURED"] is False
    assert projection["COMPARATOR_PROJECTION_METHOD"] == "CONSERVATIVE_PRIMARY_MAX_PROXY"
    assert projection["PROJECTION_BASIS"] == "N20_EQUIVALENT_COST_PROJECTION"
    assert projection["SCALING_UNCERTAINTY"] == "UNMEASURED_N_EFFECT"


def test_comparator_cost_is_the_conservative_primary_max_within_family_and_B():
    projection = _full_matrix_projection(_summaries())
    cost_basis = {
        (item["family"], item["B"], item["statistic"]): item
        for item in projection["cost_basis_by_family_B_statistic"]
    }

    for family, comparators in {
        "gamma": ("KS",),
        "exponential": ("KS",),
        "negative_binomial": ("DISCRETE_KS", "PEARSON_HELPER"),
    }.items():
        for B in B_VALUES:
            primary = [cost_basis[(family, B, statistic)] for statistic in STATISTICS]
            expected_median = max(item["median_seconds"] for item in primary)
            expected_p95 = max(item["p95_seconds"] for item in primary)
            for statistic in comparators:
                comparator = cost_basis[(family, B, statistic)]
                assert comparator["median_seconds"] == expected_median
                assert comparator["p95_seconds"] == expected_p95
                assert comparator["cost_class"] == "PROXY_PROJECTED_COST"
                assert comparator["timing_basis"] == "CONSERVATIVE_PRIMARY_MAX_PROXY"


def test_projection_separates_eligible_outer_raw_outer_and_inner_workload():
    projection = _full_matrix_projection(_summaries())

    assert projection["eligible_outer_target"] == 2_112_000
    assert projection["raw_outer_attempt_projection"] > 2_112_000
    assert projection["inner_bootstrap_attempt_projection"] > 2_112_000
    assert projection["composite_refit_attempt_projection"] > 2_112_000
    assert projection["NB_OBSERVED_OUTER_ELIGIBILITY_RATE"] == {
        "199": 0.5,
        "999": 0.5,
    }
    assert projection["NB_OBSERVED_INNER_RETRY_FACTOR"] == {
        "199": 1.25,
        "999": 1.25,
    }
    assert projection["NB_RAW_WORKLOAD_PROJECTION_UNCERTAIN"] is True


def test_parallel_and_memory_scenarios_are_explicitly_non_measured():
    projection = _full_matrix_projection(_summaries())
    wall_time = projection["PROJECTED_CP05_C_FULL_MATRIX_WALL_TIME"]
    memory = projection["PROJECTED_PEAK_MEMORY"]

    assert set(wall_time["scenarios"]) == {"1", "4", "8", "16"}
    assert wall_time["parallel_efficiency_measured"] is False
    assert wall_time["practical_parallel_runtime"] == "NOT_ESTABLISHED"
    serial = wall_time["scenarios"]["1"]["IDEAL_LOWER_BOUND"]
    for workers in (1, 4, 8, 16):
        scenario = wall_time["scenarios"][str(workers)]
        assert scenario["IDEAL_LOWER_BOUND"]["n20_equivalent_p95_seconds"] == (
            serial["n20_equivalent_p95_seconds"] / workers
        )
        assert scenario["PRACTICAL_ESTIMATE"] == "NOT_ESTABLISHED"
        assert scenario["parallel_efficiency_measured"] is False
        assert memory["scenarios"][str(workers)][
            "naive_upper_concurrent_memory_bytes"
        ] == workers * memory["max_observed_peak_process_memory_bytes"]
        assert memory["scenarios"][str(workers)]["measured_concurrent_memory"] is False


def test_no_budget_awaits_owner_resource_budget():
    projection = _full_matrix_projection(_summaries())

    assert projection["RESOURCE_BUDGET"] == "NOT_PREVIOUSLY_AUTHORIZED"
    assert projection["RESOURCE_ASSESSMENT"] == "AWAITING_OWNER_RESOURCE_BUDGET"
    assert projection["budget_violations"] == []


def test_adequate_authorized_budget_passes():
    projection = _full_matrix_projection(
        _summaries(),
        authorized_budget={
            "max_n20_equivalent_p95_seconds": 10**12,
            "max_peak_process_memory_bytes": 10**12,
        },
    )

    assert projection["RESOURCE_ASSESSMENT"] == "PASS"
    assert projection["budget_violations"] == []


def test_inadequate_authorized_budget_blocks_resource():
    projection = _full_matrix_projection(
        _summaries(),
        authorized_budget={
            "max_n20_equivalent_p95_seconds": 1,
            "max_peak_process_memory_bytes": 1,
        },
    )

    assert projection["RESOURCE_ASSESSMENT"] == "BLOCKED_RESOURCE"
    assert {item["metric"] for item in projection["budget_violations"]} == {
        "n20_equivalent_p95_seconds",
        "max_peak_process_memory_bytes",
    }
