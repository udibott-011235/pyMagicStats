import json
import warnings

import numpy as np
import pytest
import scipy.stats as stats

import pyMagicStat.inference as inference
from pyMagicStat.assumptions import (
    AssumptionReport,
    Estimand,
    InferenceDesign,
    InferenceValidator,
    SamplingRobustnessV3,
)
from pyMagicStat.distributions.distributions import (
    BinomialDistribution,
    PoissonDistribution,
)
from pyMagicStat.inference import (
    BootstrapCI,
    InferenceDecisionStatus,
    InferenceGuarantee,
    MethodSelector,
    PopulationProportionCI,
    capabilities_for,
    empirical_likelihood_mean_ci,
)
from pyMagicStat.inference.parametric import (
    PopulationProportionCI as HistoricalPopulationProportionCI,
)


def _proportion_report() -> AssumptionReport:
    return AssumptionReport(
        design=InferenceDesign.ONE_SAMPLE,
        estimand=Estimand.PROPORTION,
        assessments={},
    )


def _from_counts(successes, trials, **kwargs):
    return PopulationProportionCI.from_counts(
        successes,
        trials,
        **kwargs,
    ).calculate_interval()


def test_cp05_01_default_remains_wilson():
    assert PopulationProportionCI([0, 1]).calculate_interval()["method"] == "wilson"


def test_cp05_02_legacy_wilson_limits_are_preserved():
    data = np.array([1] * 30 + [0] * 20, dtype=float)
    result = PopulationProportionCI(data, alpha=0.05).calculate_interval()
    p_hat = 0.6
    z_value = stats.norm.ppf(0.975)
    z_squared = z_value**2
    denominator = 1.0 + z_squared / 50
    center = (p_hat + z_squared / 100) / denominator
    half_width = (
        z_value
        * np.sqrt(p_hat * (1.0 - p_hat) / 50 + z_squared / (4.0 * 50**2))
        / denominator
    )
    assert result["lb"] == pytest.approx(center - half_width)
    assert result["ub"] == pytest.approx(center + half_width)


def test_cp05_03_zero_legacy_incidences_preserves_historical_result():
    with pytest.warns(DeprecationWarning, match="from_counts"):
        result = PopulationProportionCI(np.ones(20), incidences=0).calculate_interval()
    assert result["estimate"] == 0.0
    assert result["lb"] == 0.0


def test_cp05_04_wald_formula_and_absence_of_clipping_are_preserved():
    with pytest.warns(UserWarning, match="Wald-specific legacy"):
        result = _from_counts(1, 10, method="wald")
    z_value = stats.norm.ppf(0.975)
    margin = z_value * np.sqrt(0.1 * 0.9 / 10)
    assert result["lb"] == pytest.approx(0.1 - margin)
    assert result["ub"] == pytest.approx(0.1 + margin)
    assert result["lb"] < 0.0


def test_cp05_05_wald_boundary_degeneracy_is_preserved():
    with pytest.warns(UserWarning):
        zero = _from_counts(0, 10, method="wald")
    with pytest.warns(UserWarning):
        one = _from_counts(10, 10, method="wald")
    assert (zero["lb"], zero["ub"]) == (0.0, 0.0)
    assert (one["lb"], one["ub"]) == (1.0, 1.0)


def test_cp05_06_method_names_remain_case_insensitive():
    assert _from_counts(4, 10, method="WiLsOn")["method"] == "wilson"


def test_cp05_07_callable_preserves_historical_counting():
    result = PopulationProportionCI(
        [1, 2, 3, 4],
        incidences=lambda value: value >= 3,
    ).calculate_interval()
    assert result["successes"] == 2
    assert result["estimate"] == 0.5


def test_cp05_08_raw_boolean_data_remains_supported():
    result = PopulationProportionCI([True, False, True]).calculate_interval()
    assert result["successes"] == 2
    assert result["estimate"] == pytest.approx(2 / 3)


def test_cp05_09_raw_integer_and_float_binary_data_are_equivalent():
    integers = PopulationProportionCI([0, 1, 1, 0]).calculate_interval()
    floats = PopulationProportionCI([0.0, 1.0, 1.0, 0.0]).calculate_interval()
    assert integers["lb"] == floats["lb"]
    assert integers["ub"] == floats["ub"]
    assert integers["estimate"] == floats["estimate"]


def test_cp05_10_empty_multidimensional_and_nonfinite_raw_data_fail():
    invalid = ([], [[0, 1]], [0, np.nan], [0, np.inf])
    for data in invalid:
        with pytest.raises(ValueError, match="finite one-dimensional"):
            PopulationProportionCI(data)


def test_cp05_11_nonbinary_raw_data_fails():
    with pytest.raises(ValueError, match="binary"):
        PopulationProportionCI([0, 0.5, 1])


def test_cp05_12_legacy_numeric_count_outside_sample_fails():
    for count in (-1, 4):
        with pytest.raises(ValueError, match="between 0 and the sample size"):
            PopulationProportionCI([0, 0, 0], incidences=count)


def test_cp05_13_six_legacy_result_structures_remain_present():
    result = _from_counts(4, 10)
    assert {"lb", "ub", "method", "estimate", "n", "assumptions"} <= result.keys()


def test_cp05_14_four_legacy_assumption_keys_remain_present():
    assumptions = _from_counts(4, 10)["assumptions"]
    assert {
        "successes",
        "failures",
        "normal_approximation_adequate",
        "normal_approximation_required",
    } <= assumptions.keys()


def test_cp05_15_caller_input_is_not_mutated():
    data = np.array([0.0, 1.0, 1.0, 0.0])
    before = data.copy()
    PopulationProportionCI(data).calculate_interval()
    np.testing.assert_array_equal(data, before)


def test_cp05_16_from_counts_zero_of_one_is_valid_for_wilson():
    result = _from_counts(0, 1)
    assert result["n"] == 1
    assert 0.0 <= result["lb"] <= result["ub"] <= 1.0


def test_cp05_17_from_counts_one_of_one_is_valid_for_wilson():
    result = _from_counts(1, 1)
    assert result["estimate"] == 1.0
    assert 0.0 <= result["lb"] <= result["ub"] <= 1.0


def test_cp05_18_clopper_pearson_accepts_both_boundaries():
    zero = _from_counts(0, 12, method="clopper_pearson")
    one = _from_counts(12, 12, method="clopper_pearson")
    assert zero["lb"] == 0.0
    assert one["ub"] == 1.0


def test_cp05_19_from_counts_uses_trials_and_discrete_estimate():
    result = _from_counts(3, 17)
    assert result["estimate"] == pytest.approx(3 / 17)
    assert result["n"] == 17


def test_cp05_20_from_counts_accepts_numpy_integers():
    result = _from_counts(np.int64(3), np.int32(8))
    assert result["successes"] == 3
    assert result["failures"] == 5


def test_cp05_21_from_counts_rejects_booleans():
    with pytest.raises(ValueError, match="not bool"):
        PopulationProportionCI.from_counts(True, 5)
    with pytest.raises(ValueError, match="not bool"):
        PopulationProportionCI.from_counts(1, False)


def test_cp05_22_from_counts_rejects_floats_including_integral_values():
    with pytest.raises(ValueError, match="not bool or float"):
        PopulationProportionCI.from_counts(3.0, 5)
    with pytest.raises(ValueError, match="not bool or float"):
        PopulationProportionCI.from_counts(3, 5.0)


def test_cp05_23_from_counts_rejects_nonpositive_trials():
    for trials in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            PopulationProportionCI.from_counts(0, trials)


def test_cp05_24_from_counts_rejects_successes_outside_support():
    for successes in (-1, 4):
        with pytest.raises(ValueError, match="0 <= successes <= trials"):
            PopulationProportionCI.from_counts(successes, 3)


def test_cp05_25_from_counts_does_not_fabricate_raw_data():
    interval = PopulationProportionCI.from_counts(2, 5)
    assert not hasattr(interval, "data")


def test_cp05_26_wilson_matches_scipy_on_deterministic_grid():
    for n in (1, 2, 5, 20, 100):
        for x in sorted({0, n // 2, n}):
            for alpha in (0.01, 0.05, 0.2):
                result = _from_counts(x, n, alpha=alpha)
                oracle = stats.binomtest(x, n).proportion_ci(
                    confidence_level=1.0 - alpha,
                    method="wilson",
                )
                assert result["lb"] == pytest.approx(oracle.low, abs=1e-14)
                assert result["ub"] == pytest.approx(oracle.high, abs=1e-14)


def test_cp05_27_clopper_pearson_matches_scipy_exact_grid():
    for n in (1, 2, 5, 20):
        for x in range(n + 1):
            result = _from_counts(x, n, method="clopper_pearson")
            oracle = stats.binomtest(x, n).proportion_ci(
                confidence_level=0.95,
                method="exact",
            )
            assert result["lb"] == pytest.approx(oracle.low, abs=1e-14)
            assert result["ub"] == pytest.approx(oracle.high, abs=1e-14)


def test_cp05_28_wilson_bounds_stay_in_unit_interval_on_cp01_subset():
    for alpha in (0.01, 0.05, 0.1):
        for n in range(1, 51):
            for x in range(n + 1):
                result = _from_counts(x, n, alpha=alpha)
                assert 0.0 <= result["lb"] <= result["ub"] <= 1.0


def test_cp05_29_wilson_complement_symmetry_is_float64_stable():
    for n in (1, 3, 10, 100):
        for x in range(n + 1):
            direct = _from_counts(x, n)
            complement = _from_counts(n - x, n)
            assert direct["lb"] == pytest.approx(1.0 - complement["ub"], abs=1e-14)
            assert direct["ub"] == pytest.approx(1.0 - complement["lb"], abs=1e-14)


def test_cp05_30_wilson_limits_are_monotone_in_successes():
    limits = [_from_counts(x, 50) for x in range(51)]
    assert all(left["lb"] <= right["lb"] for left, right in zip(limits, limits[1:]))
    assert all(left["ub"] <= right["ub"] for left, right in zip(limits, limits[1:]))


def test_cp05_31_clopper_pearson_complement_symmetry_is_preserved():
    for n in (1, 5, 20):
        for x in range(n + 1):
            direct = _from_counts(x, n, method="clopper_pearson")
            complement = _from_counts(n - x, n, method="clopper_pearson")
            assert direct["lb"] == pytest.approx(1.0 - complement["ub"], abs=1e-14)
            assert direct["ub"] == pytest.approx(1.0 - complement["lb"], abs=1e-14)


def test_cp05_32_wald_out_of_range_case_prevents_accidental_clipping():
    with pytest.warns(UserWarning):
        result = _from_counts(1, 10, method="wald")
    assert result["lb"] < 0.0


def test_cp05_33_unauthorized_method_names_fail_explicitly():
    for name in ("exact", "beta", "jeffreys", "agresti_coull", "wilsoncc", "midp"):
        with pytest.raises(ValueError, match="method must be"):
            PopulationProportionCI.from_counts(2, 5, method=name)


def test_cp05_34_clopper_pearson_rejects_fractional_legacy_incidences():
    with pytest.warns(DeprecationWarning, match="outside the supported"):
        with pytest.raises(ValueError, match="integer binomial successes"):
            PopulationProportionCI(
                np.zeros(10),
                incidences=3.7,
                method="clopper_pearson",
            )


def test_cp05_35_confidence_level_is_one_minus_alpha():
    result = _from_counts(2, 5, alpha=0.2)
    assert result["confidence_level"] == pytest.approx(0.8)


def test_cp05_36_estimand_metadata_is_proportion():
    assert _from_counts(2, 5)["estimand"] == "proportion"


def test_cp05_37_design_metadata_is_one_sample():
    assert _from_counts(2, 5)["design"] == "one_sample"


def test_cp05_38_sampling_model_metadata_is_bernoulli_binomial():
    assert _from_counts(2, 5)["sampling_model"] == "bernoulli_binomial"


def test_cp05_39_each_method_reports_approved_interval_kind():
    expected = {
        "wilson": "frequentist_score",
        "clopper_pearson": "frequentist_exact_conservative",
        "wald": "frequentist_asymptotic_legacy",
    }
    for method, interval_kind in expected.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = _from_counts(20, 40, method=method)
        assert result["interval_kind"] == interval_kind


def test_cp05_40_every_method_remains_not_calibrated_before_cp06():
    for method in ("wilson", "clopper_pearson", "wald"):
        result = _from_counts(20, 40, method=method)
        assert result["calibration_status"] == "not_calibrated"


def test_cp05_41_top_level_and_assumption_counts_agree():
    result = _from_counts(7, 13)
    assert result["successes"] == result["assumptions"]["successes"]
    assert result["failures"] == result["assumptions"]["failures"]


def test_cp05_42_input_mode_distinguishes_every_contractual_route():
    raw = PopulationProportionCI([0, 1]).calculate_interval()
    predicate = PopulationProportionCI([1, 2], incidences=lambda x: x > 1).calculate_interval()
    counts = _from_counts(1, 2)
    with pytest.warns(DeprecationWarning):
        legacy = PopulationProportionCI([0, 0], incidences=1).calculate_interval()
    with pytest.warns(DeprecationWarning):
        fractional = PopulationProportionCI([0, 0], incidences=0.5).calculate_interval()
    assert [
        raw["input_mode"],
        predicate["input_mode"],
        counts["input_mode"],
        legacy["input_mode"],
        fractional["input_mode"],
    ] == [
        "binary_data",
        "predicate",
        "counts",
        "legacy_incidences_count",
        "legacy_fractional_incidences",
    ]


def test_cp05_43_design_requirements_do_not_claim_independence_was_proved():
    result = _from_counts(2, 5)
    assert result["assumptions"]["independence"] == "unknown"
    assert result["design_requirements"] == [
        "independent_units",
        "common_success_probability",
        "bernoulli_binomial_sampling",
    ]


def test_cp05_44_independence_values_are_preserved_and_invalid_values_fail():
    for value in ("unknown", "assumed", "verified"):
        result = _from_counts(2, 5, independence=value)
        assert result["assumptions"]["independence"] == value
    with pytest.raises(ValueError, match="independence must be"):
        PopulationProportionCI.from_counts(2, 5, independence="inferred")


def test_cp05_45_complete_result_is_json_serializable():
    json.dumps(_from_counts(2, 5, method="clopper_pearson"))


def test_cp05_46_fractional_legacy_route_is_not_binomial_supported():
    with pytest.warns(DeprecationWarning):
        result = PopulationProportionCI([0] * 10, incidences=3.7).calculate_interval()
    assert result["compatibility"]["binomial_contract_supported"] is False
    assert result["calibration_status"] == "not_calibrated"


def test_cp05_47_canonical_input_routes_are_binomial_supported():
    results = [
        PopulationProportionCI([0, 1]).calculate_interval(),
        PopulationProportionCI([1, 2], incidences=lambda x: x == 2).calculate_interval(),
        _from_counts(1, 2),
    ]
    assert all(item["compatibility"]["binomial_contract_supported"] for item in results)


def test_cp05_48_only_wald_is_marked_as_a_legacy_method():
    for method in ("wilson", "clopper_pearson"):
        assert _from_counts(10, 20, method=method)["compatibility"]["legacy_method"] is False
    assert _from_counts(10, 20, method="wald")["compatibility"]["legacy_method"] is True


def test_cp05_49_integral_legacy_incidences_warns_and_recommends_from_counts():
    with pytest.warns(DeprecationWarning, match="from_counts") as recorded:
        result = PopulationProportionCI([0] * 5, incidences=2).calculate_interval()
    assert recorded[0].filename.endswith("test_proportion_ci_contract.py")
    assert result["compatibility"]["deprecated"] is True


def test_cp05_50_fractional_legacy_warning_declares_model_incompatibility():
    with pytest.warns(DeprecationWarning, match="outside the supported Bernoulli/binomial"):
        result = PopulationProportionCI([0] * 5, incidences=2.5).calculate_interval()
    assert "ordinary binomial" in result["compatibility"]["deprecation_reason"]


def test_cp05_51_callable_does_not_emit_numeric_api_deprecation():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        PopulationProportionCI([1, 2], incidences=lambda x: x == 2)
    assert not any(item.category is DeprecationWarning for item in recorded)


def test_cp05_52_from_counts_does_not_emit_legacy_api_warning():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        PopulationProportionCI.from_counts(10, 20)
    assert not recorded


def test_cp05_53_wald_low_counts_warning_is_specific_and_noncalibrating():
    with pytest.warns(
        UserWarning,
        match="Wald-specific legacy.*not a calibration guarantee.*selection",
    ):
        PopulationProportionCI.from_counts(1, 10, method="wald")


def test_cp05_54_nonwald_methods_do_not_emit_the_legacy_wald_warning():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        PopulationProportionCI.from_counts(1, 10, method="wilson")
        PopulationProportionCI.from_counts(1, 10, method="clopper_pearson")
    assert not any(item.category is UserWarning for item in recorded)


def test_cp05_55_public_inference_import_works():
    assert inference.PopulationProportionCI is PopulationProportionCI


def test_cp05_56_historical_import_points_to_the_same_class():
    assert HistoricalPopulationProportionCI is PopulationProportionCI


def test_cp05_57_public_all_contains_population_proportion_ci():
    assert "PopulationProportionCI" in inference.__all__


def test_cp05_58_v2_selector_fails_closed_for_proportion():
    decision = MethodSelector().select(_proportion_report())
    assert decision.selected_method is None


def test_cp05_59_proportion_routing_status_and_guarantee_are_not_calibrated():
    decision = MethodSelector().select(_proportion_report())
    assert decision.status is InferenceDecisionStatus.NOT_CALIBRATED
    assert decision.guarantee is InferenceGuarantee.NOT_CALIBRATED


def test_cp05_60_proportion_routing_has_no_mean_alternatives():
    decision = MethodSelector().select(_proportion_report())
    assert decision.alternatives == ()
    assert "mean-inference" in " ".join(decision.reasons)


def test_cp05_61_proportion_routing_is_not_parametric_recommended():
    assert MethodSelector().select(_proportion_report()).parametric_recommended is False


def test_cp05_62_v3_selector_also_fails_closed_for_proportion():
    decision = MethodSelector(SamplingRobustnessV3()).select(_proportion_report())
    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.NOT_CALIBRATED
    assert decision.guarantee is InferenceGuarantee.NOT_CALIBRATED


def test_cp05_63_proportion_fail_closed_does_not_evaluate_mean_robustness():
    class FailingMeanPolicy:
        POLICY_VERSION = "must-not-run"

        def evaluate(self, report):
            raise AssertionError("mean shape/outlier policy must not run")

    decision = MethodSelector(FailingMeanPolicy()).select(_proportion_report())
    assert decision.selected_method is None


def test_cp05_64_existing_one_sample_mean_routing_is_unchanged():
    report = InferenceValidator().validate_one_sample(
        np.linspace(-1.0, 1.0, 21),
        independence="assumed",
    ).report
    assert MethodSelector().select(report).selected_method == "one_sample_t"


def test_cp05_65_one_way_remains_not_calibrated():
    report = AssumptionReport(
        design=InferenceDesign.ONE_WAY,
        estimand=Estimand.GROUP_MEAN_DIFFERENCES,
        assessments={},
    )
    decision = MethodSelector().select(report)
    assert decision.selected_method is None
    assert decision.status is InferenceDecisionStatus.NOT_CALIBRATED


def test_cp05_66_proportion_capability_registry_remains_empty():
    assert capabilities_for(InferenceDesign.ONE_SAMPLE, Estimand.PROPORTION) == ()


def test_cp05_67_bootstrap_proportion_is_not_a_population_ci_method():
    assert BootstrapCI is not PopulationProportionCI
    with pytest.raises(ValueError, match="method must be"):
        PopulationProportionCI.from_counts(2, 5, method="bootstrap")


def test_cp05_68_bootstrap_is_not_an_automatic_proportion_alternative():
    serialized = MethodSelector().select(_proportion_report()).to_dict()
    assert serialized["alternatives"] == []
    assert "bootstrap" not in str(serialized["reasons"]).lower()


def test_cp05_69_empirical_likelihood_result_contract_is_unchanged():
    result = empirical_likelihood_mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.method == "empirical_likelihood"
    assert result.estimate == 3.0
    assert result.lower < result.estimate < result.upper


def test_cp05_70_one_way_contract_is_not_redefined_by_proportion_work():
    report = InferenceValidator().validate_one_way(
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        independence="assumed",
    ).report
    decision = MethodSelector().select(report)
    assert decision.status is InferenceDecisionStatus.NOT_CALIBRATED
    assert decision.selected_method is None


def test_cp05_71_binomial_and_poisson_gof_contracts_remain_available():
    binomial = BinomialDistribution(np.array([0, 1] * 20, dtype=int))
    poisson = PoissonDistribution(np.array([0, 1, 2, 3] * 20, dtype=int))
    assert binomial.validate_data() is True
    assert poisson.validate_data() is True
    assert "binomial" in binomial.evaluate_goodness_of_fit(n=1, p=0.5)["hypothesis"].lower()
    assert "poisson" in poisson.evaluate_goodness_of_fit()["hypothesis"].lower()


@pytest.mark.parametrize("alpha", (0.0, 1.0, -0.1, np.nan, np.inf))
def test_cp05_extra_invalid_alpha_fails(alpha):
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        PopulationProportionCI.from_counts(1, 2, alpha=alpha)


def test_cp05_extra_nonnumeric_legacy_incidences_fails_closed():
    with pytest.raises(ValueError, match="real count"):
        PopulationProportionCI([0, 1], incidences="1")
