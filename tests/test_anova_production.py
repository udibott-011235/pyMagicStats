from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import stats

from pyMagicStat.assumptions import Estimand, InferenceDesign, InferenceValidator
from pyMagicStat.inference import ANOVAResult, MethodSelector, OneWayANOVA, WelchANOVA
from pyMagicStat.inference.decision import InferenceDecisionStatus


def _groups():
    return (
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([2.0, 3.0, 4.0, 5.0, 7.0]),
        np.array([4.0, 5.0, 6.0, 7.0, 8.0]),
    )


def test_classical_matches_scipy_and_components():
    groups = _groups()
    result = OneWayANOVA(*groups, independence="assumed").run()
    oracle = stats.f_oneway(*groups)

    assert isinstance(result, ANOVAResult)
    assert result.statistic == pytest.approx(oracle.statistic, rel=1e-13, abs=1e-15)
    assert result.p_value == pytest.approx(oracle.pvalue, rel=1e-13, abs=1e-15)
    assert result.numerator_df == 2.0
    assert result.denominator_df == 12.0
    assert result.components["ss_total"] == pytest.approx(
        result.components["ss_between"] + result.components["ss_within"]
    )
    assert 0.0 <= result.components["eta_squared"] <= 1.0
    assert result.method_version == "classical-one-way-anova-v1"


def test_classical_k2_equals_pooled_t_squared():
    a = np.array([1.1, 2.3, 3.2, 4.1, 5.4])
    b = np.array([2.2, 2.9, 4.8, 5.1, 6.7, 7.0])
    result = OneWayANOVA(a, b, independence="assumed").run()
    t = stats.ttest_ind(a, b, equal_var=True)

    assert result.statistic == pytest.approx(float(t.statistic) ** 2, rel=1e-12)
    assert result.p_value == pytest.approx(float(t.pvalue), rel=1e-12)
    assert result.numerator_df == 1.0
    assert result.denominator_df == len(a) + len(b) - 2


def test_welch_k2_equals_welch_t_squared():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 5.0, 9.0, 13.0, 17.0, 21.0])
    result = WelchANOVA(a, b, independence="assumed").run()
    t = stats.ttest_ind(a, b, equal_var=False)

    assert result.statistic == pytest.approx(float(t.statistic) ** 2, rel=1e-12)
    assert result.p_value == pytest.approx(float(t.pvalue), rel=1e-12)
    assert result.numerator_df == 1.0
    assert result.denominator_df == pytest.approx(float(t.df), rel=1e-12)
    assert result.components["welch_correction"] == pytest.approx(1.0)
    assert result.method_version == "welch-one-way-anova-v1"


def test_welch_formula_components_are_reproducible():
    groups = _groups()
    result = WelchANOVA(*groups, independence="assumed").run()
    n = np.array([len(group) for group in groups], dtype=float)
    means = np.array([np.mean(group) for group in groups], dtype=float)
    variances = np.array([np.var(group, ddof=1) for group in groups], dtype=float)
    weights = n / variances
    W = weights.sum()
    weighted_mean = float(np.dot(weights, means) / W)
    k = len(groups)
    B = float(np.sum((1.0 - weights / W) ** 2 / (n - 1.0)))
    correction = 1.0 + 2.0 * (k - 2) / (k**2 - 1.0) * B
    F = float(np.sum(weights * (means - weighted_mean) ** 2) / (k - 1) / correction)
    df2 = float((k**2 - 1.0) / (3.0 * B))

    assert result.components["weights"] == pytest.approx(tuple(weights))
    assert result.components["weighted_mean"] == pytest.approx(weighted_mean)
    assert result.components["welch_B"] == pytest.approx(B)
    assert result.components["welch_correction"] == pytest.approx(correction)
    assert result.statistic == pytest.approx(F)
    assert result.denominator_df == pytest.approx(df2)
    assert result.p_value == pytest.approx(stats.f.sf(F, k - 1, df2))


@pytest.mark.parametrize("klass", [OneWayANOVA, WelchANOVA])
def test_invalid_alpha_and_too_few_groups_rejected(klass):
    group = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="alpha"):
        klass(group, group + 1.0, alpha=0.0)
    with pytest.raises(ValueError, match="at least two groups"):
        klass(group)


@pytest.mark.parametrize("klass", [OneWayANOVA, WelchANOVA])
def test_nonfinite_and_constant_groups_rejected(klass):
    valid = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        klass(valid, np.array([1.0, np.nan, 3.0]))
    with pytest.raises(ValueError):
        klass(valid, np.array([2.0, 2.0, 2.0]))


@pytest.mark.parametrize("klass", [OneWayANOVA, WelchANOVA])
def test_input_mutation_after_construction_does_not_change_result(klass):
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 6.0])
    engine = klass(a, b, independence="assumed")
    before = engine.run()
    a[:] = 999.0
    b[:] = -999.0
    after = engine.run()
    assert before.to_dict() == after.to_dict()


def test_result_is_deeply_effectively_immutable():
    result = OneWayANOVA(*_groups(), independence="assumed").run()
    with pytest.raises(FrozenInstanceError):
        result.statistic = 0.0
    with pytest.raises(TypeError):
        result.components["ss_between"] = 0.0
    with pytest.raises(TypeError):
        result.diagnostics["strict"] = False


def test_to_dict_is_json_ready_and_detached():
    result = WelchANOVA(*_groups(), independence="assumed").run()
    payload = result.to_dict()
    assert isinstance(payload["group_sizes"], list)
    assert isinstance(payload["components"]["weights"], list)
    payload["components"]["weights"][0] = -1.0
    assert result.components["weights"][0] > 0.0


def test_common_translation_scale_and_permutations_are_invariant():
    groups = _groups()
    for klass in (OneWayANOVA, WelchANOVA):
        baseline = klass(*groups, independence="assumed").run()
        translated = klass(
            *(group + 1000.0 for group in groups), independence="assumed"
        ).run()
        scaled = klass(*(group * -7.0 for group in groups), independence="assumed").run()
        reordered = klass(*reversed(groups), independence="assumed").run()
        permuted_within = klass(
            *(group[::-1] for group in groups), independence="assumed"
        ).run()

        for other in (translated, scaled, reordered, permuted_within):
            assert other.statistic == pytest.approx(baseline.statistic, rel=1e-12, abs=1e-14)
            assert other.p_value == pytest.approx(baseline.p_value, rel=1e-12, abs=1e-14)


def test_equal_means_constructed_case_has_zero_f_and_p_one():
    a = np.array([-1.0, 0.0, 1.0])
    b = np.array([-2.0, 0.0, 2.0])
    c = np.array([-3.0, 0.0, 3.0])
    for klass in (OneWayANOVA, WelchANOVA):
        result = klass(a, b, c, independence="assumed").run()
        assert result.statistic == pytest.approx(0.0, abs=1e-15)
        assert result.p_value == pytest.approx(1.0, abs=1e-15)


def test_diagnostic_shape_failure_does_not_auto_block_explicit_execution():
    severe = np.array([0.0] * 18 + [1.0, 1000.0])
    other = np.linspace(0.0, 2.0, 20)
    result = OneWayANOVA(severe, other, independence="assumed", strict=True).run()
    assert np.isfinite(result.statistic)
    assert any(name.startswith("shape_") for name in result.diagnostics["diagnostic_flags"])


def test_unknown_independence_is_reported_not_falsely_validated():
    result = OneWayANOVA(*_groups(), independence="unknown").run()
    assert result.diagnostics["independence"] == "unknown"
    assert "independence" in result.diagnostics["unresolved_assumptions"]
    assert result.diagnostics["automatic_selection_calibrated"] is False


def test_selector_one_way_remains_not_calibrated():
    validation = InferenceValidator().validate_one_way(*_groups(), independence="assumed")
    assert validation.report.design is InferenceDesign.ONE_WAY
    assert validation.report.estimand is Estimand.GROUP_MEAN_DIFFERENCES
    decision = MethodSelector().select(validation.report)
    assert decision.status is InferenceDecisionStatus.NOT_CALIBRATED
    assert decision.selected_method is None


def test_repeated_execution_is_deterministic():
    for klass in (OneWayANOVA, WelchANOVA):
        engine = klass(*_groups(), independence="assumed")
        assert engine.run().to_dict() == engine.run().to_dict()
