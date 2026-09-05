import inspect

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.oneway import anova_generic, anova_oneway

from pyMagicStat.inference import OneWayANOVA, WelchANOVA


RTOL = 5e-10
ATOL = 5e-12
OFFSET_RTOL = 5e-8


def _summary_arrays(groups):
    means = np.asarray([np.mean(group) for group in groups], dtype=float)
    variances = np.asarray([np.var(group, ddof=1) for group in groups], dtype=float)
    nobs = np.asarray([len(group) for group in groups], dtype=float)
    return means, variances, nobs


def _manual_classical(groups):
    means, variances, nobs = _summary_arrays(groups)
    k = len(groups)
    n_total = int(np.sum(nobs))
    grand_mean = float(np.dot(nobs, means) / n_total)
    ss_between = float(np.sum(nobs * (means - grand_mean) ** 2))
    ss_within = float(np.sum((nobs - 1.0) * variances))
    df1 = float(k - 1)
    df2 = float(n_total - k)
    ms_between = ss_between / df1
    ms_within = ss_within / df2
    f_stat = float(ms_between / ms_within)
    p_value = float(stats.f.sf(f_stat, df1, df2))
    return {
        "f": f_stat,
        "p": p_value,
        "df1": df1,
        "df2": df2,
        "grand_mean": grand_mean,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_between + ss_within,
        "ms_between": ms_between,
        "ms_within": ms_within,
    }


def _manual_welch(groups):
    means, variances, nobs = _summary_arrays(groups)
    k = len(groups)
    weights = nobs / variances
    w_sum = float(np.sum(weights))
    weighted_mean = float(np.dot(weights, means) / w_sum)
    a = float(np.sum(weights * (means - weighted_mean) ** 2) / (k - 1))
    b = float(np.sum((1.0 - weights / w_sum) ** 2 / (nobs - 1.0)))
    correction = float(1.0 + 2.0 * (k - 2) / (k**2 - 1.0) * b)
    f_stat = float(a / correction)
    df1 = float(k - 1)
    df2 = float((k**2 - 1.0) / (3.0 * b))
    p_value = float(stats.f.sf(f_stat, df1, df2))
    return {
        "f": f_stat,
        "p": p_value,
        "df1": df1,
        "df2": df2,
        "weights": weights,
        "weighted_mean": weighted_mean,
        "b": b,
        "correction": correction,
    }


def _scenarios():
    balanced = (
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        np.array([-1.0, 0.0, 1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    unbalanced = (
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
        np.array([1.0, 1.5, 2.5, 4.0, 6.5, 9.0, 12.0]),
        np.array([-3.0, -1.0, 0.0, 2.0]),
    )
    min_n = (
        np.array([0.0, 2.0]),
        np.array([1.0, 5.0]),
        np.array([-2.0, 4.0]),
    )
    size_variance = (
        np.array([-20.0, 20.0]),
        np.array([-6.0, -3.0, 0.0, 3.0, 6.0]),
        np.linspace(-2.0, 2.0, 17),
        np.linspace(1.0, 2.0, 31),
    )
    many_groups = tuple(
        np.asarray([j + 0.25 * i for j in (-2.0, -0.5, 0.75, 2.25)], dtype=float)
        for i in range(12)
    )
    near_degenerate = (
        np.array([0.0, 1e-7, 2e-7, 4e-7]),
        np.array([1e-7, 3e-7, 5e-7, 8e-7, 13e-7]),
        np.array([-2e-7, 0.0, 3e-7, 7e-7]),
    )
    return {
        "balanced_k3": balanced,
        "unbalanced_k4": unbalanced,
        "min_n2": min_n,
        "size_variance_association": size_variance,
        "many_groups_k12": many_groups,
        "near_degenerate_valid": near_degenerate,
    }


@pytest.mark.parametrize("name,groups", list(_scenarios().items()))
def test_classical_concords_with_manual_scipy_and_statsmodels(name, groups):
    result = OneWayANOVA(*groups, independence="assumed").run()
    manual = _manual_classical(groups)
    scipy_result = stats.f_oneway(*groups)
    sm_raw = anova_oneway(groups, use_var="equal")
    means, variances, nobs = _summary_arrays(groups)
    sm_summary = anova_generic(means, variances, nobs, use_var="equal")

    for oracle_f in (
        manual["f"],
        float(scipy_result.statistic),
        float(sm_raw.statistic),
        float(sm_summary.statistic),
    ):
        assert result.statistic == pytest.approx(oracle_f, rel=RTOL, abs=ATOL), name

    for oracle_p in (
        manual["p"],
        float(scipy_result.pvalue),
        float(sm_raw.pvalue),
        float(sm_summary.pvalue),
    ):
        assert result.p_value == pytest.approx(oracle_p, rel=RTOL, abs=ATOL), name

    assert result.numerator_df == pytest.approx(manual["df1"], rel=RTOL, abs=ATOL)
    assert result.denominator_df == pytest.approx(manual["df2"], rel=RTOL, abs=ATOL)
    assert result.components["ss_between"] == pytest.approx(manual["ss_between"], rel=RTOL, abs=ATOL)
    assert result.components["ss_within"] == pytest.approx(manual["ss_within"], rel=RTOL, abs=ATOL)
    assert result.components["ss_total"] == pytest.approx(manual["ss_total"], rel=RTOL, abs=ATOL)
    assert result.components["mean_square_between"] == pytest.approx(manual["ms_between"], rel=RTOL, abs=ATOL)
    assert result.components["mean_square_within"] == pytest.approx(manual["ms_within"], rel=RTOL, abs=ATOL)
    assert result.numerator_df + result.denominator_df == pytest.approx(result.n_total - 1)


@pytest.mark.parametrize("name,groups", list(_scenarios().items()))
def test_welch_concords_with_manual_and_statsmodels(name, groups):
    result = WelchANOVA(*groups, independence="assumed").run()
    manual = _manual_welch(groups)
    sm_raw = anova_oneway(groups, use_var="unequal", welch_correction=True)
    means, variances, nobs = _summary_arrays(groups)
    sm_summary = anova_generic(
        means,
        variances,
        nobs,
        use_var="unequal",
        welch_correction=True,
    )

    for oracle_f in (manual["f"], float(sm_raw.statistic), float(sm_summary.statistic)):
        assert result.statistic == pytest.approx(oracle_f, rel=RTOL, abs=ATOL), name
    for oracle_p in (manual["p"], float(sm_raw.pvalue), float(sm_summary.pvalue)):
        assert result.p_value == pytest.approx(oracle_p, rel=RTOL, abs=ATOL), name

    assert result.numerator_df == pytest.approx(manual["df1"], rel=RTOL, abs=ATOL)
    assert result.denominator_df == pytest.approx(manual["df2"], rel=RTOL, abs=ATOL)
    assert result.components["weights"] == pytest.approx(tuple(manual["weights"]), rel=RTOL, abs=ATOL)
    assert result.components["weighted_mean"] == pytest.approx(manual["weighted_mean"], rel=RTOL, abs=ATOL)
    assert result.components["welch_B"] == pytest.approx(manual["b"], rel=RTOL, abs=ATOL)
    assert result.components["welch_correction"] == pytest.approx(manual["correction"], rel=RTOL, abs=ATOL)
    assert result.denominator_df > 0.0
    assert result.components["welch_correction"] >= 1.0


def test_welch_concords_with_modern_scipy_when_available():
    if "equal_var" not in inspect.signature(stats.f_oneway).parameters:
        pytest.skip("SciPy < 1.16 has no Welch f_oneway oracle")

    for name, groups in _scenarios().items():
        result = WelchANOVA(*groups, independence="assumed").run()
        oracle = stats.f_oneway(*groups, equal_var=False)
        assert result.statistic == pytest.approx(float(oracle.statistic), rel=RTOL, abs=ATOL), name
        assert result.p_value == pytest.approx(float(oracle.pvalue), rel=RTOL, abs=ATOL), name


@pytest.mark.parametrize(
    "a,b",
    [
        (np.array([0.0, 1.0]), np.array([2.0, 5.0])),
        (np.array([1.0, 2.0, 4.0, 7.0]), np.array([-3.0, 0.0, 2.0, 9.0, 12.0])),
        (np.linspace(-2.0, 2.0, 31), np.linspace(-15.0, 15.0, 7)),
    ],
)
def test_k2_cross_engine_invariants(a, b):
    classical = OneWayANOVA(a, b, independence="assumed").run()
    pooled_t = stats.ttest_ind(a, b, equal_var=True)
    assert classical.statistic == pytest.approx(float(pooled_t.statistic) ** 2, rel=RTOL, abs=ATOL)
    assert classical.p_value == pytest.approx(float(pooled_t.pvalue), rel=RTOL, abs=ATOL)

    welch = WelchANOVA(a, b, independence="assumed").run()
    welch_t = stats.ttest_ind(a, b, equal_var=False)
    assert welch.statistic == pytest.approx(float(welch_t.statistic) ** 2, rel=RTOL, abs=ATOL)
    assert welch.p_value == pytest.approx(float(welch_t.pvalue), rel=RTOL, abs=ATOL)
    assert welch.denominator_df == pytest.approx(float(welch_t.df), rel=RTOL, abs=ATOL)


def test_large_common_offset_preserves_results_within_float64_resolution():
    groups = _scenarios()["unbalanced_k4"]
    offset = 1e12
    shifted = tuple(group + offset for group in groups)

    for klass in (OneWayANOVA, WelchANOVA):
        baseline = klass(*groups, independence="assumed").run()
        translated = klass(*shifted, independence="assumed").run()
        assert translated.statistic == pytest.approx(
            baseline.statistic, rel=OFFSET_RTOL, abs=5e-10
        )
        assert translated.p_value == pytest.approx(
            baseline.p_value, rel=OFFSET_RTOL, abs=5e-10
        )


def test_common_small_and_large_scaling_preserve_results():
    groups = _scenarios()["balanced_k3"]
    for scale in (1e-100, 1e100, -1e-100, -1e100):
        scaled = tuple(group * scale for group in groups)
        for klass in (OneWayANOVA, WelchANOVA):
            baseline = klass(*groups, independence="assumed").run()
            other = klass(*scaled, independence="assumed").run()
            assert other.statistic == pytest.approx(baseline.statistic, rel=RTOL, abs=ATOL)
            assert other.p_value == pytest.approx(baseline.p_value, rel=RTOL, abs=ATOL)


def test_heteroscedastic_group_and_observation_permutations_preserve_results():
    groups = _scenarios()["size_variance_association"]
    alternatives = (
        tuple(reversed(groups)),
        tuple(group[::-1] for group in groups),
        (groups[2], groups[0], groups[3], groups[1]),
    )
    for klass in (OneWayANOVA, WelchANOVA):
        baseline = klass(*groups, independence="assumed").run()
        for candidate_groups in alternatives:
            other = klass(*candidate_groups, independence="assumed").run()
            assert other.statistic == pytest.approx(baseline.statistic, rel=RTOL, abs=ATOL)
            assert other.p_value == pytest.approx(baseline.p_value, rel=RTOL, abs=ATOL)


def test_all_oracle_domain_outputs_are_finite_and_bounded():
    for groups in _scenarios().values():
        for klass in (OneWayANOVA, WelchANOVA):
            result = klass(*groups, independence="assumed").run()
            assert np.isfinite(result.statistic)
            assert result.statistic >= 0.0
            assert np.isfinite(result.p_value)
            assert 0.0 <= result.p_value <= 1.0
            assert np.isfinite(result.numerator_df)
            assert np.isfinite(result.denominator_df)
            assert result.numerator_df > 0.0
            assert result.denominator_df > 0.0
