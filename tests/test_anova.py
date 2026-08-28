import warnings

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

from pyMagicStat.inference import OneWayANOVA, WelchANOVA


@pytest.fixture
def one_way_groups():
    rng = np.random.default_rng(20260827)
    return (
        rng.normal(loc=0.0, scale=1.0, size=24),
        rng.normal(loc=0.4, scale=1.0, size=30),
        rng.normal(loc=1.0, scale=1.0, size=36),
    )


def test_classical_anova_matches_scipy(one_way_groups):
    result = OneWayANOVA(
        *one_way_groups,
        independence="assumed",
    ).run_test()
    reference = stats.f_oneway(*one_way_groups, equal_var=True)

    assert result["method"] == "Classical one-way ANOVA"
    assert result["statistic"] == pytest.approx(reference.statistic)
    assert result["p_value"] == pytest.approx(reference.pvalue)
    assert result["numerator_df"] == 2.0
    assert result["denominator_df"] == 87.0
    assert result["k"] == 3
    assert result["n_total"] == 90
    assert result["equal_var_requested"] is True
    assert result["variance_selection_policy"] == "explicit_classical"
    assert 0.0 <= result["eta_squared"] <= 1.0


def test_welch_anova_matches_statsmodels():
    rng = np.random.default_rng(118)
    groups = (
        rng.normal(loc=0.0, scale=0.8, size=12),
        rng.normal(loc=0.5, scale=2.5, size=25),
        rng.normal(loc=1.0, scale=5.0, size=55),
    )
    result = WelchANOVA(
        *groups,
        independence="assumed",
    ).run_test()
    reference = anova_oneway(groups, use_var="unequal", welch_correction=True)

    assert result["method"] == "Welch one-way ANOVA"
    assert result["statistic"] == pytest.approx(reference.statistic)
    assert result["p_value"] == pytest.approx(reference.pvalue)
    assert result["numerator_df"] == pytest.approx(reference.df_num)
    assert result["denominator_df"] == pytest.approx(reference.df_denom)
    assert result["equal_var_requested"] is False
    assert result["variance_selection_policy"] == "explicit_welch"


@pytest.mark.parametrize("executor", [OneWayANOVA, WelchANOVA])
def test_anova_strict_mode_accepts_a_calibrated_decision(executor, one_way_groups):
    result = executor(*one_way_groups, independence="assumed").run_test()
    assert result["inference_decision"]["status"] == "selected"


@pytest.mark.parametrize("executor", [OneWayANOVA, WelchANOVA])
def test_anova_is_invariant_to_common_translation_and_group_order(
    executor,
    one_way_groups,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        baseline = executor(*one_way_groups, strict=False).run_test()
        translated = executor(
            *(group + 1_000.0 for group in one_way_groups),
            strict=False,
        ).run_test()
        reordered = executor(*reversed(one_way_groups), strict=False).run_test()

    assert translated["statistic"] == pytest.approx(baseline["statistic"])
    assert translated["p_value"] == pytest.approx(baseline["p_value"])
    assert reordered["statistic"] == pytest.approx(baseline["statistic"])
    assert reordered["p_value"] == pytest.approx(baseline["p_value"])


def test_anova_result_contains_shared_diagnostics_and_no_post_hoc(one_way_groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = WelchANOVA(*one_way_groups, strict=False).run_test()

    assert result["assumptions"]["design"] == "one_way"
    assert result["inference_decision"]["status"] == "selected"
    assert "post_hoc" not in result
    assert "pairwise" not in result
