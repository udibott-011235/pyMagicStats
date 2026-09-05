import inspect

import numpy as np
import pytest
from scipy import stats

from pyMagicStat.inference import OneWayANOVA, WelchANOVA


OFFSET_RTOL = 5e-8
OFFSET_ATOL = 5e-10


def _groups():
    return (
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
        np.array([1.0, 1.5, 2.5, 4.0, 6.5, 9.0, 12.0]),
        np.array([-3.0, -1.0, 0.0, 2.0]),
    )


def test_large_common_offset_preserves_classical_statistic_and_pvalue():
    groups = _groups()
    shifted = tuple(group + 1e12 for group in groups)

    baseline = OneWayANOVA(*groups, independence="assumed").run()
    translated = OneWayANOVA(*shifted, independence="assumed").run()
    scipy_baseline = stats.f_oneway(*groups)
    scipy_shifted = stats.f_oneway(*shifted)

    assert scipy_shifted.statistic == pytest.approx(scipy_baseline.statistic, rel=1e-13)
    assert translated.statistic == pytest.approx(
        baseline.statistic, rel=OFFSET_RTOL, abs=OFFSET_ATOL
    )
    assert translated.p_value == pytest.approx(
        baseline.p_value, rel=OFFSET_RTOL, abs=OFFSET_ATOL
    )
    assert translated.statistic == pytest.approx(
        float(scipy_shifted.statistic), rel=OFFSET_RTOL, abs=OFFSET_ATOL
    )


def test_large_common_offset_preserves_welch_statistic_and_pvalue():
    groups = _groups()
    shifted = tuple(group + 1e12 for group in groups)

    baseline = WelchANOVA(*groups, independence="assumed").run()
    translated = WelchANOVA(*shifted, independence="assumed").run()

    assert translated.statistic == pytest.approx(
        baseline.statistic, rel=OFFSET_RTOL, abs=OFFSET_ATOL
    )
    assert translated.p_value == pytest.approx(
        baseline.p_value, rel=OFFSET_RTOL, abs=OFFSET_ATOL
    )
    assert translated.denominator_df == pytest.approx(
        baseline.denominator_df, rel=OFFSET_RTOL, abs=OFFSET_ATOL
    )

    if "equal_var" in inspect.signature(stats.f_oneway).parameters:
        scipy_shifted = stats.f_oneway(*shifted, equal_var=False)
        assert translated.statistic == pytest.approx(
            float(scipy_shifted.statistic), rel=OFFSET_RTOL, abs=OFFSET_ATOL
        )
        assert translated.p_value == pytest.approx(
            float(scipy_shifted.pvalue), rel=OFFSET_RTOL, abs=OFFSET_ATOL
        )


def test_centered_group_variances_are_translation_stable():
    groups = _groups()
    shifted = tuple(group + 1e12 for group in groups)

    baseline = OneWayANOVA(*groups, independence="assumed")
    translated = OneWayANOVA(*shifted, independence="assumed")

    assert tuple(summary.variance for summary in translated.summaries) == pytest.approx(
        tuple(summary.variance for summary in baseline.summaries),
        rel=1e-12,
        abs=1e-14,
    )
