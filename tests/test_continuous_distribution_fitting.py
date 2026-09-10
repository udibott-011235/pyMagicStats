"""Analytical and independent backend oracles for fixed-location MLEs."""

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from pyMagicStat.distributions.families import (
    ExponentialFamily, GammaFamily, NoFiniteMLEError, FitNumericalError,
)
from pyMagicStat.distributions.families import _fitting


@pytest.mark.parametrize("data,scale", [([1, 2, 3], 2.), ([0, 0, 6], 2.), ([.25], .25),
                                      ([1e308, 1e308], 1e308), ([5e-324, 5e-324], 5e-324)])
def test_exponential_closed_form_and_full_likelihood(data, scale):
    result = ExponentialFamily().fit(data)
    assert result.fitted_distribution.parameters.scale == scale
    expected = sum(-math.log(scale)-x/scale for x in data)
    assert result.log_likelihood == pytest.approx(expected, rel=2e-15)
    assert result.aic == pytest.approx(2-2*expected)
    assert result.bic == pytest.approx(math.log(len(data))-2*expected)


@pytest.mark.parametrize("data", [[0], [0., -0., 0]])
def test_exponential_zero_is_exact_no_finite_mle(data):
    with pytest.raises(NoFiniteMLEError) as error:
        ExponentialFamily().fit(data)
    assert type(error.value) is NoFiniteMLEError


@pytest.mark.parametrize("data", [[0, 1, 2], [1], [2, 2, 2]])
def test_gamma_nonfinite_mle_boundaries_do_not_call_backend(monkeypatch, data):
    def forbidden(*args, **kwargs):
        pytest.fail("boundary must be classified before backend invocation")
    monkeypatch.setattr(stats.gamma, "fit", forbidden)
    with pytest.raises(NoFiniteMLEError):
        GammaFamily().fit(data)


@pytest.mark.parametrize("data", [[.1, .2, .8, 2, 5], [1, 2, 4, 10], [1e-5, 1, 100]])
def test_gamma_scipy_mle_and_complete_likelihood(data):
    shape, _, scale = stats.gamma.fit(data, floc=0, method="MLE")
    result = GammaFamily().fit(data)
    assert result.fitted_distribution.parameters.shape == pytest.approx(shape, rel=2e-14)
    assert result.fitted_distribution.parameters.scale == pytest.approx(scale, rel=2e-14)
    # Independent scalar formula includes normalization, including log Gamma(a).
    ll = sum((shape-1)*math.log(x)-x/scale-math.lgamma(shape)-shape*math.log(scale) for x in data)
    assert result.log_likelihood == pytest.approx(ll, rel=2e-13)
    assert result.aic == pytest.approx(4-2*ll)
    assert result.bic == pytest.approx(2*math.log(len(data))-2*ll)


def test_gamma_both_frozen_arguments_are_explicit(monkeypatch):
    real_fit = stats.gamma.fit
    calls = []
    def spy(data, *args, **kwargs):
        calls.append((data.copy(), args, kwargs.copy()))
        return real_fit(data, *args, **kwargs)
    monkeypatch.setattr(stats.gamma, "fit", spy)
    GammaFamily().fit([1, 2, 4])
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], [1, 2, 4])
    assert calls[0][1] == ()
    assert calls[0][2] == {"floc": 0, "method": "MLE"}


@pytest.mark.parametrize("output", [None, {}, [2, 0, 1], (2, 1), (2, 0, 1, 9),
    ("2", 0, 1), (True, 0, 1), (2, False, 1), (2, 0, None), (2+0j, 0, 1),
    (np.nan, 0, 1), (np.inf, 0, 1), (0, 0, 1), (-1, 0, 1), (2, 1, 1),
    (2, np.nan, 1), (2, 0, 0), (2, 0, -1), (2, 0, np.inf), (2, 0, np.nan),
    (np.finfo(float).max, 0, 1), (2, 0, np.nextafter(0., 1.))])
def test_gamma_malformed_or_unusable_backend_output_fails_closed(monkeypatch, output):
    def backend(*args, **kwargs):
        warnings.warn("backend warning does not make invalid output valid", RuntimeWarning)
        return output
    monkeypatch.setattr(stats.gamma, "fit", backend)
    with pytest.raises(FitNumericalError):
        GammaFamily().fit([1, 2, 4])


@pytest.mark.parametrize("error", [ValueError("backend"), OverflowError("backend"), FloatingPointError("backend")])
@pytest.mark.parametrize("cls,target", [(GammaFamily, "gamma"), (ExponentialFamily, "exponential")])
def test_backend_errors_preserve_cause_after_public_validation(monkeypatch, error, cls, target):
    def backend(*args, **kwargs):
        raise error
    if target == "gamma":
        monkeypatch.setattr(stats.gamma, "fit", backend)
    else:
        monkeypatch.setattr(_fitting, "_mean", backend)
    with pytest.raises(FitNumericalError) as caught:
        cls().fit([1, 2, 4])
    assert caught.value.__cause__ is error
    with pytest.raises(ValueError):
        cls().fit([-1, 2])


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_control_exceptions_are_not_translated(monkeypatch, error):
    def backend(*args, **kwargs):
        raise error
    monkeypatch.setattr(stats.gamma, "fit", backend)
    with pytest.raises(type(error)) as caught:
        GammaFamily().fit([1, 2, 4])
    assert caught.value is error


@pytest.mark.parametrize("cls,target", [(GammaFamily, "gamma"), (ExponentialFamily, "exponential")])
def test_warnings_are_stable_ordered_and_context_is_restored(monkeypatch, cls, target):
    original = stats.gamma.fit if target == "gamma" else _fitting._mean
    def backend(*args, **kwargs):
        warnings.warn(r"first C:\private\machine\file.py line 123 object at 0xABCD", RuntimeWarning)
        warnings.warn("second /home/user/private.py", UserWarning)
        return original(*args, **kwargs)
    if target == "gamma":
        monkeypatch.setattr(stats.gamma, "fit", backend)
    else:
        monkeypatch.setattr(_fitting, "_mean", backend)
    filters, errstate = list(warnings.filters), np.geterr().copy()
    result = cls().fit([1, 2, 4])
    assert result.warnings == (
        "RuntimeWarning: first <path> line <number> object at <address>",
        "UserWarning: second <path>",
    )
    assert warnings.filters == filters
    assert np.geterr() == errstate


@pytest.mark.parametrize("data", [[10**400], [2**53+1]])
@pytest.mark.parametrize("cls", [GammaFamily, ExponentialFamily])
def test_continuous_unsafe_float64_values_are_public_errors(cls, data):
    with pytest.raises(ValueError):
        cls().fit(data)


def test_numerically_unrepresentable_scale_is_not_a_mathematical_boundary():
    with pytest.raises(FitNumericalError):
        ExponentialFamily().fit([0, 0, 5e-324])


def test_nonfinite_final_likelihood_is_rejected(monkeypatch):
    monkeypatch.setattr(_fitting.math, "fsum", lambda values: float("inf"))
    with pytest.raises(FitNumericalError):
        GammaFamily().fit([1, 2, 4])


@pytest.mark.parametrize("path", [
    r"C:\Users\Jane Doe\private.py:123",
    r"\\server\share\private.py line 42",
    "/home/Jane Doe/private.py:123",
    r"C:\Users\Jane Doe\private.py line 42",
    r"\\server\share\private.py:123",
    "/home/Jane Doe/private.py line 42",
])
@pytest.mark.parametrize("cls", [GammaFamily, ExponentialFamily])
def test_embedded_paths_and_source_coordinates_are_fully_redacted(monkeypatch, path, cls):
    original = stats.gamma.fit if cls is GammaFamily else _fitting._mean
    def backend(*args, **kwargs):
        warnings.warn("first at " + path, RuntimeWarning)
        warnings.warn("ordinary ratio x/y; 42 iterations; object 0xABCD", UserWarning)
        return original(*args, **kwargs)
    if cls is GammaFamily:
        monkeypatch.setattr(stats.gamma, "fit", backend)
    else:
        monkeypatch.setattr(_fitting, "_mean", backend)
    filters, state = list(warnings.filters), np.geterr().copy()
    result = cls().fit([1, 2, 4])
    suffix = " line <number>" if " line " in path else ""
    assert result.warnings == (
        "RuntimeWarning: first at <path>" + suffix,
        "UserWarning: ordinary ratio x/y; 42 iterations; object <address>",
    )
    assert warnings.filters == filters and np.geterr() == state


@pytest.mark.parametrize("message", [
    "divide by zero encountered in log", "ratio x/y is 0.5", "3/4 observations; 42 iterations",
    "step / correction is undefined", "solver 1.2 returned 3.4", "no convergence after 123 iterations",
])
def test_ordinary_warning_messages_remain_stable(message):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        warnings.warn(message, RuntimeWarning)
    assert _fitting._warning_strings(records) == ("RuntimeWarning: " + message,)
