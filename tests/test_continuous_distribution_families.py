import copy

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyMagicStat.distributions import ExponentialFamily, GammaFamily
from pyMagicStat.distributions.families import ParameterizedContinuousDistribution


PROBABILITY_METHODS = ("pdf", "logpdf", "cdf", "logcdf", "sf", "logsf")


@pytest.fixture(params=["gamma", "exponential"])
def bound_and_backend(request):
    if request.param == "gamma":
        return (
            GammaFamily().bind(shape=2.5, scale=3.0),
            stats.gamma(a=2.5, loc=0.0, scale=3.0),
        )
    return (
        ExponentialFamily().bind(scale=3.0),
        stats.expon(loc=0.0, scale=3.0),
    )


@pytest.mark.parametrize("method", PROBABILITY_METHODS)
def test_probability_methods_match_scipy_at_central_and_tail_points(
    bound_and_backend, method
):
    bound, backend = bound_and_backend
    x = np.array([-1.0, 0.0, 1e-12, 0.25, 2.0, 25.0, 1e3])

    actual = getattr(bound, method)(x)
    expected = getattr(backend, method)(x)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_ppf_matches_scipy_at_endpoints_central_and_tail_probabilities(
    bound_and_backend,
):
    bound, backend = bound_and_backend
    q = np.array([0.0, 1e-12, 0.25, 0.5, 0.999999999999, 1.0])

    actual = bound.ppf(q)
    expected = backend.ppf(q)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert actual[0] == 0.0
    assert actual[-1] == np.inf


@pytest.mark.parametrize("method", (*PROBABILITY_METHODS, "ppf"))
def test_numeric_scalar_results_are_python_float(bound_and_backend, method):
    bound, _ = bound_and_backend
    assert type(getattr(bound, method)(np.float64(0.5))) is float
    assert type(getattr(bound, method)(1)) is float


@pytest.mark.parametrize("method", (*PROBABILITY_METHODS, "ppf"))
def test_array_results_are_float64_and_preserve_multidimensional_shape(
    bound_and_backend, method
):
    bound, _ = bound_and_backend
    values = np.array([[0.1, 0.25, 0.5], [0.75, 0.9, 1.0]], dtype=np.float32)

    result = getattr(bound, method)(values)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == values.shape


def test_zero_dimensional_ndarray_remains_an_ndarray(bound_and_backend):
    bound, _ = bound_and_backend
    result = bound.cdf(np.array(0.5))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == ()


def test_lists_and_pandas_inputs_return_unlabelled_ndarrays(bound_and_backend):
    bound, backend = bound_and_backend
    values = [0.25, 0.5, 1.0]

    list_result = bound.pdf(values)
    series_result = bound.pdf(pd.Series(values, name="x"))
    frame = pd.DataFrame({"x": values, "y": [2.0, 3.0, 4.0]})
    frame_result = bound.pdf(frame)

    assert isinstance(list_result, np.ndarray)
    assert isinstance(series_result, np.ndarray)
    assert isinstance(frame_result, np.ndarray)
    assert frame_result.shape == frame.shape
    np.testing.assert_allclose(list_result, backend.pdf(values), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(series_result, backend.pdf(values), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(frame_result, backend.pdf(frame), rtol=0.0, atol=0.0)


def test_finite_points_outside_support_are_evaluated_without_clipping(
    bound_and_backend,
):
    bound, backend = bound_and_backend
    assert bound.support.contains(-1.0) is False
    assert bound.pdf(-1.0) == float(backend.pdf(-1.0)) == 0.0
    assert bound.cdf(-1.0) == float(backend.cdf(-1.0)) == 0.0
    assert bound.logpdf(-1.0) == float(backend.logpdf(-1.0)) == -np.inf


@pytest.mark.parametrize("method", PROBABILITY_METHODS)
@pytest.mark.parametrize(
    "bad",
    [
        [],
        np.array([], dtype=float),
        [np.nan],
        [np.inf],
        [-np.inf],
        ["not-numeric"],
        [1 + 0j],
        True,
    ],
)
def test_probability_queries_fail_closed(bound_and_backend, method, bad):
    bound, _ = bound_and_backend
    with pytest.raises((TypeError, ValueError)):
        getattr(bound, method)(bad)


def test_probability_query_rejects_real_value_not_representable_as_float64(
    bound_and_backend,
):
    bound, _ = bound_and_backend
    with pytest.raises(ValueError, match="representable as float64"):
        bound.cdf(10**10000)


@pytest.mark.parametrize(
    "bad",
    [
        [],
        np.array([], dtype=float),
        np.nan,
        np.inf,
        -np.inf,
        -0.01,
        1.01,
        [0.5, np.nan],
        [0.0, 1.01],
        ["not-numeric"],
        1 + 0j,
        True,
    ],
)
def test_ppf_queries_fail_closed(bound_and_backend, bad):
    bound, _ = bound_and_backend
    with pytest.raises((TypeError, ValueError)):
        bound.ppf(bad)


def test_integer_seed_is_deterministic_and_matches_scipy_generator_mapping(
    bound_and_backend,
):
    bound, backend = bound_and_backend

    first = bound.rvs(size=(4, 3), rng=20260906)
    second = bound.rvs(size=(4, 3), rng=np.int64(20260906))
    expected = backend.rvs(
        size=(4, 3),
        random_state=np.random.default_rng(20260906),
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, expected)


def test_caller_generator_is_used_directly_and_advances(bound_and_backend):
    bound, backend = bound_and_backend
    caller = np.random.default_rng(314159)
    oracle = np.random.default_rng(314159)
    before = copy.deepcopy(caller.bit_generator.state)

    actual = bound.rvs(size=8, rng=caller)
    expected = backend.rvs(size=8, random_state=oracle)

    np.testing.assert_array_equal(actual, expected)
    assert caller.bit_generator.state != before
    assert caller.bit_generator.state == oracle.bit_generator.state


def test_integer_sampling_does_not_mutate_process_global_numpy_state(
    bound_and_backend,
):
    bound, _ = bound_and_backend
    original = np.random.get_state()
    try:
        np.random.seed(271828)
        before = np.random.get_state()
        bound.rvs(size=10, rng=123)
        after = np.random.get_state()
    finally:
        np.random.set_state(original)

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize(
    "bad_rng",
    [
        None,
        True,
        np.bool_(False),
        -1,
        np.int64(-2),
        np.random.RandomState(1),
        np.random.SeedSequence(1),
        np.random.PCG64(1),
        object(),
    ],
)
def test_public_rng_contract_rejects_unsupported_forms(bound_and_backend, bad_rng):
    bound, _ = bound_and_backend
    with pytest.raises((TypeError, ValueError)):
        bound.rvs(size=2, rng=bad_rng)


def test_rng_is_required(bound_and_backend):
    bound, _ = bound_and_backend
    with pytest.raises(TypeError):
        bound.rvs(size=2)
    with pytest.raises(TypeError):
        bound.rvs(size=2, random_state=123)


@pytest.mark.parametrize(
    "size, expected_shape",
    [
        (0, (0,)),
        (3, (3,)),
        (np.int64(2), (2,)),
        ((2, 3), (2, 3)),
        ((2, 0), (2, 0)),
        ((), ()),
    ],
)
def test_rvs_array_return_contract(bound_and_backend, size, expected_shape):
    bound, _ = bound_and_backend
    result = bound.rvs(size=size, rng=7)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == expected_shape


def test_rvs_size_none_returns_python_float(bound_and_backend):
    bound, _ = bound_and_backend
    assert type(bound.rvs(size=None, rng=7)) is float


@pytest.mark.parametrize(
    "bad_size",
    [True, np.bool_(False), -1, np.int64(-2), (2, -1), (2, True), (2, 1.5), [2, 3]],
)
def test_rvs_rejects_invalid_size(bound_and_backend, bad_size):
    bound, _ = bound_and_backend
    with pytest.raises((TypeError, ValueError)):
        bound.rvs(size=bad_size, rng=7)


def test_continuous_bound_object_has_no_discrete_probability_api():
    bound = GammaFamily().bind(shape=2.0, scale=3.0)
    assert isinstance(bound, ParameterizedContinuousDistribution)
    assert not hasattr(bound, "pmf")
    assert not hasattr(bound, "logpmf")
    assert not hasattr(GammaFamily(), "pdf")
