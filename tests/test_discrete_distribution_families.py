import copy
from dataclasses import FrozenInstanceError
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyMagicStat.distributions import (
    DiscreteDistributionFamily,
    DistributionSupport,
    NegativeBinomialFamily,
    NegativeBinomialParameters,
    ParameterizedDiscreteDistribution,
    SupportKind,
)
from pyMagicStat.distributions.families import GammaFamily, GammaParameters


PROBABILITY_METHODS = ("pmf", "logpmf", "cdf", "logcdf", "sf", "logsf")


@pytest.fixture
def bound_and_backend():
    return (
        NegativeBinomialFamily().bind(r=2.5, p=0.35),
        stats.nbinom(n=2.5, p=0.35, loc=0),
    )


def test_public_discrete_hierarchy_and_bound_type():
    family = NegativeBinomialFamily()
    bound = family.bind(r=2, p=0.5)

    assert isinstance(family, DiscreteDistributionFamily)
    assert isinstance(bound, ParameterizedDiscreteDistribution)
    assert bound.kind is SupportKind.DISCRETE
    assert bound.family is family


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, True),
        (1, True),
        (1.0, True),
        (0.5, False),
        (1.25, False),
        (-1, False),
        (True, False),
        (np.bool_(True), False),
        (np.nan, False),
        (np.inf, False),
        (-np.inf, False),
    ],
)
def test_discrete_support_scalar_membership(value, expected):
    support = NegativeBinomialFamily().support

    assert support.kind is SupportKind.DISCRETE
    assert support.contains(value) is expected


@pytest.mark.parametrize(
    "values, expected",
    [
        (
            [0, 1.0, 0.5, -1, True, np.nan, np.inf],
            [True, True, False, False, False, False, False],
        ),
        (
            [[0, 1.0, 1.25], [True, np.bool_(False), 2]],
            [[True, True, False], [False, False, True]],
        ),
        (
            np.array([0, np.bool_(True), 2.0, 2.5], dtype=object),
            [True, False, True, False],
        ),
    ],
)
def test_discrete_support_array_membership_is_elementwise(values, expected):
    membership = NegativeBinomialFamily().support.contains(values)

    assert isinstance(membership, np.ndarray)
    assert membership.dtype == np.bool_
    assert membership.shape == np.asarray(values).shape
    np.testing.assert_array_equal(membership, np.asarray(expected, dtype=bool))


def test_discrete_support_extension_does_not_change_continuous_membership():
    assert GammaFamily().support.contains(0.5) is True
    assert NegativeBinomialFamily().support.contains(0.5) is False


def test_family_is_stateless_and_bindings_own_independent_parameters():
    family = NegativeBinomialFamily()
    first = family.bind(r=2, p=0.5)
    second = family.bind(r=3.5, p=0.25)

    assert not hasattr(family, "__dict__")
    assert first.parameters is not second.parameters
    assert first.parameters != second.parameters
    with pytest.raises(AttributeError):
        family.parameters = first.parameters


def test_parameters_canonicalize_to_python_float_and_are_immutable():
    parameters = NegativeBinomialParameters(r=np.int64(2), p=np.float32(0.5))

    assert type(parameters.r) is float
    assert type(parameters.p) is float
    assert parameters == NegativeBinomialParameters(r=2.0, p=0.5)
    with pytest.raises(FrozenInstanceError):
        parameters.r = 3.0
    with pytest.raises(FrozenInstanceError):
        parameters.p = 0.25


@pytest.mark.parametrize(
    "r",
    [2, 2.5, np.int32(3), np.float64(4.5), np.nextafter(0.0, 1.0)],
)
def test_positive_finite_integer_and_noninteger_r_are_valid(r):
    assert NegativeBinomialParameters(r=r, p=0.5).r == float(r)


def test_p_equal_one_is_valid():
    assert NegativeBinomialParameters(r=2.5, p=1).p == 1.0


@pytest.mark.parametrize(
    "r",
    [
        True,
        np.bool_(False),
        0,
        0.0,
        -1,
        np.nan,
        np.inf,
        -np.inf,
        1 + 0j,
        "2",
        None,
        pytest.param(10**10000, id="unrepresentable-integer"),
    ],
)
def test_invalid_r_fails_closed(r):
    with pytest.raises((TypeError, ValueError)):
        NegativeBinomialParameters(r=r, p=0.5)
    with pytest.raises((TypeError, ValueError)):
        NegativeBinomialFamily().bind(r=r, p=0.5)


@pytest.mark.parametrize(
    "p",
    [
        True,
        np.bool_(False),
        0,
        0.0,
        -0.1,
        1.0000001,
        np.nan,
        np.inf,
        -np.inf,
        0.5 + 0j,
        "0.5",
        None,
        pytest.param(10**10000, id="unrepresentable-integer"),
    ],
)
def test_invalid_p_fails_closed(p):
    with pytest.raises((TypeError, ValueError)):
        NegativeBinomialParameters(r=2.0, p=p)
    with pytest.raises((TypeError, ValueError)):
        NegativeBinomialFamily().bind(r=2.0, p=p)


def test_bind_exposes_only_canonical_r_p_parameters():
    family = NegativeBinomialFamily()

    with pytest.raises(TypeError):
        family.bind(r=2.0, p=0.5, loc=1)
    with pytest.raises(TypeError):
        family.bind(n=2.0, p=0.5)
    with pytest.raises(TypeError):
        family.bind(r=2.0, mu=1.0)


def test_support_and_metadata_are_canonical_and_immutable():
    family = NegativeBinomialFamily()
    support = family.support

    assert support == DistributionSupport(
        lower=0.0,
        upper=math.inf,
        lower_closed=True,
        upper_closed=False,
        kind=SupportKind.DISCRETE,
    )
    assert family.parameterization == "r-p; loc=0 fixed"
    assert family.fixed_parameters == {"loc": 0.0}
    assert family.parameter_specs["r"]["constraint"] == "finite real > 0"
    assert family.parameter_specs["p"]["constraint"] == "finite real in (0, 1]"
    assert family.backend_library == "scipy"
    assert family.backend_distribution == "scipy.stats.nbinom"
    with pytest.raises(TypeError):
        family.fixed_parameters["loc"] = 1.0
    with pytest.raises(TypeError):
        family.parameter_specs["p"]["constraint"] = "other"


def test_canonical_negative_binomial_moments_match_scipy():
    parameters = NegativeBinomialParameters(r=2.5, p=0.35)
    mean = parameters.r * (1.0 - parameters.p) / parameters.p
    variance = parameters.r * (1.0 - parameters.p) / parameters.p**2

    assert mean == pytest.approx(stats.nbinom.mean(parameters.r, parameters.p))
    assert variance == pytest.approx(stats.nbinom.var(parameters.r, parameters.p))


def test_parameterized_discrete_distribution_rejects_wrong_family_or_parameters():
    with pytest.raises(TypeError, match="DiscreteDistributionFamily"):
        ParameterizedDiscreteDistribution(
            family=GammaFamily(),
            parameters=GammaParameters(shape=2.0, scale=1.0),
        )
    with pytest.raises(TypeError, match="NegativeBinomialParameters"):
        ParameterizedDiscreteDistribution(
            family=NegativeBinomialFamily(),
            parameters=GammaParameters(shape=2.0, scale=1.0),
        )


@pytest.mark.parametrize(
    "r, p",
    [(2.0, 0.5), (2.5, 0.35), (0.5, 0.99), (25.0, 0.05)],
)
@pytest.mark.parametrize("method", PROBABILITY_METHODS)
def test_probability_methods_match_scipy(r, p, method):
    bound = NegativeBinomialFamily().bind(r=r, p=p)
    backend = stats.nbinom(n=r, p=p, loc=0)
    x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 10.0, 100.0])

    np.testing.assert_allclose(
        getattr(bound, method)(x),
        getattr(backend, method)(x),
        rtol=0.0,
        atol=0.0,
    )


def test_fractional_probability_queries_follow_discrete_scipy_semantics(
    bound_and_backend,
):
    bound, backend = bound_and_backend

    assert bound.pmf(0.5) == 0.0
    assert bound.logpmf(0.5) == -np.inf
    assert bound.pmf(-1.0) == 0.0
    assert bound.logpmf(-1.0) == -np.inf
    assert bound.cdf(0.5) == bound.cdf(0.0) == float(backend.cdf(0.0))
    assert bound.cdf(1.5) == bound.cdf(1.0) == float(backend.cdf(1.0))


@pytest.mark.parametrize("method", PROBABILITY_METHODS)
def test_probability_scalar_and_array_result_contract(bound_and_backend, method):
    bound, _ = bound_and_backend
    scalar = getattr(bound, method)(np.float64(1.5))
    values = np.array([[0.0, 0.5, 1.0], [2.0, 3.0, 10.0]], dtype=np.float32)
    array = getattr(bound, method)(values)

    assert type(scalar) is float
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float64
    assert array.shape == values.shape


def test_pandas_probability_inputs_return_unlabelled_float64_arrays(
    bound_and_backend,
):
    bound, backend = bound_and_backend
    series = pd.Series([0.0, 1.0, 2.0], name="x")
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [2.0, 3.0]})

    series_result = bound.pmf(series)
    frame_result = bound.cdf(frame)

    assert isinstance(series_result, np.ndarray)
    assert series_result.dtype == np.float64
    assert isinstance(frame_result, np.ndarray)
    assert frame_result.dtype == np.float64
    assert frame_result.shape == frame.shape
    np.testing.assert_allclose(series_result, backend.pmf(series), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(frame_result, backend.cdf(frame), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("method", PROBABILITY_METHODS)
@pytest.mark.parametrize(
    "bad",
    [
        [],
        np.array([], dtype=float),
        True,
        np.bool_(False),
        [0.0, True],
        np.array([0.0, np.bool_(True)], dtype=object),
        [np.nan],
        [np.inf],
        [-np.inf],
        [1 + 0j],
        ["1"],
    ],
)
def test_probability_queries_fail_closed(bound_and_backend, method, bad):
    bound, _ = bound_and_backend
    with pytest.raises((TypeError, ValueError)):
        getattr(bound, method)(bad)


def test_probability_query_rejects_unrepresentable_real(bound_and_backend):
    bound, _ = bound_and_backend
    with pytest.raises(ValueError, match="representable as float64"):
        bound.cdf(10**10000)


def test_ppf_canonicalizes_endpoints_and_matches_scipy_interior(bound_and_backend):
    bound, backend = bound_and_backend
    q = np.array([0.0, 1e-12, 0.25, 0.5, 0.999999, 1.0])

    result = bound.ppf(q)

    assert result.dtype == np.float64
    assert result.shape == q.shape
    assert result[0] == 0.0
    assert result[-1] == np.inf
    np.testing.assert_allclose(result[1:-1], backend.ppf(q[1:-1]), rtol=0.0, atol=0.0)
    assert type(bound.ppf(0)) is float
    assert bound.ppf(0) == 0.0
    assert bound.ppf(1) == np.inf


def test_ppf_multidimensional_and_zero_dimensional_array_contract(bound_and_backend):
    bound, backend = bound_and_backend
    q = np.array([[0.0, 0.25], [0.75, 1.0]], dtype=np.float32)

    result = bound.ppf(q)
    zero_dimensional = bound.ppf(np.array(0.5))

    assert result.dtype == np.float64
    assert result.shape == q.shape
    assert result[0, 0] == 0.0
    assert result[-1, -1] == np.inf
    np.testing.assert_allclose(result[0, 1], backend.ppf(0.25), rtol=0.0, atol=0.0)
    assert isinstance(zero_dimensional, np.ndarray)
    assert zero_dimensional.dtype == np.float64
    assert zero_dimensional.shape == ()


def test_ppf_does_not_send_endpoint_entries_to_backend(monkeypatch):
    class EndpointRejectingBackend:
        def ppf(self, q):
            assert np.all((q > 0.0) & (q < 1.0))
            return np.full(np.asarray(q).shape, 4.0)

    monkeypatch.setattr(
        NegativeBinomialFamily,
        "_backend",
        lambda self, parameters: EndpointRejectingBackend(),
    )
    bound = NegativeBinomialFamily().bind(r=2.0, p=0.5)

    np.testing.assert_array_equal(bound.ppf([0.0, 0.5, 1.0]), [0.0, 4.0, np.inf])


@pytest.mark.parametrize(
    "bad",
    [
        [],
        np.array([], dtype=float),
        -0.01,
        1.01,
        True,
        np.bool_(False),
        [0.5, True],
        np.nan,
        np.inf,
        -np.inf,
        [0.5, np.nan],
        [0.5, np.inf],
        0.5 + 0j,
        ["0.5"],
    ],
)
def test_ppf_invalid_queries_fail_closed(bound_and_backend, bad):
    bound, _ = bound_and_backend
    with pytest.raises((TypeError, ValueError)):
        bound.ppf(bad)


def test_integer_seed_is_deterministic_and_matches_scipy():
    bound = NegativeBinomialFamily().bind(r=2.5, p=0.35)
    backend = stats.nbinom(n=2.5, p=0.35, loc=0)

    first = bound.rvs(size=(4, 3), rng=20260907)
    second = bound.rvs(size=(4, 3), rng=np.int64(20260907))
    expected = backend.rvs(
        size=(4, 3),
        random_state=np.random.default_rng(20260907),
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, expected)
    assert first.dtype == np.int64


def test_caller_generator_is_used_directly_and_advances():
    bound = NegativeBinomialFamily().bind(r=2.5, p=0.35)
    backend = stats.nbinom(n=2.5, p=0.35, loc=0)
    caller = np.random.default_rng(314159)
    oracle = np.random.default_rng(314159)
    before = copy.deepcopy(caller.bit_generator.state)

    actual = bound.rvs(size=8, rng=caller)
    expected = backend.rvs(size=8, random_state=oracle)

    np.testing.assert_array_equal(actual, expected)
    assert caller.bit_generator.state != before
    assert caller.bit_generator.state == oracle.bit_generator.state


def test_integer_sampling_does_not_mutate_process_global_numpy_state():
    bound = NegativeBinomialFamily().bind(r=2.5, p=0.35)
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
    "size, expected_shape",
    [(0, (0,)), (5, (5,)), (np.int64(2), (2,)), ((2, 3), (2, 3)), ((), ())],
)
def test_discrete_rvs_array_return_contract(size, expected_shape):
    result = NegativeBinomialFamily().bind(r=2.5, p=0.35).rvs(size=size, rng=7)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64
    assert result.shape == expected_shape


def test_discrete_rvs_size_none_returns_python_int():
    result = NegativeBinomialFamily().bind(r=2.5, p=0.35).rvs(size=None, rng=7)
    assert type(result) is int


def test_p_equal_one_returns_zero_with_canonical_types():
    bound = NegativeBinomialFamily().bind(r=2.5, p=1.0)

    assert bound.rvs(size=None, rng=7) == 0
    result = bound.rvs(size=(2, 3), rng=7)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, np.zeros((2, 3), dtype=np.int64))


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
def test_discrete_rng_contract_rejects_unsupported_forms(bad_rng):
    bound = NegativeBinomialFamily().bind(r=2.5, p=0.35)
    with pytest.raises((TypeError, ValueError)):
        bound.rvs(size=2, rng=bad_rng)


def test_discrete_rng_is_required_and_random_state_alias_is_rejected():
    bound = NegativeBinomialFamily().bind(r=2.5, p=0.35)
    with pytest.raises(TypeError):
        bound.rvs(size=2)
    with pytest.raises(TypeError):
        bound.rvs(size=2, random_state=123)


@pytest.mark.parametrize(
    "bad_size",
    [True, np.bool_(False), -1, np.int64(-2), (2, -1), (2, True), (2, 1.5), [2, 3]],
)
def test_discrete_rvs_rejects_invalid_size(bad_size):
    bound = NegativeBinomialFamily().bind(r=2.5, p=0.35)
    with pytest.raises((TypeError, ValueError)):
        bound.rvs(size=bad_size, rng=7)


@pytest.mark.parametrize("method", (*PROBABILITY_METHODS, "ppf"))
def test_probability_backend_nan_raises_explicit_failure(monkeypatch, method):
    class NaNBackend:
        def __getattr__(self, operation):
            return lambda value: np.full(np.asarray(value).shape, np.nan)

    monkeypatch.setattr(
        NegativeBinomialFamily,
        "_backend",
        lambda self, parameters: NaNBackend(),
    )
    bound = NegativeBinomialFamily().bind(r=2.0, p=0.5)
    query = [0.25, 0.75] if method == "ppf" else [0.0, 1.0]

    with pytest.raises(
        FloatingPointError,
        match=rf"backend numerical failure: {method} returned NaN",
    ):
        getattr(bound, method)(query)


@pytest.mark.parametrize("error_type", [ValueError, OverflowError, FloatingPointError])
def test_rvs_backend_range_failures_are_translated_and_chained(
    monkeypatch, error_type
):
    original = error_type("backend limit")

    class FailingBackend:
        def rvs(self, **kwargs):
            raise original

    monkeypatch.setattr(
        NegativeBinomialFamily,
        "_backend",
        lambda self, parameters: FailingBackend(),
    )
    bound = NegativeBinomialFamily().bind(r=2.0, p=0.5)

    with pytest.raises(
        FloatingPointError,
        match=r"^backend numerical failure: rvs",
    ) as caught:
        bound.rvs(size=2, rng=7)
    assert caught.value.__cause__ is original


@pytest.mark.parametrize(
    "backend_result",
    [
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([0.0, 1.5]),
        np.array(["0", "1"]),
        np.array([False, True]),
        np.array([0.0, float(2**63)]),
    ],
)
def test_discrete_sample_normalizer_fails_closed(monkeypatch, backend_result):
    class InvalidSampleBackend:
        def rvs(self, **kwargs):
            return backend_result

    monkeypatch.setattr(
        NegativeBinomialFamily,
        "_backend",
        lambda self, parameters: InvalidSampleBackend(),
    )
    bound = NegativeBinomialFamily().bind(r=2.0, p=0.5)

    with pytest.raises(
        FloatingPointError,
        match=r"^backend numerical failure: rvs",
    ):
        bound.rvs(size=2, rng=7)


def test_discrete_sample_normalizer_rejects_unexpected_shape(monkeypatch):
    class WrongShapeBackend:
        def rvs(self, **kwargs):
            return np.array([0, 1, 2], dtype=np.int64)

    monkeypatch.setattr(
        NegativeBinomialFamily,
        "_backend",
        lambda self, parameters: WrongShapeBackend(),
    )
    bound = NegativeBinomialFamily().bind(r=2.0, p=0.5)

    with pytest.raises(
        FloatingPointError,
        match=r"^backend numerical failure: rvs",
    ):
        bound.rvs(size=2, rng=7)


def test_legitimate_logpmf_negative_infinity_is_preserved():
    bound = NegativeBinomialFamily().bind(r=2.5, p=0.35)
    assert bound.logpmf(-1.0) == -np.inf
