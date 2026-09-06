from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from pyMagicStat.distributions import (
    Distribution,
    ExponentialFamily,
    GammaFamily,
    NormalDistribution,
)
from pyMagicStat.distributions.families import (
    ContinuousDistributionFamily,
    DistributionFamily,
    DistributionSupport,
    ExponentialParameters,
    GammaParameters,
    ParameterizedContinuousDistribution,
    ParameterizedDistribution,
    SupportKind,
)


INVALID_POSITIVE_PARAMETERS = [
    True,
    np.bool_(False),
    0,
    0.0,
    -1,
    np.nan,
    np.inf,
    -np.inf,
    1 + 0j,
    "1",
    None,
]


def test_public_hierarchy_and_additive_legacy_exports():
    assert issubclass(ContinuousDistributionFamily, DistributionFamily)
    assert issubclass(ParameterizedContinuousDistribution, ParameterizedDistribution)
    assert GammaFamily().kind is SupportKind.CONTINUOUS
    assert ExponentialFamily().kind is SupportKind.CONTINUOUS

    # Existing package exports remain available alongside the additive families.
    assert Distribution.__name__ == "Distribution"
    assert NormalDistribution.__name__ == "NormalDistribution"


@pytest.mark.parametrize(
    "family, first_kwargs, second_kwargs",
    [
        (GammaFamily(), {"shape": 2, "scale": 3}, {"shape": 4, "scale": 5}),
        (ExponentialFamily(), {"scale": 2}, {"scale": 7}),
    ],
)
def test_family_is_stateless_and_bindings_have_independent_parameter_state(
    family, first_kwargs, second_kwargs
):
    assert not hasattr(family, "__dict__")

    first = family.bind(**first_kwargs)
    second = family.bind(**second_kwargs)

    assert first is not second
    assert first.parameters is not second.parameters
    assert first.parameters != second.parameters
    assert first.family is family
    assert second.family is family
    with pytest.raises(AttributeError):
        family.parameters = first.parameters


def test_parameter_objects_canonicalize_to_python_float_and_are_immutable():
    gamma = GammaParameters(shape=np.int64(2), scale=np.float32(3.5))
    exponential = ExponentialParameters(scale=np.int32(4))

    assert type(gamma.shape) is float
    assert type(gamma.scale) is float
    assert type(exponential.scale) is float
    assert gamma == GammaParameters(shape=2.0, scale=3.5)
    assert exponential == ExponentialParameters(scale=4.0)

    with pytest.raises(FrozenInstanceError):
        gamma.shape = 9.0
    with pytest.raises(FrozenInstanceError):
        exponential.scale = 9.0


def test_parameters_reject_real_values_not_representable_as_float64():
    with pytest.raises(ValueError, match="representable as float64"):
        GammaParameters(shape=10**10000, scale=1.0)
    with pytest.raises(ValueError, match="representable as float64"):
        ExponentialParameters(scale=10**10000)


@pytest.mark.parametrize("field", ["shape", "scale"])
@pytest.mark.parametrize("value", INVALID_POSITIVE_PARAMETERS)
def test_gamma_parameters_fail_closed(field, value):
    parameters = {"shape": 2.0, "scale": 3.0}
    parameters[field] = value
    with pytest.raises((TypeError, ValueError)):
        GammaParameters(**parameters)
    with pytest.raises((TypeError, ValueError)):
        GammaFamily().bind(**parameters)


@pytest.mark.parametrize("value", INVALID_POSITIVE_PARAMETERS)
def test_exponential_parameters_fail_closed(value):
    with pytest.raises((TypeError, ValueError)):
        ExponentialParameters(scale=value)
    with pytest.raises((TypeError, ValueError)):
        ExponentialFamily().bind(scale=value)


def test_bind_does_not_expose_loc_or_rate_aliases():
    with pytest.raises(TypeError):
        GammaFamily().bind(shape=2.0, scale=3.0, loc=1.0)
    with pytest.raises(TypeError):
        GammaFamily().bind(shape=2.0, scale=3.0, rate=1.0 / 3.0)
    with pytest.raises(TypeError):
        ExponentialFamily().bind(scale=3.0, loc=1.0)
    with pytest.raises(TypeError):
        ExponentialFamily().bind(scale=3.0, rate=1.0 / 3.0)


def test_nonnegative_continuous_support_contract_and_membership():
    support = GammaFamily().support

    assert support == ExponentialFamily().support
    assert support.lower == 0.0
    assert support.upper == math.inf
    assert support.lower_closed is True
    assert support.upper_closed is False
    assert support.kind is SupportKind.CONTINUOUS
    assert support.contains(0) is True
    assert support.contains(np.float64(1.5)) is True
    assert support.contains(-1) is False
    assert support.contains(np.inf) is False
    assert support.contains(-np.inf) is False
    assert support.contains(np.nan) is False

    values = np.array([[0.0, 1.0, -1.0], [np.nan, np.inf, 4.0]])
    membership = support.contains(values)
    assert isinstance(membership, np.ndarray)
    assert membership.dtype == np.bool_
    assert membership.shape == values.shape
    np.testing.assert_array_equal(
        membership,
        np.array([[True, True, False], [False, False, True]]),
    )


def test_support_is_immutable_and_validates_infinite_endpoint_closure():
    support = GammaFamily().support
    with pytest.raises(FrozenInstanceError):
        support.lower = -1.0
    with pytest.raises(ValueError):
        DistributionSupport(
            lower=0.0,
            upper=math.inf,
            lower_closed=True,
            upper_closed=True,
            kind=SupportKind.CONTINUOUS,
        )


@pytest.mark.parametrize("bad", ["x", 1 + 0j, True, None])
def test_support_nonnumeric_membership_fails_closed(bad):
    assert not bool(GammaFamily().support.contains(bad))


def test_support_membership_fails_closed_for_unrepresentable_real_value():
    assert GammaFamily().support.contains(10**10000) is False


@pytest.mark.parametrize("family", [GammaFamily(), ExponentialFamily()])
def test_parameterization_and_backend_metadata_are_immutable_or_derived(family):
    assert family.fixed_parameters == {"loc": 0.0}
    assert family.parameter_specs["loc"] == {
        "canonical": False,
        "fixed": True,
        "value": 0.0,
    }
    assert "loc=0 fixed" in family.parameterization
    assert family.backend_library == "scipy"
    assert family.backend_distribution.startswith("scipy.stats.")
    assert isinstance(family.backend_version, str)
    assert family.backend_version

    with pytest.raises(TypeError):
        family.fixed_parameters["loc"] = 1.0
    with pytest.raises(TypeError):
        family.parameter_specs["loc"]["value"] = 1.0


def test_parameterized_distribution_rejects_mismatched_parameter_object():
    with pytest.raises(TypeError, match="GammaParameters"):
        ParameterizedContinuousDistribution(
            family=GammaFamily(),
            parameters=ExponentialParameters(scale=2.0),
        )


def test_parameterized_distribution_is_immutable_and_delegates_metadata():
    bound = GammaFamily().bind(shape=2.0, scale=3.0)

    assert isinstance(bound, ParameterizedContinuousDistribution)
    assert bound.name == bound.family.name == "gamma"
    assert bound.kind is SupportKind.CONTINUOUS
    assert bound.support is bound.family.support
    assert bound.parameter_specs is bound.family.parameter_specs
    assert bound.fixed_parameters is bound.family.fixed_parameters
    assert bound.backend_library == "scipy"
    assert bound.backend_distribution == "scipy.stats.gamma"

    with pytest.raises(FrozenInstanceError):
        bound.parameters = GammaParameters(shape=1.0, scale=1.0)
