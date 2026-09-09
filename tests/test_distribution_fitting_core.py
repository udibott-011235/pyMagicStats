"""CP04 public ownership, input and immutable result contracts."""

import copy
from dataclasses import fields, replace
import inspect
import math
import pickle
import weakref
import gc

import numpy as np
import pytest

import pyMagicStat
import pyMagicStat.distributions as public
import pyMagicStat.distributions.families as families
from pyMagicStat.distributions.families import (
    GammaFamily, ExponentialFamily, NegativeBinomialFamily, FitResult,
    FittedDistribution, FittedContinuousDistribution, FittedDiscreteDistribution,
    DistributionFitError, FitIdentifiabilityError, NoFiniteMLEError, FitNumericalError,
)


NEW_EXPORTS = {
    "FittedDistribution", "FittedContinuousDistribution", "FittedDiscreteDistribution",
    "FitResult", "DistributionFitError", "FitIdentifiabilityError",
    "NoFiniteMLEError", "FitNumericalError",
}
CASES = [
    (GammaFamily, [1., 2., 4., 9.], ("shape", "scale"), "scipy-gamma-fixed-loc-mle-v1", "scipy.stats.gamma.fit"),
    (ExponentialFamily, [0., 1., 3.], ("scale",), "pymagicstats-exponential-closed-form-mle-v1", "closed_form"),
    (NegativeBinomialFamily, [0, 0, 1, 2, 3, 10], ("r", "p"), "pymagicstats-negative-binomial-profile-mle-v1", "scipy.optimize.brentq"),
]


def test_additive_exports_and_exact_fit_signatures():
    for name in NEW_EXPORTS:
        assert name in public.__all__ and name in families.__all__
        assert getattr(public, name) is getattr(families, name)
        assert not hasattr(pyMagicStat, name)
    assert len(public.__all__) == 18
    assert len(families.__all__) == 22
    for cls, *_ in CASES:
        signature = inspect.signature(cls.fit)
        assert list(signature.parameters) == ["self", "data"]
        assert all(p.default is inspect.Parameter.empty for p in signature.parameters.values())
        assert signature.return_annotation in (FitResult, "FitResult")
    for cls in (FitIdentifiabilityError, NoFiniteMLEError, FitNumericalError):
        assert cls.__bases__ == (DistributionFitError,)


@pytest.mark.parametrize("cls,data,estimated,estimator,solver", CASES)
def test_exact_result_fields_and_single_authoritative_state(cls, data, estimated, estimator, solver):
    result = cls().fit(data)
    assert [f.name for f in fields(result)] == [
        "fitted_distribution", "estimation_method", "fixed_parameters", "estimated_parameters",
        "n_observations", "log_likelihood", "aic", "bic", "converged", "warnings", "metadata",
    ]
    fitted = result.fitted_distribution
    assert [f.name for f in fields(fitted)] == ["parameterized_distribution"]
    bound = fitted.parameterized_distribution
    assert fitted.parameters is bound.parameters
    assert fitted.family is bound.family
    assert not hasattr(fitted, "__dict__")
    assert not hasattr(fitted, "_backend")
    for name in ("name", "kind", "support", "parameterization", "parameter_specs", "fixed_parameters",
                 "backend_library", "backend_distribution", "backend_version"):
        assert getattr(fitted, name) == getattr(bound, name)
    assert result.backend == fitted.backend_distribution
    assert result.backend_version == fitted.backend_version
    assert result.fixed_parameters == ("loc",)
    assert result.estimated_parameters == estimated
    assert result.estimation_method == "maximum_likelihood"
    assert result.converged is True
    assert type(result.n_observations) is int and result.n_observations == len(data)
    for name in ("log_likelihood", "aic", "bic"):
        assert type(getattr(result, name)) is float and math.isfinite(getattr(result, name))
    assert result.warnings == ()
    assert dict(result.metadata) == {"estimator_id": estimator, "solver_id": solver}


@pytest.mark.parametrize("cls,data,estimated,estimator,solver", CASES)
@pytest.mark.parametrize("transform", [lambda x: x, copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))])
def test_every_state_layer_remains_immutable(cls, data, estimated, estimator, solver, transform):
    result = transform(cls().fit(data))
    fitted = result.fitted_distribution
    for obj, name, value in [
        (result, "aic", 1.), (result, "backend", "other"), (result, "metadata", {}),
        (fitted, "parameters", None), (fitted, "parameterized_distribution", None),
        (fitted.parameterized_distribution, "family", None),
        (fitted.parameters, estimated[0], 1.), (fitted.support, "lower", -1),
        (fitted.family, "name", "other"), (result.metadata, "_items", ()),
    ]:
        with pytest.raises((AttributeError, TypeError)):
            setattr(obj, name, value)
    with pytest.raises(TypeError):
        result.metadata["solver_id"] = "other"
    with pytest.raises(TypeError):
        fitted.fixed_parameters["loc"] = 1
    with pytest.raises(TypeError):
        fitted.parameter_specs[estimated[0]]["fixed"] = True
    with pytest.raises(TypeError):
        result.estimated_parameters[0] = "other"
    assert fitted.cdf(1.) == pytest.approx(fitted.parameterized_distribution.cdf(1.))


@pytest.mark.parametrize("cls,data,estimated,estimator,solver", CASES)
def test_sample_is_not_retained(cls, data, estimated, estimator, solver):
    sample = np.asarray(data)
    reference = weakref.ref(sample)
    result = cls().fit(sample)
    original = result.fitted_distribution.cdf(1.)
    sample[:] = 100
    assert result.fitted_distribution.cdf(1.) == original
    del sample
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("cls,data,estimated,estimator,solver", CASES)
def test_operations_and_rng_are_delegated(cls, data, estimated, estimator, solver):
    fitted = cls().fit(data).fitted_distribution
    bound = fitted.parameterized_distribution
    discrete = cls is NegativeBinomialFamily
    assert isinstance(fitted, FittedDiscreteDistribution if discrete else FittedContinuousDistribution)
    assert not hasattr(fitted, "pdf" if discrete else "pmf")
    assert not hasattr(fitted, "logpdf" if discrete else "logpmf")
    for op in ("cdf", "logcdf", "sf", "logsf", "pmf" if discrete else "pdf", "logpmf" if discrete else "logpdf"):
        np.testing.assert_array_equal(getattr(fitted, op)([0, 1, 4]), getattr(bound, op)([0, 1, 4]))
    np.testing.assert_array_equal(fitted.ppf([0, .5, 1]), bound.ppf([0, .5, 1]))
    np.testing.assert_array_equal(fitted.rvs((2, 3), rng=71), bound.rvs((2, 3), rng=71))
    assert type(fitted.rvs(rng=71)) is (int if discrete else float)
    assert fitted.rvs(3, rng=71).dtype == (np.int64 if discrete else np.float64)
    with pytest.raises(TypeError):
        fitted.rvs(3)
    with pytest.raises(TypeError):
        fitted.rvs(3, rng=None)
    a, b = np.random.default_rng(71), np.random.default_rng(71)
    np.testing.assert_array_equal(fitted.rvs(4, rng=a), bound.rvs(4, rng=b))
    assert a.bit_generator.state == b.bit_generator.state


INVALID = [0, True, np.bool_(True), [], [[1, 2]], [True, 1], np.array([True, 2], dtype=object),
           np.array([False, True]), ["1", "2"], "123", [1+0j], [None], [np.nan], [np.inf],
           [-np.inf], [-1, 3], [object()]]


@pytest.mark.parametrize("cls", [GammaFamily, ExponentialFamily, NegativeBinomialFamily])
@pytest.mark.parametrize("data", INVALID)
def test_public_input_fails_before_numerical_boundary(cls, data):
    with pytest.raises((TypeError, ValueError)) as error:
        cls().fit(data)
    assert not isinstance(error.value, DistributionFitError)


def test_direct_construction_cannot_introduce_mutable_result_state():
    result = ExponentialFamily().fit([1, 2])
    source = dict(result.metadata)
    copied = replace(result, metadata=source)
    source["solver_id"] = "changed"
    assert copied.metadata["solver_id"] == "closed_form"
    for kwargs in ({"metadata": {"nested": {"mutable": 1}}}, {"warnings": []},
                   {"estimated_parameters": ["scale"]}, {"converged": False},
                   {"n_observations": True}, {"aic": math.inf}, {"log_likelihood": 1}):
        with pytest.raises((TypeError, ValueError)):
            replace(result, **kwargs)
    with pytest.raises(TypeError):
        FittedDistribution(None)
    with pytest.raises(TypeError):
        FittedDiscreteDistribution(result.fitted_distribution.parameterized_distribution)
    with pytest.raises(TypeError):
        FittedContinuousDistribution(NegativeBinomialFamily().bind(r=1, p=.5))


@pytest.mark.parametrize("cls,data,estimated,estimator,solver", CASES)
@pytest.mark.parametrize("construction", ["constructor", "replace"])
@pytest.mark.parametrize("invalid", [
    "empty_metadata", "missing_estimator", "missing_solver", "false_estimator",
    "false_solver", "extra_metadata", "wrong_fixed", "wrong_estimated",
    "duplicate_fixed", "duplicate_estimated", "overlap", "wrong_wrapper",
    "generic_wrapper", "wrong_aic", "wrong_bic", "changed_n", "changed_ll",
])
def test_semantic_invariants_fail_closed_for_all_public_construction_paths(
    cls, data, estimated, estimator, solver, construction, invalid,
):
    result = cls().fit(data)
    metadata = dict(result.metadata)
    if invalid == "empty_metadata":
        change = {"metadata": {}}
    elif invalid.startswith("missing_"):
        del metadata[invalid.removeprefix("missing_") + "_id"]
        change = {"metadata": metadata}
    elif invalid.startswith("false_"):
        metadata[invalid.removeprefix("false_") + "_id"] = "test"
        change = {"metadata": metadata}
    elif invalid == "extra_metadata":
        change = {"metadata": {**metadata, "backend": "invented"}}
    elif invalid == "wrong_fixed":
        change = {"fixed_parameters": ()}
    elif invalid == "wrong_estimated":
        change = {"estimated_parameters": ("invented",)}
    elif invalid == "duplicate_fixed":
        change = {"fixed_parameters": ("loc", "loc")}
    elif invalid == "duplicate_estimated":
        change = {"estimated_parameters": estimated + estimated}
    elif invalid == "overlap":
        change = {"estimated_parameters": estimated + ("loc",)}
    elif invalid == "wrong_wrapper":
        # A subclass can drop/override the interface; only the frozen concrete
        # wrapper is an admissible successful-result representation.
        class OtherWrapper(type(result.fitted_distribution)):
            __slots__ = ()
        change = {"fitted_distribution": OtherWrapper(result.fitted_distribution.parameterized_distribution)}
    elif invalid == "generic_wrapper":
        change = {"fitted_distribution": FittedDistribution(result.fitted_distribution.parameterized_distribution)}
    elif invalid == "wrong_aic":
        change = {"aic": result.aic + 1.}
    elif invalid == "wrong_bic":
        change = {"bic": result.bic + 1.}
    elif invalid == "changed_n":
        change = {"n_observations": result.n_observations + 1}
    else:
        change = {"log_likelihood": result.log_likelihood + 1.}
    with pytest.raises((TypeError, ValueError)):
        if construction == "replace":
            replace(result, **change)
        else:
            FitResult(**{**{f.name: getattr(result, f.name) for f in fields(result)}, **change})


@pytest.mark.parametrize("cls,data,estimated,estimator,solver", CASES)
def test_valid_canonical_direct_construction_and_round_trips(cls, data, estimated, estimator, solver):
    production = cls().fit(data)
    values = {f.name: getattr(production, f.name) for f in fields(production)}
    values["metadata"] = {"estimator_id": estimator, "solver_id": solver}
    direct = FitResult(**values)
    for result in (direct, replace(direct), copy.copy(direct), copy.deepcopy(direct),
                   pickle.loads(pickle.dumps(direct))):
        assert result.estimated_parameters == estimated
        assert result.fixed_parameters == ("loc",)
        assert dict(result.metadata) == values["metadata"]
        assert result.aic == 2*len(estimated)-2*result.log_likelihood
        assert result.bic == len(estimated)*math.log(len(data))-2*result.log_likelihood
        assert type(result.fitted_distribution) is type(production.fitted_distribution)
