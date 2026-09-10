"""Generalized NB MLE: independent numerical and exact-domain evidence."""

from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy import optimize, special, stats

from pyMagicStat.distributions.families import (
    NegativeBinomialFamily, FitIdentifiabilityError, NoFiniteMLEError, FitNumericalError,
)
from pyMagicStat.distributions.families import _fitting


SAMPLE = [0, 0, 1, 2, 3, 10]


@pytest.mark.parametrize("value", [0, -0., 1, 1., np.int8(1), np.int64(1), np.uint64(1)])
def test_lossless_integer_forms(value):
    data = _fitting._fit_input([value, 0, 10], discrete=True)
    assert data.dtype == np.int64
    assert data[0] == int(value)
    assert NegativeBinomialFamily().fit([value, 0, 10]).converged


@pytest.mark.parametrize("value", [.5, -1, True, np.bool_(False), np.nan, np.inf, -np.inf,
    2**63, 2**63+1, -(2**63)-1, float(2**63), np.uint64(2**63), "1", 1+0j])
def test_no_rounding_wrapping_parsing_or_boolean_coercion(value):
    with pytest.raises((TypeError, ValueError)):
        NegativeBinomialFamily().fit([0, value, 10])


def test_int64_limit_is_not_first_rounded_through_float():
    for source in ([0, 2**63-1], np.array([0, 2**63-1], dtype=np.uint64),
                   np.array([0, 2**63-1], dtype=object)):
        result = _fitting._fit_input(source, discrete=True)
        assert int(result[1]) == 2**63-1
    assert int(_fitting._fit_input([np.nextafter(float(2**63), 0)], discrete=True)[0]) == 2**63-1024
    with pytest.raises(ValueError):
        _fitting._fit_input([np.iinfo(np.int64).min], discrete=True)


@pytest.mark.parametrize("data,error", [([0, 0], FitIdentifiabilityError), ([1], NoFiniteMLEError),
    ([1, 1, 1], NoFiniteMLEError), ([0, 2], NoFiniteMLEError), ([0, 1], NoFiniteMLEError),
    ([2**63-1, 2**63-1], NoFiniteMLEError)])
def test_exact_existence_classes_before_solver(monkeypatch, data, error):
    def forbidden(*args):
        pytest.fail("ineligible sample reached solver")
    monkeypatch.setattr(_fitting, "_nb_root", forbidden)
    with pytest.raises(error) as caught:
        NegativeBinomialFamily().fit(data)
    assert type(caught.value) is error


def test_exact_overdispersion_uses_python_integer_arithmetic(monkeypatch):
    # Var = mean + 1, but float64 cannot distinguish these at this magnitude.
    a = 10**9
    m = a*a-1
    observed = []
    def stop(counts, n, total, excess):
        observed.append((n, total, excess))
        raise FitNumericalError("eligible diagnostic stop")
    monkeypatch.setattr(_fitting, "_nb_root", stop)
    with pytest.raises(FitNumericalError, match="eligible diagnostic stop"):
        NegativeBinomialFamily().fit([m-a, m+a])
    assert observed == [(2, 2*m, 4)]


# Fixed 70-digit references from bisection of the exact finite harmonic
# recurrence psi(r+x)-psi(r)=sum_{j=0}^{x-1}1/(r+j), not the production
# asymptotic digamma evaluator or Brent solver. Likelihood uses log products.
REFERENCES = [
    (SAMPLE, "0.683961663022270942078893240917581859759384440126681272532104151215084",
     "0.2041293738735170048661473201307145371109651386096319607713365622158416",
     "-12.78811507275900587642047751153196256066885718929340049613517540837715"),
    ([0, 0, 0, 0, 100], "0.03816736300972426820372528547843490306549049289844473694692599421024670",
     "0.001904733218277279944438111158583318238592809864649955849539525653657657",
     "-9.060451728951057054342167599783791217659396332379336685224213950605165"),
]


@pytest.mark.parametrize("data,r,p,ll", REFERENCES)
def test_fixed_high_precision_references_and_full_likelihood(data, r, p, ll):
    result = NegativeBinomialFamily().fit(data)
    parameters = result.fitted_distribution.parameters
    assert type(parameters.r) is float and not parameters.r.is_integer()
    assert parameters.r == pytest.approx(float(r), rel=3e-12)
    assert parameters.p == pytest.approx(float(p), rel=3e-12)
    assert result.log_likelihood == pytest.approx(float(ll), rel=2e-14)
    assert result.aic == pytest.approx(4-2*float(ll))
    assert result.bic == pytest.approx(2*math.log(len(data))-2*float(ll))
    assert sum(stats.nbinom.logpmf(data, parameters.r, parameters.p)) == pytest.approx(float(ll), abs=2e-12)
    with localcontext() as context:
        context.prec = 70
        rd = Decimal.from_float(parameters.r)
        mean = Decimal(sum(data))/len(data)
        # Bounded to the small fixed reference samples above (max 100).
        score = sum(sum(1/(rd+j) for j in range(x)) for x in data) - len(data)*(1+mean/rd).ln()
        assert abs(score) < Decimal('1e-10')


@pytest.mark.parametrize("data", [SAMPLE, [0, 0, 0, 0, 100], [0, 1, 4, 5, 20, 30]])
def test_independent_two_dimensional_likelihood_oracle(data):
    data = np.asarray(data)
    def objective(z):
        r = math.exp(z[0])
        p = special.expit(z[1])
        return -float(np.sum(stats.nbinom.logpmf(data, r, p)))
    # Optimize both coordinates independently; no profiled mean or score.
    oracles = [optimize.minimize(objective, start, method="Nelder-Mead",
                options={"maxiter": 2000, "maxfev": 4000, "xatol": 1e-10, "fatol": 1e-11})
               for start in ([0., -1.], [-2., -4.])]
    assert all(o.success for o in oracles)
    result = NegativeBinomialFamily().fit(data)
    r, p = result.fitted_distribution.parameters.r, result.fitted_distribution.parameters.p
    for oracle in oracles:
        assert r == pytest.approx(math.exp(oracle.x[0]), rel=3e-6)
        assert p == pytest.approx(special.expit(oracle.x[1]), rel=3e-6)
        assert result.log_likelihood == pytest.approx(-oracle.fun, abs=2e-11)


def test_permutation_and_replication_metamorphisms():
    original = NegativeBinomialFamily().fit(SAMPLE)
    permuted = NegativeBinomialFamily().fit(list(reversed(SAMPLE)))
    replicated = NegativeBinomialFamily().fit(SAMPLE*7)
    assert permuted.fitted_distribution.parameters == original.fitted_distribution.parameters
    assert permuted.log_likelihood == original.log_likelihood
    assert replicated.fitted_distribution.parameters.r == pytest.approx(original.fitted_distribution.parameters.r, rel=3e-12)
    assert replicated.fitted_distribution.parameters.p == pytest.approx(original.fitted_distribution.parameters.p, rel=3e-12)
    assert replicated.log_likelihood == pytest.approx(7*original.log_likelihood, rel=2e-14)


def test_solver_invocation_is_bounded_and_matches_provenance(monkeypatch):
    real = optimize.brentq
    calls = []
    def spy(function, low, high, **kwargs):
        calls.append(kwargs.copy())
        assert function(low, *kwargs["args"]) > 0 > function(high, *kwargs["args"])
        return real(function, low, high, **kwargs)
    monkeypatch.setattr(optimize, "brentq", spy)
    result = NegativeBinomialFamily().fit(SAMPLE)
    assert len(calls) == 1
    assert calls[0]["maxiter"] == 128
    assert result.metadata["solver_id"] == "scipy.optimize.brentq"
    assert result.fitted_distribution.parameters.r != pytest.approx(np.mean(SAMPLE)**2/(np.var(SAMPLE)-np.mean(SAMPLE)), rel=1e-3)


def test_bracket_budget_failure_is_numerical(monkeypatch):
    monkeypatch.setattr(_fitting, "_BRACKET_STEPS", 3)
    monkeypatch.setattr(_fitting, "_profile_score", lambda *args: 1.)
    with pytest.raises(FitNumericalError, match="bracketing budget"):
        NegativeBinomialFamily().fit(SAMPLE)


def test_actual_solver_iteration_exhaustion_is_numerical(monkeypatch):
    monkeypatch.setattr(_fitting, "_SOLVER_ITERATIONS", 1)
    with pytest.raises(FitNumericalError, match="did not converge"):
        NegativeBinomialFamily().fit(SAMPLE)


@pytest.mark.parametrize("likelihood", [float("nan"), float("inf"), -float("inf"), -1.7e308])
def test_nonfinite_final_metrics_never_return_result(monkeypatch, likelihood):
    monkeypatch.setattr(_fitting, "_nb_log_likelihood", lambda *args: likelihood)
    with pytest.raises(FitNumericalError):
        NegativeBinomialFamily().fit(SAMPLE)


@pytest.mark.parametrize("error", [ValueError("likelihood"), OverflowError("likelihood"), FloatingPointError("likelihood")])
def test_likelihood_backend_errors_preserve_cause(monkeypatch, error):
    def fail(*args):
        raise error
    monkeypatch.setattr(_fitting, "_nb_log_likelihood", fail)
    with pytest.raises(FitNumericalError) as caught:
        NegativeBinomialFamily().fit(SAMPLE)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("root,converged", [(0., False), (0., True), (float("nan"), True), (100., True)])
def test_convergence_and_residual_fail_closed(monkeypatch, root, converged):
    monkeypatch.setattr(optimize, "brentq", lambda *a, **k: (root, SimpleNamespace(converged=converged)))
    with pytest.raises(FitNumericalError):
        NegativeBinomialFamily().fit(SAMPLE)


@pytest.mark.parametrize("error", [ValueError("oracle"), OverflowError("oracle"), FloatingPointError("oracle")])
def test_solver_exception_translation(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(optimize, "brentq", fail)
    with pytest.raises(FitNumericalError) as caught:
        NegativeBinomialFamily().fit(SAMPLE)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_solver_does_not_catch_control_exceptions(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(optimize, "brentq", fail)
    with pytest.raises(type(error)):
        NegativeBinomialFamily().fit(SAMPLE)


def test_warning_capture_on_nb_solver_path(monkeypatch):
    real = optimize.brentq
    def warn(*args, **kwargs):
        warnings.warn("first", RuntimeWarning)
        warnings.warn("second", UserWarning)
        return real(*args, **kwargs)
    monkeypatch.setattr(optimize, "brentq", warn)
    filters, state = list(warnings.filters), np.geterr().copy()
    result = NegativeBinomialFamily().fit(SAMPLE)
    assert result.warnings == ("RuntimeWarning: first", "UserWarning: second")
    assert warnings.filters == filters and np.geterr() == state


@pytest.mark.parametrize("expression,expected", [
    ("[999000,1001000,999000,1001001]", "large"),
    ("[10**12-1-10**6,10**12-1+10**6]", "numerical"),
    ("[0]*99+[10**18]", "small"),
    ("[0,2**63-1]", "finite"),
    ("[10**18-1-10**9,10**18-1+10**9]", "numerical"),
])
def test_extreme_fits_have_external_timeout(expression, expected):
    code = f'''
import json, math
from pyMagicStat.distributions.families import NegativeBinomialFamily, FitNumericalError
try:
    result = NegativeBinomialFamily().fit({expression})
    params = result.fitted_distribution.parameters
    print(json.dumps(dict(r=params.r, p=params.p, ll=result.log_likelihood)))
except FitNumericalError:
    print(json.dumps(dict(error="numerical")))
'''
    run = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, timeout=30, check=True)
    result = json.loads(run.stdout)
    if expected == "numerical":
        assert result == {"error": "numerical"}
    else:
        assert "error" not in result
        assert math.isfinite(result["ll"]) and 0 < result["p"] < 1
        assert result["r"] > 0
        if expected == "large":
            assert result["r"] > 1e9
        elif expected == "small":
            assert result["r"] < .001


def test_canonical_rounding_loss_has_independent_likelihood_evidence_and_fails_closed():
    code = '''
from decimal import Decimal as D, localcontext
from unittest.mock import patch
from pyMagicStat.distributions.families import NegativeBinomialFamily, FitNumericalError
from pyMagicStat.distributions.families import _fitting
x = [10**12-1-10**6, 10**12-1+10**6]
seen = []
real = _fitting._validate_nb_canonical_pair
def inspect(counts, n, total, r, p):
    assert type(r) is float and type(p) is float
    seen.append((r,p))
    return real(counts,n,total,r,p)
with patch.object(_fitting, '_validate_nb_canonical_pair', inspect):
    try:
        NegativeBinomialFamily().fit(x)
    except FitNumericalError as error:
        assert 'canonical' in str(error)
    else:
        raise AssertionError('unsafe canonical pair returned as success')
assert len(seen) == 1
r,p = seen[0]
assert p == 0.999999999999
with localcontext() as context:
    context.prec = 100
    rd,pd = D.from_float(r),D.from_float(p)
    mean = D(sum(x))/2
    competitor = D.from_float(float(mean*pd/(1-pd)))
    # Independent Stirling oracle: every argument is > 1e11, so the
    # omitted 1/(1260*z**5) term is < 1e-58. No production helper/SciPy PMF.
    def lg(z):
        return (z-D('.5'))*z.ln()-z+D('0.9189385332046727417803297364056176398613974736377834128171515404827657')+1/(12*z)-1/(360*z**3)
    def ll(shape):
        return sum(lg(shape+v)-lg(shape)-lg(D(v)+1)+shape*pd.ln()+D(v)*(1-pd).ln() for v in x)
    gain = ll(competitor)-ll(rd)
    assert D(489) < gain < D(490)
    assert -D(31) < ll(competitor) < -D(30)
    assert competitor != rd
print('canonical rejection and independent complete-likelihood competitor PASS')
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=30, check=True)
    assert "PASS" in result.stdout


def test_representability_check_uses_ulp_budget_not_parameter_caps():
    code = '''
from pyMagicStat.distributions.families import FitNumericalError
from pyMagicStat.distributions.families import _fitting
counts=((10**12-1-10**6,1),(10**12-1+10**6,1))
total=2*(10**12-1)
# Isolate the postcondition: the larger coordinate competitor is numerically
# representable, while the smaller solved-root pair loses optimality. The
# fitting API must still reject the latter, never return the competitor.
_fitting._validate_nb_canonical_pair(counts,2,total,1.0000221222075029e24,0.999999999999)
try:
    _fitting._validate_nb_canonical_pair(counts,2,total,9.999999999972128e23,0.999999999999)
except FitNumericalError:
    pass
else:
    raise AssertionError('same p and smaller r must fail on likelihood, not a cap')
print('PASS')
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=30, check=True)
    assert "PASS" in result.stdout
