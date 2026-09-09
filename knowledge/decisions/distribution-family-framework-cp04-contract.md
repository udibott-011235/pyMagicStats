# DEC-013 — CP04 Wave 1 fitting executable contract

- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP04 — IN_PROGRESS`
- **Estado del registro:** `accepted`
- **Estado de arquitectura:** `FROZEN`
- **Estado de implementación:** `PENDING`
- **Fecha:** 2026-09-08
- **Baseline:** `origin/main` @ `b3f35d4d7b221c457e2e730bfba2b104e1d07144`
- **Rama:** `feature/distribution-family-framework-cp04-wave1-fitting`
- **Owner de arquitectura:** `statistical-software-architecture`
- **Implementación futura:** `implementation-engineering`
- **Evidencia:** `EV-011`
- **Extiende:** `DEC-010`, `DEC-011`, `DEC-012`
- **Supersedes:** ninguno

## 1. Purpose and checkpoint gates

CP04 freezes the first fitting contract for the three Wave 1 families already
implemented. This decision defines architecture and future acceptance criteria
only; it does not authorize production or test implementation.

CP04 is divided into four internal gates:

| Gate | Required result |
|---|---|
| `CP04-A` | Immutable fitted/result core and exact fit-input contract |
| `CP04-B` | Gamma and Exponential fixed-`loc` maximum-likelihood estimation |
| `CP04-C` | Generalized Negative Binomial real-`r` maximum-likelihood estimation |
| `CP04-D` | Independent adversarial implementation audit |

CP04 is `COMPLETE` only after all four gates pass. Architecture is `FROZEN`,
implementation is `PENDING`, and the stage remains `IN_PROGRESS`.

## 2. Public fitting surface

The future public fitting surface is exactly:

```python
GammaFamily.fit(data) -> FitResult
ExponentialFamily.fit(data) -> FitResult
NegativeBinomialFamily.fit(data) -> FitResult
```

Every free canonical parameter is estimated. These methods expose no public
`method`, `optimizer`, `loc`, `fixed_shape`, `fixed_scale`, `fixed_r`,
`fixed_p`, `rate`, `mu`, `alpha`, `theta` or `dispersion` argument. Structural
`loc` remains fixed internally at zero.

CP04 does not expose user-controlled fixed canonical parameters, method of
moments, MAP, Bayesian estimation or alternative estimators.

## 3. Object model and ownership

The future public concepts are:

```text
FittedDistribution
FittedContinuousDistribution
FittedDiscreteDistribution
FitResult
```

A fitted object wraps exactly one existing `ParameterizedDistribution`
instance. It must not duplicate authoritative family, canonical parameters,
parameterization, support or probability-backend state.

`FittedContinuousDistribution` delegates the existing continuous operations.
`FittedDiscreteDistribution` delegates the existing discrete operations and
preserves the integer `rvs(..., rng=...)` result contract.

`FitResult` owns fitting-process information only and contains exactly one
canonical `fitted_distribution`. Its information fields are exactly:

```text
fitted_distribution
estimation_method
fixed_parameters
estimated_parameters
n_observations
log_likelihood
aic
bic
converged
warnings
metadata
```

`backend` and `backend_version` are derived from `fitted_distribution`; they
are not stored as a second authoritative copy. `fixed_parameters` and
`estimated_parameters` contain names, never duplicated parameter values:

| Family | Fixed names | Estimated names |
|---|---|---|
| Gamma | `("loc",)` | `("shape", "scale")` |
| Exponential | `("loc",)` | `("scale",)` |
| Negative Binomial | `("loc",)` | `("r", "p")` |

`estimation_method` is exactly `"maximum_likelihood"`.

Returned results represent successful finite fits only: `converged` is
`True`; `log_likelihood`, `aic` and `bic` are finite Python `float` values;
`warnings` is an immutable tuple of strings; and `metadata` is deeply
immutable and contains no authoritative parameter copy. Failed fits raise
typed errors and never return partial results. `NaN` does not encode a
semantic state.

Fitted objects, results and their nested state must preserve immutability under
normal use, `copy.copy`, `copy.deepcopy` and pickle round trips.

## 4. Failure taxonomy

The future public hierarchy is equivalent to:

```text
DistributionFitError
├── FitIdentifiabilityError
├── NoFiniteMLEError
└── FitNumericalError
```

Invalid public input remains `TypeError` or `ValueError`.

- `FitIdentifiabilityError`: valid data does not identify a unique canonical
  fit.
- `NoFiniteMLEError`: valid data has no finite MLE in the frozen parameter
  space.
- `FitNumericalError`: a finite fit is mathematically eligible but cannot be
  computed or validated numerically.

After public-data validation, backend `ValueError`, `OverflowError` or
`FloatingPointError` becomes `FitNumericalError` with the original exception
as `__cause__`. pyMagicStat input errors are not translated. The fitting layer
does not catch `KeyboardInterrupt`, `SystemExit` or `GeneratorExit`.

## 5. Fit-input contract

Each fit method accepts one non-empty, one-dimensional, numeric, real sample.
It rejects scalar input, multidimensional input, booleans (including mixed
arrays and `numpy.bool_`), strings, complex values, NaN, either infinity and
out-of-support observations.

The implementation must not silently drop, clip, round or coerce observations,
and `FitResult` must not retain the raw sample.

- Gamma uses its frozen support semantics, but any observed zero implies no
  finite regular Gamma MLE and raises `NoFiniteMLEError`.
- Exponential accepts finite `x >= 0`; an all-zero sample raises
  `NoFiniteMLEError`.
- Negative Binomial accepts only non-negative integer-valued observations
  exactly representable as `int64`; fractional or out-of-range observations
  fail closed.

## 6. Continuous maximum likelihood

### Exponential

With `loc=0`, the estimate is analytical:

```text
scale_hat = arithmetic mean(data)
```

Mixed zero/positive observations are valid. An all-zero sample has no finite
MLE satisfying `scale > 0`.

### Gamma

`loc` is fixed at zero. The implementation uses
`scipy.stats.gamma.fit(data, floc=0)` with MLE semantics and strictly validates
the returned shape, location and scale. Positive constant data has no finite
Gamma MLE; any zero observation has no finite regular Gamma MLE. A zero,
infinite, NaN or arbitrarily capped fitted parameter is invalid.

Backend warnings are captured. They may be recorded only when every final
postcondition passes; invalid output is never accepted or silently ignored.

SciPy documents that `rv_continuous.fit` defaults to MLE and that `floc` fixes
the location parameter: [SciPy `rv_continuous.fit` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html).

## 7. Generalized Negative Binomial MLE

The canonical parameterization remains:

```text
r > 0, finite and real
0 < p <= 1, finite and real
loc = 0
```

For observations `x_i` and mean `m > 0`, profile:

```text
p_hat(r) = r / (r + m)

score(r) =
    sum_i [digamma(r + x_i) - digamma(r)]
    + n * log(r / (r + m))
```

The implementation uses deterministic bounded one-dimensional root solving on
`r > 0`. It does not use `scipy.stats.fit` as the authority for `r`, constrain
or round `r` to integers, use a stochastic optimizer, expose an internal
`alpha=1/r` parameterization, or return a method-of-moments initializer as the
MLE.

This boundary is necessary because SciPy's generic fitting API honors each
distribution's integrality metadata, and SciPy currently marks `nbinom` shape
`n` integral even though pyMagicStats' canonical generalized `r` is any
positive finite real: [SciPy `fit` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fit.html),
[SciPy `nbinom` source](https://github.com/scipy/scipy/blob/main/scipy/stats/_discrete_distns.py).

Samples are classified before numerical solving:

1. All observations zero: `FitIdentifiabilityError`; `p=1` is degenerate at
   zero but does not uniquely identify `r`.
2. At least one positive observation and population variance less than or
   equal to the mean: `NoFiniteMLEError`; the likelihood supremum occurs at
   the Poisson boundary `r -> +infinity`, which is not a fitted canonical
   value.
3. Population variance greater than the mean: one finite positive profile root
   is mathematically eligible.

For integer data, overdispersion is classified exactly with Python integer
arithmetic:

```text
n * sum(x_i**2) - sum(x_i)**2 > n * sum(x_i)
```

This is a statistical existence classification, not a floating tolerance.
The existence and uniqueness condition is supported by Aragón, Eberly and
Eberly, *Existence and uniqueness of the maximum likelihood estimator for the
two-parameter negative binomial distribution*, Statistics & Probability
Letters 15(5), 1992,
[doi:10.1016/0167-7152(92)90157-Z](https://doi.org/10.1016/0167-7152(92)90157-Z).

Solver tolerances and iteration/evaluation caps are documented computational
controls only. They never become hidden validity thresholds for `r`, `p` or
observations. If an eligible root cannot be bracketed, resolved or represented
safely, fitting raises `FitNumericalError`; it never substitutes arbitrary
`r_max` or a Poisson approximation. The final score residual, `r`, `p` and
complete log-likelihood are validated.

## 8. Likelihood and information criteria

Every successful fit uses the complete sample log-likelihood, including all
distribution-dependent terms, and natural logarithms:

```text
AIC = 2*k - 2*log_likelihood
BIC = k*log(n_observations) - 2*log_likelihood
```

`k` counts estimated free canonical parameters: Gamma `2`, Exponential `1`,
Negative Binomial `2`. Fixed `loc` is excluded.

CP04 does not implement AICc, model ranking or claims that AIC/BIC values from
different data/support conditions are automatically comparable.

## 9. Required future acceptance strategy

The future implementation must test:

- immutable ownership and absence of duplicated authoritative state;
- continuous/discrete interface separation;
- exact fit-input validation;
- Gamma parity against an independent fixed-`loc` MLE oracle;
- the analytical Exponential MLE;
- generalized real-`r` NB MLE using fixed high-precision reference cases;
- independent two-dimensional likelihood/oracle cross-checks that do not
  reproduce the production profile algorithm;
- non-integer fitted `r`;
- permutation invariance;
- replicated-data MLE invariance and correct likelihood scaling;
- all-zero, variance-equals-mean and variance-below-mean classifications;
- barely overdispersed samples with very large finite `r`;
- strongly overdispersed samples with small `r`;
- large counts and `int64` boundaries;
- numerical/backend exception translation and cause preservation;
- bounded execution;
- no stochastic sample-recovery assertions that confuse sampling error with
  estimator correctness;
- complete CP02/CP03 regression and legacy isolation.

`CP04-D` independently inspects implementation source and uses independent
oracles. Passing implementation-authored tests alone is insufficient.

## 10. Explicit non-scope

CP04 does not include parameter uncertainty, covariance, standard errors,
confidence intervals, GOF or fitted-null calibration, selectors or routing,
regression/GLM parameterizations, zero-inflated or hurdle models, Geometric,
Bernoulli or Hypergeometric fitting, `FitResult` failure objects,
user-provided optimizers, stochastic fitting, legacy `Distribution` migration,
or CP05–CP08.

## Revision condition

Return to Architecture before changing public fit signatures, ownership,
immutability, failure classification, data validation, fixed `loc=0`, the
continuous estimators, the generalized real-`r` NB profile MLE or existence
classification, complete-likelihood accounting, information criteria, future
acceptance strategy or explicit non-scope.
