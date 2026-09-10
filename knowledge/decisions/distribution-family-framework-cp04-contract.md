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

The following additions are public from both
`pyMagicStat.distributions.families` and `pyMagicStat.distributions`:

```text
FittedDistribution
FittedContinuousDistribution
FittedDiscreteDistribution
FitResult
DistributionFitError
FitIdentifiabilityError
NoFiniteMLEError
FitNumericalError
```

They are not exported from the package root. Existing public exports are not
removed or renamed.

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

`FittedDistribution` delegates the existing read-only properties
`backend_library`, `backend_distribution` and `backend_version` from its sole
wrapped `ParameterizedDistribution`. The probability backend is the backend
used by that fitted distribution; it is distinct from the estimator that
generated the fitted parameters.

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

`FitResult` exposes these read-only derived properties exactly:

```python
backend = fitted_distribution.backend_distribution
backend_version = fitted_distribution.backend_version
```

They are not independent stored state. `fixed_parameters` and
`estimated_parameters` contain names, never duplicated parameter values:

| Family | Fixed names | Estimated names |
|---|---|---|
| Gamma | `("loc",)` | `("shape", "scale")` |
| Exponential | `("loc",)` | `("scale",)` |
| Negative Binomial | `("loc",)` | `("r", "p")` |

`estimation_method` is exactly `"maximum_likelihood"`.

`metadata` contains immutable string provenance that distinguishes the
estimator from the probability backend. It contains `estimator_id` and
`solver_id` with these required meanings:

| Family | `estimator_id` | `solver_id` |
|---|---|---|
| Gamma | `scipy-gamma-fixed-loc-mle-v1` | `scipy.stats.gamma.fit` |
| Exponential | `pymagicstats-exponential-closed-form-mle-v1` | `closed_form` |
| Negative Binomial | `pymagicstats-negative-binomial-profile-mle-v1` | exact identifier of the deterministic bounded root solver actually used by production |

Neither provenance field contains or duplicates canonical fitted parameter
values. The Negative Binomial `solver_id` must name the actual production
solver rather than a generic optimizer category, and its acceptance test must
match the executed solver.

Returned results represent successful finite fits only: `converged` is
`True`; `log_likelihood`, `aic` and `bic` are finite Python `float` values;
`warnings` is an immutable tuple of strings; and `metadata` is deeply
immutable and contains no authoritative parameter copy. Failed fits raise
typed errors and never return partial results. `NaN` does not encode a
semantic state.

This reconciles CP04 with the semantic states in `DEC-010`:

- `SUCCESS` returns a `FitResult`; every mandatory metric is applicable,
  assessed and finite.
- `NOT_ESTIMATED` is represented structurally by parameter names in
  `fixed_parameters`, never by `NaN` or a fabricated estimate.
- `FAILED` is represented by the typed exception hierarchy; no partial
  `FitResult` is returned.
- No mandatory CP04 `FitResult` metric may enter `NOT_APPLICABLE` or
  `NOT_ASSESSED`, so CP04 emits no placeholder for either state. Any future
  optional metric that can enter one of those states requires a typed
  sentinel or enum and a new Architecture decision; `None` and `NaN` must not
  collapse their distinct meanings.

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
  fail closed. In particular, `0`, `-0.0`, `1`, `1.0` and NumPy integer
  equivalents are accepted. Integer-valued real inputs may canonicalize
  losslessly to `int64`; `0.5`, booleans, non-finite values and values outside
  the `int64` range are rejected. Rounding, truncation and wraparound are
  forbidden.

## 6. Continuous maximum likelihood

### Exponential

With `loc=0`, the estimate is analytical:

```text
scale_hat = arithmetic mean(data)
```

Mixed zero/positive observations are valid. An all-zero sample has no finite
MLE satisfying `scale > 0`.

### Gamma

`loc` is fixed at zero. The implementation invokes exactly:

```python
scipy.stats.gamma.fit(
    data,
    floc=0,
    method="MLE",
)
```

Acceptance tests confirm that both `floc` and `method` are passed explicitly.
The returned shape, location and scale are strictly validated. Positive
constant data has no finite Gamma MLE; any zero observation has no finite
regular Gamma MLE. A zero, infinite, NaN or arbitrarily capped fitted
parameter is invalid.

Warnings are captured and normalized on every fitting path, including Gamma,
Exponential and Negative Binomial. A warning may be recorded in a successful
immutable `warnings` tuple only after every fit postcondition passes; invalid
output is never accepted or silently ignored.

SciPy documents that `rv_continuous.fit` defaults to MLE and that `floc` fixes
the location parameter: [SciPy 1.18.0 `rv_continuous.fit` documentation](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.stats.rv_continuous.fit.html).

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
distribution's integrality metadata. Evidence-only reconnaissance under SciPy
1.18.1 observed that private `stats.nbinom._shape_info()` marks shape `n` with
`integrality=True`, even though pyMagicStats' canonical generalized `r` is any
positive finite real. Production must not depend on `_shape_info()`, and no
claim is made for every future SciPy version. The custom estimator is
authoritative because pyMagicStats itself guarantees real `r`, independently
of backend fitter metadata. See the
[SciPy 1.18.0 `fit` documentation](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.stats.fit.html)
and the version-pinned
[SciPy 1.18.1 `nbinom` source](https://github.com/scipy/scipy/blob/v1.18.1/scipy/stats/_discrete_distns.py).

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
The primary existence/uniqueness authority is Simonsen's 1976 result together
with its 1980 correction: [Simonsen (1976)](https://www.tandfonline.com/doi/abs/10.1080/03461238.1976.10405618),
[Simonsen (1980), correction](https://www.tandfonline.com/doi/abs/10.1080/03461238.1980.10408657).
Aragón, Eberly and Eberly (1992) is retained as historical context only;
[Wang (1996)](https://doi.org/10.1016/0167-7152(94)00259-2) identified a major
problem in its proof. Later computational and analytical support includes
[Bandara, Gill and Mitra (2019)](https://doi.org/10.1016/j.spl.2019.01.009)
and [Yang et al. (2026)](https://link.springer.com/article/10.1007/s00362-026-01842-x).

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
- delegated read-only fitted-distribution backend properties;
- exact derived `FitResult.backend` and `FitResult.backend_version`
  properties, all frozen estimator/solver identifiers, and absence of
  duplicated parameter state;
- the additive exports from `pyMagicStat.distributions.families` and
  `pyMagicStat.distributions`, with no new package-root exports;
- continuous/discrete interface separation;
- exact fit-input validation;
- exact Negative Binomial integer-valued canonicalization and rejection cases;
- Gamma parity against an independent fixed-`loc` MLE oracle;
- explicit Gamma backend arguments `floc=0` and `method="MLE"`;
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
- warning capture and normalization on every fitting path;
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
