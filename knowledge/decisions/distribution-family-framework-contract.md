# DEC-010 — Distribution Family Framework architecture and contracts

- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP01 — READY_FOR_ARCHITECT_REVIEW`
- **Estado del registro:** `under_review`
- **Fecha:** 2026-09-06
- **Baseline:** `origin/main` @ `402e4601df460811779b3238c2526ac12f463a67`
- **Owner de arquitectura:** `statistical-software-architecture`
- **Materialización:** `implementation-engineering`
- **Evidencia:** `EV-007`
- **Supersedes:** ninguno

## 1. Purpose and non-goals

pyMagicStats is a general statistical framework. The distribution-family
architecture uses statistically neutral concepts and must not encode behavior
or vocabulary from any particular application domain.

The purpose of this contract is to freeze the conceptual boundaries, naming,
parameterizations and future validation obligations for probability families.
CP01 is documentation and governance only.

CP01 does not implement family classes, fitted objects, estimators, GOF,
automatic family selection, model ranking, inference routing, or changes to
existing distribution APIs.

## 2. Responsibility separation

The architecture distinguishes five concepts:

| Concept | Responsibility | Explicit non-responsibility |
|---|---|---|
| Sample description | Immutable observed sample and descriptive statistics | Population-family identity |
| Probability distribution family | Parameterization, support and probability operations | Ownership of raw sample descriptives |
| Fitted distribution | A family plus frozen fitted parameters and backend identity | Raw fitting process or GOF decision |
| Goodness-of-fit assessment | Evidence about compatibility with a stated model | Proof of population-family identity |
| Shape / approximation diagnostic | Observable shape or approximation evidence | Probability-family abstraction or selector authority |

These concepts must not be collapsed into one class. Failure to reject GOF is
not proof that a population belongs to a family, and a fitted distribution is
not a raw sample-description object.

## 3. Existing-surface compatibility boundary

During CP01 the following classes remain unchanged:

- `Distribution`
- `NormalDistribution`
- `LognormalDistribution`
- `BinomialDistribution`
- `PoissonDistribution`

CP01 does not rename, deprecate, re-export, wrap or alter these classes. It does
not change their behavior, return contracts or Gate-2 semantics. Their future
migration classifications are recorded in section 18 and in `EV-007`.

## 4. Naming policy

The existing `NormalDistribution` denotes diagnostic/shape behavior rather
than the new probability-family abstraction. New family abstractions therefore
use the `*Family` convention:

- `NormalFamily`
- `GammaFamily`
- `ExponentialFamily`
- `WeibullFamily`
- `BetaFamily`
- `NegativeBinomialFamily`

No new family class is implemented in CP01.

## 5. Approved abstraction hierarchy

The conceptual hierarchy is deliberately shallow:

```text
DistributionFamily
├── ContinuousDistributionFamily
└── DiscreteDistributionFamily
```

Family-specific behavior normally uses composition with a numerical backend,
not deep inheritance.

The future `DistributionFamily` contract defines conceptually:

```text
name
kind
support
parameter_specs
validate_parameters(...)
cdf(...)
logcdf(...)
sf(...)
logsf(...)
ppf(...)
rvs(..., rng=...)
theoretical_mean(...)
theoretical_variance(...)
```

`ContinuousDistributionFamily` additionally defines `pdf(...)` and
`logpdf(...)`. `DiscreteDistributionFamily` additionally defines `pmf(...)`
and `logpmf(...)`. A discrete family has no artificial `pdf()` method, and a
continuous family has no artificial `pmf()` method.

## 6. Backend ownership policy

SciPy is the default numerical probability-distribution backend. pyMagicStats
does not reimplement mature distribution mathematics without an explicit,
documented reason.

SciPy normally owns:

- `pdf` and `logpdf` for continuous families;
- `pmf` and `logpmf` for discrete families;
- `cdf`, `logcdf`, `sf`, `logsf`, `ppf` and numerical probability
  calculations;
- numerical random-variate generation;
- mature fitting primitives where statistically appropriate.

pyMagicStats owns:

- canonical parameterization and backend conversion;
- parameter and support validation;
- stable API and result contracts;
- provenance, backend identity and version metadata;
- fit interpretation and reproducibility;
- caller-injected RNG contracts and random-state provenance;
- fail-closed behavior;
- GOF contracts in their separate assessment layer.

No implicit process-global RNG is authorized. The future exact Python RNG API
remains implementation-reviewable, but callers must be able to inject and
control reproducible RNG state. The `rvs(..., rng=...)` operation is required
for later simulation, calibration and possible parametric-bootstrap procedures.

## 7. Support contract

Support is represented explicitly and independently of raw data. The future
support object must express:

- lower and upper bounds;
- whether each finite boundary is open or closed;
- continuous versus integer support;
- infinite boundaries.

Support violations fail closed. The implementation must never silently clip,
round, discard or coerce out-of-support observations.

## 8. Fitted-distribution contract

The future immutable `FittedDistribution` is the authoritative representation
of:

- family identity;
- canonical parameters;
- parameterization;
- support;
- backend identity.

It delegates probability operations to the family/backend using frozen
parameters. It is immutable from the user perspective and remains conceptually
separate from the fitting operation, raw sample descriptives and GOF.

The approved high-level fitting direction is:

```text
family.fit(data, ...) -> FitResult
FitResult.fitted_distribution -> FittedDistribution
```

The family owns parameterization, support, validation and backend conversion.
`FitResult` contains one canonical `fitted_distribution`; it must not maintain
independent duplicated authoritative family, parameter, parameterization or
support state. Internal fitting helpers may use composition. A separate global
fitter registry is not part of the first implementation.

## 9. FitResult contract

The future immutable `FitResult` must represent:

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
backend
backend_version
metadata
```

`FittedDistribution` owns family identity, canonical fitted parameters,
parameterization, support and backend identity. `FitResult` owns information
about the fitting process. Convenience accessors may later expose family,
parameters or support through `fitted_distribution`, but they must not create a
second authoritative copy. Backend/version provenance may appear in
`FitResult` where needed while backend identity remains part of the fitted
distribution.

Not every metric applies to every family. Semantically different states must
not be collapsed into `NaN`. At minimum, the contract preserves:

- `NOT_APPLICABLE`
- `NOT_ESTIMATED`
- `NOT_ASSESSED`
- `FAILED`

The exact representation of these states may be finalized during an authorized
implementation review, but the distinctions are mandatory.

## 10. GOF separation

GOF remains a separate assessment layer and is not implemented in CP01. Family
objects must not expose identity-claiming helpers such as
`is_valid_distribution(...)` or family-specific `is_*` predicates.

Fitting parameters and then applying a nominal reference-distribution test to
the same sample does not automatically produce a calibrated p-value. GOF
calibration for fitted families belongs to `STAGE-DIST-FAMILIES-001 / CP05`.
Existing Binomial/Poisson Gate-2 behavior is not generalized automatically.

CP05 must distinguish:

```text
simple null:    H0: X ~ F(theta0), theta0 known
composite null: H0: X ~ F(theta), theta estimated from the analyzed sample
```

A nominal GOF reference distribution valid for the simple-null case must not
be assumed valid for the composite-null case. Exact family-specific EDF, CvM,
AD or bootstrap methodology remains `ARCHITECT_DECISION_REQUIRED`; this
contract selects no universal GOF test.

## 11. Family roadmap

| Wave | Families | Status in CP01 |
|---|---|---|
| Wave 1 | Gamma, Exponential, Negative Binomial | Prioritized; not implemented |
| Wave 2 | Weibull, Beta, Geometric, Hypergeometric, Pareto, Bernoulli | Preliminary contract; not implemented |
| Wave 3 / later | Logistic, Log-Logistic, Beta-Binomial, zero-inflated count models | Deferred |

The roadmap is prioritization only and creates no implementation authorization.

### Existing-family core compatibility track

Separate from the wave roadmap, future `BinomialFamily` and `PoissonFamily`
cores will provide probability-family operations beneath the validated legacy
`BinomialDistribution` and `PoissonDistribution` surfaces.

Conceptually:

```text
legacy BinomialDistribution
    -> preserved legacy GOF/diagnostics
    -> BinomialFamily for probability-family operations

legacy PoissonDistribution
    -> preserved legacy GOF/diagnostics
    -> PoissonFamily for probability-family operations
```

This compatibility track does not authorize implementation in CP01, modify
legacy APIs or Gate-2 behavior, transfer Gate-2 calibration to other families,
or deprecate existing classes. `NormalDistribution` and
`LognormalDistribution` remain `MIGRATE_LATER` and are not silently converted
into family objects.

## 12. Parameterization matrix

| Future family | Canonical parameters | Initial support | Backend | Contract status |
|---|---|---|---|---|
| `GammaFamily` | `shape > 0`, `scale > 0`; `loc=0` | `[0, +∞)` | `scipy.stats.gamma` | Wave 1 frozen |
| `ExponentialFamily` | `scale > 0`; `loc=0` | `[0, +∞)` | `scipy.stats.expon` | Wave 1 frozen |
| `NegativeBinomialFamily` | `r > 0`, `0 < p <= 1` | non-negative integers | `scipy.stats.nbinom` | Wave 1 frozen |
| `WeibullFamily` | `shape > 0`, `scale > 0`; `loc=0` | `[0, +∞)` | `scipy.stats.weibull_min` | Wave 2 preliminary |
| `BetaFamily` | `alpha > 0`, `beta > 0`; `loc=0`, `scale=1` | `[0, 1]` | SciPy beta backend | Wave 2 preliminary |
| `GeometricFamily` | `0 < p <= 1` | positive integers beginning at 1 | SciPy geometric backend | Wave 2 preliminary; parity check required |
| `HypergeometricFamily` | `population_size`, `success_states`, `draws` | finite integer support derived from parameters | SciPy hypergeometric backend | Wave 2 preliminary |
| `ParetoFamily` | `shape > 0`, positive minimum `scale` | `[scale, +∞)` | SciPy Pareto backend | Wave 2 preliminary; Type I only |
| `BernoulliFamily` | `0 <= p <= 1` | `{0, 1}` | SciPy Bernoulli backend | Wave 2 preliminary |

## 13. Wave-1 frozen parameterizations

### GammaFamily

Canonical storage is `shape` plus `scale`, both strictly positive, with
`loc=0`. Backend conversion is:

```python
scipy.stats.gamma(a=shape, loc=0, scale=scale)
```

Rate is not stored as a second canonical parameter. A future conversion may
accept `rate = 1 / scale`, while canonical storage remains `shape + scale`.

### ExponentialFamily

Canonical storage is strictly positive `scale`, with `loc=0`. Backend
conversion is:

```python
scipy.stats.expon(loc=0, scale=scale)
```

### NegativeBinomialFamily

Canonical parameters are dispersion `r > 0` and success probability
`0 < p <= 1`; `r` is not restricted to integers. For integer `r`, the family
admits the classical interpretation that `X` counts failures before `r`
successes. For non-integer `r`, `r` is a positive generalized shape/dispersion
parameter. Support remains `{0, 1, 2, ...}`.

The public canonical parameter is `r`, not `n`, because sample size already
uses `n` throughout the framework. Backend conversion is:

```python
scipy.stats.nbinom(n=r, p=p)
```

Canonical moments are:

```text
mean     = r * (1 - p) / p
variance = r * (1 - p) / p**2
```

Wave 1 does not introduce NB1, NB2 or mean-dispersion regression
parameterizations.

## 14. Wave-2 preliminary parameterizations

- `WeibullFamily`: positive shape and scale, `loc=0`, using
  `scipy.stats.weibull_min`.
- `BetaFamily`: positive alpha and beta with fixed support `[0, 1]`, `loc=0`
  and `scale=1`; arbitrary free location/scale is excluded initially.
- `GeometricFamily`: `p` in `(0, 1]`, support beginning at 1; exact SciPy
  parity must be verified before implementation.
- `HypergeometricFamily`: public names are `population_size`,
  `success_states` and `draws`; backend conversions must be documented.
- `ParetoFamily`: Pareto Type I with positive shape and minimum scale; shifted
  generalized semantics are excluded initially.
- `BernoulliFamily`: `p` in `[0, 1]` and support `{0, 1}`.

These contracts are preliminary and do not authorize implementation.

## 15. Input validation policy

Future fitting contracts fail closed for:

- empty samples;
- `NaN` and infinity;
- multidimensional data unless explicitly supported;
- nonnumeric observations;
- support violations;
- fractional observations for integer-only families;
- illegal parameter boundaries;
- structurally degenerate inputs where estimation is undefined.

No silent uncertainty conversion is permitted.

## 16. ARCHITECT_DECISION_REQUIRED

The following questions remain deliberately unresolved:

1. MLE versus method-of-moments exposure.
2. Whether additional estimators become public.
3. Finite-sample estimator behavior.
4. Parameter uncertainty.
5. Exact family-specific fitted-family GOF calibration, including EDF, CvM, AD
   and bootstrap methodology.
6. Optimization convergence policy.
7. Boundary estimates.
8. Discrete-family estimation rules.
9. Zero-heavy models.
10. Likelihood-based model-comparison semantics.
11. AIC/BIC comparability constraints.
12. Treatment of `loc` for future families.
13. Confidence intervals for fitted parameters.

Cortex must not resolve these questions in CP01.

## 17. Future test strategy

Future implementation checkpoints must define, before claiming readiness:

- contract tests for names, kinds, support and parameter specifications;
- valid and invalid parameter-boundary tests;
- support membership and fail-closed observation validation;
- continuous/discrete interface separation;
- oracle comparisons to SciPy for probability functions and moments;
- metamorphic identities such as `cdf + sf` consistency and quantile round
  trips within explicit float64 tolerances;
- extreme-tail, infinite-boundary and scale-stability cases;
- immutable fitted-object and JSON-ready result contracts;
- fixed-versus-estimated parameter accounting;
- convergence, warning and semantic-state propagation;
- reproducibility and backend-version metadata;
- caller-controlled RNG reproducibility without process-global state;
- family-specific fitting validation after the reserved estimator decisions;
- separate preregistered GOF calibration in CP05, with simple and composite
  nulls treated explicitly.

Tests must distinguish software accuracy from statistical calibration. Passing
backend-oracle tests does not validate estimation or GOF claims.

## 18. Migration classification for existing classes

| Existing class | Current responsibility | Future category | Reason |
|---|---|---|---|
| `Distribution` | Sample snapshot and descriptives with legacy assessment storage | `KEEP_AS_IS` | The sample-description role remains distinct from probability families; legacy fields require no CP01 change |
| `NormalDistribution` | Exact-normality/shape and Q-Q diagnostic | `MIGRATE_LATER` | Its name conflicts semantically with family terminology, but compatibility forbids a CP01 rename or behavior change |
| `LognormalDistribution` | Diagnostic of Gaussianity after log transform | `MIGRATE_LATER` | It is an assessment rather than a fitted family and needs a future diagnostic-layer migration |
| `BinomialDistribution` | Discrete validation, Pearson GOF, approximation diagnostic and moments helper | `WRAP_NEW_CORE` | A future wrapper can preserve the legacy surface while delegating probability operations to `BinomialFamily` and keeping GOF separate |
| `PoissonDistribution` | Discrete validation, Pearson GOF and approximation diagnostic | `WRAP_NEW_CORE` | A future wrapper can preserve compatibility while delegating probability operations to `PoissonFamily` and isolating GOF |

No class is marked `DEPRECATE_LATER` in CP01 because the approved contract
requires compatibility and does not yet authorize removal. These
classifications are recommendations for architecture review, not migrations.

## Selector boundary and revision condition

This stage does not authorize automatic family selection, automatic model
ranking, automatic inference routing or `MethodSelector` changes.

This decision must return to Architecture if a future checkpoint changes the
responsibility boundaries, naming convention, shallow hierarchy, backend
ownership, canonical parameterizations, support semantics, fitting location,
GOF boundary, selector boundary or any item currently marked
`ARCHITECT_DECISION_REQUIRED`.
