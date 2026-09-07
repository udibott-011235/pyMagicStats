# DEC-012 — CP03 discrete distribution core executable contract

- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP03 — IN_PROGRESS`
- **Estado del registro:** `accepted`
- **Estado de arquitectura:** `FROZEN`
- **Estado de implementación:** `PENDING`
- **Fecha:** 2026-09-07
- **Baseline:** `origin/main` @ `02a65c80c5da10295d6eeef42e691772d0686ca2`
- **Rama:** `feature/distribution-family-framework-cp03-discrete-core`
- **Owner de arquitectura:** `statistical-software-architecture`
- **Implementación futura:** `implementation-engineering`
- **Evidencia:** `EV-009`
- **Extiende:** `DEC-010`, `DEC-011`
- **Supersedes:** ninguno

## 1. Purpose and authorized scope

CP03 freezes the executable architecture for a discrete probability-family
core and `NegativeBinomialFamily`. This decision defines contracts only. It
does not authorize production code, tests, fitting, GOF integration or any
later checkpoint.

The future implementation may add only the minimum discrete-family surface
under `pyMagicStat/distributions/families/` plus additive public exports and
focused tests. The legacy discrete modules remain isolated.

## 2. Support contract

`SupportKind` gains exactly:

```python
DISCRETE = "discrete"
```

`DistributionSupport` remains the single immutable mathematical support value
object. CP03 does not create `DiscreteDistributionSupport`, `DomainConstraint`
or fields for lattice, step or origin.

For `kind=DISCRETE`, membership means a finite, integer-valued real number that
also satisfies the existing lower/upper bound and endpoint-closure rules.
Booleans are never numeric support members. Array-like membership is evaluated
elementwise and preserves shape.

For the Negative Binomial support `[0, +inf)`:

```text
contains(0)      = True
contains(1.0)    = True
contains(0.5)    = False
contains(-1)     = False
contains(True)   = False
contains(NaN)    = False
contains(+inf)   = False
contains(-inf)   = False
```

This deliberately models contiguous unit-integer supports. It is sufficient
for future Negative Binomial, Geometric, Bernoulli and Hypergeometric families
without introducing a generalized arbitrary-lattice abstraction in CP03.

## 3. Discrete object model

The minimum future public surface is:

```text
DiscreteDistributionFamily
ParameterizedDiscreteDistribution
NegativeBinomialParameters
NegativeBinomialFamily
SupportKind.DISCRETE
```

`DiscreteDistributionFamily` extends `DistributionFamily` and remains a
stateless descriptor. `ParameterizedDiscreteDistribution` extends
`ParameterizedDistribution`, adds `pmf` and `logpmf`, and shares `cdf`,
`logcdf`, `sf`, `logsf`, `ppf` and the explicit `rvs(..., rng=...)` contract.
Bound parameters remain immutable and are the sole authoritative parameter
state.

## 4. Negative Binomial parameterization

The canonical public construction API is:

```python
NegativeBinomialFamily().bind(r=..., p=...)
```

Parameter rules are:

- `r` is a finite real scalar with `r > 0`;
- `p` is a finite real scalar with `0 < p <= 1`;
- positive non-integer `r` is valid;
- `p=1` is valid and represents a degenerate distribution at zero;
- accepted parameters are canonicalized to Python `float`;
- booleans, NaN, either infinity, complex values, strings, `r <= 0`, `p <= 0`
  and `p > 1` are rejected.

`loc` is fixed internally to zero and is not a public parameter. The backend
mapping is exactly:

```python
scipy.stats.nbinom(n=r, p=p, loc=0)
```

The canonical moments are:

```text
mean     = r * (1 - p) / p
variance = r * (1 - p) / p**2
```

CP03 exposes no `mu`, `alpha`, `theta`, mean/dispersion aliases, link functions,
GLM parameterizations or fitting.

## 5. Probability query semantics

`pmf`, `logpmf`, `cdf`, `logcdf`, `sf` and `logsf` retain the CP02 query
contract. They accept finite real numeric scalars and non-empty array-like
inputs, preserve multidimensional shape, return Python `float` for scalar
results and `numpy.ndarray` with `float64` dtype for array results. pandas
labels are not preserved.

Inputs reject booleans anywhere, mixed bool/numeric structures, NaN, either
infinity, complex values, numeric strings and empty arrays.

Finite values outside support are valid mathematical evaluation points and are
delegated without clipping. In particular, finite non-integer values are valid
inputs to every discrete probability query; they are not rejected merely for
being fractional. Outside discrete support, `pmf` returns `0.0` and `logpmf`
returns `-inf`. The CDF methods preserve the standard right-continuous discrete
step function, including at negative finite inputs.

## 6. Quantile contract

`ppf(q)` accepts finite real numeric input with `0 <= q <= 1` and rejects
booleans, mixed bool/numeric structures, empty arrays, non-finite values and
out-of-range probabilities. Scalar results are Python `float`; array results
are `float64` ndarrays. Discrete PPF output is never cast to an integer dtype.

Endpoint results are canonicalized to the mathematical support:

```text
q == 0 -> lower support bound
q == 1 -> upper support bound
```

For Negative Binomial this is `ppf(0)=0.0` and `ppf(1)=+inf`. This intentionally
replaces SciPy's discrete `ppf(0)` sentinel `-1.0`. For `0 < q < 1`, the result
delegates to SciPy and normalizes to `float64`.

## 7. Random sampling contract

CP03 reuses the CP02 RNG and size validation unchanged. The public name is
`rng=`. Accepted RNGs are non-negative Python/NumPy integer seeds, excluding
booleans, and caller-owned `numpy.random.Generator` instances. `None`, negative
integers, `RandomState`, `SeedSequence`, `BitGenerator` and other objects are
rejected.

Integer seeds create local deterministic generators. Caller-owned generators
are passed through SciPy's `random_state=` interface and advance. No
process-global RNG is consumed. Sizes remain `None`, a non-negative integer or
a tuple of non-negative integers; booleans and negative dimensions are
rejected and zero-sized dimensions are valid.

Discrete sample normalization is distinct:

```text
size=None      -> Python int
explicit size -> numpy.ndarray, dtype int64, requested shape
```

The production implementation must add a discrete sample-result normalizer.
It must not pass discrete samples through the CP02 float64
`_normalize_result`, nor rename or behaviorally change that helper solely for
CP03.

## 8. Numerical failure contract

The existing backend-result NaN guard applies to `pmf`, `logpmf`, `cdf`,
`logcdf`, `sf`, `logsf` and `ppf`. Any backend NaN after valid input raises
`FloatingPointError` with stable backend numerical-failure semantics.
Legitimate infinities, including `logpmf` outside support and the upper PPF
endpoint, remain valid.

Mathematical parameter admissibility is distinct from backend sampling
executability. No arbitrary `r` or `p` threshold is authorized. If valid
parameters, RNG and size reach a backend that cannot sample because of numeric
or integer-range constraints, the failure must surface explicitly as a
numerical/backend sampling failure, never as invalid distribution parameters.
Exact exception translation is reserved for the production task.

Pre-merge adversarial review must use bounded-time extreme-`rvs` probes. A
reproducible unbounded backend call is unacceptable for integration.

## 9. Legacy isolation

CP03 must not modify:

```text
pyMagicStat/distributions/distributions.py
pyMagicStat/distributions/_discrete_gof.py
```

It must not change the behavior of `BinomialDistribution`,
`PoissonDistribution`, `DiscreteDistributionValidator` or the legacy Pearson
discrete GOF helpers. CP03 performs no migration of these APIs.

## 10. Future implementation tests and acceptance

The separately authorized production task must test:

- discrete support membership for scalar, mixed and multidimensional inputs;
- immutable/stateless family and bound-parameter ownership;
- all parameter boundaries, including positive non-integer `r` and `p=1`;
- exact SciPy mapping, moments and deterministic probability parity;
- fractional and negative finite evaluation points;
- the canonical `ppf(0)` and `ppf(1)` endpoint policy;
- scalar/array normalization for probability, PPF and discrete RVS outputs;
- CP02 RNG/size compatibility and isolation from global RNG state;
- backend NaN translation and legitimate infinities;
- bounded-time extreme-RVS failure probes;
- frozen legacy and CP02 regression surfaces.

The evidence in `EV-009` is pre-implementation baseline evidence only and does
not validate this future implementation.

## 11. Explicit non-goals

CP03 does not implement fitting, `FitResult`, `FittedDistribution`, GOF
integration, selector/routing, parameter estimation, `GeometricFamily`,
`BernoulliFamily`, `HypergeometricFamily`, Wave-2 families, zero-inflated
distributions or B3/UAT1 work. CP04–CP08 remain `NOT_STARTED`.

## Revision condition

Return to Architecture before changing discrete support membership, the single
support representation, canonical `r,p` parameterization, PPF endpoint policy,
RNG/size semantics, discrete result normalization, numerical-failure
classification, legacy isolation or any explicit non-goal.
