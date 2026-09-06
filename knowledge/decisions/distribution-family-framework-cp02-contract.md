# DEC-011 — CP02 continuous distribution core executable contract

- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP02 — IN_PROGRESS`
- **Estado del registro:** `accepted`
- **Fecha:** 2026-09-06
- **Baseline:** `origin/main` @ `ccff392af13d2cb52d1f3888a986ef58be0099e2`
- **Rama:** `feature/distribution-family-framework-cp02-continuous-core`
- **Owner de arquitectura:** `statistical-software-architecture`
- **Implementación:** `implementation-engineering`
- **Evidencia:** `EV-008`
- **Extiende:** `DEC-010`
- **Supersedes:** ninguno

## 1. Purpose and authorized scope

CP02 implements only the shared abstractions required by continuous
probability families, together with `GammaFamily` and `ExponentialFamily`.
It validates deterministic probability mechanics and API behavior against the
SciPy backend. It does not claim statistical calibration.

The implementation lives under:

```text
pyMagicStat/distributions/families/
├── __init__.py
├── _core.py
└── continuous.py
```

Additive exports from `pyMagicStat.distributions` are permitted. The existing
`pyMagicStat.distributions.distributions` and
`pyMagicStat.distributions._discrete_gof` modules remain unchanged.

## 2. State and ownership

`DistributionFamily` and `ContinuousDistributionFamily` are stateless
descriptors. `GammaFamily` and `ExponentialFamily` do not retain shape or scale
values.

Explicit manual parameterization is represented by immutable objects:

```text
ParameterizedDistribution
└── ParameterizedContinuousDistribution

GammaParameters
ExponentialParameters
```

The creation API is:

```python
GammaFamily().bind(shape=..., scale=...)
ExponentialFamily().bind(scale=...)
```

`bind()` creates independent immutable parameter state and never mutates the
family. It is not fitting, estimation, inference, GOF or model selection.

The canonical parameter object is the sole bound parameter source. A SciPy
frozen random-variable object is constructed lazily for operations and is not
stored as a second authoritative parameter source.

Future CP04 `FittedDistribution` must reuse one canonical
`ParameterizedDistribution` state. It must not duplicate family, parameters,
support or backend state. CP02 does not implement `FittedDistribution` or
`FitResult`.

## 3. Support contract

`DistributionSupport` is a frozen generic mathematical support value object
with:

```text
lower
upper
lower_closed
upper_closed
kind
```

`SupportKind` contains at least `CONTINUOUS` and remains extensible for a later
discrete kind without changing the representation.

Gamma and Exponential both use:

```text
lower = 0
upper = +inf
lower_closed = true
upper_closed = false
kind = CONTINUOUS
```

Required membership results are:

```text
0          -> contained
finite > 0 -> contained
finite < 0 -> not contained
+inf       -> not contained
NaN        -> not contained
```

Membership may be elementwise for array-like input. `EmpiricalSupport` is not
reused because it describes evidence provenance rather than a mathematical
domain.

## 4. Gamma parameterization

The canonical public parameters are `shape` and `scale`. Both must be finite,
real numeric scalars strictly greater than zero. Accepted values are
canonicalized to Python `float`.

The implementation rejects booleans, zero, negative values, NaN, either
infinity, complex values and nonnumeric values.

`loc` is fixed at zero. It is explicit in immutable/derived parameterization
metadata, is not configurable and is not duplicated in `GammaParameters`.
No rate alias is exposed.

The backend mapping is exactly:

```python
scipy.stats.gamma(a=shape, loc=0, scale=scale)
```

## 5. Exponential parameterization

The canonical public parameter is `scale`. It must be a finite, real numeric
scalar strictly greater than zero and is canonicalized to Python `float`.
The same invalid-input rules as Gamma apply.

`loc` is fixed at zero, explicit in parameterization metadata and not
configurable. No rate alias is exposed.

The backend mapping is exactly:

```python
scipy.stats.expon(loc=0, scale=scale)
```

## 6. Probability API

`ParameterizedDistribution` provides:

```text
cdf
logcdf
sf
logsf
ppf
rvs
```

`ParameterizedContinuousDistribution` additionally provides:

```text
pdf
logpdf
```

Continuous objects expose no artificial `pmf` or `logpmf`. Probability
operations are available on bound objects, not unparameterized families.

## 7. Query inputs and return values

`pdf`, `logpdf`, `cdf`, `logcdf`, `sf` and `logsf` accept Python numeric
scalars, NumPy numeric scalars, numeric array-like inputs and ndarrays of any
dimensionality.

Inputs must be non-empty, numeric, real and entirely finite. NaN, either
infinity and nonnumeric values fail closed. A finite point outside support is a
valid evaluation point and is passed to the backend without clipping.
Mathematically valid backend outputs such as negative infinity from a
log-density remain valid.

Return normalization is:

```text
scalar input     -> Python float
array-like input -> numpy.ndarray, dtype float64, same shape
```

pandas inputs, when supplied, are treated as array-like; labels and container
semantics are not preserved. A multidimensional grid is an evaluation grid,
not a multidimensional statistical sample.

## 8. Quantile contract

`ppf(q)` accepts numeric scalar or numeric array-like input. Every probability
must be finite and satisfy `0 <= q <= 1`. Empty, nonnumeric, non-finite or
out-of-range inputs fail closed.

Scalar and array return normalization follows the probability contract.
Infinite outputs are permitted where mathematically required:

```text
Gamma/Exponential ppf(0) = 0
Gamma/Exponential ppf(1) = +inf
```

## 9. Random sampling contract

The public API is `rvs(..., rng=...)`. `rng` is mandatory and SciPy's
`random_state` name is not exposed publicly.

Accepted caller forms are:

1. a non-negative Python or NumPy integer seed, excluding booleans;
2. a `numpy.random.Generator`.

`None`, negative integers, `RandomState`, `SeedSequence`, `BitGenerator` and
other objects are rejected. No implicit process-global RNG is used.

An integer seed creates a local reproducible `Generator`. Repeating the same
integer seed with the same parameters and size produces identical output under
the same supported backend. A caller-owned `Generator` is passed directly and
is expected to advance. This intentionally differs from specialized bootstrap
factories that clone generator state.

Internally the validated generator is passed through SciPy's supported
`random_state=` interface.

Supported sizes are `None`, a non-negative integer, or a tuple of non-negative
integers. Booleans and negative dimensions are rejected; zero is allowed.

Return normalization is:

```text
size=None      -> Python float
size=int/tuple -> numpy.ndarray, dtype float64, requested shape
```

## 10. Backend provenance

Backend metadata identifies:

```text
library              = scipy
Gamma distribution   = scipy.stats.gamma
Exponential backend  = scipy.stats.expon
backend version       = derived from installed scipy
```

This metadata is immutable or derived. It does not serialize an independent
copy of the bound canonical parameters.

## 11. Tests and acceptance

New CP02 tests cover family statelessness, immutable/canonical parameter
objects, parameter boundaries, support membership, exact SciPy mapping,
probability and quantile parity, scalar/array normalization, multidimensional
shape preservation, finite points outside support, fail-closed inputs, RNG
reproducibility and isolation, interface separation, and legacy exports.

The three existing distribution regression files remain frozen and must still
yield 49 passing tests. The registry validator, new CP02 tests, complete
distribution-related surface, `git diff --check`, changed-path audit and
prohibited-track audit must pass before handoff.

## 12. Explicit non-goals

CP02 does not implement or modify:

- fitting, `fit()`, `FitResult` or `FittedDistribution`;
- MLE, method of moments or parameter uncertainty;
- GOF, KS, AD, CvM, Pearson calibration or bootstrap GOF;
- AIC/BIC comparison, automatic family detection or model ranking;
- `MethodSelector` or inference routing;
- `NegativeBinomialFamily`, `BinomialFamily` or `PoissonFamily`;
- any Wave-2 family;
- legacy class migration;
- ANOVA, proportion-CI, empirical-likelihood, sampling-robustness, B3/UAT1 or
  Gate-2 policy.

Passing deterministic backend-oracle tests is software evidence only and does
not establish calibration.

## Revision condition

Return to Architecture before changing stateless family ownership, canonical
parameters, fixed `loc=0`, support semantics, public probability/RNG contracts,
SciPy mappings, future fitted-state ownership, explicit non-goals, or any
unresolved item in `DEC-010`.
