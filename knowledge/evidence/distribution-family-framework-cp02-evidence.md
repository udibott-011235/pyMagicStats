# EV-008 — CP02 continuous distribution core implementation evidence

- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP02 — IN_PROGRESS`
- **Estado:** `under_review`
- **Fecha:** 2026-09-06
- **Repositorio:** `udibott-011235/pyMagicStats`
- **Baseline:** `origin/main` @ `ccff392af13d2cb52d1f3888a986ef58be0099e2`
- **Rama:** `feature/distribution-family-framework-cp02-continuous-core`
- **Rol ejecutor:** `implementation-engineering` (Cortex)
- **Contrato:** `DEC-011`

## Claim

CP02 is authorized to implement deterministic continuous probability-family
mechanics for `GammaFamily` and `ExponentialFamily` under the frozen `DEC-011`
contract. This record is opened with the branch and will be completed with the
exact implementation surface, commands, environment, observations and
limitations before the local candidate handoff.

It does not claim estimation validity, GOF calibration, family selection or
authorization for CP03.

## Branch-opening evidence

```text
origin = git@github.com:udibott-011235/pyMagicStats.git
origin/main = ccff392af13d2cb52d1f3888a986ef58be0099e2
branch = feature/distribution-family-framework-cp02-continuous-core
branch base = ccff392af13d2cb52d1f3888a986ef58be0099e2
pre-existing worktree changes = none
registry baseline = PASS
frozen distribution baseline = 49 passed
```

## Authorized implementation surface

- stateless family descriptors and immutable bound distributions;
- reusable mathematical support;
- immutable Gamma and Exponential parameter objects;
- deterministic probability delegation to SciPy;
- explicit caller-controlled RNG;
- additive package exports;
- new CP02 tests and implementation evidence.

## Explicit exclusions

No fitting, estimation, parameter uncertainty, GOF, calibration, model
comparison, automatic selection, inference routing, discrete family, Wave-2
family or legacy migration is included. ANOVA, proportion-CI,
empirical-likelihood, sampling-robustness, B3/UAT1, Gate-2 policy and
`MethodSelector` remain untouched.

## Implementation evidence

Implemented surface:

```text
pyMagicStat/distributions/__init__.py
pyMagicStat/distributions/families/__init__.py
pyMagicStat/distributions/families/_core.py
pyMagicStat/distributions/families/continuous.py
tests/test_distribution_family_core.py
tests/test_continuous_distribution_families.py
```

Observed implementation facts:

- `GammaFamily` and `ExponentialFamily` are slot-only stateless descriptors;
- each `bind()` call creates a separate frozen family-specific parameter object
  inside a frozen `ParameterizedContinuousDistribution`;
- both supports are immutable `[0, +inf)` continuous supports;
- Gamma delegates exactly to `scipy.stats.gamma(a=shape, loc=0, scale=scale)`;
- Exponential delegates exactly to `scipy.stats.expon(loc=0, scale=scale)`;
- scalar results normalize to Python `float`, while array-like results preserve
  shape as `numpy.ndarray` with `float64` dtype;
- finite out-of-support query points are delegated without clipping;
- query/quantile validation rejects empty, nonnumeric and non-finite inputs;
- integer seeds create local NumPy generators; caller-owned generators are
  passed directly and advance; no process-global NumPy RNG state is consumed;
- no fitting, GOF, selection or discrete-family surface was introduced.

Environment:

```text
Python 3.12.14
NumPy 2.5.2
SciPy 1.18.1
pandas 3.0.5
pytest 9.1.1
OS: Windows
```

Validation results:

```text
python -m pytest -q \
  tests/test_distribution_family_core.py \
  tests/test_continuous_distribution_families.py
PASS — 277 passed in 5.05s

python -m pytest -q \
  tests/test_distribution_shape_contract.py \
  tests/test_distribution_integration.py \
  tests/test_distribution_gof_remediation.py
PASS — 49 passed in 5.09s

python -m pytest -q \
  tests/test_distribution_shape_contract.py \
  tests/test_distribution_integration.py \
  tests/test_distribution_gof_remediation.py \
  tests/test_distribution_family_core.py \
  tests/test_continuous_distribution_families.py
PASS — 326 passed in 5.95s

python -m pytest -q
OBSERVED — 564 passed, 3 skipped, 2 failed in 16.79s
```

The two full-suite failures are pre-existing knowledge-test drift outside the
CP02 implementation surface. At the exact baseline, `registry.json` already
contains 18 branch records while `tests/test_knowledge_base.py` asserts 16,
and baseline `BR-001.head_sha_at_decision` is `46f827dd...` while that test
asserts the older `f1725eb...`. CP02 adds `BR-019`, bringing the observed count
to 19, but does not alter the stale test or prior branch records. The registry's
canonical validator passes independently.

The registry validator, final diff check and path audits are recorded in the
handoff after the complete two-commit candidate is assembled. The candidate
SHA is the implementation commit containing this record and is reported in the
handoff; no self-referential SHA field is maintained.

## Limitations

The tests establish deterministic API/backend parity only for the declared
Gamma and Exponential parameterizations in the recorded environment. They do
not validate estimators, fitted-family behavior, GOF, calibration, family
selection, other backends or later families.

The stale knowledge-test assertions remain an out-of-scope repository risk;
they do not fail within the frozen or complete distribution-related surfaces.
