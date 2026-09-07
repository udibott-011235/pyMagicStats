# EV-008 — CP02 continuous distribution core implementation evidence

- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP02 — IN_PROGRESS`
- **Estado:** `accepted`
- **Fecha:** 2026-09-06
- **Repositorio:** `udibott-011235/pyMagicStats`
- **Baseline:** `origin/main` @ `ccff392af13d2cb52d1f3888a986ef58be0099e2`
- **Rama:** `feature/distribution-family-framework-cp02-continuous-core`
- **Rol ejecutor:** `implementation-engineering` (Cortex)
- **Contrato:** `DEC-011`
- **Implementación inicialmente aceptada por Arquitectura:** `874c03c70c028d0ca4966331b6fc91ec35613caa`
- **Remediación vigente aceptada por Arquitectura:** `e9ef63b802a8cb08ea38b32e87b206432c08b120`
- **Primera auditoría adversarial pre-merge:** `ADVERSARIAL_CHANGES_REQUIRED`
- **Reauditoría adversarial pre-merge:** `PENDING`
- **Integración:** `PENDING`

## Claim

Architecture initially accepted exact implementation candidate
`874c03c70c028d0ca4966331b6fc91ec35613caa`. After adversarial findings and
remediation, Architecture accepts exact candidate
`e9ef63b802a8cb08ea38b32e87b206432c08b120` as the current deterministic
continuous probability-family implementation under frozen contract `DEC-011`.
CP02 remains `IN_PROGRESS` because adversarial re-audit and integration are
pending.

It does not claim estimation validity, GOF calibration, family selection or
authorization for CP03.

## Architecture acceptance

Acceptance covers `DistributionFamily`, `ContinuousDistributionFamily`,
`DistributionSupport`, `SupportKind`, `ParameterizedDistribution`,
`ParameterizedContinuousDistribution`, `GammaParameters`,
`ExponentialParameters`, `GammaFamily` and `ExponentialFamily`, together with
the tested `pdf`, `logpdf`, `cdf`, `logcdf`, `sf`, `logsf`, `ppf` and `rvs`
mechanics and the frozen CP02 parameterization and RNG contracts.

It does not cover or imply `fit()`, `FitResult`, a `FittedDistribution`
implementation, parameter estimation or uncertainty, GOF, calibration,
`MethodSelector`, routing, discrete families, CP03, integration or merge.

## Adversarial findings and remediation

The first pre-merge adversarial audit returned
`ADVERSARIAL_CHANGES_REQUIRED` with `FINDING-ADV-CP02-001` (`MINOR`),
`FINDING-ADV-CP02-002` (`INFO`) and `FINDING-ADV-CP02-003` (`INFO`).

Remediation commit `e9ef63b802a8cb08ea38b32e87b206432c08b120` records:

- `ADV-CP02-001` — mixed bool/numeric query coercion is remediated:
  `pdf([0.5, True])` and `ppf([0.5, True])` raise `TypeError`, while
  `support.contains([0.5, True])` returns `[True, False]` elementwise;
- `ADV-CP02-002` — extreme Gamma backend NaN is hardened: the canonical
  positive finite parameter domain is unchanged and no arbitrary threshold is
  introduced; a SciPy NaN becomes an explicit `FloatingPointError`, while
  legitimate infinities remain valid results; a SciPy `RuntimeWarning` may
  occur before the explicit failure;
- `ADV-CP02-003` — the two failing `tests/test_knowledge_base.py` assertions
  remain pre-existing Knowledge Base debt outside CP02.

The historical Architecture acceptance of `874c03c…` is retained; the current
Architecture-accepted implementation candidate is `e9ef63b…`. Adversarial
re-audit remains `PENDING` and this record does not claim `ADVERSARIAL_PASS`.

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
PASS — 311 passed

python -m pytest -q \
  tests/test_distribution_shape_contract.py \
  tests/test_distribution_integration.py \
  tests/test_distribution_gof_remediation.py
PASS — 49 passed

python -m pytest -q \
  tests/test_distribution_shape_contract.py \
  tests/test_distribution_integration.py \
  tests/test_distribution_gof_remediation.py \
  tests/test_distribution_family_core.py \
  tests/test_continuous_distribution_families.py
PASS — 360 passed

python -m pytest -q
OBSERVED — 598 passed, 3 skipped, 2 failed

baseline main@ccff392af13d2cb52d1f3888a986ef58be0099e2
OBSERVED — 287 passed, 3 skipped, 2 failed

FULL_SUITE_DIFFERENTIAL=NO_NEW_FAILURES
```

The two full-suite failures are pre-existing knowledge-test drift outside the
CP02 implementation surface. At the exact baseline, `registry.json` already
contains 18 branch records while `tests/test_knowledge_base.py` asserts 16,
and baseline `BR-001.head_sha_at_decision` is `46f827dd...` while that test
asserts the older `f1725eb...`. CP02 adds `BR-019`, bringing the observed count
to 19, but does not alter the stale test or prior branch records. The registry's
canonical validator passes independently.

The registry validator, final diff check and path audits are recorded in the
handoff for the current accepted remediation candidate. The implementation
candidate is `e9ef63b802a8cb08ea38b32e87b206432c08b120`; no self-referential
governance-commit SHA field is maintained.

## Limitations

The tests establish deterministic API/backend parity only for the declared
Gamma and Exponential parameterizations in the recorded environment. They do
not validate estimators, fitted-family behavior, GOF, calibration, family
selection, other backends or later families.

The stale knowledge-test assertions remain an out-of-scope repository risk;
they do not fail within the frozen or complete distribution-related surfaces.

Object equality and hash semantics across separately instantiated Family
descriptors are not part of the CP02 contractual guarantee and remain an
explicit future architecture decision.

Mathematical parameter admissibility is distinct from backend numerical
resolvability. CP02 does not establish calibrated numerical operating
boundaries for Gamma shape/scale across the entire float64 domain. Backend NaN
is now an explicit failure rather than a returned probability value; this does
not claim that the entire IEEE-754 Gamma domain is numerically stable.
