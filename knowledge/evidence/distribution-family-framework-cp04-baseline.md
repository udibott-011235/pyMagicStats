# EV-011 — CP04 Wave 1 fitting architecture baseline

- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP04 — IN_PROGRESS`
- **Estado:** `accepted`
- **Arquitectura:** `FROZEN` mediante `DEC-013`
- **Implementación:** `PENDING`
- **Fecha:** 2026-09-08
- **Repositorio:** `udibott-011235/pyMagicStats`
- **Baseline:** `origin/main` @ `b3f35d4d7b221c457e2e730bfba2b104e1d07144`
- **Rama:** `feature/distribution-family-framework-cp04-wave1-fitting`
- **Contrato:** `DEC-013`
- **Rol ejecutor:** `implementation-engineering` (Cortex), materialización de arquitectura únicamente

## Claim

This record fixes the clean pre-implementation baseline for CP04 and records
the fitting architecture frozen in `DEC-013`. It demonstrates CP03 canonical
closure, the current absence of fitted/result production concepts, the
existing family contracts and the unchanged regression surface. It does not
claim that CP04 production code or tests exist.

## Exact identity and CP03 integration

The isolated checkout was created from the exact fetched canonical snapshot:

```text
origin/main = b3f35d4d7b221c457e2e730bfba2b104e1d07144
branch parent = b3f35d4d7b221c457e2e730bfba2b104e1d07144
branch = feature/distribution-family-framework-cp04-wave1-fitting
opening worktree = clean
CP03 = COMPLETE
CP04 start = architecture/governance only
```

CP03's post-merge governance lifecycle is:

```text
PR = #11
PR head = 7e38c1c62f69771282398ffd3fac118f866c5d69
merge commit = b3f35d4d7b221c457e2e730bfba2b104e1d07144
parent 1 = 28b57a2eaab0706c5b2e2dcdf6a03e5a30a0b649
parent 2 = 7e38c1c62f69771282398ffd3fac118f866c5d69
merge tree = 2e363bd357a5ec59c17690bd9e0e3bbb26061d33
integrated-head tree = 2e363bd357a5ec59c17690bd9e0e3bbb26061d33
TREE_EQUIVALENCE = PASS
```

## Current production reconnaissance

Read-only inspection at the baseline found:

```text
GammaFamily.bind(shape, scale) = PRESENT
ExponentialFamily.bind(scale) = PRESENT
NegativeBinomialFamily.bind(r, p) = PRESENT
ParameterizedDistribution = PRESENT
ParameterizedContinuousDistribution = PRESENT
ParameterizedDiscreteDistribution = PRESENT

GammaFamily.fit = ABSENT
ExponentialFamily.fit = ABSENT
NegativeBinomialFamily.fit = ABSENT
FitResult = ABSENT
FittedDistribution = ABSENT
FittedContinuousDistribution = ABSENT
FittedDiscreteDistribution = ABSENT
```

The existing families are stateless descriptors. Their immutable bound objects
contain the sole canonical parameter object and lazily delegate probability
operations to SciPy. Gamma uses canonical positive `shape, scale` with
`loc=0`; Exponential uses positive `scale` with `loc=0`; Negative Binomial
uses finite real `r>0`, `0<p<=1` with `loc=0`. Continuous and discrete query,
PPF and explicit caller-controlled RNG contracts remain those accepted in
`DEC-011` and `DEC-012`.

## Fitting authority and finite-MLE classification

SciPy's continuous fitting contract supports MLE and fixed `loc` through
`floc`, so the future Gamma implementation can delegate its fixed-location
numerical MLE and then enforce pyMagicStats postconditions:
[SciPy 1.18.0 `rv_continuous.fit` documentation](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.stats.rv_continuous.fit.html).

Generic SciPy discrete fitting is not authoritative for pyMagicStats'
generalized Negative Binomial fit. SciPy's generic API respects distribution
integrality metadata. Evidence-only runtime reconnaissance in the recorded
SciPy 1.18.1 environment observed exactly:

```text
stats.nbinom._shape_info(): shape n -> integrality=True
```

This private metadata is reconnaissance evidence only. Production must not
depend on `_shape_info()`, and this observation makes no claim for every
future SciPy version. CP03 froze canonical `r` as any positive finite real;
the custom estimator is therefore authoritative because pyMagicStats itself
guarantees real `r`, independently of backend fitter metadata. See
[SciPy 1.18.0 `fit` documentation](https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/scipy.stats.fit.html)
and the version-pinned
[SciPy 1.18.1 `nbinom` implementation](https://github.com/scipy/scipy/blob/v1.18.1/scipy/stats/_discrete_distns.py).

For mean `m>0`, `DEC-013` therefore profiles
`p_hat(r)=r/(r+m)` and solves the deterministic one-dimensional score on
`r>0`. Before solving, all-zero data is non-identifying; positive data with
population variance at or below the mean has no finite MLE in this parameter
space; and strictly overdispersed data is eligible for a unique finite root.
For integer observations, the classification uses the exact integer identity
`n*sum(x_i**2)-sum(x_i)**2 > n*sum(x_i)`, without a floating validity
tolerance.

The primary existence/uniqueness authority is Simonsen's 1976 result together
with its 1980 correction: [Simonsen (1976)](https://www.tandfonline.com/doi/abs/10.1080/03461238.1976.10405618),
[Simonsen (1980), correction](https://www.tandfonline.com/doi/abs/10.1080/03461238.1980.10408657).
Aragón, Eberly and Eberly (1992) is historical context only because
[Wang (1996)](https://doi.org/10.1016/0167-7152(94)00259-2) identified a major
problem in its proof. The result and reliable computation are reinforced by
[Bandara, Gill and Mitra (2019)](https://doi.org/10.1016/j.spl.2019.01.009)
and [Yang et al. (2026)](https://link.springer.com/article/10.1007/s00362-026-01842-x).
No extended quotation is reproduced here.

## Architecture review remediation

Architecture reviewed the original exact candidate
`f40ed49f3f006eae4f9de03199b2f942ddb4f38c` and returned
`CHANGES_REQUIRED`. This follow-up records:

```text
ARCH-CP04-001 = REMEDIATED
ARCH-CP04-002 = REMEDIATED
ARCH-CP04-003 = REMEDIATED
ARCH-CP04-004 = REMEDIATED
exact-SHA Architecture re-review = PENDING
```

- `ARCH-CP04-001` removes the unsupported separate adversarial-acceptance
  claim for PR #11 while retaining only demonstrated integration facts.
- `ARCH-CP04-002` separates probability-backend delegation from immutable
  estimator/solver provenance and reconciles the `DEC-010` semantic states.
- `ARCH-CP04-003` fixes public exports, exact Gamma invocation, discrete input
  canonicalization and warning handling.
- `ARCH-CP04-004` corrects source authority and pins SciPy references.

The Negative Binomial mathematical profile score and exact overdispersion
criterion are unchanged. This remediation remains pending exact-SHA
Architecture re-review and does not authorize implementation.

## Validation environment

The architecture-materialization environment was:

```text
Python=3.12.14
NumPy=2.5.2
SciPy=1.18.1
pytest=9.1.1
operating_system=Windows-11-10.0.26200-SP0
```

This is the local evidence-materialization environment, not a future
independent adversarial-audit environment.

## Exact baseline and candidate validation

Before documentation edits:

```text
knowledge registry validator = PASS
Knowledge Base = 7 passed, 2 failed
distribution surface = 575 passed, 2 warnings
```

After documentation edits:

```text
knowledge registry validator = PASS
Knowledge Base = 7 passed, 2 failed
KNOWLEDGE_TEST_DIFFERENTIAL = NO_NEW_FAILURES
git diff --check = PASS
```

The two failures are inherited in both Knowledge Base runs:

- `test_registry_has_unique_ids_and_exactly_the_governed_branches`
- `test_lifecycle_decisions_and_gate2_supersession_are_materialized_exactly`

New Knowledge Base failures: none. The two distribution warnings come from the
accepted extreme-Gamma backend NaN guards and are not failures.

The distribution surface command covers exactly:

```text
tests/test_distribution_shape_contract.py
tests/test_distribution_integration.py
tests/test_distribution_gof_remediation.py
tests/test_distribution_family_core.py
tests/test_continuous_distribution_families.py
tests/test_discrete_distribution_families.py
```

## Scope audit and nonclaims

The original architecture materialization changed only its eight authorized
`knowledge/**` paths. This follow-up changes only the six authorized
architecture-remediation records. Production files and test files are
unchanged. It does not implement fitted objects, fitting errors, estimators,
likelihoods, information criteria, GOF, calibration, selection or routing.

```text
CP03 = COMPLETE
CP04 = IN_PROGRESS
CP04 architecture = FROZEN
CP04 implementation = NOT_STARTED
CP05-CP08 = NOT_STARTED
```
