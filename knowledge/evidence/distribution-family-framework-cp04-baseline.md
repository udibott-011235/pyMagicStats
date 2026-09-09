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
[SciPy `rv_continuous.fit` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html).

Generic SciPy discrete fitting is not authoritative for pyMagicStats'
generalized Negative Binomial fit. SciPy's generic API respects distribution
integrality metadata, and SciPy marks the `nbinom` shape named `n` as integral;
CP03 instead froze canonical `r` as any positive finite real. See
[SciPy `fit` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fit.html)
and the authoritative
[SciPy `nbinom` implementation](https://github.com/scipy/scipy/blob/main/scipy/stats/_discrete_distns.py).

For mean `m>0`, `DEC-013` therefore profiles
`p_hat(r)=r/(r+m)` and solves the deterministic one-dimensional score on
`r>0`. Before solving, all-zero data is non-identifying; positive data with
population variance at or below the mean has no finite MLE in this parameter
space; and strictly overdispersed data is eligible for a unique finite root.
For integer observations, the classification uses the exact integer identity
`n*sum(x_i**2)-sum(x_i)**2 > n*sum(x_i)`, without a floating validity
tolerance.

The existence/uniqueness classification is supported by Aragón, Eberly and
Eberly, *Existence and uniqueness of the maximum likelihood estimator for the
two-parameter negative binomial distribution*, Statistics & Probability
Letters 15(5), 1992,
[doi:10.1016/0167-7152(92)90157-Z](https://doi.org/10.1016/0167-7152(92)90157-Z).
No extended quotation is reproduced here.

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

This architecture materialization changes only the eight authorized
`knowledge/**` paths. Production files and test files are unchanged. It does
not implement fitted objects, fitting errors, estimators, likelihoods,
information criteria, GOF, calibration, selection or routing.

```text
CP03 = COMPLETE
CP04 = IN_PROGRESS
CP04 architecture = FROZEN
CP04 implementation = NOT_STARTED
CP05-CP08 = NOT_STARTED
```
