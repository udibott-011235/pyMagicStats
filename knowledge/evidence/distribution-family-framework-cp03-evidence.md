# EV-010 — CP03 discrete core implementation and adversarial evidence

- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP03 — IN_PROGRESS`
- **Estado:** `accepted`
- **Arquitectura:** `FROZEN` mediante `DEC-012`
- **Implementación auditada:** `4f7fa09bc7ab501d21f6d27dada30ade23397588`
- **Parent:** `da1b9d51e4bfeb7f262db182cf10993f59b1162b`
- **Baseline:** `origin/main` @ `02a65c80c5da10295d6eeef42e691772d0686ca2`
- **Rama:** `feature/distribution-family-framework-cp03-discrete-core`
- **Auditoría independiente:** `ADVERSARIAL_PASS`
- **Integración:** `PENDING`
- **Fecha:** 2026-09-07

## Claim

This record materializes the independently completed pre-merge adversarial
result for the exact CP03 implementation commit
`4f7fa09bc7ab501d21f6d27dada30ade23397588`. The implementation realizes the
discrete-family contract frozen in `DEC-012` without changing the legacy
discrete implementation or introducing mathematical parameter thresholds.

The audit classification is:

```text
ADVERSARIAL_PASS
BLOCKER=0
MAJOR=0
MINOR=0
INFO=1
```

## Validated surfaces

The accepted pre-merge evidence records:

```text
FROZEN_REGRESSION=360 passed
CP03_TESTS=215 passed
DISTRIBUTION_SURFACE=575 passed
```

The validated candidate includes `SupportKind.DISCRETE`, contiguous integer
support membership, immutable Negative Binomial `r,p` parameterization,
SciPy-backed probability operations, canonical PPF endpoints, explicit RNG
ownership, `int`/`int64` discrete sampling normalization and fail-closed
backend-result handling.

## Exact implementation diff

The audited implementation delta
`da1b9d51e4bfeb7f262db182cf10993f59b1162b..4f7fa09bc7ab501d21f6d27dada30ade23397588`
contains exactly:

```text
pyMagicStat/distributions/__init__.py
pyMagicStat/distributions/families/__init__.py
pyMagicStat/distributions/families/_core.py
pyMagicStat/distributions/families/discrete.py
tests/test_discrete_distribution_families.py
```

## Bounded extreme-RVS probe matrix

Antigravity returned the following complete 16-probe matrix for the exact
implementation SHA. Every call used `size=5`, `rng=42` and its own external
10-second timeout. These observations record backend executability only; they
do not define mathematical parameter thresholds.

| r | p | size | rng | External timeout | Classification | Observed value or exception |
|---:|---:|---:|---:|---:|---|---|
| `1e-10` | `0.5` | 5 | 42 | 10 seconds | `SUCCESS` | `[0, 0, 0, 0, 0]` |
| `1e-50` | `0.5` | 5 | 42 | 10 seconds | `SUCCESS` | `[0, 0, 0, 0, 0]` |
| `1e-300` | `0.5` | 5 | 42 | 10 seconds | `SUCCESS` | `[0, 0, 0, 0, 0]` |
| `1e6` | `0.5` | 5 | 42 | 10 seconds | `SUCCESS` | `[1001531, 998847, 999612, 999906, 1001436]` |
| `1e9` | `0.5` | 5 | 42 | 10 seconds | `SUCCESS` | `[1000048402, 999963539, 999987747, 999997053, 1000045425]` |
| `1e15` | `0.5` | 5 | 42 | 10 seconds | `SUCCESS` | `[1000000048401643, 999999963539159, 999999987747424, 999999997053871, 1000000045425116]` |
| `1e50` | `0.5` | 5 | 42 | 10 seconds | `NUMERICAL_FAILURE` | `FloatingPointError: backend numerical failure: rvs could not generate samples` |
| `1e300` | `0.5` | 5 | 42 | 10 seconds | `NUMERICAL_FAILURE` | `FloatingPointError: backend numerical failure: rvs could not generate samples` |
| `2` | `1e-5` | 5 | 42 | 10 seconds | `SUCCESS` | `[209740, 183160, 308153, 234672, 308175]` |
| `2` | `1e-8` | 5 | 42 | 10 seconds | `SUCCESS` | `[209199455, 183703860, 307932808, 234602857, 307755162]` |
| `2` | `1e-15` | 5 | 42 | 10 seconds | `SUCCESS` | `[2091817326567147, 1837215524346474, 3079255352568835, 2346005086408733, 3077413635282453]` |
| `2` | `1e-18` | 5 | 42 | 10 seconds | `NUMERICAL_FAILURE` | `FloatingPointError: backend numerical failure: rvs could not generate samples` |
| `2` | `1e-25` | 5 | 42 | 10 seconds | `NUMERICAL_FAILURE` | `FloatingPointError: backend numerical failure: rvs could not generate samples` |
| `2` | `1e-50` | 5 | 42 | 10 seconds | `NUMERICAL_FAILURE` | `FloatingPointError: backend numerical failure: rvs could not generate samples` |
| `2` | `1e-300` | 5 | 42 | 10 seconds | `NUMERICAL_FAILURE` | `FloatingPointError: backend numerical failure: rvs could not generate samples` |
| `2` | `1` | 5 | 42 | 10 seconds | `SUCCESS` | `[0, 0, 0, 0, 0]` |

```text
SUCCESS=10
NUMERICAL_FAILURE=6
BACKEND_RANGE_FAILURE=0
TIMEOUT=0
```

The probes were not rerun while materializing this evidence.

## INFO-001 — bounded SciPy backend executability limitation

The bounded adversarial probes confirm that mathematical admissibility of
`r > 0` and `0 < p <= 1` does not guarantee that every extreme sampling request
is executable by the SciPy backend. This is an informational backend
executability limit, not a defect in the canonical Negative Binomial parameter
domain.

After public parameters, RNG and size pass validation, backend numerical or
integer-range failures are translated to `FloatingPointError` with stable
`backend numerical failure: rvs` semantics and the original exception chained.
No arbitrary mathematical threshold for `r` or `p` is introduced.

## Knowledge Base validation traceability

```text
PARENT_KNOWLEDGE_TESTS=7 passed, 2 failed
CANDIDATE_KNOWLEDGE_TESTS=7 passed, 2 failed
KNOWLEDGE_TEST_DIFFERENTIAL=NO_NEW_FAILURES
```

The exact inherited failures in both runs are:

- `test_registry_has_unique_ids_and_exactly_the_governed_branches`
- `test_lifecycle_decisions_and_gate2_supersession_are_materialized_exactly`

The full repository suite was not rerun for this documentation-only evidence
materialization.

## Evidence-materialization environment

The environment used to validate this evidence record was:

```text
Python=3.12.14
NumPy=2.5.2
SciPy=1.18.1
pytest=9.1.1
operating_system=Windows-11-10.0.26200-SP0
```

This is the evidence-materialization environment. It is not asserted to be the
independent Antigravity audit environment.

## Scope and isolation

The implementation is limited to the additive family core, discrete family
module, package exports and focused CP03 tests. It does not route the legacy
`BinomialDistribution`, `PoissonDistribution`,
`DiscreteDistributionValidator` or Pearson GOF helpers through the new family
API. It adds no fitting, GOF integration, selector/routing, alternative
Negative Binomial parameterization or additional discrete family.

## Status and nonclaims

```text
INTEGRATION=PENDING
CP03_OVERALL=IN_PROGRESS
CP04_CP08=NOT_STARTED
```

`ADVERSARIAL_PASS` makes the exact implementation and this evidence eligible
for Architecture review. It does not authorize or claim PR creation, merge,
canonical integration, CP03 completion, CP04 start or completion of the overall
stage.
