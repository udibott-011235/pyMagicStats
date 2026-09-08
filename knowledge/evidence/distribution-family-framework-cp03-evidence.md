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
