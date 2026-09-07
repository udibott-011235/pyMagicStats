# EV-009 — CP03 discrete core reconnaissance and baseline

- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP03 — IN_PROGRESS`
- **Estado:** `accepted`
- **Reconnaissance:** `COMPLETE`
- **Arquitectura:** `FROZEN`
- **Implementación:** `PENDING`
- **Fecha:** 2026-09-07
- **Repositorio:** `udibott-011235/pyMagicStats`
- **Baseline:** `origin/main` @ `02a65c80c5da10295d6eeef42e691772d0686ca2`
- **Rama:** `feature/distribution-family-framework-cp03-discrete-core`
- **Contrato:** `DEC-012`
- **Rol ejecutor:** `implementation-engineering` (Cortex)

## Claim

This record fixes the pre-implementation state and accepted baseline for CP03.
It demonstrates that Architecture has frozen the discrete-core contract and
that CP03 may proceed only after separate production authorization. It does not
claim that any CP03 production object exists or has been tested.

## Reconnaissance evidence

```text
CURRENT_DISCRETE_LEGACY_SURFACE =
  BinomialDistribution / PoissonDistribution / DiscreteDistributionValidator /
  Pearson GOF legacy helpers

CURRENT_NEW_FAMILY_DISCRETE_SURFACE = NONE before CP03
CURRENT_SUPPORT_CAN_REPRESENT_DISCRETE_LATTICE = NO before CP03
CURRENT_NORMALIZER_DISCRETE_RVS_COMPATIBLE = NO before CP03
NB_R_P_PARAMETERIZATION_VALID = YES
```

The accepted Negative Binomial parameterization uses `r > 0` and
`0 < p <= 1`, including positive non-integer `r` and the degenerate `p=1`
distribution at zero. Its canonical moments are
`r * (1 - p) / p` and `r * (1 - p) / p**2`, mapped to
`scipy.stats.nbinom(n=r, p=p, loc=0)`.

## Frozen regression baseline

The following results are carried forward from accepted CP02 evidence on the
canonical lineage; they were not rerun in this governance-only task:

```text
49 legacy distribution tests = PASS
311 CP02 family tests = PASS
360 combined distribution tests = PASS
knowledge registry validator = PASS
```

These results establish the pre-implementation regression baseline only. They
do not validate future CP03 behavior, Negative Binomial numerical stability or
backend sampling executability.

## Frozen implementation acceptance surface

A future CP03 candidate must demonstrate:

- `SupportKind.DISCRETE` and elementwise contiguous-integer support membership;
- the stateless `DiscreteDistributionFamily` and immutable
  `ParameterizedDiscreteDistribution` object model;
- canonical `NegativeBinomialFamily().bind(r=..., p=...)` behavior;
- exact probability, PPF endpoint and moment contracts from `DEC-012`;
- explicit RNG behavior and `int`/`int64` discrete sample normalization;
- fail-closed validation before backend delegation;
- explicit backend NaN and sampling-range failure classification;
- bounded-time extreme-RVS adversarial probes;
- unchanged CP02 continuous-family and legacy discrete behavior.

## Limitations and nonclaims

CP03 does not yet implement production code or tests. No fitting, `FitResult`,
`FittedDistribution`, GOF integration, selector/routing, parameter estimation,
Geometric, Bernoulli, Hypergeometric, Wave-2 or zero-inflated family is covered.
The architectural compatibility target for future contiguous integer supports
does not constitute their implementation.

Mathematical parameter admissibility remains distinct from backend numerical
resolvability. A future backend NaN must raise `FloatingPointError`, legitimate
infinities must remain valid, and a valid but unsampleable parameterization must
surface as a numerical/backend sampling failure rather than an invalid-parameter
error.
