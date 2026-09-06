# EV-007 — Distribution Family Framework CP01 materialization evidence

- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP01 — Family architecture and contracts`
- **Estado:** `ARCHITECT_ACCEPTED`
- **Candidato aceptado:** `c63b48eafc439de8857207fd21c1b593e38a3187`
- **Fecha:** 2026-09-06
- **Repositorio:** `udibott-011235/pyMagicStats`
- **Baseline:** `origin/main` @ `402e4601df460811779b3238c2526ac12f463a67`
- **Rama:** `feature/distribution-family-framework-cp01`
- **Rol ejecutor:** `implementation-engineering` (Cortex)
- **Producción modificada:** ninguna

## Claim

The Architect-approved CP01 contract has been materialized without changing
production behavior, and its compatibility census matches the distribution
surface present at the exact baseline.

This evidence records Architecture's acceptance of CP01. It does not implement
any family, validate an estimator, calibrate GOF, integrate BR-017 or authorize
a later checkpoint.

## Repository and scope verification

Observed before editing:

```text
origin = git@github.com:udibott-011235/pyMagicStats.git
origin/main = 402e4601df460811779b3238c2526ac12f463a67
merge-base = 402e4601df460811779b3238c2526ac12f463a67
branch = feature/distribution-family-framework-cp01
status = clean
```

The separate `audit/uat1-distribution-gof-census` candidate was not used as a
parent and was not modified.

## Review genealogy

1. `72ecdba1b60d9efb53dfe612cf7ee4beeb3e76e5` was created locally
   under the original no-push CP01 handoff rule.
2. Architecture subsequently authorized that exact candidate to be pushed
   solely so its committed content could be reviewed independently.
3. The push did not authorize a PR, merge, CP02, CP03 or implementation.
4. Architecture accepted the structural materialization and validation
   evidence, retained `72ecdba…` as the first review candidate, and requested
   one follow-up documentation/governance commit.
5. Architecture reviewed and accepted the exact follow-up candidate
   `c63b48eafc439de8857207fd21c1b593e38a3187` as CP01.

## Architect review refinements

The follow-up preserves the original architecture and adds six clarifications:

- shared probability-family operations include log-domain CDF/survival
  operations and caller-controlled reproducible sampling;
- `family.fit(...)` returns `FitResult`, whose canonical
  `fitted_distribution` owns fitted family state without duplication;
- non-integer negative-binomial `r` is a generalized positive
  shape/dispersion parameter, while the classical successes interpretation is
  limited to integer `r`;
- future `BinomialFamily` and `PoissonFamily` cores form a separate
  compatibility track beneath preserved legacy GOF/diagnostic surfaces;
- the branch chronology distinguishes local creation from the later
  review-only push authorization;
- CP05 distinguishes simple and composite GOF nulls without selecting a
  universal method.

## Current import and export boundary

Observed facts:

- `pyMagicStat.distributions.__all__` exports only `Distribution` and
  `NormalDistribution`.
- `LognormalDistribution`, `BinomialDistribution` and `PoissonDistribution`
  exist in `pyMagicStat.distributions.distributions` but are not package-level
  exports.
- `pyMagicStat.__init__` imports the distributions implementation module but
  does not re-export individual classes.
- No current class owns a general public family API containing `pdf`, `pmf`,
  `cdf`, `sf`, `ppf` or random sampling methods.

## Current-surface census

| Class | Current purpose | Imports / exports | Relevant methods | Responsibility category | Existing tests / evidence | Semantic conflict or gap | Future migration |
|---|---|---|---|---|---|---|---|
| `Distribution` | Defensive read-only snapshot of one univariate numeric sample plus descriptives | Exported from `pyMagicStat.distributions`; not individually re-exported at package root | `data`, `kurtosis`, `update_type`, reconstruction and display methods | Sample description, with legacy assessment storage | Shape-contract tests cover descriptives, dimensionality, immutability, copy/deepcopy/pickle and inference-layer input compatibility | `type` and `assessments` retain legacy state beyond pure sample description; finite-value rejection is not enforced by `univariate_sample` | `KEEP_AS_IS` |
| `NormalDistribution` | Exact-normality/shape assessment plus Q-Q regression diagnostic | Exported from `pyMagicStat.distributions` | `validate_data`, `evaluate_normality`, `evaluate_qq`, `assign_weights`, inherited `fit_test` | Shape / approximation diagnostic | Shape and integration tests cover structured assessment, exact-rejection semantics and separation from inference selection | Name resembles a probability-family object, but behavior is diagnostic; legacy weights and Q-Q accuracy are not a new family contract | `MIGRATE_LATER` |
| `LognormalDistribution` | Tests Gaussianity of `log(data)` for strictly positive samples | Implementation-module import only | `validate_data`, `evaluate_normality`, `assign_weights`, `fit_test` | Shape / model diagnostic | GOF-remediation tests cover reject, fail-to-reject and not-assessed result propagation and canonical/legacy isolation | Not a fitted lognormal family and exposes no probability operations; domain validation is implemented but lacks a dedicated non-positive regression test in the current distribution test set | `MIGRATE_LATER` |
| `BinomialDistribution` | Validates non-negative integer observations; performs Pearson GOF, normal-approximation heuristic and moments helper | Implementation-module import only | `validate_data`, `evaluate_goodness_of_fit`, `evaluate_normal_approximation`, `estimate_parameters_moments`, inherited `fit_test` | GOF assessment plus approximation diagnostic and legacy estimation helper | Gate-2 remediation tests cover parameter validation, support, mass, pooling, df, statistic/p-value reconstruction, structured decision and fail-closed states | Multiple future responsibilities are combined; current `n` denotes binomial trials and is unrelated to the reserved negative-binomial `r` naming | `WRAP_NEW_CORE` |
| `PoissonDistribution` | Validates non-negative integer observations; performs Pearson GOF and a normal-approximation heuristic | Implementation-module import only | `validate_data`, `evaluate_goodness_of_fit`, `evaluate_normal_approximation`, inherited `fit_test` | GOF assessment plus approximation diagnostic | Gate-2 remediation tests cover explicit upper tail, mass, pooling, fitted-parameter df and independent Pearson reconstruction | GOF and approximation remain combined; no separate probability-family or fitted-object contract exists | `WRAP_NEW_CORE` |

## Contract-to-code findings

### Observed facts

1. The current API predates the five-way responsibility separation and mixes
   some legacy assessment state with sample containers and validators.
2. Current `*Distribution` names do not consistently mean probability-family
   objects. The new `*Family` convention avoids extending that ambiguity.
3. The shared discrete GOF implementation preserves support mass, pools
   contiguous tails, accounts for estimated parameters and fails closed when
   Pearson degrees of freedom are insufficient.
4. Current sample normalization rejects empty, multidimensional and nonnumeric
   input but does not reject all non-finite observations.
5. The future family, support, fitted-distribution and FitResult abstractions do
   not yet exist.
6. Existing Gate-2 GOF is assessment behavior and cannot be treated as general
   fitted-family GOF calibration.

### Implementation interpretation

No frozen architectural statement is impossible on the inspected baseline.
The semantic collisions are migration constraints, not reasons to alter legacy
classes during CP01. The approved compatibility boundary and `*Family` naming
allow later work to introduce the new core without changing current behavior
in this checkpoint.

## Validation commands and results

Environment used:

```text
Python 3.12.14
NumPy 2.5.2
SciPy 1.18.1
statsmodels 0.14.6
pytest 9.1.1
OS: Windows
```

The interpreter came from an existing local pyMagicStats virtual environment;
module-path verification confirmed that `pyMagicStat` was imported from this
CP01 worktree.

Observed pre-commit results:

```text
python knowledge/tools/validate_registry.py
PASS — Knowledge registry validation passed

python -m pytest -q \
  tests/test_distribution_shape_contract.py \
  tests/test_distribution_integration.py \
  tests/test_distribution_gof_remediation.py
PASS — 49 passed in 7.07s

git diff --check
PASS

changed-path audit including untracked files
PASS — 8 paths; all under knowledge/**

prohibited-track path audit
PASS — zero B3, production, test, experiment, ANOVA, proportion-CI,
       empirical-likelihood or sampling-robustness paths changed

new-line and new-file domain-term scan
PASS — zero forbidden terms introduced

incorrect baseline SHA scan
PASS — zero occurrences
```

Architect-review follow-up validation:

```text
registry validator: PASS
distribution regression set: PASS — 49 passed in 6.64s
git diff --check: PASS
changed-path audit: PASS — 7 paths, all under knowledge/**
prohibited-track audit: PASS — zero prohibited paths changed
new-line domain-term scan: PASS — zero forbidden terms introduced
```

No production evidence is transferred or reinterpreted by these documentation
changes.

The selected distribution regression set is:

```text
tests/test_distribution_shape_contract.py
tests/test_distribution_integration.py
tests/test_distribution_gof_remediation.py
```

## Limitations

- The census is a static and regression-test-backed observation of one SHA.
- It is not statistical calibration of existing or future families.
- Backend oracle accuracy, fitting behavior and GOF calibration remain future
  checkpoint obligations.
- Migration categories are recommendations for Architecture review and do not
  authorize migration.

## Handoff target

Architectural review is `ARCHITECT_ACCEPTED` at
`c63b48eafc439de8857207fd21c1b593e38a3187`. BR-017 integration remains
`PENDING`; no PR, merge, CP02, CP03 or family implementation is authorized.
