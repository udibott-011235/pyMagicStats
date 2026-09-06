# UAT1 B3 — Distribution / GOF surface census

- Fecha: `2026-09-06`
- Checkpoint: `MANUAL UAT CHECKPOINT 1 / B3`
- Work item: `B3-CP01 — read-only surface census`
- Baseline inspected: `main@402e4601df460811779b3238c2526ac12f463a67`
- Branch: `audit/uat1-distribution-gof-census`
- Status: `CENSUS_COMPLETE / ACCURACY_CLOSURE_PENDING`
- Production changes: `none`

## Purpose

This checkpoint inventories the distribution/GOF surface that can plausibly enter Manual UAT1. It does not promote any method, alter production code, or freeze the final UAT inventory. B1 proportion-CI remains independently in progress.

## 1. Public package export boundary

`pyMagicStat.distributions.__all__` currently exports only:

```text
Distribution
NormalDistribution
```

The following implemented classes exist in `pyMagicStat.distributions.distributions` but are not exported by the package namespace:

```text
LognormalDistribution
BinomialDistribution
PoissonDistribution
```

`pyMagicStat.__init__` imports the distributions module but does not re-export individual distribution classes.

### Architectural consequence

Before B3/B4 can freeze the UAT baseline, the project must explicitly decide what “public API” means for this checkpoint:

1. strict package exports (`__all__`), or
2. any class importable from the implementation module.

Until that decision, non-exported validators are `candidate_internal_surface`, not automatically UAT-authorized.

## 2. `Distribution` container

### Current behavior

`Distribution` is an immutable/read-only snapshot of one univariate numeric sample and stores sample descriptives:

```text
n
mean
median
std
var
skewness
excess_kurtosis
mode
q1
q3
iqr
min
max
range
```

It preserves defensive-copy semantics and read-only reconstruction through copy/deepcopy/pickle tests.

### Evidence already present

Existing tests cover:

- canonical univariate sample normalization;
- empty/multidimensional/non-numeric rejection;
- defensive immutable snapshot;
- descriptive equivalence to NumPy/SciPy;
- copy/deepcopy/pickle reconstruction;
- integration with assumption/inference layers.

### B3 status

`candidate_for_uat = yes`

Remaining B3 work is an explicit accuracy table over the entire descriptive surface, including boundary/corner inputs and JSON/display contract where relevant. This is not a distribution-family PDF/CDF object; no population distribution parameterization should be inferred from this container.

## 3. `NormalDistribution`

### Actual contract

Despite its name, `NormalDistribution` is currently a normality/shape validator over sample data. It does not expose a fitted Normal population object with PDF/CDF/PPF methods.

Its structured path delegates to `ShapeAssessment` and returns:

- Shapiro evidence;
- D'Agostino evidence;
- skewness/excess kurtosis/departure magnitude;
- Q-Q regression diagnostics.

The legacy `distribution.type["Normal"]` boolean is explicitly compatibility-only and is not consumed as permission/prohibition for parametric inference.

### Evidence already present

Tests establish structured shape assessment, absence of the legacy KS path, separation between exact-normality rejection and material departure, small-sample `not_assessed`, and non-use of the legacy boolean by the inference selector.

### Open B3 questions

- Q-Q regression numerical/oracle accuracy is not yet separately frozen as a UAT claim.
- `assign_weights()` still exists with legacy heuristic weights but is not part of the structured decision path; it should not enter UAT simply because the method exists.
- The UAT inventory must name this capability as a *normality/shape diagnostic*, not as a full Normal distribution implementation.

### B3 status

`candidate_for_uat = yes_with_scope_definition`

## 4. `LognormalDistribution`

### Actual contract

The validator requires strictly positive data and evaluates exact Gaussianity of `log(data)`. The structured result explicitly states that failure to reject Gaussianity of log(data) does not demonstrate that the original population is lognormal.

### Evidence already present

Gate-2 tests cover:

- rejection when log(data) rejects exact Gaussianity;
- fail-to-reject semantics without identity claim;
- `not_assessed` propagation;
- positive-domain validation;
- structured storage separate from legacy mirrors.

### B3 status

`candidate_internal_surface / not package-exported`

If included in UAT1, it requires an explicit export/scope decision first. It should be tested as a lognormality diagnostic, not as PDF/CDF/quantile functionality.

## 5. `BinomialDistribution`

### Actual contract

This is a discrete-data validator with Pearson chi-square GOF plus a separate normal-approximation diagnostic.

Important structural behavior already implemented:

- `n` is a required structural parameter for GOF;
- fixed `n,p` uses zero estimated parameters;
- fixed `n` with estimated `p` accounts for one estimated parameter in df;
- observations outside `[0,n]` are `not_assessed`;
- invalid `n/p` fails explicitly;
- full binomial support is retained before pooling;
- contiguous tail pooling preserves mass;
- insufficient degrees of freedom returns `not_assessed` rather than a fabricated p-value.

### Evidence already present

Gate-2 tests compare df/statistic/p-value structure to independently reconstructed Pearson calculations and explicitly exercise tail anomalies that would be lost by naive low-expected-cell deletion.

### Open B3 questions

- This class is not package-exported.
- `estimate_parameters_moments()` is a separate legacy/helper capability and does not yet have a frozen UAT accuracy claim.
- `evaluate_normal_approximation()` uses the rule `n*p*(1-p) >= 9`; that heuristic must be treated as a separate capability and cannot inherit GOF validation automatically.

### B3 status

`candidate_internal_surface / GOF evidence strong / ancillary methods not yet closed`

## 6. `PoissonDistribution`

### Actual contract

Pearson chi-square GOF estimates lambda from the sample mean, retains an explicit unobserved upper-tail cell, performs contiguous tail pooling, and subtracts one fitted parameter when computing degrees of freedom.

### Evidence already present

Tests cover:

- mass preservation including unobserved upper tail;
- df reconstruction;
- high-lambda tail pooling;
- independent reconstruction of Pearson statistic and chi-square p-value;
- structured decision semantics.

### Open B3 questions

- This class is not package-exported.
- `evaluate_normal_approximation()` uses `lambda >= 9`; this heuristic is not automatically validated by GOF tests.

### B3 status

`candidate_internal_surface / GOF evidence strong / normal-approximation heuristic separate`

## 7. Shared discrete GOF engine

The Gate-2 remediation introduced a shared Pearson GOF path and contiguous tail pooling. Existing tests establish important structural invariants:

- expected and observed mass conservation;
- contiguous pooled cells;
- no silent deletion/renormalization of sparse tails;
- minimum expected count enforcement after pooling;
- parameter-estimation-aware degrees of freedom;
- fail-closed `not_assessed` when valid Pearson df cannot be formed.

This is strong implementation/structural evidence, but B3 still requires a final accuracy matrix against independent mathematical/oracle references over the UAT-included domain.

## 8. Surface that does **not** currently exist

On the inspected baseline there is no general public distribution-family API exposing, as pyMagicStats-owned methods:

```text
pdf
pmf
cdf
sf
ppf / quantile
rvs / sampling
population moments from fitted parameters
fitted support object
```

Therefore B3 must not invent UAT requirements for such functions. SciPy calls used internally by validators are implementation dependencies, not pyMagicStats public distribution APIs.

## 9. Preliminary UAT census classification

| Capability | Exported | Existing evidence | Preliminary UAT classification |
|---|---:|---|---|
| `Distribution` sample container/descriptives | yes | strong contract tests | `INCLUDE_CANDIDATE` |
| `NormalDistribution` structured normality/shape diagnostics | yes | strong shape/integration tests | `INCLUDE_CANDIDATE_WITH_SCOPE` |
| Q-Q regression subdiagnostic | via NormalDistribution | implementation tests indirect | `ACCURACY_CLOSURE_REQUIRED` |
| legacy `type` / `update_type` | compatibility/deprecated | deprecation tests | `EXCLUDE_FROM_POSITIVE_UAT_CLAIMS` |
| `NormalDistribution.assign_weights()` | method exists | legacy heuristic | `EXCLUDE_PENDING_REVIEW` |
| `LognormalDistribution` diagnostic | no package export | Gate-2 tests | `INTERNAL_CANDIDATE` |
| Binomial Pearson GOF | no package export | strong Gate-2 structural/oracle tests | `INTERNAL_CANDIDATE` |
| Poisson Pearson GOF | no package export | strong Gate-2 structural/oracle tests | `INTERNAL_CANDIDATE` |
| Binomial moment estimator | no package export | not frozen as UAT claim | `EXCLUDE_PENDING_REVIEW` |
| Binomial normal-approx rule | no package export | heuristic exists | `EXCLUDE_PENDING_REVIEW` |
| Poisson normal-approx rule | no package export | heuristic exists | `EXCLUDE_PENDING_REVIEW` |
| generic PDF/CDF/PMF/quantile family API | no | not implemented | `NOT_APPLICABLE` |

## 10. B3 next checkpoints

```text
B3-CP01  surface census                       COMPLETE
B3-CP02  define UAT public/export boundary    NEXT
B3-CP03  descriptive + diagnostic oracle map  pending
B3-CP04  discrete GOF accuracy closure        pending
B3-CP05  invalid/boundary/DataFrame cases      pending
B3-CP06  final B3 inclusion/exclusion freeze   pending
```

No production change should occur in B3-CP02. The immediate architecture task is to decide the export/public boundary and freeze exact claims before adding any new tests or modifying APIs.

## 11. Interaction with B1

B1 proportion-CI remains the first open UAT blocker and its heavy CP06-D execution is still running independently. B3 preparation may continue in read-only/documentation mode, but the final Manual UAT1 inventory must not be frozen until B1 and the final B3 closure both complete.
