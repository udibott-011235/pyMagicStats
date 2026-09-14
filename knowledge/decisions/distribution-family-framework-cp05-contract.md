# DEC-014 — CP05 fitted-family GOF calibration contract and preregistration

- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP05-C0 — HOLDOUT_COMMITMENT_DEPOSITED / EXECUTION_NOT_STARTED`
- **Estado del registro:** `accepted`
- **Estado de arquitectura:** `ACCEPTED`
- **Estado de implementación:** `CP05-B COMPLETE; CP05-C AUTHORIZED_TO_START / EXECUTION_NOT_STARTED`
- **Fecha:** 2026-09-12
- **Baseline CP05-B:** `main` @ `5eb179be578594aa900a29bf5ae2f5540e05ffa2`
- **Rama CP05-B integrada:** `feature/distribution-family-framework-cp05-b-harness`
- **Baseline CP05-C0:** `main` @ `c8df1bdab55aabf10e048e31aed61fd0d09cb5f6`
- **Rama CP05-C0:** `docs/cp05-c-holdout-commitment`
- **Owner de arquitectura:** `statistical-software-architecture`
- **Diseño matemático:** ChatGPT / Arquitectura; borrador no canónico identificado en EV-013
- **Rematerialización documental y tests:** Cortex / Implementación
- **Implementación CP05-B:** `implementation-engineering`
- **QA CP05-B:** `adversarial-statistical-qa`
- **Evidencia de apertura:** `EV-013`
- **Extiende:** `DEC-010`, `DEC-013`
- **Supersedes:** ninguno

## Post-merge governance projection de CP05-A — 2026-09-13 (histórico)

The statuses above project the state effective on integration of this closure.
PR #14 already integrated the audited documentary candidate `2caf234cf1bfa8c66dd0317986803ff443ca3194`
through `main@3d9db61cf7414ce7fe3d94819b5f9e005fff527f`.
EV-013 records independent bundle/publication audits and integration evidence.
CP05-A architecture is accepted, integration complete and governance closed
under this projection. CP05 overall remains IN_PROGRESS; CP05-B–D remain
NOT_STARTED. No executable GOF, numerical precision, RNG invariance,
performance, empirical error control, power or production validity is certified.
The mathematical contract below is unchanged.

## 1. Purpose and checkpoint split

CP05 defines how goodness-of-fit (GOF) is assessed for the three Wave 1
families after the fitting contract accepted in CP04. It resolves the CP01
reservation about fitted-family calibration only for `GammaFamily`,
`ExponentialFamily` and `NegativeBinomialFamily` with the CP04 estimators and
fixed `loc=0` contracts.

CP05 is divided into four independently authorized checkpoints:

| Checkpoint | State at this decision | Required result |
|---|---|---|
| `CP05-A` | `COMPLETE` | Reviewed architecture, statistical-risk contract and preregistration |
| `CP05-B` | `COMPLETE` | Reproducible research harness and independent software oracles; software-correctness scope only |
| `CP05-C` | `AUTHORIZED_TO_START` | Commitment gate satisfied; exploratory calibration, power characterization and performance evaluation not started |
| `CP05-D` | `NOT_STARTED` | Sealed confirmatory holdout and independent adversarial audit |

CP05-A is documentation and governance only. It does not authorize production
code, a research harness, simulation, calibration, publication, PR or merge.
CP05 is not complete until all four checkpoints pass.

## CP05-B post-merge governance closure — 2026-09-13

PR #16 integrated the independently audited candidate
`75529e4415558c1abef6166432ebbafafd00a812` into
`main@9fac41a38ed6583356b0e305a856dca7a3096530`. The merge tree is
`71f7b72fedee0fd4dbcdd1e41f208d53056fdb2f`. `FINDING-CP05B-001` is CLOSED
after remediation and independent reaudit; the technical verdict is PASS.

The CP05-B integration, governance and overall checkpoint are COMPLETE/CLOSED.
The claim scope is strictly `SOFTWARE_CORRECTNESS_ONLY`: calibration, type-I
control, power, method selection and production suitability remain unvalidated.
CP05 remains IN_PROGRESS; CP05-C, CP05-D and CP06–CP08 are NOT_STARTED. No
CP05-C or CP05-D execution occurred, no holdout was accessed, and no holdout
secret was generated. The mathematical contract below is unchanged.

The repository branch/ruleset bypass required for PR #16 remains procedural
debt rather than statistical debt. Restricted Windows pytest environments may
require an explicit `--basetemp`.

## CP05-C0 holdout commitment gate — 2026-09-13

The Project Owner deposited only the public SHA-256 commitment below. The
plaintext CP05-D namespace remains stored off-repository, was not disclosed to
or accessed by Cortex, and is not represented anywhere in this repository.

```text
CP05_D_NAMESPACE_COMMITMENT_SHA256=0d15aa19ff174fba06e3b06817e288b78e6168d4a775061cd4766e94c3c1896b
COMMITMENT_STATUS=DEPOSITED
HOLDOUT_COMMITMENT=DEPOSITED
SECRET_STORED_OFF_REPO=YES
SECRET_DISCLOSED=NO
HOLDOUT_SECRET_ACCESSED=NO
HOLDOUT_SECRET_DISCLOSED=NO
HOLDOUT_EXECUTED=NO
CP05_C=AUTHORIZED_TO_START
CP05_C_EXECUTION=NOT_STARTED
CP05_D=NOT_STARTED
CP05_OVERALL=IN_PROGRESS
```

This satisfies only the precondition gate required before CP05-C. It does not
execute or authorize any particular experiment command, calibration, power
analysis, method selection or CP05-D activity. The mathematical contract below
is unchanged.

## 2. Estimand, population, design and experimental unit

The primary calibration estimand for each preregistered cell is the rejection
probability of one fully specified GOF procedure under its stated null:

\[
\pi_c = P_c\{p_{MC} \le 0.05\}.
\]

The population is the family/parameter/sample-size/null-type combination in
that cell. The design is Monte Carlo sampling from that population. The
experimental unit is one independently generated dataset together with the
complete assessment performed on it. Inner bootstrap replicates are not
independent outer experimental units and must never be used as the Wilson
denominator.

For composite Negative Binomial cells, the primary estimand is explicitly
conditional on the observed dataset admitting the finite CP04 MLE. The
unconditional assessment probability is a separate applicability estimand;
it must be reported and must not be substituted for conditional type-I error.

## 3. Null contracts

Two null types are distinct and must be carried in every artifact and result:

### 3.1 Simple known-parameter null

\[
H_0^S: X_1,\ldots,X_n \overset{iid}{\sim} F(\theta_0),
\]

where `theta0` is fixed independently of the observed sample. Neither the
observed dataset nor a Monte Carlo replicate is fitted. This path is required
as an oracle and reference case; it is not evidence that a simple-null
reference distribution is valid after fitting.

### 3.2 Composite fitted-parameter null

\[
H_0^C: X_1,\ldots,X_n \overset{iid}{\sim} F(\theta),
\qquad \theta\text{ unknown}.
\]

The observed sample is fitted with the exact CP04 estimator. Parametric
bootstrap samples are generated from that fitted distribution, and every
eligible bootstrap sample is refitted with the same estimator before its test
statistic is computed. Treating the observed estimate as known in the
replicates is forbidden because it changes the null and can produce a
conservative, low-power assessment.

## 4. Candidate statistics; no default selected in CP05-A

CP05-A preregisters candidates but selects no universal or family-specific
production default.

| Family class | Primary candidates | Comparators only |
|---|---|---|
| Gamma, Exponential | Anderson–Darling (`AD`), Cramér–von Mises (`CVM`) | Kolmogorov–Smirnov (`KS`) |
| Negative Binomial | discrete support-weighted `AD`, discrete support-weighted `CVM` | discrete `KS`; existing Pearson Gate-2 helper |

Only the primary candidates are eligible for promotion. Comparators are run
for context and regression evidence but cannot become the selected CP05
procedure without a new architecture decision and preregistration.

For ordered continuous observations and fitted CDF values
`u[i] = F(x[i]; theta)`:

\[
W^2 = \frac{1}{12n} + \sum_{i=1}^{n}
\left(u_i-\frac{2i-1}{2n}\right)^2,
\]

\[
A^2 = -n - \frac{1}{n}\sum_{i=1}^{n}(2i-1)
\left[\log u_i + \log(1-u_{n+1-i})\right].
\]

The implementation must use stable log-CDF/log-survival operations where
available. Silent clipping that changes the statistic is forbidden.

For Negative Binomial support points `j = 0, 1, ...`, let `p_j` be fitted PMF,
`H_j` fitted CDF, `S_j` the number of observations not exceeding `j`, and
`Z_j = S_j - n H_j`. The preregistered support-weighted candidates are

\[
W_d^2 = \frac{1}{n}\sum_{j\ge0} Z_j^2 p_j,
\qquad
A_d^2 = \frac{1}{n}\sum_{j\ge0}
\frac{Z_j^2 p_j}{H_j(1-H_j)}.
\]

CP05-B must implement these infinite-support sums with a documented numerical
remainder certificate and compare them with an independent higher-precision
oracle. The omitted positive tail contribution must be bounded by
`max(1e-14, 1e-12 * abs(partial_sum))`; the completed float64 statistic must
agree with the independent oracle within
`abs(error) <= 1e-11 * max(1, abs(T_oracle))`. A truncation without a bound,
randomized jitter, continuous critical values or silent tie breaking is
forbidden. Failure to certify this accuracy produces `FAILED`, not an
approximate statistic.

### 4.1 Stable upper tail of discrete Anderson–Darling

This clarification is required by the Owner for rematerialization. It preserves
the statistic, the remainder certificate and the oracle tolerance above.

Define `p_j = PMF(j)`, `H_j = CDF(j)`, `q_j = SF(j) = 1 - H_j`,
`S_j = number of observations <= j` and `Z_j = S_j - n H_j`.
The equality defining survival is mathematical, not an instruction to subtract
a rounded CDF. For every `j >= max(x_i)`, including equality:

\[
S_j=n,\qquad Z_j=nq_j.
\]

The summand before the exterior factor `1/n` is

\[
\frac{Z_j^2p_j}{H_j(1-H_j)}
=n^2\frac{q_jp_j}{H_j}.
\]

The complete contribution to `A_d^2`, including `1/n`, is

\[
n\frac{q_jp_j}{H_j}.
\]

CP05-B must use `sf/logsf`, `logpmf` and `logcdf`; it must not compute
`1-cdf` or directly evaluate the original quotient when `H_j` rounds to
`1.0`. The complete contribution can be evaluated from

\[
\log(n)+\log q_j+\log p_j-\log H_j.
\]

Equivalently, using backend log functions, evaluate the complete term from
`log(n) + logsf(j) + logpmf(j) - logcdf(j)`.

CP05-B must retain the established remainder certificate and compare this
stable branch against the independent higher-precision oracle. Tests must
explicitly verify algebraic equivalence and the boundary `j = max(x_i)`.
If accuracy cannot be certified, evaluation must fail explicitly; silent
clipping is forbidden. No executable statistic or harness is introduced here.

The Pearson helper is a historical comparator only. Gate-2 evidence and
thresholds do not transfer to this fitted-family assessment.

## 5. Monte Carlo and parametric-bootstrap contract

For a statistic where larger values are less compatible with the null, the
Monte Carlo p-value is exactly

\[
p_{MC}=\frac{b+1}{B+1},\qquad
b=\sum_{k=1}^{B} I(T_k^* \ge T_{obs}).
\]

The plus-one correction and the `>=` tie rule are mandatory. A reported zero
p-value is impossible. Rejection means `p_MC <= alpha`; the nominal level is
fixed at `alpha = 0.05`.

The research harness must support `B in {199, 999}` as preregistered candidate
configurations. Monte Carlo discreteness is part of each assessed procedure;
results from different `B` values are different cells and must not be pooled.
CP05-C may promote one `B` per family/statistic only through the selection rule
in section 10. No other `B` can enter CP05-D without a new preregistration.

### 5.1 Composite refitting

The observed fit and every replicate fit must retain CP04 estimator, solver,
backend and warning provenance. A mock, method-of-moments substitute,
warm-start that changes the solution, or fit using the generating parameter is
not an admissible refit.

### 5.2 Negative Binomial eligibility

The CP04 finite-MLE distinction is binding:

- an all-zero observed sample is non-identifying;
- a positive sample with population variance no greater than its mean has no
  finite MLE in the accepted real-`r` parameterization;
- only a sample with a successful finite CP04 MLE is GOF-eligible.

An ineligible observed dataset returns `NOT_ASSESSED`; it is not a
non-rejection. In composite Negative Binomial bootstrap, mathematically
ineligible replicate samples may be rejected and redrawn until exactly `B`
eligible refits are obtained, with a hard cap of `100 * B` attempted draws.
All attempts and ineligibility reasons are counted. Reaching the cap returns
`NOT_ASSESSED` for the whole assessment and no p-value.

A numerical/backend failure is not mathematical ineligibility. It terminates
the assessment as `FAILED` and must never be silently redrawn. For Gamma and
Exponential, any replicate-fit or statistic-evaluation failure likewise
terminates the assessment as `FAILED`.

## 6. Reproducibility and random-state ownership

No process-global RNG, order-dependent worker stream or undocumented retry is
permitted. CP05-B must derive one independent seed for each stochastic unit as

```text
SHA256(namespace || canonical_cell_id || raw_outer_index || purpose || raw_inner_index)
```

using UTF-8 strings, NUL separators and unsigned big-endian conversion of the
first 128 digest bits into a NumPy `SeedSequence` entropy integer. The public
development namespace is exactly:

```text
pyMagicStats/STAGE-DIST-FAMILIES-001/CP05-C/v1
```

This makes results invariant to process count, shard count, batch size and
execution order. `raw_outer_index` counts every generated outer dataset,
including ineligible Negative Binomial draws; `eligible_outer_index` is a
separate monotonically increasing label assigned only after eligibility is
known. Likewise `raw_inner_index` counts every bootstrap attempt, including
an ineligible redraw. A redraw must not reuse a seed or renumber later raw
attempts. Re-running the same manifest must reproduce raw counts, eligibility
mapping and digests bit-for-bit in the frozen environment.

The CP05-D namespace remains secret. Before CP05-C begins, the decision owner
must deposit its SHA-256 commitment in the accepted preregistration record.
The plaintext namespace is revealed to the independent CP05-D executor only
after code, dependency lock, selected method, selected `B`, cell manifest and
analysis command are frozen. At CP05-A materialization the commitment was
`PENDING_OWNER` (historical state). CP05-C0 has now deposited the public
commitment recorded above without disclosing or accessing the plaintext.
CP05-D remains `NOT_STARTED` and cannot start before the remaining freeze and
authorization requirements are satisfied.

## 7. Calibration cells

A cell key is the ordered tuple

```text
(phase, null_type, family, statistic, n, canonical_parameters, B)
```

Canonical JSON uses sorted keys, decimal strings without binary-float repr
variation, and the family parameter names frozen in DEC-010/DEC-013.

### 7.1 Mandatory development matrix (`CP05-C`)

- `n in {20, 50, 100, 250}`.
- Gamma: `shape in {0.25, 0.5, 1, 2, 10}`, `scale = 1`.
- Exponential: `scale = 1`.
- Negative Binomial: `r in {0.25, 1, 5, 20}` crossed with
  `p in {0.1, 0.5, 0.9}`.
- Both simple and composite null contracts.
- Every primary candidate and comparator in section 4, and both preregistered
  `B` values.
- `R_C = 2,000` eligible outer units per cell.

For composite Negative Binomial cells, outer draws continue until `R_C`
eligible observed fits are obtained or `100 * R_C` total draws are attempted.
The raw-draw denominator, eligibility count and reason histogram are mandatory
outputs. Other families do not replace failed units; any failure blocks that
cell.

### 7.2 Mandatory confirmatory matrix (`CP05-D`)

The primary holdout uses the same `n` and parameter matrix with fresh sealed
randomness, composite nulls only, the selected statistic and selected `B`, and
`R_D = 5,000` eligible outer units per cell. Simple-null holdout cells are
software controls and cannot substitute for composite-null cells.

The adversarial holdout adds these off-grid cells:

- Gamma: `n in {30, 75, 150}`, `shape in {0.35, 3, 30}`, and
  `scale in {1e-6, 1e6}`.
- Exponential: `n in {30, 75, 150}` and `scale in {1e-6, 1e6}`.
- Negative Binomial: `n in {30, 75, 150}`, `r in {0.5, 2, 10, 50}` and
  `p in {0.25, 0.75, 0.97}`.

The same Negative Binomial eligibility and attempt rules apply. Resource
pressure may pause the checkpoint as `BLOCKED`; it may not reduce `R_D`, drop
cells or change parameters after unblinding.

## 8. Primary risk criterion

Each cell records `x` rejections among `R` eligible outer units. Let
`z = 1.959963984540054`. The two-sided 95% Wilson upper endpoint is

\[
U_W = \frac{\hat p + z^2/(2R) +
z\sqrt{\hat p(1-\hat p)/R + z^2/(4R^2)}}{1+z^2/R},
\qquad \hat p=x/R.
\]

The owner-approved liberal bound is binding:

```text
CELL_ACCEPTED iff Wilson95_upper(type_I_error) <= 0.065
```

No rounding is applied before comparison. The rule is cell-wise; there is no
pooling, averaging, tolerance budget or post-hoc cell removal. Promotion
requires every mandatory primary holdout cell to pass. Confidence intervals
are monitoring bounds, not a claim of familywise 95% simultaneous coverage.

Applicability, numerical-failure rate and inner-retry burden are reported with
their own Wilson intervals. They are not folded into the type-I numerator or
used to make an ineligible sample look conservative.

## 9. CP05-B software-oracle gates

Before any calibration claim, the harness must pass deterministic tests for:

1. hand-calculated continuous `AD`, `CVM` and `KS` fixtures;
2. independent arbitrary-precision Negative Binomial support-sum fixtures,
   including a certified remainder bound;
3. simple-null comparisons against an external authoritative implementation
   where contracts coincide;
4. trace proof that composite observed and replicate data are refitted exactly
   once per eligible sample with CP04 estimators;
5. plus-one p-value, `>=` ties and boundary cases `b in {0, B}`;
6. separation of mathematical ineligibility, numerical failure and cap
   exhaustion;
7. seed derivation and invariance across serial, sharded and reordered runs;
8. manifest/schema validation, atomic checkpoints and resume equivalence;
9. environment, source SHA, dependency and raw-output digest capture;
10. regression isolation from production and all CP04 tests.

These are software-correctness gates only. Passing them is not evidence of
type-I calibration, power or production suitability.

## 10. Exploratory selection rule (`CP05-C`)

Selection occurs separately for the continuous and Negative Binomial tracks.
Before the full matrix, CP05-C runs a non-claiming performance preflight with
`R_PREFLIGHT = 20` on Gamma `(n=20, shape=0.25, scale=1)`, Exponential
`(n=20, scale=1)`, and Negative Binomial
`(n=20, r=0.25, p=0.1)` and `(n=20, r=20, p=0.9)`, for both `B` values and
primary candidates. It records median/p95 wall time, peak memory, eligibility
and projected full-matrix cost. If the projection exceeds the separately
authorized resource budget, CP05-C becomes `BLOCKED_RESOURCE`; the matrix is
not downsampled and no calibration claim is made.

Within a track, a primary candidate is eligible only if:

- all CP05-B gates pass;
- all mandatory development cells complete without unaccounted failures;
- every matched primary-candidate development cell satisfies the section 8
  Wilson rule;
- its median and 95th-percentile per-assessment runtime and peak-memory metrics
  are recorded in the frozen environment.

Among eligible candidates, maximize the minimum preregistered alternative
rejection rate across the track; break ties within `0.01` absolute power by
smaller 95th-percentile runtime, then smaller peak memory, then the fixed order
`AD`, `CVM`. Comparator outcomes never enter promotion or tie breaking. For
the same statistic, prefer `B=199` only when every
matched development cell differs from `B=999` by at most `0.005` absolute
rejection rate; otherwise use `B=999`.

The exploratory alternative set is:

- Gamma: Weibull shape `0.7` and `1.3`, and moment-matched Lognormal;
- Exponential: Weibull shape `0.7` and `1.3`;
- Negative Binomial: Poisson with matched mean, 20% zero-inflated Negative
  Binomial, and a Poisson-lognormal mixture with matched mean.

Each alternative cell exercises the composite fitted procedure, uses
`R_POWER = 2,000` eligible outer units and the same outer eligibility/cap
accounting as its corresponding composite-null cell.

Exact alternative parameter conversion must be frozen in the CP05-B manifest
before CP05-C. Power is descriptive and used only by the deterministic
selection rule; CP05-A makes no minimum-power claim.

If no candidate is eligible, the track is `NO_METHOD_PROMOTED`. It is forbidden
to weaken the risk limit, edit the cell matrix or choose by narrative judgment.
The resulting method/configuration manifest and its digest must be frozen
before holdout unsealing.

## 11. Confirmatory and adversarial rules (`CP05-D`)

CP05-D must run from a clean independent checkout of the frozen candidate. The
auditor verifies source/config/environment digests before receiving the
plaintext holdout namespace. Raw results are immutable append-only artifacts.

Any of the following invalidates the complete holdout run:

- method, `B`, matrix, statistic, estimator or dependency changes after
  unsealing;
- missing or duplicated outer indices;
- unexplained retry, drop, overwrite or manual rerun;
- mismatch between commitment and revealed namespace;
- code/config digest mismatch;
- use of development results in a holdout decision not preregistered here.

A defect correction requires a new versioned preregistration, a new secret
namespace commitment and an entirely new holdout. The invalidated holdout
remains preserved as evidence and is never selectively reused.

## 12. Research result contract; public API deferred

Every experimental assessment record must contain at least:

```text
schema_version
source_sha, config_digest, environment_digest
phase, canonical_cell_id, raw_outer_index, eligible_outer_index
null_type, family_id, canonical_parameters, n
statistic_id, statistic_value, alpha, B
exceedance_count, p_value, reject
observed_fit_provenance
replicate_attempts, raw_inner_indices, eligible_refits, ineligibility_counts
failure_state, failure_reason
seed_derivation_version, output_digest
```

Allowed terminal semantics are `ASSESSED`, `NOT_APPLICABLE`, `NOT_ASSESSED`
and `FAILED`; they may not be collapsed into `NaN` or a boolean. CP05-A does
not freeze a public Python class, export, function signature or production
default. Those require post-calibration architecture review and separate
implementation authorization.

## 13. Interpretation boundary

A GOF assessment evaluates compatibility with one stated family/null/procedure.
It does not prove that the data-generating population belongs to that family.
It must not rank families, auto-select a model, route inference, alter
`MethodSelector`, or compare AIC/BIC across candidates. “Not rejected” is the
strongest permitted non-rejection wording.

## 14. Explicit non-scope and stop conditions

CP05-A excludes production code, production/statistical tests, harness code,
simulation, calibration execution, performance claims, new families,
alternative estimators (Knowledge Base governance tests are the sole test-file
exception), uncertainty intervals for fitted parameters, family
selection/ranking, routing, push, PR and merge.

Stop and return to Architecture/Owner if any work changes the CP04 estimators,
fixed `loc=0`, canonical parameterization, null definition, statistic formulas,
retry/failure accounting, seed derivation, cell matrix, `alpha`, Wilson bound,
selection rule, holdout isolation, result semantics or public-API boundary.

## 15. Authorities

- SciPy `goodness_of_fit` documents the fitted-null parametric-bootstrap
  pattern, refitting each Monte Carlo sample and the plus-one p-value:
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html>.
- Stute, Manteiga and Quindimil (1993) provide the parametric-bootstrap GOF
  foundation: <https://doi.org/10.1007/BF02613687>.
- Phipson and Smyth (2010) motivate nonzero Monte Carlo p-values:
  <https://doi.org/10.2202/1544-6115.1585>.
- Lockhart, Spinelli and Stephens (2007) treat Cramér–von Mises statistics for
  discrete distributions with estimated parameters:
  <https://doi.org/10.1002/cjs.5550350111>.

These sources inform the contract but do not validate pyMagicStats. Only the
preregistered CP05 evidence can support a project calibration claim.
