# EV-013 — CP05-A GOF architecture baseline and preregistration

- **Estado:** `under_review`
- **Fecha:** 2026-09-12
- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint:** `CP05-A — IN_PROGRESS`
- **Repositorio:** `udibott-011235/pyMagicStats`
- **Baseline:** `main` @ `6409717ebdfdd34d41d41c36983dda82de935e6b`
- **Rama:** `feature/distribution-family-framework-cp05-gof-calibration`
- **Contrato propuesto:** `DEC-014`
- **Rol de materialización:** `implementation-engineering`

## Claim

This record fixes the clean CP05-A opening baseline and preserves the
preregistered GOF design proposed by ChatGPT/Architecture in DEC-014 for
independent Antigravity audit of the new exact SHA.
It demonstrates only repository identity, CP04 canonical closure, existing
Wave 1 fitting surfaces and documentation-level preregistration. It does not
claim that a harness, GOF implementation, calibration result, performance
result or production API exists.

## Provenance and clean rematerialization

ChatGPT / Arquitectura produced the noncanonical architectural draft
`17bf06639ad84a18c26865b46cdecf26dc3ab9ed`, tree
`c33a1b2a63994e6f8725819a3e590384472ae6fd`.
Antigravity reported `TECHNICAL_DESIGN_DISPOSITION=USABLE` but rejected that
SHA for merge because of `ROLE_DRIFT`:
`GOVERNANCE_DISPOSITION=REQUIRES_REMATERIALIZATION`,
`CAN_CURRENT_SHA_BE_MERGED=NO`, `VERDICT=FAIL_DO_NOT_MERGE`.
That verdict is not a PASS for this new candidate.

Cortex / Implementación performs the clean documentary rematerialization from
`6409717ebdfdd34d41d41c36983dda82de935e6b` in a new clone directly from the
authorized remote. The verified bundle is used only in a separate reference
repository; its base-to-draft diff is applied as uncommitted file changes.
No draft ref or commit object is imported into the target repository.
The new commit must have the baseline as its sole parent; the noncanonical
draft does not belong to its genealogy. No cherry-pick, merge or rebase is used.

- **Design authority:** ChatGPT / `statistical-software-architecture`.
- **Materialization, registry and governance tests:** Cortex / `implementation-engineering`.
- **Independent next reviewer:** Antigravity / `adversarial-statistical-qa`.
- **After audit:** ChatGPT interprets the findings; the Project Owner decides
  separately. No self-review or acceptance is claimed.

Transfer integrity verified before documentary changes:

```text
bundle_file = pymagicstats-cp05-audit-17bf066.bundle
bundle_sha256 = 72f13ba7d6b47b956cb45530cc8b63e02aef11c047ad6254dd6d785d0c3301c3
manifest_file = pymagicstats-cp05-audit-17bf066-manifest.json
manifest_sha256 = 2980caaeda60569ee2b884e4edf3932805b5b6d64760532ef5f4f3669cd42b7a
bundle_verify = complete history; bundle is okay
noncanonical_draft_in_genealogy = false
```

DEC-014 section 4 records the Owner-required stable discrete-AD tail
clarification: the internal summand has factor n squared; the complete
contribution has factor n. The remaining mathematical design, risk limits,
matrices, selection, seeds and holdout are reused unchanged. This is
documentary clarification, not implementation or a numerical-oracle result.

## Exact identity and CP04 closure

The local branch was opened from the exact canonical snapshot:

```text
HEAD before branch = 6409717ebdfdd34d41d41c36983dda82de935e6b
main = 6409717ebdfdd34d41d41c36983dda82de935e6b
origin/main observed locally = 6409717ebdfdd34d41d41c36983dda82de935e6b
branch parent = 6409717ebdfdd34d41d41c36983dda82de935e6b
branch = feature/distribution-family-framework-cp05-gof-calibration
opening worktree = clean
```

The owner-certified PR #13 closure is:

```text
PR_NUMBER = 13
PR_STATE = CLOSED
PR_MERGED = YES
MERGE_METHOD = merge commit
MERGE_SHA = 6409717ebdfdd34d41d41c36983dda82de935e6b
FIRST_PARENT = 2b6e1263b8489592030b0838cd3851f193fbfd7f
SECOND_PARENT = 9da985d1770ac2ec6bb542e2233d6e882e56d1c2
MERGE_TREE = 5df9177b4e8a1c112d169208ee7e37bb75ee2588
EXPECTED_TREE_MATCH = PASS
GITHUB_SIGNATURE = VERIFIED
SOURCE_IMPLEMENTATION_BRANCH = PRESERVED
SOURCE_GOVERNANCE_BRANCH = PRESERVED
EXTRA_COMMITS = NONE
```

Local `git log -1 --format='%H %P %T'` reproduced the merge SHA, both parents
and the tree. No remote mutation was performed during CP05-A.

## Existing production surface

The transferred draft records the following baseline reconnaissance; Cortex
reuses this design context without claiming a new production or calibration audit:

```text
GammaFamily.fit(data) -> FitResult = PRESENT
ExponentialFamily.fit(data) -> FitResult = PRESENT
NegativeBinomialFamily.fit(data) -> FitResult = PRESENT
Gamma fixed-loc MLE = PRESENT
Exponential fixed-loc MLE = PRESENT
Negative Binomial generalized real-r MLE = PRESENT
FitResult / fitted distribution provenance = PRESENT

Wave 1 fitted-family GOF public API = ABSENT
CP05 research harness = ABSENT
CP05 calibration artifacts = ABSENT
CP05 holdout commitment = PENDING_OWNER
```

Legacy Pearson discrete GOF helpers remain separate and do not supply CP05
calibration evidence.

## Preregistered statistical risk

DEC-014 fixes the primary estimand, simple/composite null split, refit rule,
candidate statistics, Monte Carlo plus-one p-value, seed derivation, failure
accounting, matrices, selection rule and sealed holdout protocol.

The owner-approved risk criterion is:

```text
alpha = 0.05
cell acceptance = Wilson 95% upper endpoint for type-I error <= 0.065
pooling across cells = FORBIDDEN
mandatory-cell deletion after inspection = FORBIDDEN
```

For composite Negative Binomial cells, type-I error is conditional on an
eligible observed finite CP04 MLE. Applicability and numerical failures use
separate denominators and intervals.

## Risk register at opening

| ID | Risk | Contract control | CP05-A status |
|---|---|---|---|
| `R-CP05-001` | Reusing simple-null references after estimation | Composite refit on every eligible replicate | Controlled by design; untested |
| `R-CP05-002` | Zero/anti-conservative Monte Carlo p-values | `(b+1)/(B+1)` and `>=` ties | Controlled by design; untested |
| `R-CP05-003` | Worker/shard RNG dependence | Per-unit SHA-256 seed derivation | Controlled by design; untested |
| `R-CP05-004` | Silent denominator manipulation | Explicit eligible/raw/failure counters | Controlled by design; untested |
| `R-CP05-005` | NB finite-MLE conditioning hidden from users | Conditional estimand plus separate applicability | Controlled by design; untested |
| `R-CP05-006` | Discrete infinite-tail truncation error | Certified remainder and independent oracle | Open for CP05-B |
| `R-CP05-007` | AD tail underflow/overflow | Stable log-domain evaluation or fail closed | Open for CP05-B |
| `R-CP05-008` | Development overfitting / holdout leakage | Seed commitment and freeze-before-reveal | Commitment pending Owner |
| `R-CP05-009` | Infeasible nested-bootstrap cost | Measured CP05-C performance gate; no reduced matrix | Open for CP05-C |
| `R-CP05-010` | Family selection or identity overclaim | Separate assessment layer and wording boundary | Controlled by architecture |

## Current phase status and nonclaims

```text
CP04_IMPLEMENTATION = COMPLETE
CP04_INTEGRATION = COMPLETE
CP04_GOVERNANCE = CLOSED
CP04_OVERALL = COMPLETE
CP05-A = IN_PROGRESS
CP05-B = NOT_STARTED
CP05-C = NOT_STARTED
CP05-D = NOT_STARTED
CP05_OVERALL = IN_PROGRESS
STAGE_DIST_FAMILIES_001 = IN_PROGRESS
```

CP05-A changes documentation, registry projections and Knowledge Base tests
only. It does not add or modify production modules, GOF routines, simulation
runners or calibration results. No calibration command was executed.

## Validation record

Cortex ran the baseline registry validator and Knowledge Base suite in the new
clone before edits: PASS and 9 passed. Final documentary validation commands are
`python knowledge/tools/validate_registry.py`,
`python -m pytest -q tests/test_knowledge_base.py` and `git diff --check`.
The final result is required before creating the local candidate:

```text
registry validator = PASS
Knowledge Base tests = 9 passed
production-path diff = EMPTY
git diff --check = PASS
full suite = NOT_RUN (documentation-only scope)
```

These checks validate governance consistency only. They cannot establish
statistical calibration or software correctness for a future harness.

## Authorization boundary

The current authorization permits a local CP05-A architecture candidate only.
It does not authorize production, calibration, push, PR, merge or CP05-B–D.
Antigravity must independently audit the new exact candidate SHA first,
followed by ChatGPT interpretation and a separate Project Owner decision.
Any commit change requires a new exact-SHA audit. DEC-014 and EV-013 remain
under_review; CP05-A remains IN_PROGRESS.
