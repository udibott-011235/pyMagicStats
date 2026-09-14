# EV-013 — CP05-A GOF architecture baseline and preregistration

- **Estado:** `accepted`; CP05-A y CP05-B cerrados
- **Fecha:** 2026-09-12
- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Checkpoint vigente:** `CP05-B — COMPLETE / GOVERNANCE_CLOSED`
- **Repositorio:** `udibott-011235/pyMagicStats`
- **Baseline:** `main` @ `6409717ebdfdd34d41d41c36983dda82de935e6b`
- **Rama:** `feature/distribution-family-framework-cp05-gof-calibration`
- **Contrato aceptado:** `DEC-014`
- **Rol de materialización:** `implementation-engineering`

## Claim de apertura (histórico)

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

## Phase status at rematerialization (historical)

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

## Rematerialization validation record (historical)

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

## Rematerialization authorization boundary (historical)

The current authorization permits a local CP05-A architecture candidate only.
It does not authorize production, calibration, push, PR, merge or CP05-B–D.
Antigravity must independently audit the new exact candidate SHA first,
followed by ChatGPT interpretation and a separate Project Owner decision.
Any commit change requires a new exact-SHA audit. DEC-014 and EV-013 remain
under_review; CP05-A remains IN_PROGRESS.

## CP05-A post-merge closure — 2026-09-13

This section projects the governance state effective when the local BR-026
closure is integrated. It preserves the historical opening, rejected-draft
verdict and unexecuted risk register above.

### Accepted candidate and independent audits

The audited candidate is `2caf234cf1bfa8c66dd0317986803ff443ca3194`,
parent `6409717ebdfdd34d41d41c36983dda82de935e6b`, tree
`0375d8ea1260fa9825ad7ff904e76736611b8563`. Its transfer artifacts were:

- bundle SHA-256: `7c3e8282b2e8af04347ee9efb70fdbe653ed202af6c465f401fdd96194821d5e`;
- identity-manifest SHA-256: `ab373827ec63de6aabbadea560b7eea9ff37eac20492d19b2a2ff18e03028f82`.

The Owner supplied Antigravity's independent bundle audit: IDENTITY_CHECK,
BUNDLE_INTEGRITY, FSCK_FULL_STRICT, GENEALOGY_CHECK, SCOPE_CHECK and
EVIDENCE_REPRODUCTION PASS; PRODUCTION_DIFF EMPTY; GOV_001–004 CLOSED;
STAT_001 CLOSED; TECHNICAL_DESIGN_DISPOSITION USABLE;
GOVERNANCE_DISPOSITION COMPLIANT; VERDICT PASS.
The independent remote publication audit of PR #14 also passed, followed by
Architecture PASS and separate Owner merge authorization.
These findings concern the documentary CP05-A candidate only. The new closure
SHA requires its own independent review.

The rejected noncanonical draft
`17bf06639ad84a18c26865b46cdecf26dc3ab9ed` remains excluded from the
candidate and merge genealogy. ROLE_DRIFT history is preserved, not relabeled
as a successful original materialization.

### PR #14 integration evidence

```text
PR_NUMBER=14
PR_STATE=CLOSED
PR_MERGED=YES
MERGE_METHOD=merge commit
MERGE_SHA=3d9db61cf7414ce7fe3d94819b5f9e005fff527f
FIRST_PARENT=6409717ebdfdd34d41d41c36983dda82de935e6b
SECOND_PARENT=2caf234cf1bfa8c66dd0317986803ff443ca3194
MERGE_TREE=0375d8ea1260fa9825ad7ff904e76736611b8563
CANDIDATE_TREE_MATCH=PASS
SOURCE_BRANCH_PRESERVED=YES
EXTRA_COMMITS=NONE
```

The remote reads and fresh clone confirmed the merge parents and tree.
The merge includes a PGP signature. GitHub's Git commit API reports
`verification.verified=true`; this is GitHub's verification report, not a
claim that Cortex independently performed local cryptographic verification.

Remote CI returned zero commit statuses and zero check runs for the candidate.
`CI_STATUSES=NONE_REPORTED` is compensated only by independently reproduced
registry validation, nine Knowledge Base tests and diff/scope checks within
this documentary scope. No executable GOF, numerical accuracy, RNG invariance,
runtime viability, type-I calibration, power or production validity follows.

### Disclosed publication event

The authorized ordinary non-forced branch-creation push automatically reported:

```text
Bypassed rule violations for refs/heads/feature/distribution-family-framework-cp05-gof-calibration:

- Cannot create ref due to creations being restricted.
```

No ruleset was changed and no manual bypass was requested. Main remained
unchanged during publication. The Project Owner accepted preserving this
procedural event. Future branch-creation ruleset alignment remains separate
governance debt and does not alter the audited candidate.

### Projected state and next role at CP05-A closure (historical)

```text
DEC_014=ACCEPTED
EV_013=ACCEPTED
CP05_A_ARCHITECTURE=ACCEPTED
CP05_A_INTEGRATION=COMPLETE
CP05_A_GOVERNANCE=CLOSED
CP05_A_OVERALL=COMPLETE
CP05_OVERALL=IN_PROGRESS
CP05_B_AT_CP05_A_CLOSURE=NOT_STARTED
CP05_C=NOT_STARTED
CP05_D=NOT_STARTED
CP06_CP08=NOT_STARTED
```

BR-025 is archived / fully_contained / merged with its source ref preserved.
BR-001 records main at the PR #14 merge. BR-026 remains under_review /
same_head / pending at its clean opening snapshot, without self-reference.

The CP05-D holdout commitment remains PENDING_OWNER before CP05-C.
No holdout secret was created or inspected. The only current authorization is
one local documentary closure commit; no push, PR, merge or CP05-B–D.
Next: independent Antigravity audit of the exact closure SHA, then ChatGPT
interpretation and a separate Project Owner decision.

## CP05-B software harness closure — 2026-09-13

### Identity, integration and independent audit

```text
CP05_B_CANDIDATE_SHA=75529e4415558c1abef6166432ebbafafd00a812
CP05_B_MERGE_SHA=9fac41a38ed6583356b0e305a856dca7a3096530
CP05_B_MERGE_TREE=71f7b72fedee0fd4dbcdd1e41f208d53056fdb2f
ADVERSARIAL_AUDIT=PASS
FINDING_CP05B_001=CLOSED
```

PR #16 integrated the exact remediated candidate by merge commit. Its first
parent is `5eb179be578594aa900a29bf5ae2f5540e05ffa2`, its second parent is the
candidate above, and its tree equals the candidate tree. The source branch
`feature/distribution-family-framework-cp05-b-harness` remains preserved.

The independent audit closed `FINDING-CP05B-001` after the Parquet artifact
serialization remediation. The technical verdict is PASS within the exact
claim scope `SOFTWARE_CORRECTNESS_ONLY`.

### Validation evidence

```text
RESEARCH_TESTS=43 passed
CP04_REGRESSION=313 passed
DISTRIBUTION_REGRESSION=888 passed
PRODUCTION_REGRESSION=1128 passed, 3 inherited skips
REGISTRY_VALIDATOR=PASS
COMPILEALL=PASS
DIFF_CHECK=PASS
```

### Canonical state and claim limits

```text
CP05_A_INTEGRATION=COMPLETE
CP05_A_GOVERNANCE=CLOSED
CP05_A_OVERALL=COMPLETE
CP05_B_INTEGRATION=COMPLETE
CP05_B_GOVERNANCE=CLOSED
CP05_B_OVERALL=COMPLETE
CP05_OVERALL=IN_PROGRESS
CP05_C=NOT_STARTED
CP05_D=NOT_STARTED
CP06_CP08=NOT_STARTED
CP05_B_CLAIM_SCOPE=SOFTWARE_CORRECTNESS_ONLY
CALIBRATION_VALIDATED=NO
TYPE_I_CONTROL_VALIDATED=NO
POWER_VALIDATED=NO
METHOD_SELECTED=NO
PRODUCTION_SUITABILITY_VALIDATED=NO
CP05_C_EXECUTED=NO
CP05_D_EXECUTED=NO
HOLDOUT_ACCESSED=NO
HOLDOUT_SECRET_GENERATED=NO
```

Repository branch/ruleset governance required a PR #16-specific bypass and
remains procedural debt, not statistical debt. Restricted Windows pytest
environments may require explicit `--basetemp`.

BR-028 records the clean opening snapshot for this documentary closure. No
CP05-C implementation or execution is authorized by this state change.
