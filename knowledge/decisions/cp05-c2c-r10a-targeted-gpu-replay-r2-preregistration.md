# DEC-022 — CP05-C2C R10-A targeted GPU replay R2 preregistration

**Status:** `proposed`

**Date:** 2026-09-24

**Owner:** Project Owner — Ehud Bottaro

**Architecture:** ChatGPT; implementation: Cortex; independent QA: Antigravity

**Base / scientific execution SHA:** `d2abd57e65bb7433eff81872a4f6510144d7c267`

**Scientific tree:** `70b20d1f9cbc20d6cd77de4c25b3017a6413932c`

**Documentary branch:** `docs/cp05-c2c-r10a-targeted-gpu-replay-r2-preregistration`

**Supersedes:** none. DEC-019 is preserved as consumed, failed historical
execution authorization, not amended or reused.

## Context, provenance and authorization boundary

The Project Owner supplied the architectural acceptance of the scientific
candidate above and the following independent Antigravity result for that
exact candidate:

```text
VERDICT=PASS
AGY_QA_R3_01_CLOSED=PASS
READY_FOR_NEXT_PREREGISTRATION=YES
READY_FOR_GPU_REPLAY=NO
```

These are attributed Owner-supplied independent QA results, not a new Cortex
audit or a verification of remote audit artifacts. Cortex does not independently
close AGY-QA-R3-01 here. The PASS is not transferred to the new documentary
commit. This decision remains proposed for architectural review.

DEC-019's failed targeted replay and [EV-017](../evidence/cp05-c2c-r10a-targeted-replay-r1-failure.md)
remain historical. [DEC-020](cp05-c2c-r10a-r3-value-grid-separation.md), the
historical AGY-QA-R3-01 FAIL preserved in
[EV-018](../evidence/cp05-c2c-r10a-r3-required-quantities-audit-failure.md), and
[DEC-021](cp05-c2c-r10a-r3-r1-required-quantities.md) remain unchanged.
The reported R3-R1 PASS is a subsequent result, not an erasure of those failures.

```text
DEC019_REUSED_AS_EXECUTION_AUTHORIZATION=NO
REPLAY_VERSION=R2
EXECUTION_SHA=d2abd57e65bb7433eff81872a4f6510144d7c267
READY_FOR_GPU_REPLAY=NO
```

This task materializes only a new proposed preregistration, its versioned frozen
identity manifest, and registry/index entries. As with DEC-019, preregistration
and harness implementation are separate stages. No harness is implemented,
adapted or run here. Future harness work and actual GPU execution require
separate authorization.

## Scientific question and limits

Does candidate `d2abd57e65bb7433eff81872a4f6510144d7c267` implement, on a real
GPU for this historical directed set, the same CPU/CUDA mathematical procedure
required by [DEC-016](cp05-c2b-cuda-equivalence-preregistration.md), including
the structural contracts of DEC-020 and DEC-021?

The unit of comparison is a frozen observed/bootstrap identity, and for MC a
complete frozen outer with its observed record and eligible bootstrap sequence.
This is directed software-equivalence evidence, not a population error-rate
estimand. It does not answer Type-I calibration, power, performance, production
readiness, full C2C equivalence or CP05-D. No performance optimization is in scope.

## Frozen identity artifact and semantic equivalence to R1

The new artifact is
[cp05-c2c-r10a-targeted-gpu-replay-r2-identities.json](cp05-c2c-r10a-targeted-gpu-replay-r2-identities.json).
Its schema version is `cp05-c2c-r10a-targeted-replay-v2`, decision is
`DEC-022`, `replay_version` is `R2`, and `replay_execution_sha` is the
exact EXECUTION_SHA above. It also binds the scientific tree.

Both workload arrays are semantically identical to R1, including all record
metadata, list order, raw indices and gaps. They are copied into a new versioned
artifact, not rediscovered from results. The old manifest remains immutable
historical evidence and is not a mutable authorization for R2.

Source blobs inspected at the exact base commit:

| Source | SHA-256 of exact Git blob bytes |
|---|---|
| `knowledge/decisions/cp05-c2c-r10a-targeted-gpu-replay-identities.json` | `39e01237b3497be542b2c81e3d96ba0de78278bf8f22bd7bc86bed8fc5f2492a` |
| `experiments/distribution_gof/cuda_calibration/targeted_replay_r1/frozen_identity_manifest.json` | `efb550af870213160ed450c222043a0c73cc415a7aecb633c62d61118d72b36d` |

The canonical source blob uses LF; the frozen R1 copy uses CRLF. Their parsed
JSON objects are equal. These byte hashes are provenance, not a requirement
that R2 metadata have the same bytes as R1.

The ordered workload payload consists of the two original keys
`persisted_failure_record_identities` and `mc_failed_outers` with their entire
lists and all nested fields. Its SHA-256 is
`3fe47fef69fc2dfac152dd48eac75841fe25a107b0da6194c06a5d14a7be1395`,
computed over ASCII bytes of Python
`json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)`.
Dictionary keys are sorted for this digest; array order is never sorted or
normalized. This same payload digest must hold for both R1 sources and R2.

Source archive/digest hashes, historical classifications, gates, topology,
counts, p-values, reject flags and evidence limitations are retained as R9
provenance only. The source archives are not re-audited in this task. In
particular, historical distribution-value evidence was `NOT_PERSISTED`;
neither it nor missing historical sample digests may be invented.

## Workload A: persisted R9 failure union

Freeze exactly the original 134 identities in
`persisted_failure_record_identities`:

```text
WORKLOAD_A_IDENTITIES=134
WORKLOAD_A_OBSERVED=10
WORKLOAD_A_BOOTSTRAP=124
WORKLOAD_A_FAMILY=negative_binomial
```

Selection is the persisted R9 classification / fit / statistic / hard-failure
union (C/F/S/H), not a complete inventory of all possible DEC-016 failures.
Retain cell, record type, raw outer and raw inner identity exactly. No discovery,
replacement, pruning or enlargement is allowed after results.

## Workload B: complete historical MC reconstruction

Freeze exactly the original 12 `mc_failed_outers`. For each, reconstruct the
listed observed identity and exactly its 15 eligible bootstrap identities.
Preserve the stored outer order and each bootstrap sequence, including original
`raw_inner_index` gaps. Never normalize the indices to 0..14 or replace a
listed eligible identity to obtain a passing result.

Workload B may overlap Workload A and includes records beyond A. The 134 count
is not the total number of GPU evaluations. Uniqueness is required within A,
across B outer identities and within each B bootstrap sequence; cross-workload
overlap is legitimate, not grounds to discard a frozen identity.

MC compares **new replay CPU versus new replay CUDA** on the matched frozen
identities. The inherited `r9_b_cpu`, `r9_b_cuda`, old p-values and old reject
flags are historical context only, never acceptance oracles. An incomplete
outer or a changed eligible sequence fails closed.

```text
R_EQ=8
B_EQ=15
alpha=0.05
p=(b+1)/(B+1)
ties=>=
```

These parameters, seed derivation, fixtures, retry/eligibility policy and the
mathematical procedure remain frozen in the execution candidate and DEC-016.
The replay does not introduce a new bootstrap policy.

## Scientific and structural acceptance gates

Require all of the following, retaining exactly the DEC-016 tolerances:

```text
CLASSIFICATION_MISMATCH=0
CUDA_NONCONVERGENCE=0
FIT_GATE_FAILURE=0
DISTRIBUTION_VALUE_GATE_FAILURE=0
STATISTIC_GATE_FAILURE=0
UNEXPLAINED_DISCREPANCY=0
MC_BOOTSTRAP_IDENTITY_MISMATCH=0
MC_OUTERS_RECONSTRUCTED=12
MC_EXCEEDANCE_COUNT_MISMATCH=0
MC_REJECT_DECISION_MISMATCH=0
```

In addition, persist explicit structural failure reasons and require zero
occurrences of every reason below, not just an aggregate passing value gate:

```text
EVALUATION_POINT_IDENTITY_MISMATCH=0
DISTRIBUTION_VALUE_LENGTH_MISMATCH=0
MISSING_CPU_DISTRIBUTION_QUANTITY=0
MISSING_CUDA_DISTRIBUTION_QUANTITY=0
MISSING_BOTH_DISTRIBUTION_QUANTITY=0
UNEXPECTED_DISTRIBUTION_QUANTITY=0
```

The Negative Binomial family contract defines exactly:
`pmf, logPMF, cdf, sf, logCDF, logSF`. Adapter keys do not define the comparison
universe. Missing CPU, CUDA or both quantities, and unexpected keys in CPU,
CUDA or both, fail closed. Preserve DEC-021 structural evidence, including
engine attribution for extra keys. A structural violation cannot become PASS
through later numerical fit/statistic agreement or through missing evidence.

## R3 value-grid / statistic-support separation

For Negative Binomial, the two frozen mathematical objects remain distinct:

```text
VALUE_EVALUATION_GRID=0..max(sample)
GOF_STATISTIC_SUPPORT=full certified_support.indices
```

The known historical identity is:

```text
negative_binomial|r=0.25,p=0.1|n=20|AD|composite|raw_outer=2|raw_inner=7
sample max=9
value grid=0..9
certified statistic support=0..169
```

This is a historical regression anchor from DEC-020/EV-017, not a sample
regenerated or executed in this documentary task. Do not conflate the two
grids or truncate certified statistic support to the value grid.

## Execution identity, fresh output and stopping policy

Before any future execution, the separately authorized harness must verify that
the scientific checkout HEAD is exactly EXECUTION_SHA and its tree is
`70b20d1f9cbc20d6cd77de4c25b3017a6413932c`. The documentary commit that
contains DEC-022 and this manifest is **not** the execution SHA. The R1 harness
must not be assumed R2-compatible merely because the workload is unchanged.

```text
FRESH_OUTPUT_REQUIRED=YES
CHECKPOINT_REUSE=NO
PRIOR_FAILED_OUTPUT_REUSE=NO
AUTO_RESUME=NO
AUTO_RERUN=NO
```

Any unexpected failure requires preserving evidence, stopping and returning
to architecture. A failed gate is not permission to rerun. Never alter
tolerances, fixtures, seeds, workloads or gates after seeing results.

## Documentary validation and impact

Baseline checks verified the local scientific HEAD/tree and clean worktree.
The registry validator passed before materialization. Validation of this
candidate is documentary only: registry validation; strict JSON parsing;
manifest cardinality, identity consistency and uniqueness; exact ordered
semantic comparison with both historical Git blobs; and Git diff checks.

Reproducible acceptance checks:

1. Run `python -B knowledge/tools/validate_registry.py`.
2. Parse the new manifest and both historical manifests read with binary
   `git show <base>:<path>` output; require no duplicate JSON keys or non-finite
   JSON constants. Compare both complete workload arrays for equality, not
   merely cardinality or unordered membership; verify the payload digest above.
3. Verify 134 unique A identities, 10 observed, 124 bootstrap, all NB; 12
   unique B outers, each with its matching observed identity and 15 unique
   bootstrap identities. Recompose identities from cell/raw indices and
   preserve all sequence gaps and order.
4. Verify the new version, decision, exact execution SHA/tree, source hashes
   and retained historical metadata. Existing registry records must remain
   identical; DEC-022 is the only new record.
5. Run `git diff --check` and `git diff --cached --check`; require exactly
   these four changed paths and no changed frozen historical/scientific files:

```text
knowledge/decisions/cp05-c2c-r10a-targeted-gpu-replay-r2-preregistration.md
knowledge/decisions/cp05-c2c-r10a-targeted-gpu-replay-r2-identities.json
knowledge/decisions/README.md
knowledge/registry.json
```

Local documentary validation used Windows PowerShell and repository-local
`.venv/Scripts/python.exe`, Python 3.12.14, with `-B` and only standard-library
JSON/hash/subprocess checks. Registry validation, strict JSON parsing, ordered
R1/R2 semantic equality, cardinalities, uniqueness, identity recomposition,
source/payload hashes, decision links and frozen-contract presence all passed.
Prior registry records are unchanged; the changed-file set is exactly the four
paths above. Git reported its existing LF-to-CRLF working-copy warnings; no EOL
configuration or attributes were changed. Seed: not applicable; no RNG was run.
Final staged diff and commit identity checks are recorded in the handoff.
These checks do not run scientific tests, load CUDA, reconstruct samples or execute GPU.
There are no changes to APIs, scientific code, tests, DEC-016/018/019/020/021,
EV-017/018 or `targeted_replay_r1/*`. The new manifest is a documentary
contract; an executable R2 harness validator remains future authorized work.

## Alternatives, limits and next role

Reusing DEC-019 as execution authorization would erase the boundary of a
consumed failed run and is rejected. Changing the workload would answer a
different question and is rejected. Reusing the same semantic identity set in
a new R2 artifact preserves the historical directed question while binding the
remediated scientific candidate explicitly.

Any identity drift, missing evidence, request to change a frozen contract or
unexpected execution failure requires return to architecture, not automatic
repair. The remaining limit is that the R2 harness and real-GPU behavior have
not been validated by this task.

Only one local documentary commit is authorized, with message
`preregister R10-A targeted GPU replay R2`. No push, PR, merge, main change,
Quantum, CUDA/GPU execution, targeted replay, full campaign, CP05-D or holdout
access. Next role: ChatGPT architecture for review of this exact new candidate;
no independent approval or execution authorization is inferred from local
documentary checks.
