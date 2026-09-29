# DEC-023 — CP05-C2C R10-A targeted GPU replay R3 environment-remediated preregistration

- Status: `proposed`; not independently audited or accepted by Cortex.
- Date recorded: 2026-09-27.
- Owner: Project Owner — Ehud Bottaro.
- Architecture: ChatGPT; documentary implementation: Cortex.
- Reviewers: ChatGPT architecture and Antigravity independent adversarial QA.
- Supersedes: none. DEC-022 and its consumed failed R2 execution remain immutable history.
- Repository: `udibott-011235/pyMagicStats`.
- Documentary branch: `docs/cp05-c2c-r10a-targeted-gpu-replay-r3-preregistration`.

## Context, scientific question and exact identities

This is a new proposed documentary preregistration, not a replay, harness
implementation, independent audit or execution authorization. The base is the
Owner-designated documentary/harness candidate. Local object identities were
verified; a fresh remote check was not completed (SSH host-key verification
failed; an HTTPS read attempt was stopped without a result). No SSH repair or
remote configuration change was made. Remote publication remains Owner-reported.

```text
BASE_SHA=2c1a5e99db086651efd046921d48ef8187a5e0d7
BASE_TREE=cd2e4e5220c3c3734d7b7796a58a7172f1a4f84c
SCIENTIFIC_SHA=d2abd57e65bb7433eff81872a4f6510144d7c267
SCIENTIFIC_TREE=70b20d1f9cbc20d6cd77de4c25b3017a6413932c
EXECUTION_SHA=d2abd57e65bb7433eff81872a4f6510144d7c267
EXECUTION_TREE=70b20d1f9cbc20d6cd77de4c25b3017a6413932c
REPLAY_VERSION=R3
READY_FOR_GPU_REPLAY=NO
```

R3 asks exactly the scientific question in
[DEC-022](cp05-c2c-r10a-targeted-gpu-replay-r2-preregistration.md): does that
unchanged scientific candidate implement the same fixed-data CPU/CUDA procedure
required by [DEC-016](cp05-c2b-cuda-equivalence-preregistration.md), including
[DEC-020](cp05-c2c-r10a-r3-value-grid-separation.md) and
[DEC-021](cp05-c2c-r10a-r3-r1-required-quantities.md), on the frozen directed
historical workload on a real GPU?

The units are observed/bootstrap identities and complete MC outers. This is
directed software-equivalence evidence, not a population error-rate estimand.
Only the environment readiness prerequisite changes. The documentary commit
is not the scientific execution SHA. No PASS transfers to this new commit.

## Consumed R2 failure and linked evidence

[EV-019](../evidence/cp05-c2c-r10a-targeted-replay-r2-environment-remediation.md)
preserves the Owner/Quantum report separately from this prospective contract.
R2 was executed exactly once and failed at the first Workload A identity:

```text
R2_EXECUTION_COUNT=1
R2_EXECUTION_STATUS=FAILED
R2_AUTHORIZATION_CONSUMED=YES
R2_AUTO_RERUN=NO
R2_AUTO_RESUME=NO
R2_FAILURE_CLASS=GPU_ENVIRONMENT_DEPENDENCY
R2_ROOT_CAUSE=CUDA_WHEEL_LIBRARY_PATH_NOT_VISIBLE_TO_DYNAMIC_LOADER
FIRST_FAILURE_IDENTITY=negative_binomial|r=0.25,p=0.1|n=20|AD|composite|raw_outer=2|raw_inner=7
```

Persisted reason reported by the Owner:

```text
CuPy failed to load libnvrtc.so.13:
OSError: libnvrtc.so.13: cannot open shared object file
```

The failure does not establish a scientific discrepancy, R10-A solver failure,
value-grid regression or harness semantic failure. These labels mean **not
established by this event**, not proof that such failures can never occur:

```text
SCIENTIFIC_DISCREPANCY=NO
R10A_SOLVER_FAILURE=NO
VALUE_GRID_REGRESSION=NO
HARNESS_SEMANTIC_FAILURE=NO
```

The R2 output and archive are immutable historical evidence; neither this
decision nor the environment remediation resets the consumed R2 authorization.
No R2 archive path, checksum, execution timestamp or independent verification
was supplied for this task; none is invented. The R1 failure in
[EV-017](../evidence/cp05-c2c-r10a-targeted-replay-r1-failure.md) also remains
unchanged. The Owner-reported strong smoke and one-record diagnostic in EV-019
are not Workload A/B execution, MC results or acceptance oracles.

## Frozen R3 manifest and exact reconstruction

The new artifact is
[cp05-c2c-r10a-targeted-gpu-replay-r3-identities.json](cp05-c2c-r10a-targeted-gpu-replay-r3-identities.json).
Its schema is `cp05-c2c-r10a-targeted-replay-v3`, decision `DEC-023`, and
replay version `R3`. It binds the unchanged scientific SHA/tree above.

Its immediate source is the exact Git blob at BASE_SHA:

```text
SOURCE_PATH=experiments/distribution_gof/cuda_calibration/targeted_replay_r2/frozen_identity_manifest.json
SOURCE_SHA256=9099c2ab099468fcb26381d07b303e79f06da394d487dfee923bfe090310910f
SOURCE_SIZE_BYTES=117577
SOURCE_LF=2963
SOURCE_CRLF=0
ORDERED_PAYLOAD_SHA256=3fe47fef69fc2dfac152dd48eac75841fe25a107b0da6194c06a5d14a7be1395
```

Reconstruction is documentary only: preserve both complete JSON array values
`persisted_failure_record_identities` and `mc_failed_outers`, including every
nested historical field, number, list order and raw-index gap. Their raw UTF-8
array-value spans are copied unchanged from the source blob. Only top-level
`schema_version`, `decision`, `replay_version`, `identity_semantics` and
`execution_authorization` change, and `r2_identity_source` is added. Every
other top-level member is unchanged, including the original `identity_source`
provenance chain. No source manifest is edited and no samples are regenerated.

For semantic verification, strictly parse both manifests (reject duplicate
keys and non-finite constants), compare both entire arrays in order, and form
`payload` from exactly those two named keys. Hash ASCII bytes of
`json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
allow_nan=False)`. Only dictionary keys are sorted; arrays are never sorted.
The digest above must match both R2 and R3. Whole-manifest byte identity is not
expected because R3 authorization/version metadata is new.

## Unchanged workloads and scientific acceptance

```text
NAMESPACE=CP05-C2C
R_EQ=8
B_EQ=15
alpha=0.05
ties=>=
p=(b+1)/(B+1)
reject=p<=0.05
WORKLOAD_A_IDENTITIES=134
WORKLOAD_A_OBSERVED=10
WORKLOAD_A_BOOTSTRAP=124
WORKLOAD_B_OUTERS=12
WORKLOAD_B_ELIGIBLE_BOOTSTRAPS_PER_OUTER=15
```

A retains the ordered R9 persisted C/F/S/H failure union, not a rediscovered
inventory. B retains each ordered observed identity and its exact 15 eligible
bootstraps, including raw-inner gaps. Compare new CPU with new CUDA on the same
arrays. Historical R9 classifications, gates, counts, p-values and reject flags
remain provenance, never acceptance oracles. Missing historical sample digests
or distribution-value evidence must not be invented. Cross-workload overlap
is retained. No rediscover, reorder, replace, prune or enlarge operation is
permitted; no fixture, seed derivation, retry/eligibility or RNG change is allowed.

All gates and tolerances in DEC-016/020/021/022 remain unchanged, as does
[DEC-018](cp05-c2c-r10a-r2-root-certification.md) root certification. Required
final counters remain:

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

Require zero in both structural item and distinct-record views for:

```text
EVALUATION_POINT_IDENTITY_MISMATCH
DISTRIBUTION_VALUE_LENGTH_MISMATCH
MISSING_CPU_DISTRIBUTION_QUANTITY
MISSING_CUDA_DISTRIBUTION_QUANTITY
MISSING_BOTH_DISTRIBUTION_QUANTITY
UNEXPECTED_DISTRIBUTION_QUANTITY
```

Retain diagnostic separation `EXPLAINED_FAILURE != UNEXPLAINED_DISCREPANCY`;
schema completeness is not numerical agreement. NB quantities remain exactly
`pmf,logPMF,cdf,sf,logCDF,logSF`. Missing or extra quantities fail closed.
Value-grid/support separation is unchanged: distribution values on
`0..max(sample)`, GOF statistic on full `certified_support.indices`.
The EV-019 single-record values are not tolerance adjustments or an oracle.

## Mandatory R3 environment preflight, before scientific output

The reported R2 failure demonstrates that a simple CuPy sum is insufficient:
it may pass without exercising NVRTC. Every future separately authorized R3
harness/execution must verify **all** of the following in the intended execution
environment, before creating the scientific results directory and before
starting any frozen scientific identity:

```text
CUDA_DEVICE_COUNT>=1
FLOAT64_BASIC_SMOKE=PASS
LIBCUDART_SO_13_LOAD=PASS
LIBNVRTC_SO_13_LOAD=PASS
NVRTC_VERSION=(13,0)
RAWKERNEL_NVRTC_COMPILE=PASS
RAWKERNEL_NVRTC_LAUNCH=PASS
RAWKERNEL_NUMERICAL_RESULT=PASS
CUPY_ELEMENTWISE_REDUCTION=PASS
CUPYX_SCIPY_GAMMALN=PASS
CUPYX_SCIPY_BETAINC=PASS
```

The RawKernel check must actually exercise the NVRTC backend (not a cached
result standing in for compilation), launch and verify its numerical result.
Record the effective `CUDA_PATH`, `LD_LIBRARY_PATH` and `CUDA_WHEEL_ROOT`,
library/NVRTC versions and the individual prerequisite results explicitly.
Preflight evidence can be reported outside the not-yet-created scientific
results directory; it must not require creating that directory to detect failure.
The precise executable preflight implementation and its tests are future
separately authorized harness work, not implemented or certified here.

On any prerequisite failure, stop without starting an identity or creating
scientific output:

```text
EXECUTION_NOT_STARTED=YES
AUTHORIZATION_NOT_CONSUMED=YES
OUTPUT_NOT_CREATED=YES
```

That describes a future explicit R3 authorization only; it does not grant one
now and does not resurrect R2. It is not permission for automatic retries.
Once the first scientific identity starts:

```text
AUTHORIZATION_CONSUMED=YES
AUTO_RERUN=NO
AUTO_RESUME=NO
```

Exact scientific HEAD/tree, clean checkout, module-origin checks and fresh
output remain mandatory. An environment PASS alone never authorizes execution.
The existing R2 harness creates output before its minimal GPU smoke; it is
historical and must not be assumed to satisfy this new R3 ordering contract.

## Freshness, stopping policy and claim boundary

```text
FRESH_OUTPUT_REQUIRED=YES
CHECKPOINT_REUSE=NO
R1_OUTPUT_REUSE=NO
R2_OUTPUT_REUSE=NO
AUTO_RESUME=NO
AUTO_RERUN=NO
```

At the first scientific failure, preserve the failed record and available
failure/context, partial summary and digests; stop remaining A/B/MC work and
return to architecture. Never edit evidence or change acceptance after results.
Even a future complete R3 PASS permits only:

> The preregistered R10-A targeted GPU replay R3 passed for the frozen directed historical workload.

It does not establish `FULL_C2C_EQUIVALENCE`, `TYPE_I_CALIBRATION`, `POWER`,
`PERFORMANCE`, `PRODUCTION_READINESS` or `CP05_D_READINESS`.

## Alternatives, validation, impact and review trigger

Reusing consumed R2 authorization is rejected. Changing scientific code,
identity selection or tolerances to accommodate an environment failure is
rejected. R3 preserves the same scientific question with a stronger readiness
prerequisite and a new versioned documentary manifest.

Allowed changes are exactly this decision, the R3 manifest, EV-019,
`knowledge/decisions/README.md` and `knowledge/registry.json`. Existing registry
records remain unchanged; only DEC-023 and EV-019 are added and the update date
advances. No API, scientific/harness code, tests or historical evidence changes.

Reproducible documentary validation:

1. `python -B knowledge/tools/validate_registry.py` before and after changes.
2. Strict JSON, exact scientific SHA/tree and metadata checks.
3. Raw array-span byte equality and exact ordered semantic R2/R3 equality;
   unchanged payload hash; identity recomposition, 134/10/124 and 12/15 counts,
   uniqueness within A, B outers and each B bootstrap list.
4. Decision/index/registry/evidence links and exact five-file allowlist.
5. `git diff --check` and staged diff check; no edits to DEC-016/018/019/020/021/022,
   historical evidence or any scientific/harness file; final blob checks.

Use standard-library document/JSON/hash checks only, not scientific tests,
sample generation or GPU execution. Local checks are reproducibility checks,
not a Cortex audit or promotion to accepted status. Missing external archive
identifiers and unverified Owner-reported results remain explicit limitations.
Any contract drift or new execution failure requires architecture review.

Only one local documentary commit is authorized. No push, PR, merge, main
change, Quantum access, CUDA/GPU or R3 execution, full 1152 campaign, CP05-D or
holdout access. Next role: ChatGPT architecture on the exact new SHA; independent
Antigravity QA and any future harness/publication/execution need their own gates.
