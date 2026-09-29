# DEC-025 — CP05-C2C R10-A targeted GPU replay R4 preregistration

- Status: `accepted`.
- Date recorded: 2026-09-27.
- Owner: Project Owner — Ehud Bottaro.
- Architecture: ChatGPT; documentary implementation: Cortex.
- Reviewers: ChatGPT architecture and Antigravity independent adversarial QA.
- Supersedes: none; historical decisions and consumed executions remain unchanged.
- Repository: `udibott-011235/pyMagicStats`.
- Branch: `docs/cp05-c2c-r10a-targeted-gpu-replay-r4-preregistration`.

## Scope, authority and frozen candidate

The initial authorization covered documentary materialization only. The Project
Owner subsequently accepted DEC-025 and authorized its documentary closure,
then R4 harness implementation in a separate child commit of the closure SHA.
Neither phase authorizes GPU, Quantum or replay execution. EV-021 remains
reserved for future R4 evidence and is not created here.

The Owner supplied the architecture review and independent Antigravity result
for `659413864082de6c120bfb6886828b0c361b8803`, tree
`0e59e3ea6cc4dadf6916c60a7f6e833ebc3e95b6`:

```text
ARCHITECTURE_REVIEW=PASS
ADVERSARIAL_QA=PASS
OWNER_ACCEPTANCE=YES
BLOCKER=0
MAJOR=0
MINOR=0
NOTE=0
VERDICT=PASS
READY_FOR_DEC025_ACCEPTANCE=YES
```

This records Owner-supplied independent review, not a new Cortex audit.
No PASS transfers to the closure or future harness SHA. The frozen manifest
is unchanged; its documentary-origin metadata is preserved, not rewritten
as an execution authorization. The following block preserves the original
preregistration identities and readiness at initial materialization:

```text
BASE_SHA=4afaf54306567c24c7d9f5d66e5ee55102c4aac0
BASE_TREE=fcac12ae1f9eabf3d854680342e55e76af5000b8
BASE_PARENT=843d932a8f63282d4dfc1e7f70b4a0a7b7b2b6b2
SCIENTIFIC_SHA=d2abd57e65bb7433eff81872a4f6510144d7c267
SCIENTIFIC_TREE=70b20d1f9cbc20d6cd77de4c25b3017a6413932c
REPLAY_VERSION=R4
DEC024_STATUS=accepted
READY_FOR_R4_IMPLEMENTATION=NO
READY_FOR_GPU_REPLAY=NO
```

The documentary commit is not the scientific execution SHA. Any future
execution requires an exact scientific HEAD/tree, clean checkout, validated
module origins and a separately authorized harness.

R4 inherits the contracts of
[DEC-016](cp05-c2b-cuda-equivalence-preregistration.md),
[DEC-018](cp05-c2c-r10a-r2-root-certification.md),
[DEC-020](cp05-c2c-r10a-r3-value-grid-separation.md),
[DEC-021](cp05-c2c-r10a-r3-r1-required-quantities.md),
[DEC-022](cp05-c2c-r10a-targeted-gpu-replay-r2-preregistration.md),
[DEC-023](cp05-c2c-r10a-targeted-gpu-replay-r3-preregistration.md) and accepted
[DEC-024](cp05-c2c-r10a-mc-reference-exact-tie-adjudication.md).
All inherited scientific rules remain unchanged. The sole future behavioral
delta from R3 is DEC-024 MC equivalence adjudication: raw count disagreement
remains recorded, but a fully certified exact-tie crossing is not by itself
a blocking adjudicated-equivalence failure. No historical gate or result is
rewritten.

The question is directed CPU/CUDA software equivalence on frozen matched
samples and complete MC outers, under DEC-024. This is not a population
error-rate estimand, statistical calibration, power or performance experiment.

## Immutable R3 and evidence boundary

[EV-020](../evidence/cp05-c2c-r10a-r3-mc-comparison-cliff.md) remains
`validated_with_limits`: R9 Git provenance and DEC-024 reasoning were
independently checked, while R3 runtime/output/archive and its checksum remain
Owner/Quantum-reported; archive bytes were not independently retrieved or
hashed by Cortex/Antigravity. No new archive verification occurs here.

```text
R3_EXECUTION_STATUS=FAILED
R3_AUTHORIZATION_CONSUMED=YES
AUTO_RERUN=NO
AUTO_RESUME=NO
```

R4 is neither a resume nor a reinterpretation of R3. R3 remains failed under
its historical contract even if a future R4 passes accepted adjudication.
R9/R3 values may be used only as historical provenance, never as R4 acceptance
oracles.

## Immediate manifest source and exact workload preservation

The new artifact is
[cp05-c2c-r10a-targeted-gpu-replay-r4-identities.json](cp05-c2c-r10a-targeted-gpu-replay-r4-identities.json).
Its immediate source is the exact
[frozen R3 manifest](../../experiments/distribution_gof/cuda_calibration/targeted_replay_r3/frozen_identity_manifest.json)
Git blob at BASE_SHA, not a regenerated identity inventory or an EOL-transformed
working-tree serialization:

```text
SOURCE_GIT_COMMIT=4afaf54306567c24c7d9f5d66e5ee55102c4aac0
SOURCE_PATH=experiments/distribution_gof/cuda_calibration/targeted_replay_r3/frozen_identity_manifest.json
SOURCE_GIT_BLOB_ID=8aa2aa0d8a5f3c875a2a70e8f7b6d460c40713ba
SOURCE_GIT_BLOB_SHA256=98170a403081a326c5c76be19fdebaf520dfbe1ea341b4cfef657d5aa4b3dfe6
SOURCE_SIZE_BYTES=118053
SOURCE_LF=2969
SOURCE_CRLF=0
ORDERED_PAYLOAD_SHA256=3fe47fef69fc2dfac152dd48eac75841fe25a107b0da6194c06a5d14a7be1395
```

The source was materialized in harness commit
`9282d5b082c4977ee12a7c6e0a54d908ee7283ff`, from the documentary R3 manifest
in `bff34ba8062433713177871b5c9e8927a0359ec9`. The new
`r3_identity_source` records the immediate BASE blob and its byte properties;
existing `r2_identity_source` and `identity_source` chains are retained.

Only top-level `schema_version` (v4), `decision` (DEC-025),
`replay_version` (R4), `identity_semantics`, `historical_fields_policy`
and `execution_authorization` are updated, and `r3_identity_source` is added.
All other top-level values remain unchanged. The complete raw JSON array
values for both keys are copied byte-for-byte:

```text
persisted_failure_record_identities
mc_failed_outers
```

Both entire parsed arrays, including nested metadata and sequence order, must
be identical to R3 and R2. Compute the ordered semantic digest from exactly
these two keys using ASCII bytes of
`json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)`.
Only dictionary keys are sorted; array order is never sorted. Strict JSON
rejects duplicate keys and nonfinite values. Whole-manifest R3/R4 byte hashes
are not expected to match because version/provenance metadata changes.

## Workloads and unchanged scientific mathematics

```text
NAMESPACE=CP05-C2C
R_EQ=8
B_EQ=15
WORKLOAD_A_IDENTITIES=134
WORKLOAD_A_OBSERVED=10
WORKLOAD_A_BOOTSTRAP=124
WORKLOAD_B_OUTERS=12
WORKLOAD_B_ELIGIBLE_BOOTSTRAPS_PER_OUTER=15
WORKLOAD_B_RECORDS_PER_OUTER=16
WORKLOAD_B_TOTAL_RECORDS=192
```

A is the ordered frozen R9 persisted failure union C/F/S/H. B is the ordered
12 historical outers, each with its observed identity and the exact ordered
15 eligible bootstrap identities. Preserve all raw-index gaps and cross-workload
overlap; require uniqueness within A, across B outers and within each B bootstrap
list. Recompose identities from the stored cell/raw indices for validation,
without generating any samples in this documentary task.

Do not rediscover, reorder, replace, prune, enlarge or regenerate identities;
do not change seeds, raw-index gaps, eligibility or retry accounting.
Future CPU and CUDA must independently evaluate the same reconstructed arrays.
Historical missing sample digests or distribution evidence must not be invented.

No change to fitting, classification, distribution, value grid, certified
support, AD/CvM, RNG, eligibility, retry accounting or NB solver mathematics
is allowed. DEC-018 root certification remains intact. NB distribution values
remain on `0..max(sample)`; the statistic uses full `certified_support.indices`.
Required NB quantities remain exactly `pmf, logPMF, cdf, sf, logCDF, logSF`.
All DEC-016 fit/value/statistic tolerances and DEC-021 fail-closed completeness
checks remain unchanged. Any scientific change requires a new architectural
decision, not a harness workaround.

## Raw MC procedure remains unchanged

```text
MC_EXCEEDANCE(T_boot,T_obs) := T_boot >= T_obs
p_MC=(b+1)/(B+1)
B_EQ=15
alpha=0.05
reject := p_MC <= 0.05
```

CPU and CUDA calculate their own raw values separately. Prohibited:

```text
isclose tie
epsilon tie
ULP tie
rounding tie
quantized tie
shared CPU statistic for CUDA
corrected CUDA b
corrected CUDA p
```

`STATISTIC_RTOL` is not changed and is never a tie tolerance.

### Fixed B_EQ limitation: reject boundary is not informative

```text
B_EQ=15
alpha=0.05
p_min=(0+1)/(15+1)=0.0625
p_min > alpha
raw_reject_cpu=false
raw_reject_cuda=false
REJECT_DECISION_GATE_PRESERVED=YES
REJECT_BOUNDARY_EVIDENCE_AT_ALPHA_0_05=NO
```

For every evaluable MC outer, both raw reject decisions must be false because
even the smallest plus-one p-value exceeds alpha. The exact reject-agreement
gate is retained, but this replay cannot validate decision behavior near
`alpha=0.05`. This does not invalidate directed equivalence evidence on
statistics, individual indicators, raw counts and exact-tie adjudication.
Neither B_EQ nor alpha changes. TYPE_I_CALIBRATION, POWER,
PRODUCTION_READINESS and FULL_C2C_EQUIVALENCE remain out of scope.

## Sole new semantics: future DEC-024 adjudication

For every bootstrap `j`, retain:

```text
T_cpu_boot
T_cuda_boot
T_cpu_obs
T_cuda_obs
cpu_exceedance
cuda_exceedance
```

The canonical predicate is exactly:

```text
CPU_REFERENCE_EXACT_TIE(j) :=
    finite(T_cpu_boot[j])
    AND finite(T_cpu_obs)
    AND T_cpu_boot[j] == T_cpu_obs
```

A crossing must meet **all fifteen DEC-024 prerequisites**: exact finite CPU
tie; unequal raw indicators; observed classification, fit, distribution-value
and statistic gates PASS; bootstrap classification, fit, distribution-value
and statistic gates PASS; all four CPU/CUDA statistics finite; exact frozen
bootstrap identity; expected frozen/reconstructed sample digest and seed
identity; no structural failure reason; no CUDA non-convergence/failure.
Missing or unverified evidence cannot certify a crossing.

Certification direction is necessarily:

```text
cpu_exceedance=true
cuda_exceedance=false
```

Any `cpu_exceedance=false, cuda_exceedance=true` is FAIL.
Unequal CPU statistics are not ties even for `nextafter`, one ULP, a difference
below statistic tolerance or `isclose` returning true. Individual provenance
is required; a count allowance alone never certifies.

## Signed accounting and separate raw evidence

Always persist raw results without overwriting:

```text
raw_b_cpu
raw_b_cuda
raw_p_cpu
raw_p_cuda
raw_reject_cpu
raw_reject_cuda

indicator_mismatch_count
certified_reference_exact_tie_crossing_count
unexplained_indicator_mismatch_count
```

For an outer with `raw_b_cpu != raw_b_cuda`, explained equivalence requires
all individual mismatches certified under DEC-024 and all the following:

```text
indicator_mismatch_count
==
certified_reference_exact_tie_crossing_count

unexplained_indicator_mismatch_count == 0

raw_b_cpu - raw_b_cuda
==
certified_reference_exact_tie_crossing_count

raw_reject_cpu == raw_reject_cuda
```

The count difference is **signed**, not merely absolute: every certifiable
crossing is CPU true / CUDA false. Opposite-direction mismatches cannot cancel
certified crossings. If `raw_b_cpu == raw_b_cuda`, any individual indicator
mismatch is still FAIL; count equality cannot conceal cancellation.
Neither a bound `abs(b_cpu-b_cuda) <= N` nor net count difference without
individual evidence is an acceptance rule.

## Individual, structural, per-outer and global gates

The inherited individual counters must remain zero:

```text
CLASSIFICATION_MISMATCH=0
CUDA_NONCONVERGENCE=0
FIT_GATE_FAILURE=0
DISTRIBUTION_VALUE_GATE_FAILURE=0
STATISTIC_GATE_FAILURE=0
UNEXPLAINED_DISCREPANCY=0
MC_BOOTSTRAP_IDENTITY_MISMATCH=0
```

Both structural item-count and distinct-record-count views must be zero for:

```text
EVALUATION_POINT_IDENTITY_MISMATCH
DISTRIBUTION_VALUE_LENGTH_MISMATCH
MISSING_CPU_DISTRIBUTION_QUANTITY
MISSING_CUDA_DISTRIBUTION_QUANTITY
MISSING_BOTH_DISTRIBUTION_QUANTITY
UNEXPECTED_DISTRIBUTION_QUANTITY
```

R4 must persist an explicit per-outer surface:

```text
MC_RAW_EXCEEDANCE_COUNT_MATCH
MC_REFERENCE_EXACT_TIE_CROSSING_COUNT
MC_UNEXPLAINED_INDICATOR_MISMATCH
MC_REJECT_DECISION_MATCH
MC_EQUIVALENCE_ADJUDICATED
```

Global PASS requires complete traversal, all inherited gates and:

```text
WORKLOAD_A_RECORDS_EXECUTED=134
WORKLOAD_B_RECORDS_EXECUTED=192
MC_OUTERS_RECONSTRUCTED=12
MC_UNEXPLAINED_INDICATOR_MISMATCH=0
MC_REJECT_DECISION_MISMATCH=0
MC_EQUIVALENCE_ADJUDICATED=true
```

All 12 outers must individually meet adjudicated equivalence and exact reject
agreement. No partial traversal may pass.
`MC_EXCEEDANCE_COUNT_MISMATCH` can no longer act alone as a blocking boolean:
a fully certified raw mismatch is permitted only under accepted DEC-024.
Preserve the observed raw mismatch count separately, for example as
`MC_RAW_EXCEEDANCE_COUNT_MISMATCH_OUTERS`; never artificially set it to zero.
Unexplained failures remain failures, not certified crossings.

## Preregistered adversarial cases for future implementation

These are required future tests, not tests implemented or run in this task.
PASS paths below still require every other individual, provenance and accounting
gate; an exact tie by itself is not sufficient.

1. Exact CPU tie + CUDA false: potentially certifiable.
2. `nextafter` below: NOT certifiable.
3. `nextafter` above: NOT certifiable.
4. One ULP below: NOT certifiable.
5. One ULP above: NOT certifiable.
6. Unequal CPU values with delta below `STATISTIC_RTOL`: NOT certifiable.
7. `isclose` true for non-equal CPU values: NOT certifiable.
8. CPU false / CUDA true: FAIL.
9. Equal raw b with canceling indicator mismatches: FAIL.
10. Three mismatches with net raw count difference 1: FAIL.
11. Missing observed gate: FAIL.
12. Missing bootstrap gate: FAIL.
13. Identity mismatch: FAIL.
14. Seed mismatch: FAIL.
15. Sample digest mismatch: FAIL.
16. Structural failure: FAIL.
17. CUDA non-convergence: FAIL.
18. Nonfinite statistic: FAIL.
19. Unequal CPU/CUDA reject decisions: FAIL.
20. Zero indicator mismatches + raw equality: PASS path.
21. Multiple exact CPU ties, all CPU true / CUDA false with exact accounting:
    PASS path.

The existing `mc_exact_tie` and `mc_near_comparison_cliff`
[fixtures](../../experiments/distribution_gof/cuda_calibration/cp05_c2b_adversarial_fixtures.json)
and their [evaluator](../../experiments/distribution_gof/cuda_calibration/a2_artifacts.py)
are not modified.

## Strong preflight inherited from R3

Retain the entire DEC-023/R3 readiness contract and ordering:

```text
validate_static
-> load manifest
-> exact scientific repo HEAD/tree/clean preflight
-> validate requested fresh/safe output path
-> load frozen runtime
-> module-origin verification
-> strong environment preflight
-> only if PASS create scientific output
-> provenance
-> authorization boundary at first scientific identity
-> Workload A/B
```

Before output creation or authorization consumption, require:

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

RawKernel must use NVRTC, a fresh temporary cache outside scientific output
and checkout, unique source, actual compilation, actual launch and a verified
deterministic float64 numerical result after synchronization. A prior cached
binary cannot stand in for compilation. Remove the temporary cache and restore
temporary cache environment overrides, including on failure. No scientific
RNG or experiment fixture is used for smoke checks.

Retain effective `CUDA_WHEEL_ROOT`, `CUDA_PATH`, `LD_LIBRARY_PATH`,
dependency resolution/load results, CuPy/runtime/driver/NVRTC versions, GPU
name, each individual preflight result and module origins in provenance.
Do not hardcode the Owner's home directory as a universal algorithm requirement.
No installation, loader repair or global environment modification is authorized.

Any failed preflight returns nonzero without A/B evaluation, checkpoints,
scientific counter increments or scientific output:

```text
EXECUTION_NOT_STARTED=YES
AUTHORIZATION_NOT_CONSUMED=YES
OUTPUT_NOT_CREATED=YES
AUTO_RERUN=NO
AUTO_RESUME=NO
```

Do not classify readiness failure as CPU/CUDA discrepancy.

## Authorization, freshness and stopping

A future R4 execution requires separate explicit Owner authorization.
DEC-025 does not grant it. After preflight PASS and output/provenance creation,
authorization is still unconsumed and execution not started.
`AUTHORIZATION_CONSUMED` changes to `YES` only immediately before evaluating
the first scientific identity; then `EXECUTION_STARTED=YES`.
Keep distinct persistent states for preflight failure, post-preflight/pre-identity
failure, scientific failure and complete PASS. Pre-identity failure must not
increment `UNEXPLAINED_DISCREPANCY`.

```text
AUTO_RERUN=NO
AUTO_RESUME=NO
```

Any scientific failure preserves available evidence, partial records/counters,
context and digests, then STOP and return to architecture.

R4 requires a fresh output directory, fresh authorization, fresh traversal,
fresh Workload A and fresh Workload B. Prohibited:

```text
R3 output reuse
R3 checkpoint reuse
R3 partial-record reuse
R3 authorization reuse
R3 resume
```

R3 is historical provenance only. No failed run is repaired into a pass and
no tolerances, identity selection or acceptance rules are adjusted after results.

## Claim boundary and excluded scope

A future R4 PASS may assert exclusively:

> CPU/CUDA targeted equivalence passed on the frozen R10-A workload,
> with any raw MC count differences limited to certified canonical
> CPU-reference exact-tie crossings and with identical reject decisions.

It cannot assert:

```text
RAW_MC_PVALUE_BIT_IDENTITY
GPU_NATIVE_MC_PVALUE_IDENTITY
FULL_C2C_EQUIVALENCE
FULL_1152_PASS
TYPE_I_CALIBRATION
POWER
PERFORMANCE
PRODUCTION_READINESS
CP05_D_READINESS
```

Raw CPU/GPU p-value identity remains unresolved debt if standalone GPU production
requires it. Adjudicated targeted agreement does not satisfy that requirement.

```text
FULL_PRIMARY_1152=NOT_AUTHORIZED
FULL_ADVERSARIAL_14_EXECUTION=NOT_AUTHORIZED
CP05_D=NOT_AUTHORIZED
HOLDOUT_ACCESS=NOT_AUTHORIZED
```

## Documentary validation, alternatives and handoff

Only four paths may change: this decision, its R4 identities JSON,
`knowledge/decisions/README.md` and `knowledge/registry.json`.
DEC-025 is the only new registry ID; EV-021 remains reserved but uncreated.
Do not edit accepted DEC-024, any inherited decision, historical evidence,
R3 harness/manifest, fixtures, tests or scientific code.

Validation is standard-library/documentary only:

1. Verify exact base SHA/tree/parent, clean worktree and DEC-025/EV-021 free
   before materialization; verify DEC-024 accepted.
2. Run `python -B knowledge/tools/validate_registry.py`; strictly parse registry
   and manifests, rejecting duplicate keys/nonfinite values.
3. Read source manifests from exact Git blobs. Compare entire R3/R4 and R2/R4
   ordered arrays, raw array spans and payload digest; verify counts,
   uniqueness, identity recomposition, raw-inner gaps, scientific SHA/tree and
   retained provenance metadata.
4. Verify links, unchanged prior registry records, frozen decisions/harnesses/
   fixtures/scientific code and exact four-file allowlist.
5. Run working-tree and staged `git diff --check`; verify one local commit and
   clean worktree. No GPU or scientific tests/samples are run.

Reusing R3 authorization/results, modifying workload/science, introducing
near-tie thresholds or correcting CUDA raw results are rejected alternatives.
R4 instead preserves the scientific question and records DEC-024 adjudication
with exact signed accounting. Missing identities, certification prerequisites
or authority require STOP and architectural review, not automatic repair.

One local commit is authorized: `preregister R4 exact-tie adjudicated replay`.
No push, PR, merge, main modification, Quantum, GPU/replay, full campaign,
CP05-D or holdout access. This accepted documentary contract is not a harness
implementation or equivalence result. Next role: ChatGPT architecture.
The Owner's subsequent implementation authorization is limited to the separate
Phase 2 harness commit; GPU preflight and replay remain unauthorized.
