# DEC-027 — CP05-C2C R11 Boundary-Sensitive Decision Equivalence Preregistration

- Status: `accepted`.
- Date recorded: 2026-10-01.
- Owner: Project Owner — Ehud Bottaro.
- Architecture: ChatGPT — statistical/software architecture.
- Documentary implementation: Cortex — implementation engineering.
- Independent QA: Antigravity — adversarial statistical/software QA.
- Related decisions/evidence: `DEC-014`, `DEC-016`, `DEC-024`, `DEC-025`,
  `DEC-026`, `EV-021`.
- Supersedes: none.

## Purpose and limited scope

R11 addresses the specific scientific limitation preserved by `EV-021`:

```text
B_EQ=15
alpha=0.05
p_min=(0+1)/(15+1)=0.0625
p_min > alpha
```

R10-A therefore could not provide informative evidence about the
reject/non-reject decision boundary at `alpha=0.05`. R11 is not a full
CPU/CUDA equivalence campaign. It is a limited boundary-sensitive research
equivalence experiment under `DEC-026`.

## Research and production boundary

```text
EXECUTION_CLASS=research-reference
CANONICAL_REFERENCE=NumPy/SciPy
CUDA_ROLE=research instrumentation only

CUDA_IS_PRODUCTION_BACKEND=NO
CUDA_IS_PUBLIC_API=NO
PRODUCTION_TESTS_REQUIRE_GPU=NO
CPU_REFERENCE_REMAINS_CANONICAL=YES
```

R11 does not change the production API, production dependencies, production
scientific semantics or production tests.

### Frozen scientific-code identity

```text
R11_SCIENTIFIC_BASE_SHA=d7412911dcf8402f01a502fe0d756813ec10bbee
R11_SCIENTIFIC_BASE_TREE=421c07a3466a689e8c49da5c1cfe25fb946a0a5b
PREEXISTING_SCIENTIFIC_CODE_MUTATION=PROHIBITED
```

The future R11 implementation may add isolated R11 builder, harness, tests and
documentary files. It must not alter, relative to
`R11_SCIENTIFIC_BASE_SHA`, any pre-existing scientific implementation used by
the CPU-reference or CUDA-candidate evaluation path. This prohibition includes
existing fitting, RNG/seed semantics, generation, classification, distribution
values, statistic evaluation, certified support and CUDA numerical
implementation.

If implementation discovers that any such pre-existing scientific file must
change:

```text
STOP
R11_PREREGISTRATION_REQUIRES_REVISION=YES
GPU_EXECUTION=PROHIBITED
```

A new architecture decision and review are required before continuing. The
future evidence artifact must record all of:

```text
R11_SCIENTIFIC_BASE_SHA/TREE
R11_BUILDER_SHA/TREE
R11_HARNESS_SHA/TREE
```

Implementation identity must not be confused with scientific-reference
identity.

## Frozen R11 Monte Carlo contract

```text
ALPHA=0.05
B_R11=199
P_MC=(b+1)/(B+1)
MC_EXCEEDANCE(T_boot,T_obs) := T_boot >= T_obs
reject := p_MC <= 0.05
```

The mathematical decision surface is:

| `b` | `p_MC` | `reject` |
|---:|---:|:---|
| 8 | `9/200=0.045` | `True` |
| 9 | `10/200=0.050` | `True` |
| 10 | `11/200=0.055` | `False` |
| 11 | `12/200=0.060` | `False` |

```text
MC_REJECT_BOUNDARY_MATHEMATICALLY_INFORMATIVE=YES
```

This means only that `B=199` permits observations on both sides of the
`alpha=.05` decision threshold. It does not claim prospectively that the real
R11 scientific outers will actually fall near that boundary.

## Frozen source outer surface

The R11 scientific surface is exactly the 12 historical Monte Carlo outers
already frozen in the canonical R4 manifest:

```text
SOURCE_R4_MANIFEST=experiments/distribution_gof/cuda_calibration/targeted_replay_r4/frozen_identity_manifest.json
SOURCE_R4_MANIFEST_SHA256=3f300ff5a678e3eac4956fe10514bf7634ec0ea824f0bfef56f4688a86adb286
SOURCE_ARRAY=mc_failed_outers
R11_SOURCE_OUTERS=12
R11_RECORDS_PER_OUTER=200
R11_TOTAL_MC_RECORDS=2400
```

The array order is normative. The set must not be rediscovered, reordered,
replaced, pruned or enlarged.

To bind this decision to that historical surface without generating any
scientific sample, the documentary source projection is defined as the array
obtained by visiting `mc_failed_outers` in its stored order and retaining, for
each object, exactly these fields:

```text
outer_identity
cell_id
raw_outer_index
r9_b_cpu
r9_b_cuda
r9_p_cpu
r9_p_cuda
r9_reject_cpu
r9_reject_cuda
observed_identity
```

Serialize the resulting array as UTF-8 bytes using Python
`json.dumps(projection, ensure_ascii=False, sort_keys=True,
separators=(',', ':'), allow_nan=False)`. Dictionary keys are sorted; array
order is never sorted. The result frozen by this documentary materialization
is:

```text
R11_SOURCE_OUTER_PROJECTION_BYTES=4250
R11_SOURCE_OUTER_PROJECTION_SHA256=91d090100179f96e5d562a6d2019becd249af3b6b20405ac5cc041622fd7c8bd
```

The historical R9 fields in this projection are provenance binding only; they
are not R11 acceptance oracles.

## Future CPU-reference workload construction

DEC-027 defines, but does not execute, the future CPU-only construction. For
each frozen source outer, in frozen order:

1. Reconstruct the canonical observed sample using the existing canonical
   seed/identity semantics.
2. Fit the observed sample using the canonical NumPy/SciPy reference
   implementation.
3. Traverse `raw_inner_index = 0,1,2,...`.
4. Derive each seed with the existing canonical rule:

   ```text
   derive_seed(
       "CP05-C2C",
       cell_id,
       raw_outer_index,
       "inner_bootstrap",
       raw_inner_index
   )
   ```

5. Generate the bootstrap sample from the CPU-reference fitted parameters
   using the canonical NumPy generator semantics.
6. Materialize the exact scientific array and calculate SHA-256 over its
   canonical raw sample bytes,
   `np.ascontiguousarray(sample).tobytes(order="C")`.
7. Evaluate canonical eligibility using CPU `reference_fit(sample)`.
8. If eligible, preserve `identity`, `cell_id`, `raw_outer_index`,
   `raw_inner_index`, `seed_identity`, `sample_digest` and accepted
   ordinal/order.
9. For canonical `NB_NOT_ASSESSED` outcomes, preserve the attempt as
   `INELIGIBLE` and continue prospectively.
10. For any other CPU-reference construction failure: STOP, preserve evidence
    and do not substitute another rule.
11. Stop only after exactly 199 canonically eligible bootstrap identities have
    been obtained for that outer.

No part of that construction is performed by this documentary task.

## Retry cap

For Negative Binomial:

```text
R11_NB_RETRY_CAP=100*B_R11
R11_NB_RETRY_CAP=19900 raw attempts per outer
```

If any outer does not produce 199 canonically eligible bootstrap samples
before that cap:

```text
R11_REFERENCE_WORKLOAD_CONSTRUCTION=FAIL
GPU_EXECUTION=PROHIBITED
AUTO_RERUN=NO
AUTO_RESUME=NO
```

The cap cannot be increased after observing the result without a new
architecture decision and preregistration.

## R4 prefix invariant

R11 preserves the canonical namespace, outer identity, seed derivation, CPU
fitting semantics, generator semantics and eligibility semantics. Therefore,
the first 15 canonically eligible bootstrap identities reconstructed for each
R11 outer are expected to match the corresponding 15 frozen R4 bootstrap
identities exactly.

The R4 manifest is the historical oracle for identity, `raw_inner_index` and
ordering. The independently audited R4 run2 runtime evidence referenced by
`EV-021` is the historical oracle for seed identity and sample digest:

```text
R4_PREFIX_IDENTITY_ORACLE=experiments/distribution_gof/cuda_calibration/targeted_replay_r4/frozen_identity_manifest.json
R4_RUNTIME_ORACLE_ARCHIVE=cp05_c2c_r10a_targeted_r4_501d752_run2_PASS_EVIDENCE.tar.gz
R4_RUNTIME_ORACLE_ARCHIVE_SHA256=dd8de17dfa17ac54855f9823d053e6820a2a285aeb8467eec390aea51dcba6ad
R4_RUNTIME_CROSSINGS_REPORT=cp05_c2c_r10a_targeted_r4_501d752_run2_crossings.json
R4_RUNTIME_CROSSINGS_REPORT_SHA256=f7c35e51e0273eac73f9743010b0191cc3c33fac9dde88cdbc123200d86e10f7
R4_RUNTIME_WORKLOAD_B_RECORDS=workload_b_records.jsonl
R4_RUNTIME_WORKLOAD_B_RECORDS_SHA256=beea7dffa4a059de241703079e254a696a8ad7764cd9b6288079a84188ee5f37
```

The archive/crossings identity association above is corrected by DEC-028
against canonical EV-021. No other DEC-027 scientific contract term changes.

Before accepting the future R11 reference workload:

1. Verify the R4 manifest SHA-256.
2. Verify the runtime evidence archive SHA-256.
3. Extract and read `workload_b_records.jsonl`.
4. Verify its exact SHA-256.
5. Verify the historical observed record and first 15 bootstrap records for
   each of the 12 outers.

```text
R4_PREFIX_EXPECTED_COUNT=15
R4_PREFIX_IDENTITY_MATCH_REQUIRED=YES
R4_PREFIX_SEED_MATCH_REQUIRED=YES
R4_PREFIX_SAMPLE_DIGEST_MATCH_REQUIRED=YES
R4_OBSERVED_IDENTITY_MATCH_REQUIRED=YES
R4_OBSERVED_SEED_MATCH_REQUIRED=YES
R4_OBSERVED_SAMPLE_DIGEST_MATCH_REQUIRED=YES
```

Any prefix mismatch is FAIL and requires architectural review. It must not be
repaired by accepting the new sequence.

If the historical runtime oracle is unavailable or either runtime hash does
not match:

```text
R4_RUNTIME_ORACLE_VERIFIED=NO
R11_REFERENCE_WORKLOAD_ACCEPTED=NO
GPU_EXECUTION=PROHIBITED
```

The prefix gate must not be silently downgraded to identity-only.

```text
R4_RESULTS_REUSED=NO
R4_OUTPUT_REUSED=NO
R4_AUTHORIZATION_REUSED=NO
R4_CHECKPOINT_REUSED=NO
```

## Future frozen reference workload artifact

The future CPU-reference workload artifact must preserve at least:

- source R4 manifest SHA-256 and ordered source outer projection SHA-256;
- builder SHA and tree;
- canonical NumPy/SciPy reference SHA and tree;
- Python, NumPy and SciPy versions and environment metadata;
- for each outer: observed identity, observed seed identity and observed sample
  digest;
- for every attempted bootstrap: `raw_inner_index`, `seed_identity`,
  `sample_digest` and canonical eligibility status;
- for each accepted bootstrap: accepted order, identity, `raw_inner_index`,
  `seed_identity`, `dtype.str`, shape, C-contiguous raw sample payload bytes and
  `sample_digest`;
- attempt count, accepted count, retry-cap status, ordered workload payload
  SHA-256 and failure state if incomplete.

For every observed sample and every one of the 199 accepted bootstrap samples
per outer, preserve:

```text
identity
dtype.str
shape
C-contiguous raw sample bytes
SHA256(raw sample bytes)
```

Canonical digest bytes are exactly:

```text
np.ascontiguousarray(sample).tobytes(order="C")
```

Seeds and digests without the exact payload bytes are insufficient. The
workload artifact/archive must have its own SHA-256 and immutable ordered
manifest. Attempted-but-ineligible bootstraps continue to preserve at minimum
`raw_inner_index`, `seed_identity`, `sample_digest` and canonical eligibility
status; accepted and observed scientific records must preserve their exact
sample payload bytes.

```text
R11_SAMPLE_REGENERATION_DURING_EVALUATION=PROHIBITED
CUDA_SAMPLE_GENERATION=PROHIBITED
CPU_SAMPLE_REGENERATION_DURING_COMPARISON=PROHIBITED
```

Future CPU and CUDA scientific evaluation must deserialize and use the same
frozen sample payload. Before either numerical engine evaluates a record,
`SHA256(loaded canonical raw bytes)` must equal the frozen `sample_digest`.
Otherwise:

```text
R11_SAMPLE_IDENTITY_GATE=FAIL
NUMERICAL_EVALUATION=PROHIBITED
GPU_RUN_OR_RECORD=FAIL_CLOSED
```

CUDA must never independently regenerate a bootstrap from a seed. The seed is
provenance evidence, not the transport mechanism for the scientific sample
during CPU/CUDA comparison.

The future workload builder must be CPU-only and remain importable/testable
without CuPy/CUDA.

## GPU-independent selection

```text
POST_HOC_IDENTITY_SELECTION=NO
GPU_DEPENDENT_SELECTION=NO
CUDA_DEPENDENT_ELIGIBILITY=NO
```

CUDA may not choose which 199 samples enter the scientific workload. The
workload is frozen from CPU-reference semantics before CUDA scientific
evaluation begins.

## DEC-024 exact-tie semantics

DEC-024 remains normative. The canonical exceedance comparator is exactly
`T_boot >= T_obs`, and an exact CPU-reference tie means exactly
`T_cpu_boot == T_cpu_obs`.

The following are prohibited tie definitions: `isclose`, epsilon, rounding,
ULP allowance, relative tolerance, absolute tolerance and `nextafter`
proximity.

```text
FUZZY_TIE_RULE=NO
```

## Critical R11 decision-equivalence rule

```text
DISCREPANCY_EXPLAINED != SCIENTIFIC_DECISION_EQUIVALENCE
```

A DEC-024-certified exact-tie crossing may causally explain a raw CPU/CUDA
exceedance-count discrepancy. That explanation does not permit a scientific
decision mismatch. The critical example is:

```text
CPU:
b=10
p=11/200=0.055
reject=False

CUDA:
b=9
p=10/200=0.050
reject=True

EXACT_TIE_CERTIFIED=YES
DISCREPANCY_EXPLAINED=YES
SCIENTIFIC_DECISION_EQUIVALENCE=NO
R11_OUTER_PASS=NO
```

No adjudication may overwrite raw counts, raw p-values or raw reject
decisions.

## R11 outer acceptance contract

Prospectively:

```text
R11_OUTER_PASS :=
    all applicable classification gates PASS
    AND all applicable fit gates PASS
    AND all applicable distribution-value gates PASS
    AND all applicable statistic gates PASS
    AND MC_BOOTSTRAP_IDENTITY_MISMATCH=0
    AND MC_UNEXPLAINED_INDICATOR_MISMATCH=0
    AND every explained indicator mismatch satisfies DEC-024 exactly
    AND DEC-024 signed accounting is exact
    AND raw_reject_cpu == raw_reject_cuda
```

`raw_b_cpu == raw_b_cuda` is not mandatory. `raw_p_cpu == raw_p_cuda` is not
mandatory. Both CPU and CUDA raw counts, p-values and reject decisions must
always be persisted without correction or replacement.

## Global R11 PASS

A future scientific R11 PASS requires all of the following:

- all 12 source outers traversed completely;
- 199 eligible bootstraps evaluated per outer;
- 12/12 outers adjudicated;
- zero unexplained indicator mismatches;
- zero identity mismatches;
- zero structural failures;
- zero CUDA non-convergence/failure in applicable records;
- all applicable numerical gates PASS;
- raw reject decision agreement for every evaluable outer;
- all deterministic adversarial decision-boundary fixtures PASS;
- no incomplete traversal; and
- no post-hoc rule changes.

## Boundary fixture contract

The future deterministic logical fixtures include at least:

```text
BOUNDARY_FIXTURES_AGGREGATE_ONLY=PROHIBITED
```

For every fixture involving an indicator mismatch, exact tie, cancellation or
DEC-024 certification, the fixture must materialize a complete logical `B=199`
indicator surface and the per-bootstrap evidence required by the same R11
adjudication function used for the scientific outers. That function must derive
`b_cpu`, `b_cuda`, `p_cpu`, `p_cuda`, `reject_cpu` and `reject_cuda` from the
199 indicator records. Those aggregates must not be supplied as trusted
inputs.

Small direct unit tests of the mathematical `b -> p -> reject` mapping are
permitted in addition, but they are not sufficient to satisfy the R11
adversarial adjudication fixture gate.

1. `b_cpu=8`, `b_cuda=8`: both reject; PASS path.
2. `b_cpu=9`, `b_cuda=9`: `p=.05` exactly; both reject; PASS path.
3. `b_cpu=10`, `b_cuda=10`: both non-reject; PASS path.
4. `b_cpu=11`, `b_cuda=11`: both non-reject; PASS path.
5. DEC-024-certified crossing `CPU b=9`, `CUDA b=8`: both reject, decision
   preserved; potential PASS path if all other gates pass.
6. DEC-024-certified crossing `CPU b=10`, `CUDA b=9`: CPU non-reject, CUDA
   reject; mandatory FAIL.
7. CPU exceedance `False`, CUDA exceedance `True`: not certifiable; FAIL.
8. `nextafter` neighbor: NOT exact tie.
9. One-ULP non-equal CPU values: NOT exact tie.
10. `isclose=True` but exact equality `False`: NOT exact tie.
11. Equal raw `b` with cancelling individual indicator mismatches: FAIL.
12. Unexplained individual indicator mismatch with unchanged reject decision:
    FAIL.

These deterministic fixtures validate decision logic. They are not evidence
that the 12 real scientific outers naturally occupy the `alpha=.05`
neighborhood.

## Boundary observation metrics

Keep separate:

```text
MC_REJECT_BOUNDARY_MATHEMATICALLY_INFORMATIVE=YES
```

from the future observed evidence:

```text
SCIENTIFIC_BOUNDARY_NEIGHBORHOOD_OBSERVED :=
count of real R11 CPU-reference outers whose raw b is in {8,9,10,11}
```

The actual per-outer `b` and `p` values must also be preserved. Zero observed
near-boundary outers is not a failed logical fixture test. Passing fixtures is
not empirical evidence that the real workload sampled the boundary.

## Maximum permitted claim

If a future R11 execution passes, its maximum permitted claim is:

> CPU/CUDA decision equivalence at alpha=0.05 passed on the preregistered R11
> boundary-sensitive scope consisting of the 12 frozen historical R10-A Monte
> Carlo outers evaluated with 199 CPU-reference-generated eligible bootstrap
> samples per outer, plus deterministic decision-boundary adversarial fixtures.
>
> Any raw count discrepancy was either absent or fully DEC-024-certified, and
> no accepted discrepancy changed the scientific reject decision.

## Claims explicitly not established

R11 cannot by itself establish:

```text
FULL_C2C_EQUIVALENCE
FULL_1152_PASS
TYPE_I_CALIBRATION
TYPE_I_VALIDATED
POWER_VALIDATED
PERFORMANCE_CERTIFICATION
CUDA_PRODUCTION_BACKEND
CUDA_PRODUCTION_READINESS
PRODUCTION_READINESS
CP05_D_READINESS
HOLDOUT_VALIDATION
```

```text
FULL_1152=OUT_OF_SCOPE
TYPE_I_CALIBRATION=OUT_OF_SCOPE
POWER=OUT_OF_SCOPE
PERFORMANCE_CERTIFICATION=OUT_OF_SCOPE
PRODUCTION_READINESS=NOT_APPLICABLE
CP05_D=NOT_AUTHORIZED
HOLDOUT_ACCESS=NOT_AUTHORIZED
```

## Historical state preservation

DEC-027 does not rewrite `DEC-014`..`DEC-026` or historical evidence. R4 and
`EV-021` remain immutable historical evidence.

```text
R10A_TARGETED_EQUIVALENCE=COMPLETE
R10A_EVIDENCE_STATUS=validated_with_limits
FULL_C2C_EQUIVALENCE=NOT_ESTABLISHED
FULL_1152_PERFORMED=NO
CP05_D=NOT_STARTED
HOLDOUT_ACCESSED=NO
```

## Failure and stop policy

```text
AUTO_RERUN=NO
AUTO_RESUME=NO
```

Any future failure must preserve available evidence and stop. Automatic
repair, retry, resume, tolerance widening, identity replacement, sample
pruning, boundary redefinition, retry-cap increase, post-hoc fixture change or
post-hoc claim expansion is prohibited.

## Current authorization boundary

This task authorizes only documentary materialization of DEC-027.

```text
R11_REFERENCE_WORKLOAD_BUILDER=FUTURE_IMPLEMENTATION
R11_REFERENCE_WORKLOAD_CONSTRUCTION=NOT_AUTHORIZED
R11_HARNESS_IMPLEMENTATION=NOT_AUTHORIZED
R11_GPU_EXECUTION=NOT_AUTHORIZED

FULL_1152=NOT_AUTHORIZED
CP05_D=NOT_AUTHORIZED
HOLDOUT_ACCESS=NOT_AUTHORIZED
```

It does not authorize scientific sample generation, workload construction,
builder or harness implementation, NumPy scientific R11 generation, GPU/CUDA
execution, the full 1152 campaign, CP05-D or holdout access. Every later phase
requires its own explicit authorization.
