# EV-021 — CP05-C2C R10-A R4 targeted CPU/CUDA equivalence runtime evidence

- Status: `validated_with_limits`.
- Date recorded: 2026-09-28.
- Repository: `udibott-011235/pyMagicStats`.
- Documentary branch: `docs/cp05-c2c-r10a-r4-runtime-evidence`.
- Owner role: implementation-engineering — Cortex, documentary materialization only.
- Reviewers: statistical-software-architecture — ChatGPT;
  adversarial-statistical-qa — Antigravity; decision-owner — Ehud Bottaro.
- Supersedes: none.

## Authority, provenance and canonical identities

The Project Owner authorized only local documentary materialization of this
record and its registry entry. The runtime evidence was produced on Quantum,
then transported and verified bit-for-bit by independent QA, Antigravity.
This record transcribes that independently audited result supplied in the
Owner's materialization instruction. Cortex does not perform or claim an
independent statistical audit, archive retrieval, rehashing of the transported
runtime artifacts, GPU execution or replay in this task. The hashes below are
Antigravity-verified values, not merely unverified Owner runtime reports.

The audit applies to the frozen runtime and harness identities below. It does
not automatically certify the new documentary candidate, authorize a next
action, or close full CP05-C2C.

```text
HARNESS_SHA=501d752fc2542105b2dc3c95dd1607f59b90d94c
HARNESS_TREE=6c42e7d25f00e612f4c30a3d88e823b0a50ca933
SCIENTIFIC_SHA=d2abd57e65bb7433eff81872a4f6510144d7c267
SCIENTIFIC_TREE=70b20d1f9cbc20d6cd77de4c25b3017a6413932c
PREREGISTRATION_SHA=d481125e6981d43bbb529c8ed102c034438cbddd
MANIFEST_SHA256=3f300ff5a678e3eac4956fe10514bf7634ec0ea824f0bfef56f4688a86adb286
ORDERED_PAYLOAD_SHA256=3fe47fef69fc2dfac152dd48eac75841fe25a107b0da6194c06a5d14a7be1395
```

The harness commit is `implement R4 exact-tie adjudication harness`, descending
from the accepted [DEC-025 closure](../decisions/cp05-c2c-r10a-targeted-gpu-replay-r4-preregistration.md).
[DEC-024](../decisions/cp05-c2c-r10a-mc-reference-exact-tie-adjudication.md)
defines the canonical CPU-reference exact-tie adjudication policy. The
[documentary frozen identities](../decisions/cp05-c2c-r10a-targeted-gpu-replay-r4-identities.json),
[harness manifest](../../experiments/distribution_gof/cuda_calibration/targeted_replay_r4/frozen_identity_manifest.json)
and [R4 harness](../../experiments/distribution_gof/cuda_calibration/targeted_replay_r4/targeted_gpu_replay.py)
remain unchanged by this record.

## Frozen R4 run2 artifacts and independent byte verification

Canonical archive:

```text
cp05_c2c_r10a_targeted_r4_501d752_run2_PASS_EVIDENCE.tar.gz
SHA256=dd8de17dfa17ac54855f9823d053e6820a2a285aeb8467eec390aea51dcba6ad
```

Extracted crossings report:

```text
cp05_c2c_r10a_targeted_r4_501d752_run2_crossings.json
SHA256=f7c35e51e0273eac73f9743010b0191cc3c33fac9dde88cdbc123200d86e10f7
```

These external runtime artifacts are referenced by name and SHA256 only;
neither artifact nor any runtime binary is added to the repository here.
Antigravity physically verified the transported archive, recalculated the
eight core file hashes below, and checked all 12 internally enumerated files
in `digests.json` against their physically extracted bytes.

```text
authorization.json
dec31b0d8c1fd6e9ab5ed897a2f639f20d0db72e22c62a2605b48eb91d5b2a1b

environment.json
969e409d774315016bb38a55fb4a643419d2710dab741f3641d61b0bdeec6984

execution_manifest.json
8d0d18dffb12edbc8e32ac9e8b70f927a8ff1c91d8cdb284640b4ac4d45d9ce6

mc_results.json
4d25e9a1094c195e106966f2df2439e46fbc9d5f8de526c5ada3337c72d05782

summary.json
d5b336a33860db2b95c6dab8295b712f711afc798e0626627db3e3a8f5f9f59c

workload_a_records.jsonl
718a8365e9850fc84602fa683a2145de60872c1083a57b0f3b7475b75c857712

workload_b_records.jsonl
beea7dffa4a059de241703079e254a696a8ad7764cd9b6288079a84188ee5f37

digests.json
33ccc5b2337914ab250e2c0cdafdfc6c3c2f8963537a810e0d81e18bb585271b
```

## Audited runtime result and applicable gates

```text
R4_RUN2_RUNTIME=PASS
R4_RUN2_ADVERSARIAL_QA=PASS
TARGETED_REPLAY_PASS=YES
execution_state=COMPLETE
WORKLOAD_A_RECORDS_EXECUTED=134
WORKLOAD_B_RECORDS_EXECUTED=192
MC_OUTERS_RECONSTRUCTED=12
MC_OUTERS_ADJUDICATED=12
CLASSIFICATION_MISMATCH=0
CUDA_NONCONVERGENCE=0
FIT_GATE_FAILURE=0
DISTRIBUTION_VALUE_GATE_FAILURE=0
STATISTIC_GATE_FAILURE=0
MC_BOOTSTRAP_IDENTITY_MISMATCH=0
MC_UNEXPLAINED_INDICATOR_MISMATCH=0
MC_REJECT_DECISION_MISMATCH=0
UNEXPLAINED_DISCREPANCY=0
```

Workload A is `134 = 10 observed + 124 bootstrap`. Antigravity verified one
canonically ineligible identity:

```text
negative_binomial|r=0.25,p=0.1|n=100|CVM|composite|raw_outer=1
CPU_CLASSIFICATION=VARIANCE_NOT_GREATER_THAN_MEAN
CUDA_CLASSIFICATION=VARIANCE_NOT_GREATER_THAN_MEAN
classification_gate_pass=True
```

Its downstream gates are not applicable/null; this matched canonical
ineligibility is not a failure. The other 133 A records passed their applicable
gates. Workload B is `192 = 12 × (1 observed + 15 bootstrap)`; all B records
passed their applicable gates. “All scientific gates pass” below means all
applicable gates, not that the ineligible A identity's null gates became true.

## DEC-024 exact-tie evidence and signed accounting

```text
MC_RAW_EXCEEDANCE_COUNT_MISMATCH_OUTERS=2
MC_REFERENCE_EXACT_TIE_CROSSINGS=3
MC_UNEXPLAINED_INDICATOR_MISMATCH=0
MC_REJECT_DECISION_MISMATCH=0
```

Antigravity independently reconstructed all 12 outers using only:

```text
CPU indicator  := T_cpu_boot >= T_cpu_obs
CUDA indicator := T_cuda_boot >= T_cuda_obs
```

No `isclose`, rounding, epsilon, ULP tolerance or fuzzy comparator was used.
Raw scientific aggregates remain raw; adjudicated equivalence does not imply
raw MC count identity or raw MC p-value identity.

### Affected outer 1

```text
negative_binomial|r=1,p=0.9|n=250|AD|composite|raw_outer=7
raw_b_cpu=12
raw_b_cuda=11
signed_difference=+1

raw_inner=28
T_cpu_obs = 0.0004924344188814927
T_cpu_boot = 0.0004924344188814927
T_cuda_obs = 0.0004924344188815376
T_cuda_boot = 0.000492434418881383
CPU_EXCEEDANCE=True
CUDA_EXCEEDANCE=False
```

One canonical CPU-reference exact-tie crossing was certified.

### Affected outer 2

```text
negative_binomial|r=1,p=0.9|n=50|CVM|composite|raw_outer=4
raw_b_cpu=12
raw_b_cuda=10
signed_difference=+2

raw_inner=7
raw_inner=26
```

For both certified crossings:

```text
T_cpu_boot == T_cpu_obs
CPU_EXCEEDANCE=True
CUDA_EXCEEDANCE=False
```

CUDA is below its own observed statistic. No unprovided numerical values
are inferred for these two crossings.

### Other outers and per-outer accounting

The other 10 outers had `raw_b_cpu == raw_b_cuda`, zero certified crossings
and zero indicator mismatches. Antigravity verified both equalities for each
outer individually, not merely their aggregate totals:

```text
indicator_mismatch_count
==
certified_reference_exact_tie_crossing_count

raw_b_cpu - raw_b_cuda
==
certified_reference_exact_tie_crossing_count
```

## B_EQ=15 limitation and maximum permitted claim

```text
B_EQ=15
alpha=0.05
p_min=(0+1)/(15+1)=1/16=0.0625
p_min > alpha
MC_REJECT_BOUNDARY_INFORMATIVE=NO
```

With B=15, all evaluable reject decisions are necessarily `False`.
Consequently `MC_REJECT_DECISION_MISMATCH=0` is a necessary replay consistency
condition but provides no informative evidence about the rejection boundary
at alpha=0.05. Neither B nor alpha is changed.

The maximum permitted scientific claim is:

> CPU/CUDA targeted equivalence passed on the frozen R10-A workload,
> with raw Monte Carlo count differences limited to certified canonical
> CPU-reference exact-tie crossings and with identical reject decisions.

```text
R10A_TARGETED_EQUIVALENCE=COMPLETE
R10A_EVIDENCE_STATUS=validated_with_limits
```

This closes only R10-A targeted equivalence, not full CP05-C2C.
EV-021 does **not** demonstrate:

- raw MC p-value identity;
- raw MC count identity;
- full CP05-C2C equivalence;
- full 1152 equivalence campaign;
- alpha=.05 reject-boundary validation;
- type-I calibration;
- power validation;
- performance certification;
- production readiness;
- CP05-D validation;
- holdout validation.

```text
FULL_1152_PERFORMED=NO
CP05_D_ACCESSED=NO
HOLDOUT_ACCESSED=NO
```

## Historical run1: operational antecedent only

```text
RUN1_INTERRUPTED_EVIDENCE_SHA256=5ec5de03bbaef9fcd2d1b39fbb46a33faa16e53f40c407bdea620ed362e5f799
```

Run1 consumed its authorization and started execution. It persisted 30/134
Workload-A records and zero Workload-B records, with no summary. No scientific
gate failure was established: termination was abrupt and its root cause
remains unclassified. Run1 is not characterized as a statistical FAIL and is
not evidence used to complete run2. Run1 and run2 data are not combined.

```text
RUN1_REUSED=NO
RESUME_PERFORMED=NO
```

## Audited run2 operational supervisor

```text
HARNESS_INVOCATION_COUNT=1
PYTHON_EXIT_CODE=0
PYTHON_SIGNAL_NUMBER=NONE
WRAPPER_EXIT_CODE=0
WRAPPER_COMPLETE=YES
RERUN_PERFORMED=NO
RESUME_PERFORMED=NO
FULL_1152_PERFORMED=NO
CP05_D_ACCESSED=NO
HOLDOUT_ACCESSED=NO
RUN1_REUSED=NO
```

Operational supervision changed neither science, harness, manifest nor payload.
These are audited run2 operational facts, not operations executed by Cortex
during this documentary task.

## Independent adversarial QA result

Antigravity physically verified the transported archive and reported:

```text
R4_RUN2_ARCHIVE_SHA256_VERIFIED=YES
CORE_FILE_HASHES_VERIFIED=YES
HARNESS_IDENTITY_VERIFIED=YES
SCIENTIFIC_IDENTITY_VERIFIED=YES
FROZEN_IDENTITY_VERIFIED=YES
WORKLOAD_A_COMPLETE=YES
WORKLOAD_B_COMPLETE=YES
ALL_SCIENTIFIC_GATES_PASS=YES
MC_OUTERS_RECONSTRUCTED=12
MC_OUTERS_ADJUDICATED=12
SIGNED_ACCOUNTING_VERIFIED=YES
CANCELLATION_FALSE_PASS_POSSIBLE=NO
NEAR_TIE_CERTIFIABLE=NO
OPPOSITE_DIRECTION_CERTIFIABLE=NO
MC_REJECT_BOUNDARY_INFORMATIVE=NO
SINGLE_INVOCATION_VERIFIED=YES
BLOCKER=0
MAJOR=0
MINOR=0
NOTE=0
AUDIT_VERDICT=PASS
READY_FOR_EV021_MATERIALIZATION=YES
```

This audit belongs to Antigravity, not Cortex. Cortex's work is limited to
documentary transcription and Knowledge Base/registry validation on the
authorized local candidate. No new decision is created, no execution is
authorized, and publication, PR, merge and modification of main remain outside
this task. The next role is ChatGPT for architectural review of the documentary
candidate; no next action is authorized here.
