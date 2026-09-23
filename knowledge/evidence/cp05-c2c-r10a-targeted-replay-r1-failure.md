# EV-017 — DEC-019 targeted replay R1 failure

**Status:** `proposed`

**Recorded by:** Cortex — Implementation Engineering

**Source:** external execution reported by Project Owner Ehud Bottaro

**Date recorded:** 2026-09-23

## Owner-reported observation, not independent archive verification

The Owner reports one manual DEC-019 execution on Quantum. Cortex did not run
that replay and has not inspected or independently verified the external tar
archive. Its supplied checksum is preserved as a reported identity only.

```text
DEC019_EXECUTION_STATUS=FAIL
FAIL_FAST=PASS
FIRST_FAILURE_IDENTITY=negative_binomial|r=0.25,p=0.1|n=20|AD|composite|raw_outer=2|raw_inner=7
FIRST_FAILURE_CLASS=DISTRIBUTION_VALUE_GATE_FAILURE
ROOT_CAUSE=VALUE_GRID_AND_STATISTIC_SUPPORT_CONFLATION
REAL_DISTRIBUTION_VALUE_DISCREPANCY=NOT_ESTABLISHED
DEC019_FAILED_REPLAY_ARCHIVE_SHA256=687437aa6298ee2c83cb6dcb5da758e2dae52963a189b2b2b5b2e073a527a3ed
ARCHIVE_VERIFICATION=OWNER_REPORTED_EXTERNAL_EVIDENCE
```

Reported first-record gates and numerical values:

```text
CPU_CLASSIFICATION=ELIGIBLE
CUDA_CLASSIFICATION=ELIGIBLE
CLASSIFICATION_GATE_PASS=true
CPU_R=0.0870774587911683
CUDA_R=0.08707745879116827
CPU_P=0.0651252163579915
CUDA_P=0.06512521635799148
CPU_LL=-22.635035251876968
CUDA_LL=-22.63503525187696
FIT_GATE_PASS=true
CPU_STATISTIC=0.2295262676915133
CUDA_STATISTIC=0.22952626769151654
STATISTIC_ABS_ERROR=3.247402347028583e-15
STATISTIC_ALLOWED_TOLERANCE=2e-11
STATISTIC_GATE_PASS=true
DISTRIBUTION_VALUE_GATE_PASS=false
```

All six quantities (pmf, logPMF, cdf, sf, logCDF, logSF) reportedly emitted
`evaluation_point=null`, `cpu_value=null`, `cuda_value=null`,
`abs_error=Infinity`, `allowed_tolerance=null`, `passed=false`.
These are structural sentinels, not measured numerical PMF/CDF errors.

## Local reconstruction and architectural interpretation

On base `94a2d530e768da68085b18692c7eb6c34903e608`, the existing deterministic
CP05 fixture machinery reconstructs raw outer `2` and raw inner `7` under
namespace `CP05-C2C`, without running a replay or modifying fixtures/RNG.
The reconstructed sample SHA-256 is:

```text
bca89c41ebd5857f10a8b9908766c6dd9486eb0bd0bbb6813304823a83c15a0b
```

This matches the exact-regression digest in section 8 of the Owner's R3
instruction. Section 2 supplied a 63-character transcription ending in
`...15a0`; that discrepancy is retained here, not treated as a second valid
SHA-256. The full 64-character expected digest is confirmed by reconstruction.

The sample maximum is `9`; unchanged DEC-014 support certification stops at
`169`. Source inspection shows CPU values on `0..9` and the base CUDA adapter
values on `0..169`. This supports the frozen architectural diagnosis:
`EVALUATION_POINT_IDENTITY_MISMATCH=ESTABLISHED`.
It does not establish a numerical CPU/CUDA distribution-value discrepancy.

[DEC-020](../decisions/cp05-c2c-r10a-r3-value-grid-separation.md) separates
value points from statistic support and adds fail-closed identity evidence.
DEC-019 and its frozen manifests/harness/tests remain historical and unchanged.

## Local software validation boundary

No-GPU tests cover the exact deterministic sample, extended support, separate
adapter consumers, all six value gates, tolerances, continuous point semantics,
and mismatched grids/list lengths. The candidate surface is mocked for routing;
no actual CUDA computation or CPU/CUDA equivalence is claimed. Existing R10-A
and R10-A-R2 tests use their existing NumPy/SciPy numerical isolation.

Baseline and candidate validation commands/results are recorded in the final
implementation handoff. No external replay archive verification, GPU execution,
targeted replay, full campaign, CP05-D or holdout access occurs in this work.
