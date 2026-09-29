# EV-020 — R3 MC comparison cliff reported by the Project Owner / Quantum

- Status: `validated_with_limits`; not `accepted`.
- Date recorded: 2026-09-27; external execution timestamp not supplied.
- Recorded by: Cortex — Implementation Engineering.
- Source: Project Owner Ehud Bottaro / Quantum, supplied in the DEC-024 instruction.
- Reviewers: ChatGPT architecture, Antigravity independent QA and Project Owner.
- Related accepted decision: [DEC-024](../decisions/cp05-c2c-r10a-mc-reference-exact-tie-adjudication.md).

## Validation scope and limits

The Project Owner supplied Antigravity's independent PASS for candidate
`843d932a8f63282d4dfc1e7f70b4a0a7b7b2b6b2`, tree
`bc885233840ece806c35b7b1738967c3f7dedf96`, with exact remote identity, exactly
four files in scope and no blocker, major, minor or note. The subsequent explicit
Owner acceptance closes DEC-024, not the R3 execution or external archive.
The evidence status is limited precisely as follows:

- Frozen R9 Git provenance: independently verified.
- DEC-024 adversarial reasoning: independently audited.
- R3 runtime/output/archive: Project Owner / Quantum reported.
- Archive bytes: not independently retrieved or hashed by Cortex or Antigravity.

These independent review facts are recorded from the Owner-supplied audit;
Cortex performed no new independent runtime/archive audit during closure.
The checksum below remains Owner/Quantum-reported. R3 remains FAILED with
consumed authorization; validation does not permit rerun, resume or retroactive
PASS, and does not authorize R4 implementation or execution.

## Provenance and immutable execution status

Runtime observations and the archive checksum below are Owner/Quantum-reported.
Cortex did not access Quantum, run GPU/replay, retrieve the external archive or
independently hash its bytes. The supplied checksum is preserved as an
attributed identifier, not a locally verified archive. Archive location and
execution timestamp were not supplied and are not invented. This document
does not replace or modify the original R3 output/archive.

```text
BASE_SHA=9282d5b082c4977ee12a7c6e0a54d908ee7283ff
BASE_TREE=f2f55f1e90df24d24a3d113bda8da7be4e24f6c9

R3_EXECUTION_COUNT=1
R3_EXECUTION_STATUS=FAILED
R3_AUTHORIZATION_CONSUMED=YES
AUTO_RERUN=NO
AUTO_RESUME=NO

SCIENTIFIC_SHA=d2abd57e65bb7433eff81872a4f6510144d7c267
SCIENTIFIC_TREE=70b20d1f9cbc20d6cd77de4c25b3017a6413932c
HARNESS_SHA=9282d5b082c4977ee12a7c6e0a54d908ee7283ff

R3_EVIDENCE_ARCHIVE_SHA256=ac24de530d8a6e835806ed77c862d66b0853793479912aa2d8a779e0725c0d78
```

[DEC-023](../decisions/cp05-c2c-r10a-targeted-gpu-replay-r3-preregistration.md),
the [R3 harness](../../experiments/distribution_gof/cuda_calibration/targeted_replay_r3/targeted_gpu_replay.py),
its tests, manifest and the consumed failed execution remain immutable.

## Reported stopping point and counters

```text
FAILED_OUTER=negative_binomial|r=1,p=0.9|n=250|AD|composite|raw_outer=7

WORKLOAD_A_RECORDS_EXECUTED=134
WORKLOAD_B_RECORDS_EXECUTED=144
MC_OUTERS_RECONSTRUCTED=9

CLASSIFICATION_MISMATCH=0
CUDA_NONCONVERGENCE=0
FIT_GATE_FAILURE=0
DISTRIBUTION_VALUE_GATE_FAILURE=0
STATISTIC_GATE_FAILURE=0
MC_BOOTSTRAP_IDENTITY_MISMATCH=0
UNEXPLAINED_DISCREPANCY=0

MC_EXCEEDANCE_COUNT_MISMATCH=1
MC_REJECT_DECISION_MISMATCH=0
```

These are the reported partial R3 counters at failure, not completed-workload
acceptance. Zero individual statistic-gate failures does not remove the raw MC
count failure required by the historical R3 contract.

## Reported individual crossing (values preserved exactly)

```text
CROSSING_IDENTITY=negative_binomial|r=1,p=0.9|n=250|AD|composite|raw_outer=7|raw_inner=28

sample_digest=4fb5bb9ec09143f2fa078c45405ca39a3e48a60af765fc2d5e4a5349f7f42075
seed_identity=186433253709330584616195533534134922757

CPU_OBSERVED=0.0004924344188814927
CPU_BOOTSTRAP=0.0004924344188814927
CPU_MARGIN=0.0
CPU_EXCEEDANCE=true

CUDA_OBSERVED=0.0004924344188815376
CUDA_BOOTSTRAP=0.000492434418881383
CUDA_MARGIN=-1.5460722979643293e-16
CUDA_EXCEEDANCE=false

BOOTSTRAP_CPU_CUDA_ABS_ERROR=1.0972125985553305e-16
STATISTIC_ALLOWED_TOLERANCE=2e-11

RAW_B_CPU=12
RAW_B_CUDA=11
RAW_P_CPU=0.8125
RAW_P_CUDA=0.75
REJECT_CPU=false
REJECT_CUDA=false

CROSSING_COUNT=1
```

This is an exact CPU reference tie in the reported values, with CPU `>=`
true and CUDA `>=` false. The reported bootstrap CPU/CUDA statistic error is
below the unchanged individual statistic tolerance, yet the raw exceedance
count and p-value differ. Identical reject decisions do not retroactively make
the failed R3 MC gate pass.

## Locally verified frozen R9 provenance

Cortex read the exact Git blob in BASE_SHA at the
[frozen R3 manifest](../../experiments/distribution_gof/cuda_calibration/targeted_replay_r3/frozen_identity_manifest.json).
Its `mc_failed_outers` entry for the same failed outer contains:

```text
R9_B_CPU=12
R9_B_CUDA=11
R9_P_CPU=0.8125
R9_P_CUDA=0.75
R9_REJECT_CPU=false
R9_REJECT_CUDA=false
```

These correspond exactly to the manifest fields `r9_b_cpu`, `r9_b_cuda`,
`r9_p_cpu`, `r9_p_cuda`, `r9_reject_cpu`, `r9_reject_cuda`. The ordered
frozen bootstrap list also contains the exact `raw_inner=28` identity.
No sample was regenerated and no seed/digest was independently recomputed.

Comparing that locally verified historical provenance with the attributed R3
raw counts, p-values and reject flags yields:

```text
R9_R3_MC_TOPOLOGY_REPRODUCED=YES
```

This label is limited to the reported R3 versus frozen R9 count/p-value/reject
topology for this outer. It is not independent reproduction of R3 execution,
proof of identical numerical mechanisms in R9, or a PASS under either contract.

## Interpretation, limitations and next role

This design case motivates the prospective canonical CPU-reference exact-tie
adjudication in DEC-024. It supplies no authority for fuzzy comparisons,
rounding, near-tie certification or widening tolerances. Full individual
certification prerequisites must be checked in a future authorized
implementation/execution; this evidence record does not itself certify them
from aggregate counters or from the uninspected archive.

R3 remains FAILED with consumed authorization. No automatic rerun/resume or
retroactive adjudication is allowed. A future R4 must be separately authorized
and fresh after DEC-024 acceptance/audit. No raw p-value bit identity,
GPU-native p-value identity, full C2C equivalence, calibration, power,
performance, production readiness or CP05-D readiness is established.

Only documentary and Git-blob checks were performed by Cortex. Next role:
ChatGPT architecture; archive verification remains for separately authorized
review. No code, historical evidence, fixtures, R3 harness or raw results are
changed; no GPU, replay, Quantum, CP05-D or holdout access occurred.
