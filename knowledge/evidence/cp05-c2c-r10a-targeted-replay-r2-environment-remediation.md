# EV-019 — R2 failure and environment remediation reported by the Project Owner

- Status: `proposed`.
- Date recorded: 2026-09-27; external execution timestamps not supplied.
- Recorded by: Cortex — Implementation Engineering.
- Source: Project Owner Ehud Bottaro / Quantum, supplied in the DEC-023 instruction.
- Reviewers: ChatGPT architecture, Antigravity independent QA and Project Owner.
- Related decision: [DEC-023](../decisions/cp05-c2c-r10a-targeted-gpu-replay-r3-preregistration.md).

## Provenance and observation boundary

Every runtime observation below is attributed to the Owner/Quantum, not a
Cortex execution or independent audit. The supplied instruction is the source;
no external archive, log bundle or checksum was provided or independently
verified here. No checksum, command transcript, execution timestamp or missing
artifact is fabricated. Preserve R2 outputs/archive in place as immutable
history. This record is not a replacement for the original artifacts.

The documentary base is `2c1a5e99db086651efd046921d48ef8187a5e0d7`, tree
`cd2e4e5220c3c3734d7b7796a58a7172f1a4f84c`. The frozen scientific identity is
`d2abd57e65bb7433eff81872a4f6510144d7c267`, tree
`70b20d1f9cbc20d6cd77de4c25b3017a6413932c`, namespace `CP05-C2C`.
[DEC-022](../decisions/cp05-c2c-r10a-targeted-gpu-replay-r2-preregistration.md)
and [EV-017](cp05-c2c-r10a-targeted-replay-r1-failure.md) remain unchanged.

## Owner-reported single R2 execution

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

Reported persisted failure at the first identity of Workload A:

```text
CuPy failed to load libnvrtc.so.13:
OSError: libnvrtc.so.13: cannot open shared object file
```

This event does not establish the following failures; the `NO` values mean
not established by this loader failure, not universal absence of defects:

```text
SCIENTIFIC_DISCREPANCY=NO
R10A_SOLVER_FAILURE=NO
VALUE_GRID_REGRESSION=NO
HARNESS_SEMANTIC_FAILURE=NO
```

## Owner-reported environment remediation and strong smoke

```text
Python=3.12.3
NumPy=1.26.4
SciPy=1.16.2
CuPy=13.6.0
CUDA_DRIVER=13.0
CUDA_RUNTIME=13.0
NVRTC_VERSION=13.0
GPU=NVIDIA GeForce RTX 5060 Ti
nvidia-cuda-nvrtc=13.0.88
nvidia-cuda-runtime=13.0.96
```

The already-installed wheels contained:

```text
/home/udibott/.venv_gpu/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13
/home/udibott/.venv_gpu/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc-builtins.so.13.0
/home/udibott/.venv_gpu/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13
```

The valid remediation reported was temporary environment exposure only:

```text
CUDA_WHEEL_ROOT=/home/udibott/.venv_gpu/lib/python3.12/site-packages/nvidia/cu13
CUDA_PATH=$CUDA_WHEEL_ROOT
LD_LIBRARY_PATH=$CUDA_WHEEL_ROOT/lib
```

No installation, upgrade or system modification occurred according to the
Owner. Cortex did not apply these variables, connect to Quantum or run a smoke.
Reported results:

```text
NVRTC_LOAD=PASS
RAWKERNEL_NVRTC=PASS
ELEMENTWISE_REDUCTION=PASS
GAMMALN=PASS
BETAINC_CDF=PASS
BETAINC_SF=PASS
STRONG_CUDA13_NVRTC_SMOKE=PASS
```

## Owner-reported exact historical-record diagnostic

After the loader remediation, the Owner reports reconstruction of only the
failed identity above, not execution of Workload A, Workload B or MC. Its raw
outer/inner identity remains `2` / `7`; the numerical seed was not supplied
and is not invented. The reported values are:

```text
sample_digest=bca89c41ebd5857f10a8b9908766c6dd9486eb0bd0bbb6813304823a83c15a0b
sample_max=9
CPU_CLASSIFICATION=ELIGIBLE
CUDA_CLASSIFICATION=ELIGIBLE
SUPPORT_STOP=169
SUPPORT_SIZE=170
REMAINDER_BOUND=2.0360815661183532e-13
CUDA_FAILURE_REASON=None
CUDA_SOLVER_CONVERGED=True
CPU_r=0.0870774587911683
CUDA_r=0.08707745879116827
CPU_p=0.0651252163579915
CUDA_p=0.06512521635799148
CPU_LOG_LIKELIHOOD=-22.635035251876968
CUDA_LOG_LIKELIHOOD=-22.63503525187696
CPU_STATISTIC=0.2295262676915133
CUDA_STATISTIC=0.22952626769151654
VALUE_EVALUATION_GRID=0..9
GOF_STATISTIC_SUPPORT=0..169
CUDA_DISTRIBUTION_QUANTITIES=pmf,logPMF,cdf,sf,logCDF,logSF
```

The following labels describe this one-record probe, not the preceding failed
R2 run (which did start and consumed its authorization):

```text
EQUIVALENCE_CLAIM=NO
TARGETED_REPLAY_EXECUTED=NO
SCIENTIFIC_WORKTREE_STILL_CLEAN=YES
```

## Interpretation, limits and next role

The Owner-supplied architectural diagnosis is loader visibility, motivating
DEC-023's stronger pre-output NVRTC readiness contract. A simple CuPy reduction
is not sufficient evidence that NVRTC can load/compile. The reported successful
probe is not an equivalence, calibration, performance or production claim, and
does not reopen R2 or authorize R3. R3 retains exactly the DEC-022 scientific
question, candidate, tolerances and ordered workloads.

Cortex only materializes this attributed evidence and validates local documents.
No scientific results were reproduced, no independent QA verdict is issued,
and no historical archive is rewritten. Missing external artifact identifiers
remain a provenance limitation for architectural/independent review. Next:
ChatGPT architecture; future harness work, publication and execution require
separate authorization. No GPU, targeted replay, full campaign, CP05-D or
holdout access is part of this task.
