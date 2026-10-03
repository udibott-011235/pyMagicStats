# DEC-029 — CP05-C2C R11 CUDA Execution-Readiness Oracle and Attempt-1 Infrastructure Disposition

- Status: `accepted`.
- Date recorded: 2026-10-03.
- Owner: Project Owner — Ehud Bottaro.
- Architecture: ChatGPT — statistical/software architecture.
- Documentary implementation: Cortex — implementation engineering.
- Independent QA: Antigravity — adversarial statistical/software QA.
- Related decisions/evidence: [DEC-023](cp05-c2c-r10a-targeted-gpu-replay-r3-preregistration.md),
  [DEC-024](cp05-c2c-r10a-mc-reference-exact-tie-adjudication.md),
  [DEC-026](research-acceleration-boundary.md),
  [DEC-027](cp05-c2c-r11-boundary-sensitive-decision-equivalence-preregistration.md),
  [DEC-028](cp05-c2c-r11-r4-runtime-oracle-identity-correction.md),
  [EV-021](../evidence/cp05-c2c-r10a-r4-runtime-evidence.md),
  [EV-022](../evidence/cp05-c2c-r11-attempt1-infrastructure-failure.md).
- Supersedes: none.

The Project Owner accepted this decision after architecture PASS and Antigravity
PASS. Runtime facts below are attributed
Owner/Quantum evidence, transcribed in EV-022; this documentary materialization
does not independently audit the external runtime artifacts.

DEC-029 does not supersede DEC-027 or DEC-028 and does not change the R11 statistical question, alpha, B, sample identities, frozen payload, tolerance gates, DEC-024 semantics or claim boundary.

It adds:

1. immutable disposition of R11 execution attempt #1;
2. mandatory CUDA execution-readiness oracle before any future R11 scientific record evaluation;
3. a strict authorization-consumption boundary;
4. rules for a possible future R11 attempt #2.

## 1. R11 attempt #1 disposition

The attributed attempt-1 disposition is:

```text
R11_ATTEMPT_1_STATUS=INCONCLUSIVE_INFRASTRUCTURE
R11_GLOBAL_PASS=NO
SCIENTIFIC_EQUIVALENCE_CONCLUSION=NONE

EXECUTION_AUTHORIZATION_CONSUMED=YES
AUTO_RERUN=NO
AUTO_RESUME=NO
CHECKPOINT_RESUME=NO

RECORDS_EVALUATED=1
OUTERS_COMPLETED=0
INDICATOR_ADJUDICATION_RECORDS=0
REJECT_AGREEMENT_COUNT=0
```

Execution identity:

```text
R11_ATTEMPT_1_HARNESS_SHA=c6bde567372a72b1edcfda9e0df5b73e994fa6a7
R11_ATTEMPT_1_HARNESS_TREE=fa2ac2997d7f1be70d01ff6ca869f1a4fad86038

R11_BUILDER_SHA=1ace65bf9e01ab05df84a1cbca5031fac93bfa88
R11_BUILDER_TREE=0ba6d9804a7c46ab636cc47012091f7b9e80dcc6

R11_REFERENCE_WORKLOAD_SHA256=77f53d616deeafcba51aa185dae8a8d2ca60ee4172f4a6353a3aa97b75324b16
```

First and only scientific record reached:

```text
identity=negative_binomial|r=0.25,p=0.1|n=20|CVM|composite|raw_outer=0
record_type=observed

cpu_classification=ELIGIBLE
cpu_statistic=0.026299947929182596

cuda_classification=FAILED
cuda_statistic=NaN
cuda_solver_converged=false

classification_gate_pass=false
fit_gate_pass=false
distribution_value_gate_pass=false
statistic_gate_pass=false
```

Persisted CUDA failure:

```text
CuPy failed to load libnvrtc.so.13:
OSError: libnvrtc.so.13: cannot open shared object file:
No such file or directory
```

The attempt must NOT be classified as statistical or numerical CPU/CUDA disagreement.

Normative interpretation:

```text
SCIENTIFIC_DISCREPANCY_ESTABLISHED=NO
DECISION_EQUIVALENCE_FAILURE_ESTABLISHED=NO
R11_OUTER_FAILURE_ESTABLISHED=NO

CUDA_EXECUTION_READINESS_FAILURE=YES
SCIENTIFIC_QUESTION_REACHED=NO
```

The `NO` values mean “not established by this event”, not proof of universal equivalence.

Attempt #1 is immutable historical evidence.

Prohibit:

```text
ATTEMPT_1_RERUN=NO
ATTEMPT_1_RESUME=NO
ATTEMPT_1_REPAIR_IN_PLACE=NO
ATTEMPT_1_EVIDENCE_REWRITE=NO
ATTEMPT_1_RECORD_REUSE_FOR_COMPLETION=NO
```

No future successful execution may overwrite, replace or retroactively relabel attempt #1 as PASS.

## 2. Relationship with DEC-023

DEC-023 already identified the same class of weakness during R10-A:

a basic CuPy/device smoke can pass without proving NVRTC JIT readiness.

DEC-023 required:

```text
LIBNVRTC_SO_13_LOAD
NVRTC_VERSION
RAWKERNEL_NVRTC_COMPILE
RAWKERNEL_NVRTC_LAUNCH
RAWKERNEL_NUMERICAL_RESULT
CUPY_ELEMENTWISE_REDUCTION
CUPYX_SCIPY_GAMMALN
CUPYX_SCIPY_BETAINC
```

DEC-029 does not rewrite DEC-023.

The R11 failure demonstrates that this previously learned readiness requirement was not made an enforceable prerequisite in the R11 harness.

Therefore:

```text
R11_PREFLIGHT_REGRESSION_CONFIRMED=YES
REGRESSION_CLASS=EXECUTION_READINESS_GUARD
SCIENTIFIC_CODE_REGRESSION_ESTABLISHED=NO
```

DEC-029 makes the readiness requirement normative for any future R11 scientific attempt.

## 3. Observed infrastructure diagnosis

Preserve separately from attempt #1 scientific evidence that subsequent non-scientific diagnostics found:

```text
NVRTC_LIBRARY_PRESENT=YES

NVRTC_DIRECTORY=/usr/local/lib/ollama/mlx_cuda_v13

libnvrtc.so -> libnvrtc.so.13
libnvrtc.so.13 -> libnvrtc.so.13.0.88

libnvrtc-builtins.so -> libnvrtc-builtins.so.13.0
libnvrtc-builtins.so.13.0 -> libnvrtc-builtins.so.13.0.88
```

`libnvrtc.so.13`:

```text
SONAME=libnvrtc.so.13
LDD_MISSING_DEPENDENCIES=NONE
```

The initial dynamic linker lookup did not expose NVRTC through `ldconfig`.

Temporary process-scoped exposure:

```text
LD_LIBRARY_PATH=/usr/local/lib/ollama/mlx_cuda_v13:<existing path if any>
```

then produced:

```text
CuPy=13.6.0
CUDA_DRIVER=13.0
CUDA_RUNTIME=13.0
NVRTC_VERSION=13.0
DEVICE_COUNT=1
GPU=NVIDIA GeForce RTX 5060 Ti

NVRTC_DLOPEN=PASS
```

Synthetic non-scientific smoke subsequently produced:

```text
BASIC_CUDA=PASS
NVRTC_JIT=PASS
GAMMALN=PASS
DIGAMMA=PASS
POLYGAMMA=PASS
GAMMAINC=PASS
GAMMAINCC=PASS
BETAINC=PASS
SORT_NEXTAFTER=PASS

R11_CUDA_DEPENDENCY_SMOKE=PASS
```

Architectural root-cause classification:

```text
ROOT_CAUSE=
NVRTC shared library existed on Quantum but was not visible to the
dynamic loader of the original scientific process.
```

Do not describe NVRTC as absent from the machine.

## 4. No pyMagicStats dependency on Ollama

The observed location is environment evidence only.

Prohibit hard-coding:

```text
/usr/local/lib/ollama/mlx_cuda_v13
```

inside pyMagicStats scientific or harness code.

Normative:

```text
ENVIRONMENT_REPAIR_OWNER=execution supervisor / host environment
HARNESS_ENVIRONMENT_MUTATION=NO
HARNESS_AUTO_REPAIR=NO
```

The harness verifies readiness; it does not install CUDA components, create system symlinks, edit `ldconfig`, or modify `LD_LIBRARY_PATH` to repair the machine.

A launch wrapper/supervisor may provide a process-scoped library path before Python starts.

The exact filesystem source of valid CUDA libraries is not scientifically normative.

## 5. Frozen R11 CUDA dependency surface

For current R11 frozen-payload evaluation, the readiness oracle must exercise the CUDA capabilities actually used by the frozen R11 path.

Mandatory surface:

```text
CUDA_DEVICE_COUNT>=1
CUDA_RUNTIME_CALLABLE=YES

LIBNVRTC_SO_13_LOAD=PASS
NVRTC_VERSION=(13,0)

FLOAT64_ARRAY=PASS
FLOAT64_ELEMENTWISE=PASS
FLOAT64_REDUCTION=PASS

RAWKERNEL_BACKEND=NVRTC
RAWKERNEL_FRESH_COMPILE=PASS
RAWKERNEL_LAUNCH=PASS
RAWKERNEL_NUMERICAL_RESULT=PASS

CUPYX_SCIPY_GAMMALN=PASS
CUPYX_SCIPY_DIGAMMA=PASS
CUPYX_SCIPY_POLYGAMMA=PASS
CUPYX_SCIPY_GAMMAINC=PASS
CUPYX_SCIPY_GAMMAINCC=PASS
CUPYX_SCIPY_BETAINC=PASS

CUPY_SORT=PASS
CUPY_NEXTAFTER=PASS
```

The RawKernel test must actually invoke NVRTC.

A pre-existing cached compiled kernel is insufficient.

Future implementation must guarantee a fresh/empty cache namespace or equivalent proof that compilation occurred during that oracle invocation.

The smoke uses only deterministic toy arrays and must not deserialize or evaluate a real R11 scientific sample.

## 6. cuRAND and cuSOLVER

The post-failure `cp.show_config()` reported:

```text
cuRAND=UNAVAILABLE
cuSOLVER=UNAVAILABLE
```

Do not classify these as current R11 blockers.

The frozen R11 scientific path consumes frozen payloads and does not call the CUDA sample generator, `cp.random`, or a cuSOLVER/linalg path.

Therefore:

```text
CURAND_REQUIRED_BY_CURRENT_R11=NO
CUSOLVER_REQUIRED_BY_CURRENT_R11=NO
```

This is conditional on the frozen current execution path.

If future R11 implementation begins to require either surface:

```text
CUDA_DEPENDENCY_SURFACE_DRIFT=YES
STOP
ARCHITECTURE_REVIEW_REQUIRED=YES
SCIENTIFIC_EXECUTION_PROHIBITED=YES
```

No readiness requirement may be silently weakened or expanded post hoc.

## 7. Readiness oracle ordering

For any future R11 scientific attempt, the required order is:

```text
exact clean Git checkout
        ↓
frozen contract / workload / R4 oracle identity checks
        ↓
CUDA execution-readiness oracle
        ↓
CUDA_READINESS_ORACLE_PASS required
        ↓
scientific execution infrastructure may be created
        ↓
frozen sample deserialization / CPU scientific evaluation
        ↓
immediately before first actual CUDA scientific evaluator call
        ↓
scientific authorization consumption marker
        ↓
CUDA scientific evaluation
```

The readiness oracle itself is non-scientific GPU work.

It must NOT consume scientific execution authorization.

Required states before first scientific CUDA call:

```text
CUDA_READINESS_ORACLE_PASS=YES
EXECUTION_AUTHORIZATION_CONSUMED=NO
```

Then immediately before the first actual scientific CUDA evaluation:

```text
EXECUTION_AUTHORIZATION_CONSUMED=YES
```

The existing DEC-027/PR-30 rule that consumption occurs immediately before the first actual CUDA scientific evaluator remains correct.

The defect was insufficient readiness certification before reaching that point.

## 8. Readiness failure semantics

On any readiness prerequisite failure:

```text
CUDA_READINESS_ORACLE_PASS=NO
SCIENTIFIC_EXECUTION_STARTED=NO
SCIENTIFIC_RECORDS_EVALUATED=0
EXECUTION_AUTHORIZATION_CONSUMED=NO
SCIENTIFIC_OUTPUT_DIRECTORY_CREATED=NO
```

Preserve readiness evidence separately from the not-created scientific bundle.

No automatic repair or retry.

```text
AUTO_REPAIR=NO
AUTO_RERUN=NO
AUTO_RESUME=NO
```

A subsequent execution requires a fresh explicit Owner authorization.

## 9. Readiness evidence

Future implementation must persist a write-once non-scientific readiness artifact, separate from a scientific result directory that does not yet exist.

At minimum preserve:

```text
timestamp
host/platform
Python version
CuPy version
CUDA driver version
CUDA runtime version
device count
device identity
effective CUDA_PATH if any
effective LD_LIBRARY_PATH
NVRTC library load result
NVRTC version
fresh-JIT proof/cache isolation
each required smoke result
overall CUDA_READINESS_ORACLE_PASS
executing harness SHA/tree
```

Do not persist secrets.

The readiness artifact is operational/software evidence, not scientific equivalence evidence.

## 10. Future R11 attempt #2

DEC-029 does NOT authorize another scientific execution.

A possible future execution is a new attempt:

```text
R11_ATTEMPT_2=FUTURE
R11_ATTEMPT_2_AUTHORIZED=NO
```

It may reuse the already accepted immutable R11 reference workload because the scientific sample definition has not changed.

Required:

```text
REFERENCE_WORKLOAD_SHA256=
77f53d616deeafcba51aa185dae8a8d2ca60ee4172f4a6353a3aa97b75324b16

ATTEMPT_1_OUTPUT_REUSE=NO
ATTEMPT_1_RECORD_REUSE=NO
ATTEMPT_1_AUTHORIZATION_REUSE=NO
CHECKPOINT_REUSE=NO

FRESH_OUTPUT_REQUIRED=YES
FRESH_AUTHORIZATION_REQUIRED=YES
```

The new harness may change only to implement/readiness-gate sequencing and associated software evidence/tests unless separately authorized.

The canonical CPU/CUDA scientific implementations, statistical gates, adjudication logic and frozen scientific semantics remain unchanged.

If readiness implementation requires changing pre-existing scientific mathematics:

```text
STOP
DEC_029_INSUFFICIENT=YES
NEW_ARCHITECTURE_DECISION_REQUIRED=YES
GPU_EXECUTION_PROHIBITED=YES
```

## 11. Scientific contract remains frozen

Explicitly preserve:

```text
ALPHA=0.05
B_R11=199
P_MC=(b+1)/(B+1)
MC_COMPARATOR=T_boot >= T_obs
reject=p<=0.05

R11_SOURCE_OUTERS=12
R11_RECORDS_PER_OUTER=200
R11_TOTAL_RECORDS=2400

CANONICAL_REFERENCE=NumPy/SciPy
CUDA_ROLE=research instrumentation only
```

Preserve DEC-024 exact tie rule and all R11 numerical gates.

No tolerance change.

No sample selection change.

No new generator.

No CUDA-generated scientific sample.

No post-hoc pruning.

## 12. Claim boundary

Attempt #1 permits only:

> The first authorized R11 CUDA scientific execution was inconclusive because
> the process could not dynamically load `libnvrtc.so.13` at the first CUDA
> scientific record. No R11 CPU/CUDA decision-equivalence conclusion was
> reached.

The successful post-failure smoke permits only:

> The CUDA surface required by the current R11 implementation executed
> successfully under a process environment in which NVRTC was visible to the
> dynamic loader.

It does NOT establish:

```text
R11_DECISION_EQUIVALENCE
FULL_C2C_EQUIVALENCE
TYPE_I_CALIBRATION
POWER
PERFORMANCE
CUDA_PRODUCTION_READINESS
PRODUCTION_READINESS
CP05_D_READINESS
HOLDOUT_VALIDATION
```

## 13. Current authorization boundary

This documentary task authorizes only:

```text
DEC_029_DOCUMENTARY_MATERIALIZATION=YES
EV_022_DOCUMENTARY_MATERIALIZATION=YES

R11_READINESS_IMPLEMENTATION=NO
R11_HARNESS_MODIFICATION=NO
R11_ATTEMPT_2_EXECUTION=NO
GPU_SCIENTIFIC_EXECUTION=NO
PR=NO
MERGE=NO
```
