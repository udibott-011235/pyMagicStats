# EV-022 — CP05-C2C R11 attempt-1 infrastructure failure and CUDA readiness remediation

- Status: `proposed`.
- Date recorded: 2026-10-03.
- Repository: `udibott-011235/pyMagicStats`.
- Documentary branch: `docs/cp05-c2c-r11-cuda-execution-readiness-oracle`.
- Owner role: implementation-engineering — Cortex, documentary transcription only.
- Evidence attribution: Project Owner — Ehud Bottaro / Quantum runtime.
- Reviewers: statistical-software-architecture — ChatGPT;
  adversarial-statistical-qa — Antigravity; decision-owner — Ehud Bottaro.
- Related: [DEC-029](../decisions/cp05-c2c-r11-cuda-execution-readiness-oracle.md),
  [DEC-023](../decisions/cp05-c2c-r10a-targeted-gpu-replay-r3-preregistration.md),
  [DEC-024](../decisions/cp05-c2c-r10a-mc-reference-exact-tie-adjudication.md),
  [DEC-026](../decisions/research-acceleration-boundary.md),
  [DEC-027](../decisions/cp05-c2c-r11-boundary-sensitive-decision-equivalence-preregistration.md),
  [DEC-028](../decisions/cp05-c2c-r11-r4-runtime-oracle-identity-correction.md),
  [EV-021](cp05-c2c-r10a-r4-runtime-evidence.md).
- Supersedes: none.

## Attribution and limits of documentary transcription

This record transcribes Owner/Quantum runtime evidence supplied by the Project
Owner. It is **not independently audited**. Cortex did not retrieve, load,
rehash or evaluate the external R11 workload or result bundle, and did not
perform CUDA diagnostics, a scientific execution or a rerun in this task.
The byte counts, digests and runtime observations below are attributed values,
not fresh local measurements.

Persisted attempt-1 evidence, post-failure non-scientific diagnostics and
architectural interpretation are separated below. Contract labels transcribe
the supplied facts; they do not assert an unsupplied complete JSON schema or
additional runtime fields. The successful later smoke is not part of the
scientific R11 attempt.

## 1. Persisted attempt-1 evidence

External evidence directory:

```text
/home/udibott/pymagicstats-r11-evidence/r11-cuda-decision-equivalence-2026-10-03
```

### Reported bundle byte counts and hashes

```text
authorization_consumed.json
bytes=147
sha256=b130c0fbf5b99bf842d32abae52e8fa07f07b27104778fa0b82a2ded13bb9d21

boundary_fixtures.json
bytes=3501383
sha256=d167b627c56b3131c64cc307548c83c2638504e6fbb1cd97397bb564f9a2ec9d

environment.json
bytes=10104
sha256=5c0fb77e83d1281005a4d5391c370452a7d0c06c3a1aae4ed6fd2b2ec02784ee

execution_manifest.json
bytes=1686
sha256=fa018c01148bd15d72e19f6b035d9c1eb434d9f35e8a47b2a6da82ec9aae36bb

indicator_adjudication.jsonl
bytes=0
sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

outer_results.json
bytes=2
sha256=4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945

records.jsonl
bytes=7964
sha256=33b07c04db2694d96ac3f5bd3d440637424c3cb9d70ec4e4183ae71b466976e1

reference_workload.json
bytes=4511426
sha256=77f53d616deeafcba51aa185dae8a8d2ca60ee4172f4a6353a3aa97b75324b16

summary.json
bytes=731
sha256=14de8385d23e3849eeb8130aeffe993a7c166efc6332aa755a73e631efd5e5ee
```

`digests.json` excluded its own hash. No SHA for that file was supplied,
verified or invented in this record.

```text
records.jsonl_LINES=1
indicator_adjudication.jsonl_LINES=0
```

### Authorization, execution-manifest identity and summary

The single authorization was consumed. One observed record was reached, no
outer completed, and no indicator adjudication or reject agreement was
established. The documentary disposition is `INCONCLUSIVE_INFRASTRUCTURE`.

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

The reported executing harness, accepted builder and immutable workload
identities are:

```text
R11_ATTEMPT_1_HARNESS_SHA=c6bde567372a72b1edcfda9e0df5b73e994fa6a7
R11_ATTEMPT_1_HARNESS_TREE=fa2ac2997d7f1be70d01ff6ca869f1a4fad86038

R11_BUILDER_SHA=1ace65bf9e01ab05df84a1cbca5031fac93bfa88
R11_BUILDER_TREE=0ba6d9804a7c46ab636cc47012091f7b9e80dcc6

R11_REFERENCE_WORKLOAD_SHA256=77f53d616deeafcba51aa185dae8a8d2ca60ee4172f4a6353a3aa97b75324b16
```

### First and only scientific record reached

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

The persisted CUDA failure was:

```text
CuPy failed to load libnvrtc.so.13:
OSError: libnvrtc.so.13: cannot open shared object file:
No such file or directory
```

These facts preserve the unsuccessful attempt. No completed outer or scientific
equivalence outcome may be inferred from this single failed CUDA record.

## 2. Post-failure diagnostic observations

The following observations were made after the scientific attempt and are
non-scientific infrastructure evidence. They neither resume nor complete
attempt #1 and do not repair or rewrite its persisted records.

NVRTC was present on Quantum at the observed environment location:

```text
NVRTC_LIBRARY_PRESENT=YES

NVRTC_DIRECTORY=/usr/local/lib/ollama/mlx_cuda_v13

libnvrtc.so -> libnvrtc.so.13
libnvrtc.so.13 -> libnvrtc.so.13.0.88

libnvrtc-builtins.so -> libnvrtc-builtins.so.13.0
libnvrtc-builtins.so.13.0 -> libnvrtc-builtins.so.13.0.88
```

Reported library identity and dependency inspection:

```text
SONAME=libnvrtc.so.13
LDD_MISSING_DEPENDENCIES=NONE
```

The original dynamic-linker lookup did not expose NVRTC through `ldconfig`.
A temporary process-scoped exposure, supplied before Python started, used:

```text
LD_LIBRARY_PATH=/usr/local/lib/ollama/mlx_cuda_v13:<existing path if any>
```

Under that process environment the reported versions, device and load result
were:

```text
CuPy=13.6.0
CUDA_DRIVER=13.0
CUDA_RUNTIME=13.0
NVRTC_VERSION=13.0
DEVICE_COUNT=1
GPU=NVIDIA GeForce RTX 5060 Ti

NVRTC_DLOPEN=PASS
```

The subsequent deterministic synthetic dependency smoke reported:

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

The later `cp.show_config()` also reported:

```text
cuRAND=UNAVAILABLE
cuSOLVER=UNAVAILABLE
```

The observed Ollama directory is environment evidence only. It is not a
pyMagicStats dependency or a path to hard-code in scientific or harness code.
Host repair belongs to the execution supervisor / environment; the harness
must verify readiness without installation, system symlink changes,
`ldconfig` edits, `LD_LIBRARY_PATH` mutation or automatic repair.

## 3. Architectural interpretation and immutable disposition

### Infrastructure failure, no established scientific discrepancy

The root-cause classification is:

```text
ROOT_CAUSE=
NVRTC shared library existed on Quantum but was not visible to the
dynamic loader of the original scientific process.
```

NVRTC was not absent from the machine: it was not visible to the dynamic loader
of the original scientific process. The event must not be classified as a
statistical or numerical CPU/CUDA disagreement.

```text
SCIENTIFIC_DISCREPANCY_ESTABLISHED=NO
DECISION_EQUIVALENCE_FAILURE_ESTABLISHED=NO
R11_OUTER_FAILURE_ESTABLISHED=NO

CUDA_EXECUTION_READINESS_FAILURE=YES
SCIENTIFIC_QUESTION_REACHED=NO
```

These `NO` values mean “not established by this event”; they are not proof
of universal CPU/CUDA equivalence.

DEC-023 already identified that a basic CuPy/device smoke can pass without
proving NVRTC JIT readiness. Its mandatory strong smoke included:

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

The R11 failure showed that this previously learned readiness requirement was
not an enforceable prerequisite in the R11 harness:

```text
R11_PREFLIGHT_REGRESSION_CONFIRMED=YES
REGRESSION_CLASS=EXECUTION_READINESS_GUARD
SCIENTIFIC_CODE_REGRESSION_ESTABLISHED=NO
```

DEC-029 does not rewrite DEC-023 or supersede DEC-027/028. It prospectively
requires deterministic synthetic CUDA execution readiness before scientific
infrastructure, sample deserialization or CPU scientific evaluation. Fresh
NVRTC compilation must be proved; a previously cached kernel is insufficient.
Readiness is non-scientific GPU work and does not consume scientific execution
authorization. Consumption remains immediately before the first actual CUDA
scientific evaluator call. A readiness failure must leave scientific records at
zero, authorization unconsumed and the scientific output directory uncreated;
readiness evidence is separate and write-once, with no automatic repair or retry.

### Conditional dependency surface

The frozen current R11 path evaluates frozen payloads and does not call the
CUDA generator, `cp.random`, or cuSOLVER/linalg. Therefore:

```text
CURAND_REQUIRED_BY_CURRENT_R11=NO
CUSOLVER_REQUIRED_BY_CURRENT_R11=NO
```

The reported unavailable cuRAND/cuSOLVER libraries are not current R11
blockers. This interpretation is conditional on that frozen execution path;
a future requirement for either surface means:

```text
CUDA_DEPENDENCY_SURFACE_DRIFT=YES
STOP
ARCHITECTURE_REVIEW_REQUIRED=YES
SCIENTIFIC_EXECUTION_PROHIBITED=YES
```

### Historical immutability and future authorization

```text
ATTEMPT_1_RERUN=NO
ATTEMPT_1_RESUME=NO
ATTEMPT_1_REPAIR_IN_PLACE=NO
ATTEMPT_1_EVIDENCE_REWRITE=NO
ATTEMPT_1_RECORD_REUSE_FOR_COMPLETION=NO
```

A future success must never overwrite, replace or retroactively relabel
attempt #1 as PASS. A possible attempt #2 remains future and unauthorized;
attempt-1 output, records, authorization and checkpoints cannot be reused for
completion. A future separately authorized attempt may reuse the accepted
immutable reference workload, but must have fresh output and fresh Owner
authorization. DEC-027 scientific semantics, DEC-024 exact ties and numerical
gates remain frozen.

### Claim boundary

Attempt #1 permits only:

> The first authorized R11 CUDA scientific execution was inconclusive because
> the process could not dynamically load `libnvrtc.so.13` at the first CUDA
> scientific record. No R11 CPU/CUDA decision-equivalence conclusion was
> reached.

The post-failure smoke permits only:

> The CUDA surface required by the current R11 implementation executed
> successfully under a process environment in which NVRTC was visible to the
> dynamic loader.

Neither result establishes R11 decision equivalence, full C2C equivalence,
type-I calibration, power, performance, CUDA production readiness, production
readiness, CP05-D readiness or holdout validation.

This documentary task creates only proposed DEC-029 and EV-022 records.
Readiness implementation, harness modification, attempt #2, GPU/scientific
execution, push, PR and merge are not authorized.
