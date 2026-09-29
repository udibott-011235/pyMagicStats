# EV-016 — R10-A targeted replay harness R1 materialization

- Registry status: `validated_with_limits`.
- Recorded: `2026-09-23`.
- Role: Cortex / implementation engineering.
- Repository: `udibott-011235/pyMagicStats`.
- Branch: `evidence/cp05-c2c-r10a-targeted-replay-harness-r1`.
- Base and preregistration: `b8255f4fdebe59bf534003b2fdcdaac687bd6543`.
- Scientific replay checkout: `649ca296c57ab237f3e93ea35ac9870e1d1bfc9d`.

## Authority and scope

The Project Owner supplied the architecture and Antigravity PASS dispositions
for the external audited artifacts and authorized their byte-identical local
materialization, evidence registration and one local commit. The review
dispositions below describe those frozen artifact bytes; they do not transfer
an independent audit verdict to the new Git commit. The new candidate remains
subject to exact-identity review. Publication, PR, merge and scientific execution
are not authorized by this record.

```text
WORK_ITEM=CP05-C2C-R10-A-TARGETED-GPU-REPLAY-HARNESS-R1
STATUS=AUDITED_READY_FOR_TARGETED_GPU_EXECUTION
PREREGISTRATION_SHA=b8255f4fdebe59bf534003b2fdcdaac687bd6543
REPLAY_EXECUTION_SHA=649ca296c57ab237f3e93ea35ac9870e1d1bfc9d
ARCHITECTURE_REVIEW=PASS
ANTIGRAVITY_AUDIT=PASS
HARNESS_R1_001=CLOSED
BLOCKER_COUNT=0
MAJOR_COUNT=0
MINOR_COUNT=0
NOTE_COUNT=0
```

The status and counts above record the supplied artifact audit disposition,
not results of a GPU run. HARNESS-R1-001 concerns stopping at the first record
or MC discrepancy, after preserving the available evidence. Record counters
are checked by the delta caused by the current record. The failure preservation
policy includes `NO_FIXTURE_CHANGE=true`.

## Artifact identity and relative paths

The versioned unit is
`experiments/distribution_gof/cuda_calibration/targeted_replay_r1/`.
Its harness, execution manifest and mock tests are copies of the three external
audited files from
`artifacts/cp05-c2c-r10a-targeted-gpu-replay-harness-b8255f4/` in the Owner's
Codex workspace. No artifact was reconstructed, reformatted or edited.

Grouping them preserves the harness's sibling-manifest path and the test's
sibling-harness path. The directory-local `.gitattributes` contains exactly:

```gitattributes
targeted_gpu_replay.py -text
frozen_identity_manifest.json -text
test_harness_r1_fail_fast.py -text
```

```text
HARNESS_SHA256=d84282a560c7952597372ee907631fbf32efb4868639c851203e9a16699e9399
TEST_SHA256=9e8d112d1cc8cbbc234d3cc661ac23834ed32b3f2d9fd33a3f90ff9da10bb577
HARNESS_BYTE_IDENTITY=PASS
MANIFEST_BYTE_IDENTITY=PASS
TEST_BYTE_IDENTITY=PASS
PATH_LAYOUT_ADAPTATION=YES
AUDITED_ARTIFACT_BYTES_CHANGED=NO
BYTE_STABLE_GIT_MATERIALIZATION=YES
GIT_EOL_PROTECTION=DISABLE_GIT_EOL_NORMALIZATION_FOR_AUDITED_ARTIFACTS
```

## Canonical manifest versus audited execution representation

```text
CANONICAL_MANIFEST_PATH=knowledge/decisions/cp05-c2c-r10a-targeted-gpu-replay-identities.json
CANONICAL_MANIFEST_GIT_BLOB_SHA256=39e01237b3497be542b2c81e3d96ba0de78278bf8f22bd7bc86bed8fc5f2492a
AUDITED_EXECUTION_MANIFEST_PATH=experiments/distribution_gof/cuda_calibration/targeted_replay_r1/frozen_identity_manifest.json
AUDITED_EXECUTION_MANIFEST_SHA256=efb550af870213160ed450c222043a0c73cc415a7aecb633c62d61118d72b36d
MANIFEST_SHA256=efb550af870213160ed450c222043a0c73cc415a7aecb633c62d61118d72b36d
BYTE_DIFFERENCE=EOL_ONLY
NORMALIZED_CONTENT_EQUIVALENCE=PASS
CANONICAL_CONTENT_DRIFT=NO
EXECUTION_MANIFEST_SOURCE=AUDITED_EXTERNAL_ARTIFACT
CANONICAL_MANIFEST_CHANGED=NO
```

The canonical Git blob has 116330 bytes and 2950 LF line endings, with no CRLF.
The audited execution artifact has 119280 bytes and 2950 CRLF line endings.
Replacing CRLF with LF in memory makes the execution bytes equal to the
canonical blob. The two byte streams retain their distinct hashes; the
canonical manifest and DEC-016/018/019 remain unchanged. Index and committed
artifact hashes must match the audited hashes without Git EOL normalization.

## Local validation and collection boundary

Validation uses Python 3.12 from the existing local `.venv`, with
`PYTHONDONTWRITEBYTECODE=1` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
No scientific runtime is loaded by the static mode or mock tests.

Commands from the repository root:

```text
python knowledge/tools/validate_registry.py
python -m py_compile experiments/distribution_gof/cuda_calibration/targeted_replay_r1/targeted_gpu_replay.py
python experiments/distribution_gof/cuda_calibration/targeted_replay_r1/targeted_gpu_replay.py --validate-static
python -m pytest -q --confcutdir=experiments/distribution_gof/cuda_calibration/targeted_replay_r1 experiments/distribution_gof/cuda_calibration/targeted_replay_r1/test_harness_r1_fail_fast.py
git diff --check
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check
```

Registry validation, compilation and static validation pass. Static validation
confirms 134 unique Workload A records (10 observed, 124 bootstrap), NB only,
12 Workload B outers and 15 ordered bootstrap identities per outer, including
the frozen gaps. It reports no full-runner calls or full-matrix execution loop.

The initial unscoped `python -m pytest -q <test-path>` invocation produced
11 setup errors: pytest attempted to import the ancestor `experiments` package,
which the audited test deliberately blocks. Restricting pytest collection with
`--confcutdir` to the artifact directory passes **11 tests and 102 subtests**.
The same option can be supplied via `PYTEST_ADDOPTS` with the original command.
The audited import guard remains active and unchanged. The scoped command is
required to reproduce this validation; the unscoped command is not claimed to
pass in this checkout.

`git diff --check` passes. The additional default staged check flags each of
the 2950 preserved CRLF endings of the execution manifest as trailing whitespace.
The staged check above passes with `cr-at-eol` enabled for that command only;
blank-at-eol, blank-at-eof and space-before-tab checks remain enabled. No Git
configuration or audited artifact bytes are changed to address these reports.

## Claim limits and handoff

This evidence certifies the targeted replay harness,
its frozen identity contract and fail-fast semantics only.

It does NOT establish:

- CUDA equivalence;
- full 1152 equivalence;
- calibration;
- performance;
- production readiness;
- R10-B/C/D completion;
- CP05-C2C completion.

```text
GPU_EXECUTION_PERFORMED=NO
TARGETED_REPLAY_PERFORMED=NO
FULL_CAMPAIGN_PERFORMED=NO
QUANTUM_TOUCHED=NO
CP05_D_ACCESSED=NO
HOLDOUT_ACCESSED=NO
```

The scientific candidate, seeds, RNG, retry semantics, fixtures and tolerances
are unchanged. The next step is Owner/architecture review of the local commit
identity and staged/committed byte hashes; the Owner controls any manual push.
The registry is the canonical evidence index; no separate index change is
required by `knowledge/evidence/README.md`.
