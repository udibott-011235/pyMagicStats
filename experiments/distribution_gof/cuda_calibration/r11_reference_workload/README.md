# R11 CPU reference-workload builder machinery

This isolated research package implements only construction, payload transport,
source binding and provenance gates from accepted
[DEC-027](../../../../knowledge/decisions/cp05-c2c-r11-boundary-sensitive-decision-equivalence-preregistration.md)
and [DEC-028](../../../../knowledge/decisions/cp05-c2c-r11-r4-runtime-oracle-identity-correction.md).
The Owner's implementation-only instruction authorizes this candidate.
Real construction, scientific sample generation, numerical comparison, CUDA,
full-1152, CP05-D and holdout access remain separately authorized phases.

No import starts execution. There is no command-line execution entry point,
comparison harness, CuPy import, RNG implementation or scientific fitting
implementation. No pre-existing scientific file is changed.

## API and future human binding

- `SourceSurface.frozen(manifest_bytes)` verifies the exact R4 manifest bytes,
  12 stored outers and normative 4250-byte ordered projection. It retains
  immutable source bytes and reconstructs fresh metadata dictionaries in stored
  order. It never generates a sample.
- `CPUAdapters` describes injected canonical interfaces. A human-authorized
  execution environment must supply bindings to the **existing frozen**
  observed construction, `reference_fit`, `derive_seed` and CPU `_generate`
  semantics. These are the existing interfaces inspected in
  `cp05_c2c_equivalence_runner.py` / `cp05_cuda_engine.py` at the frozen
  scientific SHA; this package does not import those modules because their
  transitive imports attempt CuPy. Adapter wiring and scientific execution are
  not provided or tested on the real surface in this candidate.
- `observed(row, namespace)` returns the canonical ndarray and integer observed
  seed. `reference_fit(family, sample)` returns the existing fit dictionary
  with `engine="CPU_REFERENCE"` and `parameters`. `derive_seed` receives the
  exact five canonical arguments including `"inner_bootstrap"`.
  `generate(row, fitted_parameters, seed)` passes the CPU fit parameters to
  the existing generator using the row's canonical cell metadata.
- `canonical_error_type` must be the existing custom `EngineContractError`
  in a real binding. Only an exception of that type, in an NB bootstrap fit,
  with the canonical `NB_NOT_ASSESSED:` prefix is ineligible. Other failures
  terminate the single invocation. An NB observed-fit failure also terminates.
  No exception is converted into that scientific classification.
- `ReferenceWorkloadBuilder.build(source, adapters, builder_binding=...)`
  visits raw inner indices from zero, retains every attempt and accepts exactly
  199 eligible samples. NB is limited to 19900 raw attempts; other canonical
  families have 199 attempts and fail on any unsuccessful construction. An
  eligible final permitted attempt may complete the outer. Accepted ordinals
  start at zero and follow eligible traversal order. It does not compare
  numerical CPU/CUDA results or calculate p-values.
- A builder instance is consumed before its first preflight. Neither success
  nor failure can be automatically rerun/resumed on that instance.
  `BuildFailure.artifact_bytes` contains immutable available diagnostic
  evidence, including the failure stage, observed payload if available, prior
  records and the failed attempt's seed/digest when available. Preflight
  rejection occurs before construction and raises `ContractError`.
- `binding_from_git(repository)` reads a clean committed builder's SHA/tree,
  verifies the frozen scientific tree and rejects changes to pre-existing
  scientific paths. `builder_binding=None` is an explicit placeholder.
  `bind_artifact` returns new bytes bound to a supplied SHA/tree and refuses
  replacing an existing binding. A human should obtain this binding with
  `binding_from_git` before publishing a frozen artifact.
- Environment metadata uses the actual Python version and installed NumPy/SciPy
  distribution versions. Missing SciPy rejects real frozen construction;
  synthetic tests record `UNAVAILABLE` rather than inventing a version.

## Artifact and integrity

The versioned canonical UTF-8 JSON document embeds the exact source manifest
bytes as base64, scientific SHA/tree, source hashes, builder binding,
environment and all ordered outer records. Each observed/accepted record
retains its identity, integer seed and a payload containing `dtype.str`,
original ndarray shape, base64 C-order raw bytes and their exact SHA-256.
Accepted records also include cell/outer/inner identities and accepted ordinal.
Attempts retain raw index, seed, digest, CPU eligibility status and reason.
Outer counts, retry-cap status and completion/failure states are explicit.

`SamplePayload` separates canonicalization, digest computation, serialization,
deserialization and verification. Only actual CPU NumPy arrays are accepted;
object/structured dtypes and implicit device-array conversions are rejected.
No numeric precision, endianness, values or eligibility rule is changed.
Deserialization validates dtype, shape, byte length and raw-byte digest before
returning a read-only, bytes-backed array. Scientific adapters receive these
frozen arrays, preventing mutation of the persisted sample.

The workload has an ordered payload digest over the observed record followed
by accepted records in each outer. For each record it hashes three fields,
each preceded by its unsigned 8-byte big-endian length: canonical record
metadata excluding payload, canonical dtype/shape metadata, and exact payload
bytes. The document also hashes the canonical entire workload, including
attempt provenance. `write_once(path, artifact_bytes)` creates exclusively and
returns the artifact's independent SHA-256; it refuses overwrite or resume.

`load_artifact` verifies canonical JSON, duplicate keys, finiteness, schema,
identities, counts, raw/accepted ordering, eligibility accounting, source
binding and payload integrity. Default loading rejects unbound, incomplete
or synthetic artifacts. Diagnostic loading is explicit; it never permits
numerical evaluation. A software completion status is construction status,
not scientific acceptance.

## Historical prefix gate

`compare_prefix` is a pure, mock-testable comparison. It checks observed
identity/seed/digest and the first 15 eligible bootstrap
identity/raw-index/order/seed/digest associations against the source manifest
and historical records. It cannot verify an archive on its own.

`load_for_use` composes complete frozen schema/payload integrity, the exact
source manifest, historical archive SHA-256, exact extracted workload JSONL
SHA-256 and prefix comparison. A supplied crossings report is independently
verified against its own SHA-256. Archive, crossings and workload hashes can
never substitute for each other. The archive is read in memory, with exactly
one regular workload member required; files are not extracted to disk.

Absent, malformed or mismatching history raises `OracleError` with
`R4_RUNTIME_ORACLE_VERIFIED=NO`,
`R11_REFERENCE_WORKLOAD_ACCEPTED=NO`, `GPU_EXECUTION=PROHIBITED`.
Successful integrity loading exposes records for a separately authorized
future phase; it makes no scientific equivalence or empirical acceptance claim.

## Software verification only

Run the isolated standard-library suite from the repository root:

```text
python -m unittest tests.research.test_cp05_c2c_r11_reference_workload -v
python -m knowledge.tools.validate_registry
python -m compileall -q experiments/distribution_gof/cuda_calibration/r11_reference_workload tests/research/test_cp05_c2c_r11_reference_workload.py
git diff --check
```

Every scientific call in the suite uses deterministic fake adapters and tiny
fixed arrays. No RNG, canonical scientific fit, real observed reconstruction
or scientific sample generation is invoked. The real R4 manifest is read only
for the expressly permitted documentary hash/order test. Synthetic artifacts
are tagged `SYNTHETIC_TEST` and rejected by the scientific loader.

The historical external runtime archive was not retrieved or scientifically
checked in this implementation phase. Actual prefix acceptance and canonical
adapter execution remain untested and require the later human-authorized phase.
The current software-test runtime has Python/NumPy but no SciPy/pytest; the
suite uses `unittest` and no scientific dependency. No registry, historical
decision or historical evidence file is modified.
