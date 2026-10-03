# R11 CPU/CUDA decision-equivalence harness

This isolated package consumes the accepted frozen R11 payloads and delegates
fixed-sample numerical work to the existing scientific implementations. It adds
the B=199 Monte Carlo aggregator, DEC-024 exact-tie adjudication, strict complete
run gates, logical boundary fixtures, and exclusive evidence publication.

Implementation and software testing do not authorize a scientific run. Tests
use literal toy arrays, mocked evaluators/CUDA interfaces, synthetic archives,
and temporary Git repositories. No accepted R11 artifact is needed for tests.

## Frozen contract and identity

The accepted builder is commit
`1ace65bf9e01ab05df84a1cbca5031fac93bfa88`, tree
`0ba6d9804a7c46ab636cc47012091f7b9e80dcc6`.
The accepted workload SHA256 is
`77f53d616deeafcba51aa185dae8a8d2ca60ee4172f4a6353a3aa97b75324b16`.
Scientific SHA/tree, source manifest/projection hashes, and R4 archive,
crossings, and extracted historical record hashes remain the existing frozen
builder contract. Alpha is 0.05, B is 199, and the required topology is
12 outers / 200 records per outer / 2,400 records.

Future execution derives the harness SHA/tree from its actual clean checkout.
It requires the scientific and builder Git objects and trees, unchanged
pre-existing scientific files and an unchanged entire builder surface. Builder
provenance remains independent of the readiness implementation base:
`7891f17a3af77ebdc818896b31cfd12c4179b759`, tree
`31f93907b9f57956e09fb181749b340b108f5e6f`. That base includes accepted
DEC-029 governance history. Only the prospective diff from this base is scoped
to this package and its dedicated software test. The base must be an ancestor
of the executing commit; unrelated new paths or changes fail closed.

Project imports are lazy and confined to tracked, unchanged source files in the
executing checkout. Existing project modules and subsequently resolved project
imports receive origin checks. Git source comparisons respect checkout filters,
including Windows line-ending normalization.

## Payloads and numerical delegation

Preflight reuses `r11_reference_workload.oracle.load_for_use` for the complete
frozen schema, all payloads, source association, archive/crossings/extracted
record digests, and ordered historical R4 prefix. The additional harness gate
checks the exact accepted workload byte hash and accepted builder binding.

Traversal follows the source array as stored, then observed and accepted
ordinals 0 through 198. Prospective attempts, seeds and rejected draws are
provenance; they do not supply evaluated samples. There is no generation,
resampling, eligibility selection, checkpoint, retry or resume path.

The existing payload codec creates an ndarray backed by immutable bytes.
Dtype, shape and canonical raw-byte hash are checked before each engine and
again after each engine. Both receive the same sample object. CPU evaluation,
NB tail certification when applicable, CUDA evaluation, distribution grids,
and numerical gates delegate to existing accepted functions. The CUDA adapter
requires the existing CUDA primitive and never substitutes CPU mathematics.
The original B_EQ=15 aggregator is not used.

Delegated record identities must match the frozen descriptors. Drift fails;
the harness does not repair them by replacing them with expected identities.

## Adjudication and pass conditions

For each complete outer the new adjudicator derives all 199 indicators using
exact `>=`, then computes b, (b+1)/200 and rejection at p <= 0.05. Caller-supplied
aggregate fields are ignored and left unchanged.

A certified crossing requires finite CPU statistics with exact `==`, CPU True /
CUDA False, and valid observed/bootstrap identities, digests, classifications,
numerical gates, distribution evidence and CUDA convergence. Approximate
equality and neighboring representable floats do not certify ties.

Unexplained mismatches are **mismatch count minus certified crossing count**,
consistent with accepted DEC-024 and the required fixtures. The stray asterisk
in the pasted signed-accounting paragraph is interpreted as a formatting defect,
not multiplication. Signed b_cpu - b_cuda must equal the certified count, and
raw rejection decisions must agree. Thus certified b_cpu=10 / b_cuda=9 fails.

The same adjudicator evaluates all 12 complete logical B=199 fixtures.
A fixture passes when its result matches its expected disposition; expected
scientific-outer failures remain failures. Logical fixture evidence is explicitly
marked and is separate from scientific records.

Global PASS requires all accepted preflight gates, all 12 complete/adjudicated
outers, all 2,400 valid records, all 12 outer passes and rejection agreements,
all expected fixture dispositions, and zero structural, CUDA, convergence,
identity or unexplained failures. Partial traversal cannot pass.
The count of CPU b in {8,9,10,11} is diagnostic; zero is allowed.

## Evidence and failure behavior

DEC-029 readiness runs after frozen provenance, topology and logical fixture
checks, and before scientific runtime construction, output creation or payload
deserialization. Its isolated module loads CuPy/CuPyX lazily and exercises only
deterministic toy numbers: runtime/device, normal-loader `libnvrtc.so.13`, NVRTC
13.0, float64 arrays/elementwise/reduction, NVRTC RawKernel, all six required
special functions, sort and nextafter. Every asynchronous operation is
synchronized before PASS. Toy numerical checks are operational smoke checks;
they do not change scientific tolerances or establish scientific equivalence.

RawKernel uses explicit `backend="nvrtc"` and an explicit compile call. Every
invocation uses a new UUID in the actual kernel symbol/source, making the CuPy
source cache key distinct. Evidence records the invocation identity, source
SHA256, backend and synchronized compilation before launch and result checks.
An old kernel or an existing readiness artifact cannot satisfy a new invocation.
The source cache behavior is documented in the
[CuPy compiler source](https://github.com/cupy/cupy/blob/v13.6.0/cupy/cuda/compiler.py)
and the [RawKernel API](https://docs.cupy.dev/en/v13.6.0/reference/generated/cupy.RawKernel.html).

`--readiness-output` names an exclusive JSON file outside the repository and
outside scientific `--output`. The file is reserved before hardware access and
flushed/fsynced on PASS or FAIL when publication is possible. It includes host,
software/device versions, only the effective CUDA_PATH/LD_LIBRARY_PATH environment
values, every gate, fresh-JIT proof, harness SHA/tree and failure details.
No host repair, environment mutation, retry, cuRAND or cuSOLVER call is used.
Existing evidence is never overwritten. A publication failure stops execution
and retains any bytes already written.

Readiness failure leaves scientific execution unstarted, zero scientific
records, authorization unconsumed and the scientific directory uncreated.
On readiness PASS, future scientific execution binds execution_manifest.json
to `CUDA_READINESS_ORACLE_PASS=true` and the exact separate readiness artifact
SHA256. Readiness itself does not consume scientific execution authorization.

A new output directory must resolve outside the executing repository. Existing
output directories and duplicate artifact names are rejected. JSONL streams
are created exclusively and appended/flushed sequentially. Each raw result is
captured as detached immutable JSON bytes before later engine work.
Adjudication never mutates record or raw-result inputs.

The bundle publishes:

- execution_manifest.json
- reference_workload.json (exact input bytes)
- records.jsonl
- indicator_adjudication.jsonl
- outer_results.json
- boundary_fixtures.json
- environment.json
- summary.json
- digests.json

Finite numerical evidence round-trips through canonical JSON. Nonfinite
failure values receive explicit `nonfinite_float` tags instead of invalid JSON.
Opaque fitted objects and likelihood callables receive type/name metadata;
their numeric result fields are retained. Raw partial CPU/CUDA results survive
engine, certification, gate and adjudication failures.

Authorization is unconsumed throughout preflight and CPU-only work. Immediately
before the first CUDA scientific evaluator call, a write-once
authorization_consumed.json marker is published and the state becomes consumed.
Any subsequent failure preserves available evidence and stops. A publication
error reports the actual consumption state and retains files already written.
There is no automatic rerun, resume, repair, skip or tolerance widening.

digests.json covers every other published file, including the consumption marker
when present. Its own self-hash is explicitly excluded to avoid self-reference.

## CLI and software verification

`python -m experiments.distribution_gof.cuda_calibration.r11_decision_equivalence --validate-static`
checks the frozen dimensions and logical fixtures without loading R11 payloads,
constructing a scientific runtime or requiring SciPy/CuPy.

The non-scientific `--validate-readiness` mode requires `--workload`,
`--r4-archive`, `--r4-crossings`, `--readiness-output` and `--require-gpu`.
It may validate frozen schema/provenance but never deserializes samples or
constructs a scientific runtime/bundle. It prohibits `--output` and explicitly
reports zero scientific records and unconsumed authorization. Running this
hardware mode requires separate execution authorization; this implementation
was tested only with fakes.

The separate future `--execute` mode requires those same arguments plus
`--output`. There are no mutable scientific
contract options, option abbreviations or resume switches.

Run the three software suites:

```text
python -m unittest tests.research.test_cp05_c2c_r11_reference_workload tests.research.test_cp05_c2c_r11_canonical_adapter tests.research.test_cp05_c2c_r11_decision_equivalence -v
python -m knowledge.tools.validate_registry
python -m compileall -q experiments/distribution_gof/cuda_calibration/r11_reference_workload experiments/distribution_gof/cuda_calibration/r11_decision_equivalence tests/research/test_cp05_c2c_r11_reference_workload.py tests/research/test_cp05_c2c_r11_canonical_adapter.py tests/research/test_cp05_c2c_r11_decision_equivalence.py
git diff --check
```

The new suite covers the requested implementation contracts:

| Requirements | Coverage |
| --- | --- |
| Workload hash, R4 oracle, mandatory crossings/builder binding | PreflightTests |
| Dirty checkout, scientific/builder mutation, identity capture and import origin | GitIdentityTests |
| Frozen sample identity before both engines, no CPU fallback, delegated NB support | CanonicalDelegationTests |
| 199 records, derived aggregates, exact >= and ==, signed accounting, rejection and cancelling/unexplained failures | AdjudicationTests |
| All 12 boundary dispositions, nonfinite values and each required certification gate | AdjudicationTests |
| Stored traversal order, 2,400 mock evaluations, complete evidence and exact input copy | ExecutionSoftwareTests |
| Tampering, incomplete topology, failures before/after consumption, no retry and publication failure | ExecutionSoftwareTests |
| No regeneration/old aggregator, fixed B/alpha/hash, no resume/mutable CLI, import/static safety | StaticAndArtifactTests |
| Raw-result immutability, explicit nonfinite evidence and no overwrite | StaticAndArtifactTests |
| Full mandatory surface, loader/version failure, async/numerical failure, fresh source vs cached kernel, write-once evidence and environment isolation | ReadinessProbeTests |
| Readiness-only safety, failure before runtime/output/deserialization, sequencing, immediate consumption callback and manifest digest | ReadinessIntegrationTests |
| Accepted documentary history, distinct implementation base/ancestor, unchanged science/builder and rejection of prospective unrelated changes | GitIdentityTests |

No pre-existing scientific code or builder implementation is modified.
