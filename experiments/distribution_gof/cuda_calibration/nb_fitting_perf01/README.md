# CP05-C2D-PERF-01 — Negative Binomial Fitting Performance Benchmark

Implementation Engineering delivery, based on
`2a8041579eaf4a4855c84525766f22770a6ba605` (tree
`7b3bd60c29a3aec4617a8efebaef5db64995a480`). Research-only FIT comparison:

| Engine | Implementation | Production backend |
| --- | --- | --- |
| A `CANONICAL_CPU` | Unmodified `_fitting.fit_negative_binomial`: Decimal score/likelihood, Brent, canonical pair validation | Current scientific baseline |
| B `EXPERIMENTAL_FAST_CPU_FLOAT64` | Independent NumPy/SciPy bounded profile solve in `fast_cpu.py` | `NO` |
| C `CUDA` | Unmodified `cuda_candidate.fit_negative_binomial` | Existing experimental candidate |

No production changes, public API wiring, automatic routing, DEC, R11 decision
re-adjudication, holdout access or architectural backend decision. This delivery
does **not** execute real PERF-01 or GPU performance on Quantum.

## Frozen input

The sole numerical input is the existing accepted R11 payload artifact:
SHA256 `77f53d616deeafcba51aa185dae8a8d2ca60ee4172f4a6353a3aa97b75324b16`,
12 source outers, `B_R11=199`, exactly 2400 records. The existing R11
`verify_workload` enforces this exact artifact hash, complete schema, source
manifest, builder identity, all payload digests and historical R4 archive/
crossings/prefix gates. PERF-01 adds NB family, record count and unique identity
checks. Neither the builder nor any generator is called. Samples deserialize
from immutable stored bytes, in observed/accepted order, preserving raw indices,
seeds, dtype, shape, identity and digest. There is no hash, seed, sample-size,
record-count or tolerance override on the CLI.

## Qualification and timing

The fixed order is A, B, C batch 1, 32, 128, 512, full-compatible-batch. Each CPU
engine warms up on the first stored record. Each CUDA configuration warms up on
the first `min(32, first compatible batch size)` stored records. Warm-up is
excluded from all measured durations/outcome counts and CUDA allocator peaks.
Each configuration then attempts **one** ordered pass over all 2400 records.

A and B's measured results become the complete scientific qualification evidence;
there is no second fitting campaign. For every record B must match classification
and convergence, produce successful finite fits, preserve sample identity and
digest, pass the existing `NB_TRANSFORM_ATOL=1e-8` checks on log(r)/logit(p), and
pass existing `NB_OBJECTIVE_RTOL=1e-9` likelihood agreement against A's actual
Decimal-based fitted likelihood. A solver failure also closes eligibility.
The existing flat-objective exception requires downstream gates; this FIT-only
benchmark does not apply it. No scientific tolerance is changed. Every failure,
including both raw fit results, is retained in the bundle. If B fails, its wall
time/throughput and a clearly labeled raw diagnostic speedup remain available;
its scientific speedups versus A/C are null. No B adjustment occurs inside a run.

CUDA only packs consecutive equal-shape/equal-dtype records; it never sorts,
pads, regenerates, drops or substitutes samples. A requested integer batch size
is a cap; tails or shorter compatible runs are explicitly reported. Full mode
uses each maximal contiguous compatible run. Resource/shape exceptions preserve
the failed batch and abort that configuration, without retrying or changing its
size. Later independently requested configurations still run once. Partial or
unavailable configurations have no usable throughput and cannot supply a best
CUDA time. Each completed CUDA configuration's fit gates are checked against A
outside the timed region; failed scientific gates exclude it from derived
speedups/best CUDA batch while keeping diagnostics.

**TOTAL HOST WALL TIME** spans each complete synchronous fitting pass, including
packing, transfers, solver validation and result download/conversion. It excludes
input file loading, imports, warm-up, gate adjudication and artifact writing.
CUDA calls `deviceSynchronize()` before and after the whole host region **and**
each transfer/compute/download region. Each stage also has CUDA events; event
durations include host submission gaps and are not presented as isolated kernel
time. Internal scalar/metadata transfers and synchronization inside the unchanged
candidate remain in the compute phase. Host stage durations are reported separately and need not sum exactly to
total host wall (which also includes result bookkeeping).

`cpu_process_seconds` is process CPU time during the same region. `peak_rss_bytes`
is a per-region 2 ms sampled RSS peak including baseline; brief transient peaks
may be missed. `gpu_memory_peak_bytes` is the maximum default CuPy memory-pool
**reserved** bytes observed after synchronous stages, after clearing warm-up
cached blocks. Cached solver intermediates contribute; allocations outside that
pool do not. These methods/limits are recorded with timings. Empty/unavailable
measurements are null. Environment evidence records CPU/RAM/platform, Python,
NumPy, SciPy, psutil, thread settings and, when requested, CUDA/CuPy/device data.

## Execution after audit and separate authorization

Requires the project CPU dependencies and psutil. CUDA execution additionally
requires the existing supported CuPy/CUDA environment. Benchmark imports and
`--help` do not import the CUDA candidate or initialize a GPU. Only `--cuda`
constructs the real CUDA adapter; it has no CPU fallback.

From a clean committed checkout, validation alone (no fitting/timing):

```bash
python -m experiments.distribution_gof.cuda_calibration.nb_fitting_perf01 \
  --workload /existing/reference_workload.json \
  --r4-archive /existing/cp05_c2c_r10a_targeted_r4_501d752_run2_PASS_EVIDENCE.tar.gz \
  --r4-crossings /existing/cp05_c2c_r10a_targeted_r4_501d752_run2_crossings.json
```

Only after audit and authorization for real execution, append:

```bash
--execute --cuda --output /new/directory/outside-checkout
```

The output directory must be new; it is never overwritten or resumed. Without
`--execute`, no performance campaign starts. Without `--cuda`, CPU-only diagnostic
execution reports CUDA configurations as not requested and `BENCHMARK_COMPLETE=false`.
Completion requires all seven configurations to finish their full pass;
scientific eligibility is reported separately. A complete campaign need not be
scientifically eligible. Final source/payload drift invalidates eligibility and
completion while retaining measured evidence. Source checks require the exact
base ancestor/tree, a clean checkout, additive changes only in this package and
the PERF-01 test file, and project modules loaded from the committed checkout.

## Minimal execution bundle

| File | Contents |
| --- | --- |
| `manifest.json` | Base/candidate identities, frozen workload hash/count, order, warm-up and pass policy |
| `scientific_eligibility.json` | B qualification, per-record gates and both raw results; preserves all failures |
| `timings.json` | One row per configuration, host/process/device/memory accounting, actual batches and failure reasons |
| `records_summary.json` | Ordered descriptors, per-configuration raw fits and CUDA FIT gate evidence |
| `environment.json` | Runtime/hardware/software/thread context |
| `summary.json` | Required wall times/throughputs, eligibility, allowed speedups, best eligible CUDA batch and completeness |
| `digests.json` | SHA256 of the other six files (no self-reference) |

Nonfinite failed results are retained as explicit `{"nonfinite": "nan"}`/
`inf` markers, never invalid JSON constants. `ARCHITECTURAL_DECISION=null`;
evidence review and any backend recommendation belong to Architecture.

## Local software tests

```bash
python -m pytest tests/research/test_cp05_c2d_perf01_nb_benchmark.py -q
```

Tests use literal toy transport data, mocked fits/events/resources and one tiny
literal B solver fixture. They never read the accepted payload artifact, run a
scientific equivalence/performance campaign, generate samples or require a GPU.
Next role: **Antigravity — adversarial implementation audit**.
