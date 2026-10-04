# CP05-C3A development null runner

Research implementation of the DEC-014 null matrix. The frozen contract has
1,056 cells: 576 AD/CVM primary cells and 480 KS/PEARSON comparators. Each real
cell requires 2,000 assessed eligible outers; B is exactly 199 or 999. Cell
identity and seeds include phase CP05-C and use the existing public development
namespace and stateless CP05-B seed derivation.

Gamma and Exponential composite cells use CP04 fitting. NB simple cells never
fit. NB composite cells import the unchanged PERF-01 fast CPU solver and expose
a research result protocol around the production fitted-distribution operations.
The research estimator ID, backend, warnings and explicit provenance identify
EXPERIMENTAL_FAST_CPU_FLOAT64 and PRODUCTION_BACKEND=NO. The adapter never
constructs a production FitResult with counterfeit canonical provenance.

`run_outer_unit` from CP05-B owns generation, observed fitting, bootstrap refits,
statistics, ties and the (b+1)/(B+1) p-value. This package supplies its NB fit
hook, adapts the output schema and orchestrates cells. NB mathematical observed
ineligibility permits the next raw outer until the 100*R_C cap. Numerical
failures, other-family failures, comparator nonassessment and inner cap
exhaustion stop the cell. Only assessed outers receive eligible indices or enter
Wilson. Inner NB mathematical ineligibility is redrawn by the unchanged core,
with the 100*B cap. Generating parameters never replace a composite refit.

NB composite entry canaries compare every observed attempt from raw index 0
through the first eligible matched fit (cap 64). They then generate bootstrap
samples from that fast fitted distribution using the normal inner seeds, through
the first matched eligible refit (cap 64). Eligible comparisons reuse PERF-01's
strict transformed-parameter and likelihood gates. Ineligible comparisons require
exact CP04 classification and convergence. There are no objective or decision
waivers. All samples are regenerated independently in the campaign; canary fits
and rows never enter its results. Observed-only periodic checks run at raw
indices divisible by 500, skipping already certified entry indices. Any mismatch
is terminal FAILED_SHADOW_ORACLE, with the raw identity and fitting evidence.

Parallelism is across cells. Within each cell, raw order is a sequential prefix;
this preserves stopping, eligible indices and retry accounting independently of
workers, batch grouping, configuration order or resume boundary. Atomic
`checkpoint.json` refers to immutable per-outer JSON segments with byte hashes
and record hashes. Resume checks scientific identity, raw prefixes, eligible
indices, inner accounting, seeds, caps, Monte Carlo arithmetic and shadow
certification before continuing. Worker/batch changes are allowed on resume.
Terminal failures are never retried. An exclusive `.execution.lock` prevents
concurrent writers. After an interrupted process leaves that lock, an operator
must verify the writer has stopped before removing the lock and resuming.

Cell bundles contain manifest, environment, outer/inner Parquet, summary,
metadata, shadow evidence and SHA-256 inventories. No full samples are retained.
Outer Parquet records also carry per-record digests. Bundle validation recomputes
accounting and unrounded Wilson intervals. The aggregator demands every expected
cell exactly once, rejects scientific use of software fixtures and reports
primary risk/shadow gates without pooling or choosing a method or B.

Performance fields separate primary outer time and shadow time. Total wall is
cumulative invocation wall through calibration and checkpoint preparation;
final bundle serialization is outside that measurement boundary. Median/p95
use assessed outer durations only. Fast fit count describes primary calls;
canonical shadow fit count describes canary/periodic calls. RSS is process RSS
sampled every 2 ms, including baseline; simultaneous cell workers share process
RSS, so this is an observed process peak, not attributable memory per thread.

The CLI supports `manifest`, `run` and `aggregate`; no scientific execution was
performed for this delivery. Future real execution requires separate Owner
authorization. Example that only writes the manifest:

```text
python -m experiments.distribution_gof.cp05_c manifest --root <output>
```

`run` accepts `--workers`, `--batch-size`, exact canonical JSON `--cell-id`
selectors and `--max-new-units` checkpoint boundaries. Without a selector it
addresses the full null matrix. `--fixture-target` explicitly requests a short
software fixture (1..1999), records that status and excludes it from scientific
aggregation. Custom generation/fitting/statistic hooks and oracle doubles are
accepted by the Python API only for explicit software fixtures. R_C in the
manifest is always 2000. `aggregate --root <output>` validates each expected cell
directory, then writes `null_matrix_summary.json`.

CLI source identity defaults to the running checkout's committed HEAD. Future
scientific runs require a clean committed checkout, the exact declared source
SHA, the authorized base/tree and only additive changes in the permitted research
paths. Python callers must supply that source SHA when building their manifests.
The matrix constructor's BASE_SHA default is useful for contract/software tests,
but cannot mislabel a real run from a later candidate. Resume binds the source
SHA independently of operational topology.

Power parameters are not preregistered completely. Power execution and method/B
selection are absent. CP05-D secrets, sealed matrices and samples are inaccessible
through this runner; its manifest accepts only CP05-C and the public namespace.
The required blocked-stage flags remain NO, including CP05_C_COMPLETE.
