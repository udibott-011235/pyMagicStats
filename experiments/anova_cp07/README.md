# CP-ANOVA-07A calibration harness

Engineering implementation of `anova-calibration-prereg-v1.1`. The normative
sources are the CP06 preregistration, compact manifest, cell-ID/E0 amendment and
v1.1 freeze note under `knowledge/`. Production ANOVA and the selector are not
modified. A successful engineering test suite does not close CP-ANOVA-07 or
establish Type-I control/power. Next step: architect review (CP-ANOVA-07B).

## Setup and review

Run from the repository root using Python >=3.10 with the project dependencies:

```text
python -m pip install -r experiments/anova_cp07/requirements.txt
python -m experiments.anova_cp07 validate
python -m pytest tests/experiments/test_anova_cp07.py tests/test_anova_production.py tests/test_anova_oracle_adversarial.py tests/test_anova_location_stability.py -q
```

The engineering suite checks all 197 identities and field mappings. It evaluates
32 parity replicas for each of the 154 unsealed cells (4,928 paired datasets /
9,856 public API comparisons), plus small reproducibility probes. It does not
run E0 or D. Storage/resume/recomposition tests inject handcrafted results;
these are synthetic test fixtures, not calibration artifacts. The 43 H cells
are only inspected as configuration; mathematical transformations for H-only
families are tested with deterministic stubs, without a holdout RNG stream.

## Proposed E0 command — DO NOT EXECUTE before review/authorization

From a clean committed technical branch, in PowerShell:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
python -m experiments.anova_cp07 run --phase E0 --workers 2 --batch-size 50 --output ../anova-cp07-E0
```

The command fixes E0 at E0-01 through E0-12, 200 replicas each, CPU only.
No replication-count or cell-subset override exists. Both environment variables
before numeric imports and `threadpoolctl` around each computation enforce one
BLAS/OpenMP thread per process. Output should be outside the checkout to keep
the candidate clean for resume. No E0/D/H run was executed during implementation.

## Identity and reproducibility

- Compact JSON is pinned by its canonical semantic SHA-256; line-ending changes
  do not change the validator's source identity. The source file's actual byte
  hash is recorded separately in runtime provenance.
- Cross products follow the authorized ordered tables. Validation compares every
  complete record and the complete configuration against the pinned source and
  materialization. Missing, duplicate, extra, reordered or inconsistent records
  and altered statistical settings fail closed.
- UTF-8 `phase|cell_id|replicate_index` is SHA-256 hashed. The first 16 bytes are
  interpreted as four explicitly **little-endian uint32** values, passed after
  the master seed to `SeedSequence`. `Generator(PCG64(...))` is explicit.
- Replica indices are zero-based. Group order is the manifest order. Mixtures
  draw a membership vector then one normal vector per group. All families use
  theoretical centering/scaling; the H1 vector is centered linspace times delta.
- Each replica generates one tuple of groups, summarizes it once, then computes
  Classical and Welch on the same summary object before returning its record.
  The sample checksum includes group lengths and little-endian float64 bytes.
- Every run (including resume/recompose) passes public API parity first for every
  active cell, indices 0..31, rtol 1e-12 / atol 1e-14. Exceptions, nonfinite
  comparisons and mismatches abort. Parity samples are not added to accounting.

## Accounting

One Parquet row contains both method results, statuses, exception messages,
warning count and sample hash. Generation, summary/kernel and nonfinite-output
failures are explicit; an error is never encoded as a non-rejection. A summary
failure counts as a kernel-path error for both methods. The other kernel still
runs when one kernel fails. Warning counts are per paired replica and repeated
on each method's summary; they should not be added across methods.

Per-method `replications_completed` counts that method's finite successful
outputs. Paired `replications_completed` counts successes of both methods.
The four paired categories must sum to that independently counted denominator
for each alpha. Rejection uses strict `p < alpha`, matching production. Requested
counts refer to the replicas assigned to the current shard; metadata separately
records the frozen full-phase count per cell. A recomposed run has full counts.

Wilson 99% intervals use z=2.5758293035489004 and completed counts; MC SE is
sqrt(rate*(1-rate)/completed). Zero completed counts yield null rates/intervals.
Complete core summaries implement the preregistered confirmatory bands. Core
execution errors invalidate interpretation, preserve accounting artifacts and
cause the runner to raise. Partial shards cannot earn a confirmatory PASS.
Robustness bands and power monotonicity flags are descriptive only.

## Shards, transactions and resume

For shard s of S, every cell contributes indices `range(s, frozen_reps, S)`.
Workers and batch size affect scheduling only. Output records are in cell-ID /
replica-index order. Each invocation writes one shard to its own output directory.
Completed directories contain exactly the seven required artifact names, with
the shard encoded in `anova_calibration_replicates-00000-of-00001.parquet`.

All files are written and verified in a sibling transaction directory, then
published by directory rename. A completed directory is immutable: `--resume`
only returns it after checking provenance, exact file inventory, SHA-256 checksums,
embedded Parquet provenance, replica coverage and recomputed metadata accounting.
It never appends replicas or overwrites corrupt output. Ordinary interruptions
clean the transaction; a process kill can leave an unpublished `.pending-*`
directory, which is not consumed by resume. Resume of an unpublished shard
recomputes that shard deterministically. Completed shards are never recomputed.

Recompose with `python -m experiments.anova_cp07 recompose --output <new-directory>
<shard-directory-0> <shard-directory-1> ...`. It requires exactly one of every
shard and matching harness/engine/manifest/version/runtime identity; it streams
the sorted union and recomputes all integer counts, rates and intervals. It
rejects missing/duplicate shards and overlapping/missing/unexpected replica keys.
Shards from different workers/batches may combine; shards from different
dependency versions or hosts do not combine under this conservative identity.

Runtime metadata records harness commit, frozen engine commit/blob, effective
preregistration version, full materialized manifest SHA-256, source byte hash,
Python/dependency versions, OS/CPU, phase, seed, alpha, workers/batch/shards,
thread settings, UTC times, parity and all error/warning counts. Parquet schema
metadata and CSV summary/disagreement columns carry provenance. The manifest is
the exact materialized configuration whose hash is referenced by these artifacts.
Metadata contains checksums of the other six artifacts; it is not self-hashed.
Checksums detect accidental corruption, not malicious rewriting of an entire
artifact directory by a party able to replace both data and checksums.

## Sealed holdout

All H phases are supported in materialization, generators, accounting, workers
and persistence, but reject execution by default before generating a sample.
There is no boolean `--allow-holdout` bypass. A future PO-authorized opening uses
`--holdout-authorization <declaration.json>`. The declaration must identify
`action: "open-holdout"`, `authorized_by: "Product Owner"`, exact `harness_sha`
and `manifest_sha256`, and explicitly assert `phase_d_complete`,
`remediations_closed`, `candidate_frozen` as JSON true. Its byte hash is recorded.
This is an explicit operational authorization record, not a cryptographic proof
of PO identity or a substitute for architectural/governance review. No such valid
declaration was created or exercised in this checkpoint. H uses master seed
2026090599 only after opening; E0/D use 2026090501.
