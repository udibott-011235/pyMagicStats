# Student t versus uncorrected empirical likelihood calibration

This standalone package runs a paired Monte Carlo calibration for a one-sample
arithmetic mean. Every replicate generates one sample and applies both the
two-sided Student t procedure and the existing, uncorrected production
empirical-likelihood test/CI to that same sample. It generates evidence only:
it does not select methods, calibrate routing, implement Bartlett correction,
or run bootstrap-t.

## Scientific domain and sealed holdout

`scenarios.py` imports the scenario objects and complete cell matrix from
`experiments.adversarial_robustness_calibration`. That existing module remains
the source of truth for distribution parameterizations, population means, and
sample sizes. The experiment adds no distribution or cell.

Before lookup or generation, a fail-closed guard rejects every reserved sample
size, lognormal sigma, Student-t df, contamination epsilon, and family in the
sealed holdout policy `sealed-blind-holdout-v1`. Every root, shard, block, cell,
and aggregate manifest records `"holdout_used": false` and the root/aggregate
metadata include the complete exclusion policy.

## Reproducibility and sharding

`--replicates-per-cell R` means **R total global replicates per cell**, not R per
shard. Shard `s` of `S` owns exactly the global IDs `s, s+S, s+2S, ... < R`.
This modulo allocation is independent of machine, order, and wall-clock time.

Each seed identity is BLAKE2b-derived from the master seed, canonical scenario
ID, shard ID, and global replicate ID using canonical JSON. Python `hash()` is
never used. Each sample can be reconstructed from the recorded seed identity,
backend metadata/version, scenario metadata, and replicate coordinates. CPU
and GPU RNG streams need not be bit-identical, but each backend is internally
deterministic. A paired-sample fingerprint is persisted as an additional audit
check.

## Running shards on Quantum

From the repository root:

```bash
python -m experiments.el_vs_t.run_calibration \
  --replicates-per-cell 50000 \
  --master-seed 20260829 \
  --backend auto \
  --workers 12 \
  --batch-size 2048 \
  --shard 0 \
  --num-shards 20 \
  --output outputs/el_vs_t
```

Run the command once per shard with a distinct `--shard`. Optional repeated
`--scenario` and `--sample-size` flags restrict smoke runs, but may select only
canonical non-holdout cells. A rerun verifies and skips checksum-complete
blocks. `--force` is the only way to recompute them. A block is first written
to a temporary file, flushed, atomically renamed, checksummed, and only then
given a completion marker. Checkpoints bound sample memory to approximately
`batch_size * n * 8` bytes plus method outputs.

For multi-process CPU work, prevent BLAS oversubscription. The runner sets the
three limits below to one before spawning workers; setting them explicitly in
the server job is still clearer:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## Storage and aggregation

`--format auto` uses Parquet when `pyarrow` or `fastparquet` is installed and
otherwise uses portable, deterministic `CSV.gz`. Replicates are stored by
shard/scenario/sample-size/block; they are never all held in memory. Aggregation
loads one cell at a time and never reruns simulation:

```bash
python -m experiments.el_vs_t.aggregate_calibration \
  --input outputs/el_vs_t \
  --output outputs/el_vs_t/summary
```

It fails on missing or duplicate shards/replicate IDs, bad checksums, missing
blocks, wrong ownership, wrong seed identity, repository SHA mismatch, alpha or
confidence mismatch, scenario-registry mismatch, method-version mismatch, or
any non-false holdout status. It produces `el_vs_t_summary.csv`,
`el_vs_t_disagreement.csv`, `el_vs_t_metadata.json`, and the mechanically
generated `el_vs_t_report.md`.

## CPU and optional GPU paths

The required CPU path uses Python, NumPy, SciPy, pandas, and pyMagicStats. No
network, API key, LLM, Codex, or interactive IDE is used. Parquet is optional;
without a Parquet engine, `CSV.gz` works with the required dependencies.

CuPy is the only optional GPU dependency. Install the CuPy package matching
the server CUDA runtime (for example `cupy-cuda12x` where appropriate); it is
deliberately not a required project dependency. Explicit `--backend gpu` fails
clearly when CuPy/CUDA is unavailable. `--backend auto` falls back to CPU and
uses GPU generation/diagnostic moments only for batches of at least 250,000
sample elements, avoiding launch/transfer overhead on small cells. GPU data
moves to host once per batch, never observation by observation.

Student t, the production `empirical_likelihood_mean_test`, the production
`empirical_likelihood_mean_ci`, and all scalar SciPy root solving intentionally
stay on CPU. No GPU EL solver exists in this harness.

Measure rather than assume acceleration:

```bash
python -m experiments.el_vs_t.benchmark \
  --scenario normal --sample-size 30 --replicates 20
```

The benchmark reports generation, diagnostics, device transfer, Student t,
EL test, EL CI, serialization, and end-to-end timings separately. It records
GPU as unavailable when appropriate and makes no speedup claim unless both
paths were actually measured.

## Statistical output contract

Per-cell summaries expose relevant denominators, Type I error and MCSE,
coverage and MCSE, CI-width mean/median/quantiles, and numerical-failure rates.
EL adds hull-outside, boundary, nonregular, and solver-failure rates. Paired
outputs count/rate both/t-only/EL-only/neither rejection and coverage outcomes,
plus finite regular width ratios and paired differences. Descriptive sample
moments are observational only and never influence execution or grouping.

The report explicitly separates **OBSERVATION**, **INTERPRETATION**, and
**POLICY — NOT DETERMINED**. No routing threshold can be inferred or activated
by this code path.
