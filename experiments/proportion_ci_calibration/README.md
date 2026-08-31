# CP-06 proportion-CI calibration harness

This directory contains the isolated experimental harness for
`STAGE-PROP-CI-001/CP-06` over candidate
`2df5b90a5395163e723f9c52aafbb91fdce96d43`.

Production endpoints are evaluated through
`PopulationProportionCI.from_counts`. Jeffreys is implemented only here and is
labelled `bayesian_comparator`. The harness does not modify production metadata,
capabilities, or routing.

The deterministic smoke checkpoint exercises several small sample sizes and
values near 100 without starting CP06-B:

```text
env -u PYTHONPATH python -m experiments.proportion_ci_calibration.run SMOKE --workers 2 --batch-size 64
```

Recommended Linux/Quantum launch for CP06-B, after reviewing the smoke
artifacts:

```text
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e . -r experiments/proportion_ci_calibration/requirements-quantum.txt
env -u PYTHONPATH python -m pytest tests/experiments/test_proportion_ci_calibration.py -ra --durations=20
env -u PYTHONPATH python -m experiments.proportion_ci_calibration.run B --workers 8 --batch-size 256
```

`--workers 8` creates eight independent `n` shards and `--batch-size 256`
bounds each SciPy PMF matrix. Completed `n` shards are cached atomically in the
system temporary directory and can be resumed without changing committed
evidence. The support is selected with a deterministic Hoeffding bound of
`1e-14`; it does not call `scipy.stats.binom.isf` or another root-finding
quantile routine.

The v2 harness routes coverage by endpoint structure. Monotone methods use the
contiguous CDF/SF path. A nonmonotone endpoint grid, including affected Wald
configurations, is evaluated as the explicit set
`{x: lower[x] <= p <= upper[x]}` with chunked log-PMF accumulation. It is never
sorted or collapsed to a forced range. Wald outputs also include probability-
weighted `P_outside` and `P_degenerate` summaries and worst cases.

The heavy checkpoints are separate, deterministic, and resumable. Do not run
them as part of harness validation:

```text
# CP06-C: n=1..5000, full preregistered p domain
env -u PYTHONPATH python -m experiments.proportion_ci_calibration.run C --workers 8 --batch-size 256

# CP06-D: the 11 stress n values through 1,000,000
env -u PYTHONPATH python -m experiments.proportion_ci_calibration.run D --workers 1 --batch-size 64

# CP06-E: endpoint partitions plus analytic or bounded adversarial candidates
env -u PYTHONPATH python -m experiments.proportion_ci_calibration.run E --workers 8 --batch-size 256

# CP06-F: audit the C/D/E trigger queue at 80 decimal digits
env -u PYTHONPATH python -m experiments.proportion_ci_calibration.high_precision --checkpoints C D E --workers 4 --digits 80
```

The pre-v2 CP06-B shards and Wald evidence are incompatible and must not be
reused. After B is rerun with this harness, C may reuse only shards whose
candidate SHA, harness/schema version, checkpoint hash, per-shard semantic
hash, payload hash, and interval hash all validate. D uses one worker because a
single `n=1_000_000` interval inventory is intentionally memory-heavy.

Endpoint grids have a separate atomic cache keyed and validated by candidate,
harness/schema version, `n`, alpha, method, and SHA-256. This lets E reuse the
API-produced grids from C/D. E excludes the base grid and expected-width
calculation; it persists `proportion_ci_cp06_e_adversarial_minima.parquet` with
the global minimum and reconstructible acceptance runs for every
`(method, alpha, n)`. Single-run regions use the analytic stationary candidate;
multi-run regions use a deterministic bounded optimizer with `xatol=5e-13`.
F refuses fewer than 80 decimal digits and consumes explicit runs without
filling their gaps.

The interval inventory records canonical SHA-256 hashes and row counts instead
of committing the enormous raw endpoint grid. Coverage, event-regime, oracle,
invariant, and worst-case summaries are persisted incrementally as Parquet.
