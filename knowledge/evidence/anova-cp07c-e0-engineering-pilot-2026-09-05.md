# CP-ANOVA-07C — E0 engineering pilot PASS

- Fecha: `2026-09-05`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07C`
- Harness branch: `engineering/cp-anova-07a-harness`
- Harness SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Engine SHA: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- Engine blob: `2d00ae2a2812b8c390125fefe244dcb4830176c5`
- Preregistration: `anova-calibration-prereg-v1.1`
- Manifest SHA-256: `affa3a1ae3c02b8081d0bdc761e6ce3725bb736899b0d2771d5d185530c0262a`
- Decision: `E0_ENGINEERING_PASS`

## Execution

E0 ran on Quantum from the named technical branch, clean candidate checkout, CPU backend, two workers, batch size 50, and all BLAS/OpenMP-related thread environment variables pinned to 1.

Frozen E0 scope:

```text
12 cells
200 replicas per cell
2,400 paired datasets
4,800 method results (Classical + Welch)
```

Runtime environment reported by the harness:

```text
Python 3.12.3
NumPy 2.5.2
SciPy 1.18.1
statsmodels 0.15.0
pandas 3.0.5
pyarrow 25.0.1
threadpoolctl 3.6.0
Linux 6.8.0-138-generic x86_64
```

## Parity gate

```text
status = PASS
cells = 12
replicas_per_cell = 32
comparisons = 768
rtol = 1e-12
atol = 1e-14
warning_count = 14
```

No parity mismatch occurred. The 14 warnings were confined to the parity/public-API diagnostic path and did not block or alter paired Monte Carlo execution.

## Monte Carlo execution accounting

The published run reports:

```text
execution_status = ACCOUNTED
paired datasets requested = 2,400
paired datasets completed = 2,400
method results requested = 4,800
method results completed = 4,800
generation errors = 0
kernel errors = 0
nonfinite outputs = 0
Monte Carlo warning count = 0
```

At alpha 0.05, paired category totals across all 12 cells were:

```text
both_reject       = 197
classical_only    = 135
welch_only        = 84
neither           = 1,984
TOTAL             = 2,400
```

The paired accounting invariant is therefore satisfied exactly.

## Persistence

The run published all required E0 artifacts and reported SHA-256 checksums for:

- `anova_calibration_manifest.json`
- `anova_calibration_summary.parquet`
- `anova_calibration_summary.csv`
- `anova_calibration_replicates-00000-of-00001.parquet`
- `anova_calibration_disagreement.csv`
- `anova_calibration_report.md`

The runner's transactional publication path verifies the directory before exposing the completed output.

## Engineering sanity observations — NOT inferential evidence

E0 was explicitly not sized for statistical authorization. Its rejection rates are recorded only as engineering sanity checks that scenario construction and paired accounting are sensitive to intended design differences.

Examples at alpha 0.05:

```text
E0-01 normal/equal SD/H0:
  Classical 0.045
  Welch     0.025

E0-02 normal/heteroscedastic sizes [5,10,20], SD [4,2,1]/H0:
  Classical 0.290
  Welch     0.085

E0-03 normal/heteroscedastic sizes [5,10,20], SD [1,2,4]/H0:
  Classical 0.015
  Welch     0.070

E0-06 Student-t df3, sizes [5,10,20], SD [4,2,1]/H0:
  Classical 0.225
  Welch     0.045

E0-10 lognormal sigma 1.5/H0:
  Classical 0.260
  Welch     0.335

E0-11 normal/H1/delta 0.50:
  Classical 0.125
  Welch     0.125

E0-12 gamma shape 1/H1/delta 1.00:
  Classical 0.515
  Welch     0.520
```

These values do not authorize a selector, prove robustness, establish Type-I control, or define a preferred method. They show that the harness captures known/adversarial structural differences instead of collapsing both methods into identical accounting.

## Decision

`CP-ANOVA-07C` is **complete/PASS as an engineering checkpoint**.

This closes only the engineering pilot. It does not close calibration and does not open holdout.

Next checkpoint is `CP-ANOVA-07D — Phase D development calibration`, subject to architect scheduling/authorization and resource coordination with the still-running proportion-CI calibration on Quantum.

Phase H remains sealed.
