# CP-ANOVA-07D — Welch finite-sample adjudication

- Fecha: `2026-09-06`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07D`
- Harness SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Prior evidence: `anova-cp07d-d-core-confirmatory-failures-2026-09-06.md`
- Decision: `WELCH_FINITE_SAMPLE_BEHAVIOR_CONFIRMED`

## Oracle adjudication

Cortex adjudicated the eight preregistered Welch confirmatory failures on 96 deterministic samples drawn under the frozen candidate and seed contract.

Reported oracle evidence:

```text
samples = 96
F/p/df comparison groups = 432
scalar comparisons = 1,728
mismatches = 0
max relative error = 7.8017e-15
max absolute error = 7.1054e-15
warnings = 0
exceptions = 0
```

The candidate Welch kernel, public `WelchANOVA`, frozen independent formula, statsmodels Welch implementation and SciPy Welch implementation (where available) agreed pointwise within the preregistered deterministic tolerances.

For the four equal-population-variance failed designs, Classical also agreed with SciPy Classical.

## Adjudication

The eight D-core confirmatory failures are therefore classified as **finite-sample behavior of Welch ANOVA under the tested designs**, not as a pyMagicStats implementation defect.

The following consequences are frozen:

1. Production ANOVA code is not reopened by this finding.
2. CP-ANOVA-04/05 remain closed for implementation correctness.
3. The preregistered `[0.04, 0.06]` confirmatory band is not relaxed.
4. The eight FAIL results remain evidence exactly as observed.
5. No selective rerun, seed change or replication increase is authorized.
6. Welch is not authorized as a universal heteroscedastic fallback based on this evidence.
7. A future selector must incorporate an empirically calibrated finite-sample authorization region rather than treating the explicit Welch implementation as automatically valid for every `(k, n_i, variance pattern)`.

## Phase D continuation

The purpose of the remaining development strata is now to map behavior outside the exact normal/core region and characterize power, not to rescue the failed D-core cells.

The remaining frozen Phase D strata are authorized unchanged:

```text
D-robustness-h0   54 cells x 25,000 = 1,350,000 paired datasets
D-stress-h0       10 cells x 25,000 =   250,000 paired datasets
D-power-h1        36 cells x 20,000 =   720,000 paired datasets
```

Execution order remains:

1. `D-robustness-h0`
2. `D-stress-h0`
3. `D-power-h1`

After each stratum, inspect parity/provenance/accounting/persistence before starting the next. Statistical red/amber robustness results are evidence and must not trigger selective reruns.

## Current state

```text
D-core-h0          complete / 8 Welch confirmatory FAILs retained
Welch adjudication complete / finite-sample behavior confirmed
D-robustness-h0    authorized NEXT
D-stress-h0        authorized after D-robustness execution integrity check
D-power-h1         authorized after D-stress execution integrity check
Phase H            SEALED
selector            NOT_CALIBRATED
```
