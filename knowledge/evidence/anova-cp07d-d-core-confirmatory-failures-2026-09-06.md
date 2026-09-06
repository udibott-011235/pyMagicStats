# CP-ANOVA-07D — D-core confirmatory failures

- Fecha: `2026-09-06`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07D`
- Subcheckpoint: `D-core-h0`
- Harness SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Preregistration: `anova-calibration-prereg-v1.1`
- Replications: `50,000/cell`
- Decision: `PAUSE_FOR_WELCH_FINITE_SAMPLE_ADJUDICATION`

## Result

The supplied Quantum transcript reaches the completed artifact checksum block and contains eight explicit preregistered confirmatory `FAIL` rows. Every observed confirmatory failure is Welch at `alpha=0.05`; no Classical confirmatory failure appears in the supplied transcript.

The transcript begins mid-JSON, so this evidence does not infer the exact total count of PASS rows from the pasted terminal text alone. The eight FAIL rows are nevertheless sufficient to trigger the preregistered stop/adjudication condition before any later Phase D stratum is launched.

## Failed cells

```text
cell_id       design / SD                                 Welch Type-I   Wilson 99% CI
DCEV-k3-07    n=[5,30,30], sd=[1,1,1]                    0.05962         [0.056950, 0.062406]
DCEV-k3-08    n=[30,5,5], sd=[1,1,1]                     0.05800         [0.055366, 0.060752]
DCEV-k5-05    n=[5,8,12,20,30], sd=[1,1,1,1,1]           0.06182         [0.059104, 0.064653]
DCEV-k5-06    n=[30,20,12,8,5], sd=[1,1,1,1,1]           0.05762         [0.054994, 0.060363]
DCUV-k10-01   n=[10]*10, sd=[1,4,1,4,1,4,1,4,1,4]       0.05922         [0.056559, 0.061998]
DCUV-k3-04    n=[5,10,20], sd=[4,2,1]                    0.05894         [0.056285, 0.061712]
DCUV-k3-08    n=[5,30,30], sd=[4,1,1]                    0.05800         [0.055366, 0.060752]
DCUV-k5-03    n=[5,8,12,20,30], sd=[4,3,2,1.5,1]        0.06532         [0.062531, 0.068224]
```

All listed rows report:

```text
replications_requested = 50000
replications_completed = 50000
generation_error_count = 0
kernel_error_count = 0
nonfinite_count = 0
warning_count = 0
```

## Interpretation boundary

This is a **statistical calibration failure of the preregistered Welch authorization region**, not yet evidence of an implementation defect.

Reasons not to label this a code bug at this point:

1. CP-ANOVA-05 deterministic oracle work already established pointwise agreement of the production Welch kernel with the independently specified formula and reference implementations over deterministic scenarios.
2. The failures are structured: they concentrate in small/unbalanced designs, moderate/high `k`, and adverse variance-size patterns rather than appearing randomly across all normal cells.
3. Several failures occur even when population SDs are equal but group sizes are highly unequal. Under Gaussian equal-variance conditions Classical ANOVA retains its exact F model, whereas Welch uses a finite-sample approximation and need not inherit Classical's exactness.
4. Published Monte Carlo literature reports that Welch/W procedures can have unsatisfactory Type-I control for small samples and/or a moderate-to-large number of groups, and older simulation work specifically reports that with equal variances but unequal sample sizes W may be inferior to the Classical F procedure.

The preregistered acceptance band must **not** be relaxed after observing these results. No selective rerun is authorized.

## Required adjudication before continuing Phase D

Next: targeted, non-calibration oracle adjudication on the eight failed cell designs.

Purpose:

- verify the exact production Welch p-values against independent SciPy and statsmodels implementations on deterministic replicas from those same cell definitions;
- confirm that the observed Type-I inflation belongs to Welch finite-sample behavior rather than a pyMagicStats implementation divergence;
- do not generate a replacement 50k calibration and do not change any threshold.

If pointwise oracle parity passes, classify the eight failures as a **finite-sample authorization boundary** for Welch and then continue the remaining preregistered Phase D strata unchanged to map robustness/stress/power.

If pointwise oracle parity fails, reopen CP-ANOVA-04/05 as an implementation blocker before any further Monte Carlo.

Until adjudication is closed:

```text
D-core-h0          EXECUTED / CONFIRMATORY FAILURES PRESENT
D-robustness-h0    PAUSED
D-stress-h0        PAUSED
D-power-h1         PAUSED
Phase H            SEALED
selector            NOT_CALIBRATED
```
