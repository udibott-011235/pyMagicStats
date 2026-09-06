# CP-ANOVA-07D — D-robustness-h0 result

- Fecha: `2026-09-06`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07D`
- Phase: `D-robustness-h0`
- Harness SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Preregistration: `anova-calibration-prereg-v1.1`
- Decision: `EXECUTION_INTEGRITY_PASS / DESCRIPTIVE_ROBUSTNESS_MAPPED`

## Execution integrity

Frozen scope:

```text
54 cells
25,000 replicas/cell
1,350,000 paired datasets
2,700,000 method results
alpha grid = [0.01, 0.05, 0.10]
```

The supplied Quantum result reaches the final checksum block and returns to the shell prompt. The complete paired-accounting section contains all 54 cell IDs at all three alpha values, each with `replications_requested = replications_completed = 25,000`, and the four paired categories sum exactly to 25,000 for every cell/alpha combination.

Under the frozen harness, successful final publication implies `verify_directory()` completed. That verifier requires parity status PASS with 32 replicas/cell, exact identity/provenance, frozen seed/alpha/counts, exact row sequence, checksum validation and reconstructed accounting equality. Therefore this run is accepted as an execution-integrity PASS.

## Robustness classification at alpha = 0.05

The source terminal paste begins partway through the summary JSON, so the alpha=.05 method rates for the first missing summary records were reconstructed exactly from the complete paired counts:

```text
Classical rejection count = both_reject + classical_only
Welch rejection count     = both_reject + welch_only
```

This yields the complete 54-cell x 2-method map.

Frozen descriptive bands:

```text
green: abs(rate - .05) <= .01
amber: .01 < abs(rate - .05) <= .025
red:   abs(rate - .05) > .025
```

Counts:

```text
             green   amber   red   total
Classical       30       8    16      54
Welch           23      22     9      54
```

These are descriptive robustness classifications, not confirmatory authorization gates.

## Structural pattern

The most damaging design across several non-normal families is `R04`:

```text
sizes = [5,10,20]
sd    = [4,2,1]
```

where the smallest group carries the largest variance. Classical becomes severely liberal across many families. Examples at alpha=.05:

```text
gamma_shape_4       Classical 0.26416   Welch 0.08084
gamma_shape_1       Classical 0.26616   Welch 0.13372
lognormal_sigma_0p5 Classical 0.26272   Welch 0.09772
lognormal_sigma_1p2 Classical 0.28388   Welch 0.24860
student_t_df_5      Classical 0.24472   Welch 0.04772
student_t_df_3      Classical 0.23616   Welch 0.04348
mixture symmetric   Classical 0.21356   Welch 0.04668
contamination 5%    Classical 0.24284   Welch 0.11400
```

Thus heteroscedasticity direction and distributional shape interact strongly; neither Classical nor Welch can be authorized from a single variance diagnostic alone.

The reverse variance pattern `R05` (`sizes=[5,10,20]`, `sd=[1,2,4]`) often suppresses Classical rejection, while Welch may remain near nominal for some families but becomes liberal for strongly skewed families. This reinforces the need for a calibrated decision region rather than a one-rule fallback.

## Extremes at alpha=.05

Across the full 54-cell map reconstructed from paired accounting:

```text
Classical minimum = 0.00840  (DRH0-F06-R05, student_t_df_3)
Classical maximum = 0.28388  (DRH0-F04-R04, lognormal_sigma_1p2)
Welch minimum     = 0.01868  (DRH0-F04-R01, lognormal_sigma_1p2)
Welch maximum     = 0.24860  (DRH0-F04-R04, lognormal_sigma_1p2)
```

The lognormal sigma=1.2 family is especially hostile to both methods in small/adverse designs. Those observations remain descriptive evidence and do not modify the preregistered design.

## Decision

`D-robustness-h0` is complete for Phase D evidence collection.

No rerun, threshold change, seed change, family removal or selective replication increase is authorized.

Next:

```text
D-stress-h0     AUTHORIZED NEXT
D-power-h1      authorized after D-stress execution-integrity review
Phase H         SEALED
selector         NOT_CALIBRATED
```

The remaining phases continue on the exact same frozen harness candidate and preregistration.
