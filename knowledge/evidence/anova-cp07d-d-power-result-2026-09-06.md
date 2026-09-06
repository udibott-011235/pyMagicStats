# CP-ANOVA-07D — D-power-h1 result

- Fecha: `2026-09-06`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07D`
- Phase: `D-power-h1`
- Harness SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Engine SHA: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- Preregistration: `anova-calibration-prereg-v1.1`
- Decision: `EXECUTION_INTEGRITY_PASS / POWER_SANITY_PASS`

## Execution integrity

Frozen scope:

```text
36 cells
20,000 replicas/cell
720,000 paired datasets
1,440,000 method results
alpha grid = [0.01, 0.05, 0.10]
```

Quantum reported:

```text
execution_status = ACCOUNTED
parity = PASS
parity cells = 36
parity replicas/cell = 32
parity comparisons = 2,304
parity warnings = 0
workers = 12
batch_size = 200
backend = cpu
```

All 216 method/alpha rows report `replications_requested = replications_completed = 20,000` with zero generation errors, zero kernel errors, zero nonfinite outputs and zero Monte Carlo warnings. The 108 paired-accounting rows satisfy the four-category accounting invariant exactly. Six final artifact checksums were published.

## Power monotonicity

The harness defines `power_monotonicity_flag=True` only if, within the same family/design/method/alpha, rejection rate decreases as `delta_range` increases.

All 216 summary rows report:

```text
power_monotonicity_flag = false
```

Therefore all 12 base designs, both methods and all three alpha values satisfy the preregistered descriptive monotonicity sanity check across:

```text
delta_range = 0.25 -> 0.50 -> 1.00
```

This is a sanity check only; no minimum power gate was preregistered.

## Alpha=.05 power map

```text
        Classical                    Welch
        d=.25   d=.50   d=1.00       d=.25   d=.50   d=1.00
P01     .07280  .14195  .46160       .07110  .13635  .43285
P02     .12155  .38500  .93805       .11910  .37795  .93325
P03     .06870  .13550  .44280       .06705  .12795  .40330
P04     .07290  .13440  .43845       .07445  .12750  .37805
P05     .25830  .27680  .33000       .06060  .07180  .10025
P06     .01195  .01465  .02360       .05455  .06720  .12550
P07     .07125  .16400  .50620       .07590  .17155  .50740
P08     .10635  .32360  .70620       .13410  .38980  .79150
P09     .07445  .18255  .57640       .06930  .18255  .58940
P10     .07450  .23515  .56115       .08525  .30890  .66795
P11     .06630  .13680  .46715       .05520  .13210  .47835
P12     .14525  .45050  .90830       .15385  .49785  .92810
```

Frozen design identities:

```text
P01 normal k3 n10 equal SD
P02 normal k3 n30 equal SD
P03 normal k5 n10 equal SD
P04 normal [5,10,20] equal SD
P05 normal [5,10,20] SD [4,2,1]
P06 normal [5,10,20] SD [1,2,4]
P07 gamma shape 1 k3 n10 equal SD
P08 lognormal sigma 1.2 k3 n10 equal SD
P09 Student-t df3 k3 n10 equal SD
P10 asymmetric contamination 5% k3 n10 equal SD
P11 Laplace k5 n10 equal SD
P12 symmetric mixture k3 n30 equal SD
```

## Interpretation boundary

Raw rejection under H1 must not be interpreted as a fair method-power ranking when the corresponding method does not control Type I in that design region.

This is especially visible for `P05` and `P06`: the variance-size direction strongly changes Classical rejection, consistent with the H0 calibration/stress evidence. Therefore Phase D power evidence supports sensitivity characterization and monotonicity, not a universal statement that one method is more powerful.

Likewise, higher Welch H1 rejection in some skewed families must be interpreted together with its H0 size distortion in comparable regions.

## Decision

`D-power-h1` is complete.

All four Phase D strata have now executed:

```text
D-core-h0          complete
D-robustness-h0    complete
D-stress-h0        complete
D-power-h1         complete
```

No additional development Monte Carlo, selective rerun or parameter change is authorized under preregistration v1.1.

Next: `CP-ANOVA-07E — Phase D interpretation and candidate freeze`.

Phase H remains sealed pending explicit Product Owner authorization.
