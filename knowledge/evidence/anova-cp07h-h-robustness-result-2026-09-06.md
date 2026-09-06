# CP-ANOVA-07H — H-robustness holdout result

- Fecha: `2026-09-06`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07H`
- Phase: `H-robustness`
- Harness SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Engine SHA: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- Preregistration: `anova-calibration-prereg-v1.1`
- Holdout authorization SHA-256: `ed0ed744e460bbf35a45cae6d8fecc52b5ab212dc18dfdd664d9dacf4b06113b`
- Decision: `EXECUTION_INTEGRITY_PASS / HOLDOUT_ROBUSTNESS_MAPPED`

## Execution integrity

Frozen scope:

```text
21 cells
25,000 replicas/cell
525,000 paired datasets
1,050,000 method results
alpha grid = [0.01, 0.05, 0.10]
holdout master seed = 2026090599
```

Quantum reported:

```text
execution_status = ACCOUNTED
parity = PASS
parity cells = 21
parity replicas/cell = 32
parity comparisons = 1,344
parity warnings = 0
workers = 12
batch_size = 200
backend = cpu
```

All 126 method/alpha summary rows report zero generation errors, zero kernel errors, zero nonfinite outputs and zero Monte Carlo warnings. All 63 paired-accounting rows close exactly. Six final artifact checksums were published.

## Robustness bands at alpha=.05

Complete holdout map:

```text
             green   amber   red   total
Classical       11       3     7      21
Welch            7       8     6      21
TOTAL           18      11    13      42
```

These are descriptive holdout robustness classifications, not confirmatory authorization gates.

## Design-level structure

Frozen holdout designs:

```text
HRD01 = sizes [7,7,7],   sd [1,1,1]
HRD02 = sizes [6,15,40], sd [1,1,1]
HRD03 = sizes [6,15,40], sd [3.5,2,1]
```

### HRD01 — balanced, equal SD

Across the seven holdout families:

```text
Classical: 4 green / 3 amber / 0 red
Welch:     4 green / 3 amber / 0 red
```

Rejection-rate ranges at alpha=.05:

```text
Classical 0.03280 .. 0.05036
Welch     0.02756 .. 0.05000
```

Thus neither method exhibits a red robustness failure in this small but balanced/equal-SD region, although both can become conservative for skew/heavy-tail families.

### HRD02 — unbalanced, equal SD

```text
Classical: 7 green / 0 amber / 0 red
Welch:     2 green / 3 amber / 2 red
```

Classical remains near nominal across all seven holdout families under equal population SD despite size imbalance.

Welch is materially more fragile under nonnormality in the same equal-SD unbalanced design. Examples at alpha=.05:

```text
gamma_shape_2          Welch 0.07380  amber
lognormal_sigma_0p8    Welch 0.09904  red
pareto_alpha_3p5       Welch 0.10972  red
weibull_shape_1p5      Welch 0.07000  amber
```

This independently extends the finite-sample finding beyond normality: Welch cannot be treated as a cost-free replacement for Classical when variances are equal but group sizes are unbalanced.

### HRD03 — unbalanced, adverse heteroscedasticity

```text
Classical: 0 green / 0 amber / 7 red
Welch:     1 green / 2 amber / 4 red
```

Classical is severely liberal for every holdout family:

```text
range = 0.27364 .. 0.31192
```

Welch improves substantially but is not universally robust:

```text
range = 0.05092 .. 0.20520
```

Examples:

```text
student_t_df_7         Welch 0.05092  green
beta_2_5               Welch 0.06964  amber
contamination_2pct     Welch 0.06960  amber
gamma_shape_2          Welch 0.09424  red
weibull_shape_1p5      Welch 0.08116  red
lognormal_sigma_0p8    Welch 0.15144  red
pareto_alpha_3p5       Welch 0.20520  red
```

Thus Welch clearly mitigates variance-size antagonism relative to Classical, but strong skew/tail structure can still leave substantial Type-I inflation.

## Cross-phase interpretation

The sealed holdout reproduces the development conclusions rather than overturning them:

1. Classical is excellent in the exact equal-variance model, including many unequal-n settings.
2. Classical can fail catastrophically when the smallest groups carry the largest variances.
3. Welch is often the safer heteroscedastic procedure, but has finite-sample/group-count costs even under equal variance.
4. Under skewed/heavy-tailed families plus adverse heteroscedasticity, Welch can also become badly liberal.
5. The decision engine therefore requires a calibrated multidimensional routing region; no single normality or variance-equality pretest can choose the procedure reliably.

No result in H-robustness changes the frozen thresholds, seeds, scenario definitions or prior confirmatory H-core result.

## Decision

`H-robustness` is complete and valid.

No rerun, threshold relaxation, scenario removal or production modification is authorized.

The final preregistered holdout phase is now authorized:

```text
H-power   AUTHORIZED NEXT
```

After H-power finishes, stop for final CP-ANOVA-07 statistical closure. Selector remains `NOT_CALIBRATED` until that closure explicitly defines any supported routing region.
