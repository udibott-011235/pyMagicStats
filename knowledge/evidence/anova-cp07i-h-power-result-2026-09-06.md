# CP-ANOVA-07I — H-power holdout result

- Fecha: `2026-09-06`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07I`
- Phase: `H-power`
- Harness SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Engine SHA: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- Preregistration: `anova-calibration-prereg-v1.1`
- Holdout authorization SHA-256: `ed0ed744e460bbf35a45cae6d8fecc52b5ab212dc18dfdd664d9dacf4b06113b`
- Decision: `EXECUTION_INTEGRITY_PASS / POWER_SANITY_PASS`

## Execution integrity

Frozen scope:

```text
12 cells
20,000 replicas/cell
240,000 paired datasets
480,000 method results
alpha grid = [0.01, 0.05, 0.10]
holdout master seed = 2026090599
```

Quantum reported:

```text
execution_status = ACCOUNTED
parity = PASS
parity cells = 12
parity replicas/cell = 32
parity comparisons = 768
parity warnings = 0
workers = 12
batch_size = 200
backend = cpu
```

All 72 method/alpha rows completed 20,000 replications with zero generation errors, zero kernel errors, zero nonfinite outputs and zero Monte Carlo warnings. All 36 paired-accounting rows close exactly. Six final artifact checksums were published.

## Power monotonicity

The harness defines `power_monotonicity_flag=True` only when rejection rate decreases as `delta_range` increases within the same family/method/alpha.

All 72 summary rows report:

```text
power_monotonicity_flag = false
```

Therefore there are zero monotonicity violations over the frozen effect sequence:

```text
0.25 -> 0.50 -> 1.00
```

No minimum power threshold was preregistered; this is a descriptive sanity check only.

## Alpha=.05 holdout sensitivity map

```text
family                     method      d=.25    d=.50    d=1.00
gamma_shape_2              Classical   .07290   .14505   .48465
gamma_shape_2              Welch       .07415   .14705   .46865
lognormal_sigma_0p8        Classical   .07465   .18890   .57725
lognormal_sigma_0p8        Welch       .08315   .21080   .60450
student_t_df_7             Classical   .07510   .15040   .47080
student_t_df_7             Welch       .07125   .14500   .45430
weibull_shape_1p5          Classical   .07195   .14790   .47510
weibull_shape_1p5          Welch       .07185   .14615   .45795
```

The holdout reproduces the expected monotone sensitivity pattern. Raw H1 rejection-rate differences between Classical and Welch are not interpreted as universal power superiority because size distortion differs across design regions; power comparisons must be conditioned on valid Type-I control.

## Decision

`H-power` is complete and valid.

All preregistered holdout strata are now complete:

```text
H-core-normal  complete
H-robustness   complete
H-power        complete
```

No further Monte Carlo is authorized under preregistration v1.1. Next: final `CP-ANOVA-07` statistical closure and evidence-to-policy translation.
