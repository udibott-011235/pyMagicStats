# CP-ANOVA-07 — Final statistical closure

- Fecha: `2026-09-06`
- Stage: `STAGE-ANOVA-001`
- Candidate harness: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Frozen production engine: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- Preregistration: `anova-calibration-prereg-v1.1`
- Holdout authorization SHA-256: `ed0ed744e460bbf35a45cae6d8fecc52b5ab212dc18dfdd664d9dacf4b06113b`
- Final decision: `COMPLETE_WITH_LIMITATIONS`

## Scope completed

Development evidence:

```text
D-core-h0          42 x 50,000
D-robustness-h0    54 x 25,000
D-stress-h0        10 x 25,000
D-power-h1         36 x 20,000
```

Sealed holdout evidence:

```text
H-core-normal      10 x 50,000
H-robustness       21 x 25,000
H-power            12 x 20,000
```

All executed strata closed with parity PASS, ACCOUNTED execution, frozen identities/seeds/replication counts and immutable artifacts. Deterministic oracle adjudication previously established that the observed Welch calibration failures are finite-sample statistical behavior rather than an implementation divergence.

## Confirmatory result

### Classical one-way ANOVA

Classical passes the preregistered confirmatory H-core gates in every eligible equal-population-variance cell.

The evidence supports the implementation and the classical exact-model claim for independent Gaussian groups with common population variance over the tested design region.

This does not imply arbitrary robustness outside that model. Development and holdout evidence show severe Type-I distortion when variance magnitude is associated with group size, especially when the smallest groups have the largest variances.

### Welch one-way ANOVA

Welch does **not** pass the preregistered global H-core authorization claim.

Holdout failures:

```text
HCN-03: normal, k=7, n_i=7, equal SD
Type-I = 0.05960
CI99 = [0.0569308, 0.0623860]

HCN-08: normal, k=7, n_i=7, heterogeneous SD
Type-I = 0.06202
CI99 = [0.0592993, 0.0648569]
```

These independently reproduce the finite-sample/group-count behavior found in development. Welch therefore cannot be authorized as a universal heteroscedastic fallback or as a cost-free replacement for Classical.

## Robustness evidence

Across development and holdout, the principal structural conclusions are stable:

1. Equal-variance Gaussian settings strongly support Classical.
2. Classical can become catastrophically liberal or conservative under variance-size antagonism.
3. Welch often mitigates heteroscedasticity-induced distortion, but its finite-sample accuracy depends on group count, per-group sample sizes and design structure.
4. Nonnormality interacts with imbalance and heteroscedasticity; strong skew/heavy tails can make Welch materially liberal as well.
5. A binary variance pretest followed by `Classical if equal / Welch if unequal` is not statistically justified by the evidence.
6. A binary normality test is also insufficient for method routing.

## Power evidence

Both development and holdout power phases have zero monotonicity violations across the frozen delta sequence. This supports implementation sanity under H1.

No minimum-power authorization was preregistered. Raw rejection-rate differences between Classical and Welch must not be interpreted as power superiority where Type-I control differs.

## Production/API status

The following are statistically and architecturally supported as **explicit methods**:

```text
OneWayANOVA(...).run()
WelchANOVA(...).run()
```

subject to their documented assumptions and limitations.

The evidence does **not** authorize an automatic selector to choose between them yet.

Current routing status remains:

```text
ONE_WAY selector = NOT_CALIBRATED
```

No universal Welch fallback is authorized.

## Evidence-to-policy boundary

The current calibration is sufficient to close CP-ANOVA-07 as an evidence program, but not sufficient to derive a post-hoc hard multidimensional selector boundary directly from observed development/holdout cells. Doing so would reuse the holdout to optimize routing and would destroy its role as independent validation.

A future selector stage must preregister its routing policy before any new validation evidence is generated. It may use the present evidence to motivate candidate routing variables such as:

```text
k
min(n_i)
size imbalance
variance-ratio magnitude
variance-size association
residual-shape diagnostics
independence metadata
```

but thresholds and routing rules must be specified prospectively and validated on new evidence.

## Final checkpoint state

```text
CP-ANOVA-01..06   COMPLETE
CP-ANOVA-07       COMPLETE_WITH_LIMITATIONS

Classical explicit method   SUPPORTED WITH DOCUMENTED ASSUMPTIONS
Welch explicit method       SUPPORTED WITH DOCUMENTED LIMITATIONS
Welch universal fallback    NOT AUTHORIZED
Automatic ONE_WAY selector  NOT_CALIBRATED

Further prereg-v1.1 MC      NOT AUTHORIZED
```

## Next architectural stage

Next work should not modify the frozen ANOVA candidate merely to make the observed cells pass. The appropriate next stage is an **evidence-to-policy / selector preregistration** stage, separate from CP-ANOVA-07, or UAT/inventory closure if selector work remains explicitly outside the current release gate.
