# CP-ANOVA-07G — H-core-normal holdout result

- Fecha: `2026-09-06`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07G`
- Phase: `H-core-normal`
- Harness SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Engine SHA: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- Preregistration: `anova-calibration-prereg-v1.1`
- Holdout authorization SHA-256: `ed0ed744e460bbf35a45cae6d8fecc52b5ab212dc18dfdd664d9dacf4b06113b`
- Decision: `EXECUTION_INTEGRITY_PASS / WELCH_GLOBAL_H_CORE_GATE_FAIL`

## Execution integrity

Frozen scope:

```text
10 cells
50,000 replicas/cell
500,000 paired datasets
1,000,000 method results
alpha grid = [0.01, 0.05, 0.10]
holdout master seed = 2026090599
```

Quantum reported:

```text
execution_status = ACCOUNTED
parity = PASS
parity cells = 10
parity replicas/cell = 32
parity comparisons = 640
parity warnings = 0
workers = 12
batch_size = 200
backend = cpu
```

All 60 method/alpha rows completed 50,000 replications with zero generation errors, zero kernel errors, zero nonfinite outputs and zero Monte Carlo warnings. All 30 paired accounting rows close exactly. Six immutable artifact checksums were published.

## Confirmatory gates at alpha=.05

Classical is confirmatory only in equal-population-SD cells. All five eligible Classical cells pass:

```text
HCN-01  0.05108  PASS
HCN-02  0.05132  PASS
HCN-03  0.05056  PASS
HCN-04  0.04776  PASS
HCN-05  0.04878  PASS
```

Welch is confirmatory over all ten normal holdout cells. Eight pass and two fail:

```text
HCN-01  0.05096  PASS
HCN-02  0.05086  PASS
HCN-03  0.05960  FAIL  CI99=[0.0569308, 0.0623860]
HCN-04  0.04802  PASS
HCN-05  0.05500  PASS
HCN-06  0.05694  PASS
HCN-07  0.05040  PASS
HCN-08  0.06202  FAIL  CI99=[0.0592993, 0.0648569]
HCN-09  0.05082  PASS
HCN-10  0.04958  PASS
```

Therefore the preregistered global H-core authorization claim for Welch is not satisfied.

## Structural replication of Phase D finding

The two holdout failures reproduce the finite-sample pattern found in development rather than contradicting it.

`HCN-03`:

```text
normal
k = 7
sizes = [7,7,7,7,7,7,7]
equal population SD
Welch Type-I = 0.05960
```

Classical on the same exact-model design is 0.05056 and PASS. This independently confirms that Welch can become liberal solely from finite-sample/group-count structure even without heteroscedasticity.

`HCN-08`:

```text
normal
k = 7
sizes = [7,7,7,7,7,7,7]
sd = [1,2,1,3,1.5,2.5,4]
Welch Type-I = 0.06202
```

This independently confirms the interaction between moderate/high k, small per-group n and heterogeneous variance patterns.

By contrast Welch passes the k=4 unbalanced heteroscedastic designs HCN-06/07 and both k=2 unequal-variance designs HCN-09/10. This does not define a complete authorization boundary, but it strengthens the evidence that k and finite group sizes must be explicit routing variables.

## Classical descriptive heteroscedastic behavior

Classical was correctly not gated in unequal-SD holdout cells and shows the expected severe directional distortion:

```text
HCN-06  Classical = 0.37778
HCN-07  Classical = 0.00386
HCN-08  Classical = 0.08490
HCN-09  Classical = 0.29498
HCN-10  Classical = 0.00080
```

These results independently reinforce that a variance-equality p-value pretest cannot justify a simple Classical/Welch binary switch.

## Decision

`H-core-normal` execution is complete and valid.

The holdout does **not** authorize Welch universally across the preregistered normal design region. The two FAIL rows are retained exactly as observed; no rerun, threshold relaxation, seed change, scenario removal or replication increase is allowed.

This is a statistical result, not an execution or implementation blocker. Production implementation correctness remains closed based on the deterministic oracle evidence.

The remaining already-authorized holdout strata continue unchanged to characterize out-of-model generalization and H1 sensitivity:

```text
H-robustness   AUTHORIZED NEXT
H-power        authorized after H-robustness integrity review
```

Selector remains `NOT_CALIBRATED`. No automatic Welch fallback is authorized.
