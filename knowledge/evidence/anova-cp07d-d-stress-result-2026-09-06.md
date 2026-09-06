# CP-ANOVA-07D — D-stress-h0 result

- Fecha: `2026-09-06`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07D`
- Phase: `D-stress-h0`
- Harness SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Engine SHA: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- Preregistration: `anova-calibration-prereg-v1.1`
- Decision: `EXECUTION_INTEGRITY_PASS / STRESS_BEHAVIOR_MAPPED`

## Execution integrity

Frozen scope:

```text
10 cells
25,000 replicas/cell
250,000 paired datasets
500,000 method results
alpha grid = [0.01, 0.05, 0.10]
```

Quantum reported:

```text
execution_status = ACCOUNTED
parity = PASS
parity cells = 10
parity replicas/cell = 32
parity comparisons = 640
parity warnings = 12
workers = 12
batch_size = 200
backend = cpu
```

All six thread-control environment variables were pinned to 1.

The complete 60 method/alpha rows report the frozen 25,000 completed replicas with zero generation errors, zero kernel errors, zero nonfinite outputs and zero Monte Carlo warnings. The complete 30 paired-accounting rows satisfy exactly:

```text
both_reject + classical_only + welch_only + neither = 25,000
```

for every cell/alpha pair.

The final output contains the six required artifact checksums. Execution integrity is accepted.

## Stress map at alpha = 0.05

```text
cell       family / design                                     Classical    Welch
DSH0-01    normal; n=[2,2,2], equal SD                         0.04996      0.02928
DSH0-02    normal; n=[2,2,2,2,2], equal SD                     0.05148      0.07880
DSH0-03    normal; n=[2,5,20], SD=[8,2,1]                      0.58368      0.11044
DSH0-04    normal; n=[2,5,20], SD=[1,2,8]                      0.00040      0.04848
DSH0-05    lognormal sigma=1.5; n=[5,5,5], equal SD            0.02196      0.01260
DSH0-06    Student-t df=2.5; n=[5,5,5], equal SD               0.03756      0.02836
DSH0-07    asymmetric contamination 10%; n=[10,10,10]          0.03248      0.02456
DSH0-08    normal; 20 groups x n=5, equal SD                   0.04864      0.12772
DSH0-09    lognormal sigma=1.5; n=[5,10,20], SD=[4,2,1]        0.30552      0.33968
DSH0-10    Student-t df=2.5; n=[5,10,20], SD=[1,2,4]           0.00860      0.03420
```

## Interpretation

These stress cells were preregistered as descriptive/no-gate cases. Their purpose is to expose failure modes, not to define a production authorization region by themselves.

Important patterns:

1. `DSH0-03` demonstrates severe variance-size antagonism: Classical Type-I rises to 58.368%, while Welch still over-rejects at 11.044%.
2. Reversing the variance-size association (`DSH0-04`) drives Classical almost to zero rejection (0.040%) while Welch is near nominal (4.848%).
3. Large group count with tiny groups (`DSH0-08`: k=20, n_i=5) leaves Classical near nominal (4.864%) but Welch becomes strongly liberal (12.772%), reinforcing the D-core finite-sample finding for Welch as k grows.
4. Strong skew plus adverse heteroscedasticity (`DSH0-09`) is hostile to both procedures: Classical 30.552%, Welch 33.968%.
5. Heavy tails and the opposite variance-size association (`DSH0-10`) make Classical extremely conservative (0.860%); Welch is less extreme but still conservative (3.420%).

These observations strengthen the architecture requirement that method selection must depend on a calibrated multidimensional region rather than a single normality or variance-equality switch.

## Decision

`D-stress-h0` is complete for Phase D evidence collection.

No production change, rerun, threshold change, seed change or replication adjustment is authorized from this descriptive phase.

Next:

```text
D-power-h1      AUTHORIZED NEXT
Phase H         SEALED
selector         NOT_CALIBRATED
```

After D-power finishes, stop and perform Phase D interpretation/candidate-freeze review before any Product Owner holdout authorization can be considered.
