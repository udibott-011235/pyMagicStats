# CP-ANOVA-07D — Phase D execution authorization

- Fecha: `2026-09-05`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07D`
- Harness branch: `engineering/cp-anova-07a-harness`
- Harness SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Preregistration: `anova-calibration-prereg-v1.1`
- Prior checkpoint: `CP-ANOVA-07C = complete/PASS`
- Decision: `AUTHORIZED_FOR_PHASE_D`

## Scope

Phase D is authorized because Quantum is no longer running competing test/calibration workloads and E0 closed cleanly with parity/accounting/persistence PASS.

Authorized phases, unchanged from preregistration:

```text
D-core-h0        42 cells × 50,000 = 2,100,000 paired datasets
D-robustness-h0  54 cells × 25,000 = 1,350,000 paired datasets
D-stress-h0      10 cells × 25,000 =   250,000 paired datasets
D-power-h1       36 cells × 20,000 =   720,000 paired datasets
TOTAL                                4,420,000 paired datasets
                                      8,840,000 method results
```

No replication counts, seeds, scenarios, alpha grid, acceptance rules or manifest fields may change.

## Execution strategy

Run each Phase D stratum as a separate immutable output directory. Do not combine the four strata into one invocation or artifact directory.

Recommended order:

1. `D-core-h0`
2. `D-robustness-h0`
3. `D-stress-h0`
4. `D-power-h1`

Use the exact named technical branch at the exact harness SHA, clean tree, CPU backend, 12 workers, batch size 200, and one BLAS/OpenMP thread per process.

Each stratum already contains its own full frozen replication count in the manifest. No count override is authorized.

## Stop conditions

After each stratum, stop and inspect its terminal metadata before starting the next one. Do not proceed automatically if any of these occur:

- parity status != PASS;
- generation_error_count > 0;
- kernel_error_count > 0;
- nonfinite_count > 0;
- execution_status != ACCOUNTED;
- harness/engine/manifest identity mismatch;
- unexpected output overwrite/resume behavior;
- production checkout becomes dirty;
- any confirmatory D-core cell reports `INVALID_EXECUTION`.

A statistical confirmatory gate `FAIL` in D-core is not an execution failure and must not be hidden or rerun selectively. It is evidence to be adjudicated after Phase D. Do not change tolerances or replicate counts to rescue it.

## Resource settings

Because Quantum has no competing test workload, 12 worker processes are authorized. BLAS/OpenMP-related environment variables remain pinned to one thread per process to prevent oversubscription.

If system responsiveness or memory pressure becomes problematic, the worker count may be reduced without changing the statistical experiment because worker count is not part of RNG identity. Increasing beyond 12 is not pre-authorized in this checkpoint.

## Holdout

Phase H remains sealed. No holdout authorization file is created by this decision.

After all four Phase D strata are complete, stop at `CP-ANOVA-07D` and return the four metadata/results sets for architect interpretation. Do not open or execute any H phase.
