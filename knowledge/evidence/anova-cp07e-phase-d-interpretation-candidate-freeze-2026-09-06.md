# CP-ANOVA-07E — Phase D interpretation and candidate freeze

- Fecha: `2026-09-06`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07E`
- Harness branch: `engineering/cp-anova-07a-harness`
- Harness SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Engine SHA: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- Engine blob: `2d00ae2a2812b8c390125fefe244dcb4830176c5`
- Preregistration: `anova-calibration-prereg-v1.1`
- Manifest SHA-256: `affa3a1ae3c02b8081d0bdc761e6ce3725bb736899b0d2771d5d185530c0262a`
- Decision: `PHASE_D_COMPLETE / CANDIDATE_FROZEN_FOR_HOLDOUT_CONSIDERATION`

## Phase D completion

The frozen development program is complete:

```text
D-core-h0          42 cells x 50,000
D-robustness-h0    54 cells x 25,000
D-stress-h0        10 cells x 25,000
D-power-h1         36 cells x 20,000
```

No further Phase D Monte Carlo, selective rerun, seed change, threshold change, family removal, cell removal, or replication adjustment is authorized under preregistration v1.1.

## Implementation correctness

Deterministic oracle work and the directed Welch adjudication found no implementation divergence.

The eight D-core confirmatory failures are retained exactly as observed and classified as finite-sample behavior of Welch rather than a pyMagicStats code defect.

Production ANOVA is therefore not reopened by Phase D.

## Statistical interpretation

Phase D does **not** support a universal automatic selector rule such as:

```text
heteroscedasticity -> Welch
```

or:

```text
normality failure -> reject ANOVA
```

The evidence instead shows a multidimensional authorization problem involving at least:

- number of groups `k`;
- group-size vector `n_i`;
- direction/magnitude of variance-size association;
- distributional skew/heavy tails/contamination;
- explicit method (Classical vs Welch).

Classical remains exact under its Gaussian/common-variance model but can become dramatically liberal or conservative under variance-size antagonism. Welch protects against some heteroscedastic configurations but has measurable finite-sample size distortion in small/unbalanced/high-k designs and can fail badly under strong skew plus adverse heteroscedasticity.

## Development calibration implications

At alpha=.05, D-robustness classified:

```text
             green   amber   red
Classical       30       8    16
Welch           23      22     9
```

D-stress then exposed deliberately extreme regions, including:

```text
normal n=[2,5,20], sd=[8,2,1]:   Classical .58368, Welch .11044
normal n=[2,5,20], sd=[1,2,8]:   Classical .00040, Welch .04848
normal k=20, n_i=5 equal SD:     Classical .04864, Welch .12772
lognormal sigma=1.5 adverse SD:  Classical .30552, Welch .33968
```

These stress outcomes are descriptive and are not converted into post-hoc gates.

## Power interpretation

All 12 base designs, both methods and alpha values passed the preregistered monotonicity sanity check across:

```text
delta_range = 0.25 -> 0.50 -> 1.00
```

No minimum-power threshold was preregistered.

Raw H1 rejection differences are not used as method superiority claims when H0 Type-I control differs between methods or design regions.

## Candidate freeze semantics

`candidate frozen` means only:

- no production or harness changes before holdout;
- exact harness/engine/manifest identity fixed;
- Phase D interpretation fixed before viewing holdout;
- observed development failures remain part of the candidate evidence.

It does **not** mean:

- selector is calibrated;
- ANOVA is universally authorized;
- Welch is universally preferred;
- production UAT1 blocker is closed;
- holdout has passed.

## Holdout readiness

The candidate is now ready for Product Owner consideration of CP-ANOVA-07F holdout opening.

If the Product Owner authorizes opening, the holdout declaration must bind exactly:

```text
action = open-holdout
authorized_by = Product Owner
harness_sha = b211dcd61be02a234386947305bc1a1c9cfffde7
manifest_sha256 = affa3a1ae3c02b8081d0bdc761e6ce3725bb736899b0d2771d5d185530c0262a
phase_d_complete = true
remediations_closed = true
candidate_frozen = true
```

Until that explicit authorization exists:

```text
H-core-normal  SEALED
H-robustness   SEALED
H-power        SEALED
selector       NOT_CALIBRATED
```
