# CP-ANOVA-06 — Authorized amendment v1.1: cell IDs + exact E0 subset

- Fecha: `2026-09-05`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-06`
- Version: `anova-calibration-prereg-v1.1`
- Supersedes only the incomplete identification layer of `anova-calibration-prereg-v1`.
- Statistical design change: **NONE**.
- Allowed scope of this amendment: materialize stable `cell_id` values and freeze the exact E0 engineering subset.
- Prohibited interpretation: this amendment does not alter distributions, group sizes, SD multipliers, alpha grid, replications, seeds, H0/H1 definitions, acceptance bands, metrics, holdout families, or holdout opening policy.

## 1. Canonical cell identity contract

Every Monte Carlo cell has one and only one stable `cell_id`.

The seed key remains exactly:

```text
phase|cell_id|replicate_index
```

with UTF-8 encoding, SHA-256, first four uint32 values of the digest, and the phase-appropriate frozen master seed passed to `numpy.random.SeedSequence`.

`cell_id` is immutable once defined here. Workers, batching, sharding, file order, execution order, hostname, process id, timestamp and Python `hash()` MUST NOT contribute to identity or RNG.

## 2. Already explicit phases

The following IDs already present in the manifest are authoritative as written:

- `D-core-h0`: 42 cells (`DCEV-*` and `DCUV-*`).
- `D-stress-h0`: 10 cells (`DSH0-01` through `DSH0-10`).
- `H-core-normal`: 10 cells (`HCN-01` through `HCN-10`).

No renaming is permitted.

## 3. D-robustness-h0 — exact 54 IDs

Family order is frozen:

```text
F01 = gamma_shape_4
F02 = gamma_shape_1
F03 = lognormal_sigma_0p5
F04 = lognormal_sigma_1p2
F05 = student_t_df_5
F06 = student_t_df_3
F07 = laplace
F08 = mixture_symmetric_5pct_scale6
F09 = contamination_asymmetric_5pct_loc10
```

Design order is frozen:

```text
R01 = sizes [5,5,5],          sd [1,1,1]
R02 = sizes [30,30,30],       sd [1,1,1]
R03 = sizes [5,10,20],        sd [1,1,1]
R04 = sizes [5,10,20],        sd [4,2,1]
R05 = sizes [5,10,20],        sd [1,2,4]
R06 = sizes [10,10,10,10,10], sd [1,1,1,1,1]
```

Canonical ID is:

```text
DRH0-F<family_index>-R<design_index>
```

Therefore the exact set is the Cartesian product:

```text
DRH0-F01-R01 ... DRH0-F01-R06
DRH0-F02-R01 ... DRH0-F02-R06
DRH0-F03-R01 ... DRH0-F03-R06
DRH0-F04-R01 ... DRH0-F04-R06
DRH0-F05-R01 ... DRH0-F05-R06
DRH0-F06-R01 ... DRH0-F06-R06
DRH0-F07-R01 ... DRH0-F07-R06
DRH0-F08-R01 ... DRH0-F08-R06
DRH0-F09-R01 ... DRH0-F09-R06
```

Total: `9 * 6 = 54` cells.

All are `H0`.

## 4. D-power-h1 — exact 36 IDs

Base-cell order is frozen:

```text
P01 = normal, k3 n=10 balanced, equal sd
P02 = normal, k3 n=30 balanced, equal sd
P03 = normal, k5 n=10 balanced, equal sd
P04 = normal, sizes [5,10,20], sd [1,1,1]
P05 = normal, sizes [5,10,20], sd [4,2,1]
P06 = normal, sizes [5,10,20], sd [1,2,4]
P07 = gamma_shape_1, k3 n=10, equal sd
P08 = lognormal_sigma_1p2, k3 n=10, equal sd
P09 = student_t_df_3, k3 n=10, equal sd
P10 = contamination_asymmetric_5pct_loc10, k3 n=10, equal sd
P11 = laplace, k5 n=10, equal sd
P12 = mixture_symmetric_5pct_scale6, k3 n=30, equal sd
```

Delta order is frozen:

```text
D01 = delta_range 0.25
D02 = delta_range 0.50
D03 = delta_range 1.00
```

Canonical ID is:

```text
DPH1-P<base_index>-D<delta_index>
```

Exact set:

```text
DPH1-P01-D01 ... DPH1-P01-D03
DPH1-P02-D01 ... DPH1-P02-D03
DPH1-P03-D01 ... DPH1-P03-D03
DPH1-P04-D01 ... DPH1-P04-D03
DPH1-P05-D01 ... DPH1-P05-D03
DPH1-P06-D01 ... DPH1-P06-D03
DPH1-P07-D01 ... DPH1-P07-D03
DPH1-P08-D01 ... DPH1-P08-D03
DPH1-P09-D01 ... DPH1-P09-D03
DPH1-P10-D01 ... DPH1-P10-D03
DPH1-P11-D01 ... DPH1-P11-D03
DPH1-P12-D01 ... DPH1-P12-D03
```

Total: `12 * 3 = 36` cells.

All are `H1` and use the already frozen mean-vector rule:

```text
mu_i = delta_range * centered_linspace(-0.5, 0.5, k)
```

## 5. H-robustness — exact 21 IDs

Holdout family order is frozen:

```text
F01 = gamma_shape_2
F02 = lognormal_sigma_0p8
F03 = student_t_df_7
F04 = weibull_shape_1p5
F05 = pareto_alpha_3p5
F06 = beta_2_5
F07 = contamination_asymmetric_2pct_loc10
```

Holdout robustness designs are frozen:

```text
HRD01 = sizes [7,7,7],   sd [1,1,1]
HRD02 = sizes [6,15,40], sd [1,1,1]
HRD03 = sizes [6,15,40], sd [3.5,2,1]
```

Canonical ID:

```text
HRH0-F<family_index>-HRD<design_index>
```

Exact set:

```text
HRH0-F01-HRD01 ... HRH0-F01-HRD03
HRH0-F02-HRD01 ... HRH0-F02-HRD03
HRH0-F03-HRD01 ... HRH0-F03-HRD03
HRH0-F04-HRD01 ... HRH0-F04-HRD03
HRH0-F05-HRD01 ... HRH0-F05-HRD03
HRH0-F06-HRD01 ... HRH0-F06-HRD03
HRH0-F07-HRD01 ... HRH0-F07-HRD03
```

Total: `7 * 3 = 21` cells.

All are `H0` and remain sealed.

## 6. H-power — exact 12 IDs

Holdout power family order is frozen:

```text
F01 = gamma_shape_2
F02 = lognormal_sigma_0p8
F03 = student_t_df_7
F04 = weibull_shape_1p5
```

All use:

```text
sizes [10,10,10]
sd    [1,1,1]
```

Delta order:

```text
D01 = 0.25
D02 = 0.50
D03 = 1.00
```

Canonical ID:

```text
HPH1-F<family_index>-D<delta_index>
```

Exact set:

```text
HPH1-F01-D01 ... HPH1-F01-D03
HPH1-F02-D01 ... HPH1-F02-D03
HPH1-F03-D01 ... HPH1-F03-D03
HPH1-F04-D01 ... HPH1-F04-D03
```

Total: `4 * 3 = 12` cells.

All are `H1` and remain sealed.

## 7. Exact E0 engineering subset — 12 cells

E0 is **not** a statistical-evidence phase. It gets its own stable IDs so its RNG stream cannot accidentally collide with Phase D.

The exact E0 cells are:

```text
E0-01  normal; sizes [5,5,5]; sd [1,1,1]; H0
E0-02  normal; sizes [5,10,20]; sd [4,2,1]; H0
E0-03  normal; sizes [5,10,20]; sd [1,2,4]; H0
E0-04  gamma_shape_1; sizes [5,5,5]; sd [1,1,1]; H0
E0-05  lognormal_sigma_1p2; sizes [5,10,20]; sd [1,1,1]; H0
E0-06  student_t_df_3; sizes [5,10,20]; sd [4,2,1]; H0
E0-07  mixture_symmetric_5pct_scale6; sizes [10,10,10,10,10]; sd [1,1,1,1,1]; H0
E0-08  contamination_asymmetric_5pct_loc10; sizes [5,10,20]; sd [1,2,4]; H0
E0-09  normal; sizes [2,2,2]; sd [1,1,1]; H0
E0-10  lognormal_sigma_1p5; sizes [5,10,20]; sd [4,2,1]; H0
E0-11  normal; sizes [10,10,10]; sd [1,1,1]; H1; delta_range 0.50
E0-12  gamma_shape_1; sizes [10,10,10]; sd [1,1,1]; H1; delta_range 1.00
```

Each uses exactly `200` replications.

Total E0 generated datasets:

```text
12 * 200 = 2,400
```

E0 MUST NOT contain any holdout-only family or use `holdout_master_seed`.

E0 uses `development_master_seed = 2026090501`.

## 8. Cell-count invariant

The implementation MUST materialize exactly:

```text
E0                 12
D-core-h0          42
D-robustness-h0    54
D-stress-h0        10
D-power-h1         36
H-core-normal      10
H-robustness       21
H-power            12
```

Evidence phases excluding E0:

```text
42 + 54 + 10 + 36 + 10 + 21 + 12 = 185 cells
```

Including E0:

```text
197 total stable cell IDs
```

The harness MUST assert uniqueness across all 197 IDs at load time.

## 9. Implementation rule

Cortex may generate cross-product records programmatically from the ordered tables above, but the resulting `cell_id`, family, sizes, SD vector, hypothesis and delta MUST exactly equal this amendment.

Programmatic materialization is an implementation detail; it does not authorize a change in ordering or naming.

A load-time manifest validator MUST reject:

- duplicate `cell_id`;
- missing phase cell;
- unexpected phase cell;
- size/SD length mismatch;
- unsupported family;
- H0 cell with delta;
- H1 power cell without delta;
- holdout family in E0 or Phase D;
- development/stress family substituted into sealed holdout where a holdout-only family is specified.

## 10. Authorization after blocker

The prior `BLOCKED_PREFLIGHT` was valid because v1 did not fully materialize the identity layer and did not specify E0.

This amendment resolves exactly those two blockers.

Cortex is authorized to resume CP-ANOVA-07A **only after** verifying this file and the restored manifest on `audit/anova-calibration-preregistration`.

No Monte Carlo phase may be executed as part of this clarification. Cortex must still stop at `READY_FOR_ARCHITECT_REVIEW` with only the proposed E0 command.
