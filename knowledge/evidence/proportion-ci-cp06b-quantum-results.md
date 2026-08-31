# CP-06B — Quantum calibration results (`n=1..200`)

**Stage:** `STAGE-PROP-CI-001`  
**Checkpoint:** `CP06-B`  
**Status:** `pass`  
**Execution host:** Quantum  
**Production candidate:** `2df5b90a5395163e723f9c52aafbb91fdce96d43`  
**Harness SHA:** `c7ece2118075343e322ea2792f1d700d9f77334c`  
**Backend:** CPU / SciPy float64  
**Workers:** 8  
**Batch size:** 256

## Execution summary

Quantum completed `n=1..200` with no STOP condition:

- `CP06-B complete in 72.079s`
- wall time: `1:13.94`
- CPU utilization: `766%`
- max resident set size: `429448 kB`
- swaps: `0`
- exit status: `0`
- cached shards: `200/200`

## 1. Worst coverage by method

| n | alpha | method | coverage_min | max_undercoverage | tier | worst_p | origin |
|---:|---:|---|---:|---:|---|---:|---|
| 139 | 0.200 | clopper_pearson | 0.8 | 0.000 | nominal_like | 3.126606e-01 | nextafter |
| 94 | 0.200 | jeffreys | 0.0 | 0.800 | critical_shortfall | 0.0 | boundary |
| 1 | 0.200 | wald | 0.0 | 0.800 | critical_shortfall | 1.0e-12 | grid |
| 95 | 0.001 | wilson | 0.0 | 0.999 | critical_shortfall | 0.0 | boundary |

Interpretation: Wilson/Jeffreys/Wald can show severe or critical frequentist undercoverage in boundary/extreme discrete regimes. This is statistical behavior, not by itself a harness or implementation defect. CP-04 explicitly preregistered preservation and high-precision review of these cells.

## 2. Clopper–Pearson exact-coverage gate

`violations: 0`

No evaluated CP06-B cell violated `coverage >= 1-alpha-1e-12`.

## 3. Invariants

| method | lower monotonic failures | upper monotonic failures | bounds failures | NaN | max complement error |
|---|---:|---:|---:|---:|---:|
| clopper_pearson | 0 | 0 | 0 | 0 | 5.329071e-15 |
| jeffreys | 0 | 0 | 0 | 0 | 6.217249e-15 |
| wald | 2047 | 2047 | 12093 | 0 | 3.330669e-16 |
| wilson | 0 | 0 | 0 | 0 | 3.330669e-16 |

Wald failures are expected consequences of the approved unclipped legacy formula; they are not interpreted as production implementation defects because formula fidelity is verified independently.

## 4. Confidence nesting

All methods:

- lower nesting failures: `0`
- upper nesting failures: `0`

## 5. Oracle agreement

| method | oracle | max lower error | max upper error | rows |
|---|---|---:|---:|---:|
| clopper_pearson | scipy_binomtest | 8.715251e-15 | 5.329071e-15 | 8309 |
| clopper_pearson | statsmodels_beta | 0.0 | 5.329071e-15 | 8309 |
| jeffreys | statsmodels_jeffreys | 0.0 | 6.217249e-15 | 8309 |
| wald | independent_unclipped_formula | 0.0 | 0.0 | 8309 |
| wilson | scipy_binomtest | 3.330669e-16 | 3.330669e-16 | 8309 |
| wilson | statsmodels_wilson | 4.773959e-15 | 4.773959e-15 | 8309 |

All applicable oracle gates are comfortably inside the preregistered `1e-12` tolerance for `n<=5000`.

## 6. High-precision trigger queue

| method | trigger | cells | worst undercoverage |
|---|---|---:|---:|
| jeffreys | material_minimum_or_endpoint | 1400 | 0.999 |
| wilson | material_minimum_or_endpoint | 1400 | 0.999 |

No `cp_undercoverage` trigger was observed.

These 2,800 rows are an audit queue, not 2,800 defects. CP06-F must resolve them according to the CP-04 80-digit high-precision protocol.

## 7. Minimum retained PMF mass

All methods reported `1.0` at printed precision in CP06-B summaries. The smoke harness previously established the warning-free Hoeffding truncation strategy against complete PMF summation with omitted mass bounded at `1e-14`.

## Architectural interpretation

`CP06-B: PASS`

The checkpoint demonstrates:

- harness stability at `n<=200`;
- Wilson and Clopper–Pearson numerical integrity;
- exact/conservative CP coverage gate preserved across the evaluated domain;
- no nesting failures;
- oracle agreement at approximately machine precision;
- Wald implementation fidelity despite known legacy pathologies;
- preregistered high-precision queue generated for Wilson/Jeffreys boundary minima.

This PASS does **not** promote production `calibration_status`, authorize routing, or complete CP-06. Remaining work is CP06-C through CP06-I.
