# DEC-016 — CP05-C2B CPU/CUDA equivalence preregistration

- Status: `accepted`
- Date: 2026-09-20
- Reference SHA: `e759137b5f1f10f97746fb241dc2fe6c02ef7061`
- Reference engine: `pyMagicStats CP04 + DEC-014 float64 implementation`
- Experimental engine: `CP05 CUDA/RAPIDS float64 experimental engine`
- EQUIVALENCE_EXECUTED=`NO`

## Scope

This preregistration asks whether the CUDA engine implements the same mathematical procedure as CP04/DEC-014. It does not estimate Type-I error, power, speed, or production suitability. Quantum, GPU equivalence execution, generator-sanity execution, benchmarking, calibration, CP05-D and holdout access are out of scope. Both engines must use float64; float32, result-changing mixed precision and TensorFloat32 cannot pass the gate. Fixed observed and bootstrap arrays are supplied to both engines, isolating mathematics from backend RNG streams.

## Frozen gates

- **A categorical:** 100% exact agreement for `ELIGIBLE`, `ALL_ZERO_NON_IDENTIFYING`, `VARIANCE_NOT_GREATER_THAN_MEAN`, `NOT_ASSESSED`, `FAILED`, `RETRY_CAP_EXHAUSTED`.
- **B fits:** Exponential scale and Gamma shape/scale use `abs(cuda-cpu) <= max(5e-12, 5e-10*abs(cpu))`. NB uses `abs(log(r_cuda)-log(r_cpu)) <= 1e-8`, `abs(logit(p_cuda)-logit(p_cpu)) <= 1e-8`, and `abs(LL_cuda-LL_cpu) <= 1e-9*max(1,abs(LL_cpu))`. A flat objective may record `PARAMETERIZATION_DIFFERENCE_ON_FLAT_OBJECTIVE` only with equal category, objective, downstream PMF/CDF and GOF statistic gates.
- **C values:** CDF, SF, logCDF, logSF and PMF/logPMF use `abs(cuda-cpu) <= max(5e-13, 5e-11*abs(cpu))`; extreme values are assessed in log space without clipping.
- **D GOF:** AD/CvM use `abs(T_cuda-T_cpu) <= 2e-11*max(1,abs(T_cpu))`. Where an oracle exists, both engines separately meet DEC-014 oracle requirements.
- **E MC:** the integer exceedance count and reject decision are exact. `p_MC=(b+1)/(B+1)` and ties use `T* >= T_obs`.
- **F batching/RNG:** `1×1`, `2×3`, and one memory-compatible larger partition retain outer/inner indices, category, count, retry accounting and logical results. SHA-256 seed identities agree exactly across backend, batching, order and resume boundaries.

## Fixture matrix and artifacts

The fixed primary composite matrix has 144 cells: Gamma shapes `0.25,0.5,1,2,10`, Exponential scale `1`, NB `r={0.25,1,5,20}` and `p={0.1,0.5,0.9}`, each at `n={20,50,100,250}` and `AD/CVM`. `R_EQ=8`; `B_EQ=15`. Deterministic adversarial fixtures cover NB eligibility cliffs, sparse/heavy-tail NB, Gamma shape 0.25, tail/CDF extremes, and exact/near MC ties.

Generator validation is a separate non-GOF gate: structural contracts are exact; the future sanity check has `N_GENERATOR_SANITY=1_000_000`, `|z_mean| <= 5`, and `|z_variance| <= 5` per frozen fixture.

The future artifact contract is `equivalence_manifest.json`, `fixture_manifest.json`, comparison parquet files, `batch_invariance.json`, `rng_identity.json`, `generator_sanity.json`, `environment.json`, `summary.json`, and `digests.json`. Initial `summary.json` begins `equivalence_gate_passed=false` and `calibration_claim=false`.

Any unexplained discrepancy fails the gate. No tolerance, fixture, category or solver tuning after results is permitted without a new architecture decision, candidate SHA, preregistration and independent audit.
