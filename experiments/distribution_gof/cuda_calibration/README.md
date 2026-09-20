# CP05-C2A CUDA/RAPIDS experimental engine

This directory is an **experimental-only**, non-production prototype for a
future Quantum/RAPIDS host. It is not routed through `pyMagicStat`, is not a
public API, and makes no calibration claim.

Supported future procedures are composite-null Gamma, Exponential and Negative
Binomial GOF with AD or CvM. The script keeps DEC-014 formulas explicit:
continuous AD receives stable `logcdf`/`logsf` values without clipping;
discrete NB uses the certified-support formulas; bootstrap p-values use
plus-one and `>=`; and NB all-zero / variance-not-greater-than-mean samples are
mathematically ineligible rather than failures.

NB discrete AD/CvM primitives require caller-provided certified support and
truncation. The caller must demonstrate a DEC-014-compatible tail-remainder
bound; the engine never silently extends or hides the support truncation.

RNG identity is SHA-256 over namespace, canonical cell ID, raw outer index,
purpose and raw inner index. GPU random generation is deliberately separate;
bitwise CPU/GPU identity is not claimed. Batching is shaped as
`outer_batch × bootstrap_batch × n` and neither dimension requires retaining a
full campaign in VRAM.

The intended bundle is `manifest.json`, `results.parquet`, `accounting.json`,
`environment.json`, `summary.json`, and `digests.json`, all marked
`CUDA_RAPIDS_EXPERIMENTAL` until the CP05-C2B equivalence gate passes.

Expected future Quantum command (not authorized in C2A):

```text
python experiments/distribution_gof/cuda_calibration/cp05_cuda_engine.py --family gamma --parameters '{"shape":2,"scale":1}' --n 50 --statistic AD --null-type composite --B 199 --R 20 --outer-batch-size 4 --bootstrap-batch-size 32 --seed-namespace <authorized-namespace> --output <run-dir>
```

Known limitations: CuPy/RAPIDS is optional locally; C2A intentionally has no
long-run executor, no CPU↔CUDA tolerances, no performance claim, and no
equivalence or calibration result. R1's small composite fixture runner calls
the CP04 reference MLE for observed samples and every bootstrap replicate;
that preserves the objective and failure semantics but is not CPU↔CUDA
equivalence evidence.

CP05-C2C adds `cp05_c2c_equivalence_runner.py`: a fail-closed, fixed-data
DEC-016 runner contract. It accepts only the frozen `R_EQ=8`, `B_EQ=15`,
144-cell/1152-outer identity design and records that bootstrap fixtures are
constructed with `CPU_REFERENCE_FITTED_PARAMETERS`. It exposes exactly the
preregistered artifact bundle, three fixed batch partitions, fixture digests,
and the seven-case `N=1_000_000` generator-sanity path. It has no calibration
mode, no partial PASS artifact, and requires `--require-gpu`; execution on a
CUDA/Quantum host remains separately authorized.
