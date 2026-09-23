# DEC-020 — CP05-C2C R10-A-R3 distribution-value grid / GOF-support separation

**Status:** `proposed`

**Owner:** Project Owner — Ehud Bottaro

**Architecture:** ChatGPT; implementation: Cortex; independent QA: Antigravity

**Base:** `94a2d530e768da68085b18692c7eb6c34903e608`

**Branch:** `experiments/cp05-c2c-r10a-r3-value-grid-separation`

## Frozen architectural contract

Statistical identity remains unchanged.

Distribution-value evaluation points and GOF tail-support are separate
mathematical objects. The Owner-authorized R3 contract formalizes the existing
CPU reference value grid; it does not introduce a new grid or change DEC-014,
DEC-016, DEC-018 or historical DEC-019.

For Negative Binomial:

```python
CANONICAL_VALUE_POINTS = tuple(range(max(sample) + 1))
CERTIFIED_STATISTIC_SUPPORT = certified_support.indices
```

CPU and CUDA independently derive `CANONICAL_VALUE_POINTS` using
`canonical_distribution_value_points(cell, sample)`. Both evaluate PMF,
logPMF, CDF, SF, logCDF and logSF on those points with their own fitted
parameters. Neither engine receives the other's fitted result.

For continuous families the helper preserves sorted-unique observed-sample
semantics. The immutable tuple representation does not alter the points.

The CPU statistic remains the CP04 / DEC-014 reference. The CUDA statistic
continues receiving the complete certified support and unchanged remainder
bound. Tail points beyond the sample maximum must not enter the value grid or
be removed from the statistic support.

## Motivation and evidence boundary

[EV-017](../evidence/cp05-c2c-r10a-targeted-replay-r1-failure.md) records the
Owner-reported single failed DEC-019 replay. Classification, fit and statistic
gates passed at the first failed identity; the distribution-value gate emitted
six length-mismatch sentinels. In the base adapter, CUDA value points were the
extended statistic support while CPU points stopped at the sample maximum.
This establishes a point-identity mismatch, not a numerical PMF/CDF discrepancy.
The external archive has not been independently audited by Cortex.

## Fail-closed comparison and evidence compatibility

`evaluate_fixed_record` first compares the two point vectors exactly, including
order. Different vectors, even of equal length, are never zipped. For matching
vectors, every value list must cover the entire point vector; equal but short
or long value lists also fail closed rather than being truncated by `zip`.

The minimal additive evidence field is `failure_reason` on structural-failure
items in `distribution_evidence`:

- `EVALUATION_POINT_IDENTITY_MISMATCH`: different point vectors.
- `DISTRIBUTION_VALUE_LENGTH_MISMATCH`: values do not cover the matching grid.

Existing structural sentinels (null point/values/tolerance, infinite absolute
error, `passed=false`) remain for compatibility. The reason explicitly marks
them as structural, not measured numerical discrepancies. Numerical comparison
items retain their existing schema and concrete point/value/tolerance evidence.
Empty evidence cannot pass. No gate is bypassed.

DEC-016 remains exactly:

```text
abs(cuda-cpu) <= max(5e-13, 5e-11*abs(cpu))
```

## Invariants and no-GPU acceptance

| ID | Contract / regression |
|---|---|
| R3-I01 | CPU and CUDA evaluation-point vectors are exactly identical. |
| R3-I02 | NB points are the integers zero through sample maximum inclusive. |
| R3-I03 | Extending certified support does not change value points. |
| R3-I04 | Complete support and remainder bound reach `candidate_statistic`. |
| R3-I05 | Both adapters independently apply the same value-grid helper. |
| R3-I06 | Point identity is checked before any value comparison. |
| R3-I07 | Equal-length different vectors fail closed without numerical zipping. |
| R3-I08 | Absolute/relative DEC-016 value tolerances are unchanged. |
| R3-I09 | Each of the six NB quantities remains a required comparison gate. |
| R3-I10 | Solver and statistical tail mathematics remain untouched. |

The exact regression uses existing `fixed_observed`, `reference_fit`,
`derive_seed` and `_generate` machinery, namespace `CP05-C2C`, cell
`negative_binomial|r=0.25,p=0.1|n=20|AD|composite`, raw outer `2`, raw inner `7`.
It does not replace or hard-code a sample. Reconstructed SHA-256:
`bca89c41ebd5857f10a8b9908766c6dd9486eb0bd0bbb6813304823a83c15a0b`.
The sample maximum is `9`; DEC-014 certification stops at `169`. Values use
`tuple(range(10))`; the statistic receives `tuple(range(170))` and the original
remainder bound. Mock candidate primitives prove routing, not CUDA numerics.

Regression source:
`tests/research/test_cp05_c2c_r10a_r3_value_grid.py`.

## Reproducible local validation

Environment: Windows, Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1;
repository-local `.venv/Scripts/python.exe`. Tests run with
`PYTHONDONTWRITEBYTECODE=1` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

```text
python -m pytest -q tests/research/test_cp05_c2c_equivalence_runner.py tests/research/test_cp05_c2c_a2_artifact_integration.py tests/research/test_cp05_cuda_nb_r10a.py tests/research/test_cp05_cuda_equivalence_preregistration.py tests/research/test_cp05_cuda_engine.py tests/research/test_cp05_c2c_r10a_r3_value_grid.py
python knowledge/tools/validate_registry.py
python -m py_compile experiments/distribution_gof/cuda_calibration/cp05_c2c_equivalence_runner.py tests/research/test_cp05_c2c_r10a_r3_value_grid.py
git diff --check
```

The first four pytest files also define the pre-edit baseline: **83 passed**
in 1328.79 seconds. The complete candidate selection above reports **121 passed**
in 1325.11 seconds, with no `cupy`/`cupyx` modules loaded.
No thresholds, samples, test bodies or selection filters in existing tests are
changed. The new R3 file alone reports **30 passed** in 10.55 seconds. Registry
validation, `py_compile`, working-tree and staged `git diff --check` pass.
AST comparison confines edits to the two adapters and `evaluate_fixed_record`,
plus the new helper. Existing registry records and frozen artifacts are unchanged.
The final candidate identity accompanies the implementation handoff.

## Frozen scope and handoff

No changes to `cuda_candidate.py`, `nb_support.py`, DEC-016/018/019, RNG,
fixtures, seeds, retry, classification, R_EQ/B_EQ, root certification,
flat-objective exception, MC, AD/CvM equations or tail tolerances/mathematics.
The three audited `targeted_replay_r1` artifacts and their EOL protection remain
byte-identical. No GPU, Quantum, targeted replay, full campaign, CP05-D or
holdout execution/access is authorized here.

One local implementation commit is authorized, without push, PR or merge.
Local passing tests do not establish GPU equivalence or calibration. This
decision remains proposed pending architecture review and independent audit
of the new candidate SHA. A future replay requires separate preregistration.
