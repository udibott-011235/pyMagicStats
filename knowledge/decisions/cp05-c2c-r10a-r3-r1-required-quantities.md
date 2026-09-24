# DEC-021 — CP05-C2C R10-A-R3-R1 required distribution quantity completeness

**Status:** `proposed`

**Owner:** Project Owner — Ehud Bottaro

**Architecture:** ChatGPT; implementation: Cortex; independent QA: Antigravity

**Base:** `3d13ac4ae5410cc6957f68cf4a60ebf34c22729f`

**Branch:** `experiments/cp05-c2c-r10a-r3-r1-required-quantities`

## Motivation and historical boundary

DEC-020 remains conceptually valid: distribution-value points and certified
statistic support are separate mathematical objects. Its first implementation
candidate, `3d13ac4ae5410cc6957f68cf4a60ebf34c22729f`, failed Antigravity's
independent audit with `AGY-QA-R3-01`, severity `MAJOR`, verdict `FAIL`.
[EV-018](../evidence/cp05-c2c-r10a-r3-required-quantities-audit-failure.md)
preserves that result. Neither this remediation nor passing local tests erases
the historical FAIL or constitutes independent closure of the finding.

The failed implementation iterated CPU adapter dictionary items. A quantity
omitted by CPU, or by both engines, was not compared and could disappear from
the gate. Nonempty evidence for the remaining quantities was insufficient to
establish the complete family contract.

## Frozen rule

THE FAMILY CONTRACT DEFINES THE COMPARISON UNIVERSE.

ADAPTER OUTPUT DOES NOT DEFINE THE COMPARISON UNIVERSE.

The single canonical source is `required_distribution_quantities(family)` in
`cp05_c2c_equivalence_runner.py`. It returns exactly:

```python
# Negative Binomial
("pmf", "logPMF", "cdf", "sf", "logCDF", "logSF")

# Gamma and Exponential
("cdf", "sf", "logCDF", "logSF")
```

Unknown families raise `C2CError`; no contract is inferred from adapter keys.
For the three frozen families, both engine key sets must equal the canonical
required set exactly. Missing and unexpected quantities fail closed.
Numerical iteration follows canonical tuple order, not adapter insertion order.

## Evaluation sequence and structural evidence

`evaluate_fixed_record` performs these ordered checks:

1. Exact evaluation-point identity, including order.
2. Exact required-quantity completeness for both engines, including rejection
   of extra keys.
3. Exact value-vector length against the matching point vector, for every
   required quantity.
4. Existing DEC-016 numerical comparison at each matched point.
5. Distribution gate: point identity, quantity completeness, nonempty evidence
   and all evidence items passing.

All structural checks finish before any numerical distribution comparison.
A structural failure anywhere prevents numerical comparison of the other
quantities too. Later fit/statistic evaluation cannot turn the failed
distribution gate into PASS. Existing downstream gating remains unchanged.

Each missing required quantity receives an explicit evidence item with one of:

```text
MISSING_CPU_DISTRIBUTION_QUANTITY
MISSING_CUDA_DISTRIBUTION_QUANTITY
MISSING_BOTH_DISTRIBUTION_QUANTITY
```

Each extra key receives `UNEXPECTED_DISTRIBUTION_QUANTITY` and an `engines`
list containing `CPU`, `CUDA`, or both. Extra-key evidence is deterministically
sorted by quantity. These use the existing structural sentinel:

```text
evaluation_point=null
cpu_value=null
cuda_value=null
allowed_tolerance=null
abs_error=Infinity
passed=false
```

Null value fields denote an unperformed pointwise comparison, not a measured
numeric discrepancy. Missing-engine identity is carried by `failure_reason`;
extra-engine identity is carried by `engines`. An entirely absent adapter
mapping reports every required quantity as missing.

R3 point-identity and value-length failure reasons remain unchanged. A point
mismatch is recorded first; simultaneous missing/extra quantities are still
reported explicitly. Length checks require matching points and complete keys.
No structural failure is reinterpreted as a numerical PMF/CDF discrepancy.

## Regression contract

`tests/research/test_cp05_c2c_r10a_r3_r1_required_quantities.py` independently
states the expected sets and adds 73 parameterized cases:

- 18 NB omissions: each of six quantities absent from CPU, CUDA or both.
- 24 continuous omissions: four quantities, three engine cases, two families.
- 9 extra-key cases: each family and engine combination.
- 9 absent-mapping cases: every required missing quantity has evidence.
- 3 canonical-contract cases and one unknown-family rejection.
- 3 successful canonical-order cases despite different dictionary order.
- 5 nonempty-evidence probes: missing quantity, wrong points, wrong lengths,
  numeric disagreement, and a complete passing control.
- One combined point/completeness failure with ordered structural evidence.

Structural probes replace the numerical comparison with a function that fails
if called. The 30 existing R3 cases remain unmodified and cover the exact
DEC-019 sample digest, sample maximum 9, certified support stop 169, value
points 0..9, full statistic support 0..169, continuous sorted-unique semantics,
point/order/length/empty-evidence failures and unchanged DEC-016 tolerances.

## Validation and limits

The baseline is the previous candidate's recorded 121 passing no-GPU tests;
that result did not detect the later independent MAJOR finding. The same six
test files plus the new R3-R1 file are required for this candidate. No existing
test expectation, selection filter or test body is changed.

Environment: Windows, repository-local `.venv/Scripts/python.exe`, Python
3.12.14, NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1.
Use `PYTHONDONTWRITEBYTECODE=1` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

```text
python -m pytest -q tests/research/test_cp05_c2c_equivalence_runner.py tests/research/test_cp05_c2c_a2_artifact_integration.py tests/research/test_cp05_cuda_nb_r10a.py tests/research/test_cp05_cuda_equivalence_preregistration.py tests/research/test_cp05_cuda_engine.py tests/research/test_cp05_c2c_r10a_r3_value_grid.py tests/research/test_cp05_c2c_r10a_r3_r1_required_quantities.py
python knowledge/tools/validate_registry.py
python -m py_compile experiments/distribution_gof/cuda_calibration/cp05_c2c_equivalence_runner.py tests/research/test_cp05_c2c_r10a_r3_r1_required_quantities.py
git diff --check
git diff --cached --check
```

The focused R3 plus R3-R1 selection reports **103 passed** in 22.74 seconds.
An additional implementation-only sensitivity probe loads the base evaluator
from its Git blob in memory: all six CPU-only and all six both-engine omissions
incorrectly pass there; all six CUDA-only omissions fail. The remediated
evaluator rejects all 18 omissions. This reproduces the reported mechanism
locally without rewriting the base or claiming an independent QA verdict.
The complete selection reports **194 passed** in 1233.40 seconds: all 121
previous tests plus 73 new R3-R1 cases, with no CuPy/CuPyX modules loaded.
Registry validation, `py_compile`, working-tree and staged diff checks pass.
AST comparison confirms only `evaluate_fixed_record` changed among existing
runner functions; the only new top-level function is the canonical quantity
helper. Historical artifacts and all existing registry records are unchanged.
The final local commit identity is recorded in the implementation handoff.
Tests establish local implementation behavior only, not GPU equivalence,
statistical calibration or independent QA approval.

## Frozen scope

No changes to `cuda_candidate.py`, `nb_support.py`, canonical value grid,
DEC-016 tolerances, DEC-018 root certification, historical DEC-019 or manifest,
the audited replay harness/tests, R_EQ/B_EQ, RNG, seeds, fixtures, retry,
classification, NB solver, flat-objective exception, MC, AD/CvM equations,
tail certification or remainder bound. DEC-020 and EV-017 remain historical
and unchanged; this decision supplements DEC-020 rather than superseding it.

Only one new local commit is authorized. No push, PR, merge, GPU, Quantum,
targeted replay, full campaign, CP05-D or holdout access. The next role is
ChatGPT architecture, followed by independent audit of the exact new SHA;
this candidate does not inherit an audit PASS and does not authorize replay.
