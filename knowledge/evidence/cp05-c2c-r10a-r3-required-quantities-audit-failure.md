# EV-018 — R10-A-R3 independent audit failure — missing required quantities

**Status:** `proposed`

**Recorded by:** Cortex — Implementation Engineering

**Source:** Antigravity independent adversarial QA, supplied in the
Project Owner's R3-R1 implementation instruction.

## Immutable historical audit result

```text
AUDITED_SHA=3d13ac4ae5410cc6957f68cf4a60ebf34c22729f
AUDITED_TREE=3981d9053668e4d949676e61a090bc4e9b2ab5a9
FINDING=AGY-QA-R3-01
SEVERITY=MAJOR
VERDICT=FAIL
NB_REQUIRED_QUANTITIES_COMPLETE=FAIL
MISSING_CPU_QUANTITY_FAIL_CLOSED=FAIL
MISSING_CUDA_QUANTITY_FAIL_CLOSED=PASS
READY_FOR_NEXT_PREREGISTRATION=NO
```

Cortex is recording Antigravity's reported independent result, not claiming
to have performed that audit or independently verified a remote artifact.
The local audited SHA/tree are the authorized remediation base. No new remote
inspection, GPU execution or replay is part of this work.

## Reported observations

| Removed quantity | Missing CPU only: gate | Missing CUDA only: gate | Missing both: gate |
|---|---|---|---|
| pmf | True | False | True |
| logPMF | True | False | True |
| cdf | True | False | True |
| sf | True | False | True |
| logCDF | True | False | True |
| logSF | True | False | True |

The missing-CPU observations refer to separate removal of each of the six
quantities, not to an empty CPU dictionary. The defect was the adapter-defined
comparison universe:

```python
for quantity, cpu_values in cpu.get("distribution_values", {}).items():
```

A quantity omitted by CPU, including an omission by both engines, did not
produce a required failing comparison. Remaining passing evidence could
therefore approve an incomplete record. This violates DEC-020 R3-I09: each of
the six NB quantities remains a required comparison gate. It is a structural
completeness defect, not evidence of a numerical CPU/CUDA PMF/CDF discrepancy.

## Remediation boundary

[DEC-021](../decisions/cp05-c2c-r10a-r3-r1-required-quantities.md) freezes the
family-defined required sets, missing/extra-key evidence, and structural checks
before numerical comparison. Its tests include independent omission probes,
extra keys, continuous families and nonempty-evidence counterexamples.

The historical candidate and its FAIL are not rewritten. DEC-020's value-grid
and statistic-support separation remains conceptually valid. Local remediation
tests cannot close `AGY-QA-R3-01` on Antigravity's behalf or establish readiness
for the next preregistration. The new SHA requires architecture review and
independent adversarial QA. No push, PR, merge or experimental execution is
authorized by this evidence record.
