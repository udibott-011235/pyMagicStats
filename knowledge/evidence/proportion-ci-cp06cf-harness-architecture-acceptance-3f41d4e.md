# CP-06 C–F harness architecture acceptance — `3f41d4e`

**Stage:** `STAGE-PROP-CI-001`  
**Work item:** `CP06-C-F-HARNESS-CORRECTION-3`  
**Accepted/frozen harness SHA:** `3f41d4ea0d193968c0bbe7080e49cdbe784bf1ac`  
**Parent:** `6c8861cc26e4d2967be87a9bf853895bfce63fb5`  
**Frozen production candidate:** `2df5b90a5395163e723f9c52aafbb91fdce96d43`  
**Architecture status:** `ACCEPTED FOR QUANTUM EXECUTION`  
**Statistical calibration status:** unchanged / `not_calibrated`

## Scope verification

GitHub compare confirms `3f41d4e...` is exactly one commit ahead of `6c8861c...`. The correction changes only:

- `experiments/proportion_ci_calibration/harness.py`
- `experiments/proportion_ci_calibration/high_precision.py`
- `experiments/proportion_ci_calibration/run.py`
- `tests/experiments/test_proportion_ci_calibration.py`

No production path under `pyMagicStat/` changed.

The remote branch `experiments/proportion-ci-calibration` resolves exactly to `3f41d4ea0d193968c0bbe7080e49cdbe784bf1ac`.

## Round-3 findings closed

### High-precision structural predicates — CLOSED

CP06-F now re-evaluates the actual paired predicate at >=80 digits rather than resolving structural findings from independent endpoint agreement alone:

- complement symmetry pairs `x` with `n-x`;
- endpoint monotonicity pairs adjacent outcomes and preserves lower/upper kind;
- confidence nesting compares the wider and narrower alpha endpoints at the same outcome;
- bounds and NaN/Inf are evaluated directly at high precision.

Persistent HP violations are classified `confirmed_structural_violation` with `resolved=false`. Float64-only failures restored by HP are classified `float64_structural_artifact` with `resolved=true`.

### Structural queue context — CLOSED

The queue carries reconstructible paired context including `complement_x`, `x_left`, `x_right`, `endpoint_kind`, `alpha_wider` and `alpha_narrower`.

### Shard semantic provenance — CLOSED

Shard/result semantics are now versioned independently as `cp06-shard-schema-v3`; cache paths, checkpoint hashes, shard hashes, provenance validation and metadata include this identity. v2/legacy shards cannot silently feed the new run.

Endpoint payload semantics did not change, so the endpoint cache deliberately retains `cp06-harness-schema-v2` identity and remains reusable after SHA/provenance validation.

### Classification semantics — CLOSED

`confirmed_exact_coverage` is reserved for Clopper–Pearson. Wilson/Wald/Jeffreys without HP shortfall use `confirmed_no_shortfall_at_audited_cell`; Jeffreys remains explicitly a `bayesian_comparator` and does not acquire a frequentist exactness claim.

## Test evidence supplied by implementation agent

Focused harness suite reported:

```text
70 passed
0 failed
0 skipped
0 warnings
8.24 s
```

Heavy B/C/D/E/F execution was not performed by the implementation agent, preserving the execution split: Cortex/Codex validates the harness in small tests; Quantum owns confirmatory/heavy execution.

## Evidence-transfer consequence

Previous CP06-B coverage evidence is not globally transferable because the Wald acceptance-set semantics and later trigger/provenance semantics changed after the original B run.

Therefore Quantum must regenerate CP06-B under the frozen harness SHA before C consumes B shards. Old B raw artifacts should be preserved separately for traceability but must not be reused as v3 evidence.

## Execution decision

The architecture HOLD is lifted for harness SHA `3f41d4ea0d193968c0bbe7080e49cdbe784bf1ac`.

Authorized next sequence:

1. Quantum checkout exact frozen harness SHA and verify frozen production ancestry.
2. Preserve prior B artifacts as historical/non-transferable evidence.
3. Rerun CP06-B under shard schema v3 and inspect gates/results.
4. If B is clean, execute C.
5. Before full D, benchmark a bounded stress shard and record wall/RSS; performance may influence execution strategy but must not change the preregistered statistical domain without a new architecture decision.
6. Execute D, E and F sequentially only after preceding evidence is accepted.

No PR, merge, selector activation or production calibration-status change is authorized by this acceptance.
