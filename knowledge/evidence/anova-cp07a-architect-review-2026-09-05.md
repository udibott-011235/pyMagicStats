# CP-ANOVA-07A — Architect review

- Fecha: `2026-09-05`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-07B`
- Harness branch: `engineering/cp-anova-07a-harness`
- Candidate SHA: `b211dcd61be02a234386947305bc1a1c9cfffde7`
- Parent/base: `6f4c62dcb8ac6a45009ad9b32c8aa52aa92e622f`
- Review branch: `audit/anova-cp07a-architect-review`
- Decision: `APPROVED_FOR_E0`

## Scope integrity

`b211dcd...` is exactly one commit ahead of the frozen v1.1 preregistration base. The diff contains nine added files only:

- `experiments/anova_cp07/README.md`
- `experiments/anova_cp07/__init__.py`
- `experiments/anova_cp07/__main__.py`
- `experiments/anova_cp07/accounting.py`
- `experiments/anova_cp07/manifest.py`
- `experiments/anova_cp07/requirements.txt`
- `experiments/anova_cp07/runner.py`
- `experiments/anova_cp07/simulation.py`
- `tests/experiments/test_anova_cp07.py`

No production ANOVA, selector, robustness policy, frozen knowledge source or `main` file is modified by the candidate.

## Manifest and identity

The harness pins preregistration `anova-calibration-prereg-v1.1`, validates the compact manifest against a canonical SHA-256, materializes and compares the complete records, and asserts exactly 197 unique IDs across the frozen phase counts.

RNG identity follows the frozen contract:

```text
phase|cell_id|replicate_index
UTF-8 -> SHA-256 -> first four little-endian uint32 -> SeedSequence([master_seed, ...]) -> PCG64
```

Worker count, batching, sharding and scheduling are not RNG inputs.

## Paired execution

Each replica generates one tuple of groups, computes one summary tuple through `_summarize_groups`, then runs `_classical_kernel` and `_welch_kernel` on that same summary object. The selector and historical `OneWayRobustness` are not part of the calibration execution path.

The public-API parity gate uses 32 deterministic replicas per active cell with `rtol=1e-12`, `atol=1e-14`, and aborts on nonfinite output or mismatch.

Cortex reports the five unsealed phases produced 9,856 kernel/public-API comparisons with zero mismatches, corresponding to 154 unsealed cells × 32 replicas × 2 methods.

## Accounting

Errors are represented explicitly as generation/kernel/nonfinite states and are not treated as non-rejections. Method-specific completed denominators count only finite successful results. Paired accounting independently counts replicas in which both methods succeeded and enforces:

```text
both + classical_only + welch_only + neither == paired_completed
```

Wilson 99% intervals use the preregistered z constant. Confirmatory Type-I gates are applied only to complete D-core/H-core outputs, with Classical gated only under equal SD and Welch gated over normal equal/unequal SD designs. Partial shards cannot receive a confirmatory PASS.

## Persistence, shards and resume

Artifacts are transactionally written into a sibling pending directory, checksumed and verified before atomic directory publication. Completed artifacts are immutable. Resume verifies provenance/checksums/accounting and does not append or recompute a valid completed shard. Recomposition requires a complete unique shard set, merges replica keys deterministically and rebuilds accounting.

## Holdout

H phases fail closed without explicit authorization before generating samples. The runner validates a future opening declaration against the exact harness SHA and manifest SHA and requires `phase_d_complete`, `remediations_closed`, and `candidate_frozen` to be true. There is no generic boolean holdout bypass in the CLI.

This is a governance guard, not a cryptographic identity proof, which is acceptable for the current project contract.

## Test evidence

Cortex handoff reports:

```text
86 tests passed
8 warnings
197 IDs validated
9,856 parity comparisons
0 mismatches
```

The architect inspected the committed test suite and found explicit coverage for manifest drift, exact RNG replay, paired sample/summary execution, workers/batches/shards reproducibility, public API parity, holdout fail-closed behavior, error accounting, Wilson interval reference values, transaction/recompose/resume, interrupted publication, selector exclusion, generator transforms, power offsets, confirmatory gates and frozen engine provenance.

No GitHub CI/check status is registered for this commit; the reported test run is therefore handoff evidence rather than an independent remote CI run. This does not block the engineering-only E0 phase, but Phase D must not be authorized solely from E0.

## Decision

`CP-ANOVA-07A` implementation is accepted and `CP-ANOVA-07B` architect review is complete.

Candidate `b211dcd61be02a234386947305bc1a1c9cfffde7` is **APPROVED_FOR_E0** only.

E0 remains engineering-only:

- 12 frozen E0 cells;
- 200 replicas/cell;
- 2,400 datasets total;
- CPU only;
- no statistical PASS/FAIL interpretation;
- no D phase authorization;
- no H phase opening.

The E0 execution must use the exact candidate SHA above from a clean checkout and write artifacts outside the repository. After E0, stop at `CP-ANOVA-07C` and return artifacts/accounting/performance/reproducibility evidence for architect review before Phase D.
