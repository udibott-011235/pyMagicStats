# DEC-019 — CP05-C2C R10-A targeted GPU replay preregistration

- Estado: `proposed`
- Fecha: 2026-09-22
- Owner: `decision-owner`
- Revisores: `statistical-software-architecture`, `implementation-engineering`, `adversarial-statistical-qa`
- Supersedes: none
- Depends on: DEC-016 and DEC-018

## Contexto y estimando

This decision preregisters a future targeted GPU replay after the R10-A
Negative Binomial remediation. It selects historical identities from the
authenticated R9 evidence bundle without executing CUDA, replaying records, or
making an equivalence claim.

The frozen identities are:

```text
REPLAY_EXECUTION_SHA=649ca296c57ab237f3e93ea35ac9870e1d1bfc9d
R9_EVIDENCE_SHA=01c9c0759d3ffec1bbd95cb5aa3f67793aab544b
SOURCE_ARCHIVE_SHA256=2bda9bf446a64d61d035f0948c19570f6177c1ac2ac650968720edb2b6887269
SOURCE_DIGESTS_JSON_SHA256=3782b8f15977131afb6bcb416a3ac356a55dc4fb38f40a6006309dc88f9ca93a
NAMESPACE=CP05-C2C
R_EQ=8
B_EQ=15
```

The archive digest, the complete `digests.json` digest, and every artifact
digest declared by `digests.json` were verified before deriving the selection.
The machine-readable selection is frozen in
[`cp05-c2c-r10a-targeted-gpu-replay-identities.json`](cp05-c2c-r10a-targeted-gpu-replay-identities.json).

## Evidencia R9 persistida y limitación histórica

The three comparison parquet artifacts were joined one-to-one by `identity`.
Their shared `cell_id`, `record_type`, `raw_outer_index`, and
`raw_inner_index` fields agree exactly; `family` agrees between the fit and
statistic artifacts. The historical selection is defined only by persisted
fields:

```text
C = classification gate persisted as false
F = fit gate persisted as false
S = statistic gate persisted as false
H = cuda_failure_reason persisted as non-null/non-empty
A = C union F union S union H
```

The mandatory name for `A` is **R9 persisted failure union (C/F/S/H)**. The
observed topology is:

```text
WORKLOAD_A_PERSISTED_FAILURE_UNION=134
WORKLOAD_A_OBSERVED=10
WORKLOAD_A_BOOTSTRAP=124

CLASSIFICATION_GATE_FALSE=72
FIT_GATE_FALSE=131
STATISTIC_GATE_FALSE=131
CUDA_HARD_FAILURE_RECORDS=71

FIT_STAT_INTERSECTION=129
FIT_ONLY=2
STATISTIC_ONLY=2
PURE_CLASSIFICATION_ONLY=1
FAILED_FAMILIES=negative_binomial only
```

The persisted R9 artifacts do not contain a record-level
`distribution_value_gate_pass` field or enough persisted values to derive it:

```text
DISTRIBUTION_VALUE_GATE_PERSISTENCE=NOT_PERSISTED
```

Therefore, the 134 identities represent exclusively the historical union of
persisted classification, fit, statistic, and hard CUDA failures. They are not
called all R9 failures, all DEC-016 failures, or a complete R9 failure set. It
cannot be claimed that they include every R9 record that might have failed any
DEC-016 gate. Historical sample digests were also not persisted and are not
invented or reconstructed as evidence after the fact.

The five identities explaining the non-identical fit/statistic sets are frozen
without reinterpretation:

### Fit-only

```text
negative_binomial|r=20,p=0.9|n=250|AD|composite|raw_outer=5|raw_inner=2
negative_binomial|r=20,p=0.9|n=250|CVM|composite|raw_outer=2|raw_inner=9
```

Both remain historical `fit_gate_pass=false` records even though their R9
diagnostic says `FLAT_OBJECTIVE=true` and `flat_objective_used=null`.

### Statistic-only

```text
negative_binomial|r=1,p=0.5|n=50|AD|composite|raw_outer=1|raw_inner=2
negative_binomial|r=5,p=0.9|n=250|AD|composite|raw_outer=7|raw_inner=14
```

### Pure-classification-only

```text
negative_binomial|r=1,p=0.9|n=50|CVM|composite|raw_outer=6
```

## Alternativas consideradas

1. Select only the 131 fit failures. Rejected because it would omit two
   statistic-only records and the pure classification mismatch.
2. Describe the union as every historical DEC-016 failure. Rejected because
   the distribution-value gate was not persisted at record level.
3. Regenerate R9 arrays to infer missing historical evidence. Rejected because
   regenerated results are not persisted R9 evidence and would violate the
   read-only provenance boundary.
4. Freeze the persisted `C/F/S/H` union and separately reconstruct complete MC
   outers. Selected because both workloads follow mechanically from the
   authenticated artifacts without changing historical results.

## Decisión y razón

### Workload A — record-level remediation replay

Workload A will execute exactly the 134 identities in the R9 persisted failure
union. For each identity, the future executor must check out
`649ca296c57ab237f3e93ea35ac9870e1d1bfc9d` and reconstruct the array from its
`cell_id`, `raw_outer_index`, optional `raw_inner_index`, namespace, frozen seed
derivation, and frozen generator/retry semantics. Persisted or modified R9
arrays must not be supplied as replay inputs.

For every record actually executed, the future replay evaluates all applicable
current DEC-016 gates, including classification, fit, distribution values, and
statistic. A future distribution-value failure is a replay failure even though
that gate could not be used to select the historical 134 identities.

### Workload B — complete MC outer reconstruction

The R9 statistic artifact mechanically yields:

```text
MC_EVALUATED_OUTERS=970
R9_MC_FAILED_OUTERS=12
MC_UNEVALUABLE_OUTERS=38
WORKLOAD_B_MC_OUTERS=12
```

For every one of the 12 historical discrepancy outers, Workload B reconstructs
and evaluates one observed record plus exactly 15 eligible bootstrap records.
The frozen R9 exceedance counts differ for each selected outer, while all 12
historical CPU and CUDA reject decisions are `false`. The manifest records each
outer, both counts and p-values, both decisions, the observed identity, and the
15 bootstrap identities ordered by `raw_inner_index`.

Before interpreting any MC result, each outer must satisfy the following
reconstruction precondition. The executor reconstructs the observed record from
its frozen statistical identity, applies the frozen seed derivation, generates
bootstrap attempts in `raw_inner_index` order, and applies the frozen eligibility
and retry semantics until exactly `B_EQ=15` eligible bootstraps have been
obtained. Those 15 reconstructed eligible identities must match exactly, and in
the same `raw_inner_index` order, the 15 `bootstrap_identities` frozen for that
outer in the manifest.

Historical `raw_inner_index` values can contain gaps caused by ineligible
attempts. They must not be assumed to be `0..14`, normalized, or replaced by
ordinal positions. If the reconstructed eligible sequence differs from the
manifest, `MC_BOOTSTRAP_IDENTITY_MISMATCH` is nonzero and the executor must
`PRESERVE_EVIDENCE` and `STOP`; it must not search for or substitute a different
set of 15 replicas.

Workload A contains 134 historical persisted identities. Workload B can
reevaluate Workload A identities and can require additional identities from the
same 12 outers. Consequently, 134 is not the total number of GPU evaluations in
the replay. This decision does not freeze a unique-evaluation total.

## Gate preregistrado del replay futuro

The record-level replay gate is:

```text
CLASSIFICATION_MISMATCH=0
CUDA_NONCONVERGENCE=0
FIT_GATE_FAILURE=0
DISTRIBUTION_VALUE_GATE_FAILURE=0
STATISTIC_GATE_FAILURE=0
UNEXPLAINED_DISCREPANCY=0
```

The MC gate is:

```text
MC_BOOTSTRAP_IDENTITY_MISMATCH=0
MC_OUTERS_RECONSTRUCTED=12
MC_EXCEEDANCE_COUNT_MISMATCH=0
MC_REJECT_DECISION_MISMATCH=0
```

The two MC comparison counters have only within-replay semantics:

```text
MC_EXCEEDANCE_COUNT_MISMATCH =
count(replay_b_cpu != replay_b_cuda)

MC_REJECT_DECISION_MISMATCH =
count(replay_reject_cpu != replay_reject_cuda)
```

`replay_b_cpu` and `replay_b_cuda` are the exceedance counts calculated by CPU
and CUDA from the same reconstructed arrays in the same future targeted replay.
Likewise, `replay_reject_cpu` and `replay_reject_cuda` are the CPU and CUDA
decisions calculated in that replay. The persisted `r9_b_cpu`, `r9_b_cuda`,
`r9_p_cpu`, `r9_p_cuda`, `r9_reject_cpu`, and `r9_reject_cuda` values are
historical provenance only, not acceptance oracles. The future replay need not
reproduce the old R9 `b_cuda`; it must establish CPU↔CUDA agreement within the
new replay.

`TARGETED_GPU_REPLAY=PASS` only if every gate above passes exactly. Any
unexpected discrepancy requires:

```text
PRESERVE_EVIDENCE
STOP
NO_AUTOMATIC_RERUN
NO_THRESHOLD_CHANGE
NO_SEED_CHANGE
NO_RNG_CHANGE
NO_RETRY_POLICY_CHANGE
NO_FIXTURE_CHANGE
```

The manifest is a selection and provenance artifact. Historical R9 outcomes
are not acceptance oracles for the future candidate.

## Invariantes congelados

DEC-016 and DEC-018 are not modified or superseded. The replay must preserve
the `CP05-C2C` namespace, `R_EQ=8`, `B_EQ=15`, primary matrix, seed derivation,
RNG, generator semantics, bootstrap eligibility and retry semantics, ties
`>=`, plus-one MC p-value, CPU reference, CUDA candidate mathematics, NB
classification, fit tolerances, distribution-value tolerances, statistic
tolerances, and the flat-objective exception.

## Límites y consecuencias

A future targeted PASS does not imply:

```text
full 1152 equivalence PASS
complete historical R9 failure closure
proof that R9 had no distribution-only failures
R10-B PASS
R10-C PASS
R10-D PASS
statistical calibration
performance validation
production readiness
CP05-C2C completion
```

This preregistration itself executes no GPU workload and makes no CUDA
equivalence, calibration, performance, or production claim. The full campaign
remains prohibited until separately authorized after the required architecture
sequence.

## Condición que obliga a revisar

A new architecture decision is required before changing the evidence source,
selection semantics, replay SHA, namespace, fixtures, seeds, RNG, generator,
bootstrap eligibility/retry behavior, gate tolerances, failure classification,
MC tie/plus-one semantics, or either workload. A mismatch between a reconstructed
identity and its manifest coordinates also requires preserving evidence and
stopping.

## Impacto en API, código, tests y documentación

There is no production, experimental-engine, runner, public API, or test change.
The only implementation artifacts are this proposed decision, its immutable
identity manifest, and their decision index/registry entries. Execution and
independent audit remain future work.
