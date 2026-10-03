# DEC-028 — CP05-C2C R11 R4 Runtime Oracle Identity Correction

- Status: `accepted`.
- Date recorded: 2026-10-02.
- Owner: Project Owner — Ehud Bottaro.
- Architecture: ChatGPT — statistical/software architecture.
- Documentary implementation: Cortex — implementation engineering.
- Independent QA: Antigravity — adversarial statistical/software QA.
- Related decisions/evidence: `DEC-027`, `EV-021`.
- Supersedes: none.

DEC-028 is a narrow provenance-binding correction. It overrides one erroneous
artifact/digest association in DEC-027 and does not supersede DEC-027 as a
whole.

## Context

Accepted DEC-027 correctly identifies the R4 runtime archive by filename, but
accidentally assigns to that archive the SHA-256 belonging to the separately
extracted crossings report. Canonical historical evidence `EV-021`
distinguishes the archive, crossings report and extracted workload records
unambiguously.

## Canonical EV-021 identities

```text
R4_RUNTIME_ORACLE_ARCHIVE=
cp05_c2c_r10a_targeted_r4_501d752_run2_PASS_EVIDENCE.tar.gz

R4_RUNTIME_ORACLE_ARCHIVE_SHA256=
dd8de17dfa17ac54855f9823d053e6820a2a285aeb8467eec390aea51dcba6ad

R4_RUNTIME_CROSSINGS_REPORT=
cp05_c2c_r10a_targeted_r4_501d752_run2_crossings.json

R4_RUNTIME_CROSSINGS_REPORT_SHA256=
f7c35e51e0273eac73f9743010b0191cc3c33fac9dde88cdbc123200d86e10f7

R4_RUNTIME_WORKLOAD_B_RECORDS=
workload_b_records.jsonl

R4_RUNTIME_WORKLOAD_B_RECORDS_SHA256=
beea7dffa4a059de241703079e254a696a8ad7764cd9b6288079a84188ee5f37
```

## Decision

DEC-028 corrects only the value associated with
`R4_RUNTIME_ORACLE_ARCHIVE_SHA256` in DEC-027. The incorrect association:

```text
PASS_EVIDENCE.tar.gz
->
f7c35e51e0273eac73f9743010b0191cc3c33fac9dde88cdbc123200d86e10f7
```

is replaced by:

```text
PASS_EVIDENCE.tar.gz
->
dd8de17dfa17ac54855f9823d053e6820a2a285aeb8467eec390aea51dcba6ad
```

The value `f7c35e51e0273eac73f9743010b0191cc3c33fac9dde88cdbc123200d86e10f7`
remains valid historical evidence, bound only to the crossings JSON artifact.
`workload_b_records.jsonl` and its digest remain unchanged. No historical
artifact is regenerated, reinterpreted, replaced or modified.

## Narrow supersession semantics

```text
DEC027_R4_ARCHIVE_SHA_BINDING_CORRECTED=YES
DEC027_OTHER_CONTENT_SUPERSEDED=NO
EV021_CHANGED=NO
R4_HISTORICAL_EVIDENCE_CHANGED=NO
```

## Scientific invariants unchanged

DEC-028 overrides exactly one erroneous evidence-binding value in DEC-027.
All other accepted DEC-027 scientific and governance terms remain normative.

```text
ALPHA=0.05
B_R11=199
R11_SOURCE_OUTERS=12
R11_TOTAL_MC_RECORDS=2400

R11_SCIENTIFIC_BASE_SHA=
d7412911dcf8402f01a502fe0d756813ec10bbee

R11_SCIENTIFIC_BASE_TREE=
421c07a3466a689e8c49da5c1cfe25fb946a0a5b
```

DEC-028 does not alter:

- source outer identities or order;
- source projection SHA;
- seed derivation;
- NumPy/SciPy reference semantics;
- generator semantics;
- eligibility semantics;
- `B=199`;
- `alpha=.05`;
- retry cap;
- exact sample-byte contract;
- R4 prefix requirements;
- DEC-024 exact-tie semantics;
- R11 outer PASS;
- global R11 PASS;
- boundary fixtures;
- maximum permitted R11 claim; or
- historical R10-A conclusions.

## Future R11 workload acceptance gate

The future R11 workload acceptance gate must independently:

1. Verify the R4 frozen identity manifest hash.
2. Verify the R4 archive hash equals
   `dd8de17dfa17ac54855f9823d053e6820a2a285aeb8467eec390aea51dcba6ad`.
3. Verify the extracted `workload_b_records.jsonl` hash equals
   `beea7dffa4a059de241703079e254a696a8ad7764cd9b6288079a84188ee5f37`.
4. Verify the historical observed plus first 15 bootstrap
   identity/seed/sample-digest prefix.
5. If the crossings report is supplied or used, verify its own hash equals
   `f7c35e51e0273eac73f9743010b0191cc3c33fac9dde88cdbc123200d86e10f7`.

```text
ARTIFACT_HASH_SUBSTITUTION=PROHIBITED
ORACLE_IDENTITY_DOWNGRADE=PROHIBITED
```

No artifact may satisfy another artifact's identity gate merely because its
digest appears elsewhere in historical evidence.

## Authorization boundary

```text
R11_REFERENCE_WORKLOAD_BUILDER=NOT_AUTHORIZED_BY_DEC028
R11_REFERENCE_WORKLOAD_CONSTRUCTION=NOT_AUTHORIZED
R11_HARNESS_IMPLEMENTATION=NOT_AUTHORIZED
R11_GPU_EXECUTION=NOT_AUTHORIZED

FULL_1152=NOT_AUTHORIZED
CP05_D=NOT_AUTHORIZED
HOLDOUT_ACCESS=NOT_AUTHORIZED
```

DEC-028 is a documentary provenance correction only. It does not authorize a
builder or harness implementation, workload or sample construction, GPU/CUDA
execution, full-1152, CP05-D or holdout access.
