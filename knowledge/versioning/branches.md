# Branch lifecycle

> This document is a human-readable projection. knowledge/registry.json is canonical.

Observación materializada: `2026-08-30`. Consulte `EV-003` para la evidencia
Git reproducible, `EV-005` para la integración de Gate 2 y `DEC-006` para la
autoridad de lifecycle.

| ID | Rama | Status | Relación | Integración | Ahead/behind | HEAD observado | Siguiente acción resumida |
|---|---|---|---|---|---:|---|---|
| BR-001 | `main` | `accepted` | `canonical` | `not_applicable` | 0/0 | `f1725ebdfebcb667c053420e4cb4c1e35048f9e0` | ninguna sin decisión expresa |
| BR-002 | `audit/global-main-a0881c4` | `archived` | `fully_contained` | `not_applicable` | 0/8 | `a0881c479bcc0496f79d0f8477d53a41a91907d9` | conservar archivada |
| BR-003 | `docs/project-knowledge-base` | `archived` | `fully_contained` | `merged` | 0/17 | `0a853ba4f25dd160bd8f182e221744280cd980a8` | integrada vía PR #1; conservar archivada |
| BR-004 | `experiments/el-vs-t-calibration-harness` | `archived` | `fully_contained` | `merged` | 0/12 | `05bc7106cca40fafc64ea78433f637ddbdfe48c5` | conservar archivada |
| BR-005 | `feature/anova-engine` | `under_review` | `diverged` | `pending` | 4/21 | `9ebbe4fd1f6b9f847be75f7add09fee609ebe383` | esperar decisión; no es merge candidate |
| BR-006 | `feature/empirical-likelihood-mean` | `archived` | `fully_contained` | `merged` | 0/13 | `427d75b4ea2f72a0e6c6aabbc5b79084721c698e` | conservar archivada |
| BR-007 | `fix/el-ci-numerical-convergence` | `archived` | `fully_contained` | `merged` | 0/9 | `c3c3834f177b8161fb25a9028251a755360a7ee9` | conservar archivada; integrada vía PR #2 |
| BR-008 | `fix/el-vs-t-calibration-accounting` | `archived` | `fully_contained` | `merged` | 0/11 | `51d74e74386eed1c0fe4cc4e90b394dc119acc85` | conservar archivada |
| BR-009 | `fix/el-vs-t-cupy-generator-compatibility` | `archived` | `fully_contained` | `merged` | 0/10 | `c8dd9ab949f12801944cb465fc5bba8186a70134` | conservar archivada |
| BR-010 | `fix/gate2-major-remediation` | `superseded` | `fully_contained` | `not_planned` | 0/8 | `a0881c479bcc0496f79d0f8477d53a41a91907d9` | no borrar; placeholder histórico |
| BR-011 | `fix/gate2-distribution-gof-remediation` | `superseded` | `fully_contained` | `merged` | 0/7 | `0fc71c90c15f7c82b55ba650de742265d492df33` | incorporada indirectamente como ancestro de BR-012 vía PR #3; conservar |
| BR-012 | `fix/gate2-adversarial-remediation` | `archived` | `fully_contained` | `merged` | 0/6 | `9a87c5d48dba8b8a172b5386d7318e7f37ec98fe` | integrada vía PR #3; conservar archivada |
| BR-013 | `refactor/distribution-shape-contract` | `archived` | `fully_contained` | `merged` | 0/17 | `46b9f9fa7cee47466154541ea086ada5f5a4e1eb` | conservar archivada |
| BR-014 | `refactor/inference-capability-routing` | `archived` | `fully_contained` | `merged` | 0/14 | `763ceeaab86f1ede85eb204a02249df0194346ba` | conservar archivada |
| BR-015 | `refactor/inference-engine` | `archived` | `fully_contained` | `merged` | 0/22 | `2eb302f9a5ac07b57192af7d7b6451f672835ca4` | conservar archivada |
| BR-016 | `refactor/sampling-robustness-v3` | `archived` | `fully_contained` | `merged` | 0/15 | `12d5167bdf6dedec748d890b77f3ad683ba22bae` | conservar archivada |

## Supersesión Gate 2

```text
BR-010 placeholder
   └─ superseded por BR-011 candidato materializado
         └─ superseded por BR-012 candidato adversarial integrado vía PR #3
```

La relación se codifica únicamente desde el registro nuevo mediante
`supersedes`; no se duplica con `superseded_by`.
