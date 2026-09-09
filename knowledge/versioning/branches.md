# Branch lifecycle

> This document is a human-readable projection. knowledge/registry.json is canonical.

Observación inicial materializada: `2026-08-30`; apertura de `BR-017`, cierre
post-merge mediante `BR-018`, apertura CP02 mediante `BR-019` y cierre
post-merge CP02 mediante `BR-020` materializados: `2026-09-06`; apertura CP03
y evidencia adversarial pre-merge mediante `BR-021`, y cierre post-merge CP03
mediante `BR-022`: `2026-09-07`; apertura CP04 Wave 1 fitting mediante
`BR-023`: `2026-09-08`. Consulte `EV-003`
para la evidencia Git reproducible inicial, `EV-005` para la integración de
Gate 2, `EV-007` para la apertura de CP01 del Distribution Family Framework y
`DEC-006` para la autoridad de lifecycle.

| ID | Rama | Status | Relación | Integración | Ahead/behind | HEAD observado | Siguiente acción resumida |
|---|---|---|---|---|---:|---|---|
| BR-001 | `main` | `accepted` | `canonical` | `not_applicable` | 0/0 | `b3f35d4d7b221c457e2e730bfba2b104e1d07144` | CP03 canónicamente cerrado; no modificar directamente; CP04 arquitectura congelada |
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
| BR-017 | `feature/distribution-family-framework-cp01` | `archived` | `fully_contained` | `merged` | 0/1 | `3f9acd5a51ce38ae62b9800d50efb0949c6531f0` | conservar la rama remota; no borrar |
| BR-018 | `docs/distribution-family-framework-cp01-post-merge` | `under_review` | `same_head` al abrir | `pending` | 0/0 al abrir | `46f827dd107aa9e6f940f0de085fbb91075ff049` | esperar revisión de Arquitectura; sin PR, merge ni CP02/CP03 |
| BR-019 | `feature/distribution-family-framework-cp02-continuous-core` | `archived` | `fully_contained` | `merged` | 0/1 | `b3d116f2f3e82b74ab6fb5f8337e217104c3bca6` | integrada vía PR #8; preservar rama remota |
| BR-020 | `docs/distribution-family-framework-cp02-post-merge` | `archived` | `fully_contained` | `merged` | 0/1 | `37bc8de97185794e9ef14b33d4ed00d3da1a660e` | integrada vía PR #9; preservar rama remota |
| BR-021 | `feature/distribution-family-framework-cp03-discrete-core` | `archived` | `fully_contained` | `merged` | 0/1 | `a1e4d61f0026f8407506d039788bb2df2eafa680` | integrada vía PR #10; preservar rama remota |
| BR-022 | `docs/distribution-family-framework-cp03-post-merge` | `archived` | `fully_contained` | `merged` | 0/1 | `7e38c1c62f69771282398ffd3fac118f866c5d69` | integrada vía PR #11; preservar rama remota |
| BR-023 | `feature/distribution-family-framework-cp04-wave1-fitting` | `under_review` | `same_head` al abrir | `pending` | 0/0 al abrir | `b3f35d4d7b221c457e2e730bfba2b104e1d07144` | revisar arquitectura CP04; implementación y publicación no autorizadas |

### Cronología de BR-017

- La rama se abrió localmente en el baseline registrado y produjo
  `72ecdba1b60d9efb53dfe612cf7ee4beeb3e76e5` bajo una regla sin push.
- Arquitectura autorizó posteriormente el push de ese SHA exacto sólo para
  revisión independiente de su contenido versionado.
- El push no autorizó PR, merge, CP02, CP03 ni implementación.
- La revisión preservó `72ecdba…` y autorizó un único commit documental de
  seguimiento y su push para reauditación arquitectónica.
- Arquitectura aceptó CP01 en el candidato exacto
  `c63b48eafc439de8857207fd21c1b593e38a3187`.
- El integration head `3f9acd5a51ce38ae62b9800d50efb0949c6531f0`
  recibió `ADVERSARIAL_PASS` con cero blockers, majors y minors.
- PR #6 integró ese head mediante el merge commit
  `46f827dd107aa9e6f940f0de085fbb91075ff049`; BR-017 está completamente
  contenido en `main`, queda archivado y su ref remota se preserva.

### Apertura de BR-018

- `docs/distribution-family-framework-cp01-post-merge` se abrió desde el
  `main@46f827dd107aa9e6f940f0de085fbb91075ff049` exacto, con cero commits
  únicos al abrir.
- `head_sha_at_decision` conserva ese snapshot de apertura; no intenta
  autorreferenciar el commit vigente de la rama.
- El candidato inicial `8d5049ebc32ea9efbaec1e6550681810a229ff66` se
  materializó y publicó; Arquitectura solicitó correcciones menores de
  consistencia de gobernanza.
- BR-018 espera revisión de Arquitectura del commit de seguimiento. No hay
  autorización para PR, merge, CP02, CP03 ni implementación de familias.

### Apertura de BR-019

- `feature/distribution-family-framework-cp02-continuous-core` se abrió desde
  `main@ccff392af13d2cb52d1f3888a986ef58be0099e2` exacto, con árbol limpio y
  cero commits únicos al abrir.
- `head_sha_at_decision` conserva el snapshot de apertura. El trabajo autorizado
  materializa `DEC-011` e implementa únicamente el core continuo,
  `GammaFamily` y `ExponentialFamily`.
- Arquitectura aceptó inicialmente la implementación exacta
  `874c03c70c028d0ca4966331b6fc91ec35613caa`.
- La primera auditoría adversarial pre-merge devolvió
  `ADVERSARIAL_CHANGES_REQUIRED`: `FINDING-ADV-CP02-001` (`MINOR`),
  `FINDING-ADV-CP02-002` (`INFO`) y `FINDING-ADV-CP02-003` (`INFO`).
- El commit `e9ef63b802a8cb08ea38b32e87b206432c08b120` corrigió
  `ADV-CP02-001` y endureció `ADV-CP02-002`; Arquitectura acepta ese SHA exacto
  como candidato vigente. `874c03c…` permanece como evidencia histórica.
- `7e009503a253308a4a8294f280461953247b1c2f` conserva el primer governance/audit
  head y el resultado `ADVERSARIAL_CHANGES_REQUIRED`.
- Antigravity reaudita el governance head exacto
  `9cf25c157a4f4114f41ae74d4e04e009392414e3` y emite `ADVERSARIAL_PASS`: cero
  blockers, majors y minors; el único INFO es `ADV-CP02-003`, deuda
  preexistente fuera de alcance.
- `b3d116f2f3e82b74ab6fb5f8337e217104c3bca6` materializa el resultado final
  como governance head pre-merge.
- PR #8 integra ese head mediante `aa5723d2cb7dbaf48e6f9059368b9fdaeeb7926c`,
  con parents `ccff392af13d2cb52d1f3888a986ef58be0099e2` y
  `b3d116f2f3e82b74ab6fb5f8337e217104c3bca6`.
- BR-019 queda `archived`, `fully_contained` y `merged`; su rama remota se
  preserva. CP03 permanece `NOT_STARTED`.

### Apertura de BR-020

- `docs/distribution-family-framework-cp02-post-merge` se abre exactamente desde
  `main@aa5723d2cb7dbaf48e6f9059368b9fdaeeb7926c`, con árbol limpio y cero
  commits únicos al abrir.
- `head_sha_at_decision` conserva ese snapshot de apertura y no intenta
  autorreferenciar el commit de gobernanza posterior.
- BR-020 materializa únicamente el cierre post-merge de CP02 para revisión de
  Arquitectura.
- Arquitectura aceptó el candidato `37bc8de97185794e9ef14b33d4ed00d3da1a660e`;
  PR #9 lo integró mediante `02a65c80c5da10295d6eeef42e691772d0686ca2`,
  con parents `aa5723d2cb7dbaf48e6f9059368b9fdaeeb7926c` y
  `37bc8de97185794e9ef14b33d4ed00d3da1a660e`.
- BR-020 queda `archived`, `fully_contained` y `merged`; su rama remota se
  preserva.

### Apertura de BR-021

- `feature/distribution-family-framework-cp03-discrete-core` se abre desde
  `main@02a65c80c5da10295d6eeef42e691772d0686ca2` exacto, con árbol limpio y
  cero commits únicos al abrir.
- En la apertura, el snapshot registrado fue
  `main@02a65c80c5da10295d6eeef42e691772d0686ca2`. Después de la implementación
  autorizada por separado y de su aceptación por Arquitectura/Owner,
  `head_sha_at_decision` se avanzó explícitamente al SHA de implementación
  auditado `4f7fa09bc7ab501d21f6d27dada30ade23397588`.
- La relación registrada es `relation_to_main=contains_main`, con
  `ahead/behind=2/0`; los commits únicos son
  `da1b9d51e4bfeb7f262db182cf10993f59b1162b` y
  `4f7fa09bc7ab501d21f6d27dada30ade23397588`.
- El commit documental que materializa esta corrección no se autorreferencia ni
  sustituye el `head_sha_at_decision` aceptado. `DEC-012` congela la
  arquitectura del core discreto y `EV-009` registra reconnaissance y baseline.
- `da1b9d51e4bfeb7f262db182cf10993f59b1162b` materializó y publicó la
  arquitectura congelada; `4f7fa09bc7ab501d21f6d27dada30ade23397588`
  implementó y publicó el candidato discreto para revisión.
- Antigravity completó la auditoría del SHA de implementación `4f7fa09…` con
  `ADVERSARIAL_PASS`: 0 BLOCKER, 0 MAJOR, 0 MINOR y 1 INFO (`INFO-001`).
- `INFO-001` registra solamente una limitación acotada de ejecutabilidad del
  backend SciPy, traducida correctamente a `FloatingPointError` sin añadir
  thresholds matemáticos.
- `EV-010` materializa 360 tests de regresión congelada, 215 tests CP03 y 575
  tests de distribución.
- `a1e4d61f0026f8407506d039788bb2df2eafa680` materializa el governance head
  pre-merge final.
- Una auditoría distinta del PR head exacto `a1e4d61…`, ejecutada desde clon
  independiente fresco, devuelve `ADVERSARIAL_PASS`: 0 BLOCKER, 0 MAJOR,
  0 MINOR y 2 INFO. `INFO-001` preserva la limitación backend acotada;
  `INFO-002` registra los dos fallos heredados de Knowledge Base. Registro y
  superficie de 575 tests pasan, el diferencial es `NO_NEW_FAILURES` y GitHub
  reporta 0 checks/workflows, 0 reviews y 0 threads sin resolver.
- PR #10 integra ese head mediante
  `28b57a2eaab0706c5b2e2dcdf6a03e5a30a0b649`, con parents exactos
  `02a65c80c5da10295d6eeef42e691772d0686ca2` y
  `a1e4d61f0026f8407506d039788bb2df2eafa680`.
- El tree del merge y el tree del head integrado coinciden exactamente en
  `13002c3ca716a3b2a2aa8914bce078afd623d9ff`.
- BR-021 queda `archived`, `fully_contained` y `merged`; su rama remota se
  preserva. CP03 queda `COMPLETE`; en ese hito CP04 permanecía `NOT_STARTED`.

### Apertura de BR-022

- `docs/distribution-family-framework-cp03-post-merge` se abre exactamente
  desde `main@28b57a2eaab0706c5b2e2dcdf6a03e5a30a0b649`, con árbol limpio y cero
  commits únicos al abrir.
- `head_sha_at_decision` conserva ese snapshot de apertura y no intenta
  autorreferenciar el commit documental posterior.
- BR-022 materializó únicamente el cierre post-merge de CP03. Arquitectura y
  auditoría aceptaron el head exacto
  `7e38c1c62f69771282398ffd3fac118f866c5d69`.
- PR #11 integró ese head mediante
  `b3f35d4d7b221c457e2e730bfba2b104e1d07144`, con parents exactos
  `28b57a2eaab0706c5b2e2dcdf6a03e5a30a0b649` y
  `7e38c1c62f69771282398ffd3fac118f866c5d69`.
- Los trees del merge y del head integrado coinciden exactamente en
  `2e363bd357a5ec59c17690bd9e0e3bbb26061d33`. BR-022 queda `archived`,
  `fully_contained` y `merged`; su rama remota se preserva.

### Apertura de BR-023

- `feature/distribution-family-framework-cp04-wave1-fitting` se abre desde el
  snapshot exacto `main@b3f35d4d7b221c457e2e730bfba2b104e1d07144`, con
  árbol limpio y cero commits únicos al abrir.
- `head_sha_at_decision` registra únicamente ese snapshot de apertura; no
  autorreferencia el futuro commit documental.
- BR-023 materializa `DEC-013` y `EV-011` para revisión arquitectónica. La
  implementación, el push, PR, merge y CP05–CP08 requieren autorizaciones
  separadas.

## Supersesión Gate 2

```text
BR-010 placeholder
   └─ superseded por BR-011 candidato materializado
         └─ superseded por BR-012 candidato adversarial integrado vía PR #3
```

La relación se codifica únicamente desde el registro nuevo mediante
`supersedes`; no se duplica con `superseded_by`.
