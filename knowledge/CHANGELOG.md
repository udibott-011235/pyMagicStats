# Historial de la base de conocimiento

## 2026-09-07 — STAGE-DIST-FAMILIES-001 / CP03

- PR #9 integra el cierre canónico de CP02 en
  `main@02a65c80c5da10295d6eeef42e691772d0686ca2`; CP01 y CP02 permanecen
  `COMPLETE` y el stage general permanece `IN_PROGRESS`.
- Se abre CP03 como `IN_PROGRESS` desde ese baseline exacto en `BR-021`.
- Se materializa `DEC-012`, arquitectura `FROZEN` para el core discreto,
  `SupportKind.DISCRETE`, `ParameterizedDiscreteDistribution` y
  `NegativeBinomialFamily().bind(r=..., p=...)`.
- Se registra `EV-009`: reconnaissance `COMPLETE`, nueva superficie discreta de
  familias inexistente antes de CP03, soporte/normalización discreta todavía no
  implementados y baseline heredado de 49/311/360 tests en PASS.
- La fase inicial de arquitectura no cambió producción, tests, legacy discreto,
  fitting, GOF, routing ni familias adicionales.
- La implementación exacta `4f7fa09bc7ab501d21f6d27dada30ade23397588`
  materializa el core discreto y `NegativeBinomialFamily` congelados en
  `DEC-012`.
- Antigravity completó la auditoría independiente pre-merge con
  `ADVERSARIAL_PASS`: 0 BLOCKER, 0 MAJOR, 0 MINOR y 1 INFO.
- `EV-010` registra 360 tests de regresión congelada, 215 tests CP03 y 575
  tests de superficie de distribución en PASS.
- `INFO-001` queda acotado a una limitación de ejecutabilidad extrema del
  backend SciPy correctamente traducida a `FloatingPointError`; no se añaden
  thresholds matemáticos para `r` o `p`.
- La integración permanece `PENDING`, CP03 permanece `IN_PROGRESS` y
  CP04–CP08 permanecen `NOT_STARTED`; no hay autorización de PR o merge.

## 2026-09-06 — STAGE-DIST-FAMILIES-001 / CP02

- Se abre CP02 como `IN_PROGRESS` desde
  `main@ccff392af13d2cb52d1f3888a986ef58be0099e2` en `BR-019`.
- Se materializa `DEC-011`, contrato ejecutable congelado para el core continuo,
  `GammaFamily` y `ExponentialFamily`.
- Se abre `EV-008` para registrar implementación, paridad SciPy, RNG,
  compatibilidad y validaciones del candidato local.
- Se implementan descriptores stateless, parámetros y distribuciones
  parametrizadas inmutables, soporte matemático explícito y las familias Gamma
  y Exponential mediante delegación a SciPy.
- Arquitectura acepta la implementación exacta
  `874c03c70c028d0ca4966331b6fc91ec35613caa`; `EV-008` pasa de
  `under_review` a `accepted`, mientras `BR-019` conserva revisión activa e
  integración `pending` hasta la auditoría adversarial pre-merge.
- La superficie remediada registra 311 tests CP02; las 49 regresiones congeladas
  y los 360 tests combinados de distribución pasan sin modificar clases legacy.
- La suite completa conserva los dos fallos preexistentes de deriva en
  `tests/test_knowledge_base.py`: baseline 287 passed / 3 skipped / 2 failed,
  candidato remediado 598 passed / 3 skipped / 2 failed y
  `FULL_SUITE_DIFFERENTIAL=NO_NEW_FAILURES`.
- El primer governance/audit head `7e009503…` conserva la auditoría
  `ADVERSARIAL_CHANGES_REQUIRED`; la remediación exacta `e9ef63b…` cierra
  `ADV-CP02-001` y endurece `ADV-CP02-002`.
- Antigravity reaudita el governance head exacto `9cf25c…` y emite
  `ADVERSARIAL_PASS` con 0 BLOCKER, 0 MAJOR, 0 MINOR y 1 INFO preexistente fuera
  de alcance (`ADV-CP02-003`). Las superficies validadas registran 49 tests de
  regresión congelada, 311 tests CP02 y 360 tests combinados de distribución.
- `b3d116f2f3e82b74ab6fb5f8337e217104c3bca6` materializa el governance head
  pre-merge final. PR #8 lo integra mediante merge commit
  `aa5723d2cb7dbaf48e6f9059368b9fdaeeb7926c`, con parents `ccff392…` y
  `b3d116f…`.
- CP02 pasa a implementación `COMPLETE`, integración `COMPLETE`, gobernanza
  `CLOSED` y estado general `COMPLETE`; BR-019 queda `archived`/`merged` y se
  abre BR-020 para materializar este cierre post-merge.
- CP03–CP08 permanecen `NOT_STARTED`; no se autoriza fitting, GOF, selección,
  familias discretas ni funcionalidad CP03. El stage general permanece
  `IN_PROGRESS`.

## 2026-09-06 — STAGE-DIST-FAMILIES-001 / CP01

- Se abre `STAGE-DIST-FAMILIES-001` desde
  `main@402e4601df460811779b3238c2526ac12f463a67` y se registra `BR-017`.
- Se materializan `DEC-009` (estado/checkpoints) y `DEC-010` (arquitectura y
  contratos congelados) para el Distribution Family Framework.
- Se añade `EV-007` con el inventario de compatibilidad de la superficie actual
  y la evidencia de validación de CP01.
- CP01 modifica sólo `knowledge/**`; no implementa familias, fitting, GOF,
  selección automática ni cambios de producción.
- La revisión arquitectónica de `72ecdba…` conserva ese primer candidato y
  aclara el contrato de sampling/RNG, ownership entre `FitResult` y
  `FittedDistribution`, semántica de Negative Binomial para `r` no entero,
  compatibilidad futura mediante `BinomialFamily`/`PoissonFamily`, cronología
  del push de revisión y distinción GOF entre null simple y compuesto.
- Arquitectura acepta CP01 en el candidato exacto
  `c63b48eafc439de8857207fd21c1b593e38a3187`; DEC-009, DEC-010 y EV-007 pasan
  a `accepted`, mientras la integración de BR-017 permanece `pending` y
  CP02–CP08 permanecen `NOT_STARTED`.
- Antigravity emite `ADVERSARIAL_PASS` para PR #6 en el integration head
  `3f9acd5a51ce38ae62b9800d50efb0949c6531f0`, con cero `BLOCKER`, `MAJOR` y
  `MINOR`.
- PR #6 se integra mediante el merge commit
  `46f827dd107aa9e6f940f0de085fbb91075ff049`; CP01 y su integración pasan a
  `COMPLETE`, BR-017 queda `archived`/`merged` y CP02–CP08 permanecen
  `NOT_STARTED`.
- Se abre `BR-018` desde el nuevo `main@46f827d…` exclusivamente para
  materializar el cierre de gobernanza post-merge de CP01.

## 2026-08-30 — KB v1.3

- Se registra la integración controlada de PR #1 (`docs/project-knowledge-base`) y PR #3 (`fix/gate2-adversarial-remediation`) en `main` (`f1725ebdfebcb667c053420e4cb4c1e35048f9e0`).
- Se añade `EV-005` registrando la evidencia inmutable de la integración de Gate 2 (árbol `238222f`, parents `e8422a7` y `9a87c5d`, suite 289 passed / 3 skipped, límites `TD-GOF-SUPPORT-001` y `FINDING-ADV-NUM-004` abiertos, no demostración de identidad distributiva por GOF, bypass automático observado al crear la rama pero no durante el merge, y preservación de ramas).
- Se actualizan las 16 ramas en `knowledge/registry.json` y `knowledge/versioning/branches.md` según las decisiones autorizadas por Product Owner y Arquitectura:
  - `BR-001` (`main`): canonical en `f1725eb`.
  - `BR-003` (`docs/project-knowledge-base`): archivada, fully_contained, merged vía PR #1.
  - `BR-010` (`fix/gate2-major-remediation`): superseded, fully_contained, preservada como placeholder histórico.
  - `BR-011` (`fix/gate2-distribution-gof-remediation`): superseded, fully_contained, merged indirectamente como ancestro de BR-012 vía PR #3.
  - `BR-012` (`fix/gate2-adversarial-remediation`): archivada, fully_contained, merged vía PR #3 (HEAD auditado `9a87c5d`).
  - Actualización de métricas reproducibles (ahead/behind, merge-base, relación) para todas las demás ramas respecto del nuevo baseline de `main`.

## 2026-08-30 — KB v1.2

- Se amplía el schema a `1.1.0` y se añade el tipo canónico `branch`.
- Se registran las 16 ramas remotas observadas, sus SHAs, relaciones con
  `main`, integración, supersesión y siguientes acciones decididas.
- Se incorporan EV-003, EV-004 y DEC-006 para el inventario forense, el
  candidato adversarial Gate 2 y la autoridad de lifecycle.
- Se añade `versioning/` como proyección humana del registro canónico.
- Se exige que toda auditoría independiente se origine en el remoto autorizado
  o en un artefacto/bundle cuyo SHA haya sido validado explícitamente.

## 2026-08-29 — KB v1.1

- Se corrige la autoridad de los roles: ChatGPT diseña, Cortex implementa,
  Antigravity audita y el Project Owner decide.
- Se añade `SYSTEM_PROMPTS.md` como fuente única del núcleo común y los system
  prompts de cada agente.
- Se separan explícitamente diseño, implementación, publicación, PR y merge.
- Se canoniza la prohibición de modificar `main` o usar bypass administrativo.
- Se añaden veredictos, severidades, condiciones de detención y handoff con SHA.
- Se reemplazan los espacios ambiguos de arquitectura/implementación e
  investigación/reproducción por espacios alineados con los roles vigentes.

## 2026-08-29 — KB v1

- Se crea el portal y registro canónico legible por máquinas.
- Se formalizan gobernanza, estados, revisión cruzada y autoridad por rol.
- Se crean espacios para arquitectura/implementación, QA adversarial e
  investigación/reproducción.
- Se indexan la calibración `mean-v2.1-2026-08`, sus artefactos, los datasets
  existentes, decisiones aceptadas y deuda numérica conocida.
- Se añade validación automática y plantilla de PR.
