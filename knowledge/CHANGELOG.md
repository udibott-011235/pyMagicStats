# Historial de la base de conocimiento

## 2026-08-31 — Manual UAT Checkpoint 1

- Se crea `DEC-007` — `MANUAL UAT CHECKPOINT 1 — CURRENT STATISTICAL CORE`.
- El checkpoint se define como una pausa de madurez del alcance estadístico actual, no como cierre del toolbox ni como declaración de completitud de pyMagicStats.
- Se fijan cuatro bloqueantes previos al UAT manual:
  - cierre completo de `STAGE-PROP-CI-001` (CP-06/07/08);
  - cierre estadístico propio de ANOVA one-way, sin transferir calibración de medias;
  - accuracy closure/censo de la superficie de distribuciones y GOF realmente expuesta en el baseline;
  - congelación de un inventario explícito de métodos admitidos al UAT.
- El Manual UAT 1 se ejecutará con invocación manual de métodos, sin `MethodSelector` ni decision engine, sobre DataFrames reales/sucios y fixtures controlados para separar matemática, implementación/API y manejo de datos.
- Un PASS sólo habilita uso operativo manual de los módulos incluidos y dentro de límites documentados.
- Se registran como deuda posterior deliberadamente fuera del hito: expansión de distribuciones, transformaciones, métodos no paramétricos, regresiones, DOE y orquestador/decision engine.
- `DEBT-001 / TD-NUM-001` permanece no bloqueante para el dominio retail/BI salvo que un caso del baseline UAT dependa de escalas subnormales.
- `DEC-007` queda enlazado desde la ruta de lectura obligatoria de `knowledge/README.md` y desde `knowledge/decisions/README.md` para visibilidad transversal de todos los agentes.

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
