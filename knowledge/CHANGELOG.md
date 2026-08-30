# Historial de la base de conocimiento

## 2026-08-30 — Agent orchestration policy

- Se adopta `knowledge/decisions/agent-orchestration-policy.md` como política operativa de distribución de carga entre agentes.
- ChatGPT queda enfocado en arquitectura matemática/software, arbitraje técnico, metodología, Gates e interpretación de evidencia.
- Codex reemplaza a Cortex como asignación vigente del rol `implementation-engineering` y queda enfocado en implementación, refactors, fixes y tests ligados al cambio.
- Antigravity amplía su rol desde QA adversarial a `QA + Repo Ops + Validation Engineering`, absorbiendo preflight Git, SHA/branch/worktree checks, fresh-clone validation, suites rutinarias, regresión, reproducibilidad, scope audit y evidencia de Gates, además de adversarial estadístico/numérico/software.
- Los hallazgos mecánicos o inequívocamente cubiertos por un contrato pueden circular `Antigravity -> Codex -> Antigravity` sin consumir arbitraje de ChatGPT.
- Los hallazgos que cambien estimando, teoría, API, método, política, threshold, fallback, garantías o aceptación escalan a ChatGPT.
- Se actualizan `GOVERNANCE.md`, `AGENT_PROTOCOL.md`, `SYSTEM_PROMPTS.md` y los tres espacios de rol.

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
