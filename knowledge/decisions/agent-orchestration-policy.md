# Política de orquestación de agentes y distribución de carga

**Estado:** accepted por decisión del Project Owner el 2026-08-30.

## Objetivo

Maximizar la capacidad total del equipo de agentes de pyMagicStats asignando cada tarea al rol con mayor ventaja marginal. La política no busca minimizar tokens como objetivo primario: busca preservar capacidad arquitectónica, reducir duplicación, mantener independencia de validación y mejorar continuidad.

## Regla de routing

Antes de escalar una tarea, clasificarla:

- Si requiere decidir qué comportamiento, contrato, método o política debe tener el sistema: **ChatGPT**.
- Si requiere implementar algo ya decidido: **Codex**.
- Si requiere comprobar, reproducir, auditar, operar Git o intentar romper un candidato: **Antigravity**.
- Si cambia prioridad, riesgo aceptado, alcance, PR o merge: **Project Owner**.

La autoridad pertenece al rol, no al modelo.

## Roles

### ChatGPT — Arquitectura matemática y de software / árbitro técnico

Responsable de arquitectura, teoría estadística, contratos, API, selección metodológica, criterios de aceptación, diseño de Gates, interpretación de calibraciones, resolución de contradicciones y recomendación final al Project Owner.

Debe evitar consumir capacidad en tareas mecánicas que Antigravity pueda certificar: `git status`, verificación rutinaria de ramas/SHA, fresh clones, suites estándar, linting, comprobación de archivos, scope checks y evidencia operacional repetitiva.

ChatGPT no implementa producción como función normal del rol ni autocertifica evidencia.

### Codex — Ingeniería principal de implementación

Responsable de convertir contratos aprobados en código, tests ligados al cambio, documentación técnica, refactors y correcciones. Recibe especificaciones cerradas con baseline, alcance, invariantes y criterios de aceptación.

Codex no redefine teoría ni es el auditor independiente final de su propio candidato. Su objetivo es producir un candidato reproducible para Antigravity.

### Antigravity — QA, Repo Ops y Validation Engineering

Responsable de una superficie deliberadamente amplia:

- preflight Git, baseline, SHA, ancestry, branch y workspace;
- fresh-clone validation;
- suites rutinarias, regresión, smoke tests, lint/type/import checks cuando apliquen;
- reproducibilidad, determinismo y validación de entornos/backends;
- scope audit, diff audit y consistencia documental;
- testing adversarial estadístico, numérico y de software;
- evidencia de Gate/release candidate;
- clasificación de hallazgos.

Antigravity distingue al menos: `REGRESSION_FAILURE`, `NUMERICAL_RISK`, `API_CONTRACT_FAILURE`, `STATISTICAL_VALIDITY_QUESTION`, `PERFORMANCE_ISSUE`, `DOCUMENTATION_MISMATCH` y `GOVERNANCE_ISSUE`.

La primera auditoría independiente de un candidato de producción sigue siendo de solo lectura. Antigravity no aprueba su propio fix.

## Flujo estándar

1. Project Owner define objetivo y riesgo.
2. ChatGPT define contrato y aceptación.
3. Antigravity realiza preflight y certifica baseline/entorno.
4. Codex implementa.
5. Antigravity valida, reproduce, audita y ataca el candidato.
6. Hallazgos mecánicos o de implementación inequívoca circulan `Antigravity -> Codex -> Antigravity` sin requerir a ChatGPT en cada iteración.
7. Hallazgos que cambien teoría, estimando, API, política, threshold, fallback, garantía o criterio de aceptación escalan a ChatGPT.
8. ChatGPT interpreta los hallazgos materiales y recomienda transición.
9. Sólo el Project Owner autoriza PR y merge.

## Política de remediación

### No requiere arbitraje arquitectónico

Typo, import, fixture, formatting, metadata, documentación trivial, bug inequívoco dentro del contrato, edge case, mutation accidental, scope drift o fallo de regresión claramente cubierto por la especificación.

Ruta: `Antigravity -> Codex -> Antigravity`.

### Requiere ChatGPT

Cambio de estimando, método, teoría, API pública, política de selección, robustez, threshold, fallback, semántica de estados, garantías, calibración o contradicción metodológica.

Ruta: `Antigravity -> ChatGPT -> nueva/ajustada especificación -> Codex`.

## Evidencia compacta

Los agentes no transfieren logs completos salvo necesidad diagnóstica. El handoff debe priorizar:

- acción;
- resultado;
- evidencia mínima;
- conclusión;
- SHA exacto.

Ejemplo válido: `pytest -q -> 255 passed in 6.32s, exit_code=0`.

Ante fallo se agrega únicamente el test, traceback relevante, input, expected y actual necesarios para reproducirlo.

## Fresh clones

Gate validation, release candidates, merge candidates, instalación y reproducibilidad final se validan preferentemente desde un fresh clone aislado. Antigravity es el responsable preferido de esta tarea.

## Anti-patrones

- Triple revisión completa del mismo trabajo por ChatGPT, Codex y Antigravity.
- Usar ChatGPT como terminal para controles rutinarios.
- Usar Codex como único auditor de su propia implementación.
- Enviar todo hallazgo a ChatGPT aunque no cambie el contrato.
- Mantener estado crítico únicamente en memoria conversacional.

## Continuidad

Todo trabajo material termina cada etapa con rama, SHA, estado, evidencia mínima y siguiente rol. GitHub y `knowledge/` son la memoria compartida; ningún agente debe ser un single point of failure por agotamiento de contexto o capacidad.

## Distribución orientativa de trabajo

No son cuotas de tokens ni límites rígidos:

- ChatGPT: 60–70% arquitectura/metodología, 15–20% interpretación, 10–15% revisión, mínimo trabajo mecánico.
- Codex: 65–80% implementación, 15–25% tests asociados, resto investigación localizada necesaria para implementar.
- Antigravity: 25–35% testing, 20–30% adversarial, 20–25% Repo Ops/Git, 10–20% scope/documentación, resto validaciones genéricas.

La asignación concreta siempre sigue la ventaja marginal del rol y no una cuota.