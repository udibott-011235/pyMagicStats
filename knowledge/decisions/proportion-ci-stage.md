# STAGE-PROP-CI-001 — Revisión y ampliación de intervalos de proporción

**Estado general:** `in_progress`  
**Fecha de apertura:** 2026-08-30  
**Baseline canónico:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`  
**Rama de registro:** `docs/proportion-ci-stage`  
**Owner:** Product Owner  
**Arquitectura estadística:** ChatGPT  
**Implementación:** Cortex/Codex, sólo después de aprobación explícita  
**QA adversarial:** Antigravity, sobre SHA candidato exacto

## Objetivo

Revisar y ampliar el contrato de intervalos para una proporción poblacional Bernoulli/binomial sin introducir selección automática no calibrada. El stage debe preservar compatibilidad razonable con `PopulationProportionCI` y separar explícitamente garantía estadística, implementación numérica, calibración y routing.

## Estado inicial observado

En `main`, `PopulationProportionCI` soporta actualmente `wilson` y `wald`; Wilson es el default. El contrato actual acepta datos binarios, predicado de incidencia o un conteo de incidencias. La lógica mantiene además un diagnóstico legacy de aproximación normal basado en éxitos y fracasos >= 10. No existe evidencia canónica suficiente para ampliar esto a un selector automático de métodos.

## Checkpoints

| Checkpoint | Objetivo | Estado | Pendientes individuales | Criterio de salida |
|---|---|---|---|---|
| CP-01 | Censo y reconstrucción del contrato actual | `in_progress` | Inventariar API pública/interna, tests, docs, dependencias, invariantes, usos, compatibilidad y deuda; confirmar tratamiento de `incidences` fraccionario; detectar reglas legacy y cualquier routing implícito | Informe reproducible, sólo lectura, fijado al baseline exacto, sin propuesta de implementación |
| CP-02 | Especificación estadística | `pending` | Fijar estimando, diseño Bernoulli/binomial, supuestos; decidir métodos candidatos y semántica de garantías; resolver casos `x=0`, `x=n`, one/two-sided, aggregate vs raw data y parámetros inválidos | Spec arquitectónica aprobada por Product Owner |
| CP-03 | Contrato de API y compatibilidad | `pending` | Diseñar API nueva/extendida sin romper innecesariamente `PopulationProportionCI`; definir resultados, estados, warnings, metadata y deprecaciones | Contrato público cerrado y tests de compatibilidad especificados |
| CP-04 | Diseño de calibración | `pending` | Definir grid de `n`, `p`, alpha/confidence; cobertura exacta por enumeración binomial cuando sea posible; ancho, conservadurismo, monotonicidad, simetrías, boundary behavior y referencias externas | Protocolo pre-registrado antes de mirar resultados finales/holdout |
| CP-05 | Implementación candidata | `pending` | Implementar sólo métodos aprobados y tests unitarios/metamórficos; no activar routing automático | SHA candidato reproducible en rama aislada |
| CP-06 | Calibración y evidencia | `pending` | Ejecutar harness aprobado; producir summary/metadata/evidence; cuantificar cobertura y límites por método | Evidencia suficiente para clasificar cada método como validado, limitado o no calibrado |
| CP-07 | Auditoría adversarial | `pending` | Antigravity audita el SHA exacto; boundaries, precisión numérica, invariancias, regresión y claims estadísticos | Cero hallazgos bloqueantes o nueva iteración con SHA nuevo |
| CP-08 | Decisión de integración | `pending` | ChatGPT interpreta evidencia; Product Owner decide alcance final, PR y merge | Sólo Product Owner autoriza PR/merge |

## Hipótesis de arquitectura a revisar, no aprobadas todavía

- Mantener Wilson como candidato general-purpose y Wald sólo como método explícito/legacy, nunca como fallback automático por `n`.
- Evaluar Clopper–Pearson como intervalo exacto/conservador.
- Evaluar Jeffreys sólo si se conserva una separación semántica correcta entre intervalo bayesiano creíble y CI frecuentista; no mezclar etiquetas.
- Agresti–Coull, continuidad corregida, mid-P u otros métodos no entran automáticamente: requieren justificación de valor incremental.
- No usar `successes >= 10 and failures >= 10` como selector universal de método.
- Conteos agregados deben ser enteros; aceptar éxitos fraccionarios requiere un estimando/modelo distinto y no debe heredarse silenciosamente.
- Survey weights, effective sample size y finite-population correction quedan fuera de alcance hasta diseño explícito.

## Invariantes del stage

1. Ningún checkpoint posterior puede darse por completado por el mero hecho de que los tests de software estén verdes.
2. `fail_to_reject`, cobertura nominal aproximada o una referencia externa no equivalen por sí solos a autorización de routing automático.
3. La calibración debe separar desempeño estadístico de fallos numéricos.
4. No se transferirá evidencia de media, ANOVA, EL o GOF a intervalos de proporción.
5. Cada SHA nuevo requiere su propia evidencia y auditoría; un PASS no se transfiere.
6. `main` permanece protegido y no se modifica directamente durante el stage.

## Trabajo posterior ya acordado

Después de cerrar este stage, el siguiente bloque del roadmap es el Gate de calibración de diagnósticos de distribución/GOF (Shapiro, D’Agostino, Anderson, Q-Q, KS/Lilliefors y huecos de muestra pequeña), seguido —sólo con evidencia suficiente— por el motor automático de decisión de método de bondad y ajuste.
