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

## Estado observado tras CP-01

CP-01 fue completado mediante auditoría read-only sobre el baseline exacto. El evidence record es `knowledge/evidence/proportion-ci-cp01-census.md`. El censo confirmó Wilson como default, Wald como ruta explícita, ausencia de calibración específica de cobertura y varios gaps de contrato. También detectó un defecto MAJOR independiente: el routing genérico puede seleccionar `one_sample_t` para `Estimand.PROPORTION`; este comportamiento debe quedar fail-closed antes de cualquier routing automático de proporciones.

CP-02 dispone ya de una especificación arquitectónica propuesta en `knowledge/decisions/proportion-ci-cp02-spec.md` y permanece `under_review` hasta aprobación explícita del Product Owner.

## Checkpoints

| Checkpoint | Objetivo | Estado | Pendientes individuales | Criterio de salida |
|---|---|---|---|---|
| CP-01 | Censo y reconstrucción del contrato actual | `complete` | Ninguno. Evidencia registrada en `knowledge/evidence/proportion-ci-cp01-census.md` | Informe reproducible read-only completado sobre baseline exacto |
| CP-02 | Especificación estadística | `under_review` | Aprobación del Product Owner de estimando/diseño; Wilson candidato principal; Clopper–Pearson candidato exacto/conservador; Wald legacy explícito; Jeffreys sólo con semántica bayesiana separada; boundary policy; fail-closed para `Estimand.PROPORTION`; alcance bilateral mínimo | Spec arquitectónica aprobada por Product Owner |
| CP-03 | Contrato de API y compatibilidad | `pending` | Diseñar API agregada `(successes,trials)`; transición de `incidences` fraccionario; export público; esquema de resultado; deprecaciones; relación con BootstrapCI; contrato fail-closed del router | Contrato público cerrado y tests de compatibilidad especificados |
| CP-04 | Diseño de calibración | `pending` | Definir grid de `n`, `p`, alpha/confidence; cobertura exacta por enumeración binomial cuando sea posible; ancho, conservadurismo, monotonicidad, simetrías, boundary behavior y referencias externas | Protocolo pre-registrado antes de mirar resultados finales/holdout |
| CP-05 | Implementación candidata | `pending` | Implementar sólo métodos aprobados y tests unitarios/metamórficos; corregir routing incompatible de proporción; no activar routing automático | SHA candidato reproducible en rama aislada |
| CP-06 | Calibración y evidencia | `pending` | Ejecutar harness aprobado; producir summary/metadata/evidence; cuantificar cobertura y límites por método | Evidencia suficiente para clasificar cada método como validado, limitado o no calibrado |
| CP-07 | Auditoría adversarial | `pending` | Antigravity audita el SHA exacto; boundaries, precisión numérica, invariancias, regresión y claims estadísticos | Cero hallazgos bloqueantes o nueva iteración con SHA nuevo |
| CP-08 | Decisión de integración | `pending` | ChatGPT interpreta evidencia; Product Owner decide alcance final, PR y merge | Sólo Product Owner autoriza PR/merge |

## Arquitectura propuesta en CP-02 — pendiente de aprobación

- Estimando: una única proporción poblacional `p=P(Y=1)` bajo diseño Bernoulli/binomial ordinario.
- Mantener Wilson como candidato principal y default candidato.
- Incorporar Clopper–Pearson como candidato frecuentista exacto/conservador.
- Mantener Wald únicamente como método legacy explícito; nunca default ni fallback automático.
- Admitir Jeffreys sólo como candidato comparador y, si llega a exponerse, con semántica explícita de credible interval bayesiano; no etiquetarlo como CI frecuentista.
- No autorizar por ahora Agresti–Coull, Wilson-CC, mid-P ni otras variantes para producción.
- Preservar inicialmente el alcance bilateral; one-sided requerirá criterio de calibración propio.
- Tratar `x=0` y `x=n` como realizaciones válidas, no errores.
- No utilizar `successes >= 10 and failures >= 10` como selector universal ni garantía.
- Diseñar una entrada agregada explícita `(successes, trials)` con enteros; no reinterpretar éxitos fraccionarios como binomiales.
- Survey weights, effective sample size, clustering y finite-population correction quedan fuera de alcance.
- `MethodSelector` debe fallar cerrado para `Estimand.PROPORTION` mientras no exista una capability registrada y calibrada; nunca transferir `one_sample_t` desde el estimando media.
- `BootstrapCI(stat="proportion")` permanece como superficie separada hasta contrato y calibración específicos.

## Invariantes del stage

1. Ningún checkpoint posterior puede darse por completado por el mero hecho de que los tests de software estén verdes.
2. `fail_to_reject`, cobertura nominal aproximada o una referencia externa no equivalen por sí solos a autorización de routing automático.
3. La calibración debe separar desempeño estadístico de fallos numéricos.
4. No se transferirá evidencia de media, ANOVA, EL o GOF a intervalos de proporción.
5. Cada SHA nuevo requiere su propia evidencia y auditoría; un PASS no se transfiere.
6. `main` permanece protegido y no se modifica directamente durante el stage.

## Trabajo posterior ya acordado

Después de cerrar este stage, el siguiente bloque del roadmap es el Gate de calibración de diagnósticos de distribución/GOF (Shapiro, D’Agostino, Anderson, Q-Q, KS/Lilliefors y huecos de muestra pequeña), seguido —sólo con evidencia suficiente— por el motor automático de decisión de método de bondad y ajuste.
