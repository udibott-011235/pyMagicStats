# STAGE-PROP-CI-001 — Revisión y ampliación de intervalos de proporción

**Estado general:** `in_progress`  
**Fecha de apertura:** 2026-08-30  
**Baseline canónico:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`  
**Rama de registro:** `docs/proportion-ci-stage`  
**Owner:** Product Owner  
**Arquitectura estadística:** ChatGPT  
**Implementación:** Cortex/Codex, sólo después de aprobación explícita  
**QA adversarial:** Antigravity, sobre SHA candidato exacto

## Estado tras diseño de CP-03

CP-01 fue completado mediante auditoría read-only sobre el baseline exacto y quedó registrado en `knowledge/evidence/proportion-ci-cp01-census.md`.

CP-02 fue aprobado explícitamente por el Product Owner el 2026-08-30. La especificación aceptada está en `knowledge/decisions/proportion-ci-cp02-spec.md`.

CP-03 dispone ahora de un contrato de API y compatibilidad completo en `knowledge/decisions/proportion-ci-cp03-api-contract.md`. El documento fija firmas públicas, `from_counts(successes,trials)`, compatibilidad legacy, deprecaciones, metadata, métodos públicos, boundaries, separación de bootstrap, fail-closed del selector y 71 tests obligatorios para la futura implementación. Permanece `under_review` hasta aprobación explícita del Product Owner.

## Checkpoints

| Checkpoint | Objetivo | Estado | Pendientes individuales | Criterio de salida |
|---|---|---|---|---|
| CP-01 | Censo y reconstrucción del contrato actual | `complete` | Ninguno. Evidencia registrada en `knowledge/evidence/proportion-ci-cp01-census.md` | Informe reproducible read-only completado sobre baseline exacto |
| CP-02 | Especificación estadística | `complete` | Ninguno. Spec aceptada en `knowledge/decisions/proportion-ci-cp02-spec.md` | Aprobación explícita del Product Owner registrada |
| CP-03 | Contrato de API y compatibilidad | `under_review` | Aprobación explícita del Product Owner del contrato `proportion-ci-cp03-api-contract.md`; registry deberá indexar CP-01/CP-02/CP-03 antes de una futura integración | Contrato público aprobado; sin cambios de producción |
| CP-04 | Diseño de calibración | `pending` | Definir grid de `n`, `p`, alpha/confidence; cobertura exacta por enumeración binomial cuando sea posible; ancho, conservadurismo, monotonicidad, simetrías, boundary behavior y referencias externas | Protocolo pre-registrado antes de mirar resultados finales/holdout |
| CP-05 | Implementación candidata | `pending` | Implementar sólo métodos aprobados y tests unitarios/metamórficos; corregir routing incompatible de proporción; no activar routing automático | SHA candidato reproducible en rama aislada |
| CP-06 | Calibración y evidencia | `pending` | Ejecutar harness aprobado; producir summary/metadata/evidence; cuantificar cobertura y límites por método | Evidencia suficiente para clasificar cada método como validado, limitado o no calibrado |
| CP-07 | Auditoría adversarial | `pending` | Antigravity audita el SHA exacto; boundaries, precisión numérica, invariancias, regresión y claims estadísticos | Cero hallazgos bloqueantes o nueva iteración con SHA nuevo |
| CP-08 | Decisión de integración | `pending` | ChatGPT interpreta evidencia; Product Owner decide alcance final, PR y merge | Sólo Product Owner autoriza PR/merge |

## Arquitectura aceptada en CP-02

- Estimando: una única proporción poblacional `p=P(Y=1)` bajo diseño Bernoulli/binomial ordinario.
- Wilson es candidato principal y default candidato.
- Clopper–Pearson es candidato frecuentista exacto/conservador.
- Wald permanece únicamente como método legacy explícito; nunca default ni fallback automático.
- Jeffreys sólo entra como comparador y, si alguna vez se expone, con semántica explícita de credible interval bayesiano; no como CI frecuentista.
- Agresti–Coull, Wilson-CC, mid-P y otras variantes no están autorizadas para producción en este stage.
- El alcance inicial permanece bilateral; one-sided requiere calibración propia.
- `x=0` y `x=n` son realizaciones válidas.
- `successes >= 10 and failures >= 10` no es selector universal ni garantía.
- La entrada agregada futura debe representar `(successes, trials)` enteros; éxitos fraccionarios no se reinterpretan como binomiales.
- Survey weights, effective sample size, clustering y finite-population correction permanecen fuera de alcance.
- `MethodSelector` debe fallar cerrado para `Estimand.PROPORTION` mientras no exista capability registrada y calibrada; nunca transferir `one_sample_t` desde el estimando media.
- `BootstrapCI(stat="proportion")` permanece separado hasta contar con contrato y calibración específicos.

## Contrato propuesto en CP-03 — pendiente de aprobación

- Preservar `PopulationProportionCI(data, alpha=0.05, incidences=None, method="wilson", *, independence="unknown")`.
- Añadir `PopulationProportionCI.from_counts(successes, trials, alpha=0.05, method="wilson", *, independence="unknown")` como API agregada canónica con conteos enteros.
- Re-exportar `PopulationProportionCI` desde `pyMagicStat.inference` sin retirar la ruta `.parametric`.
- Métodos públicos del stage: `wilson`, `clopper_pearson`, `wald`.
- Mantener `calculate_interval()` como dict JSON-serializable con claves legacy y metadata nueva.
- `incidences` numérico queda temporalmente compatible con `DeprecationWarning`; los fraccionarios no se consideran soportados por el contrato binomial y Clopper–Pearson los rechaza.
- Wald permanece legacy explícito, sin clipping y sin fecha de retirada automática.
- `BootstrapCI(stat="proportion")` permanece separado y nunca es fallback.
- `MethodSelector` debe devolver `NOT_CALIBRATED`/`NOT_CALIBRATED`, `selected_method=None`, sin alternativas de media para `Estimand.PROPORTION` hasta que exista capability calibrada.
- CP-05 deberá satisfacer la matriz de 71 tests especificada antes de cualquier afirmación de implementación correcta.

## Invariantes del stage

1. Ningún checkpoint posterior puede darse por completado por el mero hecho de que los tests de software estén verdes.
2. `fail_to_reject`, cobertura nominal aproximada o una referencia externa no equivalen por sí solos a autorización de routing automático.
3. La calibración debe separar desempeño estadístico de fallos numéricos.
4. No se transferirá evidencia de media, ANOVA, EL o GOF a intervalos de proporción.
5. Cada SHA nuevo requiere su propia evidencia y auditoría; un PASS no se transfiere.
6. `main` permanece protegido y no se modifica directamente durante el stage.

## Trabajo posterior ya acordado

Tras aprobación de CP-03 se abre CP-04, preregistro del experimento/calibración. CP-05 seguirá pendiente y no podrá comenzar por la sola aprobación de CP-03.

Después de cerrar este stage, el siguiente bloque del roadmap es el Gate de calibración de diagnósticos de distribución/GOF (Shapiro, D’Agostino, Anderson, Q-Q, KS/Lilliefors y huecos de muestra pequeña), seguido —sólo con evidencia suficiente— por el motor automático de decisión de método de bondad y ajuste.
