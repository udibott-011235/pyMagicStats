# STAGE-PROP-CI-001 — Revisión y ampliación de intervalos de proporción

**Estado general:** `in_progress`  
**Fecha de apertura:** 2026-08-30  
**Baseline canónico:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`  
**Rama de registro:** `docs/proportion-ci-stage`  
**Owner:** Product Owner  
**Arquitectura estadística:** ChatGPT  
**Implementación:** Cortex/Codex, sólo después de aprobación explícita  
**QA adversarial:** Antigravity, sobre SHA candidato exacto

## Estado tras aprobación de CP-04

CP-01 fue completado mediante auditoría read-only y está registrado en `knowledge/evidence/proportion-ci-cp01-census.md`.

CP-02 fue aprobado explícitamente por el Product Owner y está registrado en `knowledge/decisions/proportion-ci-cp02-spec.md`.

CP-03 fue aprobado explícitamente por el Product Owner. El contrato de API y compatibilidad está en `knowledge/decisions/proportion-ci-cp03-api-contract.md` y su aceptación en `knowledge/decisions/proportion-ci-cp03-acceptance.md`.

CP-04 fue aprobado explícitamente por el Product Owner. El preregistro exhaustivo está en `knowledge/experiments/proportion-ci-cp04-preregistration.md` y su aceptación en `knowledge/decisions/proportion-ci-cp04-acceptance.md`.

La indexación de CP-01/CP-02/CP-03 en `knowledge/registry.json` se delegó como mantenimiento documental independiente a QA/Antigravity en rama separada. Esa sincronización no altera los contratos aceptados ni bloquea la implementación.

CP-05 queda listo para implementación por Cortex/Codex, pero requiere respetar estrictamente CP-02, CP-03 y CP-04. La calibración final, Monte Carlo shadow y holdout pertenecen a CP-06 y no deben ejecutarse como evidencia final durante CP-05.

## Checkpoints

| Checkpoint | Objetivo | Estado | Pendientes individuales | Criterio de salida |
|---|---|---|---|---|
| CP-01 | Censo y reconstrucción del contrato actual | `complete` | Indexación en registry delegada como mantenimiento documental | Informe reproducible read-only completado sobre baseline exacto |
| CP-02 | Especificación estadística | `complete` | Indexación en registry delegada como mantenimiento documental | Aprobación explícita del Product Owner registrada |
| CP-03 | Contrato de API y compatibilidad | `complete` | Indexación en registry delegada; no cambia el contrato aceptado | Contrato público aprobado; sin cambios de producción |
| CP-04 | Diseño/preregistro de calibración | `complete` | Ninguno dentro de arquitectura; preregistro congelado salvo enmienda explícita previa a resultados afectados | Protocolo preregistrado y aprobado antes de implementación/calibración |
| CP-05 | Implementación candidata | `ready_for_implementation` | Cortex/Codex debe implementar únicamente el contrato aprobado, satisfacer los 71 tests, corregir routing de proporción fail-closed y producir SHA candidato; no activar routing automático ni ejecutar holdout final | SHA candidato reproducible en rama aislada con tests contractuales verdes y handoff completo |
| CP-06 | Calibración y evidencia | `pending` | Ejecutar el harness preregistrado sobre SHA congelado; producir summary/metadata/evidence; cuantificar cobertura y límites por método | Evidencia suficiente para clasificar cada método como validated_with_limits, not_calibrated o rejected_for_claim |
| CP-07 | Auditoría adversarial | `pending` | Antigravity audita el SHA exacto y la evidencia CP-06; boundaries, precisión numérica, invariancias, regresión y claims estadísticos | Cero hallazgos bloqueantes o nueva iteración con SHA nuevo |
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

## Contrato aceptado en CP-03

- Preservar `PopulationProportionCI(data, alpha=0.05, incidences=None, method="wilson", *, independence="unknown")`.
- Añadir `PopulationProportionCI.from_counts(successes, trials, alpha=0.05, method="wilson", *, independence="unknown")` como API agregada canónica con conteos enteros.
- Re-exportar `PopulationProportionCI` desde `pyMagicStat.inference` sin retirar la ruta `.parametric`.
- Métodos públicos del stage: `wilson`, `clopper_pearson`, `wald`.
- Mantener `calculate_interval()` como dict JSON-serializable con claves legacy y metadata nueva.
- `incidences` numérico queda temporalmente compatible con `DeprecationWarning`; los fraccionarios no se consideran soportados por el contrato binomial y Clopper–Pearson los rechaza.
- Wald permanece legacy explícito, sin clipping y sin fecha de retirada automática.
- `BootstrapCI(stat="proportion")` permanece separado y nunca es fallback.
- `MethodSelector` debe devolver `NOT_CALIBRATED`, `selected_method=None`, sin alternativas de media para `Estimand.PROPORTION` hasta que exista capability calibrada.
- CP-05 deberá satisfacer la matriz de 71 tests contractuales especificada antes de cualquier afirmación de implementación correcta.

## Preregistro aceptado en CP-04

- Métodos: Wilson, Clopper–Pearson y Wald; Jeffreys como comparador no productivo.
- Alpha: `0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.200`.
- `n=1..5000` exhaustivo, más stress points hasta `1,000,000`.
- Probabilidades: anchors extremos, grid lineal interior de 9,999 puntos, grid event-scale `lambda=np`, endpoints y vecinos inducidos por los intervalos.
- Cobertura y ancho esperados por enumeración binomial determinista cuando sea computable.
- Búsqueda adversarial de mínimos de cobertura dentro de regiones inducidas por endpoints, no sólo evaluación del grid.
- Estratificación por `min(np,n(1-p))` sin convertirla en thresholds de routing.
- Tiers preregistrados de undercoverage.
- Gates matemático-numéricos separados por método.
- High-precision audit a 80 dígitos para celdas materiales/sospechosas.
- Monte Carlo shadow: 128 celdas × 1,000,000 + 512 × 250,000 = 256 millones de draws.
- Holdout confirmatorio de 10,000 celdas generado sólo después del freeze del SHA de CP-05.
- Reproducibilidad, invariancia a shard/backend y prohibición explícita de calibration hacking.
- Ningún resultado de CP-06 autoriza por sí solo routing automático.

## Invariantes del stage

1. Ningún checkpoint posterior puede darse por completado por el mero hecho de que los tests de software estén verdes.
2. Cobertura nominal aproximada o una referencia externa no equivalen por sí solas a autorización de routing automático.
3. La calibración debe separar desempeño estadístico de fallos numéricos.
4. No se transferirá evidencia de media, ANOVA, EL o GOF a intervalos de proporción.
5. Cada SHA nuevo requiere su propia evidencia y auditoría; un PASS no se transfiere.
6. `main` permanece protegido y no se modifica directamente durante el stage.
7. El preregistro de CP-04 queda congelado antes de observar resultados finales utilizados para decisiones de política.
8. Un backend acelerado puede cambiar rendimiento, nunca el dominio estadístico ni las celdas evaluadas.
9. CP-05 no puede cambiar `calibration_status` a un estado favorable ni registrar capacidades automáticas antes de CP-06/CP-07.

## Trabajo posterior ya acordado

CP-05 produce un candidato congelado. CP-06 ejecuta la calibración preregistrada sobre ese SHA. CP-07 realiza auditoría adversarial independiente. Sólo después se llega a CP-08.

Después de cerrar este stage, el siguiente bloque del roadmap es el Gate de calibración de diagnósticos de distribución/GOF, seguido —sólo con evidencia suficiente— por el motor automático de decisión de método de bondad de ajuste.