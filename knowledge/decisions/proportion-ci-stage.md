# STAGE-PROP-CI-001 — Revisión y ampliación de intervalos de proporción

**Estado general:** `in_progress`  
**Fecha de apertura:** 2026-08-30  
**Baseline canónico:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`  
**Rama de registro:** `docs/proportion-ci-stage`  
**Owner:** Product Owner  
**Arquitectura estadística:** ChatGPT  
**Implementación:** Cortex/Codex  
**QA adversarial:** Antigravity, sobre SHA candidato exacto

## Estado actual

CP-01 fue completado mediante auditoría read-only y está registrado en `knowledge/evidence/proportion-ci-cp01-census.md`.

CP-02 fue aprobado explícitamente por el Product Owner y está registrado en `knowledge/decisions/proportion-ci-cp02-spec.md`.

CP-03 fue aprobado explícitamente por el Product Owner. El contrato de API y compatibilidad está en `knowledge/decisions/proportion-ci-cp03-api-contract.md` y su aceptación en `knowledge/decisions/proportion-ci-cp03-acceptance.md`.

CP-04 fue aprobado explícitamente por el Product Owner. El preregistro exhaustivo está en `knowledge/experiments/proportion-ci-cp04-preregistration.md` y su aceptación en `knowledge/decisions/proportion-ci-cp04-acceptance.md`.

CP-05 fue aceptado sobre el SHA candidato congelado `2df5b90a5395163e723f9c52aafbb91fdce96d43`. La aceptación está en `knowledge/decisions/proportion-ci-cp05-acceptance.md`. La revisión arquitectónica previa está en `knowledge/evidence/proportion-ci-cp05-architecture-review-c452a05.md`; su único hallazgo MINOR, `CP05-AR-001`, quedó cerrado al ampliar el test Wilson #28 al grid CP-01 completo `alpha={0.01,0.05,0.10}`, `n=1..200`, `x=0..n`.

CP-06 queda abierto exclusivamente para ejecutar la calibración preregistrada de CP-04 sobre el SHA congelado `2df5b90a5395163e723f9c52aafbb91fdce96d43`. Ningún cambio de código productivo puede hacerse durante esa ejecución sin invalidar la transferencia de evidencia al candidato.

La sincronización previa del registry por Antigravity permanece en rama documental separada y será consolidada junto con CP-04/CP-05 antes de futura integración.

## Checkpoints

| Checkpoint | Objetivo | Estado | Pendientes individuales | Criterio de salida |
|---|---|---|---|---|
| CP-01 | Censo y reconstrucción del contrato actual | `complete` | Consolidación final del registry antes de integración | Evidencia reproducible read-only completada |
| CP-02 | Especificación estadística | `complete` | Consolidación final del registry antes de integración | Spec aprobada |
| CP-03 | Contrato de API y compatibilidad | `complete` | Consolidación final del registry antes de integración | Contrato público aprobado |
| CP-04 | Diseño/preregistro de calibración | `complete` | Ninguno; preregistro congelado | Protocolo aprobado antes de implementación/calibración |
| CP-05 | Implementación candidata | `complete` | Ninguno; SHA congelado `2df5b90a5395163e723f9c52aafbb91fdce96d43` | 77 tests contractuales verdes, suite completa verde, revisión arquitectónica cerrada |
| CP-06 | Calibración y evidencia | `in_progress` | Ejecutar íntegramente el preregistro CP-04 sobre el SHA congelado; producir outputs, metadata, high-precision audit, shadow MC y holdout | Evidencia suficiente para clasificar cada método y congelar artefactos reproducibles |
| CP-07 | Auditoría adversarial | `pending` | Antigravity audita SHA exacto y evidencia CP-06 | Cero hallazgos bloqueantes o nueva iteración con SHA nuevo |
| CP-08 | Decisión de integración | `pending` | ChatGPT interpreta evidencia; Product Owner decide alcance final, PR y merge | Sólo Product Owner autoriza PR/merge |

## Contrato estadístico vigente

- Estimando: una única proporción poblacional `p=P(Y=1)` bajo diseño Bernoulli/binomial ordinario.
- Wilson permanece default candidato.
- Clopper–Pearson permanece exacto/conservador.
- Wald permanece legacy explícito, nunca default ni fallback automático.
- Jeffreys es únicamente comparador bayesiano no productivo.
- `from_counts(successes,trials)` es la API agregada canónica.
- `incidences` numérico permanece en transición con deprecación; fraccionarios no tienen soporte binomial.
- `BootstrapCI(stat="proportion")` permanece separado.
- `MethodSelector` falla cerrado para `Estimand.PROPORTION` y no transfiere capacidades de media.
- `calibration_status` permanece `not_calibrated` durante CP-06 hasta decisión posterior; ejecutar calibración no modifica código ni metadata productiva automáticamente.

## Preregistro CP-04 congelado

- Métodos productivos evaluados: Wilson, Clopper–Pearson, Wald.
- Comparador no productivo: Jeffreys.
- Alpha: `0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.200`.
- `n=1..5000` exhaustivo más stress hasta `n=1,000,000`.
- `p`: anchors extremos, grid lineal interior, event-scale `lambda=np`, endpoints y vecinos inducidos.
- Cobertura y expected width por enumeración binomial determinista.
- Búsqueda adversarial de mínimos entre regiones inducidas por endpoints.
- Estratificación por `min(np,n(1-p))` sin convertirla en regla de routing.
- High-precision audit a 80 dígitos para celdas sospechosas/materiales.
- Shadow Monte Carlo total: 256 millones de draws.
- Holdout: 10,000 celdas generado sólo después del freeze de CP-05.
- Prohibición de calibration hacking y de modificar thresholds/grid tras observar resultados.
- Ningún resultado de CP-06 autoriza por sí solo routing automático.

## Invariantes del stage

1. Tests verdes no equivalen a validez estadística.
2. Cobertura aproximada o coincidencia con un oráculo no autoriza routing.
3. Separar desempeño estadístico de fallos numéricos.
4. No transferir evidencia de media, ANOVA, EL o GOF.
5. Todo SHA nuevo requiere evidencia propia; un PASS no se transfiere.
6. `main` permanece protegido.
7. CP-04 queda congelado antes de resultados confirmatorios.
8. El backend puede cambiar rendimiento, nunca el dominio estadístico.
9. CP-06 no puede alterar el SHA candidato ni promover `calibration_status` automáticamente.
10. El holdout deja de ser holdout si se usa para ajustar el mismo SHA; todo fix requiere nuevo SHA y nueva evaluación independiente.

## Hito transversal posterior al stage

El cierre de CP-08 no constituye por sí solo autorización para construir el orquestador de decisión. `STAGE-PROP-CI-001` es uno de los bloqueantes de entrada de `DEC-007 — MANUAL UAT CHECKPOINT 1 — CURRENT STATISTICAL CORE`.

Después de este stage, el proyecto todavía debe cerrar el bloque estadístico de ANOVA, realizar el accuracy closure/censo de la superficie de distribuciones y GOF que vaya a entrar al baseline, y congelar un inventario explícito de métodos antes de ejecutar el primer UAT manual con DataFrames reales/sucios.

El Manual UAT 1 probará herramientas aisladas mediante invocación explícita; no probará `MethodSelector` ni un decision engine. Un PASS de ese hito habilitará sólo uso manual de los módulos incluidos dentro de sus límites documentados. Nuevas distribuciones, transformaciones, métodos no paramétricos, regresiones, DOE y el orquestador permanecen como desarrollo posterior salvo decisión explícita del Product Owner.

Referencia canónica del plan: `knowledge/decisions/manual-uat-checkpoint-1.md`.

## Próximo paso

Ejecutar CP-06 sobre `2df5b90a5395163e723f9c52aafbb91fdce96d43`, conservar artefactos reproducibles y entregar resultados a arquitectura antes de CP-07.
