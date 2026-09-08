# Evidencia vigente indexada

## EV-001 — Calibración de robustez de la media

- **Política:** `mean-v2.1-2026-08`
- **Estado:** `validated_with_limits`
- **Diseño:** inferencia t bilateral de una media independiente.
- **Matriz:** 19 escenarios × 8 tamaños × 1,000 réplicas = 152,000 muestras.
- **Semilla:** `20260826`.
- **Artefactos:** runner, metadata JSON y resumen CSV versionados.
- **Demuestra:** comportamiento empírico de cobertura/error tipo I y decisiones
  de robustez dentro de la matriz declarada.
- **No demuestra:** validez para ANOVA, potencia, pruebas unilaterales,
  dependencia, diseños agrupados o toda distribución posible.

Fuentes canónicas:

- `Docs/sampling-robustness-calibration.md`
- `experiments/robustness_calibration.py`
- `experiments/results/sampling_robustness_metadata.json`
- `experiments/results/sampling_robustness_summary.csv`

## EV-002 — Diagnósticos de residuos para diseño de una vía

- **Estado:** `validated_with_limits` como diagnóstico de software.
- **Demuestra:** el contrato documentado evalúa residuos centrados por grupo,
  balance, heterocedasticidad e independencia declarada.
- **No demuestra:** que ANOVA o Welch ANOVA estén implementados/calibrados en
  `main`, ni que la política de una media se transfiera a múltiples grupos.

Fuente canónica: `Docs/inference-engine.md` y tests de supuestos en `main`.

## EV-003 — Deuda numérica de escalas subnormales

- **Estado:** `open`, no bloqueante para retail/BI actual.
- **Riesgo:** la tolerancia de degeneración puede perder invariancia de escala
  por underflow en magnitudes extremas de `float64`.
- **Criterio de cierre:** normalización segura y tests metamórficos de escala.

Fuente canónica: `Docs/technical-debt.md`.

## EV-004 — Gate 2 adversarial clear — 9a87c5d

- **Estado:** `validated_with_limits`
- **Candidato:** `9a87c5d48dba8b8a172b5386d7318e7f37ec98fe`
- **Parent directo:** `0fc71c90c15f7c82b55ba650de742265d492df33`
- **Rama:** `fix/gate2-adversarial-remediation`
- **Demuestra:** validación estadística focalizada del candidato Gate 2 (10,000 configuraciones metamórficas, 54/54 Poisson, 12/12 Binomial, 28 passed en pruebas focales, cero defectos críticos o mayores en el alcance).
- **Límites:** `TD-GOF-SUPPORT-001` y `FINDING-ADV-NUM-004` permanecen fuera de alcance; GOF no demuestra identidad distributiva ni autoriza merge.

Fuente canónica: `knowledge/evidence/gate2-adversarial-clear-9a87c5d.md`.

## EV-005 — Integración controlada de Gate 2 en main

- **Estado:** `validated_with_limits`
- **Merge SHA:** `f1725ebdfebcb667c053420e4cb4c1e35048f9e0`
- **Parents:** `e8422a74cef7d3eebc1f807666e9388acd407794`, `9a87c5d48dba8b8a172b5386d7318e7f37ec98fe`
- **Tree:** `238222f324e33c1c3cc19d25c0483474671ecb87`
- **Integración:** PR #3 (`fix/gate2-adversarial-remediation` -> `main`)
- **Demuestra:** integración controlada del árbol auditado en EV-004; igualdad exacta con el rehearsal; 289 passed, 3 skipped (exclusivamente por CuPy/CUDA); bypass automático en creación de rama pero no en merge; ramas Gate 2 preservadas.
- **Límites:** `TD-GOF-SUPPORT-001` y `FINDING-ADV-NUM-004` abiertos y fuera de alcance; GOF no demuestra identidad distributiva.

Fuente canónica: `knowledge/evidence/gate2-integration-f1725eb.md`.

## EV-007 — Distribution Family Framework CP01

- **Estado:** `accepted`
- **CP01:** `COMPLETE`
- **Integración:** `COMPLETE` mediante PR #6
- **Candidato aceptado:** `c63b48eafc439de8857207fd21c1b593e38a3187`
- **Integration head:** `3f9acd5a51ce38ae62b9800d50efb0949c6531f0`
- **Merge SHA:** `46f827dd107aa9e6f940f0de085fbb91075ff049`
- **Antigravity pre-merge:** `ADVERSARIAL_PASS`
- **Baseline:** `main@402e4601df460811779b3238c2526ac12f463a67`
- **Stage:** `STAGE-DIST-FAMILIES-001`
- **Demuestra:** materialización trazable, aceptación arquitectónica del
  contrato congelado en `c63b48e…`, auditoría adversarial del integration head
  e integración mediante merge commit en `main@46f827d…`, sin cambios de
  producción.
- **No demuestra:** implementación de familias, validez de estimadores,
  calibración GOF ni autorización de CP02.
- **Revisión arquitectónica:** `72ecdba…` queda preservado como primer
  candidato, `c63b48e…` como candidato arquitectónico aceptado y `3f9acd5…`
  como integration head preservado en el merge commit `46f827d…`.

Fuente canónica:
`knowledge/evidence/distribution-family-framework-cp01-evidence.md`.

## EV-008 — Distribution Family Framework CP02 continuous core

- **Estado:** `accepted`
- **CP02:** `COMPLETE`
- **Baseline:** `main@ccff392af13d2cb52d1f3888a986ef58be0099e2`
- **Rama:** `feature/distribution-family-framework-cp02-continuous-core`
- **Contrato:** `DEC-011`
- **Implementación aceptada por Arquitectura:**
  `e9ef63b802a8cb08ea38b32e87b206432c08b120`
- **Governance head auditado:**
  `9cf25c157a4f4114f41ae74d4e04e009392414e3`
- **Governance head pre-merge final:**
  `b3d116f2f3e82b74ab6fb5f8337e217104c3bca6`
- **Auditoría adversarial pre-merge:** `ADVERSARIAL_PASS`
- **Clasificación:** 0 BLOCKER, 0 MAJOR, 0 MINOR, 1 INFO preexistente
  fuera de alcance (`FINDING-ADV-CP02-003`).
- **Integración:** `COMPLETE` mediante PR #8 en
  `main@aa5723d2cb7dbaf48e6f9059368b9fdaeeb7926c`.
- **Parents del merge:** `ccff392af13d2cb52d1f3888a986ef58be0099e2`,
  `b3d116f2f3e82b74ab6fb5f8337e217104c3bca6`.
- **Evidencia de apertura:** identidad de baseline, árbol limpio, registro
  válido y 49 regresiones congeladas antes de implementar.
- **Evidencia de implementación:** 311 tests CP02, 49 regresiones congeladas y
  360 tests en la superficie de distribución combinada; paridad directa con
  SciPy para Gamma/Exponential, RNG explícito y compatibilidad de exports.
- **Diferencial de suite completa:** baseline 287 passed / 3 skipped / 2 failed;
  candidato 598 passed / 3 skipped / 2 failed; `NO_NEW_FAILURES`.
- **Demuestra:** mecánica determinista y contrato API en el entorno registrado.
- **No demuestra:** fitting, estimación, GOF, calibración, selección automática
  ni autorización de CP03; tampoco garantiza semántica de igualdad/hash entre
  instancias separadas de descriptores Family. El stage general permanece
  `IN_PROGRESS`; CP03 está `IN_PROGRESS` con integración `PENDING`, y
  CP04–CP08 siguen `NOT_STARTED`.

Fuente canónica:
`knowledge/evidence/distribution-family-framework-cp02-evidence.md`.

## EV-009 — Distribution Family Framework CP03 discrete core baseline

- **Estado:** `accepted`
- **CP03:** `IN_PROGRESS`
- **Reconnaissance:** `COMPLETE`
- **Arquitectura:** `FROZEN`
- **Implementación:** `PENDING`
- **Baseline:** `main@02a65c80c5da10295d6eeef42e691772d0686ca2`
- **Rama:** `feature/distribution-family-framework-cp03-discrete-core`
- **Contrato:** `DEC-012`
- **Superficie legacy actual:** `BinomialDistribution`,
  `PoissonDistribution`, `DiscreteDistributionValidator` y helpers Pearson GOF.
- **Nueva superficie discreta de familias antes de CP03:** ninguna.
- **Gaps confirmados antes de implementar:** `DistributionSupport` todavía no
  expresa membership entero discreto y el normalizador CP02 de resultados no
  satisface el contrato `int`/`int64` de RVS discreto.
- **Baseline heredado:** 49 regresiones legacy, 311 tests CP02 y 360 tests
  combinados de distribución en PASS; no se reejecutaron en esta tarea de
  gobernanza.
- **Demuestra:** reconnaissance completo y contrato discreto congelado.
- **No demuestra:** implementación CP03, ejecutabilidad numérica de RVS extremo,
  fitting, GOF, routing, familias adicionales ni autorización de producción.

Fuente canónica:
`knowledge/evidence/distribution-family-framework-cp03-baseline.md`.

## EV-010 — Distribution Family Framework CP03 implementation

- **Estado:** `accepted`
- **CP03:** `IN_PROGRESS`
- **Arquitectura:** `FROZEN` mediante `DEC-012`
- **Implementación auditada:**
  `4f7fa09bc7ab501d21f6d27dada30ade23397588`
- **Baseline:** `main@02a65c80c5da10295d6eeef42e691772d0686ca2`
- **Rama:** `feature/distribution-family-framework-cp03-discrete-core`
- **Auditoría adversarial pre-merge:** `ADVERSARIAL_PASS`
- **Clasificación:** 0 BLOCKER, 0 MAJOR, 0 MINOR, 1 INFO.
- **Regresión congelada:** 360 passed.
- **Tests CP03:** 215 passed.
- **Superficie de distribución:** 575 passed.
- **Diff exacto de implementación:** cinco rutas: exports de
  `pyMagicStat/distributions`, exports y core de `families`, implementación
  discreta y `tests/test_discrete_distribution_families.py`; la lista canónica
  exacta consta en EV-010.
- **Probes RVS extremos acotados:** matriz completa de 16 casos en EV-010,
  todos con `size=5`, `rng=42` y timeout externo de 10 segundos: 10 `SUCCESS`,
  6 `NUMERICAL_FAILURE`, 0 `BACKEND_RANGE_FAILURE` y 0 `TIMEOUT`.
- **INFO-001:** limitación acotada de ejecutabilidad del backend SciPy para
  sampling extremo; los fallos backend numéricos/de rango se traducen a
  `FloatingPointError` después de la validación pública, sin thresholds
  matemáticos para `r` o `p`.
- **Validación de Knowledge Base:** parent 7 passed / 2 failed; candidato 7
  passed / 2 failed; `NO_NEW_FAILURES`. Los dos fallos heredados exactos y el
  entorno de materialización constan en EV-010. La suite completa del
  repositorio no se reejecutó para esta corrección documental.
- **Integración:** `PENDING`.
- **Demuestra:** implementación del contrato discreto, paridad SciPy,
  endpoints PPF canónicos, RNG explícito, normalización `int`/`int64`, guards
  fail-closed y aislamiento legacy en el SHA auditado.
- **No demuestra:** integración canónica, finalización de CP03, fitting, GOF,
  routing, familias discretas adicionales ni autorización de CP04.

Fuente canónica:
`knowledge/evidence/distribution-family-framework-cp03-evidence.md`.
