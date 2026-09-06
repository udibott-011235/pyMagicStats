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

- **Estado:** `under_review`
- **CP02:** `IN_PROGRESS`
- **Baseline:** `main@ccff392af13d2cb52d1f3888a986ef58be0099e2`
- **Rama:** `feature/distribution-family-framework-cp02-continuous-core`
- **Contrato:** `DEC-011`
- **Evidencia de apertura:** identidad de baseline, árbol limpio, registro
  válido y 49 regresiones congeladas antes de implementar.
- **Evidencia de implementación:** 277 tests CP02, 49 regresiones congeladas y
  326 tests en la superficie de distribución combinada; paridad directa con
  SciPy para Gamma/Exponential, RNG explícito y compatibilidad de exports.
- **Candidato:** commit local de implementación identificado en el handoff; no
  publicado.
- **Demuestra:** mecánica determinista y contrato API en el entorno registrado.
- **No demuestra:** fitting, estimación, GOF, calibración, selección automática
  ni autorización de CP03.

Fuente canónica:
`knowledge/evidence/distribution-family-framework-cp02-evidence.md`.
