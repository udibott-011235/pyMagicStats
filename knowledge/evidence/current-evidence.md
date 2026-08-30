# Evidencia inicial indexada

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

## EV-005 — Integración controlada de Gate 2 en main

- **Estado:** `validated_with_limits`
- **Merge SHA:** `f1725ebdfebcb667c053420e4cb4c1e35048f9e0`
- **Parents:** `e8422a74cef7d3eebc1f807666e9388acd407794`, `9a87c5d48dba8b8a172b5386d7318e7f37ec98fe`
- **Tree:** `238222f324e33c1c3cc19d25c0483474671ecb87`
- **Integración:** PR #3 (`fix/gate2-adversarial-remediation`)
- **Verificación:** igualdad exacta con el rehearsal; 289 passed, 3 skipped (exclusivamente por CuPy/CUDA).
- **Límites:** `TD-GOF-SUPPORT-001` y `FINDING-ADV-NUM-004` abiertos y fuera de alcance; GOF no demuestra identidad distributiva.
- **Gobernanza:** bypass automático observado al crear la rama, pero no durante el merge; ramas Gate 2 preservadas.

Fuente canónica: `knowledge/evidence/gate2-integration-f1725eb.md`.
