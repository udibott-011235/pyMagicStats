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

## EV-002 — Diagnóstico de una vía en main

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

