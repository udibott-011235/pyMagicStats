# CP-ANOVA-06 — Preregistration freeze

- Fecha: `2026-09-05`
- Rama: `audit/anova-calibration-preregistration`
- Base: `501e97db102567f2ee225da3dccb026b027667c8`
- Preregistration spec commit: `60d1fc58d50766c485d07d15e54693ccb6bbbe8e`
- Manifest commit: `f5ffc7ee75d11503b422181af2e4ef49ea65e0c9`
- Estado: `complete/frozen`

## Artefactos congelados

1. `knowledge/experiments/anova-cp06-calibration-preregistration.md`
2. `knowledge/experiments/anova-cp06-calibration-manifest.json`

## Decisiones principales

- Classical y Welch se calibran como métodos explícitos; no se calibra selector.
- Phase E0 es engineering-only y no puede usarse como evidencia.
- Phase D separa core H0, robustness H0, stress H0 y power H1.
- Phase H es holdout sellado con familias y diseños no vistos en desarrollo.
- Classical/Welch reciben exactamente la misma muestra por réplica.
- Seeds son independientes de workers/shards/order y derivadas con SHA-256 + SeedSequence.
- Gate principal alpha=0.05 usa Wilson 99% CI.
- Classical confirmatory: normal + equal variance + min n>=5, CI99 completamente dentro `[0.04,0.06]`.
- Welch confirmatory: toda celda normal core con min n>=5, equal/unequal variance, CI99 completamente dentro `[0.04,0.06]`.
- Heterocedastic Classical y no-Gaussian robustness son caracterización, no selector/policy authorization.
- Power no tiene PASS threshold; se reporta comparativamente.
- Harness usa summaries+kernels de producción y debe superar parity gate contra API pública.
- No se ejecuta Monte Carlo pesado en CP-ANOVA-06.

## Siguiente checkpoint

`CP-ANOVA-07A — Cortex harness implementation`

Cortex debe implementar el harness y los artefactos exactamente contra el preregistro/manifest congelados. No puede modificar scenarios, seeds, replications, acceptance bands, holdout, accounting o execution semantics sin reabrir CP-ANOVA-06.

Antes de cualquier corrida de evidencia, ChatGPT debe realizar `CP-ANOVA-07B` auditando el harness y su parity/reproducibility suite.
