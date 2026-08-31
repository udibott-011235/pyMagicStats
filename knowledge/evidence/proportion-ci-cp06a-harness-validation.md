# CP-06A — Validación del harness de calibración de intervalos de proporción

**Stage:** `STAGE-PROP-CI-001`  
**Estado:** `validated_with_limits`  
**Rol:** experiment/calibration engineering + statistical-software-architecture review  
**Candidato productivo congelado:** `2df5b90a5395163e723f9c52aafbb91fdce96d43`  
**Harness experimental:** `experiments/proportion-ci-calibration` @ `c7ece2118075343e322ea2792f1d700d9f77334c`

## Propósito

CP06-A valida que el harness experimental está materializado, reproducible y suficientemente estable para trasladar la carga computacional de CP06-B→I a Quantum. No constituye calibración estadística final de Wilson, Clopper–Pearson, Wald ni Jeffreys.

## Aislamiento confirmado

GitHub confirma que `c7ece2118075343e322ea2792f1d700d9f77334c` es exactamente un commit sobre `2df5b90a5395163e723f9c52aafbb91fdce96d43` y añade únicamente scripts, tests y artefactos experimentales. `pyMagicStat/` permanece intacto.

## Validación focal reportada

El handoff de Cortex/Codex reporta:

- 17 pruebas focalizadas verdes en 5.94 s;
- `n = 1, 2, 5, 10, 97, 101`;
- probabilidades extremas próximas a 0 y 1;
- comparación contra suma PMF completa con tolerancia absoluta `5e-14`;
- simetría `p ↔ 1-p` con tolerancia `5e-14`;
- invariancia entre batches con tolerancia `5e-15`;
- masa PMF mínima retenida `0.9999999999999964`;
- cero `RuntimeWarning` de SciPy;
- smoke multiproceso 2 workers / batch 64 completado en 8.551 s.

## Corrección del harness

Durante un intento previo de CP06-B, `scipy.stats.binom.isf` no logró acotar raíces para probabilidades extremadamente cercanas a 1. La ejecución fue detenida y el problema se clasificó como defecto del harness, no del código productivo.

El harness publicado elimina esa dependencia para selección de soporte y usa una cota determinista de Hoeffding con masa omitida máxima `1e-14`, contrastada contra suma PMF completa en el smoke test.

## Artefactos materializados

- `experiments/proportion_ci_calibration/__init__.py`
- `experiments/proportion_ci_calibration/harness.py`
- `experiments/proportion_ci_calibration/run.py`
- `experiments/proportion_ci_calibration/README.md`
- `experiments/proportion_ci_calibration/requirements-quantum.txt`
- `tests/experiments/test_proportion_ci_calibration.py`
- artefactos smoke bajo `experiments/results/proportion_ci_cp06_smoke_*`

Dependencias adicionales fijadas para Quantum:

- `mpmath==1.4.1`
- `pyarrow==25.0.1`

## Límite explícito

No se ejecutaron CP06-B–H, los 256 millones de draws ni el holdout. Por tanto no existe todavía evidencia para cambiar `calibration_status`, registrar capabilities automáticas ni clasificar definitivamente un método.

## Decisión

`CP06-A: COMPLETE / validated_with_limits`

Siguiente trabajo: ejecutar CP06-B→I exclusivamente en Quantum sobre el harness `c7ece2118075343e322ea2792f1d700d9f77334c`, manteniendo inmutable el candidato productivo `2df5b90a5395163e723f9c52aafbb91fdce96d43`.