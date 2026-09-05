# CP-ANOVA-04 — Directed regression evidence

- Fecha: `2026-09-05`
- Candidate code SHA validado: `5a116a4e8672dadd3fe57a51f4186f70d1440afd`
- Rama: `feature/anova-production-candidate`
- Host: `quantum`
- Entorno: Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1, statsmodels 0.15.0, pytest 9.1.1
- Ejecución: un solo hilo por librería BLAS/OpenMP, baja prioridad (`nice -n 10`)

## Alcance de la regresión

```text
tests/test_anova_production.py
tests/test_assumptions.py
tests/test_inference_selector.py
tests/test_inference_capabilities.py
```

## Resultado

```text
collected 52 items
52 passed in 1.71s
```

## Qué valida esta regresión

Además del PASS unitario de ANOVA, esta corrida confirma que el candidate no introdujo regresiones observables en:

- DataQualityAssessment y diagnósticos de supuestos;
- residuos/diseño one-way existentes;
- MethodSelector legacy/v2;
- routing v3 por capabilities;
- semántica Welch como default de dos muestras donde ya correspondía;
- ausencia de reglas artificiales de tamaño mínimo;
- estado ONE_WAY sin selección automática calibrada.

## Freeze de CP-ANOVA-04

Con:

1. smoke unitario `18/18 PASS`;
2. regresión dirigida `52/52 PASS`;
3. candidate code SHA exacto reproducible;
4. sin modificación de `main`;
5. sin cambio del selector ONE_WAY;
6. sin ejecución Monte Carlo;

se considera **CP-ANOVA-04 complete/frozen** sobre:

```text
5a116a4e8672dadd3fe57a51f4186f70d1440afd
```

Los commits documentales posteriores no cambian el SHA del código auditado. Cualquier modificación futura de `pyMagicStat/inference/anova.py`, sus exports o tests del candidate reabre CP-ANOVA-04 y requiere un nuevo SHA de freeze.

## Siguiente checkpoint

`CP-ANOVA-05 — deterministic/oracle validation`

Debe ampliar validación contra oráculos independientes y casos adversariales numéricos sin iniciar todavía la calibración Monte Carlo de CP-ANOVA-06/07.
