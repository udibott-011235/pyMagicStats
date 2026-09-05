# CP-ANOVA-04 — Remediated directed regression and freeze

- Fecha: `2026-09-05`
- Rama: `fix/anova-location-stability`
- Candidate SHA ejecutado: `83bfe547563d977f1ed0dd0f43629c281744488c`
- Production-code remediation commit: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- Host: `quantum`
- Entorno: Python 3.12.3, pytest 9.1.1
- Ejecución: single-thread BLAS/OpenMP, `nice -n 10`

## Alcance de regresión

```text
tests/test_anova_production.py
tests/test_anova_location_stability.py
tests/test_assumptions.py
tests/test_inference_selector.py
tests/test_inference_capabilities.py
```

## Resultado

```text
collected 55 items
55 passed in 1.69s
```

## Freeze

CP-ANOVA-04 vuelve a estado **complete/frozen** sobre el candidate exacto:

```text
83bfe547563d977f1ed0dd0f43629c281744488c
```

El candidate anterior `5a116a4e8672dadd3fe57a51f4186f70d1440afd` permanece como `superseded_candidate` por la deuda de estabilidad ante location común grande detectada por CP-ANOVA-05.

El freeze actual incluye:

- producción Classical/Welch explícita;
- summaries localmente centrados para estabilidad numérica;
- preservación de API y method versions;
- selector ONE_WAY aún `NOT_CALIBRATED`;
- sin cambios en optimizer/orchestrator;
- sin Monte Carlo/calibración;
- smoke específico location: `21/21 PASS`;
- regresión dirigida completa: `55/55 PASS`.

Cualquier cambio posterior en `pyMagicStat/inference/anova.py`, exports públicos o tests contractuales de producción reabre CP-ANOVA-04 y exige nuevo freeze SHA.

## Siguiente checkpoint

Reanudar **CP-ANOVA-05 — deterministic/oracle validation** desde este candidate remediado, conservando el preregistro original y la adjudicación documentada del oracle Welch de SciPy bajo offsets grandes.
