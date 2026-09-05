# CP-ANOVA-05 — Deterministic / oracle validation PASS

- Fecha: `2026-09-05`
- Rama: `fix/anova-location-stability`
- Candidate ejecutado: `f5200fc1a191f24d9f48a998eeae8abb46107817`
- Production-code remediation: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- CP-ANOVA-04 freeze lógico: `83bfe547563d977f1ed0dd0f43629c281744488c`
- Host: `quantum`
- Entorno: Python 3.12.3, pytest 9.1.1; SciPy moderno con `f_oneway(..., equal_var=False)` disponible.
- Ejecución: single-thread BLAS/OpenMP, `nice -n 10`.

## Resultado

```text
collected 41 items
41 passed, 8 warnings in 1.93s
```

## Cobertura validada

La suite PASS cubre:

- Classical vs fórmula independiente;
- Classical vs SciPy;
- Classical vs statsmodels raw y summary-based;
- Welch vs fórmula independiente;
- Welch vs statsmodels raw y summary-based;
- Welch vs SciPy moderno en dominio ordinario;
- `k=2`: Classical = pooled Student `t²`;
- `k=2`: Welch = Welch `t²` y df Satterthwaite;
- `n_i=2`;
- `k=12`;
- tamaños desbalanceados;
- asociación adversarial tamaño-varianza;
- grupos casi degenerados pero aceptados por DQA;
- offset común `1e12`;
- escalas comunes `1e-100`, `1e100` y signos negativos;
- permutaciones de grupos y observaciones;
- outputs F/p/df finitos y acotados en el dominio probado.

## Adjudicación del oracle Welch bajo gran offset

Se confirmó que el path SciPy Welch `f_oneway(..., equal_var=False)` calcula medias y varianzas por grupo directamente sobre las muestras absolutas y puede perder estabilidad numérica ante una gran location común. Para el escenario `+1e12`, el oracle SciPy se usa sobre datos centrados por un único origen común, una transformación matemáticamente equivalente para ANOVA. El candidate remediado preserva F/p/df respecto a su baseline sin requerir relajar tolerancias.

## Warnings

Los 8 warnings observados provienen de SciPy en escenarios deliberadamente extremos (`n=2`, shape diagnostics y escalas extremas). No produjeron NaN/Inf en los outputs ANOVA del candidate ni discrepancias de oráculos. Se registran como comportamiento de dependencias/diagnósticos, no como fallo del engine ANOVA.

## Interpretación

CP-ANOVA-05 queda **complete/frozen**.

PASS significa corrección matemática y estabilidad numérica determinista dentro del dominio probado. No autoriza todavía robustez poblacional, control de error tipo I, potencia ni selección automática Classical/Welch.

El siguiente checkpoint es **CP-ANOVA-06 — calibration preregistration**. Debe congelar matriz de escenarios, seeds, replicaciones/precisión, métricas primarias, criterios de decisión y holdout antes de cualquier corrida Monte Carlo.
