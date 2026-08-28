# Calibración ANOVA (`anova-v1-2026-08`)

## Alcance y reproducibilidad

Esta política cubre exclusivamente la prueba global de igualdad de medias para
`k >= 3` grupos independientes. No calibra medidas repetidas, bloques,
regresión, comparaciones post-hoc ni pruebas de rangos.

El runner versionado es
[`experiments/anova_calibration.py`](../experiments/anova_calibration.py). La
corrida final se ejecutó así:

```bash
python -m experiments.anova_calibration \
  --replications 100 \
  --nominal-sizes 10 25 60 \
  --effect-sizes 0 0.8 \
  --seeds 20260827 20260828 20260829 \
  --output-dir experiments/results
```

Son 15 escenarios × 3 tamaños × 2 hipótesis × 3 seeds × 100 réplicas =
27 000 datasets. Cada réplica usa un `SeedSequence` derivado de seed, escenario,
tamaño, hipótesis y número de réplica. El
[CSV por celda](../experiments/results/anova_calibration_summary.csv) y la
[metadata](../experiments/results/anova_calibration_metadata.json) se conservan
en Git.

Cada fila registra error tipo I o potencia de Classical y Welch, decisión de
`OneWayRobustness`, frecuencia Welch/`INSUFFICIENT`, elegibilidad Classical,
falsos `ACCEPTABLE`, conservadurismo y diagnósticos observados. Las tres seeds se
mantienen separadas para auditar sensibilidad Monte Carlo.

## Matriz

- Normal homocedástica balanceada, desbalanceada y con cinco grupos.
- Normal heterocedástica balanceada.
- Desbalance peligroso con grupo pequeño/varianza grande.
- Asociación opuesta, grupo grande/varianza grande.
- Gamma moderada y exponencial severa.
- Lognormal moderada y severa.
- Student-t con 3 df y Laplace.
- Mezcla simétrica de colas, mezcla sesgada y contaminación positiva de 5%.
- Tamaños nominales 10, 25 y 60; los perfiles desbalanceados son
  `0.5n, n, 2n`.
- H0 y H1 con separación máxima estandarizada de medias de 0.8.

## Resultados principales

Promediando las 135 filas H0 (cada una conserva su seed):

| Métrica | Resultado |
|---|---:|
| Error tipo I Welch, sin selección | 0.0477 |
| Error tipo I Classical, sin selección | 0.0575 |
| Selección Welch por la política | 0.5313 |
| `ACCEPTABLE` | 0.1264 |
| `CAUTION` | 0.4048 |
| `INSUFFICIENT` | 0.4687 |

La comparación decisiva fue el desbalance heterocedástico:

| Escenario H0 | Tamaños | Classical | Welch | Selección Welch |
|---|---|---:|---:|---:|
| Grupo pequeño, varianza grande | 5/10/20 | 0.253 | 0.053 | 0.000 |
| Grupo pequeño, varianza grande | 12/25/50 | 0.257 | 0.050 | 0.923 |
| Grupo pequeño, varianza grande | 30/60/120 | 0.223 | 0.063 | 1.000 |
| Grupo grande, varianza grande | 5/10/20 | 0.007 | 0.063 | 0.000 |
| Grupo grande, varianza grande | 12/25/50 | 0.007 | 0.023 | 0.887 |
| Grupo grande, varianza grande | 30/60/120 | 0.020 | 0.070 | 1.000 |

Classical puede ser muy liberal o extremadamente conservador según el signo de
la asociación tamaño-varianza. Welch mantuvo la tasa mucho más cerca del nivel
nominal. Esto, junto con su pérdida pequeña de potencia bajo normalidad y
homocedasticidad, respalda `welch_anova` como default calibrado.

Para H1 normal homocedástica balanceada, la potencia Welch/Classical fue
0.277/0.280 en n=10, 0.717/0.733 en n=25 y 0.983/0.983 en n=60. En cinco grupos
fue 0.280/0.293, 0.733/0.740 y 0.980/0.977. La mayor diferencia observada en el
subconjunto homocedástico fue el perfil desbalanceado n=25: 0.643 vs 0.683.

## Política resultante

La versión `anova-v1-2026-08` no copia los thresholds de la política de una
media. Evalúa primero constraints estructurales y después rutas específicas:

- mínimo absoluto de 8 observaciones por grupo;
- `ACCEPTABLE` sólo con diagnósticos por grupo y residuo estandarizado
  compatibles, sin extremos y `n_min >= 15`;
- ruta cautelosa pequeña: |skew|≤1.25, |kurtosis excedente|≤4 y máximo 10% de
  extremos detectados;
- ruta moderada: `n_min >= 25`, `N >= 75`, |skew|≤1.75,
  |kurtosis|≤6 y extremos≤8%;
- ruta grande: `n_min >= 50`, `N >= 150`, |skew|≤3,
  |kurtosis|≤15 y extremos≤10%;
- un extremo dentro de grupo con modified-z≥8 activa un guardrail
  `INSUFFICIENT` antes de cualquier relajación por shape;
- `equal_var=None` selecciona Welch; Classical requiere `equal_var=True` y
  evidencia compatible en magnitud, Brown-Forsythe/Levene, Fligner, razón de
  varianzas y alineación tamaño-varianza.

Un `p > .05` no demuestra homocedasticidad y nunca basta para seleccionar
Classical.

## Escenarios severos y abstención

La política fue deliberadamente conservadora en zonas no respaldadas:

| Escenario H0 | n nominal | `INSUFFICIENT` |
|---|---:|---:|
| Lognormal severa | 10 | 0.967 |
| Lognormal severa | 25 | 0.993 |
| Lognormal severa | 60 | 0.993 |
| Contaminación 5% | 10 | 0.847 |
| Contaminación 5% | 25 | 0.987 |
| Contaminación 5% | 60 | 0.997 |
| Exponencial | 10 | 0.820 |
| Exponencial | 25 | 0.787 |
| Exponencial | 60 | 0.417 |

El guardrail de influencia redujo la selección H0 global de 0.552 a 0.531 sin
cambiar ninguna tasa no seleccionada de Classical o Welch.

## Falsos `ACCEPTABLE`, conservadurismo y seed

La definición operativa marca una fila seed como falso `ACCEPTABLE` si su tasa
condicional queda fuera de 2.5%–7.5%. Hubo 33/135 filas H0 marcadas; muchas
tenían muy pocas réplicas `ACCEPTABLE`, por lo que el cociente es discreto e
inestable. Al agregar las tres seeds y exigir al menos 10% de `ACCEPTABLE`, sólo
dos celdas quedaron apenas por encima del límite: normal balanceada n=25
(7.53%) y normal con grupo grande/varianza grande n=60 (7.65%). Con 300
réplicas, ambas diferencias están dentro de incertidumbre Monte Carlo relevante
y requieren revisión independiente, no una afirmación de garantía.

Hubo 20/135 filas seed marcadas como excesivamente conservadoras. Se concentran
en colas/mixtures con selección escasa o en celdas donde la selección condiciona
fuertemente el denominador. Este costo es visible en `insufficient_rate`; no se
oculta elevando la tasa incondicional de rechazo.

Las tasas Welch por seed variaron típicamente en incrementos de un punto
porcentual y, por usar 100 réplicas, rangos como 3%–9% son compatibles con ruido
Monte Carlo. La política debe reevaluarse con más réplicas y nuevas seeds si una
auditoría propone mover thresholds.

## Limitaciones y revisión

- Seleccionar y contrastar con los mismos datos no otorga control condicional
  exacto. Algunas tasas condicionales en subconjuntos pequeños se alejaron del
  nominal aunque la tasa Welch no seleccionada estuviera controlada.
- La matriz usa tres y cinco grupos; más grupos requieren ampliación.
- La potencia usa una sola configuración global de effect size; no sustituye un
  estudio de contrastes específicos.
- No hay calibración para dependencia, clusters, repeated measures o post-hoc.
- La política está implementada, pero no se considera aprobada para merge hasta
  una revisión adversarial independiente de runner, resultados y thresholds.

