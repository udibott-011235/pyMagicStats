# Calibración de `SamplingRobustness` (`mean-v2.1-2026-08`)

## Objetivo y reproducibilidad

Esta calibración evalúa la inferencia t de una media; no valida ANOVA ni el
intervalo chi-square de una varianza. El runner versionado es
[`experiments/robustness_calibration.py`](../experiments/robustness_calibration.py)
y se ejecutó así:

```bash
python -m experiments.robustness_calibration \
  --replications 1000 \
  --sample-sizes 10 20 30 40 50 80 100 200 \
  --seed 20260826 \
  --output-dir experiments/results
```

Son 19 escenarios × 8 tamaños × 1 000 réplicas = 152 000 muestras. Cada celda
usa un flujo independiente derivado de `numpy.random.SeedSequence`. El
[resumen agregado](../experiments/results/sampling_robustness_summary.csv) y los
[metadatos](../experiments/results/sampling_robustness_metadata.json) se guardan
en Git. El CSV comprimido por réplica se genera localmente y se excluye para no
añadir varios megabytes reproducibles al repositorio.

Con 1 000 réplicas, el error estándar Monte Carlo de una tasa nominal de 5% es
0.69 puntos porcentuales. Se usó 3.5%–6.5% como banda práctica de auditoría; no
es una prueba universal de validez.

## Matriz de distribuciones

- Normal y Laplace centradas.
- Exponencial centrada.
- Gamma con shape 2, 4 y 9.
- Lognormal con sigma 0.25, 0.5, 1.0 y 1.25, restando su media teórica.
- Student-t con 3, 5, 10 y 30 grados de libertad.
- Mezcla simétrica: 95% N(0,1) + 5% N(0,8).
- Mezcla sesgada: 90% N(-0.5,1) + 10% N(4.5,1), centrada por su media.
- Contaminación positiva de 1%, 5% y 10% con N(10,1) sobre N(0,1), centrada
  por la media poblacional exacta de la mezcla.

En cada réplica se registran skewness, kurtosis excedente, fracción de outliers
detectada por MAD/IQR, estado de `ShapeAssessment`, decisión real de
`SamplingRobustness`, cobertura del CI t y rechazo bilateral de H0 en la prueba
t de una muestra. Para esta prueba bilateral, cobertura y error tipo I son
complementarios; se conservan ambos campos para hacer explícitos los dos
contratos solicitados.

## Resultados que determinan los umbrales

La tabla muestra celdas representativas. Los valores son proporciones sobre
1 000 réplicas.

| Escenario | n | Cobertura | Error tipo I | Mediana abs. skew | Mediana abs. kurt. exc. |
|---|---:|---:|---:|---:|---:|
| Normal | 10 | 0.952 | 0.048 | 0.43 | 0.80 |
| Exponencial | 40 | 0.927 | 0.073 | 1.52 | 2.23 |
| Exponencial | 80 | 0.944 | 0.056 | 1.71 | 3.28 |
| Gamma shape 4 | 40 | 0.947 | 0.053 | 0.80 | 0.66 |
| Lognormal sigma 0.5 | 40 | 0.946 | 0.054 | 1.17 | 1.34 |
| Lognormal sigma 0.5 | 80 | 0.949 | 0.051 | 1.35 | 2.32 |
| Lognormal sigma 1.0 | 80 | 0.912 | 0.088 | 2.75 | 9.39 |
| Lognormal sigma 1.0 | 200 | 0.938 | 0.062 | 3.37 | 15.58 |
| Lognormal sigma 1.25 | 200 | 0.896 | 0.104 | 4.33 | 24.64 |
| Student-t df 3 | 40 | 0.960 | 0.040 | 0.67 | 2.19 |
| Student-t df 3 | 200 | 0.952 | 0.048 | 0.69 | 5.86 |
| Mezcla simétrica 5% | 200 | 0.952 | 0.048 | 1.92 | 22.52 |
| Mezcla sesgada 10% | 80 | 0.951 | 0.049 | 1.50 | 2.44 |
| Outliers positivos 5% | 80 | 0.918 | 0.082 | 3.12 | 11.12 |
| Outliers positivos 5% | 200 | 0.945 | 0.055 | 3.10 | 10.46 |

Conclusiones de política:

1. Se conservan n≥40, |skew|≤1 y |kurtosis excedente|≤3 como escalón
   moderado. Gamma shape 4 y lognormal sigma 0.5 apoyan el corte; exponencial
   queda fuera por skew y todavía exhibe 7.3% de error.
2. Se conservan n≥80, |skew|≤2 y |kurtosis excedente|≤7 como escalón grande.
   Exponencial, gamma y lognormal sigma 0.5 entran en una zona razonable; la
   lognormal sigma 1.0 excede ambos límites y permanece inflada.
3. Los outliers se expresan como fracción, no como veto binario: máximo 2.5%
   en el escalón n≥40 y 10% en n≥80. La mezcla sesgada al 10% tuvo 4.9% de
   error en n=80 cuando las métricas de forma quedaron acotadas.
4. Para n≥200 se añade una ruta exclusivamente `caution`, con |skew|≤2,
   |kurtosis excedente|≤25 y outliers detectados≤10%. Está respaldada por t(3)
   y la mezcla simétrica; no autoriza las lognormales severamente asimétricas.
5. Toda aceptación basada en aproximación asintótica se etiqueta `caution`.
   `acceptable` queda reservado a forma directamente compatible y ausencia de
   extremos detectados.
6. La revisión v2.1 aplica los constraints antes de cualquier alivio por shape:
   una muestra con outliers sólo puede recibir `caution` si satisface uno de los
   escalones calibrados. En particular, n=12 con 1/12 outliers es
   `insufficient`, aunque el diagnóstico de shape sea `pass`.

La repetición completa de v2.1 mantuvo idénticas las 152 tasas de cobertura,
error tipo I y métricas diagnósticas de v2. Sólo cambió la decisión: 77 celdas
aumentaron su tasa `insufficient` (máximo +0.19). Las proporciones globales
quedaron en 0.3180 `acceptable`, 0.2220 `caution` y 0.4600 `insufficient`.

## Por qué `ShapeAssessment.FAIL` no decide solo

El baseline anterior (`b757fde`) convertía cualquier `FAIL` en
`insufficient`. La matriz mostró contraejemplos directos:

- mezcla simétrica, n=200: `FAIL` en 97.9% de réplicas y error tipo I de 4.8%;
- Student-t df 3, n=200: `FAIL` en 44.3% y error tipo I de 4.8%;
- lognormal sigma 1.25, n=200: `FAIL` en 99.7% y error tipo I de 10.4%.

Por tanto, `FAIL` describe forma severa pero no determina robustez sin n,
asimetría, kurtosis y outliers. La política final puede emitir `caution` ante un
`FAIL` de colas pesadas en n≥200, mientras conserva `insufficient` para la
asimetría severa. El test automatizado con una muestra t(3) fija este contrato.

## Alcance y límites

- La matriz calibra únicamente inferencia t bilateral de una media con muestras
  independientes; no cubre potencia, pruebas unilaterales, dependencia ni
  diseños agrupados.
- Diagnosticar y seleccionar con la misma muestra no crea una garantía formal
  de error condicional. En especial, una contaminación poblacional rara puede
  no aparecer en una muestra; ninguna regla basada sólo en datos observados
  puede detectar esa cola ausente.
- Las métricas muestrales son ruidosas. Los cortes son una política conservadora
  y versionada, no teoremas ni fronteras universales.
- No se implementó ANOVA. La calibración debe ampliarse antes de reutilizarse
  para decisiones de múltiples grupos.
