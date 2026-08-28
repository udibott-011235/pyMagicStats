# Contrato entre `Distribution`, shape e inferencia

## Fuente canónica

`Distribution.data` conserva una copia defensiva, univariada y de solo lectura
del ndarray observado. Cambiar el array fuente no altera el snapshot y una
mutación in-place de `Distribution.data` falla explícitamente. `Distribution`
calcula una sola familia coherente de descriptivos muestrales: varianza y desviación con
`ddof=1`, skewness con `bias=False`, y curtosis excedente con `fisher=True` y
`bias=False`. El nombre público no ambiguo es `excess_kurtosis`.

El contrato de solo lectura se restaura al reconstruir la instancia mediante
pickle o `copy.deepcopy`. `copy.copy` comparte el mismo snapshot, lo cual es
seguro porque el ndarray compartido permanece read-only.

`ShapeAssessment.assess(distribution)` reutiliza los descriptivos de la
instancia. `ShapeAssessment.assess(distribution.data)` sigue siendo compatible y
aplica exactamente las mismas convenciones canónicas.

Las capas de validación e inferencia desenvuelven ese mismo snapshot, por lo que
`InferenceValidator`, `OneSampleTTest` y `TwoSampleTTest` aceptan instancias de
`Distribution` además de los array-like existentes. Los datos vacíos y las
matrices 2D se rechazan con `ValueError`: este contrato representa una sola
muestra univariada y no aplana matrices silenciosamente.

## Dos preguntas distintas

El diagnóstico de shape separa:

1. evidencia de Shapiro-Wilk y D'Agostino contra normalidad gaussiana exacta;
2. magnitud observada mediante skewness y curtosis excedente;
3. estado descriptivo derivado de esa magnitud.

Las métricas `shapiro_rejects_exact_normality`,
`dagostino_rejects_exact_normality` y `exact_normality_rejected` resumen la
primera pregunta. `departure_magnitude` vale `mild`, `moderate`, `severe` o
`not_assessed` y responde la segunda. Los umbrales descriptivos son:

- `severe`: |skewness| > 2 o |curtosis excedente| > 7;
- `moderate`: |skewness| > 1 o |curtosis excedente| > 3;
- `mild`: métricas observadas por debajo de esos límites.

Un p-value menor que alpha no eleva por sí solo la magnitud a `moderate` ni a
`severe`. Por ejemplo, una muestra grande puede rechazar gaussianidad exacta y
mostrar a la vez una desviación observada leve.

## Decisión inferencial

El flujo sigue siendo:

```text
data -> descriptivos/shape -> SamplingRobustness -> MethodSelector -> método
```

`SamplingRobustness` combina tamaño, skewness, curtosis excedente y outliers con
la política calibrada existente. `MethodSelector` consume ese resultado; no
consume `Distribution.type` ni un booleano de normalidad. Por ello un rechazo
formal no bloquea automáticamente `one_sample_t`, `paired_t`, `welch_t` o
`student_t`.

## Compatibilidad y deprecaciones

`Distribution.kurtosis` continúa como alias deprecado de
`Distribution.excess_kurtosis`. `Distribution.type` y `update_type()` son API
legacy para validadores de distribución.

Cuando `NormalDistribution` conserva temporalmente `type["Normal"]`, el valor
significa solamente que las pruebas formales disponibles no rechazaron
normalidad exacta. No autoriza ni prohíbe inferencia paramétrica. El resultado
estructurado canónico queda en `distribution.assessments["normality"]`.

La API pública recomendada es:

```python
from pyMagicStat.distributions import Distribution, NormalDistribution
```

La simulación pequeña y reproducible de este contrato está en
`experiments/shape_contract_simulation.py`. Compara Normal, Student-t con 10 y 5
grados de libertad, y lognormales moderada y severa para n=30, 100 y 750. No
recalibra los umbrales de robustez.
