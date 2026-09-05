# CP-ANOVA-05 blocker — common-location numerical stability

- Fecha: `2026-09-05`
- Audit SHA: `9335fcfccffda0a6786895abba460c4359fc6562`
- Production code SHA bajo prueba: `5a116a4e8672dadd3fe57a51f4186f70d1440afd`
- Host: Quantum
- Resultado de suite: `37 passed, 1 failed, 8 warnings in 2.00s`
- Estado: `BLOCKER`, reabre CP-ANOVA-04

## Fallo reproducible

Escenario `unbalanced_k4` con offset común `1e12`:

```text
baseline Classical F   = 3.185834368134492
translated Classical F = 3.1857834745060285
```

La diferencia excede la tolerancia preregistrada de location invariance.

El ULP de float64 alrededor de `1e12` es aproximadamente `1.220703125e-4`. Los datos concretos del escenario siguen siendo representables con resolución suficiente, y `scipy.stats.f_oneway` Classical conserva el F baseline cuando recibe los datos raw desplazados. Por tanto no corresponde adjudicar el fallo simplemente como imposibilidad de representación del input.

## Causa raíz

El candidate congelado calcula por grupo:

```python
mean = np.mean(group)
variance = np.var(group, ddof=1)
```

y Classical/Welch luego operan sobre diferencias entre medias absolutas.

Para datos con gran location común y dispersión pequeña/moderada, `np.mean(group)` pierde bits de baja significancia al acumular números del orden de `1e12`. Posteriormente calcular `mean_i - grand_mean` resta números grandes y cercanos y amplifica esa pérdida para la señal entre grupos.

En el escenario reproducible, por ejemplo, el tercer grupo tiene media baseline `5.214285714285714`, mientras `np.mean(group + 1e12) - 1e12` queda aproximadamente en `5.2142333984375`.

La varianza de ese mismo grupo también presenta una desviación pequeña con cálculo directo sobre la location grande, mientras calcularla después de restar un origen local recupera la varianza baseline.

## Adjudicación de oráculos

- SciPy Classical raw-data es estable en este caso porque centra conjuntamente los datos antes de las sumas de cuadrados.
- Las fórmulas summary-based que reciben únicamente medias absolutas float64 pueden reproducir la misma pérdida del candidate; no son oráculos independientes suficientes para este edge case.
- Welch raw implementations basadas en medias absolutas pueden presentar la misma deuda de location; para adjudicar Welch debe usarse una representación matemáticamente equivalente centrada antes de generar summaries.

## Remediación arquitectónica requerida

No usar `np.longdouble` como solución principal: su precisión/plataforma no es un contrato portable suficiente.

Mantener arquitectura cacheable por grupo mediante summaries localmente centrados:

```text
origin_i       = one observed value from group i
mean_offset_i  = mean(group_i - origin_i)
variance_i     = var(group_i - origin_i, ddof=1)
ss_within_i    = (n_i - 1) * variance_i
```

Para combinar grupos, elegir un `reference_origin` entre los summaries y construir:

```text
relative_mean_i = (origin_i - reference_origin) + mean_offset_i
```

Classical y Welch calculan F/df/p a partir de `relative_mean_i`, no de medias absolutas.

Los outputs descriptivos `group_means`, `grand_mean` y `weighted_mean` pueden reconstruirse en coordenadas absolutas para API, entendiendo que su representación float64 puede redondearse; el estadístico no debe depender de esa reconstrucción.

Esta estructura sigue siendo:

- O(N) para construir summaries;
- O(k) por evaluación;
- cacheable por grupo para el futuro optimizador;
- independiente de un origen global fijado al crear el summary.

## Gobernanza

Por stop condition de CP-ANOVA-05:

- CP-ANOVA-05 queda `blocked/remediation_required`;
- CP-ANOVA-04 se reabre;
- el freeze SHA `5a116a4e...` queda como `superseded_candidate`, no como candidate final;
- la remediación debe vivir en una rama técnica separada;
- después de la corrección se repiten CP-ANOVA-04 smoke/regression y luego toda CP-ANOVA-05 desde el nuevo SHA.
