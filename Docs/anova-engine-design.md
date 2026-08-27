# Diseño del motor ANOVA de una vía

## Estado y objetivo

Este documento define la integración de inferencia one-way en el motor existente.
Es un contrato previo a la implementación. Mientras no estén completos los
estadísticos, tests y calibración, `MethodSelector` debe conservar
`status="not_calibrated"` y `selected_method=None` para
`InferenceDesign.ONE_WAY`.

El alcance inicial son `k >= 3` grupos independientes. Se implementarán ANOVA
clásico y ANOVA de Welch, sin pruebas post-hoc. Tukey, Games-Howell y otros
procedimientos de comparaciones múltiples pertenecen a una fase posterior.

## Preguntas estadísticas y estimands

El estimand primario es el vector de medias poblacionales y la hipótesis global:

```text
H0: mu_1 = mu_2 = ... = mu_k
H1: al menos una media difiere
```

Tanto ANOVA clásico como Welch preservan esa pregunta global sobre medias. No
identifican qué grupos difieren y un rechazo no autoriza comparaciones por pares
sin control adicional de multiplicidad.

Las alternativas deben etiquetar su pregunta distinta:

- una prueba de permutación sobre un estadístico de medias conserva la pregunta
  bajo una hipótesis de intercambiabilidad que debe justificarse;
- bootstrap de contrastes estima el contraste de medias especificado y requiere
  remuestreo dentro de cada grupo;
- Kruskal-Wallis contrasta diferencias en distribuciones de rangos y sólo admite
  una interpretación simple de location shift bajo condiciones adicionales. No
  es «ANOVA no paramétrico de medias» ni un reemplazo automático.

## Pipeline arquitectónico

El flujo reutiliza las cuatro capas del inference engine:

1. `InferenceValidator.validate_one_way` normaliza grupos y crea un
   `AssumptionReport`.
2. Una política `OneWayRobustness` interpreta exclusivamente reportes ONE_WAY.
   Tendrá versión independiente de `SamplingRobustness`; no heredará sin
   evidencia los umbrales `mean-v2.1-2026-08`.
3. `MethodSelector` decide entre `classical_anova`, `welch_anova` o ningún
   método. La selección sólo se habilita después de calibración.
4. `OneWayANOVA` y `WelchANOVA` ejecutan el estadístico solicitado y adjuntan el
   reporte y la decisión; no vuelven a implementar diagnósticos.

No se transforman datos ni se ejecutan bootstrap, permutaciones o métodos de
rangos de manera implícita.

## Contrato de `validate_one_way`

### Entrada y calidad por grupo

La API permanece alineada con el validator actual:

```python
validation = InferenceValidator(alpha=0.05).validate_one_way(
    group_a,
    group_b,
    group_c,
    independence="assumed",
)
```

Debe exigir al menos tres grupos. Cada grupo se convierte a un vector float64 y
registra:

- `n`, dimensionalidad y finitud;
- datos faltantes o infinitos;
- varianza muestral y tolerancia numérica;
- varianza cero o numéricamente despreciable;
- número de valores distintos.

Una falla de calidad es estructural y se rechaza antes de pruebas de forma o
varianza. El validator puede admitir matemáticamente `n >= 2`, pero la política
de robustez debe tratar tamaños muy pequeños como evidencia insuficiente según
la calibración. La deuda técnica de escalas subnormales permanece separada en
`Docs/technical-debt.md`.

### Independencia

`independence` conserva `unknown`, `assumed` y `verified`. Es metadato del
diseño: no se deduce a partir de correlaciones, orden o valores observados.
Medidas repetidas, bloques, clusters y pares quedan fuera de este contrato.

### Forma y residuos

No se evalúan todos los valores pooled como si procedieran de una población
común. Diferencias genuinas entre medias no deben parecer falta de normalidad.

Para cada grupo `i` se calculan residuos centrados
`e_ij = y_ij - mean(y_i)` y se conservan como `relevant_samples`. El reporte
incluye diagnóstico de forma por grupo. También se construye un diagnóstico
agregado sobre residuos estandarizados dentro de grupo, no sobre observaciones
pooled. Ese agregado resume evidencia, pero nunca sustituye los diagnósticos por
grupo ni funciona como veto aislado.

### Outliers e influencia

Los extremos se buscan dentro de cada grupo y sobre residuos estandarizados. Se
registran conteo y fracción por grupo, máximo entre grupos y total. Así una media
de grupo legítimamente distinta no se etiqueta como outlier.

La primera versión no intentará estimar Cook's distance ni leverage de un modelo
general. Los casos con uno o pocos puntos que dominan una media deben producir
`INSUFFICIENT` o `CAUTION` según evidencia de simulación, no sólo por el p-value
de una prueba de forma.

### Varianzas y balance

El assessment de varianza informa, sin convertir `p > alpha` en prueba de
igualdad:

- varianzas y tamaños por grupo;
- razón máximo/mínimo de varianzas;
- razón máximo/mínimo de tamaños;
- asociación entre tamaño y varianza, porque grupos pequeños con varianza grande
  constituyen una combinación especialmente peligrosa para ANOVA clásico;
- Brown-Forsythe/Levene centrado en mediana;
- Fligner-Killeen como diagnóstico robusto;
- Bartlett sólo como evidencia secundaria cuando la forma compatible con
  normalidad sea razonablemente defendible.

Los p-values son diagnósticos junto con razones de magnitud y balance. Ninguno
selecciona por sí solo Classical o Welch.

## Métodos candidatos

### ANOVA clásico

Usa sumas de cuadrados entre y dentro de grupos:

```text
SS_between = sum_i n_i (mean_i - grand_mean)^2
SS_within  = sum_i sum_j (y_ij - mean_i)^2
F = [SS_between / (k - 1)] / [SS_within / (N - k)]
```

Sus grados de libertad son `k - 1` y `N - k`. El contrato exige grupos
independientes, varianza común razonablemente defendible y error dentro de grupo
compatible con la robustez calibrada. `equal_var=True` será la solicitud
explícita del usuario si se conserva este parámetro en el selector.

### ANOVA de Welch

Para `w_i = n_i / s_i^2`, `W = sum_i w_i` y media ponderada
`mean_w = sum_i w_i mean_i / W`:

```text
A = sum_i w_i (mean_i - mean_w)^2 / (k - 1)
B = sum_i [(1 - w_i/W)^2 / (n_i - 1)]
F_W = A / [1 + 2(k - 2)B/(k^2 - 1)]
df1 = k - 1
df2 = (k^2 - 1)/(3B)
```

La implementación se contrastará con `statsmodels.stats.oneway.anova_oneway`
usando corrección de Welch. Welch es candidato serio a default por su robustez
ante heterocedasticidad, pero no se fijará sólo por analogía con la prueba t. La
matriz calibrará control de tipo I y pérdida de potencia en condiciones
homocedásticas antes de activar esa política.

## Robustez y selección

`OneWayRobustness` devuelve los niveles existentes `ACCEPTABLE`, `CAUTION` e
`INSUFFICIENT`, acompañados de razones. Sus constraints duros —calidad,
degeneración, tamaños mínimos y límites de influencia que resulten calibrados—
se evalúan antes de cualquier relajación por shape.

La política candidata del selector es:

- falla estructural o robustez insuficiente: ningún método;
- robustez suficiente y `equal_var=True`: ANOVA clásico solicitado
  explícitamente, siempre que la política no lo desaconseje de forma dura;
- robustez suficiente y `equal_var=False`: Welch explícito;
- `equal_var=None`: default decidido por calibración. Welch sólo se convertirá
  en default si controla tipo I sin conservadurismo/pérdida de potencia
  inaceptable en la matriz homocedástica;
- mientras falten implementación, tests o evidencia: `NOT_CALIBRATED`.

`InferenceDecision.parametric_recommended` se extenderá para reconocer sólo los
identificadores realmente implementados. Las alternativas one-way conservarán
su estimand explícito.

## API propuesta y resultados

Ejecución explícita:

```python
classical = OneWayANOVA(
    group_a,
    group_b,
    group_c,
    independence="assumed",
    strict=True,
).run_test()

welch = WelchANOVA(
    group_a,
    group_b,
    group_c,
    independence="assumed",
    strict=True,
).run_test()
```

Ambas clases devuelven al menos:

- `method`, `statistic`, `p_value`, `alpha`, `reject_null` y `txt`;
- grados de libertad numerador/denominador;
- `k`, `n_total`, tamaños, medias y varianzas de grupos;
- `assumptions` e `inference_decision` serializados;
- política de selección solicitada cuando corresponda.

No devuelven resultados post-hoc. La ejecución directa de Classical solicita al
selector `equal_var=True`; Welch solicita `equal_var=False`. Con `strict=True`
se rechaza una decisión sin método compatible. `strict=False` permite auditar el
estadístico solicitado con warning y diagnóstico, sin convertirlo en método
recomendado.

## Backward compatibility

No existe una clase ANOVA paramétrica pública previa que conservar. Se mantienen
sin cambios las APIs auditadas de una muestra, pareada, dos muestras, bootstrap
y varianza poblacional. `validate_one_way` ya existe: se amplía su reporte sin
eliminar las claves actuales `data_quality_group_i`, `shape_group_i`,
`outliers_group_i`, `variance` e `independence`.

`kruskalWallisTest` permanece disponible con su contrato histórico, pero no se
invoca automáticamente ni se presenta como prueba de igualdad de medias.

## Estrategia de testing

### Unitarios y referencias

- rechazo de menos de tres grupos, datos no finitos, grupos degenerados y n
  insuficiente;
- residuos centrados por grupo y agregado estandarizado sin pooling de medias;
- outliers detectados dentro de grupo;
- métricas Brown-Forsythe/Levene, Fligner, Bartlett, balance y razón de varianza;
- fórmulas Classical frente a `scipy.stats.f_oneway(equal_var=True)` cuando la
  versión instalada lo permita, con fórmula manual como referencia estable;
- Welch frente a `statsmodels.stats.oneway.anova_oneway(use_var="unequal")`;
- invariancia a traslación común, permutación de observaciones y orden de grupos;
- resultados deterministas y serializables;
- selector incapaz de recomendar identificadores no implementados;
- contratos `strict`, independencia y alternativas/estimands.

### Adversariales

- tamaños mínimos y un grupo casi degenerado;
- outlier único dominante con test de normalidad que no rechaza;
- varianza grande en el grupo pequeño y varianza pequeña en el grupo grande;
- medias muy separadas con residuos normales, para impedir diagnóstico pooled;
- colas pesadas simétricas frente a sesgo severo;
- escalas y offsets grandes dentro de float64 ordinario;
- permutaciones de labels y orden de grupos;
- protección contra selección Classical por un único `p > alpha`.

## Estrategia de calibración

El runner será reproducible, versionado y separado de la calibración de medias.
Usará `SeedSequence`, metadata completa y un CSV resumen versionado. Incluirá:

- normal con varianzas iguales y distintas;
- diseños balanceados y desbalanceados;
- asociación positiva y negativa entre `n_i` y `s_i^2`, incluida la combinación
  peligrosa de grupo pequeño/varianza grande;
- skewness moderada y severa;
- colas pesadas;
- outliers y mezclas;
- tamaños pequeños, moderados y grandes;
- H0 verdadera y H1 con varios effect sizes;
- varias seeds de auditoría.

Por celda se medirán error tipo I o potencia, frecuencia de selección Classical,
Welch e `INSUFFICIENT`, falsos `ACCEPTABLE`, conservadurismo y sensibilidad a la
seed. La calibración comparará el método seleccionado con ambos procedimientos,
no sólo su decisión final.

La política recibirá una versión independiente —por ejemplo
`anova-v1-2026-08`— únicamente cuando los resultados justifiquen sus thresholds.
No se copiarán automáticamente `n=40/80`, skewness `1/2` ni kurtosis `3/7` de
la política de una media.

## Riesgos conocidos y revisión requerida

- La selección y el test usan los mismos datos; el control condicional tras
  selección no es un teorema y debe auditarse empíricamente.
- Tests de forma y varianza tienen poca potencia en muestras pequeñas y exceso
  de potencia en muestras grandes.
- Heterocedasticidad, desbalance y no normalidad interactúan; thresholds
  marginales pueden ocultar celdas peligrosas.
- Welch puede perder potencia en algunos escenarios homocedásticos; Classical
  puede inflar tipo I bajo asociaciones adversas de tamaño/varianza.
- Outliers raros no observados no pueden inferirse desde la muestra.
- La inferencia global no resuelve comparaciones múltiples.

La rama no se declarará lista para merge al terminar la implementación. Debe
quedar preparada para revisión adversarial independiente de fórmulas, API,
tests, selector y calibración.

