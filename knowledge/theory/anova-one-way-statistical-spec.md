# STAGE-ANOVA-001 / CP-ANOVA-02 — One-way ANOVA statistical specification

- Fecha de congelación: `2026-09-05`
- Rama: `audit/anova-statistical-closure`
- Base de trabajo: `main@402e4601df460811779b3238c2526ac12f463a67`
- Estado: `frozen statistical specification`
- Autoridad: Product Owner + statistical-software-architecture
- Implementación: pendiente de CP-ANOVA-03/04

## 1. Alcance

Esta especificación cubre inferencia global sobre medias para grupos independientes en un diseño one-way.

Estimando:

```text
vector de medias poblacionales (mu_1, ..., mu_k)
```

Hipótesis global:

```text
H0: mu_1 = mu_2 = ... = mu_k
H1: al menos una media poblacional difiere
```

Métodos Step 1:

1. Classical one-way ANOVA.
2. Welch one-way ANOVA.

Fuera de alcance de este stage:

- post-hoc Tukey/Games-Howell;
- repeated measures;
- bloques, clusters o dependencia intra-grupo;
- factorial ANOVA/interacciones;
- ANCOVA/regresión;
- selección automática Classical/Welch;
- transformación automática;
- fallback automático a Kruskal-Wallis;
- DOE y optimización de experimentos.

La integración futura con `optimization/orchestrator.py` está registrada separadamente y no modifica este scope.

## 2. Dominio del diseño

### 2.1 Número de grupos

Se soporta **`k >= 2`**.

Razones:

- one-way ANOVA está matemáticamente definido para dos o más grupos;
- para `k=2` existen invariantes fuertes contra las pruebas t;
- evita un caso artificial en workflows futuros que reduzcan un conjunto de niveles hasta dos grupos.

Con `k=2`:

```text
F_classical = t_student^2
F_welch     = t_welch^2
```

con igualdad correspondiente de p-values bilaterales y grados de libertad compatibles.

### 2.2 Unidad independiente

La unidad independiente no se deduce de los valores observados.

`independence` continúa siendo metadata del diseño con los estados existentes:

- `unknown`
- `assumed`
- `verified`

Una ejecución numérica puede calcularse para auditoría aun cuando la independencia sea desconocida, pero una afirmación de inferencia validada no puede presentar independencia como satisfecha si está `unknown`.

El comportamiento exacto de `strict=True/False` se congelará en CP-ANOVA-03.

### 2.3 Calidad mínima por grupo

Cada grupo debe ser:

- unidimensional;
- numérico;
- finito;
- `n_i >= 2`;
- con al menos dos valores distintos;
- con varianza muestral positiva y no numéricamente despreciable bajo el contrato vigente de `DataQualityAssessment`.

Aunque Classical ANOVA puede producir un número en algunas configuraciones degeneradas, pyMagicStats rechazará grupos degenerados para mantener un contrato común, interpretable y compatible con Welch.

La deuda de escalas subnormales float64 permanece separada como `TD-NUM-001` y no se convierte en bloqueante de este stage.

## 3. Variable relevante para supuestos

No se probará normalidad sobre observaciones pooled de todos los grupos.

Para cada grupo:

```text
e_ij = y_ij - mean_i
```

Los diagnósticos de forma e influencia se aplican dentro del grupo sobre estos residuos centrados.

Un agregado de residuos estandarizados por grupo puede utilizarse como diagnóstico secundario, pero:

- no sustituye los diagnósticos por grupo;
- no funciona como veto aislado;
- no demuestra Gaussianidad cuando no rechaza normalidad.

La forma observada, Shapiro-Wilk, D'Agostino, skewness, kurtosis y outliers son evidencia diagnóstica; no constituyen una prueba lógica del modelo poblacional.

## 4. Classical one-way ANOVA

### 4.1 Modelo estadístico

La ruta Classical corresponde al modelo de errores independientes con varianza común y, para la distribución F exacta finito-muestral, errores Gaussianos dentro de cada grupo.

La igualdad de varianzas no se inferirá de `p > alpha` en Levene/Brown-Forsythe. Los tests de varianza son diagnostics y pueden aportar evidencia de conflicto, no demostrar igualdad.

La política futura puede utilizar evidencia empírica de robustez fuera del modelo exacto únicamente después de una calibración ANOVA específica.

### 4.2 Fórmulas canónicas

Para `k` grupos, tamaños `n_i`, medias `m_i`, varianzas muestrales `s_i^2` y `N = sum(n_i)`:

```text
grand_mean = sum_i(n_i * m_i) / N

SS_between = sum_i n_i * (m_i - grand_mean)^2
SS_within  = sum_i (n_i - 1) * s_i^2
SS_total   = SS_between + SS_within

df_between = k - 1
df_within  = N - k

MS_between = SS_between / df_between
MS_within  = SS_within / df_within

F = MS_between / MS_within
p = sf_F(F; df_between, df_within)
```

### 4.3 Effect metric descriptiva

Para one-way Classical puede exponerse:

```text
eta_squared = SS_between / SS_total
```

cuando `SS_total > 0`.

`eta_squared` es una métrica descriptiva de tamaño de efecto/varianza explicada. No sustituye el test global ni implica causalidad.

Omega-squared y otras correcciones quedan fuera de Step 1 salvo decisión posterior explícita.

## 5. Welch one-way ANOVA

### 5.1 Modelo estadístico

Welch ANOVA preserva la misma hipótesis global sobre medias sin exigir varianza común.

Su estadístico usa la aproximación Welch-Satterthwaite. No debe etiquetarse como una prueba F exacta finito-muestral equivalente al Classical bajo su modelo.

Cualquier afirmación de robustez frente a no-normalidad, colas pesadas, skewness o tamaños pequeños requiere calibración específica del stage ANOVA.

### 5.2 Fórmulas canónicas

Con:

```text
w_i = n_i / s_i^2
W   = sum_i w_i
m_w = sum_i(w_i * m_i) / W
```

se define:

```text
A = [sum_i w_i * (m_i - m_w)^2] / (k - 1)

B = sum_i [ (1 - w_i / W)^2 / (n_i - 1) ]

correction = 1 + [2 * (k - 2) / (k^2 - 1)] * B

F_W = A / correction

df1 = k - 1
df2 = (k^2 - 1) / (3 * B)

p = sf_F(F_W; df1, df2)
```

Cada `s_i^2` debe ser positiva bajo el contrato de calidad; no se admiten pesos infinitos por grupos degenerados.

### 5.3 Effect size

No se reutilizará `eta_squared` Classical como si fuera un effect size Welch equivalente.

Una métrica de efecto específica para Welch puede añadirse posteriormente con teoría, referencia y validación propias. Esto es especialmente relevante para la futura integración con el optimizador.

## 6. Separación computation vs inference authorization

El motor se diseñará en dos niveles conceptuales:

### A. Kernel estadístico

Calcula de forma determinista los estadísticos a partir de grupos válidos/resúmenes válidos.

No decide por sí mismo si el diseño real justifica la interpretación inferencial.

### B. Capa de validación/autorización

Adjunta:

- diseño;
- estimando;
- independencia;
- diagnostics de residuos;
- diagnostics de varianza;
- status de soporte;
- razones y límites.

La existencia de un F y un p-value computables no se convertirá automáticamente en una recomendación del método.

`MethodSelector` debe permanecer `NOT_CALIBRATED` para `InferenceDesign.ONE_WAY` durante Step 1.

## 7. Arquitectura de cálculo y eficiencia

### 7.1 Resúmenes suficientes por grupo

El kernel Classical y Welch debe depender de resúmenes por grupo:

```text
n_i
mean_i
variance_i (ddof=1)
ss_within_i = (n_i - 1) * variance_i
```

Una ejecución ordinaria:

1. valida/normaliza los datos;
2. calcula cada resumen una sola vez en `O(N)`;
3. calcula Classical o Welch desde esos resúmenes en `O(k)`.

No se requiere concatenar todos los datos para calcular F.

### 7.2 Razón arquitectónica

Esto mejora:

- eficiencia;
- testabilidad de fórmulas;
- reproducibilidad;
- estabilidad de interfaces;
- futura integración con `optimization/orchestrator.py`.

En Step 2, un optimizador que evalúe repetidamente subconjuntos de grupos podrá cachear resúmenes y reevaluar candidatos en `O(k)` por subconjunto en lugar de volver a recorrer todas las observaciones.

Esta optimización futura no autoriza todavía una API pública basada exclusivamente en summary statistics; CP-ANOVA-03 decidirá la interfaz.

## 8. Outputs mínimos requeridos

### 8.1 Campos compartidos

Ambos métodos deben exponer al menos:

- `method`;
- `statistic`;
- `p_value`;
- `alpha`;
- `reject_null`;
- `numerator_df`;
- `denominator_df`;
- `k`;
- `n_total`;
- `group_sizes`;
- `group_means`;
- `group_variances`;
- `design = one_way`;
- `estimand = group_mean_differences`;
- `assumptions` / diagnostics;
- metadata de versión del método/contrato.

### 8.2 Classical adicionales

- `grand_mean`;
- `ss_between`;
- `ss_within`;
- `ss_total`;
- `mean_square_between`;
- `mean_square_within`;
- `eta_squared`.

### 8.3 Welch adicionales

- `weights`;
- `weighted_mean`;
- `welch_correction`;
- término `B` o metadata suficiente para reproducir `df2`.

Los nombres finales de API se congelarán en CP-ANOVA-03, pero la información matemática no podrá eliminarse sin revisar esta especificación.

## 9. Oráculos independientes

### Classical

Oráculos mínimos:

1. fórmula canónica independiente implementada desde summaries;
2. `scipy.stats.f_oneway(*groups)` compatible con el mínimo SciPy declarado por el proyecto;
3. `statsmodels.stats.oneway.anova_oneway(groups, use_var="equal")` como segundo oracle externo cuando corresponda;
4. para `k=2`, `scipy.stats.ttest_ind(..., equal_var=True)` mediante `F = t^2`.

No se usará `equal_var=True` en `scipy.stats.f_oneway` como requisito del test suite mientras `pyproject.toml` permita SciPy <1.16.

### Welch

Oráculos mínimos:

1. fórmula canónica independiente;
2. `statsmodels.stats.oneway.anova_oneway(groups, use_var="unequal", welch_correction=True)`;
3. SciPy `f_oneway(..., equal_var=False)` como oracle secundario sólo cuando SciPy >=1.16 esté disponible;
4. para `k=2`, Welch t mediante `scipy.stats.ttest_ind(..., equal_var=False)` y `F = t^2`.

Un oracle externo no reemplaza la fórmula independiente; ambos deben concordar.

## 10. Invariantes obligatorios

Para ambos métodos, dentro de tolerancia float64 apropiada:

1. **orden de grupos:** permutar grupos no cambia F ni p;
2. **orden interno:** permutar observaciones dentro de un grupo no cambia el resultado;
3. **traslación común:** sumar una constante común a todos los valores no cambia F ni p;
4. **escala común no nula:** multiplicar todos los valores por la misma constante no cambia F ni p;
5. **repetición:** la misma entrada produce exactamente la misma estructura/resultados deterministas;
6. **k=2:** equivalencia F/t² descrita arriba;
7. **medias exactamente iguales con dispersión positiva idéntica:** configuraciones construidas para `SS_between=0` producen `F=0`, `p=1`;
8. **identidad Classical:** `SS_total = SS_between + SS_within` dentro de tolerancia;
9. **grados de libertad Classical:** `df_between + df_within = N - 1`;
10. **rango:** `p in [0,1]`, `F >= 0`, `eta_squared in [0,1]` cuando esté definido.

## 11. Casos adversariales obligatorios

El candidate debe probar al menos:

- grupos mínimos `n_i=2`;
- diseños muy desbalanceados;
- grupo pequeño con varianza grande;
- grupo grande con varianza grande;
- varianzas casi degeneradas pero por encima del contrato de calidad;
- offsets grandes con dispersión ordinariamente representable en float64;
- escalas comunes grandes y pequeñas dentro del dominio no subnormal;
- outlier dominante;
- skewness severa;
- colas pesadas;
- medias muy separadas con residuos internamente compatibles;
- datos no finitos;
- grupos constantes/degenerados;
- permutaciones de labels, grupos y observaciones.

Estos casos prueban implementación y límites; no autorizan por sí solos robustez inferencial.

## 12. Qué significa PASS antes de calibración

Un candidate puede obtener **PASS de implementación matemática** si:

- concuerda con fórmulas independientes;
- concuerda con oráculos externos dentro de tolerancias justificadas;
- satisface invariantes;
- maneja correctamente errores/degeneraciones;
- es determinista;
- no altera datos de entrada;
- no reintroduce selección automática ONE_WAY;
- conserva complexity `O(N)` para summaries + `O(k)` para el kernel.

Este PASS **no significa todavía** que Classical o Welch estén autorizados automáticamente para datos no Gaussianos, pequeños, contaminados o seleccionados adaptativamente.

## 13. Evidencia posterior requerida

Después del candidate matemático se requerirá una preregistración ANOVA propia para estudiar, como mínimo:

- error tipo I;
- potencia;
- balance/desbalance;
- asociación entre tamaños y varianzas;
- número de grupos;
- tamaños por grupo;
- Gaussianidad y desviaciones de forma;
- colas pesadas;
- contaminación/outliers;
- heterocedasticidad;
- estabilidad Monte Carlo;
- holdout independiente;
- comportamiento de cualquier política futura de abstención/selección.

La calibración histórica `anova-v1-2026-08` se conserva como piloto y no cuenta como cierre final.

## 14. Relación con el optimizador de experimentos

Step 1 no modifica `optimization/orchestrator.py`.

Sin embargo, la arquitectura summary-based y los outputs estructurados son requisitos deliberados para permitir un Step 2 donde:

```text
orchestrator -> ANOVA evaluator -> F / p-value / df / metadata propia
```

sin obligar a ANOVA a imitar el contrato de Kruskal-Wallis.

## 15. Condiciones que obligan a revisar esta especificación

Debe reabrirse CP-ANOVA-02 si se pretende:

- cambiar el estimando;
- excluir `k=2`;
- aceptar grupos degenerados;
- cambiar las fórmulas Welch;
- interpretar un diagnóstico como prueba lógica de normalidad/homocedasticidad;
- habilitar selección automática antes de calibración;
- incorporar post-hoc, factorial, repeated measures o DOE dentro del mismo contrato;
- definir effect size Welch como equivalente directo de eta-squared Classical.
