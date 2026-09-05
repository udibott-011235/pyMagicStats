# EV-ANOVA-PREFLIGHT — ANOVA statistical closure preflight

- Fecha: `2026-09-05`
- Rama de trabajo: `audit/anova-statistical-closure`
- Base técnica observada: `main@402e4601df460811779b3238c2526ac12f463a67`
- Rama histórica inspeccionada: `feature/anova-engine@9ebbe4fd1f6b9f847be75f7add09fee609ebe383`
- Rol: `statistical-software-architecture`
- Estado: `evidence / preflight complete`

## Objetivo

Establecer qué parte del trabajo ANOVA histórico puede reutilizarse para cerrar el bloqueante estadístico ANOVA previo al Manual UAT 1, sin transferir evidencia de calibración de medias/t-tests ni reactivar automáticamente el selector ONE_WAY.

El estimando primario permanece como diferencias entre medias de grupos independientes y la hipótesis global:

```text
H0: mu_1 = mu_2 = ... = mu_k
H1: al menos una media poblacional difiere
```

## Estado Git observado

`main` y `feature/anova-engine` están divergidos. La comparación actual muestra que la rama histórica conserva cuatro commits únicos pero está veinticuatro commits detrás de `main`; su merge-base es `33f28bd0487d4a4aff05253d013ba5f010493430`.

Por tanto la rama histórica no es un candidato seguro para merge o rebase mecánico. Se trata como fuente de evidencia y código de referencia. El cierre se desarrolla desde `main` actual en una rama nueva.

## Activos útiles de la rama histórica

La rama histórica contiene piezas técnicamente valiosas:

- `pyMagicStat/inference/anova.py` con implementaciones explícitas de Classical one-way ANOVA y Welch one-way ANOVA;
- tests de concordancia con SciPy/statsmodels;
- invariancia a traslación común y orden de grupos;
- tests adversariales de outlier dominante, heterocedasticidad y no pooling de medias;
- un diseño correcto de diagnóstico sobre residuos centrados dentro de cada grupo;
- un harness reproducible `experiments/anova_calibration.py` con escenarios homocedásticos, heterocedásticos, desbalanceados, sesgados, heavy-tail, mezclas y contaminación;
- documentación explícita de que Kruskal-Wallis no debe presentarse como reemplazo automático de una prueba de igualdad de medias.

Estos activos deben rescatarse selectivamente, no por cherry-pick global.

## Estado relevante de `main`

### Validator

`InferenceValidator.validate_one_way` ya existe en `main` y:

- normaliza cada grupo de forma independiente;
- construye residuos centrados `y_ij - mean_i`;
- aplica shape/outlier diagnostics por grupo sobre esos residuos;
- evalúa heterocedasticidad mediante `VarianceAssessment`;
- mantiene independencia como metadato externo del diseño;
- permite actualmente dos o más grupos.

La decisión de trabajar con residuos dentro de grupo es estadísticamente correcta para este diseño: apilar observaciones de grupos con medias distintas puede crear una forma pooled no normal aunque los errores dentro de cada grupo sean compatibles con el modelo.

### Selector

`MethodSelector` falla cerrado para `InferenceDesign.ONE_WAY`: devuelve `selected_method=None`, `status=NOT_CALIBRATED` y `guarantee=NOT_CALIBRATED`.

Ese comportamiento debe preservarse durante el cierre del estadístico. Validar Classical/Welch para invocación manual no autoriza selección automática.

### Capability registry

El registry de capacidades actual contiene garantías explícitas para inferencia de una media, pero no registra todavía capacidades ONE_WAY. ANOVA no debe añadirse allí hasta que se defina qué garantía concreta puede sostener cada ruta.

### Variance diagnostics

El `VarianceAssessment` actual es más simple que el usado por la rama ANOVA histórica. Conserva varianzas, ratio de varianzas, Levene/Brown-Forsythe por mediana, ratio de tamaños y heterogeneidad. La rama histórica añadía Fligner, Bartlett, asociación tamaño-varianza y el caso adversarial `small_group_large_variance`.

La antigua política de selección Classical dependía de métricas que ya no existen en `main`, por lo que no puede reutilizarse sin una nueva decisión arquitectónica y nueva evidencia.

## Hallazgos de la calibración histórica

La calibración `anova-v1-2026-08` es útil como piloto exploratorio pero no como evidencia suficiente de producción.

La corrida documentada usó:

- 15 escenarios;
- tamaños nominales 10, 25 y 60;
- H0 y una sola separación H1 de 0.8;
- tres seeds;
- 100 réplicas por seed/celda;
- 27,000 datasets totales.

Problemas para una afirmación de calibración final:

1. 100 réplicas por seed producen incertidumbre Monte Carlo grande alrededor de alpha=0.05; incluso agregando tres seeds siguen siendo sólo 300 réplicas por celda.
2. Las tasas condicionales tras selección usan subconjuntos menores y son aún más inestables.
3. La política y sus thresholds fueron evaluados en la misma familia de escenarios usada para diseñarlos; no existe holdout independiente preregistrado.
4. El propio informe histórico reconoce falsos `ACCEPTABLE`, conservadurismo, sensibilidad a seed y necesidad de revisión independiente.
5. La evidencia de selección no es necesaria para cerrar inicialmente los estadísticos explícitos y añade un problema post-selection innecesario para Manual UAT 1.

Conclusión: `anova-v1-2026-08` queda clasificada como evidencia piloto/histórica y no transfiere status `calibrated` al código nuevo.

## Compatibilidad de oráculos

El test histórico de Classical usa `scipy.stats.f_oneway(..., equal_var=True)`. El proyecto declara `scipy>=1.11`, mientras que el parámetro `equal_var` fue añadido en SciPy 1.16.

El cierre debe evitar una dependencia accidental de SciPy >=1.16 sólo para ejecutar el oracle. Opciones válidas:

- Classical: `scipy.stats.f_oneway(*groups)` para todo el rango declarado, más fórmula independiente/manual;
- Welch: `statsmodels.stats.oneway.anova_oneway(..., use_var="unequal", welch_correction=True)` para statsmodels >=0.14;
- SciPy >=1.16 puede añadirse como oracle secundario de Welch cuando esté disponible, sin convertirlo en requisito mínimo.

## Contrato estadístico provisional

El cierre se separa en dos planos.

### Plano A — estadísticos explícitos

Validar e implementar de forma reproducible:

- Classical one-way ANOVA;
- Welch one-way ANOVA;
- H0 global de igualdad de medias;
- grados de libertad, F y p-value;
- invariantes algebraicos y numéricos;
- límites y degeneraciones;
- diagnósticos anexos sin convertirlos automáticamente en selector.

### Plano B — selección automática

Fuera del cierre inicial. `MethodSelector` permanece `NOT_CALIBRATED` para ONE_WAY hasta una decisión posterior y evidencia propia.

## Preguntas que debe resolver la especificación

1. Alcance público `k>=2` versus `k>=3`. SciPy/statsmodels y el validator actual aceptan dos grupos; si se conserva `k>=2`, debe añadirse el invariante Classical F = Student t^2 y Welch F = Welch t^2 para dos grupos.
2. Separar garantía exacta/paramétrica bajo modelo Gaussiano externo de cualquier ruta empíricamente robusta frente a no-normalidad.
3. Definir si la igualdad de varianzas de Classical es una declaración externa del modelo, un requisito estricto, o sólo una elección explícita del usuario acompañada por diagnostics. No debe inferirse únicamente de `p > alpha` en Levene/Brown-Forsythe.
4. Definir política para independencia `unknown`: para inferencia autorizada debe fallar cerrado o quedar explícitamente fuera del dominio validado.
5. Definir qué métricas de efecto, si alguna, entran en Manual UAT 1; no mezclar su validación con la del estadístico F sin especificación.

## Oráculos y referencias externas seleccionados

- SciPy `stats.f_oneway`: oracle Classical compatible con versiones antiguas; desde SciPy 1.16 también ofrece Welch con `equal_var=False`.
- statsmodels `stats.oneway.anova_oneway`: oracle independiente para Standard/Welch/Brown-Forsythe, con Welch como `use_var="unequal"`.
- Minitab One-Way ANOVA: referencia de comportamiento aplicado; normalidad se inspecciona sobre residuos y al no asumir varianzas iguales se usa Welch.

Ninguno de estos oráculos sustituye una especificación matemática independiente ni una calibración propia del dominio de pyMagicStats.

## Veredicto de preflight

`feature/anova-engine@9ebbe4fd...` **NO es merge candidate**.

Se rescatan sus fórmulas, tests, escenarios y decisiones conceptuales que sigan siendo compatibles con `main`; se descartan su autorización de selección automática y el status implícito de calibración.

El siguiente checkpoint es congelar la especificación estadística y el plan de oráculos antes de escribir el candidato ANOVA sobre `main` actual.
