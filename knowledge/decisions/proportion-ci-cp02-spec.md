# CP-02 — Especificación estadística propuesta para intervalos de proporción

**Stage:** `STAGE-PROP-CI-001`  
**Estado:** `under_review`  
**Baseline:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`  
**Evidencia de entrada:** `knowledge/evidence/proportion-ci-cp01-census.md`  
**Owner de decisión:** Product Owner  
**Arquitectura:** statistical-software-architecture

## 1. Estimando y diseño

El estimando soportado por este stage es una única proporción poblacional

`p = P(Y=1)`

bajo un diseño Bernoulli/binomial ordinario con unidades independientes, probabilidad de éxito común y número de ensayos fijado/observado. La implementación no puede demostrar independencia, representatividad ni procedencia; esas condiciones deben permanecer explícitas como supuestos de diseño, no inferirse de la muestra.

Quedan fuera de alcance de este stage: survey weights, effective sample size, clustering, repeated measures, finite population correction, tasas con exposición, proporciones multinomiales y éxitos fraccionarios derivados de ponderaciones.

## 2. Representaciones de entrada

Se preservará la ruta raw binaria y la ruta callable por compatibilidad. Se diseñará una representación agregada explícita `(successes, trials)` donde ambos sean enteros y cumplan `0 <= successes <= trials`, `trials >= 1`.

La aceptación actual de `incidences` fraccionario se clasifica como legacy incompatibility con el modelo binomial. No se reinterpretará silenciosamente como un modelo ponderado. CP-03 deberá definir una transición de compatibilidad/deprecación antes de un rechazo estricto si se conserva el constructor actual.

## 3. Métodos aprobados como candidatos del stage

### 3.1 Wilson score — candidato principal

- Permanece como default candidato.
- Intervalo bilateral score sin corrección de continuidad.
- Se evaluará en toda la región binomial, incluidos `x=0` y `x=n`.
- No dependerá de una regla `successes>=10 and failures>=10` para existir o seleccionarse.
- Su equivalencia algebraica actual con SciPy se considera evidencia numérica, no calibración de cobertura.

### 3.2 Clopper–Pearson — candidato exacto/conservador

- Se incorpora a CP-04/CP-06 como candidato frecuentista exacto por inversión binomial.
- Debe etiquetarse como exacto en el sentido de control de cobertura discreta, no como intervalo de ancho óptimo.
- El conservadurismo debe medirse explícitamente en calibración.

### 3.3 Wald — legacy explícito

- Se conserva únicamente por compatibilidad/uso explícito mientras CP-03 determine la transición.
- Nunca será default ni fallback automático.
- Su existencia no debe condicionarse a `successes>=10 and failures>=10`; esa regla podrá permanecer temporalmente como warning legacy pero no como garantía.
- Su cobertura, límites fuera de `[0,1]` y degeneración en boundaries se cuantificarán en CP-04/CP-06 como evidencia de limitaciones.

### 3.4 Jeffreys — candidato comparador, no CI frecuentista por defecto

- Puede entrar en el harness como comparador debido a su utilidad práctica y disponibilidad como oráculo.
- Si se expone públicamente, deberá etiquetarse como intervalo/credible interval bayesiano basado en prior Beta(1/2,1/2), no mezclarse semánticamente con garantías frecuentistas.
- No será seleccionado automáticamente en este stage.

### 3.5 Métodos no aprobados todavía

Agresti–Coull, Wilson con corrección de continuidad, mid-P e inversión numérica alternativa no entran en producción por defecto en este stage. Pueden aparecer como comparadores si CP-04 demuestra valor experimental claro, pero requieren nueva decisión arquitectónica para exposición pública.

## 4. Lateralidad

El alcance mínimo obligatorio de producción permanece bilateral para compatibilidad. CP-03 podrá diseñar `alternative`/`side` de forma extensible, pero no se implementarán intervalos one-sided hasta que CP-04 registre su criterio de validación independiente. No se transferirá automáticamente la calibración bilateral.

## 5. Boundaries

`x=0` y `x=n` son casos válidos del modelo, no errores de input.

- Wilson debe devolver límites válidos en `[0,1]`.
- Clopper–Pearson debe seguir su definición exacta, incluyendo límites 0 o 1 cuando corresponda.
- Wald puede conservar su comportamiento legacy en una ruta explícita, con limitaciones visibles.

Ningún método debe inventar clipping salvo que forme parte de su definición contractual. En particular, no se modificará Wald silenciosamente sólo para hacerlo parecer válido.

## 6. Semántica de garantías

El resultado futuro debe separar al menos:

- `method`;
- `estimand`;
- `design`;
- `confidence_level`;
- `interval_kind`/garantía;
- `calibration_status`;
- supuestos de diseño requeridos;
- información de compatibilidad/deprecación cuando aplique.

Categorías conceptuales propuestas:

- Wilson: frequentist score / calibration pending until CP-06;
- Clopper–Pearson: frequentist exact-conservative / calibration still required for project-level claims;
- Wald: frequentist asymptotic legacy / known limitations;
- Jeffreys: Bayesian credible interval / no frequentist routing claim by default.

Una fórmula conocida o una implementación de referencia no convierte automáticamente un método en `validated_with_limits` dentro de pyMagicStats.

## 7. Regla legacy successes/failures >= 10

La regla no puede ser selector universal ni prueba de validez. CP-03 debe renombrarla/deprecar su semántica si permanece visible. CP-04 podrá medir su relación con desempeño de Wald, pero cualquier threshold futuro deberá derivarse de evidencia y conservarse como condición específica del método, no como regla global de proporciones.

## 8. Routing y hallazgo CP01-F-003

Antes de permitir routing automático de proporciones, `MethodSelector` debe fallar cerrado para `Estimand.PROPORTION` cuando no exista una capacidad registrada y calibrada para ese estimando.

El comportamiento observado en CP-01, donde `Estimand.PROPORTION` puede recibir `selected_method="one_sample_t"`, se considera un defecto de separación de estimando y debe corregirse en la futura implementación candidata. La corrección no autoriza todavía seleccionar Wilson automáticamente; el estado correcto mientras no exista capability calibrada debe ser equivalente a `NOT_CALIBRATED`/`REVIEW_REQUIRED`, nunca una prueba de media.

## 9. Bootstrap de proporción

`BootstrapCI(stat="proportion")` no se fusionará silenciosamente con `PopulationProportionCI`. CP-03 debe definir si se mantiene como superficie separada y aclarar su estimando/entrada. No se podrá registrar como alternativa automática hasta contar con calibración específica y contrato binario explícito.

## 10. Decisiones que quedan para CP-03

- forma exacta del constructor/API agregada;
- transición de `incidences` fraccionario;
- export público de `PopulationProportionCI` o nueva fachada;
- esquema exacto del objeto/dict de resultado;
- estrategia de deprecación de Wald y metadata legacy;
- integración fail-closed con capability routing sin activar selección automática;
- relación pública con `BootstrapCI(stat="proportion")`.

## 11. Condición de aprobación de CP-02

CP-02 puede marcarse `complete` sólo después de aprobación explícita del Product Owner de estas decisiones. Hasta entonces permanece `under_review` y no autoriza implementación.