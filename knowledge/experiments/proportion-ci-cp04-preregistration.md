# CP-04 — Preregistro exhaustivo de calibración de intervalos de proporción

**Stage:** `STAGE-PROP-CI-001`  
**Estado:** `under_review`  
**Tipo:** preregistro confirmatorio antes de implementación/calibración  
**Baseline de producción:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`  
**Baseline documental de entrada:** `docs/proportion-ci-stage` @ `c435ccfdb959e0f980aae58d9f912e86349afb93`  
**Contratos de entrada:** CP-02 (`accepted`) y CP-03 (`accepted`)  
**Owner de decisión:** Product Owner  
**Arquitectura:** statistical-software-architecture

## 1. Propósito

Caracterizar de manera reproducible, determinista y adversarial el comportamiento de los intervalos bilaterales de una proporción Bernoulli/binomial aprobados para `STAGE-PROP-CI-001`.

El experimento debe responder, separadamente:

1. ¿La implementación reproduce correctamente la definición matemática/oráculos independientes?
2. ¿Cuál es la cobertura frecuentista real como función de `n`, `p` y `alpha`?
3. ¿Dónde existe undercoverage, conservadurismo, degeneración o salida del espacio paramétrico?
4. ¿Cuál es el ancho esperado y cómo cambia por región del espacio paramétrico?
5. ¿Se preservan simetrías, nesting por nivel de confianza, monotonicidad estructural y boundaries?
6. ¿Los resultados son estables numéricamente en tamaños y probabilidades extremos?

Este experimento **no selecciona automáticamente un método**, no habilita routing y no transfiere evidencia a otros estimandos.

## 2. Métodos incluidos

### Producción candidata

- `wilson` — score bilateral, default candidato.
- `clopper_pearson` — exacto/conservador por inversión binomial.
- `wald` — legacy explícito; se estudia para cuantificar límites, no para rehabilitarlo como default.

### Comparador no productivo

- `jeffreys` — intervalo creíble bayesiano Beta(1/2,1/2), únicamente como benchmark descriptivo de ancho/cobertura frecuentista observada. No se convierte en CI frecuentista ni en método de producción por este experimento.

No se incluyen como candidatos de producción Agresti–Coull, Wilson-CC, mid-P u otras variantes. Podrán mencionarse en discusión externa, pero no participan en gates ni routing.

## 3. Niveles de confianza

Se preregistran los siguientes `alpha`:

```text
0.001
0.005
0.010
0.025
0.050
0.100
0.200
```

correspondientes a niveles bilaterales:

```text
99.9%
99.5%
99.0%
97.5%
95.0%
90.0%
80.0%
```

`alpha=0.05` es el nivel focal principal. Los demás niveles son confirmatorios/stress y no pueden descartarse después de ver resultados.

## 4. Grid de tamaños muestrales

### 4.1 Exhaustivo discreto

Evaluar **cada entero**:

```text
n = 1, 2, 3, ..., 5000
```

No se muestrea este rango: se recorre completo.

### 4.2 Extensión de escala

Añadir obligatoriamente:

```text
7500
10000
15000
20000
30000
50000
75000
100000
250000
500000
1000000
```

Objetivo: detectar pérdida de precisión, inestabilidad de colas y supuestos asintóticos falsamente tranquilizadores.

### 4.3 Estratos de reporte

Los resultados deben resumirse al menos por:

```text
1–5
6–10
11–20
21–30
31–50
51–100
101–250
251–500
501–1000
1001–2000
2001–5000
>5000
```

Los estratos son descriptivos; no son reglas de routing.

## 5. Grid de probabilidades

No se utilizará únicamente un grid lineal, porque eso subrepresenta `p≈0/1`, precisamente donde los intervalos discretos presentan los cliffs más importantes.

La búsqueda usa la unión de cuatro familias.

### 5.1 Anchors fijos

Incluir exactamente, cuando estén en `[0,1]`:

```text
0
1e-12
1e-10
1e-9
1e-8
1e-7
1e-6
1e-5
1e-4
2.5e-4
5e-4
1e-3
2.5e-3
5e-3
0.01
0.025
0.05
0.10
0.20
0.30
0.40
0.50
```

y sus complementos `1-p`.

### 5.2 Grid lineal interior

Para cada `n` del rango exhaustivo, evaluar un grid lineal mínimo:

```text
p = 0.0001, 0.0002, ..., 0.9999
```

(9,999 puntos), además de `0` y `1`.

La implementación puede vectorizar/shardear esta matriz, pero no reducirla.

### 5.3 Grid event-scale

Para capturar el régimen donde el número esperado de éxitos/fracasos permanece pequeño aun con `n` grande, definir `lambda = n*p`.

Usar:

- 801 puntos log-espaciados entre `1e-6` y `100`;
- anchors adicionales `lambda in {0.01,0.025,0.05,0.1,0.25,0.5,0.75,1,1.5,2,3,4,5,7.5,10,15,20,30,40,50,75,100}`.

Para cada `n`:

```text
p = lambda/n
```

cuando `p <= 0.5`, más el complemento `1-p`.

Esta capa es obligatoria para no concluir falsamente que `n` grande elimina problemas de boundary cuando `np` o `n(1-p)` siguen siendo pequeños.

### 5.4 Puntos inducidos por endpoints del intervalo

Para cada `n`, `alpha`, método y `x=0..n`, calcular los límites `L_x`, `U_x`.

Añadir como candidatos de cobertura:

- cada endpoint único interior;
- `nextafter(endpoint, 0)`;
- `nextafter(endpoint, 1)`;
- midpoint entre endpoints adyacentes cuando sea representable.

Estos puntos capturan cambios discretos del conjunto de valores `x` cuyo intervalo contiene `p`.

## 6. Enumeración frecuentista determinista

Para un método `m`, tamaño `n`, probabilidad verdadera `p` y nivel `alpha`:

```text
C_m(n,p,alpha)
  = sum_{x=0}^n 1[p in I_m(x,n,alpha)] * BinomPMF(x; n,p)
```

Ésta es la métrica primaria de cobertura.

No se estimará mediante simulación cuando la enumeración binomial sea computable.

También calcular:

```text
ExpectedWidth_m(n,p,alpha)
  = sum_x width(I_m(x,n,alpha)) * BinomPMF(x;n,p)
```

Para Wald calcular adicionalmente:

```text
P_outside = P(lb < 0 or ub > 1)
P_degenerate = P(ub == lb)
```

Para todos los métodos calcular:

```text
undercoverage = max(0, (1-alpha) - coverage)
excess_coverage = max(0, coverage - (1-alpha))
```

## 7. Búsqueda adversarial de mínimos de cobertura

El grid anterior no se considera suficiente para afirmar que se encontró el peor caso.

### 7.1 Conjunto de aceptación

Para un `p` dado, identificar:

```text
A(p) = {x : L_x <= p <= U_x}
```

Para métodos cuyos límites sean monotónicos en `x`, verificar primero esa monotonicidad y representar `A(p)` como rango contiguo `[a,b]`.

Entonces:

```text
coverage(p) = P(a <= X <= b), X~Binomial(n,p)
```

### 7.2 Partición por endpoints

Los endpoints de todos los intervalos inducen regiones de `p` donde `[a,b]` no cambia.

Dentro de cada región, evaluar:

- ambos extremos mediante límites laterales apropiados;
- midpoint;
- todo extremo estacionario interior de la función de cobertura.

Para un rango aceptado fijo `[a,b]`, usar la identidad derivada de la distribución binomial para localizar la raíz interior de la derivada cuando exista. La implementación debe hacerlo en log-space mediante `gammaln`/equivalente para evitar overflow combinatorio.

Si una implementación no puede justificar analíticamente esa raíz, debe usar un optimizador acotado determinista con tolerancia absoluta de `p <= 1e-12` y demostrar por tests que reproduce la solución analítica en casos donde ésta esté disponible.

### 7.3 Métodos no monotónicos

Si un método viola monotonicidad de endpoints en `x`, no se fuerza un rango contiguo. Se vuelve a enumeración explícita del conjunto `A(p)` y el hallazgo se registra como anomalía estructural.

### 7.4 Salida obligatoria

Para cada `(method, alpha, n)` registrar:

- mínimo global encontrado de cobertura;
- `p` donde ocurre;
- conjunto/rango de `x` que cubre ese `p`;
- distancia respecto a nominal;
- si el mínimo ocurrió en boundary inducido, raíz interior o grid;
- verificación float64 y, para mínimos materiales, alta precisión.

## 8. Estratos estadísticos por número esperado de eventos

Reportar por separado las regiones:

```text
min(np, n(1-p)) < 0.5
0.5 <= min(...) < 1
1 <= min(...) < 2
2 <= min(...) < 5
5 <= min(...) < 10
10 <= min(...) < 20
20 <= min(...) < 30
>= 30
```

Estos estratos **no son thresholds de selección**. Sirven para mostrar directamente si una regla basada en 5, 10, 20 o 30 eventos tendría o no relación estable con cobertura.

La regla legacy `successes>=10 and failures>=10` se estudia únicamente como objeto descriptivo para Wald. No se valida por decreto ni se convierte en gate global.

## 9. Métricas principales

### Primarias

1. cobertura frecuentista exacta/determinista;
2. máximo undercoverage;
3. mínimo de cobertura por `n/alpha`;
4. expected width;
5. exceso de cobertura/conservadurismo;
6. probability mass de intervalos Wald fuera de `[0,1]`;
7. probability mass de intervalos degenerados.

### Secundarias

8. ancho máximo y mínimo por `x`;
9. mediana/percentiles de ancho ponderados por Binomial;
10. ratio de expected width vs Wilson;
11. ratio de expected width vs Clopper–Pearson;
12. simetría complementaria;
13. nesting por nivel de confianza;
14. monotonicidad de límites respecto de `x`;
15. estabilidad float64 vs alta precisión;
16. runtime y memoria por backend/shard.

## 10. Tiers descriptivos de undercoverage

Para evitar reinterpretar los resultados después de verlos, preregistrar los siguientes rótulos **descriptivos**, no reglas universales de validez:

```text
nominal_like:       deficit <= 0.005
mild_shortfall:     0.005 < deficit <= 0.015
material_shortfall: 0.015 < deficit <= 0.030
severe_shortfall:   0.030 < deficit <= 0.050
critical_shortfall: deficit > 0.050
```

Ejemplo: en un intervalo nominal 95%, cobertura 92% implica déficit 3 puntos porcentuales y entra en `material_shortfall`.

Estos tiers no autorizan routing. Su propósito es que un déficit observado hoy y el mismo déficit observado mañana reciban el mismo lenguaje.

## 11. Gates matemático-numéricos preregistrados

### 11.1 Wilson

Debe cumplir todos:

- equivalencia con oráculo independiente SciPy en celdas compartidas;
- error absoluto de límites `<=1e-12` para `n<=5000` en float64 ordinario;
- error `<=1e-10` para stress cells hasta `n=1e6`, salvo demostración high-precision de que la diferencia procede del oráculo;
- límites dentro de `[0,1]` con tolerancia `5e-15`;
- simetría complementaria de límites `<=1e-12` (`<=1e-10` en stress extremo);
- monotonicidad de lower/upper respecto de `x`;
- nesting al aumentar confidence level;
- ninguna afirmación de cobertura nominal uniforme sobre todo `p` salvo que la evidencia lo demuestre.

La cobertura puede quedar por debajo del nominal en regiones discretas. Eso no se ocultará: se clasificará por los tiers de §10 y limitará cualquier claim posterior.

### 11.2 Clopper–Pearson

Debe cumplir todos:

- equivalencia con `scipy.stats.binomtest(...).proportion_ci(method="exact")` en celdas compartidas;
- tolerancias numéricas iguales a Wilson;
- límites en `[0,1]`;
- boundaries exactos según definición;
- simetría complementaria;
- nesting por confidence level;
- cobertura determinista `>= 1-alpha - 1e-12` en toda celda evaluada.

Toda aparente violación de la última condición mayor que `1e-12` exige recálculo high-precision antes de clasificar el método o la implementación.

### 11.3 Wald

El gate es de fidelidad legacy, no de calidad estadística:

- reproducir exactamente la fórmula aprobada, sin clipping;
- mantener simetría algebraica;
- registrar todos los casos fuera de `[0,1]` y degenerados;
- cuantificar cobertura y deficits sin descartar celdas desfavorables;
- no presentar el threshold legacy `10/10` como garantía.

Una cobertura pobre no constituye un bug si reproduce correctamente Wald; constituye evidencia para limitar su uso.

### 11.4 Jeffreys comparador

- reproducir el oráculo de referencia seleccionado;
- etiquetar resultados como `bayesian_comparator`;
- medir cobertura frecuentista sólo como propiedad observada, no como garantía semántica;
- ninguna métrica de Jeffreys puede activar producción o routing en este stage.

## 12. Invariantes metamórficos

Para cada método donde matemáticamente corresponda:

### Complement symmetry

```text
L(x,n) ~= 1 - U(n-x,n)
U(x,n) ~= 1 - L(n-x,n)
```

### Confidence nesting

Si `alpha_1 < alpha_2`:

```text
L(alpha_1) <= L(alpha_2)
U(alpha_1) >= U(alpha_2)
```

### Estimate consistency

```text
estimate == successes/trials
```

### Representation equivalence

Para conteos enteros:

```text
raw binary data
predicate equivalent
from_counts
legacy integral incidences
```

deben producir el mismo `estimate` y, para el mismo método, los mismos límites dentro de tolerancia, aunque metadata/warnings difieran.

### Shard invariance

Los outputs deterministas deben ser idénticos independientemente de:

- número de workers;
- orden de `n`;
- orden de `p`;
- tamaño de batch;
- CPU vs backend acelerado, dentro de tolerancia explícita.

## 13. Auditoría high-precision

Toda celda que cumpla cualquiera de estas condiciones se recalcula con precisión arbitraria mínima de **80 dígitos decimales**:

- aparente violación de cobertura exacta de Clopper–Pearson;
- diferencia de oráculo > tolerancia;
- límite fuera de `[0,1]` para Wilson/CP;
- violación de simetría/nesting > tolerancia;
- mínimo de cobertura Wilson/Jeffreys situado a menos de `1e-10` de un endpoint inducido;
- undercoverage Wilson clasificado `severe` o `critical`;
- cualquier NaN/Inf/underflow/overflow en PMF/CDF.

El reporte debe conservar tanto float64 como high-precision y declarar cuál gobierna la interpretación.

## 14. Monte Carlo shadow audit

La enumeración determinista es autoridad para cobertura. Se añade Monte Carlo únicamente para intentar detectar bugs en el harness o en el mapeo de intervalos.

### 14.1 Celdas críticas

Seleccionar **128 celdas** determinísticamente a partir de los peores/minimos encontrados en el barrido de desarrollo, cubriendo todos los métodos de producción, alphas y estratos de `n`.

Ejecutar:

```text
1,000,000 réplicas por celda
```

Total máximo: 128 millones de draws binomiales por pasada.

### 14.2 Celdas amplias

Seleccionar adicionalmente **512 celdas** estratificadas y ejecutar:

```text
250,000 réplicas por celda
```

Total: 128 millones de draws adicionales.

Total shadow audit preregistrado: **256 millones de draws**.

### 14.3 Criterio

Para cada celda comparar la cobertura Monte Carlo con la cobertura enumerada usando:

```text
|coverage_MC - coverage_exact| <= max(5 * MC_SE, 0.001)
```

Una discrepancia mayor no modifica la verdad matemática por votación: abre un hallazgo del harness hasta explicar la diferencia.

### 14.4 RNG

- usar generador explícito/versionado;
- seed derivada de un master seed y del identificador canónico de la celda mediante hash estable;
- el resultado debe ser invariante a shard/worker/batch;
- registrar master seed, algoritmo RNG y versión.

## 15. Holdout confirmatorio posterior al freeze de CP-05

No se revelará una lista fija de holdout antes de congelar el SHA candidato de implementación.

Después del freeze de CP-05, el Product Owner genera/registra un master seed nuevo. A partir de él se generan **10,000 celdas holdout** sin modificar thresholds ni código.

Distribución preregistrada:

- `alpha`: uniforme discreto sobre el conjunto de §3;
- `n`: mezcla 60% log-uniform integer `1..5000`, 30% log-uniform integer `5001..100000`, 10% log-uniform integer `100001..1000000`;
- `p`: mezcla
  - 30% Uniform(0,1),
  - 30% log-boundary y su complemento,
  - 30% event-scale `lambda/n` con lambda log-uniform `1e-6..100` y espejo,
  - 10% exact/boundary-neighbor points derivados de endpoints.

Las 10,000 celdas se evalúan por enumeración/búsqueda determinista según corresponda.

No se permite ajustar implementación, tiers ni criterios usando el holdout y luego conservar el mismo veredicto. Cualquier fix genera nuevo SHA y una nueva evaluación de holdout con seed nueva.

## 16. Independencia de oráculos

Al menos:

- Wilson: comparar contra SciPy y, en subconjunto, statsmodels;
- Clopper–Pearson: SciPy binomtest exact y, en subconjunto, statsmodels beta;
- Wald: fórmula independiente y statsmodels normal;
- Jeffreys: statsmodels o implementación Beta independiente.

Los oráculos no se usarán como sustituto ciego de producción: sirven para detectar discrepancias.

## 17. Outputs obligatorios

El harness futuro debe producir, como mínimo:

```text
experiments/results/proportion_ci_calibration_metadata.json
experiments/results/proportion_ci_interval_grid.parquet
experiments/results/proportion_ci_coverage_summary.parquet
experiments/results/proportion_ci_worst_cases.csv
experiments/results/proportion_ci_event_regimes.csv
experiments/results/proportion_ci_invariants.csv
experiments/results/proportion_ci_high_precision_audit.csv
experiments/results/proportion_ci_mc_shadow.csv
experiments/results/proportion_ci_holdout_summary.csv
```

Si el volumen de `interval_grid` o `coverage_summary` es demasiado grande para Git, el raw exhaustivo puede permanecer reconstruible/no versionado y se debe versionar:

- metadata completa;
- hashes;
- comando;
- schema;
- summaries/worst cases;
- procedimiento determinista de reconstrucción.

No se degrada el grid sólo para reducir tamaño del repositorio.

## 18. Metadata reproducible obligatoria

Registrar:

- repo;
- branch;
- candidate SHA;
- experiment version;
- CP-04 spec SHA;
- Python;
- NumPy;
- SciPy;
- statsmodels;
- backend CPU/GPU;
- hardware visible;
- OS;
- RNG/seed para MC;
- número de workers;
- batch size;
- comandos exactos;
- timestamps;
- hashes de outputs;
- tolerancias;
- número total de `(n,p,alpha,method)` evaluados por capa;
- cantidad de high-precision rechecks;
- failures/NaN/Inf/excepciones;
- celdas excluidas: idealmente cero; toda exclusión debe explicarse.

## 19. Paralelización y hardware

El cálculo puede explotar CPU multinúcleo y GPU cuando sea útil, pero el backend no forma parte de la definición estadística.

Reglas:

- usar bloques por `(alpha,n)` o esquema equivalente reproducible;
- evitar construir tensores gigantes innecesarios que agoten VRAM/RAM;
- preferir CDF/SF/logPMF estables a sumar PMF ingenuamente en colas extremas;
- la aceleración no puede cambiar el conjunto de celdas;
- una ejecución acelerada debe ser contrastada contra CPU float64 en un subconjunto preregistrado;
- diferencias por backend fuera de tolerancia son hallazgo, no ruido descartable.

## 20. Criterios de clasificación en CP-06

### `validated_with_limits`

Un método puede recibir esta clasificación del proyecto sólo si:

1. pasa gates matemático-numéricos correspondientes;
2. su mapa de cobertura/width está completo dentro del dominio preregistrado;
3. toda región `material/severe/critical` queda explícitamente documentada;
4. no existen discrepancias de high-precision sin resolver;
5. shadow MC no muestra inconsistencia del harness;
6. holdout posterior al freeze no revela defecto de implementación/contabilidad;
7. Antigravity audita el SHA exacto posteriormente en CP-07.

### `not_calibrated`

Se mantiene si faltan celdas, outputs, holdout, high-precision requerido o si un hallazgo impide caracterizar el método de forma fiable.

### `rejected_for_claim`

Se utiliza para una afirmación concreta (por ejemplo, “cobertura nominal uniforme” o “Wald seguro cuando 10/10”) si la evidencia la contradice. No implica que la fórmula matemática deje de existir como legacy.

## 21. Regla explícita sobre routing

Incluso un resultado `validated_with_limits` en CP-06 **no autoriza automáticamente MethodSelector**.

El routing de proporciones requiere una decisión posterior que especifique qué garantía se promete, qué regiones son admisibles y cómo se comporta el sistema fuera de ellas.

Hasta entonces:

```text
Estimand.PROPORTION -> selected_method=None -> NOT_CALIBRATED/REVIEW_REQUIRED
```

según el contrato CP-03.

## 22. Prohibiciones contra p-hacking/calibration hacking

Después de aprobar CP-04 no se puede:

- eliminar un alpha porque produce malos resultados;
- eliminar n pequeños o p extremos del reporte principal;
- cambiar tiers de undercoverage después de observarlos;
- cambiar tolerancias para hacer desaparecer un fallo;
- seleccionar sólo celdas favorables;
- usar el holdout para ajustar y conservarlo como holdout;
- sustituir enumeración por Monte Carlo cuando la primera contradiga una expectativa;
- reinterpretar Jeffreys como frecuentista porque su cobertura sea favorable;
- convertir `successes>=10` en regla porque una región parcial parezca mejor.

Toda modificación posterior del preregistro debe quedar versionada como enmienda **antes** de ejecutar el análisis afectado y justificar por qué no responde a resultados ya observados.

## 23. Condición de salida de CP-04

CP-04 puede marcarse `complete/accepted` únicamente tras aprobación explícita del Product Owner de:

- métodos y comparador;
- alphas;
- `n=1..5000` exhaustivo + stress hasta 1e6;
- cuatro familias de `p`;
- búsqueda adversarial entre endpoints;
- métricas y tiers;
- gates por método;
- auditoría high-precision;
- 256 millones de draws de shadow Monte Carlo;
- holdout de 10,000 celdas generado después del freeze;
- outputs y reproducibilidad;
- prohibición de routing automático.

La aprobación de CP-04 **no autoriza aún implementación**. CP-05 sólo comienza con una autorización separada del Product Owner.