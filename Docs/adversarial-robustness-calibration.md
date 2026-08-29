# Calibración adversarial de robustez para inferencia de la media

## Estado y alcance

**CALIBRATION EVIDENCE READY**

Esta fase evaluó el contrato vigente entre `ShapeAssessment`,
`OutlierAssessment`, `SamplingRobustness` y `MethodSelector` sin modificar la
política productiva. El baseline fue
`5a0bca910b09451a3e037b313ee70afe3e3757ca`, la política observada fue
`mean-v2.1-2026-08` y se asumió independencia por diseño. No se modificaron
ANOVA, `main` ni `feature/anova-engine`.

Las expresiones **falso positivo** y **falso negativo de política** se usan aquí
como proxies operacionales, no como etiquetas poblacionales conocidas:

- falso positivo: `acceptable` o `caution` condicionado a desempeño claramente
  alejado de α=.05/cobertura=.95;
- falso negativo: `insufficient` condicionado a desempeño compatible con el
  objetivo nominal.

No se fijó una tolerancia productiva nueva (por ejemplo, error Tipo I ≤.06).

## Metodología

### Diseño Monte Carlo

Se ejecutaron **237.800 réplicas en 368 celdas** con una estrategia adaptativa:

- 200 réplicas por celda para una exploración de 26 escenarios y los tamaños
  `5, 8, 10, 15, 20, 30, 40, 50, 80, 100, 200, 500, 750, 2000`;
- 10.000 réplicas en cada una de las cinco celdas adversariales críticas;
- 5.000 réplicas para normal pura en `n=3,4,5,8,10,15,20`, para normal pura en
  `n=30,100,750,2000,10000`, y para las doce contaminaciones en `n=100`.

Las familias fueron N(0,1), Student-t con df 30/10/5/3, lognormal con σ
.25/.50/1.00, contaminación normal simétrica y asimétrica con ε
.001/.005/.01/.025/.05/.10, bimodales simétrica y asimétrica, mezclas normales
con medias o varianzas distintas, y Gamma con shape 2/4. Student-t, lognormal,
Gamma y las mezclas se centraron por su media exacta y se estandarizaron por su
varianza poblacional cuando ésta existía. La transformación afín no cambia el
estadístico t ni su cobertura.

Para cada réplica, el script llamó a los componentes públicos de validación y
selección. El one-sample t-test y su CI bilateral se calcularon para todas las
réplicas, incluso cuando la política produjo `insufficient`; esto es necesario
para estimar desempeño condicionado sin ocultar la rama rechazada por la
política.

Se registraron los descriptivos, diagnósticos de forma, detección de extremos,
decisión, método seleccionado, estadístico t, p-value, decisión de contraste,
CI y cobertura. También se calculó el contrafactual diagnóstico

`influence_ratio = |mean_full - mean_without_extremes| / SE_full`.

Retirar extremos no se interpreta como recomendación: el filtrado condicionado
al dato cambia la distribución muestral y puede introducir sesgo.

### Reproducibilidad e incertidumbre

- seed global: `20260828`;
- estrategia: `numpy.random.SeedSequence`, un stream por celda ordenada y un
  seed uint64 registrado por réplica;
- Python 3.12.13, NumPy 2.5.2, SciPy 1.18.1;
- commit al inicio: `5a0bca910b09451a3e037b313ee70afe3e3757ca`;
- timestamp y matriz completa de réplicas/seeds documentados en metadata.

Todas las tasas incluyen intervalos Wilson 95%. Como referencia de precisión
Monte Carlo, con p=.05 el error estándar es aproximadamente .00218 para 10.000
réplicas y .00308 para 5.000. Las celdas exploratorias de 200 réplicas sirven
solo para localizar señales; no sostienen recomendaciones por sí solas.

## Evidencia

### BUG-03 — extremidad no equivale a influencia

En normal pura, `OutlierAssessment` termina alertando casi siempre al crecer n,
aunque el t-test conserva su comportamiento nominal. Las razones de influencia
de la tabla se condicionan a que se detectó al menos un extremo.

| n | Réplicas | P(extremo > 0) | Tipo I total | Cobertura total | Mediana influencia/SE | P90 influencia/SE | Δt medio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 5.000 | .0984 | .0474 | .9526 | .533 | .885 | .667 |
| 100 | 5.000 | .1024 | .0514 | .9486 | .335 | .398 | .362 |
| 750 | 5.000 | .3176 | .0468 | .9532 | .133 | .166 | .139 |
| 2.000 | 5.000 | .6016 | .0476 | .9524 | .083 | .166 | .092 |
| 10.000 | 5.000 | .9912 | .0504 | .9496 | .046 | .143 | .065 |

En n=10.000 hubo WARN en 4.956/5.000 muestras, pero el error Tipo I condicionado
a WARN fue .0500 y la cobertura .9500. Por tanto, el conteo identifica rareza
con probabilidad acumulativa creciente, no daño material a la inferencia.

En n=30, las muestras con extremos detectados tuvieron error Tipo I .0346 y
cobertura .9654 antes de retirar nada. El contrafactual de retirar esos extremos
redujo la cobertura a .8659. Esto no prueba que todo extremo sea inocuo; sí
prueba que la advertencia actual no tiene dirección causal ni debe equivaler a
“el punto perjudica la inferencia”.

### BUG-04 — subprotección confirmada en regiones específicas

Resultados de las cinco celdas confirmatorias:

| Escenario | Rama | Denominador | Tasa de rama | Tipo I [IC95%] | Cobertura [IC95%] |
|---|---|---:|---:|---:|---:|
| lognormal σ=.25, n=20 | total | 10.000 | 1.000 | .0553 [.0510,.0600] | .9447 [.9400,.9490] |
|  | acceptable | 7.388 | .7388 | .0620 [.0567,.0677] | .9380 [.9323,.9433] |
| lognormal σ=.50, n=50 | total | 10.000 | 1.000 | .0573 [.0529,.0620] | .9427 [.9380,.9471] |
|  | acceptable | 2.541 | .2541 | .1015 [.0904,.1139] | .8985 [.8861,.9096] |
|  | caution | 470 | .0470 | .0702 [.0504,.0970] | .9298 [.9030,.9496] |
| lognormal σ=1.00, n=30 | total | 10.000 | 1.000 | .1128 [.1067,.1192] | .8872 [.8809,.8933] |
|  | acceptable | 460 | .0460 | .3826 [.3393,.4278] | .6174 [.5722,.6607] |
| Student-t(df=5), n=20 | total | 10.000 | 1.000 | .0482 [.0442,.0526] | .9518 [.9474,.9558] |
|  | acceptable | 6.507 | .6507 | .0547 [.0494,.0605] | .9453 [.9395,.9506] |
| bimodal simétrica, n=300 | acceptable | 10.000 | 1.000 | .0494 [.0453,.0538] | .9506 [.9462,.9547] |

La auditoría previa queda confirmada para lognormal moderada/severa y no queda
confirmada como fallo material para Student-t(df=5) ni para la bimodal simétrica
definida aquí. La bimodal rechazó normalidad exacta en 100% de las réplicas, pero
su error Tipo I y cobertura fueron nominales; esto respalda mantener los tests
de normalidad como evidencia descriptiva y no como veto binario.

El condicionamiento descubre un mecanismo de selección importante. En
lognormal σ=1, las muestras que casualmente parecen “mild” tienen bias medio
-.105 y una razón media SE/SD empírico .614; el subconjunto visualmente benigno
es precisamente el que más subestima la incertidumbre.

### Contaminación e influencia

Las contaminaciones asimétricas en n=100 muestran falsos positivos de política
con denominadores confirmatorios útiles:

| ε | Rama | Denominador | Tasa de rama | Tipo I [IC95%] | Cobertura |
|---:|---|---:|---:|---:|---:|
| .005 | acceptable | 2.711 | .5422 | .0819 [.0721,.0928] | .9181 |
| .010 | acceptable | 1.676 | .3352 | .1718 [.1545,.1906] | .8282 |
| .025 | acceptable | 359 | .0718 | .7409 [.6932,.7835] | .2591 |

En la mezcla asimétrica, una muestra sin contaminantes observados puede parecer
normal y aun estar centrada por debajo de la media poblacional de la mezcla.
Así, “no observé contaminación” no es evidencia de que el proceso generador no
la tenga. La política basada solo en forma observada no puede resolver este
problema de procedencia/modelo.

Los 24 casos `acceptable` para ε=.05 y los 5 para contaminación simétrica
ε=.10 no se interpretan: sus denominadores son demasiado pequeños. Ésta es la
razón de conservar explícitamente cada denominador.

Como contrapunto, contaminación simétrica ε=.10 tuvo error Tipo I total .0480 y
cobertura .9520, pero 4.948/5.000 muestras fueron `insufficient`; dentro de esa
rama, Tipo I fue .0481 [IC95% .0425,.0544]. Es un falso negativo operacional y
otra señal de que número de extremos y daño a la media no son sinónimos.

### BUG-05 — n pequeño y semántica

Normal pura, que satisface exactamente el modelo del t-test, produjo:

| n | Tipo I total [IC95%] | Cobertura total | Tasa acceptable | Tasa insufficient |
|---:|---:|---:|---:|---:|
| 3 | .0520 [.0462,.0585] | .9480 | .3868 | .6132 |
| 4 | .0496 [.0439,.0560] | .9504 | .4486 | .5514 |
| 5 | .0530 [.0471,.0596] | .9470 | .6304 | .3696 |
| 8 | .0494 [.0437,.0558] | .9506 | .7720 | .2280 |
| 10 | .0500 [.0443,.0564] | .9500 | .8072 | .1928 |
| 15 | .0486 [.0430,.0549] | .9514 | .8500 | .1500 |
| 20 | .0486 [.0430,.0549] | .9514 | .8840 | .1160 |

El resultado no justifica `n >= X`. Demuestra algo más preciso: con n=3 el
t-test es válido **si** el modelo gaussiano es verdadero, mientras que tres
observaciones no demuestran empíricamente robustez frente a alternativas. La
etiqueta `acceptable` mezcla actualmente compatibilidad descriptiva con soporte
del modelo; en n pequeño debería distinguirse “válido bajo supuesto gaussiano
externo” de “robustez demostrada/calibrada”.

### BUG-06 — discontinuidades exactas

Se evaluaron grids sintéticos densos contra la política productiva intacta. Una
perturbación de .005 en skew/kurtosis, .0005 en fracción o una unidad en n fue
suficiente para los siguientes saltos:

| Serie | Primer valor tras el límite | Transición |
|---|---:|---|
| skew, n=40 | 1.005 | acceptable → insufficient |
| skew, n=80 | 2.005 | caution → insufficient |
| kurtosis, n=40 | 3.005 | caution → insufficient |
| kurtosis, n=80 | 7.005 | caution → insufficient |
| kurtosis, n=200 | 25.005 | caution → insufficient |
| outlier fraction, n=40 | .0255 | caution → insufficient |
| outlier fraction, n=80 | .1005 | caution → insufficient |
| tamaño moderado | 40 | insufficient → caution |
| tamaño grande | 80 | insufficient → caution |
| tamaño heavy-tail | 200 | insufficient → caution |

Son discontinuidades de la regla, no incertidumbre Monte Carlo. Dos muestras
arbitrariamente próximas pueden recibir acciones categóricas no adyacentes, y
el sistema no comunica distancia al límite ni error de medición de skewness,
kurtosis o fracción de extremos.

### Falsos positivos, falsos negativos y regiones peligrosas

**Falsos positivos confirmados:** `acceptable` bajo lognormal σ=.50/n=50,
lognormal σ=1/n=30 y contaminación asimétrica ε=.005/.01/.025 en n=100. En las
dos últimas regiones el desempeño es demasiado deficiente para explicarse por
ruido Monte Carlo. `caution` también fue deficiente para lognormal σ=.50/n=50,
aunque el IC es más ancho por n=470.

**Falsos negativos confirmados:** normal exacta en n=3/4 y contaminación
simétrica moderada/fuerte en n=100 muestran tasas altas de `insufficient` pese a
desempeño total o condicionado nominal. En normal n=3, por ejemplo, 3.066 casos
fueron `insufficient` y tuvieron Tipo I .0496 [IC95% .0424,.0578].

**Señales exploratorias que requieren confirmación:** bimodal asimétrica en n
pequeño, Gamma shape=2 en n pequeño, y varias combinaciones asimétricas fuera de
n=100. Sus 200 réplicas por celda localizan regiones, pero no se usan como
evidencia final.

## Recomendaciones conceptuales — no implementadas

1. **Separar procedencia del supuesto y evidencia observada.** Un resultado
   debería distinguir al menos: modelo gaussiano respaldado externamente,
   evidencia de forma no informativa, aproximación empíricamente calibrada y
   evidencia adversa. `acceptable` no debería significar simultáneamente todas
   esas cosas.
2. **Separar extremidad de influencia.** Mantener el detector descriptivo, pero
   calibrar una dimensión continua de impacto sobre media/SE/CI. La razón de
   influencia estudiada es candidata diagnóstica, no una regla lista para
   producción. También debe modelarse que P(al menos un extremo) crece con n.
3. **Representar riesgo del proceso generador.** La ausencia de un contaminante
   en la muestra no elimina una contaminación rara. Metadatos del diseño,
   conocimiento del mecanismo y análisis de sensibilidad deben poder degradar
   la confianza aun si skew/kurtosis muestral parecen benignos.
4. **Sustituir cortes aislados por evidencia combinada.** Investigar un score
   continuo calibrado a error Tipo I/cobertura, con bandas de transición e
   incertidumbre diagnóstica. Luego mapearlo a tiers de acción; no asumir que un
   score por sí solo corrige el sesgo post-selección.
5. **Calibrar el significado de `caution`.** La salida debería informar riesgo
   estimado, soporte Monte Carlo y alternativas que preserven el estimando, no
   solo una categoría. Antes de cambiar thresholds se necesita confirmar las
   regiones exploratorias y comparar procedimientos de media robustos o
   resampling bajo los mismos procesos generadores.

## Limitaciones

- El estudio cubre inferencia one-sample IID de la media; no extrapola a
  diseños pareados, two-sample ni ANOVA.
- La independencia se declaró por diseño y no fue simulada bajo dependencia.
- Solo n=100 recibió confirmación exhaustiva para todas las contaminaciones;
  otros tamaños de esas familias son exploratorios.
- Los intervalos Wilson son marginales por tasa y no ajustan multiplicidad
  entre 368 celdas.
- Las tasas condicionadas describen el comportamiento operativo de una regla
  data-dependent. No son propiedades incondicionales de la familia y pueden
  exhibir sesgo de selección; precisamente por eso son necesarias para auditar
  la política.
- El contrafactual de retirar extremos no valida eliminación automática ni
  conserva necesariamente el mismo mecanismo muestral.

## Artefactos reproducibles

- `experiments/adversarial_robustness_calibration.py`
- `experiments/results/robustness_calibration_summary.csv`
- `experiments/results/robustness_calibration_metadata.json`
- `experiments/results/robustness_threshold_cliffs.csv`
- `experiments/results/robustness_calibration_replicates.csv.gz` (artefacto
  local comprimido, ignorado por Git)
- `tests/test_adversarial_robustness_calibration.py`

**CALIBRATION EVIDENCE READY**
