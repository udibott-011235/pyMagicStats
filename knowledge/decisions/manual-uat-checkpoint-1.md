# DEC-007 — Manual UAT Checkpoint 1 for the current statistical core

- Estado: `accepted`
- Fecha: `2026-08-31`
- Owner: `decision-owner`
- Arquitectura: `statistical-software-architecture`
- Revisores: `implementation-engineering`, `adversarial-statistical-qa`
- Supersedes: ninguno

## 1. Decisión

Se fija un checkpoint transversal de proyecto denominado:

> **MANUAL UAT CHECKPOINT 1 — CURRENT STATISTICAL CORE**

Este checkpoint **no significa que pyMagicStats esté completo**, ni que el toolbox estadístico esté cerrado. Su propósito es detener el desarrollo incremental en un punto controlado, validar operacionalmente las herramientas estadísticas ya incluidas en el alcance actual y establecer una baseline manual confiable antes de introducir un orquestador/decision engine.

La condición que se quiere demostrar no es “la librería ya tiene todos los métodos”, sino:

> Las herramientas estadísticas incluidas en el alcance del checkpoint han cerrado sus problemas estadísticos conocidos, producen resultados correctos cuando se invocan explícitamente y pueden operar sobre un DataFrame real/sucio sin mezclar todavía errores de selección automática con errores matemáticos, de implementación o de manejo de datos.

## 2. Motivación

La prueba de uso debe preceder al orquestador porque un resultado incorrecto bajo selección automática puede provenir de capas distintas:

1. error o limitación estadística del método;
2. error de implementación/API de la herramienta;
3. problema de representación, limpieza o validación del DataFrame;
4. error de selección/routing del orquestador.

El Manual UAT 1 elimina deliberadamente la cuarta capa. Cada método se invoca de forma explícita. De esta manera, si una prueba falla, la causa puede localizarse entre matemática, implementación/API y manejo de datos antes de añadir la complejidad del routing.

## 3. Alcance del checkpoint

El checkpoint cubre **las herramientas que estén implementadas y estadísticamente aceptadas al momento de congelar el baseline UAT**, no familias futuras todavía ausentes/incompletas.

Como mínimo, el baseline debe incluir las capacidades actuales que sobrevivan sus respectivos gates:

- contenedores/descriptivos de distribución expuestos en el baseline;
- pruebas/diagnósticos de distribución y GOF ya integrados, dentro del alcance realmente validado;
- inferencia de media y sus métodos ya aceptados/limitados;
- pruebas t/Student/Welch dentro de sus contratos aprobados;
- bootstrap únicamente donde exista contrato reproducible y estimando explícito;
- empirical likelihood únicamente dentro de su alcance documentado y sin transferirle claims que no posee;
- intervalos de una proporción cuando `STAGE-PROP-CI-001` finalice CP-06/07/08;
- ANOVA one-way únicamente después de cerrar su validación estadística propia; los diagnósticos de residuos existentes no bastan para incluir el estadístico ANOVA en el baseline.

El censo final de métodos UAT se congelará inmediatamente antes de ejecutar el checkpoint. Una función presente en el repositorio pero no validada no entra automáticamente al baseline.

## 4. Entrada obligatoria al Manual UAT 1

### 4.1 Bloqueante B1 — cierre del stage de proporciones

`STAGE-PROP-CI-001` debe completar:

- CP-06: calibración preregistrada completa, incluyendo barrido determinista, stress, búsqueda adversarial, high-precision audit, Monte Carlo shadow y holdout según CP-04;
- CP-07: auditoría adversarial sobre SHA y evidencia exactos;
- CP-08: decisión explícita de alcance/integración del Product Owner.

No basta con que Wilson/CP/Wald tengan tests unitarios verdes. Los claims finales deben ser los que sobrevivan la calibración y la auditoría.

### 4.2 Bloqueante B2 — cierre estadístico de ANOVA

La evidencia actual de residuos para diseño one-way se considera insuficiente para validar el estadístico ANOVA. Antes del UAT deben quedar resueltos, como mínimo:

- contrato del estimando/diseño y unidad independiente;
- validación del estadístico one-way contra oráculos independientes;
- tratamiento explícito de heterocedasticidad y límites de aplicabilidad;
- evaluación de residuos/errores en el diseño correcto, no normalidad marginal indiscriminada;
- calibración propia de Type-I error/cobertura o evidencia equivalente para las variantes que se pretendan exponer;
- comportamiento bajo grupos desbalanceados, varianzas distintas, tamaños pequeños y desviaciones de normalidad relevantes;
- política fail-closed para métodos todavía no calibrados;
- comparación focal contra herramientas externas cuando sea útil (por ejemplo SciPy/statsmodels y fixtures JASP/Minitab), sin convertir coincidencia por mayoría en autoridad matemática.

La calibración de medias/t-tests no se transfiere a ANOVA.

### 4.3 Bloqueante B3 — accuracy closure de distribuciones/GOF actualmente expuestos

Antes del UAT se hará un censo del API de distribución realmente expuesto en el baseline y se cerrará la exactitud de cada componente incluido. El gate debe distinguir:

- construcción/representación del objeto de distribución;
- parámetros y estadísticos descriptivos;
- CDF/PMF/PDF/quantiles cuando existan en el API;
- límites y soporte;
- invariantes estructurales;
- GOF/diagnóstico asociado;
- casos de boundary y datos inválidos;
- equivalencia con un oráculo independiente o definición matemática reproducible;
- tolerancias float64 explícitas y high precision sólo donde sea necesario.

Gate 2 y GOF ya integrados constituyen evidencia de entrada, pero **no equivalen por sí solos a declarar completa toda la superficie de distribuciones del repositorio**. El censo final decide qué funciones están realmente cubiertas.

### 4.4 Bloqueante B4 — censo de métodos del baseline UAT

Antes de la prueba se debe producir una tabla congelada por método con:

- nombre/API pública;
- estimando o cantidad calculada;
- diseño/población aplicable;
- estado de validación;
- evidencia/oráculo;
- límites conocidos;
- inputs válidos/invalidables;
- outputs y metadata esperados;
- si se permite uso manual operativo;
- si queda excluido del checkpoint.

No se incorporarán métodos “porque existen en código”.

## 5. Contrato de la prueba de uso

El UAT será una prueba operacional, no otra calibración estadística masiva y no una prueba del selector.

### 5.1 Datos

Usar al menos un DataFrame realista de trabajo con suciedad intencional y una copia/reference fixture controlada. Debe cubrir, donde tenga sentido:

- `NaN`/missing values;
- dtype numérico correcto y columnas numéricas representadas como texto;
- valores constantes o casi constantes;
- grupos pequeños y grupos desbalanceados;
- categorías vacías/no observadas cuando el API las reciba;
- outliers/extremos;
- valores fuera del soporte de un método;
- duplicados cuando sean plausibles en datos transaccionales;
- orden de filas alterado;
- índices no consecutivos;
- mezcla de columnas relevantes e irrelevantes;
- errores deliberados de input que deban fallar cerrado.

La prueba no debe “limpiar mágicamente” datos para hacer pasar el método. Debe distinguir claramente qué limpieza es responsabilidad del usuario, qué coerción está permitida por contrato y qué condición debe producir error/warning/UNKNOWN.

### 5.2 Ejecución

Los métodos se invocan **manualmente y de forma explícita**. No se permite que `MethodSelector`, un futuro decision engine o una regla heurística elija el método que se evalúa.

Cada caso UAT debe conservar:

- input/caso canónico o hash del dataset;
- método exacto invocado;
- parámetros;
- resultado pyMagicStats;
- resultado/reference cuando exista;
- diferencia/tolerancia;
- warnings/errors;
- metadata de supuestos/limitaciones;
- clasificación PASS / PASS_WITH_LIMITS / FAIL / NOT_APPLICABLE / UNKNOWN.

### 5.3 Tres capas que deben quedar separadas

#### Capa 1 — matemática

¿El método aislado produce el resultado estadísticamente correcto dentro de su contrato?

#### Capa 2 — implementación/API

¿La función acepta inputs válidos, rechaza inputs inválidos y devuelve una salida estable, interpretable y reproducible?

#### Capa 3 — DataFrame real

¿El método se comporta correctamente cuando los datos provienen de un flujo real, con missing values, tipos imperfectos y otros problemas operacionales, sin ocultar violaciones de contrato?

El orquestador será una cuarta capa futura y queda fuera de este checkpoint.

## 6. Criterio de salida del checkpoint

El checkpoint puede cerrarse como `validated_with_limits` cuando:

1. todos los bloqueantes B1–B4 estén cerrados;
2. el baseline de métodos UAT esté congelado;
3. ningún método incluido tenga un hallazgo estadístico/numérico bloqueante abierto;
4. los casos UAT manuales produzcan resultados compatibles con sus referencias y contratos;
5. errores de datos produzcan comportamiento explícito y no resultados silenciosamente incorrectos;
6. cualquier limitación sobreviviente quede documentada por método;
7. Antigravity realice una pasada adversarial sobre el UAT/baseline exacto;
8. ChatGPT interprete los hallazgos separando matemática, implementación y datos;
9. el Product Owner acepte el alcance operativo resultante.

## 7. Significado del PASS

Un PASS de Manual UAT 1 autoriza únicamente esta afirmación:

> pyMagicStats puede utilizarse manualmente para los módulos incluidos en el baseline UAT y dentro de sus límites documentados.

No autoriza afirmar que:

- la librería está completa;
- todos los métodos estadísticos deseables existen;
- toda función presente en el repositorio está validada;
- el selector/orquestador es correcto;
- una herramienta validada para un estimando transfiere evidencia a otro;
- nuevas distribuciones, transformaciones, no paramétricos, regresiones o DOE están cubiertos.

## 8. Uso operativo posterior

Después del checkpoint, el Product Owner podrá usar manualmente los métodos aceptados en trabajo diario. La selección del método sigue siendo deliberada/expresa. Los resultados deberán respetar las limitaciones conocidas y estados fail-closed de la librería.

Este uso operativo manual sirve además como fuente de nuevos casos reales y regresiones, pero un caso de producción observado no sustituye una validación estadística formal cuando se incorpore una familia nueva.

## 9. Deuda posterior deliberadamente fuera del hito

Las siguientes líneas permanecen abiertas después del Manual UAT 1 y **no bloquean** el checkpoint salvo que una función de esa familia sea incluida explícitamente en el baseline UAT:

### P1 — expansión de distribuciones

Agregar y validar nuevas familias de distribución según prioridad de uso. Cada familia deberá tener soporte, parametrización, invariantes, oráculos y GOF claramente delimitados. No se asumirá que Gate 2 valida distribuciones todavía no incorporadas.

### P2 — transformaciones

Definir e implementar transformaciones estadísticas con contrato explícito: dominio, invertibilidad cuando aplique, manejo de ceros/negativos, estimación de parámetros, efecto sobre el estimando y prevención de leakage si se usan en pipelines.

### P3 — métodos no paramétricos

Incorporar pruebas/estimadores no paramétricos por estimando y diseño, con manejo de ties, small-sample/exact versus asymptotic behavior, paired/independent structure, effect sizes y límites de interpretación. “No paramétrico” no se tratará como fallback universal ante cualquier fallo de normalidad.

### P4 — regresiones

Censar y endurecer la superficie de regresión existente y futura. La presencia de código en `pyMagicStat/models/regression.py` no equivale a validación. El stage correspondiente deberá cubrir especificación del modelo, diseño, residuos, colinealidad, heterocedasticidad, influencia, intervalos/tests, predicción, encoding, missing data y referencias independientes según la familia incorporada.

### P5 — DOE

Diseñar DOE como familia propia: estructura del diseño, aleatorización, replicación, bloques, interacciones, aliasing/confounding, análisis y diagnóstico. No debe construirse como una simple extensión del ANOVA one-way.

### P6 — orquestador / decision engine

Permanece deliberadamente diferido hasta disponer de una baseline manual de herramientas confiables. El Manual UAT 1 es prerequisito necesario, no suficiente: el Product Owner decidirá qué amplitud del toolbox debe existir antes de activar un stage de routing.

Cuando se abra ese stage, la prueba del orquestador reutilizará los casos manuales como golden baseline: si el resultado del pipeline automático difiere, se podrá separar un error de selección de un error de cálculo.

## 10. Deuda numérica existente no bloqueante

`DEBT-001 / TD-NUM-001` (escalas subnormales float64 en DataQualityAssessment) permanece abierta y no bloquea el Manual UAT 1 para el dominio retail/BI mientras no aparezca un caso del baseline que dependa de esas escalas. Si aparece, se reclasifica como bloqueante.

## 11. Orden de trabajo desde el estado actual

```text
STAGE-PROP-CI-001
  CP-06 -> CP-07 -> CP-08
        |
        v
ANOVA statistical closure
        |
        v
current distribution/GOF API accuracy closure + baseline census
        |
        v
freeze Manual UAT 1 method inventory
        |
        v
MANUAL UAT CHECKPOINT 1 — CURRENT STATISTICAL CORE
        |
        +--> manual operational use of validated modules
        |
        +--> new distributions / transformations / nonparametrics /
             regressions / DOE / other toolbox stages
        |
        `--> decision-engine stage only after explicit PO decision
```

El orden entre ANOVA y el accuracy closure final de distribuciones puede cambiar por conveniencia de ejecución, pero ambos son bloqueantes antes de congelar el baseline UAT.

## 12. Regla para todos los agentes

Al mencionar este checkpoint, los agentes deben usar lenguaje de alcance limitado. Las expresiones “toolbox completa”, “librería validada” o equivalentes quedan prohibidas salvo evidencia futura que realmente cubra ese alcance.

La denominación canónica es:

`MANUAL UAT CHECKPOINT 1 — CURRENT STATISTICAL CORE`

El checkpoint es una pausa de verificación y una puerta a uso manual productivo de los módulos validados; no es el final del desarrollo estadístico de pyMagicStats.
