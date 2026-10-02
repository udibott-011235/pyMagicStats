# DEC-026 — Research Acceleration Boundary

- Estado: `proposed`
- Fecha: 2026-09-29
- Owner: `decision-owner`
- Revisores: `statistical-software-architecture`, `implementation-engineering`,
  `adversarial-statistical-qa`
- Supersedes: ninguno
- Decisiones relacionadas: `DEC-015`, `DEC-016`, `DEC-024`, `DEC-025`

## Contexto

pyMagicStats necesita producir evidencia científica a escalas que pueden hacer
útil la aceleración CUDA, sin convertir esa conveniencia computacional en una
nueva semántica estadística, una dependencia de producción o un requisito de
la API pública. Esta decisión separa la superficie productiva de la
infraestructura experimental y fija cómo validar prospectivamente la evidencia
generada con aceleradores.

## Decisión normativa

```text
PRODUCTION_REFERENCE = NumPy/SciPy
CUDA_IS_PRODUCTION_BACKEND = NO
CUDA_IS_PUBLIC_API = NO
PUBLIC_API_REQUIRES_GPU = NO
PRODUCTION_TESTS_REQUIRE_GPU = NO
CUDA_IS_RESEARCH_INSTRUMENTATION = YES
CPU_REFERENCE_REMAINS_CANONICAL = YES
CUDA_MAY_GENERATE_LARGE_SCALE_EVIDENCE = YES
CUDA_EVIDENCE_REQUIRES_REFERENCE_VALIDATION = YES
```

La identidad que rige la arquitectura es:

```text
NumPy/SciPy = production + canonical scientific reference
CUDA/CuPy = research instrumentation only
```

La API pública, el código de producción, sus dependencias transitivas y sus
tests permanecen en el ecosistema Python/NumPy/SciPy aprobado. CUDA/CuPy no es
ni será interpretada por esta decisión como backend productivo, API pública,
fuente de nuevas semánticas científicas o requisito para instalar, utilizar o
validar producción.

CUDA puede acelerar exclusivamente calibraciones, simulaciones, estudios de
cobertura, estudios de error tipo I, estudios de potencia, bootstrap
experimental, Monte Carlo, estudios de robustez y generación de evidencia
científica. El harness y sus dependencias deben permanecer aislados del código
de librería.

## Clases de validación y experimento

1. **Software/reference validation.** Establece las semánticas canónicas y la
   corrección de la implementación NumPy/SciPy. No depende de GPU y no queda
   sustituida por velocidad, volumen de simulación ni concordancia CUDA.
2. **Accelerated exploratory/calibration experiments.** CUDA puede producir
   evidencia exploratoria o de calibración dentro de un claim, escenarios y
   métricas explícitos. Requiere validación contra la referencia y límites que
   impidan promover sus resultados a una afirmación confirmatoria no
   prerregistrada.
3. **Confirmatory scientific experiments.** El objetivo científico, el claim,
   el alcance de equivalencia y la estrategia de cross-validation CPU se
   prerregistran antes de ejecutar el acelerador. La interpretación
   confirmatoria sólo puede usar la evidencia que supere esos gates.

Estas clases son distintas: validar software no demuestra una propiedad
estadística; acelerar una calibración no redefine el procedimiento; y un
experimento confirmatorio no puede heredar suficiencia de una exploración no
prerregistrada.

## Equivalencia proporcional y cross-validation CPU

Toda evidencia CUDA debe declarar prospectivamente su referencia
NumPy/SciPy, el claim experimental, las operaciones y decisiones que requieren
equivalencia, los escenarios de comparación, métricas, tolerancias, tratamiento
de discrepancias y estrategia de cross-validation CPU. El diseño se congela
antes de la ejecución CUDA.

El grado de equivalencia CPU↔CUDA exigido es proporcional al claim:

- un claim sobre una operación o región limitada requiere evidencia para esa
  operación o región;
- un claim sobre decisiones científicas requiere equivalencia de esas
  decisiones, además de la fidelidad numérica pertinente;
- un claim global requiere cobertura global defendible.

La equivalencia universal no es requisito para utilizar CUDA como
infraestructura de investigación cuando el claim es explícitamente limitado.
Las zonas no validadas permanecen fuera de inferencia.

La cross-validation CPU es prospectiva: su muestreo, escenarios, cantidades,
seeds o identidades, métricas, criterios de aceptación y artefactos se fijan
antes de observar el resultado acelerado. Para evidencia confirmatoria debe
incluir los endpoints científicos decisivos y una muestra CPU suficiente para
el claim prerregistrado; no puede seleccionarse retrospectivamente sólo donde
CUDA concordó.

## Interpretación específica para CP05-C2C

CP05-C2C does not certify a CUDA production backend.

```text
CUDA_PRODUCTION_READINESS=NOT_APPLICABLE
```

Normativamente:

```text
CUDA_RESEARCH_TRUST_FOR_LIMITED_CLAIM
does not require
FULL_C2C_EQUIVALENCE
```

cuando `equivalence_scope` está prerregistrado, la cross-validation CPU está
prerregistrada, los gates correspondientes pasaron y el claim permanece
estrictamente dentro de ese scope.

```text
FULL_C2C_EQUIVALENCE
and
FULL_1152
```

son claims de cobertura más fuertes y separados. Su ausencia impide afirmar
equivalencia global, pero no bloquea automáticamente un claim research limitado
que haya sido validado correctamente bajo las condiciones anteriores.

## Discrepancias

Una explicación causal o numérica puede coexistir con una falta de equivalencia
de decisión:

```text
DISCREPANCY_EXPLAINED=YES
SCIENTIFIC_DECISION_EQUIVALENCE=NO
```

Explicar una discrepancia no la convierte en equivalencia científica. Una
discrepancia que cambia la decisión científica no puede eliminarse mediante
fuzzy tolerances, redondeo, reclasificación post hoc ni ampliación retrospectiva
del umbral. Debe conservarse en los artefactos, limitar o invalidar el claim y
volver a arquitectura cuando corresponda.

## Evidencia mínima

Un experimento acelerado registra como mínimo: referencia canónica;
identidad del acelerador y del harness; evidencia y alcance de equivalencia;
diseño prospectivo de cross-validation CPU; entorno; seeds e identidades RNG;
artefactos y digests; claim; denominadores; fallos; discrepancias; y
limitaciones. La velocidad nunca constituye validación estadística.
Performance may justify use of the research accelerator, but performance
evidence remains separate from equivalence and statistical validity.

## Relación con decisiones y evidencia históricas

DEC-026 redefine prospectivamente cómo se interpreta la suficiencia de C2C y
se apoya en las fronteras y lecciones de `DEC-015`, `DEC-016`, `DEC-024` y
`DEC-025`. No altera retrospectivamente `DEC-014`..`DEC-025`, R10-A ni `EV-021`,
ni reescribe sus scopes, gates, resultados o estados.

Se preserva expresamente:

```text
R10A_TARGETED_EQUIVALENCE=COMPLETE
R10A_EVIDENCE_STATUS=validated_with_limits
FULL_C2C_EQUIVALENCE=NOT_ESTABLISHED
FULL_1152_PERFORMED=NO
CP05_D=NOT_STARTED
HOLDOUT_ACCESSED=NO
```

## Límites y consecuencias

Esta decisión no autoriza implementar o ejecutar un harness, acceder a un
holdout, cambiar producción, ejecutar GPU/CUDA, iniciar R11, ejecutar la matriz
1152 ni comenzar CP05-D. Tampoco declara equivalencia para operaciones o claims
que no cuenten con evidencia proporcional y prerregistrada.

## Condición que obliga a revisar

Requiere una nueva decisión cualquier propuesta de hacer CUDA/CuPy parte de la
API, producción, dependencias productivas o tests de producción; cambiar la
referencia canónica; ampliar un claim más allá de su equivalence scope; o
modificar después de observar resultados los gates de equivalencia o
cross-validation.

## Impacto en API, código, tests y documentación

No cambia API, código productivo, código experimental ni tests de ejecución.
Coordina exclusivamente gobernanza, protocolo, prompts, perspectivas de rol,
índice de decisiones y registro de la Knowledge Base.
