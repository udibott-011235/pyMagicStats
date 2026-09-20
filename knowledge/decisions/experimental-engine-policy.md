# DEC-015 — Experimental-engine policy and execution boundary

- Estado: `accepted`
- Fecha: 2026-09-20
- Owner: `decision-owner`
- Revisores: `statistical-software-architecture`, `adversarial-statistical-qa`
- Supersedes: ninguno

## Contexto y estimando

La corrección de una implementación y la calibración estadística de un
procedimiento son preguntas distintas. Confundirlas antes de comprometer una
campaña Monte Carlo puede convertir una implementación de referencia correcta
en un motor experimental inapropiado, o convertir evidencia de ingeniería en
una reclamación de propiedades estadísticas. La distinción aplica también a
GOF de distribuciones, intervalos de confianza, bootstrap, Empirical
Likelihood, ANOVA, robustez de muestreo y futuros estudios Monte Carlo.

## Decisión y razón

Se canonizan cuatro gates independientes.

### IMPLEMENTATION_VALIDATION_GATE

Pregunta: **¿pyMagicStats implementa correctamente el procedimiento
estadístico especificado?** La evidencia incluye tests unitarios, oráculos de
referencia, invariantes matemáticos, comparaciones entre bibliotecas,
semántica de fallos y RNG, y QA adversarial. Su objeto es el producto, no una
propiedad poblacional.

### STATISTICAL_CALIBRATION_GATE

Pregunta: **¿cómo se comporta el procedimiento definido bajo poblaciones
conocidas?** Type-I, cobertura, comportamiento finito, aplicabilidad, tasas
de fallo y, sólo con autorización separada, potencia, pertenecen a este gate.
Su objeto es la abstracción estadística; no vuelve a demostrar que el código
de producción funciona.

### EXPERIMENT_ENGINE_DECISION_GATE

Antes de un Monte Carlo costoso se debe decidir explícitamente qué se mide,
qué implementación sirve de referencia, qué motor experimental se usará, qué
gate de equivalencia necesita y qué piloto de coste medir. El flujo obligatorio
es: especificación estadística → validación de referencia → selección del
motor → equivalencia referencia↔experimental → piloto pequeño → calibración
grande → auditoría independiente. No se inicia una calibración larga con el
código productivo sólo porque sea una referencia válida.

### AGENT_EXECUTION_BOUNDARY

Los agentes interactivos no son runtime, scheduler, monitor de experimentos
largos ni babysitter Monte Carlo. Un **Experiment Executor** debe ejecutar los
procesos independientes y checkpointables en un host autorizado (Quantum,
Laptop, CI/job runner u otro host controlado). Esta decisión no interviene ni
modifica experimentos que ya estén en ejecución.

## CP05-C2A

`experiments/distribution_gof/cuda_calibration/` es un motor experimental
aislado para una futura ejecución Quantum/RAPIDS. No es producción, API
pública, selector de métodos ni evidencia de calibración. Hasta un gate de
equivalencia preregistrado, todo artefacto declara
`engine=CUDA_RAPIDS_EXPERIMENTAL`, `production_engine=false`,
`equivalence_gate_passed=false` y `calibration_claim=false`.

Las tolerancias CPU↔CUDA permanecen `NOT_YET_PREREGISTERED`. CP05-C2B deberá
definir fixtures, celdas adversariales, cantidades de comparación, tolerancias
absolutas/relativas, equivalencia de decisión y clasificación de fallos antes
de ejecutar Quantum.

## Límites y consecuencias

DEC-014 no cambia: conserva fórmulas, parámetros, MLE/refit compuesto,
ineligibilidad NB, plus-one/ties y semántica RNG. CP05-C2A no ejecuta
calibración, matriz completa, CP05-D, holdout, benchmark GPU ni equivalencia
CPU↔GPU.

## Condición que obliga a revisar

Una modificación al estimando, semántica nula, fitting/refitting, RNG,
denominador, estadístico, clasificación de fallos o tolerancias de equivalencia
requiere decisión arquitectónica antes de ejecución.
