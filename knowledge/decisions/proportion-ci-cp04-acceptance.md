# CP-04 — Aceptación del preregistro de calibración

**Stage:** `STAGE-PROP-CI-001`  
**Estado:** `accepted`  
**Fecha de aprobación:** 2026-08-30  
**Product Owner:** aprobación explícita recibida mediante instrucción de proseguir  
**Preregistro aprobado:** `knowledge/experiments/proportion-ci-cp04-preregistration.md`  
**Baseline de producción:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`

## Decisión

El Product Owner aprueba el preregistro exhaustivo de CP-04 y habilita la preparación de CP-05.

Quedan fijados antes de observar resultados finales destinados a claims del proyecto:

- Wilson, Clopper–Pearson y Wald como métodos de producción bajo evaluación;
- Jeffreys únicamente como comparador bayesiano;
- siete niveles de confianza entre 80% y 99.9%;
- recorrido exhaustivo `n=1..5000` y stress hasta `n=1,000,000`;
- cuatro familias de puntos `p`, incluyendo grid interior, extremos logarítmicos, régimen `lambda=np` y puntos inducidos por endpoints;
- cobertura frecuentista por enumeración binomial determinista como autoridad primaria;
- expected width, undercoverage, conservadurismo, degeneración y masa Wald fuera de `[0,1]`;
- búsqueda adversarial de mínimos de cobertura entre endpoints;
- estratificación por `min(np,n(1-p))` sin convertirla en regla de routing;
- tiers preregistrados de déficit de cobertura;
- gates matemático-numéricos separados por método;
- auditoría con precisión arbitraria mínima de 80 dígitos para celdas materiales/sospechosas;
- shadow audit Monte Carlo de 256 millones de draws;
- holdout confirmatorio de 10,000 celdas generado únicamente después del freeze del SHA candidato;
- invariancia por shard/worker/batch/backend dentro de tolerancias explícitas;
- prohibición de modificar grid, tiers, tolerancias o holdout después de observar resultados desfavorables;
- prohibición de habilitar routing automático por la sola calibración de CP-06.

## Límite de autorización

Esta aceptación habilita únicamente la preparación/implementación de CP-05 mediante una autorización separada de implementación. No autoriza por sí sola ejecución de CP-06, apertura de PR, merge ni modificación de `main`.

La instrucción del Product Owner de `proseguir` se interpreta como autorización para preparar el handoff de CP-05, no como autorización para que Arquitectura implemente producción.