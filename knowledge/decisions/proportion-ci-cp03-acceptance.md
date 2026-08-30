# CP-03 — Aceptación del contrato de API y compatibilidad

**Stage:** `STAGE-PROP-CI-001`  
**Estado:** `accepted`  
**Fecha de aprobación:** 2026-08-30  
**Product Owner:** aprobación explícita recibida  
**Contrato aprobado:** `knowledge/decisions/proportion-ci-cp03-api-contract.md`  
**Baseline de producción:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`

## Decisión

El Product Owner aprueba CP-03 y habilita el avance a CP-04.

Quedan aceptados, dentro del alcance declarado por el contrato CP-03:

- preservación del constructor existente `PopulationProportionCI(...)`;
- `from_counts(successes, trials, ...)` como API agregada canónica;
- export público desde `pyMagicStat.inference` preservando la ruta histórica `.parametric`;
- métodos públicos futuros del stage: `wilson`, `clopper_pearson`, `wald`;
- Wilson como default preservado;
- Wald como legacy explícito, sin clipping y sin retirada programada en este stage;
- transición mediante `DeprecationWarning` para `incidences` numérico legacy;
- compatibilidad temporal de `incidences` fraccionario sólo para Wilson/Wald, marcada como fuera del contrato binomial;
- resultado dict compatible con claves legacy y metadata nueva;
- alcance exclusivamente bilateral;
- `BootstrapCI(stat="proportion")` separado;
- corrección fail-closed de `MethodSelector` para `Estimand.PROPORTION`, sin selección automática de Wilson ni transferencia de métodos de media;
- matriz de 71 tests contractuales exigidos para la futura implementación CP-05.

## Límites

Esta aceptación no autoriza implementación, calibración, publicación, PR, merge ni routing automático. CP-04 debe preregistrar la calibración antes de observar resultados finales destinados a justificar claims del proyecto.

## Siguiente checkpoint

`CP-04 — Diseño/preregistro de calibración`: habilitado para arquitectura.