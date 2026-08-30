# EV-PROP-CI-CP01 — Censo del contrato actual de intervalos de proporción

**Estado:** `validated_with_limits`  
**Fecha:** 2026-08-30  
**Baseline auditado:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`  
**Stage documental observado:** `docs/proportion-ci-stage` @ `5906a834ce070c0596f8a9efd9bddc813953e90c`  
**Rol ejecutor:** adversarial-statistical-qa  
**Naturaleza:** censo read-only; no constituye aprobación de implementación ni calibración estadística.

## Resultado de salida

`CP-01: COMPLETE — READY FOR ARCHITECTURE`

El auditor confirmó `origin/main` y el SHA analizado en `402e4601df460811779b3238c2526ac12f463a67`, checkout en detached HEAD, árbol limpio y ausencia de modificaciones locales. La gobernanza, registry, teoría y rol adversarial fueron cargados antes del análisis.

## Contrato reconstruido

`PopulationProportionCI` implementa intervalos bilaterales para una proporción con:

- Wilson como método predeterminado;
- Wald sólo mediante `method="wald"`;
- datos binarios raw;
- conteo agregado mediante `incidences`;
- clasificación mediante callable/predicate;
- `n` derivado siempre de `len(data)`;
- metadata legacy `successes >= 10 and failures >= 10`;
- resultado basado en diccionario.

La clase no se exporta desde `pyMagicStat.inference`, no está conectada a `MethodSelector` y convive con una ruta separada `BootstrapCI(stat="proportion")` con contrato distinto.

## Evidencia numérica reproducida

Para Wilson se comparó un grid determinista de 60,900 combinaciones con `alpha in {0.01, 0.05, 0.10}`, `n=1..200` y `x=0..n`.

Resultados reportados:

- diferencia máxima contra SciPy Wilson: `3.33e-16`;
- error máximo de simetría complementaria: `3.33e-16`;
- cero violaciones de monotonicidad;
- cero límites fuera de `[0,1]`;
- exclusiones aparentes del estimador únicamente por redondeo <= `2.22e-16`, ninguna mayor que `1e-14`.

Esto demuestra equivalencia algebraica y propiedades numéricas básicas en el grid observado. **No demuestra cobertura frecuentista calibrada.**

Para Wald, 4,141 de las 60,900 combinaciones del mismo grid produjeron límites fuera de `[0,1]`; `x=0` y `x=n` generan intervalos degenerados. El warning basado en éxitos/fracasos menores que 10 no impide el cálculo.

## Hallazgos principales

| ID | Severidad | Tipo | Resumen |
|---|---|---|---|
| CP01-F-001 | MAJOR | statistical-contract gap | `incidences` acepta éxitos fraccionarios, por ejemplo `3.7/10`, incompatibles con una realización binomial ordinaria. |
| CP01-F-002 | MAJOR | statistical-contract gap | Independencia, probabilidad común y procedencia del diseño no están representadas en el contrato. |
| CP01-F-003 | MAJOR | software defect | `Estimand.PROPORTION` puede recibir `selected_method="one_sample_t"` desde el routing genérico, con semántica de media incompatible con el estimando. |
| CP01-F-004 | MODERATE | compatibility risk | En modo agregado, `n=len(data)` aunque los valores de `data` son ignorados. |
| CP01-F-005 | MODERATE | calibration gap | No existe calibración estadística, evidencia adversarial ni record aceptado específico para cobertura de Wilson/Wald de producción. |
| CP01-F-006 | MODERATE | legacy debt | Wald puede salir de `[0,1]` y degenerar en boundaries. |
| CP01-F-007 | MODERATE | statistical-contract gap | `successes>=10 and failures>=10` se presenta como adecuación de aproximación normal sin calibración específica y puede interpretarse como garantía. |
| CP01-F-008 | MODERATE | statistical-contract gap | `BootstrapCI(stat="proportion")` mantiene una superficie paralela y no validada específicamente para proporciones. |
| CP01-F-009 | MINOR | documentation gap | Export público ausente y documentación/tests mínimos. |
| CP01-F-010 | MINOR | compatibility risk | Tipos aceptados por `incidences` son más amplios que la anotación/documentación. |

## Cobertura de tests encontrada

Sólo se identificaron dos tests directos de `PopulationProportionCI`: uno de fórmula Wilson/default sobre una muestra Bernoulli simulada y otro que protege `incidences=0`. No existe cobertura directa suficiente para Wald, `x=n`, `n=1`, niveles de confianza distintos, inputs adversariales, callable, conteos fraccionarios, simetría, monotonicidad, cobertura frecuentista o la ruta bootstrap de proporción.

## Evidencia estadística encontrada

No se encontró calibración formal específica de intervalos de proporción en `Docs/`, `knowledge/`, `experiments/` o `examples/`. La calibración de medias, Empirical Likelihood, ANOVA, GOF Binomial, robustez y los Wilson usados para incertidumbre Monte Carlo no se transfieren a este contrato.

## Oráculos disponibles sin cambiar dependencias

El entorno actual ya dispone de referencias para validación/calibración:

- SciPy: Wilson, Wilson con corrección, Clopper-Pearson/exact y distribución binomial;
- statsmodels: Wald/normal, Wilson, Clopper-Pearson/beta, Agresti-Coull, Jeffreys y binomial-test inversion.

Estos recursos son oráculos potenciales; no constituyen una decisión de implementación.

## Límite de esta evidencia

Este record documenta el estado factual de CP-01. No aprueba métodos nuevos, no autoriza routing automático, no decide compatibilidad y no convierte equivalencia numérica con un oráculo en calibración estadística.