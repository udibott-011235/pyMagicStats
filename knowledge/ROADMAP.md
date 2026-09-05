# pyMagicStats — Roadmap operativo y trabajo pendiente

**Estado:** activo  
**Actualizado:** 2026-09-04  
**Rama documental:** `docs/proportion-ci-stage`  
**Owner:** Product Owner

Este documento es la **vista humana consolidada del trabajo pendiente**. No sustituye los registros formales de decisiones, evidencia o experimentos: enlaza esos registros para que el estado del proyecto pueda leerse desde un único punto.

## Estado actual

### STAGE-PROP-CI-001 — intervalos de una proporción

- CP-01 — censo/contrato actual: **complete**.
- CP-02 — especificación estadística: **complete**.
- CP-03 — contrato API/compatibilidad: **complete**.
- CP-04 — preregistro de calibración: **complete / frozen**.
- CP-05 — candidato productivo: **complete / frozen** sobre `2df5b90a5395163e723f9c52aafbb91fdce96d43`.
- CP-06 — calibración/evidencia: **in_progress**.
  - CP06-A harness validation: **complete**.
  - CP06-B deterministic sweep: **PASS**.
  - CP06-C exhaustive `n=1..5000`: **PASS**.
  - CP06-D stress `n=7,500..1,000,000`: **PASS**.
  - CP06-E adversarial minima search: **running** sobre harness congelado `3f41d4ea0d193968c0bbe7080e49cdbe784bf1ac`.
  - CP06-F high-precision adjudication: **pending**.
  - Shadow Monte Carlo preregistrado: **pending**.
  - Holdout preregistrado: **pending**.
- CP-07 — auditoría adversarial de SHA/evidencia exactos: **pending**.
- CP-08 — interpretación final + decisión explícita de integración: **pending**.

### Evidencia ya obtenida en CP06

- Clopper–Pearson: cobertura exacta/conservadora sin violaciones del gate observado hasta el stress de `n=1,000,000`.
- Wilson: estructura limpia en bounds, monotonicidad, nesting, simetría y concordancia con oráculos; los casos extremos sospechosos quedan pendientes de adjudicación HP en CP06-F.
- Wald: implementación fiel a la fórmula legacy; patologías estructurales caracterizadas y no tratadas como bug de implementación.
- Jeffreys: comparador bayesiano no productivo; no se le transfieren garantías frecuentistas.
- Reproducibilidad: el inventario de intervalos de `n=100000` produjo hash idéntico bajo batching distinto (`512` vs `256`).

CP06 **caduca** cuando E/F + shadow MC + holdout estén cerrados según CP04, CP07 audite el SHA/evidencia exactos y CP08 produzca la decisión del Product Owner. No se amplía el dominio por inercia una vez satisfechos los criterios preregistrados.

## Bloqueantes antes del Manual UAT 1

### B1 — cerrar STAGE-PROP-CI-001

Completar CP06, CP07 y CP08 sin transferir PASS a un SHA distinto.

### B2 — cierre estadístico de ANOVA

Pendiente validar específicamente el estadístico ANOVA y su diseño, incluyendo:

- estimando, diseño y unidad independiente;
- oráculos independientes;
- heterocedasticidad;
- residuos/errores en el diseño correcto;
- calibración propia de Type-I error/cobertura o evidencia equivalente;
- grupos desbalanceados, tamaños pequeños y desviaciones relevantes de normalidad;
- política fail-closed;
- comparación focal con SciPy/statsmodels y fixtures externos cuando aporte evidencia.

La calibración de medias/t-tests no se transfiere a ANOVA.

### B3 — accuracy closure de distribuciones/GOF expuestos

Hacer un censo del API realmente expuesto y cerrar, por componente incluido:

- construcción/representación;
- parámetros y descriptivos;
- PDF/PMF/CDF/quantiles cuando existan;
- soporte y límites;
- invariantes;
- GOF/diagnóstico asociado;
- boundaries e inputs inválidos;
- oráculos/definiciones reproducibles;
- tolerancias float64 y HP sólo cuando sea necesario.

Gate 2 es evidencia de entrada, no una declaración de que toda la superficie de distribuciones está validada.

### B4 — congelar inventario del baseline UAT

Antes del UAT debe existir una tabla por método con API, estimando, diseño, estado de validación, evidencia, límites, inputs, outputs/metadata y autorización o exclusión para uso manual.

## MANUAL UAT CHECKPOINT 1 — CURRENT STATISTICAL CORE

Después de B1–B4 se ejecutará un UAT manual con métodos invocados explícitamente sobre fixtures controlados y DataFrames realistas/sucios. Debe separar tres capas:

1. matemática;
2. implementación/API;
3. manejo de DataFrame real.

El selector/orquestador queda fuera. Un PASS habilita sólo uso manual de los módulos incluidos y dentro de sus límites documentados.

## Deuda posterior al Manual UAT 1

- **P1 — nuevas distribuciones:** agregar y validar familias según prioridad de uso.
- **P2 — transformaciones:** dominio, invertibilidad, ceros/negativos, parámetros, estimando y leakage.
- **P3 — no paramétricos:** exact/asymptotic, ties, paired/independent, effect sizes y límites; nunca fallback universal.
- **P4 — regresiones:** especificación, residuos, colinealidad, heterocedasticidad, influencia, intervalos/tests, predicción, encoding y missing data.
- **P5 — DOE:** aleatorización, replicación, bloques, interacciones, aliasing/confounding, análisis y diagnóstico.
- **P6 — decision engine/orquestador:** sólo después de una baseline manual confiable y decisión explícita del Product Owner.

## Deuda transversal abierta

`DEBT-001 / TD-NUM-001` — escalas subnormales float64 en DataQualityAssessment. Permanece abierta y no bloqueante para el dominio retail/BI mientras ningún caso del baseline dependa de esas escalas; si aparece uno, se reclasifica.

## Orden de trabajo

```text
STAGE-PROP-CI-001
  CP06 -> CP07 -> CP08
        |
        v
ANOVA statistical closure
        |
        v
distribution / GOF accuracy closure
        |
        v
freeze Manual UAT 1 inventory
        |
        v
MANUAL UAT CHECKPOINT 1
        |
        +--> manual operational use
        |
        +--> P1..P5 toolbox expansion
        |
        `--> P6 decision engine only by explicit PO decision
```

## Registros formales relacionados

- Stage de proporciones: [`decisions/proportion-ci-stage.md`](decisions/proportion-ci-stage.md)
- Preregistro CP04: [`experiments/proportion-ci-cp04-preregistration.md`](experiments/proportion-ci-cp04-preregistration.md)
- Manual UAT 1 / DEC-007: [`decisions/manual-uat-checkpoint-1.md`](decisions/manual-uat-checkpoint-1.md)
- Evidencia: [`evidence/`](evidence/)
- Registro estructurado: [`registry.json`](registry.json)

## Regla de actualización

Esta vista debe actualizarse cuando cambie el estado de un checkpoint o bloqueante. Las afirmaciones formales siguen requiriendo su registro de evidencia/decisión y SHA exacto cuando corresponda.