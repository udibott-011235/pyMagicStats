# CP-05 — Revisión arquitectónica del candidato c452a05

**Stage:** `STAGE-PROP-CI-001`  
**Estado:** `under_review`  
**Rol:** statistical-software-architecture  
**Candidato revisado:** `feature/proportion-ci-contract` @ `c452a050c4f4856c2e49dbde889685768e964759`  
**Baseline:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`  
**Naturaleza:** revisión estática/contractual previa a CP-06; no sustituye auditoría adversarial CP-07.

## Preflight

GitHub confirma que el candidato está exactamente un commit por encima del baseline autorizado y modifica únicamente:

- `pyMagicStat/inference/__init__.py`;
- `pyMagicStat/inference/parametric.py`;
- `pyMagicStat/inference/selector.py`;
- `tests/test_proportion_ci_contract.py`.

No se observan cambios incidentales en ANOVA, EL, GOF, Bootstrap, capability registry ni `main`.

## Conformidad observada

La revisión del código candidato no identificó desviaciones conocidas en los siguientes puntos del contrato CP-03:

- Wilson permanece como default y conserva la fórmula legacy;
- Clopper–Pearson bilateral usa la inversión beta aprobada y trata `x=0/x=n` como casos válidos;
- Wald permanece legacy, explícito y sin clipping;
- `from_counts(successes,trials)` exige conteos enteros y no fabrica `data` dummy;
- se preserva el constructor legacy y la ruta callable;
- `incidences` numérico integral/fraccionario emite la deprecación aprobada y los fraccionarios no se presentan como binomiales soportados;
- metadata `estimand/design/sampling_model/interval_kind/calibration_status` permanece explícita;
- `calibration_status="not_calibrated"` no fue promovido;
- export desde `pyMagicStat.inference` añadido preservando `.parametric`;
- `MethodSelector` corta `Estimand.PROPORTION` antes de la política de medias y devuelve `selected_method=None`, `NOT_CALIBRATED`, sin alternativas;
- no se registró capability automática de proporción;
- `BootstrapCI(stat="proportion")` permanece separado.

El handoff de implementación reporta `366 passed, 3 skipped, 1 warning` para la suite completa; esta revisión no trata ese resultado como calibración estadística.

## Hallazgo CP05-AR-001

**Severidad:** `MINOR`  
**Tipo:** test-contract coverage gap

El test `test_cp05_28_wilson_bounds_stay_in_unit_interval_on_cp01_subset` recorre:

- `alpha in {0.01,0.05,0.10}`;
- `n=1..50`;
- todos los `x=0..n`.

Sin embargo, el test obligatorio #28 de CP-03 exige mantener bounds `[0,1]` en el **grid de CP-01**. CP-01 define ese grid como 60,900 combinaciones con:

- `alpha in {0.01,0.05,0.10}`;
- `n=1..200`;
- `x=0..n`.

Por tanto, el test tiene el ID correcto pero cubre sólo un subconjunto del criterio aprobado. No se observó evidencia de que Wilson falle en `n=51..200`; el hallazgo es una brecha de cumplimiento del test contractual, no un defecto estadístico demostrado.

### Criterio de cierre

Modificar únicamente el test CP05-28 para recorrer `n=1..200` completo, ejecutar al menos:

1. `tests/test_proportion_ci_contract.py`;
2. full suite;
3. `git diff --check`;

crear un nuevo commit sin amend/rebase y publicar un nuevo SHA candidato.

Todo nuevo SHA invalida el estado favorable del SHA anterior hasta una nueva revisión focalizada.

## Observación no bloqueante sobre coerción legacy

El baseline histórico convertía `incidences` mediante `float(...)`, mientras el candidato restringe la ruta numérica a `numbers.Real`; por ello entradas no numéricas pero float-convertibles como `"1"` dejan de aceptarse. CP-03 define la compatibilidad transitoria para **`incidences` numérico** y no exige preservar strings float-convertibles. Se considera fuera del contrato soportado, no hallazgo bloqueante.

## Veredicto de arquitectura

`CONDITIONAL PASS — CP-05 permanece under_review`.

No se autoriza CP-06 sobre `c452a050c4f4856c2e49dbde889685768e964759` hasta cerrar CP05-AR-001 y revisar el nuevo SHA.

No se autoriza PR ni merge.