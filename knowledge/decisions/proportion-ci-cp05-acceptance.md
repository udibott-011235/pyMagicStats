# CP-05 — Aceptación del candidato de implementación de intervalos de proporción

**Stage:** `STAGE-PROP-CI-001`  
**Estado:** `accepted`  
**Fecha:** 2026-08-30  
**Rol de revisión:** statistical-software-architecture  
**Baseline de producción:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`  
**Rama candidata:** `feature/proportion-ci-contract`  
**SHA aceptado/congelado:** `2df5b90a5395163e723f9c52aafbb91fdce96d43`

## Decisión

Se acepta CP-05 como candidato de implementación para iniciar CP-06.

El SHA `2df5b90a5395163e723f9c52aafbb91fdce96d43` queda congelado como objeto de calibración. Toda modificación posterior de código productivo genera un SHA nuevo y requiere nueva revisión antes de transferir evidencia.

## Evidencia de implementación

El handoff de Cortex/Codex reporta:

- 77 tests contractuales verdes;
- suite completa `366 passed, 3 skipped, 1 warning, 0 failed`;
- worktree limpio;
- local HEAD y remote HEAD idénticos;
- no PR, merge, rebase ni modificación de `main`.

La revisión arquitectónica del SHA previo `c452a050c4f4856c2e49dbde889685768e964759` identificó únicamente `CP05-AR-001`, una brecha MINOR del test #28: el supuesto grid CP-01 estaba truncado a `n<=50`.

El SHA actual corrige únicamente ese hallazgo. GitHub confirma que el delta `c452a050..2df5b90` modifica exclusivamente `tests/test_proportion_ci_contract.py`, y el test #28 recorre ahora el grid completo preregistrado de CP-01:

```text
alpha in {0.01, 0.05, 0.10}
n = 1..200
x = 0..n
```

## Alcance aceptado

El candidato conserva:

- Wilson como default;
- Clopper–Pearson bilateral;
- Wald legacy sin clipping;
- `from_counts(successes,trials)` sin dummy data;
- metadata y deprecaciones CP-03;
- export público desde `pyMagicStat.inference` preservando `.parametric`;
- fail-closed de `MethodSelector` para `Estimand.PROPORTION`;
- ausencia de capability automática de proporción;
- `calibration_status="not_calibrated"`;
- Bootstrap de proporción separado.

## Límites de la aceptación

Esta aceptación afirma conformidad de implementación con el contrato observado y suficiencia para iniciar calibración. No afirma cobertura estadística, validez universal, superioridad de método ni autorización de routing automático.

CP-06 debe ejecutar exclusivamente el preregistro aceptado de CP-04 sobre este SHA exacto.

No se autoriza PR ni merge por esta aceptación.

## Siguiente checkpoint

`CP-06 — Calibración y evidencia`: autorizado sobre `2df5b90a5395163e723f9c52aafbb91fdce96d43`.