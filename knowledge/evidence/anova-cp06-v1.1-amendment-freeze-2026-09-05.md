# CP-ANOVA-06 preregistration v1.1 — identity amendment freeze

- Fecha: `2026-09-05`
- Stage: `STAGE-ANOVA-001`
- Checkpoint: `CP-ANOVA-06`
- Rama: `audit/anova-calibration-preregistration`
- Estado: `complete/frozen`
- Version efectiva: `anova-calibration-prereg-v1.1`

## Motivo

CP-ANOVA-07A preflight detectó dos omisiones legítimas en v1:

1. faltaban `cell_id` autorizados para fases materializadas por cross-product;
2. E0 no tenía subconjunto exacto congelado.

No se había ejecutado E0, Phase D ni Phase H, por lo que la aclaración ocurre antes de cualquier evidencia Monte Carlo.

## Resolución

Fuente normativa añadida:

`knowledge/experiments/anova-cp06-cell-id-e0-amendment-v1.1.md`

La enmienda:

- materializa/reglamenta 197 IDs estables totales;
- fija E0 en 12 celdas / 2,400 datasets;
- conserva íntegramente diseño estadístico, seeds, replicaciones, criterios y holdout de v1;
- exige unicidad y validación estructural de IDs al cargar el harness.

El manifest fue restaurado tras un commit intermedio defectuoso de edición y el estado vigente de la rama ya no contiene el placeholder. El commit intermedio no es una fuente válida de configuración.

## Gobernanza

Cortex puede reanudar CP-ANOVA-07A leyendo en conjunto:

1. preregistro v1;
2. manifest restaurado;
3. amendment v1.1;
4. este freeze note.

Si existe conflicto entre la identificación incompleta de v1 y la enmienda v1.1, **v1.1 gobierna exclusivamente `cell_id` y E0**. Para todo aspecto estadístico restante gobierna el preregistro original.

Phase H sigue sellada. No se autoriza ejecutar E0/D/H hasta la revisión arquitectónica del harness.
