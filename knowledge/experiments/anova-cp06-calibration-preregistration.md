# CP-ANOVA-06 — Calibration preregistration

- Fecha: `2026-09-05`
- Stage: `STAGE-ANOVA-001`
- Rama: `audit/anova-calibration-preregistration`
- Base: `501e97db102567f2ee225da3dccb026b027667c8`
- CP-ANOVA-04 freeze lógico: `83bfe547563d977f1ed0dd0f43629c281744488c`
- Production remediation commit: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- `pyMagicStat/inference/anova.py` blob congelado: `2d00ae2a2812b8c390125fefe244dcb4830176c5`
- CP-ANOVA-05: `complete/frozen`
- Preregistro: `anova-calibration-prereg-v1`
- Manifest: `knowledge/experiments/anova-cp06-calibration-manifest.json`
- Naturaleza: diseño congelado antes de cualquier Monte Carlo de evidencia.

## 1. Pregunta estadística

Caracterizar y, donde el modelo lo permite, validar el comportamiento de error tipo I y potencia de **Classical one-way ANOVA** y **Welch one-way ANOVA** como métodos explícitos para grupos independientes.

Este stage **NO calibra ni habilita un selector automático**. `MethodSelector(ONE_WAY)` permanece `NOT_CALIBRATED`. Tampoco se usa `OneWayRobustness` histórico para filtrar, condicionar o seleccionar resultados.

La calibración responde separadamente:

1. ¿Classical mantiene nivel nominal bajo su modelo exacto gaussiano homocedástico?
2. ¿Welch mantiene nivel nominal bajo Gaussianidad con varianzas iguales o desiguales en diseños razonables?
3. ¿Cómo se degradan ambos métodos fuera del modelo exacto ante skew, heavy tails, contaminación y heterocedasticidad asociada al tamaño?
4. ¿Cuál es el trade-off de potencia entre ambos métodos en alternativas representativas?
5. ¿Los resultados generalizan a un holdout no usado durante implementación/ajuste del harness?

## 2. Correcciones respecto al piloto histórico

El piloto histórico `feature/anova-engine/experiments/anova_calibration.py` se conserva como evidencia exploratoria, no como calibración final.

CP-ANOVA-06 corrige explícitamente:

- no mezcla calibración del método con política de selección;
- no condiciona resultados a `acceptable/caution/insufficient`;
- no usa `classical_eligible` para declarar validez;
- usa muchas más repeticiones por celda;
- separa desarrollo y holdout;
- preregistra criterios antes de ejecutar;
- usa ambos métodos sobre **exactamente la misma muestra** por réplica;
- conserva contadores de discordancia Classical/Welch;
- exige paridad del harness con la API pública antes de la corrida de evidencia;
- hace la semilla independiente de workers, batching, sharding y orden de ejecución.

## 3. Fases congeladas

### Phase E0 — engineering pilot

- No es evidencia estadística.
- `200` replicaciones por celda sobre un subconjunto mínimo del manifest.
- Objetivo: accounting, resume, metadata, determinismo, performance y escritura de artefactos.
- Ninguna decisión sobre Type-I/power puede derivarse de E0.

### Phase D — development calibration

Tres tiers:

- `D-core-h0`: 50,000 replicaciones por celda.
- `D-robustness-h0`: 25,000 replicaciones por celda.
- `D-stress-h0`: 25,000 replicaciones por celda.
- `D-power-h1`: 20,000 replicaciones por celda y tamaño de efecto.

### Phase H — sealed holdout

No se ejecuta hasta que:

1. harness y agregación estén congelados;
2. Phase D haya terminado;
3. cualquier remediation derivada de Phase D haya sido cerrada y el candidate vuelva a congelarse;
4. Product Owner autorice abrir holdout.

Después de abrir Phase H no se cambian fórmulas, tolerancias, acceptance bands, familias, diseños, seeds ni agregación para “hacer pasar” resultados. Un fallo confirmatorio obliga a reabrir el stage y crear una nueva versión futura de preregistro/holdout.

## 4. Alpha

Cada réplica genera p-values una sola vez. Se reportan rejection rates en:

```text
alpha_grid = [0.01, 0.05, 0.10]
primary_alpha = 0.05
```

Sólo `alpha=0.05` forma parte del gate confirmatorio de esta versión. `0.01` y `0.10` son sensibilidad secundaria y no se usan para mover criterios post-hoc.

## 5. Distribuciones

Todas las familias se generan con media teórica 0 y varianza teórica 1 antes de aplicar multiplicadores de escala y mean offsets.

### Development families

- `normal`
- `gamma_shape_4`: `(Gamma(4)-4)/sqrt(4)`
- `gamma_shape_1`: exponencial estandarizada
- `lognormal_sigma_0p5`
- `lognormal_sigma_1p2`
- `student_t_df_5`: multiplicar por `sqrt((df-2)/df)`
- `student_t_df_3`: misma estandarización
- `laplace`: `scale=1/sqrt(2)`
- `mixture_symmetric_5pct_scale6`: 95% N(0,1), 5% N(0,6), dividir por `sqrt(0.95 + 0.05*36)`
- `contamination_asymmetric_5pct_loc10`: 95% N(0,1), 5% N(10,1), centrar por `0.5` y dividir por `sqrt(1 + 0.05*0.95*100)`

### Stress-only families

- `lognormal_sigma_1p5`
- `student_t_df_2p5` (varianza finita, colas extremadamente pesadas)
- `contamination_asymmetric_10pct_loc10`

### Holdout-only families

No aparecen en Phase D con esos parámetros:

- `gamma_shape_2`
- `lognormal_sigma_0p8`
- `student_t_df_7`
- `weibull_shape_1p5`
- `pareto_alpha_3p5`
- `beta_2_5`
- `contamination_asymmetric_2pct_loc10`

Para Weibull/Pareto/Beta se estandariza usando sus medias y varianzas teóricas, no usando mean/variance de la muestra simulada.

## 6. Diseños y heterocedasticidad

`sd_multipliers` son desviaciones estándar; por tanto `[1,2,4]` implica variance ratio 16.

### D-core-h0 — normal / gates primarios

#### Equal variance — 24 diseños

**k=2**

- `[5,5]`, `[10,10]`, `[30,30]`, `[100,100]`
- `[5,20]`, `[10,40]`, `[30,120]`

**k=3**

- `[5,5,5]`, `[10,10,10]`, `[30,30,30]`, `[100,100,100]`
- `[5,10,20]`, `[20,10,5]`
- `[5,30,30]`, `[30,5,5]`

**k=5**

- `[5,5,5,5,5]`, `[10,10,10,10,10]`, `[30,30,30,30,30]`, `[100,100,100,100,100]`
- `[5,8,12,20,30]`, `[30,20,12,8,5]`

**k=10**

- `[5]*10`, `[10]*10`, `[30]*10`

Todos con `sd_multipliers=[1]*k`.

#### Unequal variance — 18 diseños

**k=2**

- sizes `[5,20]` con sd `[4,1]` y `[1,4]`
- sizes `[10,40]` con sd `[4,1]` y `[1,4]`

**k=3**

- balanced `[5,5,5]`, `[10,10,10]`, `[30,30,30]` con sd `[1,2,4]`
- sizes `[5,10,20]` con sd `[4,2,1]` y `[1,2,4]`
- sizes `[10,30,60]` con sd `[4,2,1]` y `[1,2,4]`
- sizes `[5,30,30]`, sd `[4,1,1]`
- sizes `[30,5,5]`, sd `[4,1,1]`

**k=5**

- balanced n=10 y n=30 con sd `[1,1.5,2,3,4]`
- sizes `[5,8,12,20,30]` con sd `[4,3,2,1.5,1]`
- sizes `[5,8,12,20,30]` con sd `[1,1.5,2,3,4]`

**k=10**

- balanced n=10 con sd `[1,4,1,4,1,4,1,4,1,4]`

Los 42 diseños core se simulan con `normal`. Ambos métodos se calculan en cada réplica.

### D-robustness-h0 — 54 celdas

Cross-product exacto de 9 familias:

```text
gamma_shape_4
gamma_shape_1
lognormal_sigma_0p5
lognormal_sigma_1p2
student_t_df_5
student_t_df_3
laplace
mixture_symmetric_5pct_scale6
contamination_asymmetric_5pct_loc10
```

por 6 diseños:

1. sizes `[5,5,5]`, sd `[1,1,1]`
2. sizes `[30,30,30]`, sd `[1,1,1]`
3. sizes `[5,10,20]`, sd `[1,1,1]`
4. sizes `[5,10,20]`, sd `[4,2,1]`
5. sizes `[5,10,20]`, sd `[1,2,4]`
6. sizes `[10,10,10,10,10]`, sd `[1,1,1,1,1]`

### D-stress-h0

Escenarios explícitos, descriptivos/no gate:

1. normal, sizes `[2,2,2]`, sd equal
2. normal, sizes `[2]*5`, sd equal
3. normal, sizes `[2,5,20]`, sd `[8,2,1]`
4. normal, sizes `[2,5,20]`, sd `[1,2,8]`
5. lognormal sigma 1.5, sizes `[5,5,5]`, sd equal
6. Student-t df 2.5, sizes `[5,5,5]`, sd equal
7. asymmetric contamination 10%, sizes `[10,10,10]`, sd equal
8. normal, `k=20`, sizes `[5]*20`, sd equal
9. lognormal sigma 1.5, sizes `[5,10,20]`, sd `[4,2,1]`
10. Student-t df 2.5, sizes `[5,10,20]`, sd `[1,2,4]`

### D-power-h1

12 base cells, cada una con:

```text
delta_range = [0.25, 0.50, 1.00]
```

El vector de medias es:

```text
mu_i = delta_range * centered_linspace(-0.5, 0.5, k)
```

por lo que `max(mu)-min(mu)=delta_range` en unidades de la varianza base estandarizada. No se normaliza por pooled SD después de aplicar heterocedasticidad.

Base cells:

1. normal, k3 n=10 balanced, equal sd
2. normal, k3 n=30 balanced, equal sd
3. normal, k5 n=10 balanced, equal sd
4. normal, sizes `[5,10,20]`, equal sd
5. normal, sizes `[5,10,20]`, sd `[4,2,1]`
6. normal, sizes `[5,10,20]`, sd `[1,2,4]`
7. gamma shape 1, k3 n=10 equal sd
8. lognormal sigma 1.2, k3 n=10 equal sd
9. Student-t df3, k3 n=10 equal sd
10. asymmetric contamination 5%, k3 n=10 equal sd
11. Laplace, k5 n=10 equal sd
12. symmetric mixture, k3 n=30 equal sd

## 7. Holdout congelado

### H-core-normal — confirmatory

50,000 replicaciones/celda:

1. k4 sizes `[7,7,7,7]`, equal sd
2. k4 sizes `[25,25,25,25]`, equal sd
3. k7 sizes `[7]*7`, equal sd
4. k7 sizes `[25]*7`, equal sd
5. k4 sizes `[6,15,40,80]`, equal sd
6. k4 sizes `[6,15,40,80]`, sd `[3.5,2,1.5,1]`
7. k4 sizes `[6,15,40,80]`, sd `[1,1.5,2,3.5]`
8. k7 sizes `[7]*7`, sd `[1,2,1,3,1.5,2.5,4]`
9. k2 sizes `[7,35]`, sd `[3,1]`
10. k2 sizes `[35,7]`, sd `[3,1]`

### H-robustness — holdout-only families

25,000 replicaciones/celda. Cross-product de 7 holdout families por 3 diseños:

1. k3 sizes `[7,7,7]`, equal sd
2. k3 sizes `[6,15,40]`, equal sd
3. k3 sizes `[6,15,40]`, sd `[3.5,2,1]`

### H-power

20,000 replicaciones/celda. Sólo 4 holdout families (`gamma_shape_2`, `lognormal_sigma_0p8`, `student_t_df_7`, `weibull_shape_1p5`) con k3 sizes `[10,10,10]`, equal sd, y `delta_range=[0.25,0.50,1.00]`.

## 8. Semillas y reproducibilidad

Master seeds congeladas:

```text
development_master_seed = 2026090501
holdout_master_seed     = 2026090599
```

La seed de una réplica NO depende de worker, shard, batch size u orden de ejecución.

Derivación obligatoria:

1. cada celda tiene `cell_id` estable del manifest;
2. construir canonical UTF-8 string `phase|cell_id|replicate_index`;
3. SHA-256 del string;
4. usar los primeros 4 uint32 del digest junto al master seed en `numpy.random.SeedSequence`;
5. generar una única muestra por réplica y usarla para Classical y Welch.

Python `hash()` está prohibido para seeds.

## 9. Execution path del harness

El Monte Carlo principal usa los componentes de producción:

```text
_summarize_groups -> _classical_kernel / _welch_kernel
```

sobre los mismos summaries por réplica.

Antes de ejecutar evidencia, el harness debe pasar un **production parity gate**:

- 32 réplicas deterministas por cada cell_id del manifest activo;
- comparar kernel path con `OneWayANOVA(..., independence="assumed").run()`;
- comparar kernel path con `WelchANOVA(..., independence="assumed").run()`;
- tolerancia `rtol=1e-12`, `atol=1e-14` para F/p/df en dominio ordinario;
- cualquier mismatch aborta la fase.

No se llama `MethodSelector` ni `OneWayRobustness` dentro de la calibración.

## 10. Métricas obligatorias

Por `cell_id`, método y alpha:

- `replications_requested`
- `replications_completed`
- `generation_error_count`
- `kernel_error_count`
- `nonfinite_count`
- `rejection_count`
- `rejection_rate`
- Wilson 99% CI (`ci99_low`, `ci99_high`)
- `mc_standard_error`

Como Classical y Welch usan la misma muestra, guardar además:

- `both_reject_count`
- `classical_only_reject_count`
- `welch_only_reject_count`
- `neither_reject_count`
- `classical_minus_welch_rejection_rate`

Para H1 se reportan estas mismas métricas como potencia.

### Shadow diagnostics

Para `replicate_index % 10 == 0` se permite guardar en artefacto separado:

- variance ratio observado;
- max absolute sample skewness;
- max absolute excess kurtosis;
- max IQR-outlier fraction.

Son **exploratorios** y no intervienen en PASS/FAIL de CP-ANOVA-07 ni habilitan selector.

## 11. Intervalos y precisión Monte Carlo

Wilson score interval al 99%, con:

```text
z = 2.5758293035489004
```

En core H0 con 50k reps y p≈0.05, MC SE esperado ≈0.000975 y half-width 99% ≈0.0025.

No se amplían replicaciones selectivamente después de observar resultados para rescatar una celda. Si una fase completa requiere más precisión, se preregistra una nueva versión y se rerun de forma uniforme en el estrato afectado.

## 12. Criterios de PASS / interpretación

### Hard execution gates

Para toda celda válida de D-core y H-core:

```text
generation_error_count = 0
kernel_error_count = 0
nonfinite_count = 0
replications_completed = replications_requested
```

Cualquier violación bloquea la interpretación.

### Classical confirmatory gate

En celdas **normal + equal variance + min(group_n)>=5** de D-core y H-core:

```text
Wilson 99% CI del Type-I rate en alpha=0.05
DEBE quedar completamente dentro de [0.04, 0.06].
```

### Welch confirmatory gate

En todas las celdas **normal** de D-core/H-core con `min(group_n)>=5`, incluyendo equal y unequal variance:

```text
Wilson 99% CI del Type-I rate en alpha=0.05
DEBE quedar completamente dentro de [0.04, 0.06].
```

Las celdas con `n<5` son stress/characterization, no confirmatory gates.

### Classical bajo heterocedasticidad

No tiene gate de nivel nominal. Se cuantifica la distorsión para documentar por qué no debe inferirse homocedasticidad de un pretest y para informar una futura política, pero CP-06/07 no seleccionan métodos.

### Robustness bands fuera del modelo exacto

Son descriptivas, no PASS del engine:

- `green`: `|rate-0.05| <= 0.01`
- `amber`: `0.01 < |rate-0.05| <= 0.025`
- `red`: `|rate-0.05| > 0.025`

Siempre se muestra Wilson 99% CI junto a la banda.

### Power

No tiene umbral de PASS. Se reporta:

- power Classical;
- power Welch;
- diferencia pareada;
- discordance counts;
- monotonicity flags por delta como sanity check, sin convertirlos en autorización de método.

## 13. Artefactos obligatorios

Por fase:

```text
anova_calibration_manifest.json
anova_calibration_metadata.json
anova_calibration_summary.parquet
anova_calibration_summary.csv
anova_calibration_replicates-<shard>.parquet
anova_calibration_disagreement.csv
anova_calibration_report.md
```

Metadata mínima:

- git SHA del harness;
- production engine SHA/blob;
- preregistration version;
- manifest SHA-256;
- Python/NumPy/SciPy/statsmodels versions;
- OS/CPU;
- alpha grid;
- master seed;
- phase;
- workers;
- batch size;
- shard id/count;
- requested/completed counts;
- start/end UTC;
- warnings/exceptions accounting.

## 14. Reanudar / paralelizar

- CPU solamente en esta versión.
- Puede usar múltiples procesos.
- Cada proceso debe fijar BLAS/OpenMP a 1 thread.
- Sharding no cambia seeds.
- Un shard terminado es inmutable y checksumed.
- Resume debe saltar sólo shards/cells con checksum válido y metadata idéntica al manifest/harness SHA.
- Nunca mezclar artefactos de distintos SHA/manifest/version en un mismo resumen final.

## 15. Stop conditions

Reabrir CP-ANOVA-04/05 o invalidar la corrida si:

- parity gate falla;
- production ANOVA cambia después del freeze;
- manifest cambia sin nueva versión;
- seed stream depende de workers/shards/order;
- Classical/Welch no usan la misma muestra por réplica;
- una celda confirmatoria produce errores/no-finite;
- se inspecciona holdout antes de freeze de Phase D candidate;
- se cambia criterio de aceptación después de observar resultados;
- el harness condiciona resultados a selector/diagnósticos.

## 16. CP-ANOVA-07 — subcheckpoints posteriores

Después de este preregistro:

- `CP-ANOVA-07A` — Cortex implementa harness exactamente contra manifest.
- `CP-ANOVA-07B` — ChatGPT audita harness vs preregistro + parity/reproducibility.
- `CP-ANOVA-07C` — engineering pilot E0; no evidencia.
- `CP-ANOVA-07D` — Phase D development calibration.
- `CP-ANOVA-07E` — interpretación y freeze del candidate antes de holdout.
- `CP-ANOVA-07F` — Product Owner autoriza apertura de Phase H.
- `CP-ANOVA-07G` — holdout + evidence report final.

No se ejecuta ninguna corrida Monte Carlo pesada como parte de CP-ANOVA-06.
