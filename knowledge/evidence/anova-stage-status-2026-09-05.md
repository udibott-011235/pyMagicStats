# STAGE-ANOVA-001 — Statistical closure status

- Fecha: `2026-09-05`
- Rama arquitectónica: `audit/anova-statistical-closure`
- Rama técnica actual: `fix/anova-location-stability`
- Base original: `main@402e4601df460811779b3238c2526ac12f463a67`
- Estado global: `in_progress`
- Objetivo: cerrar Classical one-way ANOVA y Welch one-way ANOVA como métodos explícitos, precisos, eficientes y auditables antes de cualquier selector automático o integración con el optimizador.

## Checkpoints

| Checkpoint | Estado | Resultado / criterio |
|---|---|---|
| CP-ANOVA-01 — preflight y census | `complete` | Rama histórica tratada como evidencia, no merge candidate; arquitectura actual inspeccionada. |
| CP-ANOVA-02 — statistical specification | `complete/frozen` | Contrato matemático y dominios congelados en `knowledge/theory/anova-one-way-statistical-spec.md`. |
| CP-ANOVA-03 — API + execution contract | `complete/frozen` | API, ANOVAResult, strict behavior, separación validation/summaries/kernel y exports congelados en `knowledge/decisions/anova-api-execution-contract.md`. |
| CP-ANOVA-04 — production candidate | `complete/frozen` | Candidate remediado: `83bfe547563d977f1ed0dd0f43629c281744488c`; smoke location-stability `21/21 PASS`; directed regression `55/55 PASS`. Candidate anterior `5a116a4e...` queda superseded por deuda de location stability. |
| CP-ANOVA-05 — deterministic/oracle validation | `complete/frozen` | Suite completa sobre `f5200fc1a191f24d9f48a998eeae8abb46107817`: `41/41 PASS`, con oráculos Classical/Welch independientes, SciPy/statsmodels, k=2 t², adversarial sizes/scales/offsets/permutations y adjudicación explícita del oracle SciPy Welch bajo gran offset. |
| CP-ANOVA-06 — calibration preregistration | `next` | Congelar escenarios, seeds, precision targets, primary metrics, criterios de decisión y holdout antes de corrida pesada. |
| CP-ANOVA-07 — calibration/evidence | `pending` | Type-I/power/robustness evidence; no usar piloto histórico como evidencia final. |
| CP-ANOVA-08 — adversarial audit | `pending` | Antigravity audita SHA exacto, teoría, implementación, resultados y debt. |
| CP-ANOVA-09 — Product Owner interpretation | `pending` | Decidir integración/manual UAT y cualquier habilitación posterior. |

## Candidate congelado CP-ANOVA-04

```text
83bfe547563d977f1ed0dd0f43629c281744488c
```

La remediación de production code vive en:

```text
376677ca32dfd1e3f5b5b64bec48e3160c35d5a9
```

El candidate anterior:

```text
5a116a4e8672dadd3fe57a51f4186f70d1440afd
```

queda `superseded_candidate` después de que CP-ANOVA-05 detectara pérdida de invariancia de location al sumar un offset común `1e12`.

Evidencia actual:

- `knowledge/evidence/anova-location-stability-remediation-smoke-2026-09-05.md`
- `knowledge/evidence/anova-location-stability-remediation-freeze-2026-09-05.md`
- `knowledge/evidence/anova-cp05-deterministic-oracle-pass-2026-09-05.md`

Cualquier cambio posterior en `pyMagicStat/inference/anova.py`, exports relacionados o tests contractuales del candidate reabre CP-ANOVA-04 y requiere nuevo freeze SHA.

## Decisiones congeladas en CP-ANOVA-02

- Diseño: grupos independientes, one-way.
- Estimando: diferencias/equivalencia global de medias poblacionales.
- `k >= 2`.
- Classical y Welch son métodos explícitos distintos.
- Normalidad se diagnostica sobre residuos centrados dentro de grupos, no pooled raw values.
- Levene/Brown-Forsythe es diagnóstico; `p > alpha` no demuestra homocedasticidad.
- Independence es metadata externa; no se infiere de los valores.
- Selector ONE_WAY permanece `NOT_CALIBRATED`.
- Classical expone F, p, df, SS, MS y eta² descriptivo.
- Welch expone F de Welch, p, df efectivos, weights y correction metadata.
- Welch no reutiliza eta² Classical como si fuera un effect size equivalente.
- Kernel orientado a summaries: `O(N)` para resumir input + `O(k)` para cálculo.
- k=2 debe satisfacer `F_classical=t_student²` y `F_welch=t_welch²`.
- SciPy Classical oracle compatible con la dependencia mínima actual; Welch usa statsmodels como oracle principal y SciPy `equal_var=False` sólo cuando versión >=1.16 esté disponible.
- No post-hoc, no fallback a Kruskal, no transforms automáticas, no DOE, no optimizer integration en Step 1.

## Estabilidad numérica congelada

La representación interna de summaries utiliza centración local por grupo para evitar pérdida de precisión al combinar medias con una gran location común. La evaluación Classical/Welch usa medias relativas a un origen común y conserva el contrato `O(N)+O(k)` y la futura capacidad de caching.

No se adoptó `np.longdouble` como requisito de precisión portable ni se relajaron tolerancias para ocultar el fallo.

## Product direction posterior

Después del cierre ANOVA, un Step 2 actualizará `optimization/orchestrator.py` para trabajar con evaluadores específicos por prueba. ANOVA entregará F/p-value/df y sus métricas propias; Kruskal-Wallis conservará las suyas. El orquestador será motor operativo, no definición universal de estadísticos.

La arquitectura summary-based permite cachear summaries por grupo y reevaluar subconjuntos del futuro optimizador en `O(k)` después del resumen inicial, sin contaminar el motor ANOVA Step 1 con lógica de optimización.

## Protección de trabajo paralelo

Este stage no debe:

- tocar `main` directamente;
- modificar o detener CP06-E de proportion CI;
- lanzar CP-ANOVA-07 ni otra corrida pesada en Quantum antes de congelar CP-ANOVA-06 y recibir autorización del Product Owner;
- reutilizar el selector histórico `anova-v1-2026-08` como calibrado;
- inferir robustez/type-I/power a partir del PASS determinista de CP-ANOVA-05.

## Siguiente acción autorizada

**CP-ANOVA-06 — calibration preregistration.**

Debe congelar antes de generar muestras:

1. hipótesis y preguntas primarias de calibración;
2. familias/distribuciones y parámetros;
3. tamaños de muestra y `k`;
4. patrones de balance/desbalance y asociación tamaño-varianza;
5. heterocedasticidad;
6. shape/tails/skew/outliers/contamination;
7. H0/H1 y tamaños de efecto;
8. seeds/master seed y esquema de generación reproducible;
9. número de replicaciones o regla de precisión Monte Carlo;
10. métricas primarias (Type-I) y secundarias (power/diagnostics);
11. umbrales de aceptación y zonas de revisión;
12. holdout independiente y reglas anti-leakage;
13. accounting de fallos/errores numéricos;
14. resource plan para Quantum.

CP-ANOVA-06 es documental/preregistro. No debe iniciar todavía la corrida Monte Carlo de CP-ANOVA-07.
