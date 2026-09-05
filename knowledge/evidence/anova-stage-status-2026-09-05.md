# STAGE-ANOVA-001 — Statistical closure status

- Fecha: `2026-09-05`
- Rama arquitectónica: `audit/anova-statistical-closure`
- Rama técnica: `feature/anova-production-candidate`
- Base original: `main@402e4601df460811779b3238c2526ac12f463a67`
- Estado global: `in_progress`
- Objetivo: cerrar Classical one-way ANOVA y Welch one-way ANOVA como métodos explícitos, precisos, eficientes y auditables antes de cualquier selector automático o integración con el optimizador.

## Checkpoints

| Checkpoint | Estado | Resultado / criterio |
|---|---|---|
| CP-ANOVA-01 — preflight y census | `complete` | Rama histórica tratada como evidencia, no merge candidate; arquitectura actual inspeccionada. |
| CP-ANOVA-02 — statistical specification | `complete/frozen` | Contrato matemático y dominios congelados en `knowledge/theory/anova-one-way-statistical-spec.md`. |
| CP-ANOVA-03 — API + execution contract | `complete/frozen` | API, ANOVAResult, strict behavior, separación validation/summaries/kernel y exports congelados en `knowledge/decisions/anova-api-execution-contract.md`. |
| CP-ANOVA-04 — production candidate | `complete/frozen` | Candidate code SHA `5a116a4e8672dadd3fe57a51f4186f70d1440afd`; Quantum smoke `18/18 PASS`; directed regression `52/52 PASS`. |
| CP-ANOVA-05 — deterministic/oracle validation | `next` | Ampliar oráculos independientes, invariantes y adversarial numerical tests sobre el SHA congelado, sin Monte Carlo. |
| CP-ANOVA-06 — calibration preregistration | `pending` | Congelar escenarios, seeds, precision targets, primary metrics y holdout antes de corrida pesada. |
| CP-ANOVA-07 — calibration/evidence | `pending` | Type-I/power/robustness evidence; no usar piloto histórico como evidencia final. |
| CP-ANOVA-08 — adversarial audit | `pending` | Antigravity audita SHA exacto, teoría, implementación, resultados y debt. |
| CP-ANOVA-09 — Product Owner interpretation | `pending` | Decidir integración/manual UAT y cualquier habilitación posterior. |

## Candidate congelado CP-ANOVA-04

```text
5a116a4e8672dadd3fe57a51f4186f70d1440afd
```

Este SHA contiene la implementación y tests del production candidate. Commits documentales posteriores no modifican el código auditado.

Evidencia:

- `knowledge/evidence/anova-production-quantum-smoke-2026-09-05.md`
- `knowledge/evidence/anova-directed-regression-2026-09-05.md`

Cualquier cambio posterior en `pyMagicStat/inference/anova.py`, exports relacionados o tests del candidate reabre CP-ANOVA-04 y requiere nuevo freeze SHA.

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

## Decisiones congeladas en CP-ANOVA-03

- API pública: `OneWayANOVA(...).run()` y `WelchANOVA(...).run()`.
- Resultado canónico: `@dataclass(frozen=True) ANOVAResult`, con `to_dict()` JSON-ready.
- Classical y Welch comparten metadata base pero conservan `components` específicos.
- Kernels internos trabajan desde summaries por grupo y no llaman al selector.
- `strict` gobierna hard failures/autorización, no selección automática de método.
- WARN de shape/outliers/variance no cambia silenciosamente Classical ↔ Welch.
- `unknown` independence no se presenta como independencia verificada.
- `scipy.stats.f.sf` para p-values.
- Errores de contrato producen excepciones explícitas; no se ocultan con catches genéricos.
- El candidate no modifica `MethodSelector(ONE_WAY) -> NOT_CALIBRATED`.
- ANOVA no importa ni depende de `pyMagicStat.optimization`.

## Product direction posterior

Después del cierre ANOVA, un Step 2 actualizará `optimization/orchestrator.py` para trabajar con evaluadores específicos por prueba. ANOVA entregará F/p-value/df y sus métricas propias; Kruskal-Wallis conservará las suyas. El orquestador será motor operativo, no definición universal de estadísticos.

La arquitectura summary-based permitirá cachear `n`, media y varianza/SSW por grupo y reevaluar subconjuntos del futuro optimizador en `O(k)` después del resumen inicial, sin contaminar el motor ANOVA Step 1 con lógica de optimización.

## Protección de trabajo paralelo

Este stage no debe:

- tocar `main` directamente;
- modificar o detener CP06-E de proportion CI;
- utilizar los recursos de Quantum para corrida pesada antes de que se preregistre CP-ANOVA-06 y el Product Owner autorice la ejecución;
- reutilizar el selector histórico `anova-v1-2026-08` como calibrado.

## Siguiente acción autorizada

**CP-ANOVA-05 — deterministic/oracle validation.**

Debe ejecutarse sobre el candidate congelado y ampliar, como mínimo:

1. Classical vs SciPy en matrices deterministas adicionales;
2. Classical vs statsmodels `use_var="equal"`;
3. Welch vs statsmodels `use_var="unequal", welch_correction=True`;
4. Welch vs SciPy `equal_var=False` cuando esté disponible;
5. k=2 t² en múltiples escalas y desbalances;
6. offsets grandes, escalas pequeñas/grandes no subnormales;
7. varianzas muy desiguales y tamaños adversariales;
8. permutaciones e identidades de descomposición;
9. verificación de tolerancias y cualquier límite numérico reproducible.

No se inicia todavía Monte Carlo de robustez/type-I/power.
