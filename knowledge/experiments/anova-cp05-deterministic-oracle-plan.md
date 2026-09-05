# CP-ANOVA-05 — Deterministic / oracle validation plan

- Fecha: `2026-09-05`
- Production code freeze bajo prueba: `83bfe547563d977f1ed0dd0f43629c281744488c`
- Estado: `preregistered / resumed after numerical remediation`
- Naturaleza: determinista; **no Monte Carlo**

## Objetivo

Intentar falsar la implementación matemática/numerical de Classical one-way ANOVA y Welch one-way ANOVA antes de cualquier estudio de calibración inferencial.

## Oráculos independientes

### Classical
1. `pyMagicStat.inference.OneWayANOVA`;
2. fórmula Classical independiente implementada sólo en tests;
3. `scipy.stats.f_oneway(*groups)`;
4. `statsmodels.stats.oneway.anova_oneway(groups, use_var="equal")`;
5. `statsmodels.stats.oneway.anova_generic(means, variances, nobs, use_var="equal")`.

### Welch
1. `pyMagicStat.inference.WelchANOVA`;
2. fórmula Welch-Satterthwaite independiente implementada sólo en tests;
3. `statsmodels.stats.oneway.anova_oneway(groups, use_var="unequal", welch_correction=True)`;
4. `statsmodels.stats.oneway.anova_generic(means, variances, nobs, use_var="unequal", welch_correction=True)`;
5. `scipy.stats.f_oneway(*groups, equal_var=False)` cuando disponible.

Para el edge case de **gran offset común**, SciPy Welch se adjudica sobre datos centrados por un origen común matemáticamente equivalente, porque el path raw `equal_var=False` mostró sensibilidad numérica reproducible a la location absoluta. Esta adjudicación no modifica la tolerancia preregistrada.

## Escenarios deterministas

- balanced `k=3`;
- unbalanced `k=4`;
- mínimo `n_i=2`;
- heterocedasticidad fuerte con asociación tamaño-varianza;
- muchos grupos `k>=10`;
- offset común grande con dispersión representable;
- escala común pequeña no subnormal;
- escala común grande sin overflow del kernel;
- grupos casi degenerados pero válidos según DQA.

Ninguno usa RNG.

## Invariantes

- raw-data oracle == summary-statistics oracle donde el oracle summary no herede la deuda bajo adjudicación;
- `k=2` Classical == pooled Student `t^2`;
- `k=2` Welch == Welch `t^2` + df Satterthwaite;
- scale invariance;
- translation invariance;
- group/order invariance;
- Classical `SS_total = SS_between + SS_within`;
- Classical `df_between + df_within = N - 1`;
- Welch `df2 > 0`, correction >= 1;
- p-values `[0,1]`, F finito/no negativo dentro del dominio.

## Tolerancias congeladas

```text
ordinary rel = 5e-10
ordinary abs = 5e-12
offset rel   = 5e-8
offset abs   = 5e-10
```

No se amplían tolerancias después de observar fallos sin adjudicación documentada.

## PASS

PASS requiere suite completa sobre SHA exacto, oráculos obligatorios concordantes, no cambios adicionales en production code y evidencia de entorno/comando/resultado. PASS significa corrección matemática/numerical determinista en el dominio probado; no significa calibración inferencial, robustez poblacional ni autorización de selector automático.
