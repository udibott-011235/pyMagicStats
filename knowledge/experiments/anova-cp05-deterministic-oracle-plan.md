# CP-ANOVA-05 — Deterministic / oracle validation plan

- Fecha: `2026-09-05`
- Rama: `audit/anova-oracle-validation`
- Parent: `e458eb7f675b54753fc91f2f76c572113145f3c8`
- Production code freeze bajo prueba: `5a116a4e8672dadd3fe57a51f4186f70d1440afd`
- Estado: `preregistered`
- Naturaleza: determinista; **no Monte Carlo**

## 1. Objetivo

Intentar falsar la implementación matemática/numerical de Classical one-way ANOVA y Welch one-way ANOVA antes de cualquier estudio de calibración inferencial.

CP-ANOVA-05 no decide todavía cuándo seleccionar automáticamente Classical o Welch y no convierte tests deterministas en evidencia de robustez poblacional.

## 2. Oráculos independientes

### Classical

Se exige concordancia entre:

1. `pyMagicStat.inference.OneWayANOVA`;
2. fórmula Classical independiente implementada sólo en tests;
3. `scipy.stats.f_oneway(*groups)`;
4. `statsmodels.stats.oneway.anova_oneway(groups, use_var="equal")`;
5. `statsmodels.stats.oneway.anova_generic(means, variances, nobs, use_var="equal")`.

### Welch

Se exige concordancia entre:

1. `pyMagicStat.inference.WelchANOVA`;
2. fórmula Welch-Satterthwaite independiente implementada sólo en tests;
3. `statsmodels.stats.oneway.anova_oneway(groups, use_var="unequal", welch_correction=True)`;
4. `statsmodels.stats.oneway.anova_generic(means, variances, nobs, use_var="unequal", welch_correction=True)`;
5. `scipy.stats.f_oneway(*groups, equal_var=False)` cuando el SciPy instalado exponga `equal_var` (SciPy >=1.16).

La suite no incrementa el mínimo SciPy declarado por el proyecto: el oracle Welch de SciPy moderno es condicional; statsmodels permanece oracle obligatorio compatible con el contrato de dependencias.

## 3. Escenarios deterministas preregistrados

La matriz debe incluir al menos:

1. balanced, `k=3`;
2. unbalanced, `k=4`;
3. mínimo `n_i=2`;
4. heterocedasticidad fuerte con asociación tamaño-varianza;
5. muchos grupos (`k>=10`);
6. offset común grande con dispersión representable;
7. escala común pequeña, no subnormal;
8. escala común grande, sin overflow de momentos diagnósticos relevantes;
9. grupos casi degenerados pero claramente por encima del umbral vigente de `DataQualityAssessment`.

Ninguno usa RNG.

## 4. Invariantes especiales

Además de los invariantes ya cubiertos en CP-ANOVA-04:

- raw-data oracle == summary-statistics oracle;
- `k=2` Classical == pooled Student `t^2` en múltiples configuraciones;
- `k=2` Welch == Welch `t^2` y df Satterthwaite en múltiples configuraciones;
- scale invariance para escalas positivas y negativas no degeneradas;
- translation invariance bajo offset común grande dentro del dominio float64;
- group/order invariance en escenarios heterocedásticos;
- Classical `SS_total = SS_between + SS_within`;
- Classical `df_between + df_within = N - 1`;
- Welch `df2 > 0`, correction >= 1;
- p-values dentro de `[0,1]`, F finito y no negativo dentro del dominio aceptado.

## 5. Tolerancias

Las comparaciones ordinarias usan float64 y deben ser estrictas:

```text
relative tolerance: 5e-10
absolute tolerance: 5e-12
```

Para invariancia bajo offset común grande se admite una tolerancia separada de `5e-8` relativa, porque el propio input float64 pierde resolución absoluta al trasladarse a magnitudes grandes. El escenario se construirá con dispersión suficientemente mayor que el ULP del offset para que una desviación material siga siendo detectable.

No se ampliarán tolerancias después de observar un fallo sin documentar antes la causa numérica y demostrar que la divergencia pertenece al oracle/input float64 y no al candidate.

## 6. Stop conditions / blockers

CP-ANOVA-05 se detiene y reabre CP-ANOVA-04 si ocurre cualquiera:

- discrepancia reproducible con fórmula independiente;
- discrepancia reproducible con >=2 oráculos externos en dominio común;
- `F`, p o df no finitos para inputs aceptados no extremos;
- violación de `k=2 -> t^2`;
- pérdida material de invariancia de escala/traslación;
- summary-based kernel diverge de raw-data oracles;
- candidate acepta una configuración que su propio contrato estructural declara inválida;
- candidate requiere cambiar fórmulas/API congeladas para pasar.

Un fallo aislado de un oracle externo debe adjudicarse antes de modificar production code.

## 7. PASS de CP-ANOVA-05

PASS requiere:

- suite adversarial completa en el SHA/branch exacto;
- todos los oráculos obligatorios concordantes;
- SciPy Welch moderno concordante cuando disponible;
- cero cambios en production code respecto a `5a116a4e...`, salvo que un fallo haya reabierto formalmente CP-ANOVA-04;
- evidencia de entorno, comando y resultado registrada en Knowledge Base.

## 8. Límite de interpretación

PASS aquí significa **corrección matemática/numerical determinista dentro del dominio probado**. No significa control de error tipo I, potencia, robustez a no-normalidad o política automática validada. Eso pertenece a CP-ANOVA-06/07.
