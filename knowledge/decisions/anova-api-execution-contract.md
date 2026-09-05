# CP-ANOVA-03 — API and execution contract

- Fecha: `2026-09-05`
- Rama: `audit/anova-statistical-closure`
- Base: `main@402e4601df460811779b3238c2526ac12f463a67`
- Estado: `frozen`

## 1. Public surface

Step 1 expondrá dos métodos explícitos:

```python
OneWayANOVA(*groups, alpha=0.05, independence="unknown", strict=True)
WelchANOVA(*groups, alpha=0.05, independence="unknown", strict=True)
```

Ambos deben ofrecer:

```python
.run() -> ANOVAResult
```

Se evitará reutilizar el patrón histórico `run_test()` como contrato principal de ANOVA. Si posteriormente se requiere un adapter para `optimization/orchestrator.py`, éste envolverá `.run()` sin modificar el motor ANOVA.

## 2. Result type

Se utilizará un resultado estructurado e inmutable:

```python
@dataclass(frozen=True)
class ANOVAResult:
    method: str
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    numerator_df: float
    denominator_df: float
    k: int
    n_total: int
    group_sizes: tuple[int, ...]
    group_means: tuple[float, ...]
    group_variances: tuple[float, ...]
    assumptions: AssumptionReport
    diagnostics: Mapping[str, Any]
    components: Mapping[str, Any]
    method_version: str
```

Debe ofrecer `to_dict()` JSON-ready.

No se devolverá solamente un diccionario libre; el dataclass será el contrato canónico y `to_dict()` la interfaz serializable.

## 3. Classical components

`components` incluirá al menos:

```text
grand_mean
ss_between
ss_within
ss_total
mean_square_between
mean_square_within
eta_squared
```

## 4. Welch components

`components` incluirá al menos:

```text
weights
weighted_mean
welch_B
welch_correction
```

No se expondrá eta-squared Classical como effect size Welch.

## 5. Internal separation

La implementación deberá separar explícitamente:

```text
input validation / assumptions
        ↓
group summaries
        ↓
classical kernel OR welch kernel
        ↓
ANOVAResult assembly
```

Los kernels no deben llamar a `MethodSelector` ni decidir si la prueba está recomendada.

Los kernels aceptarán una estructura interna de summaries equivalente a:

```python
@dataclass(frozen=True)
class _GroupSummary:
    n: int
    mean: float
    variance: float
    ss_within: float
```

Esto permite testing directo de fórmulas y futura reutilización/caching por el optimizador.

## 6. Validation

Antes del kernel:

- `alpha` debe satisfacer `0 < alpha < 1`;
- debe haber `k >= 2` grupos;
- cada grupo se normaliza a array float64 unidimensional;
- cada grupo debe pasar el contrato vigente de calidad de datos;
- labels no son requisito del kernel Step 1;
- inputs no se mutan.

Se reutilizará `InferenceValidator.validate_one_way()` para diagnostics y normalización, salvo que CP-ANOVA-04 demuestre una incompatibilidad concreta que deba corregirse en la misma rama técnica con tests dedicados.

## 7. Strict behavior

`strict` no debe significar "seleccionar automáticamente Classical o Welch".

### `strict=True`

La ejecución falla cerrada (`ValueError`) cuando existe un **FAIL estructural** que hace inválido el diseño/contrato de ejecución, por ejemplo:

- data quality FAIL;
- independence explícitamente contradictoria/no independiente si el modelo actual puede representarlo;
- otro FAIL hard del `AssumptionReport` que haga el diseño no autorizable.

Un `WARN` diagnóstico de normalidad, outliers o heterocedasticidad no selecciona otro método y no bloquea automáticamente un método explícitamente solicitado. El resultado conserva los warnings en `assumptions`.

### `strict=False`

Permite computar el estadístico cuando la matemática es computable pese a un estado inferencial no autorizable/no resuelto, pero **nunca** salta los hard data-quality checks requeridos por el kernel.

Debe emitir `UserWarning` con la razón y conservarla en diagnostics/assumptions.

## 8. Independence

Se conserva el contrato actual:

```text
unknown
assumed
verified
```

`unknown` no será tratado como independencia demostrada.

Durante Step 1, `unknown` puede permitir cálculo explícito, pero el resultado debe mostrar el estado real y no afirmar validación completa del diseño. No se añadirá un selector automático por esta razón.

## 9. Explicit method semantics

Instanciar `OneWayANOVA` significa que el usuario solicita Classical ANOVA.
Instanciar `WelchANOVA` significa que solicita Welch ANOVA.

El sistema:

- diagnostica;
- calcula;
- informa límites;
- no sustituye silenciosamente el método.

En particular:

```text
OneWayANOVA + heterocedasticity warning != auto Welch
WelchANOVA + variance homogeneity != auto Classical
```

## 10. Selector and capability registry

CP-ANOVA-04 no debe modificar la semántica actual de:

```text
MethodSelector(ONE_WAY) -> NOT_CALIBRATED
```

No se registrará todavía una capability automática ONE_WAY como production-calibrated.

Los métodos explícitos pueden existir antes de que exista routing automático.

## 11. Numerical implementation

El kernel debe:

- calcular summaries una sola vez;
- usar operaciones float64 estables;
- evitar concatenación global innecesaria;
- calcular p-values con `scipy.stats.f.sf`, no `1-cdf`;
- mantener `O(N)` para summaries + `O(k)` para el cálculo;
- no crear arrays densos dependientes del rango de valores;
- no usar random state.

Para grandes offsets con pequeñas dispersiones, summaries deben usar `np.mean`/`np.var(ddof=1)` o una estrategia numéricamente igual o mejor; cualquier divergencia significativa contra oráculos en casos adversariales es blocker.

## 12. Error behavior

Errores de input/contrato son excepciones explícitas (`ValueError`/`TypeError` apropiado), no resultados con p-values NaN presentados como válidos.

No se capturará `Exception` genéricamente para convertir fallos internos en un `output_format(bool_result=False, ...)` silencioso.

## 13. Public exports

Una vez validado el candidate, se exportarán desde `pyMagicStat.inference`:

```text
ANOVAResult
OneWayANOVA
WelchANOVA
```

No se exportarán kernels internos ni `_GroupSummary`.

## 14. Method versions

Los candidatos deben usar identificadores versionados explícitos, por ejemplo:

```text
classical-one-way-anova-v1
welch-one-way-anova-v1
```

Los nombres exactos pueden ajustarse durante implementación si permanecen estables, claros y registrados en tests.

## 15. Tests required before candidate can freeze

CP-ANOVA-04 debe incluir tests unitarios para:

- result schema y immutability;
- k=2;
- k>2;
- equal/unequal n;
- Classical formula components;
- Welch weights/correction;
- invalid alpha;
- k<2;
- nonfinite/constant/degenerate groups;
- no input mutation;
- strict behavior;
- selector remains NOT_CALIBRATED;
- deterministic repeated execution;
- summary-based kernel equality with direct calculation.

Oracle/adversarial coverage completa pertenece a CP-ANOVA-05.

## 16. Future optimizer integration

El futuro adapter debe poder realizar conceptualmente:

```python
result = OneWayANOVA(*groups, ...).run()
score = result.statistic
p = result.p_value
```

u otra métrica derivada explícita de `components`, sin que ANOVA dependa de `optimization.orchestrator`.

El orquestador se modernizará después del cierre ANOVA; ANOVA no debe importar desde `pyMagicStat.optimization`.
