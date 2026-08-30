# CP-03 — Contrato de API y compatibilidad para intervalos de proporción

**Stage:** `STAGE-PROP-CI-001`  
**Estado:** `under_review`  
**Baseline de producción:** `main` @ `402e4601df460811779b3238c2526ac12f463a67`  
**Baseline documental de entrada:** `docs/proportion-ci-stage` @ `cacdabbb40e7b4984168cfa70e0f0c4077460ea3`  
**Decisión estadística de entrada:** `knowledge/decisions/proportion-ci-cp02-spec.md` (`accepted`)  
**Evidencia de entrada:** `knowledge/evidence/proportion-ci-cp01-census.md`  
**Owner de decisión:** Product Owner  
**Arquitectura:** statistical-software-architecture

## 1. Objetivo y límites

Este checkpoint cierra exclusivamente la superficie pública, compatibilidad, metadata, errores, warnings y conducta fail-closed necesaria para implementar después el contrato estadístico aprobado en CP-02.

No autoriza implementación, calibración, selección automática, PR ni merge.

No se modifica el estimando: una única proporción poblacional `p=P(Y=1)` bajo un diseño Bernoulli/binomial ordinario. El alcance de producción de este stage permanece bilateral.

## 2. Principio de compatibilidad

La API existente se preserva. No se crea una clase competidora ni se cambia el default.

La clase existente se convierte en API pública formal mediante re-export desde `pyMagicStat.inference`, pero su ruta histórica `pyMagicStat.inference.parametric.PopulationProportionCI` debe continuar funcionando.

La API canónica nueva para conteos agregados será un constructor de clase explícito `from_counts(successes, trials, ...)`; no se exigirá fabricar un contenedor `data` de longitud `n`.

## 3. Firmas públicas exactas

### 3.1 Constructor compatible para datos raw/callable/legacy

```python
class PopulationProportionCI:
    def __init__(
        self,
        data: Any,
        alpha: float = 0.05,
        incidences: Optional[Union[int, float, Callable]] = None,
        method: str = "wilson",
        *,
        independence: str = "unknown",
    ) -> None:
        ...
```

Reglas:

- Los cuatro argumentos posicionales existentes conservan significado y orden.
- `independence` es keyword-only y admite únicamente `"unknown"`, `"assumed"` o `"verified"`.
- Añadir `independence` no convierte el supuesto en observable; sólo registra procedencia declarada.
- `alpha` conserva la semántica actual; no se añade un segundo parámetro `confidence_level` al constructor durante este stage para evitar dos fuentes de verdad. `confidence_level` se deriva como `1-alpha` en el resultado.
- `method` sigue siendo case-insensitive mediante normalización a minúsculas.

### 3.2 Constructor agregado canónico

```python
@classmethod
def from_counts(
    cls,
    successes: int,
    trials: int,
    alpha: float = 0.05,
    method: str = "wilson",
    *,
    independence: str = "unknown",
) -> "PopulationProportionCI":
    ...
```

Contrato:

- `successes` y `trials` deben ser tipos enteros (`numbers.Integral`/equivalente NumPy) y `bool` se rechaza explícitamente.
- `trials >= 1`.
- `0 <= successes <= trials`.
- No se aceptan floats, aunque su valor sea entero (`3.0`); el API agregado canónico representa conteos discretos reales.
- `data` no se fabrica ni participa en el estimando.
- `estimate = successes / trials`.
- `n = trials` para compatibilidad de salida.

### 3.3 Método de cálculo

Se preserva:

```python
def calculate_interval(self) -> Dict[str, Any]:
    ...
```

No cambia a dataclass como retorno en este stage porque eso rompería consumidores existentes. Puede haber estructura interna tipada, pero `calculate_interval()` sigue devolviendo un diccionario JSON-serializable.

## 4. Export público

CP-05 deberá añadir:

```python
from pyMagicStat.inference import PopulationProportionCI
```

como ruta soportada, agregándola a `pyMagicStat/inference/__init__.py` y `__all__`.

La ruta histórica siguiente debe seguir funcionando:

```python
from pyMagicStat.inference.parametric import PopulationProportionCI
```

No se exige export desde el paquete raíz `pyMagicStat` en este stage.

## 5. Modos de entrada y compatibilidad

| Modo | Entrada | Estado CP-03 | Semántica |
|---|---|---|---|
| raw binario | `PopulationProportionCI([0,1,...])` | soportado | canónico |
| callable | `incidences=<callable>` | soportado | compatible; cada observación truthy cuenta como éxito |
| agregado nuevo | `PopulationProportionCI.from_counts(x,n)` | soportado | canónico |
| `incidences` numérico entero | constructor legacy | soportado con deprecación de API agregada | resultado preservado; migrar a `from_counts` |
| `incidences` numérico fraccionario | constructor legacy | transición temporal, fuera del contrato binomial | sólo Wilson/Wald legacy; warning y metadata explícita |
| bootstrap proportion | `BootstrapCI(stat="proportion")` | separado | no es fallback ni parte de esta API |

### 5.1 Datos raw

Cuando `incidences is None`:

- `data` debe seguir siendo un vector unidimensional, no vacío y finito;
- cada valor debe ser exactamente `0/1` después de la coerción numérica existente;
- booleanos raw siguen siendo compatibles al convertirse en `0.0/1.0`;
- `successes` se obtiene por conteo/suma y debe quedar entero;
- `trials=len(data)`.

### 5.2 Callable

Cuando `incidences` es callable:

- se preserva la aplicación observación por observación;
- cada retorno truthy cuenta como un éxito;
- el conteo resultante es entero;
- no se cambia en este stage la compatibilidad histórica de que `data` sea primero coercible a vector numérico finito;
- el resultado debe declarar `input_mode="predicate"`.

### 5.3 `incidences` numérico legacy

La ruta numérica completa se considera API agregada legacy porque obliga a suministrar `data` sólo para obtener `n`.

- Si el valor es integral y está en `[0,n]`, el resultado numérico se preserva y se emite `DeprecationWarning` indicando usar `from_counts(successes, trials)`.
- Si el valor es fraccionario y está en `[0,n]`, Wilson/Wald mantienen temporalmente el cálculo histórico para no introducir una ruptura inmediata, pero se emite `DeprecationWarning` explícito indicando que el valor queda fuera del contrato Bernoulli/binomial aprobado.
- Un conteo fraccionario nunca se etiqueta como estadísticamente soportado ni calibrado.
- `clopper_pearson` con `incidences` fraccionario debe producir `ValueError`, porque su contrato exige conteos binomiales enteros.
- `incidences=True/False` conserva el comportamiento legacy durante la transición y recibe el mismo warning de API agregada legacy; el constructor canónico `from_counts` sí rechaza booleanos.

### 5.4 Condición de retirada de `incidences` numérico

No se autoriza retirar esta ruta dentro de `STAGE-PROP-CI-001`.

CP-05 sólo introduce warning y metadata. Una retirada futura requiere simultáneamente:

1. al menos una release pública con la advertencia activa;
2. documentación de migración a `from_counts`;
3. ausencia o migración de callers internos conocidos;
4. decisión nueva y explícita del Product Owner;
5. actualización de compatibilidad y versionado.

Hasta entonces, no se convierte en `ValueError` de forma general.

## 6. Métodos públicos

El conjunto exacto autorizado para CP-05 será:

```text
wilson
clopper_pearson
wald
```

No se aceptan aliases nuevos (`exact`, `beta`, `cp`, `wilsoncc`, etc.) en este stage.

### 6.1 `wilson`

- default preservado;
- bilateral;
- `interval_kind="frequentist_score"`;
- no requiere ni consulta la regla legacy `successes>=10 and failures>=10` para poder ejecutarse;
- `calibration_status="not_calibrated"` hasta CP-06.

### 6.2 `clopper_pearson`

- nombre público definitivo: `"clopper_pearson"`;
- bilateral;
- sólo conteos enteros válidos;
- `interval_kind="frequentist_exact_conservative"`;
- los boundaries `x=0` y `x=n` son válidos;
- `calibration_status="not_calibrated"` hasta que CP-06 registre evidencia de proyecto, aunque su propiedad matemática exacta se documente separadamente.

### 6.3 `wald`

- ruta explícita legacy;
- no default;
- bilateral;
- no clipping silencioso;
- `interval_kind="frequentist_asymptotic_legacy"`;
- `calibration_status="not_calibrated"` hasta CP-06;
- se conserva el `UserWarning` cuando la regla legacy de éxitos/fracasos observados es menor que 10, pero el wording deberá dejar claro que es un diagnóstico legacy de Wald, no una garantía ni un selector;
- no se programa retirada de Wald en este stage. Marcarlo `legacy` no equivale a deprecarlo para eliminación.

### 6.4 Métodos fuera de producción

Jeffreys, Agresti–Coull, Wilson-CC, mid-P y otras variantes no se aceptan como valores de `method` en CP-05. Jeffreys podrá existir únicamente dentro del harness de CP-04/CP-06 como comparador con semántica bayesiana.

## 7. Resultado público exacto

`calculate_interval()` seguirá devolviendo un diccionario. Las claves legacy deben permanecer y las nuevas sólo se añaden.

Esquema obligatorio:

```python
{
    # Legacy estable
    "lb": float,
    "ub": float,
    "method": str,
    "estimate": float,
    "n": int,
    "assumptions": {
        "successes": int | float,
        "failures": int | float,
        "normal_approximation_adequate": bool,
        "normal_approximation_required": bool,
        # Nuevas claves compatibles
        "independence": "unknown" | "assumed" | "verified",
        "common_success_probability": "required_not_verified",
        "bernoulli_binomial_model": "required",
    },

    # Metadata nueva
    "alpha": float,
    "confidence_level": float,
    "estimand": "proportion",
    "design": "one_sample",
    "sampling_model": "bernoulli_binomial",
    "interval_kind": (
        "frequentist_score"
        | "frequentist_exact_conservative"
        | "frequentist_asymptotic_legacy"
    ),
    "calibration_status": "not_calibrated",
    "successes": int | float,
    "failures": int | float,
    "input_mode": (
        "binary_data"
        | "predicate"
        | "counts"
        | "legacy_incidences_count"
        | "legacy_fractional_incidences"
    ),
    "design_requirements": [
        "independent_units",
        "common_success_probability",
        "bernoulli_binomial_sampling",
    ],
    "compatibility": {
        "legacy_api_used": bool,
        "deprecated": bool,
        "deprecation_reason": str | None,
        "binomial_contract_supported": bool,
        "recommended_input": "binary_data" | "from_counts",
        "legacy_method": bool,
    },
}
```

### 7.1 Tipos de `successes`/`failures`

- raw, callable y `from_counts`: enteros tanto top-level como en `assumptions`;
- legacy `incidences` integral: enteros si el valor representa un entero;
- legacy fraccionario: floats, preservando el cálculo histórico, pero `compatibility.binomial_contract_supported=False`.

### 7.2 Compatibilidad de las claves legacy

Las claves `lb`, `ub`, `method`, `estimate`, `n` y `assumptions` no pueden retirarse ni renombrarse en CP-05.

Dentro de `assumptions`, las cuatro claves existentes deben seguir presentes:

- `successes`;
- `failures`;
- `normal_approximation_adequate`;
- `normal_approximation_required`.

`normal_approximation_adequate` se conserva exclusivamente como sentinel legacy de compatibilidad. La nueva documentación debe advertir que no representa una garantía general.

## 8. Errores y warnings

### 8.1 `ValueError`

Debe producirse para:

- `alpha <= 0` o `alpha >= 1`;
- `data` vacío, multidimensional o con NaN/Inf;
- raw no binario cuando `incidences is None`;
- método distinto de `wilson`, `clopper_pearson`, `wald`;
- `independence` fuera de `unknown/assumed/verified`;
- `from_counts` con bool;
- `from_counts` con tipo no entero;
- `trials < 1`;
- `successes < 0` o `successes > trials`;
- `clopper_pearson` aplicado mediante `incidences` fraccionario.

La implementación puede mejorar el mensaje de error, pero no debe convertir inputs inválidos en resultados parciales.

### 8.2 `DeprecationWarning`

Se emite con `stacklevel=2` cuando se usa `incidences` numérico como conteo agregado. Debe distinguir:

- conteo integral: API legacy; migrar a `from_counts`;
- conteo fraccionario: API legacy y además fuera del modelo Bernoulli/binomial soportado.

No se emite para `incidences` callable.

### 8.3 `UserWarning` de Wald

Se conserva para compatibilidad cuando éxitos o fracasos observados son menores que 10. El mensaje debe indicar:

- que la advertencia es específica de Wald;
- que el threshold es legacy;
- que no constituye aprobación/calibración ni selección automática.

No debe bloquear el cálculo legacy de Wald.

## 9. Boundaries

Los siguientes casos son válidos:

```text
successes = 0
successes = trials
trials = 1
```

Wilson y Clopper–Pearson deben producir su intervalo definido sin convertir esos casos en error.

Wald conserva la conducta numérica legacy, incluidos intervalos degenerados en `0/0` o `1/1`; CP-05 no debe arreglarlo mediante clipping porque alteraría el contrato numérico histórico.

## 10. Lateralidad

No se añade `alternative`, `side`, `tail` ni otra opción one-sided a la API de producción en CP-05.

Todos los resultados de este stage son bilaterales. `confidence_level=1-alpha` corresponde al intervalo bilateral definido por cada método.

La lateralidad se difiere hasta disponer de preregistro/calibración propia.

## 11. Bootstrap de proporción

`BootstrapCI(stat="proportion")` permanece como API separada.

CP-05 no debe:

- reexportarlo como método de `PopulationProportionCI`;
- incluirlo entre los valores de `method`;
- utilizarlo como fallback;
- anunciarlo como alternativa automática del selector;
- transferirle la calibración de Wilson/Clopper–Pearson.

La revisión de su contrato binario y su calibración requiere un work item separado o una decisión posterior explícita.

## 12. Contrato fail-closed de `MethodSelector`

CP01-F-003 debe corregirse en CP-05 sin seleccionar un reemplazo automático.

### 12.1 Regla

Antes de entrar en la política v2/v3 de medias, si:

```python
report.estimand is Estimand.PROPORTION
```

el selector debe devolver inmediatamente una decisión fail-closed.

Contrato del resultado:

```python
selected_method = None
status = InferenceDecisionStatus.NOT_CALIBRATED
guarantee = InferenceGuarantee.NOT_CALIBRATED
alternatives = ()
capabilities = capabilities_for(report.design, report.estimand)  # actualmente ()
```

`parametric_recommended` debe resultar `False`.

La razón debe expresar que el routing automático para proporciones no está calibrado y que ningún método de media puede transferirse al estimando proporción.

### 12.2 Robustness placeholder

Mientras `InferenceDecision` exija un `RobustnessResult`, la implementación podrá construir un resultado `INSUFFICIENT`/equivalente únicamente como soporte estructural del objeto de decisión. Ese campo no debe reinterpretarse como una evaluación de robustez de medias aplicada a Bernoulli.

### 12.3 Prohibiciones

La corrección no puede:

- seleccionar Wilson;
- seleccionar Clopper–Pearson;
- seleccionar Wald;
- registrar `automatic_selection_allowed=True` para proporciones;
- evaluar shape/outliers para decidir el método de proporción;
- devolver alternativas de media (`bootstrap_bca_mean_ci`, Wilcoxon, etc.).

El fail-closed debe ocurrir antes de que `_alternatives(InferenceDesign.ONE_SAMPLE)` produzca alternativas de media o antes de que la política de robustez de medias determine un método.

## 13. Matriz legacy -> API canónica

| Uso existente | Resultado CP-05 | Ruta recomendada |
|---|---|---|
| `PopulationProportionCI(binary_data)` | preservado | igual |
| `PopulationProportionCI(binary_data, method="wilson")` | preservado | igual |
| `PopulationProportionCI(binary_data, method="wald")` | preservado + metadata legacy | Wilson/CP según decisión explícita del usuario; no auto |
| `PopulationProportionCI(dummy, incidences=3)` | preservado + `DeprecationWarning` | `PopulationProportionCI.from_counts(3, len(dummy))` |
| `PopulationProportionCI(dummy, incidences=3.7)` | cálculo Wilson/Wald preservado temporalmente + warning fuerte + no soporte binomial | no existe migración binomial equivalente; redefinir estimando si realmente es ponderado |
| `PopulationProportionCI(data, incidences=predicate)` | preservado | igual |
| import interno desde `.parametric` | preservado | importar desde `pyMagicStat.inference` |
| `MethodSelector` + `Estimand.PROPORTION` | deja de devolver método de media | `NOT_CALIBRATED`, sin selección |

## 14. Matriz de compatibilidad contractual

| Elemento | Compatibilidad exigida |
|---|---|
| default `method="wilson"` | exacta |
| fórmula Wilson existente | exacta dentro de tolerancia float64 |
| fórmula Wald existente | exacta; sin clipping |
| claves legacy de salida | presentes y semánticamente preservadas |
| constructor positional existente | preservado |
| callable existente | preservado |
| ruta import `.parametric` | preservada |
| export nuevo `pyMagicStat.inference` | añadido |
| conteo agregado integral legacy | cálculo preservado + warning |
| conteo agregado fraccionario legacy | cálculo Wilson/Wald preservado temporalmente + warning/metadata |
| mean routing | no debe cambiar fuera del fix de estimando proporción |
| ANOVA/EL/GOF | fuera de alcance, cero cambios incidentales |

## 15. Tests obligatorios para CP-05

### 15.1 Regresión legacy

1. Default sigue siendo Wilson.
2. El test Wilson existente conserva los mismos límites dentro de tolerancia.
3. `incidences=0` conserva estimate y límites históricos.
4. `method="wald"` conserva fórmula y ausencia de clipping.
5. Wald en `x=0` conserva `[0,0]` y en `x=n` conserva `[1,1]`.
6. Método case-insensitive preservado.
7. Callable produce el mismo conteo/estimate que el comportamiento histórico.
8. Raw booleano preservado.
9. Raw `0/1` entero y float preservado.
10. NaN/Inf/multidimensional/vacío siguen fallando.
11. Raw no binario sigue fallando.
12. Conteo numérico legacy fuera de `[0,n]` sigue fallando.
13. Las seis claves/estructuras legacy (`lb`, `ub`, `method`, `estimate`, `n`, `assumptions`) permanecen.
14. Las cuatro claves legacy dentro de `assumptions` permanecen.
15. Input del caller no se muta.

### 15.2 `from_counts`

16. `from_counts(0,1)` válido para Wilson.
17. `from_counts(1,1)` válido para Wilson.
18. `from_counts(0,n)` y `(n,n)` válidos para Clopper–Pearson.
19. `from_counts(x,n)` reproduce `estimate=x/n` y `n=trials`.
20. Acepta `np.integer`.
21. Rechaza `bool`.
22. Rechaza floats, incluidos `3.0`.
23. Rechaza `trials=0` y negativos.
24. Rechaza `successes<0` y `successes>trials`.
25. No requiere ni fabrica datos raw observables en el contrato público.

### 15.3 Métodos y oráculos

26. Wilson coincide con SciPy en un grid determinista que incluya `n` pequeño, interior y boundaries.
27. Clopper–Pearson coincide con `scipy.stats.binomtest(...).proportion_ci(method="exact")` en grid determinista válido.
28. Wilson mantiene bounds `[0,1]` en el grid de CP-01.
29. Wilson mantiene simetría complementaria dentro de tolerancia float64.
30. Wilson mantiene monotonicidad observada con respecto a `x` para `n,alpha` fijados.
31. Clopper–Pearson mantiene simetría complementaria esperada dentro de tolerancia.
32. Wald conserva casos fuera de `[0,1]` previamente observados; el test debe impedir clipping accidental.
33. Nombres no autorizados (`exact`, `beta`, `jeffreys`, `agresti_coull`, `wilsoncc`, `midp`) fallan explícitamente.
34. `clopper_pearson` con legacy `incidences` fraccionario falla explícitamente.

### 15.4 Metadata

35. `confidence_level == 1-alpha`.
36. `estimand == "proportion"`.
37. `design == "one_sample"`.
38. `sampling_model == "bernoulli_binomial"`.
39. Cada método devuelve el `interval_kind` aprobado.
40. Pre-CP06, todos devuelven `calibration_status="not_calibrated"`.
41. `successes/failures` top-level concuerdan con `assumptions`.
42. `input_mode` distingue raw/predicate/counts/legacy.
43. `design_requirements` no afirma que independencia haya sido probada.
44. `independence` conserva `unknown/assumed/verified` y rechaza otros valores.
45. Resultado completo es JSON-serializable.
46. Legacy fraccionario marca `binomial_contract_supported=False`.
47. Raw/callable/from_counts válidos marcan `binomial_contract_supported=True`.
48. Wald marca `legacy_method=True`; Wilson/Clopper–Pearson, `False`.

### 15.5 Deprecaciones/warnings

49. `incidences` numérico integral emite `DeprecationWarning` y recomienda `from_counts`.
50. `incidences` fraccionario emite warning distinto que declara incompatibilidad con el modelo binomial.
51. `incidences` callable no emite warning de deprecación por este motivo.
52. `from_counts` no emite warning de API legacy.
53. Wald con éxitos/fracasos menores que 10 emite `UserWarning` específico de Wald.
54. Wilson y Clopper–Pearson no emiten el warning legacy de Wald.

### 15.6 Export y routing

55. `from pyMagicStat.inference import PopulationProportionCI` funciona.
56. La ruta histórica `.parametric` sigue funcionando y apunta a la misma clase.
57. `PopulationProportionCI` está en `pyMagicStat.inference.__all__`.
58. `MethodSelector` con `Estimand.PROPORTION` en política v2 devuelve `selected_method=None`.
59. El status anterior es `NOT_CALIBRATED` y guarantee `NOT_CALIBRATED`.
60. No devuelve alternativas de media.
61. `parametric_recommended` es `False`.
62. La misma conducta fail-closed se conserva con política v3.
63. El fail-closed de proporción no necesita shape/outlier para nombrar método; test mediante stub/spy debe demostrar que no atraviesa la selección de medias.
64. Tests existentes de routing de media permanecen verdes y con decisiones sin cambios.
65. ONE_WAY permanece `NOT_CALIBRATED` como antes.
66. Capability registry de proporciones no adquiere ninguna capability automática en CP-05.

### 15.7 Aislamiento de alcance

67. `BootstrapCI(stat="proportion")` no aparece como `method` de esta clase.
68. No aparece como alternativa automática del selector de proporción.
69. No se modifican resultados de EL.
70. No se modifican contratos ANOVA/one-way.
71. No se modifica GOF Binomial/Poisson.

## 16. Tests que NO constituyen calibración

Los tests de CP-05 pueden demostrar equivalencia matemática, compatibilidad, invariantes y conducta numérica. No pueden por sí solos cambiar `calibration_status` ni autorizar selección automática.

La cobertura frecuentista, conservadurismo y regiones de desempeño pertenecen a CP-04/CP-06.

## 17. Decisiones diferidas

Quedan explícitamente fuera de CP-03:

- criterios numéricos de aceptación de cobertura/ancho;
- grid final de calibración;
- Jeffreys como API pública;
- Agresti–Coull, Wilson-CC, mid-P;
- intervalos one-sided;
- finite population correction;
- weighted/survey proportions;
- cluster/repeated-measures proportions;
- dos proporciones o diferencias de proporciones;
- sample-size planning;
- revisión/calibración de `BootstrapCI(stat="proportion")`;
- habilitar capability/routing automático para Wilson o cualquier otro método;
- fecha/version exacta de retirada de APIs legacy más allá de la condición mínima fijada aquí.

## 18. Archivos permitidos para futura CP-05

La implementación posterior, si es autorizada, podrá requerir únicamente rutas directamente relacionadas:

- `pyMagicStat/inference/parametric.py` o un módulo nuevo focalizado de proporciones si Cortex demuestra que reduce acoplamiento sin cambiar API;
- `pyMagicStat/inference/__init__.py`;
- `pyMagicStat/inference/selector.py`;
- tests focalizados de proporciones/routing;
- documentación pública focalizada;
- records de Knowledge Base vinculados al stage.

Cambios a ANOVA, EL, GOF, políticas de medias, thresholds de robustness o `main` están prohibidos.

## 19. Registro canónico pendiente

La gobernanza exige que el conocimiento modificado esté indexado en `knowledge/registry.json` en el mismo cambio destinado a integración. Antes de promover este stage a PR deberán quedar registrados, como mínimo, la evidencia CP-01, la decisión aceptada CP-02 y el estado de CP-03. Este requisito de bookkeeping no autoriza modificar `main` ni abrir PR.

## 20. Criterio de salida de CP-03

CP-03 puede pasar de `under_review` a `complete/accepted` sólo si el Product Owner aprueba explícitamente este contrato, incluyendo:

1. firmas públicas;
2. `from_counts` como API agregada canónica;
3. estrategia de deprecación de `incidences` numérico;
4. esquema de resultado y metadata;
5. nombre `clopper_pearson`;
6. Wald como legacy sin retirada programada;
7. fail-closed del selector;
8. matriz de 71 tests obligatorios;
9. decisiones diferidas y límites de alcance.

La aprobación de CP-03 habilita **CP-04: preregistro del experimento/calibración**. No habilita todavía CP-05 ni implementación de producción.