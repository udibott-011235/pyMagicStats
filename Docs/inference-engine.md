# Motor de validación de inferencia

pyMagicStat separa tres responsabilidades:

1. `InferenceValidator` normaliza los datos y produce diagnósticos estructurados.
2. `SamplingRobustness` interpreta esos diagnósticos mediante una política explícita.
3. `MethodSelector` recomienda un procedimiento sin modificar los datos observados.

El tamaño muestral participa en la evaluación, pero `n >= 30` no constituye una
garantía ni activa remuestreo automático.

## Diagnósticos

El reporte incluye:

- tamaño, finitud, dimensionalidad y degeneración;
- skewness y kurtosis excedente;
- Shapiro-Wilk y D'Agostino cuando sus tamaños mínimos lo permiten;
- outliers extremos mediante MAD, con fallback IQR;
- Levene, razón de varianzas y balance para diseños con grupos;
- independencia como `verified`, `assumed` o `unknown`.

La independencia no se infiere mirando los valores. Si el diseño no la documenta,
el reporte conserva `not_assessed`.

## Datos relevantes por diseño

- Una muestra: observaciones usadas para estimar la media.
- Pareada: diferencias dentro de cada par.
- Dos muestras: residuos centrados dentro de cada grupo.
- Una vía: residuos centrados por grupo, balance y heterocedasticidad.

`validate_one_way` prepara los diagnósticos compartidos para ANOVA, pero esta rama
no implementa todavía el estadístico ANOVA.

## Ejemplo

```python
import numpy as np

from pyMagicStat.assumptions import InferenceValidator
from pyMagicStat.inference import MethodSelector

rng = np.random.default_rng(42)
group_a = rng.normal(10, 2, 80)
group_b = rng.normal(12, 5, 70)

validation = InferenceValidator().validate_two_sample(
    group_a,
    group_b,
    independence="assumed",
)
decision = MethodSelector().select(validation.report)

print(decision.to_dict())
# selected_method: "welch_t"
```

Welch es el valor predeterminado. Student solo se selecciona mediante
`equal_var=True`; Levene permanece disponible como diagnóstico.

## Ejecución estricta y diagnóstico

Las pruebas paramétricas rechazan por defecto una decisión `insufficient`:

```python
from pyMagicStat.inference.parametric import OneSampleTTest

test = OneSampleTTest(data, strict=True)
```

`strict=False` permite calcular el procedimiento solicitado y devuelve igualmente
`assumptions` e `inference_decision`. Esto es útil para investigación, pero no
convierte una advertencia en validación.

## Bootstrap

Bootstrap conserva explícitamente el estimando:

```python
from pyMagicStat.inference import BootstrapCI, BootstrapMeanDifferenceCI

mean_ci = BootstrapCI(
    data,
    stat="mean",
    interval_method="bca",
    random_state=42,
).compute()

difference_ci = BootstrapMeanDifferenceCI(
    group_a,
    group_b,
    interval_method="bca",
    random_state=42,
).compute()

variance_ci = BootstrapCI(
    data,
    stat="variance",
    ddof=1,
    random_state=42,
).compute()
```

Para varianza, `ddof=1` es el contrato predeterminado: el valor observado y
cada réplica usan el estimador de varianza muestral, dirigido a la varianza
poblacional convencional. `ddof=0` debe solicitarse explícitamente cuando el
estimando sea el segundo momento central empírico/MLE. El resultado registra el
`ddof` empleado.

Mann-Whitney y Kruskal-Wallis se reportan como procedimientos con estimandos de
rangos/distribuciones; no se presentan como reemplazos automáticos de pruebas de
medias.

## Varianza poblacional y chi-square

El intervalo chi-square no consume `SamplingRobustness`. Su pivote es exacto
sólo para una muestra independiente de una población normal, y un n grande no
elimina ese requisito. Por eso el contrato exige declarar el modelo:

```python
from pyMagicStat.inference.parametric import PopulationVarianceCI

interval = PopulationVarianceCI(
    data,
    population_normality="assumed",
).calculate_interval()
```

En modo estricto, `population_normality="unknown"` (predeterminado) y
`"not_assumed"` rechazan la inferencia. El diagnóstico de forma muestral puede
no contradecir, advertir o contradecir fuertemente la declaración, pero nunca
demuestra normalidad poblacional. Con `strict=False` todavía puede calcularse el
intervalo, marcado como `chi_square_validated: false`.

## Cambios de comportamiento

- Se eliminó el bootstrap denominado `apply_tlc`.
- Los errores estándar de pruebas t vuelven a ser analíticos y deterministas.
- Los intervalos de media usan cuantiles t cuando la desviación se estima.
- Los intervalos de proporción usan Wilson por defecto; `method="wald"` permanece disponible.
- Los intervalos chi-cuadrado de varianza exigen una política explícita de
  normalidad poblacional.
- `apply_transform=True` levanta `NotImplementedError`: transformar puede cambiar el estimando.
- Bootstrap acepta `random_state` y separa backend de `interval_method`.

Los umbrales de robustez son una política versionada. La versión
`mean-v2-2026-08` fue calibrada con 152 000 réplicas y 19 escenarios. Consulte
[el informe de calibración](sampling-robustness-calibration.md) y el
[runner reproducible](../experiments/robustness_calibration.py). Nuevas versiones
deben repetir esa matriz o documentar expresamente su ampliación.
