# Principios de inferencia vigentes

**Registro:** `TH-001`  
**Estado:** `accepted`  
**Alcance:** inferencia de medias y diagnósticos reutilizables para diseños de
una vía en `main`.

## 1. El estimando precede al método

Antes de seleccionar una prueba se declara qué se estima, qué grupos o pares
forman el diseño y cuál es la unidad independiente. Mann–Whitney o
Kruskal–Wallis no son reemplazos automáticos de pruebas de medias porque su
objetivo inferencial puede ser distinto.

## 2. Se diagnostica la variable relevante al diseño

- Una muestra: observaciones de la media objetivo.
- Pares: diferencias dentro de cada par.
- Dos grupos independientes: forma dentro de cada grupo/residuos centrados.
- Una vía: residuos centrados dentro de cada grupo, balance,
  heterocedasticidad e independencia del diseño.

Para ANOVA no corresponde probar normalidad sobre todos los valores agrupados:
una mezcla de grupos con medias diferentes puede parecer no normal aun cuando
los errores dentro de cada grupo satisfagan el modelo. La validación relevante
es la de residuos o errores dentro del diseño.

## 3. Normalidad no es un interruptor binario

Shapiro–Wilk, D’Agostino, skewness, kurtosis y outliers aportan evidencia con
potencia y sensibilidad diferentes. “No rechazar” no demuestra normalidad;
“rechazar” no determina por sí solo que la inferencia t sea inválida. La decisión
de robustez combina forma, tamaño, contaminación, diseño y evidencia calibrada.

## 4. El tamaño muestral no reemplaza los supuestos

`n >= 30` no es una garantía ni activa automáticamente CLT o bootstrap. El
tamaño participa como una dimensión de una política calibrada y versionada.
Todo alivio asintótico conserva estado `caution` y los límites de la calibración.

## 5. Diagnóstico, política y selección son capas distintas

1. `InferenceValidator` observa y estructura diagnósticos.
2. `SamplingRobustness` interpreta esos diagnósticos según una política
   versionada.
3. `MethodSelector` recomienda un método y alternativas sin transformar datos.

Esta separación permite auditar qué se observó, qué política se aplicó y qué
decisión resultó.

## 6. Heterocedasticidad no se decide con un pretest

Welch es el default documentado para dos grupos. Student requiere una decisión
explícita de varianzas iguales. Levene es diagnóstico, no un interruptor que
selecciona automáticamente Student o Welch.

## 7. Bootstrap conserva el estimando y es explícito

Bootstrap no “aplica el TLC” ni debe aparecer como fallback oculto. Debe
declarar estadístico, método de intervalo, backend y estado aleatorio. Con
`random_state` explícito, el resultado debe ser reproducible y no avanzar el
generador del usuario.

## 8. Reproducibilidad forma parte de la validez

Una afirmación de calibración requiere commit, entorno, semilla, comando,
matriz de escenarios y outputs. Pasar tests de software es necesario, pero no
suficiente para afirmar control de error o cobertura.

## 9. Límite actual para ANOVA

`main` contiene diagnósticos reutilizables de una vía, incluidos residuos
centrados por grupo. La selección ANOVA/Welch ANOVA no debe declararse calibrada
con la matriz de una sola media. Implementación, validación de supuestos y
calibración de error/potencia para múltiples grupos requieren evidencia propia.

## Evidencia asociada

- `Docs/inference-engine.md`
- `Docs/sampling-robustness-calibration.md`
- `experiments/robustness_calibration.py`
- `experiments/results/sampling_robustness_metadata.json`
- `experiments/results/sampling_robustness_summary.csv`

