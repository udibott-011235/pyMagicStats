# Deuda técnica

## TD-API-001: pruebas t desde estadísticos resumidos

**Estado:** pendiente; no bloqueante.

**Componente:** `OneSampleTTest`, `TwoSampleTTest`, `PairedTTest` y contratos
relacionados de inferencia de medias.

**Alcance afectado:** usos donde no existe la muestra fila por fila y sólo se
dispone de tamaño muestral, media y desviación estándar.

### Necesidad de uso

La API actual está orientada a muestras completas. En reportes agregados,
integraciones con ERP/BI, restricciones de privacidad, publicaciones y
metaanálisis pueden estar disponibles únicamente los estadísticos resumidos.
pyMagicStats no debe asumir que el analista siempre posee las observaciones
individuales cuando el método puede resolverse exactamente sin reconstruirlas.

### Alcance futuro mínimo

- Una muestra: `n`, `mean`, `std` y `mu0`.
- Dos muestras independientes: `n1`, `mean1`, `std1`, `n2`, `mean2` y `std2`.
- Variantes Welch y Student con varianza agrupada, incluyendo alternativas
  bilateral y unilaterales.
- Estadístico t, grados de libertad, p-value, diferencia estimada, error
  estándar e intervalo de confianza.
- Convención explícita de desviación estándar muestral (`ddof=1`), validación
  de dominio y metadato `input_mode = raw | summary`.
- Contrato público no ambiguo y compatible con la entrada actual de muestras.

### Límite matemático de la prueba pareada

Los resúmenes marginales de dos grupos no bastan para una prueba t pareada,
porque no contienen la variabilidad de las diferencias. La ruta resumida debe
recibir `n`, `mean_diff` y `std_diff`, o información equivalente que permita
obtener `std_diff`, como la covarianza o correlación junto con ambas
desviaciones.

Debe fallar explícitamente si sólo se proporcionan `n1`, `mean1`, `std1`,
`n2`, `mean2` y `std2` sin información sobre el emparejamiento. No se deben
simular ni reconstruir observaciones sintéticas para ocultar esa insuficiencia.

### Criterio de cierre

- La ruta resumida coincide, dentro de tolerancias documentadas, con la ruta
  basada en los datos crudos que originaron los resúmenes.
- Welch usa los grados de libertad de Welch-Satterthwaite y Student documenta
  el supuesto de varianzas iguales.
- Se cubren `n < 2`, desviación cero, no finitos, tamaños desiguales y
  heterocedasticidad fuerte.
- Tests adversariales impiden mezclar desviación poblacional y muestral.
- La prueba pareada rechaza resúmenes marginales insuficientes.
- La documentación incluye ejemplos de una muestra, dos muestras
  independientes y diferencias pareadas resumidas.
- La API conserva compatibilidad hacia atrás.

### Generalización posterior

Este pendiente establece un criterio arquitectónico más amplio: revisar cada
método para admitir entradas resumidas sólo cuando los estadísticos disponibles
sean matemáticamente suficientes. No autoriza cambios en selector, ANOVA,
empirical likelihood, bootstrap ni calibraciones no vinculadas.

## Numerical hardening for subnormal float64 scales

**Estado:** pendiente; no bloqueante para el alcance actual.  
**Componente:** `DataQualityAssessment`, diagnóstico de varianza cero o casi cero.  
**Alcance afectado:** muestras `float64` con magnitudes extremadamente pequeñas.

### Causa técnica

La comprobación actual de degeneración usa una tolerancia relativa equivalente,
aproximadamente, a:

```python
(np.finfo(float).eps * scale) ** 2
```

Cuando `scale` está en magnitudes extremas —aproximadamente desde `1e-139` y,
con mayor riesgo, alrededor de `1e-146` o inferiores— el producto y su cuadrado
pueden entrar en la región subnormal de IEEE-754 o underflowear a cero. La
tolerancia deja entonces de conservar de forma fiable la invariancia de escala.

Casos que deben incluirse en una futura evaluación:

```python
[1.0, 2.0, 3.0]
[1e-100, 2e-100, 3e-100]
[1e-140, 2e-140, 3e-140]
[1e-160, 2e-160, 3e-160]
```

### Riesgo y alcance

El diagnóstico podría clasificar de forma distinta muestras que sólo difieren
por un factor de escala, o perder la capacidad de detectar degeneración cuando
la tolerancia underflowea. No se considera bloqueante para el alcance actual de
pyMagicStat, orientado a BI, retail, wholesale, logística, pricing, inventarios
y analítica comercial, donde estas escalas no son habituales.

Debe resolverse antes de declarar soporte numérico robusto para dominios que sí
puedan operar rutinariamente en escalas extremas, entre ellos ciertos casos
farmacéuticos, bioestadísticos, físicos, aeroespaciales o de instrumentación.

### Estrategia futura recomendada

La corrección debe estudiar explícitamente la invariancia de escala. Una línea
preferente es normalizar internamente los datos para evaluar su dispersión en
una escala segura y después mapear el diagnóstico al contrato público. También
deben cubrirse valores normales, subnormales, underflow, overflow y escalas
mixtas con pruebas metamórficas que exijan la misma decisión al multiplicar una
muestra por factores finitos.

No se recomienda introducir un threshold absoluto arbitrario: trasladaría el
fallo a otra unidad de medida y rompería el carácter relativo del diagnóstico.

### Criterio de cierre

- La decisión de degeneración es invariante ante cambios de escala finitos
  representables dentro del dominio probado.
- No se producen underflow/overflow ni warnings numéricos durante el diagnóstico.
- Los casos de escala comercial ya cubiertos conservan el comportamiento
  auditado.
- La estrategia y sus límites quedan respaldados por tests numéricos dedicados.
