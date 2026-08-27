# Deuda técnica

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

