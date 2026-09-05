# CP-ANOVA-04 — Quantum production smoke evidence

- Fecha: `2026-09-05`
- Candidate SHA ejecutado: `5a116a4e8672dadd3fe57a51f4186f70d1440afd`
- Rama de candidate: `feature/anova-production-candidate`
- Host: `quantum`, Ubuntu 24.04.4 LTS, Linux 6.8.0-138-generic x86_64
- Checkout: clon aislado `~/workspace/pyMagicStats-anova-validation`
- Estado del árbol antes de instalar: limpio
- Ejecución: determinista, sin calibración Monte Carlo ni carga pesada

## Entorno

```text
Python       3.12.3
NumPy        2.5.2
SciPy        1.18.1
statsmodels  0.15.0
pytest       9.1.1
```

## Comando objetivo

```bash
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
nice -n 10 \
python -m pytest tests/test_anova_production.py -vv
```

## Resultado

```text
collected 18 items
18 passed in 1.65s
```

PASS individual confirmado para:

- Classical vs `scipy.stats.f_oneway` y componentes SS/MS;
- `k=2` Classical: `F = pooled Student t^2`;
- `k=2` Welch: `F = Welch t^2` y df concordante;
- reconstrucción independiente de weights/correction/F/df Welch;
- rechazo de alpha inválido y k<2;
- rechazo de inputs no finitos y grupos constantes;
- defensa contra mutación del input luego de construcción;
- inmutabilidad efectiva de `ANOVAResult`;
- `to_dict()` JSON-ready y desacoplado del resultado;
- invariancia a traslación, escala común y permutaciones;
- caso construido de medias iguales con `F=0`, `p=1`;
- shape diagnostic severo no actúa como selector/veto automático;
- independencia `unknown` permanece explícitamente no resuelta;
- `MethodSelector` ONE_WAY continúa `NOT_CALIBRATED`;
- ejecución repetida determinista.

## Interpretación

Este resultado cierra el smoke/unit validation del production candidate en el SHA exacto indicado. No constituye todavía CP-ANOVA-05 ni evidencia de calibración inferencial/robustez fuera del modelo.

Antes de congelar CP-ANOVA-04 se requiere una regresión dirigida sobre assumptions/routing/inference para comprobar que la nueva superficie ANOVA no alteró contratos vecinos. Si esa regresión es PASS, el candidate puede congelarse y CP-ANOVA-05 pasa a ser el siguiente checkpoint.
