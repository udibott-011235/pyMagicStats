# CP-ANOVA-04 remediation smoke — location stability

- Fecha: `2026-09-05`
- Rama: `fix/anova-location-stability`
- Candidate SHA ejecutado: `83bfe547563d977f1ed0dd0f43629c281744488c`
- Production-code remediation commit: `376677ca32dfd1e3f5b5b64bec48e3160c35d5a9`
- Host: `quantum`
- Entorno: Python 3.12.3, pytest 9.1.1
- Ejecución: single-thread BLAS/OpenMP, `nice -n 10`

## Alcance

```text
tests/test_anova_production.py
tests/test_anova_location_stability.py
```

## Resultado

```text
collected 21 items
21 passed in 1.62s
```

## Interpretación

La remediación de estabilidad ante gran location común pasa todos los tests de producción existentes y los nuevos invariantes específicos:

- Classical preserva statistic/p-value tras offset común `1e12`;
- Welch preserva statistic/p-value/df tras offset común `1e12`;
- varianzas de grupo localmente centradas son estables a traslación;
- el oracle Welch de SciPy para este edge case se adjudica sobre datos centrados por un origen común, porque su path raw `equal_var=False` calcula medias/varianzas absolutas y presenta sensibilidad numérica a grandes offsets.

Este PASS no vuelve a congelar todavía CP-ANOVA-04. Antes se requiere repetir la regresión dirigida de assumptions/selector/capabilities y después la suite completa CP-ANOVA-05 sobre el candidate remediado.
