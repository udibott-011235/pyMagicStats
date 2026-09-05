# ANOVA ↔ experiment optimizer — Product Owner direction

- Fecha: `2026-09-05`
- Rama: `audit/anova-statistical-closure`
- Estado: `roadmap direction; no implementation in current checkpoint`
- Autoridad: `decision-owner / Product Owner`

## Dirección del Product Owner

ANOVA debe cerrarse primero como una ejecución clásica precisa, eficiente, reproducible y estadísticamente auditable.

En un **Step 2 posterior**, el módulo ANOVA debe integrarse con `pyMagicStat.optimization.orchestrator` para que pueda utilizarse como herramienta operativa de evaluación dentro de workflows de experimentación.

El `orchestrator` no debe imponer una estadística universal. Su responsabilidad es coordinar búsqueda, iteración, evaluación de candidatos y ejecución. La lógica estadística y las métricas propias corresponden al método que se esté evaluando.

Ejemplos:

- Kruskal-Wallis conserva sus estadísticos y, donde corresponda a su contrato, evaluaciones complementarias con Mann-Whitney;
- ANOVA clásico debe evaluar mediante su estadístico F y p-value, junto con sus grados de libertad y componentes ANOVA;
- Welch ANOVA debe utilizar su propio estadístico F de Welch, p-value y grados de libertad efectivos;
- futuros métodos deben aportar su propio contrato estadístico sin ser forzados a imitar la salida de Kruskal-Wallis o ANOVA.

## Estado actual observado

`pyMagicStat/optimization/orchestrator.py` contiene:

1. `StatisticalEvaluator`, que instancia una clase estadística y ejecuta `run_test()`;
2. `OptimizedExperimentationIteration`, con estrategias `greedy`, `exhaustive` y `simulated_annealing`.

La implementación actual está muy acoplada al output histórico de `kruskalWallisTest`: espera `Total_SS`, `Groups`, `SSW` y p-values individuales. Esta interfaz será revisada cuando corresponda trabajar formalmente el optimizador.

No es requisito del Step 1 ANOVA adaptar el nuevo motor a ese contrato histórico.

## Contrato que ANOVA Step 1 debe preservar

Aunque el optimizador se modernizará posteriormente, la salida ANOVA debe conservar suficientes componentes matemáticos para ser reutilizable sin reescribir el motor estadístico:

### Classical one-way ANOVA

- `statistic` / F;
- `p_value`;
- `numerator_df`;
- `denominator_df`;
- `ss_between`;
- `ss_within`;
- `ss_total` o identidad verificable `ss_between + ss_within`;
- `mean_square_between`;
- `mean_square_within`;
- tamaños, medias y varianzas por grupo;
- diagnostics/assumptions;
- metadata de método y versión.

### Welch one-way ANOVA

- F de Welch;
- `p_value`;
- grados de libertad numerador y denominador;
- pesos por grupo y media ponderada cuando formen parte del cálculo;
- término/corrección de Welch;
- tamaños, medias y varianzas por grupo;
- diagnostics/assumptions;
- metadata de método y versión.

No se debe fingir que Welch posee exactamente la misma descomposición de sumas de cuadrados del ANOVA clásico.

## Responsabilidades futuras del orquestador

Cuando se abra el Step 2, el orquestador deberá convertirse en un motor operativo capaz de trabajar con evaluadores específicos por método.

Conceptualmente:

```text
ExperimentOrchestrator
        |
        +--> KruskalWallisEvaluator
        |       H / p-value / métricas propias
        |
        +--> ClassicalANOVAEvaluator
        |       F / p-value / df / SS / MS
        |
        +--> WelchANOVAEvaluator
        |       F_W / p-value / df efectivos / weights
        |
        `--> futuros evaluadores
                contrato propio del método
```

El orquestador gestiona la operación; el evaluador estadístico define qué significa evaluar un candidato.

## Nota sobre optimización e inferencia

El uso de F y p-value dentro de una estrategia de búsqueda es válido como información operativa del evaluador, pero debe distinguirse del significado confirmatorio del p-value final si el mismo dataset fue usado adaptativamente para escoger grupos/configuraciones.

Esto no bloquea el uso del optimizador. Simplemente implica que, cuando se diseñe Step 2, deberá quedar explícito si una corrida es:

- exploratoria/operativa, o
- confirmatoria con diseño previamente fijado o validación independiente.

La distinción pertenece al contrato futuro del optimizador y no amplía el alcance del cierre ANOVA actual.

## Roadmap

```text
STEP 1 — ANOVA statistical closure
    Classical ANOVA
    Welch ANOVA
    F + p-value + df
    precisión numérica
    assumptions/residuals
    oracles
    calibración/evidencia
    Manual UAT eligibility
            |
            v
STEP 2 — Integration with experiment optimizer
    actualizar StatisticalEvaluator/orchestrator
    evaluadores específicos por prueba
    ANOVA usando F + p-value
    Kruskal-Wallis conservando su lógica propia
    reproducibilidad de estrategias de búsqueda
            |
            v
FUTURE — DOE / experimental design capabilities
```

## Impacto inmediato

No se modifica el scope del checkpoint ANOVA actual. La única restricción de arquitectura es evitar outputs innecesariamente cerrados o específicos que impidan que el motor ANOVA pueda ser consumido después por un evaluador del `orchestrator`.
