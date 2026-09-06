# STAGE-DIST-FAMILIES-001 — Distribution Family Framework

- **Estado general:** `in_progress`
- **Checkpoint actual:** `CP01 — READY_FOR_ARCHITECT_REVIEW`
- **Fecha de apertura:** 2026-09-06
- **Baseline canónico:** `origin/main` @ `402e4601df460811779b3238c2526ac12f463a67`
- **Rama de trabajo:** `feature/distribution-family-framework-cp01`
- **Owner de decisión:** `decision-owner`
- **Arquitectura:** `statistical-software-architecture`
- **Implementación documental:** `implementation-engineering`
- **QA adversarial futuro:** `adversarial-statistical-qa`

## Objetivo

Crear una arquitectura estadísticamente neutral para familias de probabilidad,
distribuciones ajustadas y resultados de ajuste, sin confundir esas entidades
con descriptivos muestrales, diagnósticos de forma o evaluaciones de bondad de
ajuste.

CP01 materializa únicamente el contrato aprobado, valida su compatibilidad con
la superficie actual y registra el stage. No implementa clases de familias,
ajustes, nuevas evaluaciones GOF ni selección automática.

## Checkpoints

| Checkpoint | Estado | Resultado esperado |
|---|---|---|
| CP01 — Family architecture and contracts | `READY_FOR_ARCHITECT_REVIEW` | Contrato congelado, inventario de compatibilidad, evidencia y registro canónico |
| CP02 | `NOT_STARTED` | Requiere autorización y contrato posteriores |
| CP03 | `NOT_STARTED` | Requiere autorización y contrato posteriores |
| CP04 | `NOT_STARTED` | Requiere autorización y contrato posteriores |
| CP05 | `NOT_STARTED` | Calibración GOF para familias ajustadas; no transferible desde Gate 2 |
| CP06 | `NOT_STARTED` | Requiere autorización y contrato posteriores |
| CP07 | `NOT_STARTED` | Requiere autorización y contrato posteriores |
| CP08 | `NOT_STARTED` | Requiere autorización y contrato posteriores |

## Artefactos de CP01

- Contrato arquitectónico: [`distribution-family-framework-contract.md`](distribution-family-framework-contract.md)
- Evidencia de materialización e inventario: [`../evidence/distribution-family-framework-cp01-evidence.md`](../evidence/distribution-family-framework-cp01-evidence.md)
- Índice canónico: [`../registry.json`](../registry.json)
- Estado de rama: `BR-017`

## Alcance autorizado

CP01 permite exclusivamente cambios de documentación y gobernanza bajo
`knowledge/**`. La producción, los tests, los experimentos y la documentación
pública permanecen sin cambios.

Las clases existentes `Distribution`, `NormalDistribution`,
`LognormalDistribution`, `BinomialDistribution` y `PoissonDistribution`
conservan nombres, comportamiento, exports y contratos. El inventario de CP01
asigna una categoría de migración futura, pero no ejecuta ninguna migración.

## Invariantes del stage

1. Una descripción de muestra, una familia de probabilidad, una distribución
   ajustada, una evaluación GOF y un diagnóstico de forma son conceptos
   distintos.
2. No rechazar una hipótesis GOF no demuestra identidad distributiva.
3. El backend numérico predeterminado será SciPy; pyMagicStats conserva el
   contrato, la parametrización, la validación, la trazabilidad y el
   comportamiento fail-closed.
4. El muestreo exige RNG controlable por el caller y estado reproducible; no se
   autoriza RNG global implícito.
5. Las familias continuas y discretas exponen operaciones coherentes con su
   tipo; no se crean métodos artificiales para uniformar interfaces.
6. El soporte es explícito e independiente de los datos observados.
7. Una distribución ajustada es inmutable desde la perspectiva del usuario.
8. Los estados semánticamente distintos no se reducen a `NaN`.
9. No hay selección automática de familia, ranking de modelos ni cambios a
   `MethodSelector` en este stage.
10. Las decisiones de estimación y calibración reservadas permanecen como
   `ARCHITECT_DECISION_REQUIRED`.
11. Cada checkpoint posterior requiere autorización independiente.

## Cronología de revisión

1. El candidato `72ecdba1b60d9efb53dfe612cf7ee4beeb3e76e5` se creó
   localmente bajo la regla inicial de handoff sin push.
2. Después de `READY_FOR_ARCHITECT_REVIEW`, Arquitectura autorizó por separado
   el push de ese SHA exacto únicamente para revisar su contenido versionado.
3. Ese push no autorizó PR, merge, CP02, CP03 ni implementación.
4. La primera revisión arquitectónica conservó `72ecdba…` y solicitó un único
   commit documental de seguimiento con aclaraciones A–F.

## Dependencias y límites

El contrato consume la arquitectura congelada por el Lead Architect y el
baseline exacto indicado arriba. No consume ni modifica el track separado de
Manual UAT1 B3. Tampoco transfiere evidencia desde ANOVA, intervalos de
proporción, empirical likelihood, robustez de muestreo o Gate 2 hacia las
nuevas familias.

## Criterio de salida de CP01

CP01 queda listo para revisión arquitectónica cuando:

1. el contrato y el inventario contienen todas las decisiones congeladas;
2. el registry y la proyección de branch lifecycle son válidos;
3. los tests de distribución existentes permanecen verdes;
4. el diff contiene sólo archivos `knowledge/**` autorizados;
5. no se introdujo acoplamiento a un dominio de aplicación;
6. existe un commit local exacto y no se hizo push, PR ni merge.

## Siguiente acción

ChatGPT, en rol `statistical-software-architecture`, debe revisar el SHA
candidato exacto de CP01. Cortex se detiene después del commit local y no inicia
CP02 ni implementa ninguna familia.
