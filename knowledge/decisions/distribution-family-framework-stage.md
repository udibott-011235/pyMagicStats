# STAGE-DIST-FAMILIES-001 — Distribution Family Framework

- **Estado general:** `in_progress`
- **Checkpoint actual:** `CP03 — IN_PROGRESS / ARCHITECTURE_FROZEN`
- **Fecha de apertura:** 2026-09-06
- **Baseline canónico:** `origin/main` @ `402e4601df460811779b3238c2526ac12f463a67`
- **Rama de integración de CP01 (`merged` / `archived`):** `feature/distribution-family-framework-cp01`
- **Rama de cierre post-merge de CP01:** `docs/distribution-family-framework-cp01-post-merge`
- **Baseline de CP02:** `origin/main` @ `ccff392af13d2cb52d1f3888a986ef58be0099e2`
- **Rama de implementación de CP02:** `feature/distribution-family-framework-cp02-continuous-core`
- **Rama de cierre post-merge de CP02:** `docs/distribution-family-framework-cp02-post-merge`
- **Baseline de CP03:** `origin/main` @ `02a65c80c5da10295d6eeef42e691772d0686ca2`
- **Rama de CP03:** `feature/distribution-family-framework-cp03-discrete-core`
- **Owner de decisión:** `decision-owner`
- **Arquitectura:** `statistical-software-architecture`
- **Implementación:** `implementation-engineering`
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
| CP01 — Family architecture and contracts | `COMPLETE` | Arquitectura aceptada e integrada mediante PR #6 |
| CP02 — Continuous distribution core | `COMPLETE` | Implementación `e9ef63b…`, `ADVERSARIAL_PASS` e integración mediante PR #8 en `main@aa5723d…` |
| CP03 — Discrete distribution core | `IN_PROGRESS` | Arquitectura `FROZEN` en `DEC-012`; implementación `PENDING` |
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

## Artefactos de CP02

- Contrato ejecutable congelado: [`distribution-family-framework-cp02-contract.md`](distribution-family-framework-cp02-contract.md)
- Evidencia de implementación: [`../evidence/distribution-family-framework-cp02-evidence.md`](../evidence/distribution-family-framework-cp02-evidence.md)
- Rama de implementación integrada: `BR-019`
- Rama de cierre post-merge: `BR-020`

## Artefactos de CP03

- Contrato ejecutable congelado: [`distribution-family-framework-cp03-contract.md`](distribution-family-framework-cp03-contract.md)
- Evidencia de reconnaissance y baseline: [`../evidence/distribution-family-framework-cp03-baseline.md`](../evidence/distribution-family-framework-cp03-baseline.md)
- Rama de materialización: `BR-021`

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
5. Arquitectura revisó y aceptó el candidato exacto
   `c63b48eafc439de8857207fd21c1b593e38a3187`.
6. La aceptación cierra la revisión arquitectónica de CP01, pero no integra la
   rama ni autoriza PR, merge, CP02, CP03 o implementación.
7. Antigravity auditó PR #6 en el integration head
   `3f9acd5a51ce38ae62b9800d50efb0949c6531f0` y emitió `ADVERSARIAL_PASS`,
   con cero `BLOCKER`, `MAJOR` y `MINOR`.
8. PR #6 se integró mediante el merge commit
   `46f827dd107aa9e6f940f0de085fbb91075ff049`, con parents
   `402e4601df460811779b3238c2526ac12f463a67` y
   `3f9acd5a51ce38ae62b9800d50efb0949c6531f0`.
9. La integración completó CP01 sin cambiar comportamiento de producción ni
   autorizar checkpoints posteriores.

## Dependencias y límites

El contrato consume la arquitectura congelada por el Lead Architect y el
baseline exacto indicado arriba. No consume ni modifica el track separado de
Manual UAT1 B3. Tampoco transfiere evidencia desde ANOVA, intervalos de
proporción, empirical likelihood, robustez de muestreo o Gate 2 hacia las
nuevas familias.

## Estado de cierre arquitectónico de CP01

CP01 está `COMPLETE` y su integración está `COMPLETE` mediante PR #6 en
`main@46f827dd107aa9e6f940f0de085fbb91075ff049`. El stage general permanece
`in_progress`; CP02 está `COMPLETE`, CP03 está `IN_PROGRESS` con arquitectura
`FROZEN` e implementación `PENDING`, y CP04–CP08 permanecen `NOT_STARTED` con
autorización independiente requerida.

## Apertura autorizada de CP02

CP02 se abre desde `main@ccff392af13d2cb52d1f3888a986ef58be0099e2`
con el contrato ejecutable congelado en `DEC-011`. Su alcance se limita al core
continuo compartido, `GammaFamily`, `ExponentialFamily`, tests deterministas y
evidencia asociada. CP03–CP08 permanecen `NOT_STARTED`.

## Aceptación arquitectónica de CP02

Arquitectura aceptó inicialmente la implementación exacta
`874c03c70c028d0ca4966331b6fc91ec35613caa` del core continuo determinista
congelado en `DEC-011`. La auditoría adversarial pre-merge posterior devolvió
`ADVERSARIAL_CHANGES_REQUIRED`: `FINDING-ADV-CP02-001` (`MINOR`),
`FINDING-ADV-CP02-002` (`INFO`) y `FINDING-ADV-CP02-003` (`INFO`).

El commit `e9ef63b802a8cb08ea38b32e87b206432c08b120` corrigió
`ADV-CP02-001` y endureció `ADV-CP02-002`. Arquitectura acepta ahora ese SHA
exacto como candidato de implementación vigente de CP02; `874c03c…` se
conserva como aceptación histórica, supersedida por la remediación.

El primer governance/audit head `7e009503…` preservó el resultado
`ADVERSARIAL_CHANGES_REQUIRED`. Tras la remediación `e9ef63b…`, Antigravity
reauditó de forma independiente el governance head exacto
`9cf25c157a4f4114f41ae74d4e04e009392414e3` y emitió `ADVERSARIAL_PASS` con
0 `BLOCKER`, 0 `MAJOR`, 0 `MINOR` y 1 `INFO`. `ADV-CP02-001` queda
`CLOSED_REMEDIATED`, `ADV-CP02-002` queda `CLOSED_HARDENED` y
`ADV-CP02-003` permanece únicamente como INFO preexistente fuera de alcance.

El commit `b3d116f2f3e82b74ab6fb5f8337e217104c3bca6` materializó el resultado
adversarial como governance head pre-merge final. PR #8 integró ese head mediante
el merge commit `aa5723d2cb7dbaf48e6f9059368b9fdaeeb7926c`, cuyos parents son
`ccff392af13d2cb52d1f3888a986ef58be0099e2` y
`b3d116f2f3e82b74ab6fb5f8337e217104c3bca6`.

La integración y la gobernanza de CP02 están cerradas; CP02 queda `COMPLETE`.
El stage general permanece `IN_PROGRESS`; CP03 está `IN_PROGRESS` con
arquitectura `FROZEN` e implementación `PENDING`, y CP04–CP08 siguen
`NOT_STARTED`. CP02
no acepta ni implementa fitting, `FitResult`,
`FittedDistribution`, estimación o incertidumbre de parámetros, GOF,
calibración, `MethodSelector`, routing, familias discretas ni CP03.

## Apertura y arquitectura congelada de CP03

CP03 se abre desde `main@02a65c80c5da10295d6eeef42e691772d0686ca2` en
`BR-021`. Reconnaissance queda `COMPLETE`; `DEC-012` congela la arquitectura
del core discreto, `SupportKind.DISCRETE`, el modelo de objetos discreto y la
parametrización canónica `NegativeBinomialFamily().bind(r=..., p=...)`.

Esta materialización es exclusivamente de gobernanza. No implementa producción
ni tests, no modifica las APIs discretas legacy y no autoriza fitting, GOF,
selector/routing, otras familias discretas o CP04. CP04–CP08 permanecen
`NOT_STARTED`.

## Siguiente acción

Arquitectura debe revisar el candidato de materialización de `DEC-012` y
`EV-009`. La implementación de CP03 permanece `PENDING` y requiere autorización
separada; no hay autorización de push, PR ni merge.
