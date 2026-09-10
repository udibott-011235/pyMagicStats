# STAGE-DIST-FAMILIES-001 — Distribution Family Framework

- **Estado general:** `in_progress`
- **Checkpoint actual:** `CP04 — COMPLETE / GOVERNANCE_CLOSED`
- **Fecha de apertura:** 2026-09-06
- **Baseline canónico de apertura (CP01):** `origin/main` @ `402e4601df460811779b3238c2526ac12f463a67`
- **Rama de integración de CP01 (`merged` / `archived`):** `feature/distribution-family-framework-cp01`
- **Rama de cierre post-merge de CP01:** `docs/distribution-family-framework-cp01-post-merge`
- **Baseline de CP02:** `origin/main` @ `ccff392af13d2cb52d1f3888a986ef58be0099e2`
- **Rama de implementación de CP02:** `feature/distribution-family-framework-cp02-continuous-core`
- **Rama de cierre post-merge de CP02:** `docs/distribution-family-framework-cp02-post-merge`
- **Baseline de CP03:** `origin/main` @ `02a65c80c5da10295d6eeef42e691772d0686ca2`
- **Rama de CP03:** `feature/distribution-family-framework-cp03-discrete-core`
- **Rama de cierre post-merge de CP03:** `docs/distribution-family-framework-cp03-post-merge`
- **Baseline de CP04:** `origin/main` @ `b3f35d4d7b221c457e2e730bfba2b104e1d07144`
- **Rama de CP04:** `feature/distribution-family-framework-cp04-wave1-fitting`
- **Baseline post-merge de CP04:** `main@2b6e1263b8489592030b0838cd3851f193fbfd7f`
- **Rama de cierre post-merge de CP04:** `docs/distribution-family-framework-cp04-post-merge`
- **Owner de decisión:** `decision-owner`
- **Arquitectura:** `statistical-software-architecture`
- **Implementación:** `implementation-engineering`
- **QA adversarial:** `adversarial-statistical-qa`

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
| CP03 — Discrete distribution core | `COMPLETE` | Implementación `4f7fa09…` y PR head `a1e4d61…` con auditorías `ADVERSARIAL_PASS`; integración mediante PR #10 en `main@28b57a2…` |
| CP04 — Wave 1 fitting | `COMPLETE` | Gates CP04-A–D aceptados; implementación `6e92ef20…`, `ADVERSARIAL_PASS`, PR #12 y cierre de gobernanza en EV-012 |
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
- Evidencia de implementación y auditoría: [`../evidence/distribution-family-framework-cp03-evidence.md`](../evidence/distribution-family-framework-cp03-evidence.md)
- Rama de implementación integrada: `BR-021`
- Rama de cierre post-merge: `BR-022`

## Artefactos de CP04

- Contrato ejecutable congelado: [`distribution-family-framework-cp04-contract.md`](distribution-family-framework-cp04-contract.md)
- Evidencia de reconnaissance y baseline: [`../evidence/distribution-family-framework-cp04-baseline.md`](../evidence/distribution-family-framework-cp04-baseline.md)
- Evidencia de implementación, auditoría e integración: [EV-012](../evidence/distribution-family-framework-cp04-evidence.md)
- Rama de arquitectura/implementación integrada: `BR-023`
- Rama de cierre post-merge: `BR-024`

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
`in_progress`; CP02 y CP03 están `COMPLETE`, la implementación CP03 auditada
`4f7fa09…` y su cierre de gobernanza están integrados mediante PR #10 y
PR #11. CP04 está `COMPLETE`, con implementación e integración completas y
gobernanza cerrada según EV-012; CP05–CP08 permanecen `NOT_STARTED`.

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
El stage general permanece `IN_PROGRESS`; CP03 está `COMPLETE` con
arquitectura `FROZEN`, implementación auditada en `4f7fa09…`,
`ADVERSARIAL_PASS` e integración mediante PR #10; en ese punto CP04–CP08
seguían `NOT_STARTED`.
CP02
no acepta ni implementa fitting, `FitResult`,
`FittedDistribution`, estimación o incertidumbre de parámetros, GOF,
calibración, `MethodSelector`, routing, familias discretas ni CP03.

## Apertura y arquitectura congelada de CP03

CP03 se abre desde `main@02a65c80c5da10295d6eeef42e691772d0686ca2` en
`BR-021`. Reconnaissance queda `COMPLETE`; `DEC-012` congela la arquitectura
del core discreto, `SupportKind.DISCRETE`, el modelo de objetos discreto y la
parametrización canónica `NegativeBinomialFamily().bind(r=..., p=...)`.

La materialización inicial de arquitectura fue exclusivamente de gobernanza y
no implementó producción ni tests. El trabajo posterior autorizado produjo el
candidato exacto `4f7fa09bc7ab501d21f6d27dada30ade23397588` sin modificar las
APIs discretas legacy ni autorizar fitting, GOF, selector/routing, otras
familias discretas o CP04. En ese hito CP04–CP08 permanecían `NOT_STARTED`.

## Implementación y auditoría adversarial pre-merge de CP03

La implementación exacta `4f7fa09bc7ab501d21f6d27dada30ade23397588`
materializa el contrato discreto congelado en `DEC-012`. Antigravity completó
su auditoría de implementación y emitió `ADVERSARIAL_PASS`: 0 `BLOCKER`,
0 `MAJOR`, 0 `MINOR` y 1 `INFO` (`INFO-001`, limitación acotada del backend
SciPy correctamente traducida).

Las superficies registradas son 360 tests de regresión congelada, 215 tests
CP03 y 575 tests de distribución. `INFO-001` documenta únicamente una
limitación acotada de ejecutabilidad del backend SciPy para sampling extremo:
los fallos numéricos o de rango posteriores a la validación pública se traducen
correctamente a `FloatingPointError`, sin añadir thresholds matemáticos para
`r` o `p`.

`EV-010` preserva por separado ese resultado y la auditoría final del PR head
exacto `a1e4d61f0026f8407506d039788bb2df2eafa680`. La auditoría final se realizó
desde un clon independiente fresco y emitió `ADVERSARIAL_PASS`: 0 `BLOCKER`,
0 `MAJOR`, 0 `MINOR` y 2 `INFO`. `INFO-001` conserva la limitación acotada del
backend; `INFO-002` registra únicamente los dos fallos heredados y fuera de
alcance de Knowledge Base. El registro pasó, la superficie de distribución
registró 575 passed y el diferencial de Knowledge Base fue `NO_NEW_FAILURES`
(base y head: 7 passed, 2 failed). GitHub reportó `NO_CHECKS_REPORTED`, cero
workflow runs, cero reviews registradas y cero threads sin resolver.

El PR head final fue integrado por PR #10 mediante
el merge commit `28b57a2eaab0706c5b2e2dcdf6a03e5a30a0b649`, cuyos parents exactos son
`02a65c80c5da10295d6eeef42e691772d0686ca2` y
`a1e4d61f0026f8407506d039788bb2df2eafa680`. El tree del merge y el tree del
head integrado son idénticos:
`13002c3ca716a3b2a2aa8914bce078afd623d9ff`.

La integración y la gobernanza de CP03 están cerradas; CP03 queda `COMPLETE`.
PR #11 integró el cierre post-merge mediante
`b3f35d4d7b221c457e2e730bfba2b104e1d07144`, con parents
`28b57a2eaab0706c5b2e2dcdf6a03e5a30a0b649` y
`7e38c1c62f69771282398ffd3fac118f866c5d69`. Los trees del merge y del head
integrado coinciden en `2e363bd357a5ec59c17690bd9e0e3bbb26061d33`.
En ese hito el stage permanecía `IN_PROGRESS` y CP04 estaba abierto con
arquitectura `FROZEN`. Su cierre posterior se registra en EV-012.

## Apertura y arquitectura congelada de CP04

CP04 se abre en `BR-023` desde el snapshot exacto
`main@b3f35d4d7b221c457e2e730bfba2b104e1d07144`. `DEC-013` congela el fitting
Wave 1: fitted/result core inmutable, validación de datos, MLE fixed-`loc=0`
para Gamma y Exponential, y MLE nativo Negative Binomial para `r>0` real.

Los gates internos son CP04-A (object/input contract), CP04-B (MLE continuo),
CP04-C (MLE NB generalizado) y CP04-D (auditoría adversarial independiente).
CP04 no quedaba completo hasta que los cuatro pasaran. Aquella materialización fue
solo arquitectura/gobernanza: no implementa producción ni tests, no autoriza
push, PR o merge, y no inicia CP05–CP08.

## Cierre post-merge de CP04 — 2026-09-10

La arquitectura aceptada `9d7a9ea5dcf3f0d962e72a4a0bd2211357e3af52`
conserva el contrato matemático congelado de DEC-013. La implementación
certificada `6e92ef20aca375878964321596ba525539433f79` completa CP04-A,
CP04-B y CP04-C; CP04-D queda aceptado tras la remediación y el
`ADVERSARIAL_PASS` final. EV-011 permanece como baseline histórico inmutable.

PR #12 integró el candidato en
`main@2b6e1263b8489592030b0838cd3851f193fbfd7f`. EV-012 registra las
identidades exactas, auditorías, limitaciones de backend y validación.
CP04_IMPLEMENTATION=`COMPLETE`, CP04_INTEGRATION=`COMPLETE`,
CP04_GOVERNANCE=`CLOSED` y CP04_OVERALL=`COMPLETE`.
El stage permanece `IN_PROGRESS`; CP05–CP08 siguen `NOT_STARTED`.

## Siguiente acción

Arquitectura debe revisar el candidato local de cierre de gobernanza BR-024
y su bundle. Esta operación no publica ni integra ese candidato documental
y no inicia CP05–CP08.
