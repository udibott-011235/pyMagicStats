# Gobernanza del conocimiento y del desarrollo

## 1. Fuente de verdad

GitHub es la fuente de verdad para ramas, commits y artefactos versionados.
`knowledge/registry.json` es el índice canónico del conocimiento: los documentos
explican; el registro identifica qué existe, su estado y dependencias. El código
y los tests describen comportamiento implementado, pero no convierten por sí
solos una política estadística en válida.

Orden mínimo de evidencia:

| Afirmación | Evidencia mínima |
|---|---|
| Teórica | fuente primaria o decisión explícita con alcance y límites |
| Arquitectónica | contrato, decisión, mapa de impacto y criterios de aceptación |
| De implementación | rama, SHA, código, tests y diff asociados |
| De calibración | runner, comando, semilla, entorno, outputs, denominadores y límites |
| De dataset | procedencia, licencia, esquema, clasificación y hash/puntero |
| De estado | repositorio, rama, SHA y fecha de observación |

Cuando dos fuentes discrepan, no se borra la discrepancia. Se abre un registro
`under_review`, se conservan ambas posiciones y el Project Owner resuelve el
alcance o declara el límite con apoyo del arquitecto.

## 2. Roles y autoridad

- **Project Owner — Ehud Bottaro:** define prioridad, dominio, riesgo aceptable y
  alcance; aprueba diseño, publicación, PR y merge mediante autorizaciones
  separadas; toma la decisión final.
- **Arquitectura matemática y de software — ChatGPT:** protege la teoría,
  traduce necesidades en contratos verificables y evalúa hallazgos. No
  implementa producción ni autoriza su propio merge.
- **Ingeniería de implementación — Cortex:** implementa el contrato aprobado en
  una rama y SHA autorizados, añade pruebas y entrega evidencia. No redefine la
  teoría ni autocertifica validez.
- **QA adversarial estadístico y de software — Antigravity:** intenta refutar
  diseño, implementación y calibración. La primera auditoría es de solo lectura
  y no puede aprobar su propio fix.

La autoridad pertenece al rol, no al modelo. Un mismo agente no puede ser autor
y único revisor del mismo candidato.

Los system prompts vigentes se mantienen exclusivamente en
[`SYSTEM_PROMPTS.md`](SYSTEM_PROMPTS.md).

## 3. Fronteras de autorización

Las autorizaciones son independientes:

| Autorización | No autoriza automáticamente |
|---|---|
| Diseñar | implementar |
| Implementar | publicar la rama |
| Crear commit | hacer push |
| Publicar rama | abrir PR |
| Abrir PR | mergear |
| Recibir PASS | mergear |

Sólo el Project Owner puede autorizar PR y merge. La ejecución puede delegarse,
pero debe existir una autorización expresa para esa acción y candidato.

## 4. Protección de `main`

Todos los agentes pueden leer, hacer fetch y comparar `main`. Ningún agente
puede editar, commitear, hacer push, rebase, merge o mover directamente su
referencia. Está prohibido usar bypass administrativo para evadir esta regla.

Todo cambio parte de una rama y SHA base exactos. Un nombre de rama es
insuficiente para identificar un candidato. Cada nuevo commit requiere una
nueva auditoría antes de conservar un veredicto favorable.

## 5. Estados

| Estado | Significado |
|---|---|
| `proposed` | afirmación o artefacto nuevo, aún sin revisión |
| `under_review` | existe revisión o controversia activa |
| `accepted` | decisión vigente dentro del alcance declarado |
| `validated_with_limits` | evidencia reproducida con límites explícitos |
| `open` | deuda, gap o procedencia pendiente |
| `blocked` | no puede avanzar sin evidencia o autoridad adicional |
| `rejected` | no se acepta; se conserva la razón |
| `superseded` | reemplazado por otro ID sin reescribir historia |
| `archived` | preservado, fuera del flujo activo |

`accepted` no significa verdad universal. `validated_with_limits` nunca debe
acortarse a “validado” sin repetir sus límites.

## 6. Veredictos y severidades

- `PASS`: no quedan defectos conocidos que invaliden el alcance.
- `CONDITIONAL PASS`: requiere límites explícitos y decisión del Project Owner.
- `FAIL / DO NOT MERGE`: existe un defecto que compromete validez o software.
- `BLOCKED`: faltan evidencia, acceso, baseline, entorno o autoridad.

Severidades: `BLOCKER`, `MAJOR`, `MINOR` y `NOTE`. Un `BLOCKER` o `MAJOR`
impide merge hasta corrección y reauditación independiente.

## 7. Reglas de cambio

1. Leer registro, principios teóricos, evidencia y protocolo antes de actuar.
2. Actualizar el registro en el mismo PR que el conocimiento modificado.
3. Cambiar un contrato aceptado mediante una decisión que indique
   `supersedes`; nunca editar silenciosamente la historia.
4. Mantener sincronizados diseño, código, tests, experimentos y documentación.
5. Registrar calibraciones con runner, comando, entorno, seed, escenarios,
   denominadores, outputs y limitaciones.
6. No publicar papers con copyright sin licencia compatible.
7. No publicar PII, secretos ni datasets de licencia desconocida.
8. No mezclar alcance incidental en una rama candidata.

## 8. Definition of done

Un cambio termina cuando el contrato aprobado, el registro, la evidencia, el
código, los tests y la documentación pública cuentan la misma historia; el
validador pasa; Antigravity dejó una conclusión trazable sobre el SHA exacto;
ChatGPT interpretó los hallazgos; y el Project Owner autorizó la transición
correspondiente. Un PASS no constituye por sí solo autorización de merge.
