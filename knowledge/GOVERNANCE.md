# Gobernanza del conocimiento

## 1. Fuente de verdad

`knowledge/registry.json` es el índice canónico. Los documentos explican; el
registro identifica qué existe, su estado y sus dependencias. El código y los
tests describen comportamiento implementado, pero no convierten por sí solos
una política estadística en válida.

Orden de evidencia por tipo de afirmación:

| Afirmación | Evidencia mínima |
|---|---|
| Teórica | fuente primaria o decisión explícita con límites |
| De implementación | commit, código y tests asociados |
| De calibración | runner, comando, semilla, entorno, outputs y limitaciones |
| De dataset | procedencia, licencia, esquema, clasificación y hash/puntero |
| De estado | rama, commit y fecha de observación |

Cuando dos fuentes discrepan, no se borra la discrepancia. Se abre un registro
`under_review`, se conservan ambas posiciones y el owner estadístico resuelve o
declara el límite.

## 2. Roles y autoridad

- **Owner estadístico y de producto — Ehud Bottaro:** define el estimando, el
  dominio de uso y la decisión final de merge o aceptación.
- **Arquitectura e implementación — Codex/GPT:** traduce contratos a diseño,
  código, tests y documentación. No autocertifica validez estadística.
- **QA estadístico adversarial — Antigravity:** intenta refutar supuestos,
  calibraciones y criterios de merge. En la primera pasada audita sin corregir.
- **Investigación y reproducción — Cortex:** rastrea fuentes, reproduce
  resultados y evalúa generalización. No cambia umbrales sin decisión registrada.

Los nombres son asignaciones actuales; la autoridad pertenece al rol, no al
modelo. Un mismo agente no puede ser autor y único revisor de un registro.

## 3. Estados

| Estado | Significado |
|---|---|
| `proposed` | afirmación o artefacto nuevo, aún sin revisión |
| `under_review` | existe una revisión o controversia activa |
| `accepted` | decisión vigente dentro del alcance declarado |
| `validated_with_limits` | evidencia reproducida, con límites explícitos |
| `open` | deuda, gap o procedencia pendiente |
| `blocked` | no puede avanzar sin evidencia/autoridad adicional |
| `rejected` | no se acepta; se conserva la razón |
| `superseded` | reemplazado por otro ID, sin reescribir historia |
| `archived` | preservado, fuera del flujo activo |

`accepted` no significa verdad universal; significa contrato vigente en el
alcance indicado. `validated_with_limits` nunca debe acortarse a “validado” sin
repetir sus límites.

## 4. Reglas de cambio

1. Leer el registro, los principios teóricos, la evidencia relacionada y el
   espacio del rol antes de actuar.
2. Crear o actualizar el registro en el mismo PR que el conocimiento que cambia.
3. Para cambiar un contrato aceptado, crear una decisión que indique
   `supersedes`; no editar silenciosamente la historia.
4. Resultados de calibración requieren artefactos reproducibles. Los outputs
   grandes pueden quedar fuera de Git sólo con comando, semilla, checksum o
   procedimiento de reconstrucción.
5. Un hallazgo `BLOCKER` o `MAJOR` impide aceptar/mergear hasta corrección y
   reauditación independiente.
6. Los papers se registran por DOI/URL y ficha crítica. No se suben copias con
   copyright sin licencia compatible.
7. Datasets con PII, secretos o licencia desconocida no se publican. Se registra
   el bloqueo y, si corresponde, un dataset sintético reproducible.

## 5. Definition of done

Un cambio de conocimiento termina cuando registro, teoría/decisión, evidencia,
código/tests afectados y documentación pública cuentan la misma historia; el
validador pasa; y el rol revisor dejó una conclusión trazable.

