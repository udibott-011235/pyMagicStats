# Base de conocimiento de pyMagicStats

Este directorio es la memoria científica, arquitectónica y operativa compartida
del proyecto. Conecta teoría, decisiones, evidencia, experimentos, datasets y
revisiones mediante un registro legible por personas y máquinas.

## Objetivo

Todos los agentes trabajan desde un núcleo común y conservan su perspectiva por
rol. Una propuesta de arquitectura, una implementación y una objeción
adversarial pueden coexistir, pero ninguna crea una verdad paralela. El estado
compartido cambia sólo mediante registros trazables y revisión cruzada.

## Ruta de lectura obligatoria

1. [`ROADMAP.md`](ROADMAP.md): vista humana consolidada del trabajo pendiente,
   estado de checkpoints, bloqueantes y orden operativo vigente.
2. [`GOVERNANCE.md`](GOVERNANCE.md): autoridad, fronteras Git, estados y reglas.
3. [`SYSTEM_PROMPTS.md`](SYSTEM_PROMPTS.md): núcleo común e instrucciones de cada
   agente.
4. [`AGENT_PROTOCOL.md`](AGENT_PROTOCOL.md): ciclo, handoffs y detenciones.
5. [`registry.json`](registry.json): catálogo y estado estructurado del conocimiento.
6. [`decisions/manual-uat-checkpoint-1.md`](decisions/manual-uat-checkpoint-1.md):
   checkpoint transversal vigente para cerrar el current statistical core,
   ejecutar Manual UAT 1 y habilitar uso manual limitado antes de cualquier
   orquestador.
7. [`theory/inference-principles.md`](theory/inference-principles.md): teoría
   estadística compartida vigente.
8. El espacio del rol en [`agents/`](agents/README.md).

## Espacios de trabajo

| Espacio | Propósito | Unidad mínima |
|---|---|---|
| [`theory/`](theory/README.md) | Principios, alcance y contratos estadísticos | nota teórica versionada |
| [`papers/`](papers/README.md) | Referencias externas y lectura crítica | ficha de paper |
| [`datasets/`](datasets/README.md) | Procedencia, licencia, esquema y uso permitido | dataset card |
| [`experiments/`](experiments/README.md) | Hipótesis, comando, seeds, entorno y resultados | experiment record |
| [`evidence/`](evidence/README.md) | Mapa de evidencia y limitaciones | evidence record |
| [`decisions/`](decisions/README.md) | Decisiones, alternativas y criterios de revisión | decision record |
| [`versioning/`](versioning/README.md) | Lifecycle de ramas, integración y supersesión | branch record |
| [`agents/`](agents/README.md) | Observaciones por perspectiva de rol | role note enlazada |

## Flujo de actualización

```text
pregunta -> diseño de ChatGPT -> autorización del Project Owner
         -> implementación de Cortex -> auditoría de Antigravity
         -> interpretación de ChatGPT -> decisión del Project Owner
         -> estado aceptado/limitado/rechazado
```

Cada incorporación debe:

- tener un ID único en `registry.json` cuando modifique conocimiento canónico;
- declarar alcance, estado, rol responsable y revisores;
- enlazar evidencia y limitaciones, no sólo una conclusión;
- separar hechos observados de interpretación y decisión;
- fijar rama y SHA para afirmaciones sobre implementación;
- pasar `python knowledge/tools/validate_registry.py` y los tests pertinentes.

Consulte [`CHANGELOG.md`](CHANGELOG.md) para cambios del sistema.

La autoridad de lifecycle de ramas está fijada por
[`DEC-006`](decisions/branch-lifecycle-governance.md). La vista humana de las
ramas está en [`versioning/branches.md`](versioning/branches.md), pero
`registry.json` sigue siendo la fuente canónica estructurada.

El hito operativo transversal vigente está fijado por
[`DEC-007`](decisions/manual-uat-checkpoint-1.md). Su PASS autoriza únicamente
uso manual de los módulos incluidos en su baseline UAT y dentro de límites
documentados; no implica toolbox completa ni valida un decision engine.

Para una lectura personal rápida del estado y de todo lo pendiente, usar
[`ROADMAP.md`](ROADMAP.md) como punto de entrada.