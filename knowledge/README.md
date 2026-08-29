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

1. [`GOVERNANCE.md`](GOVERNANCE.md): autoridad, fronteras Git, estados y reglas.
2. [`SYSTEM_PROMPTS.md`](SYSTEM_PROMPTS.md): núcleo común e instrucciones de cada
   agente.
3. [`AGENT_PROTOCOL.md`](AGENT_PROTOCOL.md): ciclo, handoffs y detenciones.
4. [`registry.json`](registry.json): catálogo y estado del conocimiento.
5. [`theory/inference-principles.md`](theory/inference-principles.md): teoría
   estadística compartida vigente.
6. El espacio del rol en [`agents/`](agents/README.md).

## Espacios de trabajo

| Espacio | Propósito | Unidad mínima |
|---|---|---|
| [`theory/`](theory/README.md) | Principios, alcance y contratos estadísticos | nota teórica versionada |
| [`papers/`](papers/README.md) | Referencias externas y lectura crítica | ficha de paper |
| [`datasets/`](datasets/README.md) | Procedencia, licencia, esquema y uso permitido | dataset card |
| [`experiments/`](experiments/README.md) | Hipótesis, comando, seeds, entorno y resultados | experiment record |
| [`evidence/`](evidence/README.md) | Mapa de evidencia y limitaciones | evidence record |
| [`decisions/`](decisions/README.md) | Decisiones, alternativas y criterios de revisión | decision record |
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
