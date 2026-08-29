# Base de conocimiento de pyMagicStats

Este directorio es la memoria científica y operativa compartida del proyecto.
No es un archivo de documentos: conecta teoría, decisiones, evidencia,
experimentos, datasets y revisiones mediante un registro legible por personas y
máquinas.

## Objetivo

Todos los agentes trabajan desde un núcleo común y conservan su perspectiva por
rol. Una observación de arquitectura, una objeción estadística y una
reproducción independiente pueden coexistir, pero ninguna crea una “verdad”
paralela. El estado compartido se actualiza sólo mediante registros trazables y
revisión cruzada.

## Ruta de lectura obligatoria

1. [`GOVERNANCE.md`](GOVERNANCE.md): autoridad, estados de evidencia y reglas de
   actualización.
2. [`AGENT_PROTOCOL.md`](AGENT_PROTOCOL.md): roles, límites y formato de
   transferencia.
3. [`registry.json`](registry.json): catálogo canónico y estado de cada
   conocimiento.
4. [`theory/inference-principles.md`](theory/inference-principles.md): teoría
   estadística compartida vigente.
5. El espacio del rol que se esté desempeñando en [`agents/`](agents/README.md).

## Espacios de trabajo

| Espacio | Propósito | Unidad mínima |
|---|---|---|
| [`theory/`](theory/README.md) | Principios, alcance y contratos estadísticos | nota teórica versionada |
| [`papers/`](papers/README.md) | Referencias externas y lectura crítica | ficha de paper |
| [`datasets/`](datasets/README.md) | Procedencia, licencia, esquema y uso permitido | dataset card |
| [`experiments/`](experiments/README.md) | Hipótesis, comando, semillas, entorno y resultados | experiment record |
| [`evidence/`](evidence/README.md) | Mapa de evidencia y limitaciones | evidence record |
| [`decisions/`](decisions/README.md) | Decisiones, alternativas y criterios de revisión | decision record |
| [`agents/`](agents/README.md) | Observaciones por perspectiva de rol | role note enlazada |

## Flujo de actualización

```text
pregunta -> registro propuesto -> evidencia reproducible -> revisión cruzada
         -> decisión del owner -> estado aceptado/limitado/rechazado
         -> código, tests y documentación sincronizados
```

Cada incorporación debe:

- tener un ID único en `registry.json`;
- declarar alcance, estado, rol responsable y revisores;
- enlazar evidencia y limitaciones, no sólo una conclusión;
- separar hechos observados de interpretación y decisión;
- pasar `python knowledge/tools/validate_registry.py` y los tests.

Consulte [`CHANGELOG.md`](CHANGELOG.md) para cambios del sistema de conocimiento.

