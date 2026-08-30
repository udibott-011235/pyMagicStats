# Versioning y lifecycle de ramas

Este espacio explica cómo pyMagicStats separa la historia Git del estado
operativo de un work item.

## Fuentes de verdad

- GitHub determina refs, SHAs, parents, ancestry y pull requests existentes.
- [`registry.json`](../registry.json) determina el estado operativo acordado.
- [`branches.md`](branches.md) es una proyección humana del registro, no un
  segundo registro canónico.

Una rama puede seguir existiendo después de integrarse o supersederse. Por
tanto, existencia no equivale a actividad y un nombre de rama no sustituye al
SHA observado.

## Relación con main

| Relación | Significado |
|---|---|
| `canonical` | registro único de la rama canónica `main` |
| `same_head` | el HEAD observado coincide exactamente con `main` |
| `fully_contained` | todos los commits de la rama están contenidos en `main` |
| `contains_main` | la rama desciende de `main` y añade commits |
| `diverged` | rama y `main` contienen commits posteriores distintos al merge-base |

`ahead_of_main`, `behind_main`, `merge_base` y `unique_commits` fijan la
observación que sustenta esa relación.

## Estado de integración

| Estado | Significado |
|---|---|
| `not_applicable` | la integración no corresponde al rol de la rama |
| `pending` | existe trabajo no integrado sin decisión de candidatura |
| `merge_candidate` | Product Owner y Arquitectura lo declararon candidato |
| `merged` | el contenido ya está integrado en la rama canónica |
| `not_planned` | no se planea integrar ese registro como candidato independiente |

El `status` general conserva el lifecycle (`under_review`, `archived`,
`superseded`, etc.); `integration_state` sólo describe integración. Una rama
`archived` puede seguir existiendo en GitHub y una `merge_candidate` todavía no
está merged.

## Supersesión

El registro más nuevo incluye el ID anterior en `supersedes`. El registro
anterior conserva `status = superseded`. No se mantiene un campo inverso porque
duplicaría la relación y permitiría inconsistencias.

## Autoridad

Sólo Product Owner y Arquitectura deciden aceptación, supersesión, archivo,
candidatura e integración. Implementación y QA pueden observar SHAs, recopilar
evidencia y proponer cambios; no convierten una observación en decisión final.

## Actualización de un registro de rama

1. Obtener refs desde el remoto autorizado en un workspace independiente.
2. Registrar fecha, HEAD, merge-base, ahead/behind y commits únicos.
3. Conservar parent y PR cuando sean verificables.
4. Enlazar evidencia reproducible.
5. Solicitar a Product Owner y Arquitectura la decisión de lifecycle.
6. Actualizar `registry.json` y después regenerar manualmente `branches.md`.
7. Ejecutar `python knowledge/tools/validate_registry.py` y los tests de KB.

Un SHA con veredicto de auditoría no se mueve o rebasea para actualizarlo con
`main`: el nuevo SHA requiere una nueva evidencia.
