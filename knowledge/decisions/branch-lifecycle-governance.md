# DEC-006 — Branch lifecycle governed by Product Owner and Architecture

- Estado: `accepted`
- Fecha: `2026-08-30`
- Owner: `decision-owner`
- Revisores: `statistical-software-architecture`, `implementation-engineering`
- Evidencia: `EV-003`, `EV-004`
- Supersedes: ninguno

## Contexto

Git conserva la genealogía técnica de una rama, pero su mera existencia no
expresa si el trabajo sigue activo, fue integrado, quedó archivado o fue
superseded. El proyecto necesita preservar ambos planos sin borrar historia ni
convertir una inferencia de un agente en una decisión operativa.

## Decisión

1. GitHub determina qué refs y SHAs existen.
2. `knowledge/registry.json` determina el estado operativo del work item.
3. Una rama existente no implica trabajo activo.
4. Sólo Product Owner y Arquitectura pueden decidir los estados `accepted`,
   `superseded`, `archived`, `merge_candidate` y la integración.
5. Los agentes pueden recopilar evidencia y proponer, pero no tomar la decisión
   final de lifecycle.
6. Una rama integrada puede conservarse como `archived` hasta una futura
   decisión explícita de borrado.
7. Nunca se elimina una rama para “limpiar” antes de registrar su genealogía.
8. Todo nuevo work item debe registrar rama, base SHA, purpose, target y owner
   role.
9. Todo cierre debe registrar final SHA, evidencia, integración o supersesión y
   siguiente acción.
10. Un SHA auditado no debe cambiarse para actualizarlo con `main` sin perder el
    veredicto de auditoría asociado al SHA original.

## Supersesión

La dirección canónica se expresa únicamente mediante `supersedes`: el registro
nuevo contiene el ID anterior, y el anterior conserva `status = superseded`.
No se introduce un campo inverso `superseded_by`.

Para Gate 2, BR-011 supersede BR-010 y BR-012 supersede BR-011. Esto preserva la
cadena placeholder → candidato materializado → candidato adversarial vigente.

## Autoridad y siguiente acción

Los 16 estados iniciales fueron decididos por Product Owner y Arquitectura y se
materializan en `registry.json`. La ingeniería puede verificar consistencia y
entregar el diff, pero no alterar estas decisiones, publicar la rama o integrar
un candidato sin una autorización posterior independiente.

## Condición de revisión

La decisión debe revisarse si cambia la autoridad de roles, el modelo canónico
de registro, el repositorio remoto o la política de preservación de SHAs
auditados. Un cambio ordinario de HEAD requiere una nueva observación del
registro, no una reescritura silenciosa de esta decisión.
