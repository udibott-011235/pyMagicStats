# Protocolo de colaboración entre agentes

## Núcleo compartido, perspectivas separadas

Cada sesión empieza leyendo la ruta obligatoria de la portada. Las notas de rol
pueden desafiar el núcleo compartido, pero no lo reemplazan. Una objeción se
convierte en cambio canónico sólo después de registrarse, recibir evidencia y
pasar por revisión cruzada.

## Contrato por rol

| Rol | Pregunta principal | Entregable | No puede hacer solo |
|---|---|---|---|
| Arquitectura e implementación (Codex/GPT) | ¿Cómo expresar el contrato sin ambigüedad en API/código/tests? | diseño, implementación, pruebas y mapa de impacto | declarar válida su propia calibración |
| QA estadístico adversarial (Antigravity) | ¿En qué caso falla, sobreajusta o induce una inferencia falsa? | informe con severidad, reproducción y criterio de cierre | modificar durante la primera auditoría ni aprobar su propio fix |
| Investigación y reproducción (Cortex) | ¿Qué respalda la teoría y el resultado se reproduce/generaliza? | ficha de fuentes, réplica independiente y límites | cambiar política o umbrales por inferencia informal |
| Owner estadístico/producto (Ehud) | ¿El estimando, riesgo y alcance resuelven el problema real? | decisión final, prioridad y autorización de merge | omitir evidencia cuando la decisión afirma validez |

## Formato de transferencia

Toda transferencia entre agentes debe incluir:

```yaml
work_item: KB-<id o issue>
role: <rol desempeñado>
baseline: <rama y commit exactos>
claim: <qué se afirma o cuestiona>
evidence: <IDs y rutas>
result: <hecho observado>
interpretation: <lectura del rol>
limitations: <qué no demuestra>
open_risks: <BLOCKER|MAJOR|MINOR y razón>
next_role: <quién debe actuar>
acceptance_criteria: <condición verificable>
```

## Reglas contra la deriva

- No usar memoria de chat como fuente canónica: convertirla en un registro.
- No copiar una conclusión entre ramas sin anotar el commit de origen.
- No resolver diferencias de lenguaje renombrando el problema; definir
  estimando, población, diseño y método.
- No confundir “el test no rechazó” con “el supuesto fue demostrado”.
- No reutilizar una calibración fuera de su diseño y estimando sin nueva
  evidencia.
- No cerrar un hallazgo porque los tests pasen si el criterio estadístico sigue
  sin respuesta.

