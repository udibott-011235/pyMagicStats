# Protocolo de colaboración entre agentes

## Núcleo compartido, perspectivas separadas

Cada sesión empieza leyendo `GOVERNANCE.md`, `SYSTEM_PROMPTS.md`,
`registry.json`, los principios teóricos y el espacio del rol. Las notas de rol
pueden desafiar el núcleo compartido, pero no reemplazarlo. Una objeción se
convierte en cambio canónico sólo después de registrarse, recibir evidencia y
pasar revisión cruzada.

## Contrato por rol

| Rol | Pregunta principal | Entregable | No puede hacer solo |
|---|---|---|---|
| Arquitectura matemática y de software — ChatGPT | ¿Qué contrato produce una inferencia defendible y una API auditable? | diseño matemático, arquitectura, política, calibración y criterios de aceptación | implementar producción, aceptar su propia evidencia o autorizar merge |
| Ingeniería de implementación — Cortex | ¿Cómo implementar exactamente el contrato aprobado? | código, tests, documentación técnica, commit y handoff | redefinir teoría, cambiar alcance o autocertificar validez |
| QA adversarial — Antigravity | ¿Dónde falla, sobreajusta o induce una inferencia falsa? | informe con severidad, reproducción, veredicto y criterio de cierre | modificar en primera auditoría o aprobar su propio fix |
| Project Owner — Ehud Bottaro | ¿El estimando, riesgo y alcance resuelven el problema real? | prioridad, decisiones y autorizaciones separadas de diseño/publicación/PR/merge | convertir evidencia insuficiente en afirmación de validez |

## Ciclo obligatorio

1. El Project Owner define objetivo y alcance.
2. ChatGPT diseña el contrato y los criterios verificables.
3. El Project Owner autoriza implementación.
4. Cortex verifica baseline e implementa en rama aislada.
5. Cortex entrega SHA candidato y evidencia; no avanza a la siguiente acción Git
   sin autorización.
6. Antigravity audita el SHA remoto exacto en modo de solo lectura inicial.
7. ChatGPT interpreta el informe y recomienda aceptar, limitar o refactorizar.
8. Todo fix vuelve a Cortex y produce un nuevo SHA.
9. Todo nuevo SHA vuelve a Antigravity.
10. Sólo el Project Owner autoriza PR y merge.

## Formato de transferencia

```yaml
work_item: <ID o issue>
role: <rol desempeñado>
objective: <objetivo autorizado>
repository: udibott-011235/pyMagicStats
branch: <rama exacta>
base_sha: <SHA base>
candidate_sha: <SHA candidato o null>
claim: <qué se afirma o cuestiona>
scope:
  included: <rutas y decisiones permitidas>
  excluded: <fuera de alcance>
actions_executed: <comandos y operaciones>
files_changed: <lista o ninguno>
evidence: <IDs, rutas, comandos, seeds y outputs>
result: <hechos observados>
interpretation: <lectura propia del rol>
limitations: <qué no demuestra>
open_risks: <BLOCKER|MAJOR|MINOR|NOTE y razón>
next_role: <quién debe actuar>
acceptance_criteria: <condición verificable>
git_actions_not_performed: <push|PR|merge|rebase|main modification>
```

Hecho observado, interpretación y decisión deben aparecer separados.

## Reglas contra la deriva

- No usar memoria de chat como fuente canónica: convertir decisiones en
  registros versionados.
- No copiar una conclusión entre ramas sin anotar commit de origen y alcance.
- No resolver diferencias de lenguaje renombrando el problema: definir
  estimando, población, diseño, unidad independiente y método.
- No confundir “el test no rechazó” con “el supuesto fue demostrado”.
- No reutilizar una calibración fuera de su diseño y estimando.
- No cerrar un hallazgo porque los tests pasen si la cuestión estadística sigue
  abierta.
- No transferir un PASS a un nuevo SHA.
- No interpretar permiso para una acción Git como permiso para la siguiente.
- No continuar si baseline, rama, worktree, alcance o autorización no coinciden.

## Condiciones de detención compartidas

El agente se detiene y solicita decisión del Project Owner cuando falte una
elección que cambie estimando, política, riesgo, alcance, historia Git o
autoridad. Si falta información para concluir una auditoría, el veredicto es
`BLOCKED`, no una suposición favorable.
