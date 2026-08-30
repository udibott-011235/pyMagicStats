# Protocolo de colaboración entre agentes

## Núcleo compartido, perspectivas separadas

Cada sesión empieza leyendo `GOVERNANCE.md`, `SYSTEM_PROMPTS.md`,
`decisions/agent-orchestration-policy.md`, `registry.json`, los principios
teóricos y el espacio del rol. Las notas de rol pueden desafiar el núcleo
compartido, pero no reemplazarlo. Una objeción se convierte en cambio canónico
sólo después de registrarse, recibir evidencia y pasar revisión cruzada.

## Contrato por rol

| Rol | Pregunta principal | Entregable | No puede hacer solo |
|---|---|---|---|
| Arquitectura matemática y de software / arbitraje — ChatGPT | ¿Qué contrato produce una inferencia defendible y una API auditable? | diseño, política, calibración, Gates, interpretación y criterios de aceptación | autocertificar evidencia, absorber rutinariamente Repo Ops o autorizar merge |
| Ingeniería principal de implementación — Codex | ¿Cómo implementar exactamente el contrato aprobado? | código, tests ligados al cambio, documentación técnica, commit y handoff | redefinir teoría, cambiar alcance o autocertificar validez |
| QA, Repo Ops y Validation Engineering — Antigravity | ¿El candidato es reproducible, limpio y defendible, y dónde falla? | preflight, evidencia Git, suites, regresión, scope audit, adversarial, veredicto y criterio de cierre | aprobar su propio fix, modificar producción en la primera auditoría o cambiar el contrato |
| Project Owner — Ehud Bottaro | ¿El estimando, riesgo y alcance resuelven el problema real? | prioridad, decisiones y autorizaciones separadas de diseño/publicación/PR/merge | convertir evidencia insuficiente en afirmación de validez |

## Regla de routing

- Si la tarea requiere decidir qué debe hacer el sistema: ChatGPT.
- Si requiere implementar algo ya decidido: Codex.
- Si requiere comprobar, reproducir, operar Git o intentar romper un candidato: Antigravity.
- Si cambia prioridad, riesgo aceptado, PR o merge: Project Owner.

Los hallazgos mecánicos o inequívocamente cubiertos por el contrato circulan
`Antigravity -> Codex -> Antigravity`. Sólo se escala a ChatGPT cuando el
hallazgo exige cambiar estimando, teoría, API, método, política, threshold,
fallback, garantía o criterio de aceptación.

## Ciclo estándar

1. El Project Owner define objetivo y alcance.
2. ChatGPT diseña el contrato y criterios verificables.
3. Antigravity realiza el preflight cuando el trabajo lo requiera y certifica baseline, rama y entorno.
4. El Project Owner autoriza implementación.
5. Codex implementa en rama aislada y entrega SHA candidato.
6. Antigravity audita el SHA exacto: reproduce, corre suites, revisa scope/Repo Ops y ejecuta adversarial.
7. Los fixes mecánicos vuelven directamente a Codex y el nuevo SHA vuelve a Antigravity.
8. Los hallazgos arquitectónicos o estadísticos materiales vuelven a ChatGPT.
9. ChatGPT interpreta los hallazgos materiales y recomienda aceptar, limitar o refactorizar.
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
actions_executed: <acciones/comandos esenciales>
files_changed: <lista o ninguno>
evidence: <evidencia mínima reproducible>
result: <hechos observados>
interpretation: <lectura propia del rol>
limitations: <qué no demuestra>
open_risks: <BLOCKER|MAJOR|MINOR|NOTE y razón>
next_role: <quién debe actuar>
acceptance_criteria: <condición verificable>
git_actions_not_performed: <push|PR|merge|rebase|main modification>
```

La evidencia debe ser compacta: acción, resultado, evidencia mínima, conclusión
y SHA. Los logs completos sólo se transfieren cuando sean necesarios para
reproducir o diagnosticar un fallo.

## Reglas contra la deriva

- No usar memoria de chat como fuente canónica: convertir decisiones en registros versionados.
- No copiar una conclusión entre ramas sin anotar commit de origen y alcance.
- No resolver diferencias de lenguaje renombrando el problema: definir estimando, población, diseño, unidad independiente y método.
- No confundir “el test no rechazó” con “el supuesto fue demostrado”.
- No reutilizar una calibración fuera de su diseño y estimando.
- No cerrar un hallazgo porque los tests pasen si la cuestión estadística sigue abierta.
- No transferir un PASS a un nuevo SHA.
- No interpretar permiso para una acción Git como permiso para la siguiente.
- No continuar si baseline, rama, worktree, alcance o autorización no coinciden.
- Evitar triple revisión completa: cada rol profundiza en su especialidad.

## Condiciones de detención compartidas

El agente se detiene y solicita decisión del Project Owner cuando falte una
elección que cambie estimando, política, riesgo, alcance, historia Git o
autoridad. Codex escala a ChatGPT si un fix requiere cambiar el contrato.
Antigravity declara `BLOCKED` cuando falte evidencia necesaria para concluir.
