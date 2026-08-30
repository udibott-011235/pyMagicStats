# Espacios por perspectiva de rol

Estos espacios contienen guías y notas de trabajo, no teorías alternativas. Cada nota enlaza IDs de `registry.json`, separa hecho de interpretación y termina con una transferencia al siguiente rol.

- [`statistical-software-architecture/`](statistical-software-architecture/README.md) — ChatGPT, arquitectura matemática y de software / arbitraje técnico.
- [`implementation-engineering/`](implementation-engineering/README.md) — Codex, ingeniería principal de implementación.
- [`adversarial-statistical-qa/`](adversarial-statistical-qa/README.md) — Antigravity, QA, Repo Ops, validación rutinaria y testing adversarial.

## Routing operativo

- Decidir qué debe hacer el sistema: ChatGPT.
- Implementar un contrato ya decidido: Codex.
- Comprobar, reproducir, operar Git o intentar romper un candidato: Antigravity.
- Cambiar prioridad, riesgo aceptado, PR o merge: Project Owner.

Los hallazgos mecánicos pueden circular `Antigravity -> Codex -> Antigravity` sin escalar a ChatGPT. Los cambios de contrato escalan a arquitectura.

La política completa está en [`../decisions/agent-orchestration-policy.md`](../decisions/agent-orchestration-policy.md). El Project Owner decide sobre los registros compartidos; sus decisiones quedan en `decisions/`, no en un diario privado. Los prompts normativos viven sólo en [`../SYSTEM_PROMPTS.md`](../SYSTEM_PROMPTS.md).