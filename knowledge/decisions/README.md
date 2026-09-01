# Decisiones

Las decisiones registran por qué el proyecto adopta un contrato y bajo qué
condiciones debe revisarse. No sustituyen la evidencia; la enlazan y convierten
en una política explícita.

Decisiones iniciales indexadas:

- `DEC-001`: no usar `n >= 30` como interruptor de normalidad.
- `DEC-002`: separar evaluación, política de robustez y selección.
- `DEC-003`: Welch por defecto; Levene permanece diagnóstico.
- `DEC-004`: bootstrap explícito, reproducible y fiel al estimando.
- `DEC-005`: no transferir calibración de una media a ANOVA sin evidencia
  específica.
- `DEC-006`: lifecycle de ramas gobernado por Product Owner y Arquitectura.
- `DEC-007`: `MANUAL UAT CHECKPOINT 1 — CURRENT STATISTICAL CORE`; fija los
  bloqueantes previos al primer UAT manual, el contrato de prueba con DataFrame
  real/sucio, el alcance de uso manual posterior y la deuda futura deliberadamente
  fuera del hito (nuevas distribuciones, transformaciones, no paramétricos,
  regresiones, DOE y decision engine).

Use [`DECISION_RECORD_TEMPLATE.md`](DECISION_RECORD_TEMPLATE.md).

