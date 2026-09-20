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
- `DEC-006`: gobernanza del lifecycle de ramas por Product Owner y Arquitectura.
- `DEC-009`: apertura y checkpoints de `STAGE-DIST-FAMILIES-001`.
- `DEC-010`: arquitectura y contratos del Distribution Family Framework.
- `DEC-011`: contrato ejecutable del core continuo para CP02.
- `DEC-012`: contrato ejecutable congelado del core discreto para CP03.
- `DEC-013`: contrato ejecutable congelado de fitting Wave 1 para CP04.
- `DEC-014`: contrato y prerregistración GOF para familias Wave 1 ajustadas;
  CP05-A y CP05-B están completos. CP05-B certifica sólo corrección de software,
  no calibración, potencia, selección de método ni idoneidad de producción. El
  commitment público CP05-D está depositado; CP05-C está autorizado para
  comenzar, pero su ejecución permanece `NOT_STARTED` y CP05-D no ha comenzado.
- `DEC-015`: separa validación de implementación, calibración estadística,
  selección del motor experimental y frontera de ejecución de agentes; CP05-C2A
  es un prototipo CUDA/RAPIDS aislado, sin equivalencia ni calibración.
- `DEC-016`: prerregistra el gate CPU↔CUDA CP05-C2B sobre datos fijos y sus
  tolerancias; no constituye ejecución GPU, calibración ni benchmark.

Use [`DECISION_RECORD_TEMPLATE.md`](DECISION_RECORD_TEMPLATE.md).
