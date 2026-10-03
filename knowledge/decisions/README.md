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
- `DEC-018`: propone la certificación R10-A-R2 de la raíz Negative Binomial
  mediante bracket final, adyacencia float64 y residual derivado del bracket;
  no modifica ni reemplaza DEC-016 y no constituye ejecución GPU o calibración.
- `DEC-019`: prerregistra el replay GPU focalizado R10-A sobre la unión R9
  persistida C/F/S/H y 12 reconstrucciones MC completas; no ejecuta CUDA ni
  afirma equivalencia, calibración o cierre completo de fallos históricos.

- `DEC-020`: propone separar los puntos de evaluación de valores de distribución
  del soporte GOF certificado; preserva DEC-014/016/018/019 y no afirma
  equivalencia GPU ni calibración.

- `DEC-021`: propone el conjunto obligatorio de cantidades por familia y su
  comprobación fail-closed; registra la remediación de AGY-QA-R3-01 sin borrar
  el FAIL independiente del primer candidato DEC-020 ni autorizar replay.

- [`DEC-022`](cp05-c2c-r10a-targeted-gpu-replay-r2-preregistration.md):
  propone un replay R2 del mismo conjunto histórico 134 + 12, vinculado al
  candidato científico R3-R1 exacto y a los contratos DEC-020/021; preserva
  DEC-019 como ejecución consumida y fallida, sin reutilizar su autorización
  ni ejecutar GPU o implementar el harness R2.

- [`DEC-023`](cp05-c2c-r10a-targeted-gpu-replay-r3-preregistration.md):
  propone R3 con el mismo contrato científico y workloads ordenados de R2,
  preserva la única ejecución R2 fallida y consumida y exige preflight fuerte
  CUDA 13/NVRTC antes de crear output científico. La evidencia Owner/Quantum
  queda atribuida en EV-019; no implementa harness ni autoriza ejecución.

- [`DEC-024`](cp05-c2c-r10a-mc-reference-exact-tie-adjudication.md):
  `accepted` por el Project Owner tras arquitectura y QA independiente PASS;
  adjudicación prospectiva de equivalencia MC sólo para crossings
  certificados de empate exacto del CPU reference; conserva comparador `>=`,
  tolerancias, resultados raw y decisiones de rechazo idénticas. EV-020 atribuye
  la evidencia Owner/Quantum con estado `validated_with_limits`; R3 permanece
  FAILED y consumido. No autoriza implementación ni ejecución R4.

- [`DEC-025`](cp05-c2c-r10a-targeted-gpu-replay-r4-preregistration.md):
  `accepted` tras arquitectura/Antigravity PASS y aceptación del Owner;
  prerregistra R4 con las mismas identidades y ciencia de R3/R2,
  adjudicación MC DEC-024 aceptada y accounting firmado, sin cambiar resultados
  raw ni certificar near-ties. Hereda preflight fuerte y exige ejecución fresca
  con autorización separada; R3 permanece FAILED. Con B_EQ=15, p_min=0.0625
  impide evidencia de frontera de rechazo a alpha=0.05. No crea EV-021;
  implementación del harness autorizada separadamente, ejecución GPU no autorizada.

- [`DEC-026`](research-acceleration-boundary.md): establece NumPy/SciPy como
  superficie productiva y referencia científica canónica, y restringe
  CUDA/CuPy a instrumentación de investigación. Redefine prospectivamente cómo
  se interpreta la suficiencia de C2C mediante equivalencia y cross-validation
  CPU proporcionales al claim; no exige equivalencia universal para usar CUDA
  en research. No altera retrospectivamente `DEC-014`..`DEC-025`, R10-A ni sus
  estados y evidencia históricos.

- [`DEC-027`](cp05-c2c-r11-boundary-sensitive-decision-equivalence-preregistration.md):
  `accepted`; prerregistra prospectivamente R11 como un experimento limitado de
  equivalencia de decisión CPU/CUDA sensible a la frontera `alpha=.05`, bajo
  `DEC-026`, sobre 12 outers MC históricos congelados y `B=199`. Define la
  futura construcción CPU-reference, fixtures deterministas y accounting de
  empates exactos `DEC-024`; DEC-027 permanece `accepted`. El intento R11 #1
  inició ejecución, consumió su autorización y terminó
  `INCONCLUSIVE_INFRASTRUCTURE`, con 0 outers completados y ninguna conclusión
  de equivalencia científica CPU/CUDA. R11 no está científicamente validado ni
  completo y no afirma full-1152, backend productivo, CP05-D ni holdout.

- [`DEC-028`](cp05-c2c-r11-r4-runtime-oracle-identity-correction.md):
  `accepted`; acepta formalmente la corrección documental de provenance del
  oráculo runtime R4 de DEC-027: vincula el archive SHA canónico de EV-021 al
  PASS_EVIDENCE y conserva el digest anterior para el crossings report. No cambia
  el diseño científico R11 ni autoriza builder, construcción del workload,
  harness, GPU, full-1152, CP05-D o holdout.

- [`DEC-029`](cp05-c2c-r11-cuda-execution-readiness-oracle.md):
  `proposed`; preserva R11 attempt #1 como
  `INCONCLUSIVE_INFRASTRUCTURE`, con autorización consumida y sin
  conclusión de equivalencia; prohíbe rerun, resume y reutilización de ese
  intento. Restablece y generaliza el preflight fuerte NVRTC de DEC-023,
  exige readiness sintético determinista antes de futura evaluación científica
  R11 y conserva la ciencia DEC-027 sin cambios. La evidencia Owner/Quantum se
  atribuye en EV-022; no autoriza attempt #2.

Use [`DECISION_RECORD_TEMPLATE.md`](DECISION_RECORD_TEMPLATE.md).
