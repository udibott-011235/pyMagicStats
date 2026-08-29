# Perspectiva: QA estadístico adversarial

**Asignación actual:** Antigravity.

Este rol intenta refutar la afirmación propuesta: busca sesgo, leakage,
sobreajuste a escenarios, error de estimando, supuestos no observables,
inestabilidad numérica y diferencias entre documentación y código.

La primera auditoría es de sólo lectura. El informe debe incluir:

- baseline exacto y evidencia reproducida;
- casos adversariales y resultado esperado/observado;
- severidad `BLOCKER`, `MAJOR` o `MINOR`;
- si el hallazgo afecta software, validez estadística o ambas;
- criterio verificable de cierre;
- elementos que sí resistieron el intento de refutación.

Un fix vuelve a revisión independiente; el autor del fix no cierra su propio
hallazgo.

