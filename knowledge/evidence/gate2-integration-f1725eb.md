# EV-005 — Gate 2 integration — f1725eb

- Estado: `validated_with_limits`
- Merge SHA: `f1725ebdfebcb667c053420e4cb4c1e35048f9e0`
- Parents:
  - Parent 1 (base): `e8422a74cef7d3eebc1f807666e9388acd407794`
  - Parent 2 (head): `9a87c5d48dba8b8a172b5386d7318e7f37ec98fe`
- Tree: `238222f324e33c1c3cc19d25c0483474671ecb87`
- Integración: Pull Request #3 (`fix/gate2-adversarial-remediation` -> `main`)
- Fecha registrada: `2026-08-30`

## Resultado de integración y verificación

- Integración controlada mediante PR #3 autorizada por Product Owner.
- Verificación del árbol de merge: igualdad exacta con el rehearsal previo (`238222f324e33c1c3cc19d25c0483474671ecb87`).
- Suite completa de pruebas: 289 passed, 3 skipped.
- Skips observados: exclusivamente por CuPy/CUDA no disponible en el entorno de ejecución.
- Comportamiento de bypass: bypass automático observado durante la creación de la rama, pero no durante el merge.
- Preservación de ramas: las ramas de Gate 2 (`fix/gate2-major-remediation`, `fix/gate2-distribution-gof-remediation`, `fix/gate2-adversarial-remediation`) quedan íntegramente preservadas sin borrado.

## Límites y deuda fuera de alcance

- `TD-GOF-SUPPORT-001` y `FINDING-ADV-NUM-004` permanecen abiertos y fuera del alcance de esta remediación.
- **La bondad de ajuste (GOF) no demuestra identidad distributiva**: un test de GOF que no rechaza no constituye prueba de que los datos sigan exactamente la distribución hipotética.
- Esta evidencia es inmutable y registra los hechos observados en el SHA integrado `f1725ebdfebcb667c053420e4cb4c1e35048f9e0`.
