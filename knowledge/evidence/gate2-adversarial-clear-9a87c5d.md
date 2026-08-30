# EV-004 — Gate 2 adversarial clear — 9a87c5d

- Estado: `validated_with_limits`
- Candidato: `9a87c5d48dba8b8a172b5386d7318e7f37ec98fe`
- Parent directo: `0fc71c90c15f7c82b55ba650de742265d492df33`
- Rama: `fix/gate2-adversarial-remediation`
- Fecha registrada: `2026-08-30`

## Resultado focalizado

- `FINDING-ADV-GOF-001`: no reproducible.
- `FINDING-ADV-MUT-002`: no reproducible.
- Configuraciones metamórficas ejecutadas: 10,000.
- Escenarios Poisson: 54/54.
- Escenarios Binomial: 12/12.
- Pruebas focales Gate 2: 28 passed.
- Regresión histórica de distribuciones: 21 passed.
- Suite completa: 280 passed, 3 skipped.
- Hallazgos `CRITICAL`, `MAJOR` o `MODERATE` dentro del alcance: cero.

## Límites y deuda fuera de alcance

Permanecen fuera del alcance de esta conclusión:

- `TD-GOF-SUPPORT-001`;
- `FINDING-ADV-NUM-004`.

El candidato contiene dos commits posteriores a `main`, todavía no está
merged y es el único candidato Gate 2 vigente según DEC-006 y BR-012.

**ADVERSARIAL_CLEAR no constituye identidad distributiva ni autorización automática de merge**.

Esta evidencia sólo conserva el resultado recibido para el SHA exacto. No debe
transferirse a un SHA posterior ni usarse para ampliar la teoría estadística o
el alcance auditado.
