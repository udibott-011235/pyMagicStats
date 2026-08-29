# Registro de datasets

Todo dataset se clasifica como `input`, `synthetic`, `derived` o `external` y
recibe una dataset card. El registro no confunde los datos que alimentan un
experimento con sus resultados agregados.

## Inventario inicial

| ID | Tipo | Ruta/definición | Estado | Observación |
|---|---|---|---|---|
| `DS-001` | synthetic | `experiments/robustness_calibration.py` | accepted | 19 escenarios generados con semilla registrada; no existe raw permanente |
| `DS-002` | derived | `experiments/results/sampling_robustness_summary.csv` | accepted | resumen de 152 celdas; no es dataset de entrada |
| `DS-003` | input/fixture | `examples/df_test_2way_t_test.xlsx` | open | procedencia, licencia y contrato de columnas deben documentarse antes de reutilización científica |

Para archivos grandes, almacene un puntero estable, hash criptográfico,
licencia y procedimiento de obtención. Nunca publique PII o secretos.

Use [`DATASET_CARD_TEMPLATE.md`](DATASET_CARD_TEMPLATE.md).

