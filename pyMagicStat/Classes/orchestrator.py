import numpy as np
from scipy.stats import kruskal, mannwhitneyu

import numpy as np

class ExperimentationIteration:
    """
    Orquestador para coordinar el proceso iterativo de eliminación de grupos
    en pruebas de hipótesis (paramétricas o no) hasta alcanzar un umbral deseado de R².
    
    Se espera que el objeto 'test_obj' implemente:
      - run_test(): que retorne un diccionario con resultados (incluyendo 'Total_SS' y 'Groups').
      - remove_group(idx): que elimine el grupo en la posición idx y retorne su etiqueta.
    """
    def __init__(self, test_obj, r2_target=0.75, max_iterations=100):
        self.test_obj = test_obj
        self.r2_target = r2_target
        self.max_iterations = max_iterations
        self.history = []  # Historial de iteraciones

    def run(self):
        iteration = 0
        while iteration < self.max_iterations:
            if len(self.test_obj.groups) < 2:
                break
            # Ejecutar el test con los grupos actuales
            results = self.test_obj.run_test()
            total_ss = results.get("Total_SS", None)
            groups = results.get("Groups", [])
            global_ssw = sum(g["SSW"] for g in groups)
            global_r2 = 1 - (global_ssw / total_ss) if total_ss and total_ss > 0 else None

            # Registrar la iteración actual
            iter_info = {
                "iteration": iteration,
                "groups": [g["Label"] for g in groups],
                "global_R2": global_r2,
                "global_p_value": results.get("p_value", None),
                "removed_group": None,
                "details": results
            }
            self.history.append(iter_info)

            # Condición de parada: si se alcanza el umbral de R² o quedan menos de 2 grupos
            if global_r2 is not None and global_r2 >= self.r2_target:
                break
            if len(groups) < 2:
                break

            # Identificar el grupo con el p-valor individual más alto (menos significativo)
            max_p = -1
            idx_to_remove = None
            for i, group in enumerate(groups):
                if group["p_value"] > max_p:
                    max_p = group["p_value"]
                    idx_to_remove = i

            # Eliminar el grupo y actualizar el historial
            removed_label = self.test_obj.remove_group(idx_to_remove)
            self.history[-1]["removed_group"] = removed_label

            iteration += 1

        return self.history
