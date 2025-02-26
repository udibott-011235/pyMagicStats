import numpy as np
from itertools import combinations
import math
import random

# ---------------------- Capa de Evaluación Estadística ----------------------
# Se asume que ya disponemos de una clase especializada, por ejemplo, kruskalWallisTest.
from pyMagicStat.Classes.nonParametricHTest import kruskalWallisTest

class StatisticalEvaluator:
    """
    Esta clase encapsula la evaluación estadística utilizando una prueba especializada.
    Se encarga de:
      - Ejecutar la prueba (ej. kruskalWallisTest) con una configuración de grupos.
      - Extraer indicadores clave, por ejemplo:
           global_R2 = 1 - (suma de SSW de los grupos / Total_SS)
           p_value (valor global de la prueba)
    """
    def __init__(self, test_obj_class):
        self.test_obj_class = test_obj_class
    
    def evaluate(self, groups, labels):
        """
        Evalúa un conjunto de grupos utilizando la prueba estadística.
        Retorna un diccionario con:
          - "global_R2": el valor calculado de R².
          - "p_value": el valor p global.
          - "results": los resultados completos de la prueba.
        """
        test_obj = self.test_obj_class(*groups, labels=labels)
        results = test_obj.run_test()  # Se espera un dict con 'Total_SS' y 'Groups'
        total_ss = results.get("Total_SS", None)
        groups_info = results.get("Groups", [])
        global_ssw = sum(g["SSW"] for g in groups_info)
        global_r2 = 1 - (global_ssw / total_ss) if total_ss and total_ss > 0 else None
        
        return {
            "global_R2": global_r2,
            "p_value": results.get("p_value"),
            "results": results
        }

# ---------------------- Framework Experimental ----------------------
class OptimizedExperimentationIteration:
    """
    Framework Experimental que coordina la iteración para optimizar la configuración
    de grupos, utilizando estrategias como:
      - "greedy": elimina secuencialmente el grupo menos significativo.
      - "exhaustive": evalúa todas las combinaciones posibles (para pocos grupos).
      - "simulated_annealing": recocido simulado para explorar el espacio de soluciones.
      
    NOTA: Este framework no genera ni evalúa los estadísticos; únicamente orquesta
    el proceso iterativo usando un objeto StatisticalEvaluator.
    """
    def __init__(self, evaluator, groups, labels=None, strategy="greedy", r2_target=0.75, max_iterations=100, **kwargs):
        self.evaluator = evaluator  # Objeto StatisticalEvaluator
        self.initial_groups = groups
        if labels is None:
            self.labels = [f"Group {i+1}" for i in range(len(groups))]
        else:
            self.labels = labels
        self.strategy = strategy
        self.r2_target = r2_target
        self.max_iterations = max_iterations
        self.kwargs = kwargs
        self.history = []

    def run_greedy(self):
        groups = list(self.initial_groups)
        labels = list(self.labels)
        history = []
        for iteration in range(self.max_iterations):
            if len(groups) < 2:
                break
            eval_result = self.evaluator.evaluate(groups, labels)
            global_r2 = eval_result["global_R2"]
            iter_info = {
                "iteration": iteration,
                "groups": labels.copy(),
                "global_R2": global_r2,
                "evaluation": eval_result
            }
            history.append(iter_info)
            # Detenerse si se alcanza el umbral deseado
            if global_r2 is not None and global_r2 >= self.r2_target:
                break
            if len(groups) < 2:
                break
            # Criterio greedy: eliminar el grupo con mayor p_value
            groups_info = eval_result["results"].get("Groups", [])
            max_p = -1
            idx_to_remove = None
            for i, info in enumerate(groups_info):
                if info["p_value"] > max_p:
                    max_p = info["p_value"]
                    idx_to_remove = i
            # Eliminar el grupo seleccionado
            groups.pop(idx_to_remove)
            labels.pop(idx_to_remove)
        self.history = history
        return history

    def run_exhaustive(self):
        best_r2 = -np.inf
        best_subset = None
        best_labels = None
        best_eval = None
        history = []
        n = len(self.initial_groups)
        # Evaluar todas las combinaciones posibles (con al menos 2 grupos)
        for r in range(n, 1, -1):
            for indices in combinations(range(n), r):
                subset = [self.initial_groups[i] for i in indices]
                subset_labels = [self.labels[i] for i in indices]
                eval_result = self.evaluator.evaluate(subset, subset_labels)
                global_r2 = eval_result["global_R2"]
                history.append({
                    "subset_indices": indices,
                    "groups": subset_labels,
                    "global_R2": global_r2,
                    "evaluation": eval_result
                })
                if global_r2 is not None and global_r2 > best_r2:
                    best_r2 = global_r2
                    best_subset = subset
                    best_labels = subset_labels
                    best_eval = eval_result
        self.history = history
        return {"best_r2": best_r2, "best_groups": best_labels, "evaluation": best_eval, "history": history}

    def run_simulated_annealing(self):
        n = len(self.initial_groups)
        current_state = np.ones(n, dtype=bool)
        best_state = current_state.copy()
        initial_subset = [self.initial_groups[i] for i in range(n) if current_state[i]]
        initial_labels = [self.labels[i] for i in range(n) if current_state[i]]
        current_eval = self.evaluator.evaluate(initial_subset, initial_labels)
        current_r2 = current_eval["global_R2"]
        best_r2 = current_r2
        # Parámetros del recocido simulado
        T = self.kwargs.get("initial_temperature", 1.0)
        alpha = self.kwargs.get("cooling_rate", 0.95)
        iterations = self.max_iterations
        history = []
        for iter in range(iterations):
            neighbor = current_state.copy()
            idx = np.random.randint(0, n)
            neighbor[idx] = not neighbor[idx]
            if neighbor.sum() < 2:
                continue
            neighbor_subset = [self.initial_groups[i] for i in range(n) if neighbor[i]]
            neighbor_labels = [self.labels[i] for i in range(n) if neighbor[i]]
            neighbor_eval = self.evaluator.evaluate(neighbor_subset, neighbor_labels)
            neighbor_r2 = neighbor_eval["global_R2"]
            delta = neighbor_r2 - current_r2
            if delta > 0:
                accept = True
            else:
                accept_probability = np.exp(delta / T) if T > 0 else 0
                accept = np.random.rand() < accept_probability
            if accept:
                current_state = neighbor
                current_r2 = neighbor_r2
                if current_r2 > best_r2:
                    best_r2 = current_r2
                    best_state = current_state.copy()
            history.append({
                "iteration": iter,
                "state": current_state.copy(),
                "global_R2": current_r2
            })
            T *= alpha
        best_labels = [self.labels[i] for i in range(n) if best_state[i]]
        best_subset = [self.initial_groups[i] for i in range(n) if best_state[i]]
        self.history = history
        return {"best_r2": best_r2, "best_groups": best_labels, "evaluation": self.evaluator.evaluate(best_subset, best_labels), "history": history}

    def run(self):
        if self.strategy == "greedy":
            return self.run_greedy()
        elif self.strategy == "exhaustive":
            return self.run_exhaustive()
        elif self.strategy == "simulated_annealing":
            return self.run_simulated_annealing()
        else:
            raise ValueError("Estrategia no reconocida. Use 'greedy', 'exhaustive' o 'simulated_annealing'.")
