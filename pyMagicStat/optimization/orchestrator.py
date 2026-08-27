import numpy as np
from itertools import combinations
import math
import random
from typing import Any, Dict, List, Optional, Type, Union

# ---------------------- Capa de Evaluación Estadística ----------------------
from pyMagicStat.inference.non_parametric import kruskalWallisTest

class StatisticalEvaluator:
    """
    Encapsula la evaluación estadística utilizando una prueba especializada.
    
    Esta clase permite inicializar una prueba estadística (ej. kruskalWallisTest)
    y calcular indicadores clave (como R² global y valor p) a partir de grupos de datos.

    Parameters
    ----------
    test_obj_class : Type
        Referencia a la clase de prueba estadística a instanciar (debe tener un método `run_test()`).
    """
    def __init__(self, test_obj_class: Type) -> None:
        self.test_obj_class: Type = test_obj_class
    
    def evaluate(self, groups: List[np.ndarray], labels: List[str]) -> Dict[str, Any]:
        """
        Evalúa un conjunto de grupos utilizando la prueba estadística.

        Parameters
        ----------
        groups : List[np.ndarray]
            Lista de arreglos numpy, donde cada arreglo representa un grupo de datos.
        labels : List[str]
            Lista de etiquetas descriptivas para cada grupo.

        Returns
        -------
        Dict[str, Any]
            Diccionario con "global_R2", "p_value" y los resultados completos ("results").
        """
        test_obj = self.test_obj_class(*groups, labels=labels)
        results: Dict[str, Any] = test_obj.run_test()
        
        total_ss: Optional[float] = results.get("Total_SS", None)
        groups_info: List[Dict[str, Any]] = results.get("Groups", [])
        
        global_ssw: float = sum(g.get("SSW", 0.0) for g in groups_info)
        global_r2: Optional[float] = 1 - (global_ssw / total_ss) if total_ss and total_ss > 0 else None
        
        return {
            "global_R2": global_r2,
            "p_value": results.get("p_value"),
            "results": results
        }

# ---------------------- Framework Experimental ----------------------
class OptimizedExperimentationIteration:
    """
    Framework Experimental que coordina la iteración para optimizar la configuración
    de grupos, utilizando diversas estrategias de búsqueda heurística y exacta.

    Parameters
    ----------
    evaluator : StatisticalEvaluator
        Instancia del evaluador estadístico a utilizar.
    groups : List[np.ndarray]
        Lista inicial de grupos (arreglos de datos).
    labels : List[str], optional
        Etiquetas para cada grupo. Si no se proveen, se generan automáticamente.
    strategy : str, default="greedy"
        Estrategia de optimización a usar ('greedy', 'exhaustive', 'simulated_annealing').
    r2_target : float, default=0.75
        Umbral objetivo de R² para detener la optimización (solo usado en 'greedy').
    max_iterations : int, default=100
        Número máximo de iteraciones.
    **kwargs : Any
        Parámetros adicionales, por ejemplo, `initial_temperature` o `cooling_rate` 
        para simulated annealing.
    """
    def __init__(
        self,
        evaluator: StatisticalEvaluator,
        groups: List[np.ndarray],
        labels: Optional[List[str]] = None,
        strategy: str = "greedy",
        r2_target: float = 0.75,
        max_iterations: int = 100,
        **kwargs: Any
    ) -> None:
        self.evaluator: StatisticalEvaluator = evaluator
        self.initial_groups: List[np.ndarray] = groups
        self.labels: List[str] = labels if labels is not None else [f"Group {i+1}" for i in range(len(groups))]
        self.strategy: str = strategy
        self.r2_target: float = r2_target
        self.max_iterations: int = max_iterations
        self.kwargs: Dict[str, Any] = kwargs
        self.history: List[Any] = []

    def run_greedy(self) -> List[Dict[str, Any]]:
        """
        Ejecuta la optimización utilizando un enfoque Greedy.
        Elimina iterativamente el grupo con el mayor valor p.

        Returns
        -------
        List[Dict[str, Any]]
            Historial de iteraciones con los grupos evaluados y métricas.
        """
        groups = list(self.initial_groups)
        labels = list(self.labels)
        history: List[Dict[str, Any]] = []
        
        for iteration in range(self.max_iterations):
            if len(groups) < 2:
                break
            eval_result = self.evaluator.evaluate(groups, labels)
            global_r2 = eval_result.get("global_R2")
            
            iter_info = {
                "iteration": iteration,
                "groups": labels.copy(),
                "global_R2": global_r2,
                "evaluation": eval_result
            }
            history.append(iter_info)
            
            if global_r2 is not None and global_r2 >= self.r2_target:
                break
            if len(groups) < 2:
                break
                
            groups_info = eval_result["results"].get("Groups", [])
            max_p = -1.0
            idx_to_remove = -1
            for i, info in enumerate(groups_info):
                if info.get("p_value", 0.0) > max_p:
                    max_p = info["p_value"]
                    idx_to_remove = i
                    
            if idx_to_remove != -1:
                groups.pop(idx_to_remove)
                labels.pop(idx_to_remove)
                
        self.history = history
        return history

    def run_exhaustive(self) -> Dict[str, Any]:
        """
        Evalúa todas las combinaciones posibles de grupos para maximizar R².
        Recomendado solo para un número pequeño de grupos.

        Returns
        -------
        Dict[str, Any]
            El mejor subconjunto encontrado y el historial completo.
        """
        best_r2 = -np.inf
        best_subset: Optional[List[np.ndarray]] = None
        best_labels: Optional[List[str]] = None
        best_eval: Optional[Dict[str, Any]] = None
        history: List[Dict[str, Any]] = []
        n = len(self.initial_groups)
        
        for r in range(n, 1, -1):
            for indices in combinations(range(n), r):
                subset = [self.initial_groups[i] for i in indices]
                subset_labels = [self.labels[i] for i in indices]
                eval_result = self.evaluator.evaluate(subset, subset_labels)
                global_r2 = eval_result.get("global_R2")
                
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

    def run_simulated_annealing(self) -> Dict[str, Any]:
        """
        Usa recocido simulado (Simulated Annealing) para explorar combinaciones de grupos.

        Returns
        -------
        Dict[str, Any]
            La mejor configuración encontrada.
        """
        n = len(self.initial_groups)
        current_state = np.ones(n, dtype=bool)
        best_state = current_state.copy()
        
        initial_subset = [self.initial_groups[i] for i in range(n) if current_state[i]]
        initial_labels = [self.labels[i] for i in range(n) if current_state[i]]
        current_eval = self.evaluator.evaluate(initial_subset, initial_labels)
        current_r2 = current_eval.get("global_R2", 0.0)
        best_r2 = current_r2
        
        T = float(self.kwargs.get("initial_temperature", 1.0))
        alpha = float(self.kwargs.get("cooling_rate", 0.95))
        history: List[Dict[str, Any]] = []
        
        for iter in range(self.max_iterations):
            neighbor = current_state.copy()
            idx = np.random.randint(0, n)
            neighbor[idx] = not neighbor[idx]
            
            if neighbor.sum() < 2:
                continue
                
            neighbor_subset = [self.initial_groups[i] for i in range(n) if neighbor[i]]
            neighbor_labels = [self.labels[i] for i in range(n) if neighbor[i]]
            neighbor_eval = self.evaluator.evaluate(neighbor_subset, neighbor_labels)
            neighbor_r2 = neighbor_eval.get("global_R2", 0.0)
            
            if neighbor_r2 is None:
                continue
                
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
        return {
            "best_r2": best_r2, 
            "best_groups": best_labels, 
            "evaluation": self.evaluator.evaluate(best_subset, best_labels), 
            "history": history
        }

    def run(self) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Ejecuta la estrategia de optimización seleccionada.

        Returns
        -------
        Union[List[Dict[str, Any]], Dict[str, Any]]
            Resultados de la iteración.

        Raises
        ------
        ValueError
            Si la estrategia proporcionada no es reconocida.
        """
        if self.strategy == "greedy":
            return self.run_greedy()
        elif self.strategy == "exhaustive":
            return self.run_exhaustive()
        elif self.strategy == "simulated_annealing":
            return self.run_simulated_annealing()
        else:
            raise ValueError("Estrategia no reconocida. Use 'greedy', 'exhaustive' o 'simulated_annealing'.")
