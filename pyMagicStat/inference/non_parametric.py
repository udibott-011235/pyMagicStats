import numpy as np
from scipy.stats import kruskal, mannwhitneyu, bootstrap as _scipy_bootstrap
from numba import njit
from typing import Any, Dict, List, Optional, Tuple, Union
from pyMagicStat.utils.utils import output_format

# ----------------------------
# Numba JIT functions for specific statistics
# ----------------------------

@njit
def _numba_resample_mean(data: np.ndarray, n_resamples: int) -> np.ndarray:
    n = data.shape[0]
    res = np.empty(n_resamples)
    for i in range(n_resamples):
        s = 0.0
        for _ in range(n):
            idx = np.random.randint(0, n)
            s += data[idx]
        res[i] = s / n
    return res

@njit
def _numba_resample_median(data: np.ndarray, n_resamples: int) -> np.ndarray:
    n = data.shape[0]
    res = np.empty(n_resamples)
    temp = np.empty(n)
    for i in range(n_resamples):
        for j in range(n):
            temp[j] = data[np.random.randint(0, n)]
        # insertion sort for median
        for a in range(1, n):
            key = temp[a]
            b = a - 1
            while b >= 0 and temp[b] > key:
                temp[b + 1] = temp[b]
                b -= 1
            temp[b + 1] = key
        if n % 2:
            res[i] = temp[n // 2]
        else:
            res[i] = 0.5 * (temp[n // 2 - 1] + temp[n // 2])
    return res

@njit
def _numba_resample_variance(data: np.ndarray, n_resamples: int) -> np.ndarray:
    n = data.shape[0]
    res = np.empty(n_resamples)
    for i in range(n_resamples):
        m = 0.0
        for _ in range(n):
            m += data[np.random.randint(0, n)]
        m /= n
        v = 0.0
        for _ in range(n):
            diff = data[np.random.randint(0, n)] - m
            v += diff * diff
        res[i] = v / n
    return res

# ----------------------------
# BootstrapCI Class
# ----------------------------
class BootstrapCI:
    """
    Calcula intervalos de confianza mediante remuestreo Bootstrap.

    Parameters
    ----------
    data : Any
        Datos sobre los cuales calcular el intervalo.
    stat : str, default='mean'
        Estadístico a calcular ('mean', 'median', 'variance', 'proportion').
    method : str, default='scipy'
        Método computacional ('scipy' o 'numba').
    alpha : float, default=0.05
        Nivel de significancia.
    n_resamples : int, default=5000
        Número de iteraciones de remuestreo.
    p0 : float, optional
        Valor de referencia (umbral) para el cálculo de proporciones.
    """
    def __init__(
        self,
        data: Any,
        stat: str = 'mean',
        method: str = 'scipy',
        alpha: float = 0.05,
        n_resamples: int = 5000,
        p0: Optional[float] = None
    ) -> None:
        self.data: np.ndarray = np.array(data)
        self.stat: str = stat
        self.method: str = method
        self.alpha: float = alpha
        self.n_resamples: int = n_resamples
        self.p0: Optional[float] = p0
        
        if stat not in ('mean', 'median', 'variance', 'proportion'):
            raise ValueError(f"Stat desconocido: {stat}")
        if method not in ('numba', 'scipy'):
            raise ValueError(f"Method desconocido: {method}")

    def compute(self) -> Dict[str, Any]:
        """
        Ejecuta el cálculo del intervalo de confianza.

        Returns
        -------
        Dict[str, Any]
            Diccionario estandarizado con los límites `lb` y `ub`.
        """
        if self.method == 'numba':
            lb, ub = self._compute_numba()
        else:
            lb, ub = self._compute_scipy()
        return output_format(lb=lb, ub=ub)

    def _compute_numba(self) -> Tuple[float, float]:
        if self.stat == 'mean':
            res = _numba_resample_mean(self.data, self.n_resamples)
        elif self.stat == 'median':
            res = _numba_resample_median(self.data, self.n_resamples)
        elif self.stat == 'variance':
            res = _numba_resample_variance(self.data, self.n_resamples)
        else:  # proportion
            if self.p0 is None:
                binary = self.data
            else:
                binary = np.where(self.data >= self.p0, 1.0, 0.0)
            res = _numba_resample_mean(binary, self.n_resamples)
            
        lower: float = float(np.percentile(res, self.alpha / 2 * 100))
        upper: float = float(np.percentile(res, (1 - self.alpha / 2) * 100))
        return lower, upper

    def _compute_scipy(self) -> Tuple[float, float]:
        if self.stat == 'proportion':
            func = (lambda x: np.mean(x >= self.p0)) if self.p0 is not None else np.mean
        else:
            func = {'mean': np.mean, 'median': np.median, 'variance': np.var}[self.stat]
            
        ci = _scipy_bootstrap((self.data,), func,
                              confidence_level=1 - self.alpha,
                              n_resamples=self.n_resamples,
                              method='percentile')
        return float(ci.confidence_interval.low), float(ci.confidence_interval.high)

# ----------------------------
# Kruskal-Wallis Test class
# ----------------------------
class kruskalWallisTest:
    """
    Prueba de Kruskal-Wallis con cálculo de R² y pruebas Mann-Whitney post-hoc.

    Parameters
    ----------
    *groups : Any
        Secuencia de arreglos o listas representando cada grupo.
    alpha : float, default=0.05
        Nivel de significancia.
    labels : List[str], optional
        Etiquetas descriptivas para los grupos.
    alternative : str, default="two-sided"
        Hipótesis alternativa para Mann-Whitney.
    """
    def __init__(
        self,
        *groups: Any,
        alpha: float = 0.05,
        labels: Optional[List[str]] = None,
        alternative: str = "two-sided"
    ) -> None:
        if len(groups) < 2:
            raise ValueError("At least two groups required")
        self.groups: List[np.ndarray] = [np.array(g) for g in groups]
        self.alpha: float = alpha
        self.alternative: str = alternative
        self.labels: List[str] = labels if labels is not None else [f"Group {i+1}" for i in range(len(groups))]
        
        if len(self.labels) != len(groups):
            raise ValueError("Labels length mismatch")
            
        self._compute_r_squared()
        self.results: Dict[str, Any] = {}

    def _compute_r_squared(self) -> None:
        all_data = np.concatenate(self.groups)
        grand_mean = np.mean(all_data)
        self.ss_total: float = float(np.sum((all_data - grand_mean) ** 2))
        self.ss_within: List[float] = [float(np.sum((g - np.mean(g)) ** 2)) for g in self.groups]
        self.r_squared: List[float] = [1.0 - (ssw / self.ss_total) if self.ss_total > 0 else 0.0 for ssw in self.ss_within]

    def run_test(self) -> Dict[str, Any]:
        """
        Ejecuta la prueba global de Kruskal-Wallis y pruebas post-hoc.

        Returns
        -------
        Dict[str, Any]
            Resultados formateados incluyendo estadísticos H, valor p, grados de libertad y
            un desglose por grupo (SSW, R², p_value).
        """
        H_statistic, p_value = kruskal(*self.groups)
        df: int = len(self.groups) - 1
        p_values_per_group: List[float] = []
        
        for i, g in enumerate(self.groups):
            rest = np.concatenate([self.groups[j] for j in range(len(self.groups)) if j != i])
            _, p_indiv = mannwhitneyu(g, rest, alternative=self.alternative)
            p_values_per_group.append(float(p_indiv))
            
        self.results = {
            "H_statistic": float(H_statistic),
            "p_value": float(p_value),
            "df": df,
            "Total_SS": float(self.ss_total),
            "Groups": [
                {"Label": lab, "SSW": float(ssw), "R^2": float(r2), "p_value": float(pv)}
                for lab, ssw, r2, pv in zip(self.labels, self.ss_within, self.r_squared, p_values_per_group)
            ]
        }
        return output_format(data=self.results)

    def remove_group(self, idx: int) -> None:
        """
        Elimina un grupo por índice y recalcula R².

        Parameters
        ----------
        idx : int
            Índice del grupo a eliminar.
        """
        self.labels.pop(idx)
        self.groups.pop(idx)
        if len(self.groups) >= 1:
            self._compute_r_squared()
