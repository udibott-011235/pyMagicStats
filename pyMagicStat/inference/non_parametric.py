import numpy as np
from scipy.stats import kruskal, mannwhitneyu, bootstrap as _scipy_bootstrap
from numba import njit
from typing import Any, Dict, List, Optional, Tuple, Union
from pyMagicStat.utils.utils import output_format

# ----------------------------
# Numba JIT functions for specific statistics
# ----------------------------

@njit
def _numba_resample_mean(data: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
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
def _numba_resample_median(data: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
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
def _numba_resample_variance(
    data: np.ndarray,
    n_resamples: int,
    seed: int,
    ddof: int = 1,
) -> np.ndarray:
    np.random.seed(seed)
    n = data.shape[0]
    res = np.empty(n_resamples)
    sample = np.empty(n)
    for i in range(n_resamples):
        m = 0.0
        for j in range(n):
            sample[j] = data[np.random.randint(0, n)]
            m += sample[j]
        m /= n
        v = 0.0
        for j in range(n):
            diff = sample[j] - m
            v += diff * diff
        res[i] = v / (n - ddof)
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
    ddof : {0, 1}, default=1
        Divisor convention for ``stat='variance'``. ``ddof=1`` targets the
        conventional population variance using the same sample-variance
        estimator in the observed sample and every bootstrap resample.
        ``ddof=0`` instead bootstraps the empirical/MLE second central moment.
    """
    def __init__(
        self,
        data: Any,
        stat: str = 'mean',
        method: str = 'scipy',
        alpha: float = 0.05,
        n_resamples: int = 5000,
        p0: Optional[float] = None,
        interval_method: str = "bca",
        random_state: Optional[Union[int, np.random.Generator]] = None,
        ddof: int = 1,
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=float)
        self.stat: str = stat
        self.method: str = method.lower()
        self.alpha: float = float(alpha)
        self.n_resamples: int = int(n_resamples)
        self.p0: Optional[float] = p0
        self.interval_method = interval_method.lower()
        self.ddof = int(ddof)
        self.rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        if self.data.ndim != 1 or self.data.size < 2 or not np.all(np.isfinite(self.data)):
            raise ValueError("Bootstrap data must be a finite one-dimensional sample of size >= 2")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.n_resamples < 100:
            raise ValueError("n_resamples must be at least 100")
        
        if stat not in ('mean', 'median', 'variance', 'proportion'):
            raise ValueError(f"Stat desconocido: {stat}")
        if self.method not in ('numba', 'scipy'):
            raise ValueError(f"Method desconocido: {method}")
        if self.interval_method not in {"percentile", "basic", "bca"}:
            raise ValueError("interval_method must be 'percentile', 'basic', or 'bca'")
        if self.ddof not in {0, 1}:
            raise ValueError("ddof must be 0 or 1")
        if self.method == "numba" and self.interval_method != "percentile":
            raise ValueError("The numba backend currently supports percentile intervals only")

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
        result = output_format(lb=lb, ub=ub)
        result.update({
            "estimate": float(self._statistic(self.data)),
            "stat": self.stat,
            "backend": self.method,
            "interval_method": self.interval_method,
            "n_resamples": self.n_resamples,
        })
        if self.stat == "variance":
            result["ddof"] = self.ddof
        return result

    def _compute_numba(self) -> Tuple[float, float]:
        seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
        if self.stat == 'mean':
            res = _numba_resample_mean(self.data, self.n_resamples, seed)
        elif self.stat == 'median':
            res = _numba_resample_median(self.data, self.n_resamples, seed)
        elif self.stat == 'variance':
            res = _numba_resample_variance(
                self.data,
                self.n_resamples,
                seed,
                self.ddof,
            )
        else:  # proportion
            if self.p0 is None:
                binary = self.data
            else:
                binary = np.where(self.data >= self.p0, 1.0, 0.0)
            res = _numba_resample_mean(binary, self.n_resamples, seed)
            
        lower: float = float(np.percentile(res, self.alpha / 2 * 100))
        upper: float = float(np.percentile(res, (1 - self.alpha / 2) * 100))
        return lower, upper

    def _compute_scipy(self) -> Tuple[float, float]:
        method_name = "BCa" if self.interval_method == "bca" else self.interval_method
        ci = _scipy_bootstrap(
            (self.data,),
            self._statistic,
            confidence_level=1 - self.alpha,
            n_resamples=self.n_resamples,
            method=method_name,
            vectorized=False,
            rng=self.rng,
        )
        return float(ci.confidence_interval.low), float(ci.confidence_interval.high)

    def _statistic(self, data: np.ndarray) -> float:
        if self.stat == "proportion":
            return float(np.mean(data >= self.p0)) if self.p0 is not None else float(np.mean(data))
        if self.stat == "variance":
            return float(np.var(data, ddof=self.ddof))
        return float({"mean": np.mean, "median": np.median}[self.stat](data))


class BootstrapMeanDifferenceCI:
    """Bootstrap CI for the arithmetic mean difference of independent groups."""

    def __init__(
        self,
        data1: Any,
        data2: Any,
        *,
        alpha: float = 0.05,
        n_resamples: int = 5000,
        interval_method: str = "bca",
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        self.data1 = np.asarray(data1, dtype=float)
        self.data2 = np.asarray(data2, dtype=float)
        if any(
            sample.ndim != 1 or sample.size < 2 or not np.all(np.isfinite(sample))
            for sample in (self.data1, self.data2)
        ):
            raise ValueError("Each group must be a finite one-dimensional sample of size >= 2")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if n_resamples < 100:
            raise ValueError("n_resamples must be at least 100")
        self.alpha = float(alpha)
        self.n_resamples = int(n_resamples)
        self.interval_method = interval_method.lower()
        if self.interval_method not in {"percentile", "basic", "bca"}:
            raise ValueError("interval_method must be 'percentile', 'basic', or 'bca'")
        self.rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

    @staticmethod
    def _mean_difference(group1: np.ndarray, group2: np.ndarray) -> float:
        return float(np.mean(group1) - np.mean(group2))

    def compute(self) -> Dict[str, Any]:
        method_name = "BCa" if self.interval_method == "bca" else self.interval_method
        result = _scipy_bootstrap(
            (self.data1, self.data2),
            self._mean_difference,
            paired=False,
            vectorized=False,
            confidence_level=1.0 - self.alpha,
            n_resamples=self.n_resamples,
            method=method_name,
            rng=self.rng,
        )
        return {
            "lb": float(result.confidence_interval.low),
            "ub": float(result.confidence_interval.high),
            "estimate": self._mean_difference(self.data1, self.data2),
            "stat": "mean_difference",
            "backend": "scipy",
            "interval_method": self.interval_method,
            "n_resamples": self.n_resamples,
        }

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
