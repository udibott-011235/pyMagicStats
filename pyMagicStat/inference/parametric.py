import numpy as np
import scipy.stats as stats
from math import ceil
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pyMagicStat.utils.utils import output_format, parse_hypothesis
from pyMagicStat.assumptions import (
    AssessmentStatus,
    Estimand,
    InferenceValidator,
)
from pyMagicStat.inference.selector import MethodSelector

#------------------ Clase Base ------------------#

class ParametricMethod:
    """
    Clase base para la aplicación de métodos de estadística paramétrica.
    Valida inferencia sobre una media sin transformar ni remuestrear los datos.

    Parameters
    ----------
    data : Any
        Estructura de datos (lista, tuple, np.ndarray, pd.Series) a analizar.
    alpha : float, default=0.05
        Nivel de significancia (ej. 0.05 para un 95% de confianza).
    apply_transform : bool, default=False
        Indica si se deben aplicar transformaciones a los datos para cumplir supuestos.

    Raises
    ------
    ValueError
        Si los datos contienen valores no finitos (NaN o Inf), o si no cumplen los supuestos.
    """
    def __init__(
        self,
        data: Any,
        alpha: float = 0.05,
        apply_transform: bool = False,
        *,
        independence: str = "unknown",
        strict: bool = True,
    ) -> None:
        if apply_transform:
            raise NotImplementedError(
                "Automatic transformation is not implemented because it can change the estimand."
            )
        self.alpha = float(alpha)
        validation = InferenceValidator(alpha=alpha).validate_one_sample(
            data,
            estimand=Estimand.MEAN,
            independence=independence,
        )
        self.data = validation.samples[0]
        self.n = int(self.data.size)
        self.assumption_report = validation.report
        self.inference_decision = MethodSelector().select(validation.report)
        self.is_normal = (
            validation.report.assessments["shape"].status is AssessmentStatus.PASS
        )
        self.tlc_applied = False

        if self.inference_decision.selected_method is None:
            message = "; ".join(self.inference_decision.reasons)
            if strict:
                raise ValueError(f"Parametric mean inference is not recommended: {message}")
            warnings.warn(message, UserWarning)

    def validate_data(self) -> bool:
        """
        Valida que los datos no contengan valores infinitos o nulos.

        Returns
        -------
        bool
            True si los datos son finitos, False en caso contrario.
        """
        return bool(
            self.data.ndim == 1
            and self.data.size >= 2
            and np.all(np.isfinite(self.data))
            and np.var(self.data, ddof=1) > 0
        )

#------------------ Intervalos Paramétricos ------------------#

class NormalDistConfidenceIntervals(ParametricMethod):
    """
    Clase base para intervalos de confianza que asumen distribución normal.
    Precalcula media, desviación estándar y error estándar.
    """
    def __init__(
        self,
        data: Any,
        alpha: float = 0.05,
        apply_transform: bool = False,
        *,
        independence: str = "unknown",
        strict: bool = True,
    ) -> None:
        super().__init__(
            data,
            alpha,
            apply_transform,
            independence=independence,
            strict=strict,
        )
        self.mean: float = np.mean(self.data)
        self.std_dev: float = np.std(self.data, ddof=1)
        self.variance: float = self.std_dev ** 2
        self.effective_n: int = self.n
        self.std_error: float = self.std_dev / np.sqrt(self.n)

class PopulationMeanCI(NormalDistConfidenceIntervals):
    """
    Calcula el intervalo de confianza para la media poblacional.
    """
    def calculate_interval(self) -> Dict[str, Any]:
        """
        Calcula los límites inferior y superior del intervalo para la media.

        Returns
        -------
        Dict[str, Any]
            Diccionario conteniendo 'lb', 'ub', 'Result' y un mensaje 'txt'.
        """
        try:
            critical_value = stats.t.ppf(1 - self.alpha / 2, df=self.n - 1)
            if self.effective_n <= 1:
                return output_format(bool_result=False, txt='Sample is too small, must be > 1')
            if self.std_dev <= 0:
                return output_format(bool_result=False, txt='Standard Deviation must be > 0')

            margin_of_error = critical_value * self.std_error
            lower_bound = self.mean - margin_of_error
            upper_bound = self.mean + margin_of_error

            result = output_format(
                bool_result=True,
                txt='Confidence interval for the mean calculated with Student t quantiles',
                ub=upper_bound,
                lb=lower_bound,
            )
            result["df"] = self.n - 1
            result["std_error"] = float(self.std_error)
            result["assumptions"] = self.assumption_report.to_dict()
            result["inference_decision"] = self.inference_decision.to_dict()
            return result
        except Exception as e:
            return output_format(bool_result=False, txt=f'Error Calculating Confidence Interval for the Mean: {e}')

    def required_sample_size(self, margin_error: float) -> Dict[str, Any]:
        """
        Calcula el tamaño de muestra requerido para un margen de error dado.

        Parameters
        ----------
        margin_error : float
            Margen de error objetivo.

        Returns
        -------
        Dict[str, Any]
            Diccionario con el resultado de la validación.
        """
        try:
            z_value = stats.norm.ppf(1 - self.alpha / 2)

            if margin_error <= 0: 
                return output_format(bool_result=False, txt=f"Error must be > 0, received {margin_error}")
            if self.std_dev <= 0:
                return output_format(bool_result=False, txt="Standard Deviation must be > 0")
            n_req = ceil((z_value * self.std_dev / margin_error) ** 2)
            # Retornar n_req también sería útil en el futuro.
            return output_format(bool_result=True, txt=f"Sample Size Calculated Fine: {n_req}")
        except Exception as e:
            return output_format(bool_result=False, txt=f'Error calculating the size of the sample: {str(e)}')

class PopulationProportionCI:
    """
    Calcula el intervalo de confianza para una proporción poblacional.
    """
    def __init__(
        self,
        data: Any,
        alpha: float = 0.05,
        incidences: Optional[Union[int, float, Callable]] = None,
        method: str = "wilson",
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 1 or self.data.size < 1 or not np.all(np.isfinite(self.data)):
            raise ValueError("Proportion data must be a finite one-dimensional sample")
        self.alpha = float(alpha)
        self.n = int(self.data.size)
        self.method = method.lower()
        if self.method not in {"wilson", "wald"}:
            raise ValueError("method must be 'wilson' or 'wald'")
        if callable(incidences):
            self.incidence_ratio: float = np.mean([1 if incidences(x) else 0 for x in self.data])
        elif incidences is not None:
            count = float(incidences)
            if not 0.0 <= count <= self.n:
                raise ValueError("incidences must be between 0 and the sample size")
            self.incidence_ratio = count / self.n
        else:
            if not np.all(np.isin(self.data, (0.0, 1.0))):
                raise ValueError("data must be binary when incidences is not provided")
            self.incidence_ratio = float(np.mean(self.data))
        self.prop_std_dev: float = np.sqrt(self.incidence_ratio * (1 - self.incidence_ratio) / self.n)

    def calculate_interval(self) -> Dict[str, Any]:
        """
        Calcula los límites del intervalo para la proporción.

        Returns
        -------
        Dict[str, Any]
            Límites inferior (lb) y superior (ub).
        """
        z_value = stats.norm.ppf(1 - self.alpha / 2)
        if self.method == "wald":
            margin_of_error = z_value * self.prop_std_dev
            lb = self.incidence_ratio - margin_of_error
            ub = self.incidence_ratio + margin_of_error
        else:
            z_squared = z_value ** 2
            denominator = 1.0 + z_squared / self.n
            center = (self.incidence_ratio + z_squared / (2.0 * self.n)) / denominator
            half_width = (
                z_value
                * np.sqrt(
                    self.incidence_ratio * (1.0 - self.incidence_ratio) / self.n
                    + z_squared / (4.0 * self.n ** 2)
                )
                / denominator
            )
            lb = max(0.0, center - half_width)
            ub = min(1.0, center + half_width)
        result = output_format(lb=float(lb), ub=float(ub))
        result.update(
            {
                "method": self.method,
                "estimate": self.incidence_ratio,
                "n": self.n,
            }
        )
        return result

class PopulationVarianceCI:
    """
    Calcula el intervalo de confianza para la varianza poblacional.
    """
    def __init__(
        self,
        data: Any,
        alpha: float = 0.05,
        *,
        strict: bool = True,
        independence: str = "unknown",
    ) -> None:
        self.alpha = float(alpha)
        validation = InferenceValidator(alpha=alpha).validate_one_sample(
            data,
            estimand=Estimand.VARIANCE,
            independence=independence,
        )
        self.data = validation.samples[0]
        self.n = int(self.data.size)
        self.variance = float(np.var(self.data, ddof=1))
        self.assumption_report = validation.report
        shape = validation.report.assessments["shape"]
        if shape.status is AssessmentStatus.FAIL and strict:
            raise ValueError(
                "Chi-square variance inference requires a sufficiently Gaussian population shape; "
                "consider a bootstrap variance interval."
            )
        if shape.status is AssessmentStatus.WARN:
            warnings.warn(
                "The chi-square variance interval is sensitive to the observed non-Gaussian shape.",
                UserWarning,
            )

    def calculate_interval(self) -> Dict[str, Any]:
        """
        Calcula los límites del intervalo usando la distribución Chi-cuadrado.

        Returns
        -------
        Dict[str, Any]
            Límites inferior (lb) y superior (ub).
        """
        chi2_lower = stats.chi2.ppf(1 - self.alpha / 2, self.n - 1)
        chi2_upper = stats.chi2.ppf(self.alpha / 2, self.n - 1)
        lower = ((self.n - 1) * self.variance) / chi2_lower
        upper = ((self.n - 1) * self.variance) / chi2_upper
        result = output_format(lb=lower, ub=upper)
        result["method"] = "chi_square"
        result["assumptions"] = self.assumption_report.to_dict()
        return result


#------------------ Pruebas de Hipótesis Paramétricas ------------------#

class VarianceHomogeneityTest:
    """
    Evalúa el supuesto de homogeneidad de varianzas (homocedasticidad) entre dos o más grupos.

    Parameters
    ----------
    *groups : Any
        Arreglos o listas de datos para cada grupo a comparar.
    method : str, default='levene'
        Prueba a utilizar ('levene', 'bartlett', 'fligner').
    alpha : float, default=0.05
        Nivel de significancia para la prueba.
    center : str, default='median'
        Método de centrado para Levene ('median', 'mean', 'trimmed').
    """
    def __init__(
        self,
        *groups: Any,
        method: str = 'levene',
        alpha: float = 0.05,
        center: str = 'median'
    ) -> None:
        if len(groups) < 2:
            raise ValueError("Se requieren al menos 2 grupos para evaluar la homogeneidad de varianzas.")
        self.groups: List[np.ndarray] = [np.array(g) for g in groups]
        self.method: str = method.lower()
        self.alpha: float = alpha
        self.center: str = center
        
        for i, g in enumerate(self.groups):
            if not np.all(np.isfinite(g)):
                raise ValueError(f"El grupo {i+1} contiene valores NaN o Inf.")

    def run_test(self) -> Dict[str, Any]:
        """
        Ejecuta la prueba de homogeneidad de varianzas.

        Returns
        -------
        Dict[str, Any]
            Diccionario estandarizado con el estadístico, p-value, e indicación de igualdad de varianzas.
        """
        if self.method == 'levene':
            stat, p_val = stats.levene(*self.groups, center=self.center)
        elif self.method == 'bartlett':
            stat, p_val = stats.bartlett(*self.groups)
        elif self.method == 'fligner':
            stat, p_val = stats.fligner(*self.groups)
        else:
            raise ValueError(f"Método de homocedasticidad desconocido: '{self.method}'. Use 'levene', 'bartlett' o 'fligner'.")

        equal_variance = bool(p_val >= self.alpha)
        txt_msg = (
            f"Varianzas homogéneas (p={p_val:.4f} >= alpha={self.alpha})."
            if equal_variance
            else f"Varianzas heterogéneas (p={p_val:.4f} < alpha={self.alpha})."
        )
        
        res_dict = {
            "Result": equal_variance,
            "equal_variance": equal_variance,
            "statistic": float(stat),
            "p_value": float(p_val),
            "method": f"Levene ({self.center})" if self.method == 'levene' else self.method.capitalize(),
            "alpha": self.alpha,
            "txt": txt_msg
        }
        return output_format(data=res_dict)


class OneSampleTTest(ParametricMethod):
    """
    Prueba t de una muestra para comparar la media poblacional con un valor de referencia (popmean).

    Parameters
    ----------
    data : Any
        Muestra de datos.
    popmean : float, default=0.0
        Valor de referencia para la media bajo la hipótesis nula H0.
    alpha : float, default=0.05
        Nivel de significancia.
    ha : str, optional
        Hipótesis alternativa explícita (ej. 'data > popmean', 'data < popmean').
    alternative : str, default='two-sided'
        Hipótesis alternativa heredada ('two-sided', 'greater', 'less').
    apply_transform : bool, default=False
        Indica si se deben aplicar transformaciones a los datos para forzar normalidad.
    """
    def __init__(
        self,
        data: Any,
        popmean: float = 0.0,
        alpha: float = 0.05,
        ha: Optional[str] = None,
        alternative: str = 'two-sided',
        apply_transform: bool = False,
        independence: str = "unknown",
        strict: bool = True,
    ) -> None:
        super().__init__(
            data,
            alpha,
            apply_transform,
            independence=independence,
            strict=strict,
        )
        self.popmean: float = float(popmean)
        self.alternative, self.ha = parse_hypothesis(ha=ha, alternative=alternative, is_two_sample=False)

        self.mean: float = float(np.mean(self.data))
        self.sample_std: float = float(np.std(self.data, ddof=1))
        
        self.std_error: float = float(self.sample_std / np.sqrt(self.n))
            
        self.df: int = self.n - 1

    def run_test(self) -> Dict[str, Any]:
        """
        Ejecuta la prueba t de una muestra.

        Returns
        -------
        Dict[str, Any]
            Diccionario estandarizado con el estadístico t, valor p, df, intervalo de confianza y decisión.
        """
        if self.std_error <= 0:
            return output_format(bool_result=False, txt="La desviación estándar debe ser mayor que 0.")
            
        t_stat = (self.mean - self.popmean) / self.std_error
        
        if self.alternative == 'two-sided':
            p_val = 2.0 * float(stats.t.sf(np.abs(t_stat), df=self.df))
            t_crit = float(stats.t.ppf(1 - self.alpha / 2, df=self.df))
            ci_lower = self.mean - t_crit * self.std_error
            ci_upper = self.mean + t_crit * self.std_error
        elif self.alternative == 'greater':
            p_val = float(stats.t.sf(t_stat, df=self.df))
            t_crit = float(stats.t.ppf(1 - self.alpha, df=self.df))
            ci_lower = self.mean - t_crit * self.std_error
            ci_upper = np.inf
        else:  # less
            p_val = float(stats.t.cdf(t_stat, df=self.df))
            t_crit = float(stats.t.ppf(1 - self.alpha, df=self.df))
            ci_lower = -np.inf
            ci_upper = self.mean + t_crit * self.std_error

        reject_null = bool(p_val < self.alpha)
        txt = (
            f"Se rechaza H0 (p={p_val:.4e} < alpha={self.alpha}): La media difiere significativamente de {self.popmean}."
            if reject_null
            else f"No se rechaza H0 (p={p_val:.4e} >= alpha={self.alpha}): No hay suficiente evidencia para afirmar una diferencia significativa respecto a {self.popmean}."
        )

        res_dict = {
            "Result": reject_null,
            "reject_null": reject_null,
            "statistic": float(t_stat),
            "p_value": float(p_val),
            "df": self.df,
            "sample_mean": self.mean,
            "popmean": self.popmean,
            "std_error": self.std_error,
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "lb": float(ci_lower),
            "ub": float(ci_upper),
            "alternative": self.alternative,
            "ha": self.ha,
            "is_normal": self.is_normal,
            "tlc_applied": self.tlc_applied,
            "assumptions": self.assumption_report.to_dict(),
            "inference_decision": self.inference_decision.to_dict(),
            "txt": txt
        }
        return output_format(data=res_dict)


class PairedTTest:
    """
    Prueba t de muestras pareadas (relacionadas) para comparar la media de las diferencias entre dos muestras.

    Parameters
    ----------
    data1 : Any, optional
        Primera muestra de observaciones pareadas. Alias: df1, group1.
    data2 : Any, optional
        Segunda muestra de observaciones pareadas. Alias: df2, group2.
    df1 : Any, optional
        Alias para data1.
    df2 : Any, optional
        Alias para data2.
    group1 : Any, optional
        Alias para data1.
    group2 : Any, optional
        Alias para data2.
    popmean : float, default=0.0
        Diferencia de media esperada bajo H0 (típicamente 0.0).
    alpha : float, default=0.05
        Nivel de significancia.
    ha : str, optional
        Hipótesis alternativa explícita (ej. 'df2 > df1', 'df1 > df2', 'df1 != df2').
    alternative : str, default='two-sided'
        Hipótesis alternativa heredada ('two-sided', 'greater', 'less').
    apply_transform : bool, default=False
        Indica si se aplican transformaciones para buscar normalidad en las diferencias.
    """
    def __init__(
        self,
        data1: Any = None,
        data2: Any = None,
        *,
        df1: Any = None,
        df2: Any = None,
        group1: Any = None,
        group2: Any = None,
        popmean: float = 0.0,
        alpha: float = 0.05,
        ha: Optional[str] = None,
        alternative: str = 'two-sided',
        apply_transform: bool = False,
        independence: str = "unknown",
        strict: bool = True,
    ) -> None:
        d1_raw = data1 if data1 is not None else (df1 if df1 is not None else group1)
        d2_raw = data2 if data2 is not None else (df2 if df2 is not None else group2)

        if d1_raw is None or d2_raw is None:
            raise ValueError("Se requieren dos muestras de datos para comparar (data1/df1/group1 y data2/df2/group2).")

        if apply_transform:
            raise NotImplementedError(
                "Automatic transformation is not implemented because it can change the estimand."
            )
        validation = InferenceValidator(alpha=alpha).validate_paired(
            d1_raw,
            d2_raw,
            independence=independence,
        )
        self.d1, self.d2 = validation.samples
        self.diff = validation.relevant_samples[0]
        self.assumption_report = validation.report
        self.inference_decision = MethodSelector().select(validation.report)
        if self.inference_decision.selected_method is None:
            message = "; ".join(self.inference_decision.reasons)
            if strict:
                raise ValueError(f"Parametric paired inference is not recommended: {message}")
            warnings.warn(message, UserWarning)

        self.alpha = float(alpha)
        self.popmean = float(popmean)
        self.alternative, self.ha = parse_hypothesis(ha=ha, alternative=alternative, is_two_sample=True)
        self.n = int(self.diff.size)
        self.df = self.n - 1
        self.mean_difference = float(np.mean(self.diff))
        self.std_error = float(np.std(self.diff, ddof=1) / np.sqrt(self.n))

    def run_test(self) -> Dict[str, Any]:
        """
        Ejecuta la prueba t de muestras pareadas.

        Returns
        -------
        Dict[str, Any]
            Resultados formateados de la prueba t pareada.
        """
        t_stat = (self.mean_difference - self.popmean) / self.std_error
        if self.alternative == "two-sided":
            p_val = 2.0 * float(stats.t.sf(abs(t_stat), df=self.df))
            critical = float(stats.t.ppf(1.0 - self.alpha / 2.0, df=self.df))
            ci_lower = self.mean_difference - critical * self.std_error
            ci_upper = self.mean_difference + critical * self.std_error
        elif self.alternative == "greater":
            p_val = float(stats.t.sf(t_stat, df=self.df))
            critical = float(stats.t.ppf(1.0 - self.alpha, df=self.df))
            ci_lower = self.mean_difference - critical * self.std_error
            ci_upper = np.inf
        else:
            p_val = float(stats.t.cdf(t_stat, df=self.df))
            critical = float(stats.t.ppf(1.0 - self.alpha, df=self.df))
            ci_lower = -np.inf
            ci_upper = self.mean_difference + critical * self.std_error

        reject_null = bool(p_val < self.alpha)
        text = (
            f"Se rechaza H0 en favor de Ha ({self.ha}) (Paired t-test, p={p_val:.4e} < alpha={self.alpha}): "
            f"Existe una diferencia significativa entre los grupos pareados."
            if reject_null
            else f"No se rechaza H0 (Paired t-test, Ha: {self.ha}, p={p_val:.4e} >= alpha={self.alpha}): "
            f"No hay suficiente evidencia para afirmar una diferencia significativa entre los grupos pareados."
        )
        return output_format(data={
            "Result": reject_null,
            "reject_null": reject_null,
            "statistic": float(t_stat),
            "p_value": p_val,
            "df": self.df,
            "mean_difference": self.mean_difference,
            "popmean": self.popmean,
            "std_error": self.std_error,
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "lb": float(ci_lower),
            "ub": float(ci_upper),
            "alternative": self.alternative,
            "ha": self.ha,
            "test_type": "Paired t-test",
            "assumptions": self.assumption_report.to_dict(),
            "inference_decision": self.inference_decision.to_dict(),
            "txt": text,
        })


class TwoSampleTTest:
    """
    Prueba t para dos muestras independientes (Student t-test o Welch t-test).
    Evalúa normalidad/TLC en cada muestra y homogeneidad de varianza entre ellas.

    Parameters
    ----------
    data1 : Any, optional
        Primera muestra independiente. Alias: df1, group1.
    data2 : Any, optional
        Segunda muestra independiente. Alias: df2, group2.
    df1 : Any, optional
        Alias para data1.
    df2 : Any, optional
        Alias para data2.
    group1 : Any, optional
        Alias para data1.
    group2 : Any, optional
        Alias para data2.
    popmean : float, default=0.0
        Diferencia poblacional bajo H0 (mu1 - mu2 = popmean).
    alpha : float, default=0.05
        Nivel de significancia.
    ha : str, optional
        Hipótesis alternativa explícita (ej. 'df2 > df1', 'df1 > df2', 'df1 != df2').
    alternative : str, default='two-sided'
        Hipótesis alternativa heredada ('two-sided', 'greater', 'less').
    equal_var : Optional[bool], default=None
        Si es True, aplica la prueba t de Student (varianzas iguales).
        Si es False, aplica la prueba t de Welch (varianzas desiguales).
        Si es None, aplica Welch por defecto y reporta heterogeneidad como diagnóstico.
    homogeneity_method : str, default='levene'
        Prueba de homocedasticidad a utilizar ('levene', 'bartlett', 'fligner').
    apply_transform : bool, default=False
        Indica si se aplican transformaciones a las muestras para normalidad.
    """
    def __init__(
        self,
        data1: Any = None,
        data2: Any = None,
        *,
        df1: Any = None,
        df2: Any = None,
        group1: Any = None,
        group2: Any = None,
        popmean: float = 0.0,
        alpha: float = 0.05,
        ha: Optional[str] = None,
        alternative: str = 'two-sided',
        equal_var: Optional[bool] = None,
        homogeneity_method: str = 'levene',
        apply_transform: bool = False,
        independence: str = "unknown",
        strict: bool = True,
    ) -> None:
        d1_raw = data1 if data1 is not None else (df1 if df1 is not None else group1)
        d2_raw = data2 if data2 is not None else (df2 if df2 is not None else group2)

        if d1_raw is None or d2_raw is None:
            raise ValueError("Se requieren dos muestras de datos para comparar (data1/df1/group1 y data2/df2/group2).")

        if apply_transform:
            raise NotImplementedError(
                "Automatic transformation is not implemented because it can change the estimand."
            )
        validation = InferenceValidator(alpha=alpha).validate_two_sample(
            d1_raw,
            d2_raw,
            independence=independence,
        )
        self.data1, self.data2 = validation.samples
        self.assumption_report = validation.report
        self.inference_decision = MethodSelector().select(
            validation.report,
            equal_var=equal_var,
        )
        if self.inference_decision.selected_method is None:
            message = "; ".join(self.inference_decision.reasons)
            if strict:
                raise ValueError(f"Parametric two-sample inference is not recommended: {message}")
            warnings.warn(message, UserWarning)

        self.popmean: float = float(popmean)
        self.alpha: float = alpha
        
        self.alternative, self.ha = parse_hypothesis(ha=ha, alternative=alternative, is_two_sample=True)

        self.n1: int = len(self.data1)
        self.n2: int = len(self.data2)
        self.mean1: float = float(np.mean(self.data1))
        self.mean2: float = float(np.mean(self.data2))
        self.var1: float = float(np.var(self.data1, ddof=1))
        self.var2: float = float(np.var(self.data2, ddof=1))
        
        homo_test = VarianceHomogeneityTest(
            self.data1,
            self.data2,
            method=homogeneity_method,
            alpha=alpha,
        )
        self.homogeneity_res = homo_test.run_test()
        self.equal_var = self.inference_decision.selected_method == "student_t"

    def run_test(self) -> Dict[str, Any]:
        """
        Ejecuta la prueba t de 2 muestras independientes (Student o Welch).

        Returns
        -------
        Dict[str, Any]
            Resultados formateados de la prueba.
        """
        diff_mean = self.mean1 - self.mean2
        
        if self.equal_var:
            # Student's t-test (varianzas iguales)
            df = float(self.n1 + self.n2 - 2)
            pooled_var = ((self.n1 - 1) * self.var1 + (self.n2 - 1) * self.var2) / df
            std_error = float(np.sqrt(pooled_var * (1.0 / self.n1 + 1.0 / self.n2)))
            test_name = "Student's t-test"
        else:
            # Welch's t-test (varianzas desiguales)
            v1_n1 = self.var1 / self.n1
            v2_n2 = self.var2 / self.n2
            std_error = float(np.sqrt(v1_n1 + v2_n2))
            
            # Grados de libertad Welch-Satterthwaite
            numerator = (v1_n1 + v2_n2) ** 2
            denominator = (v1_n1 ** 2) / (self.n1 - 1) + (v2_n2 ** 2) / (self.n2 - 1)
            df = float(numerator / denominator)
            test_name = "Welch's t-test"
            
        if std_error <= 0:
            return output_format(bool_result=False, txt="El error estándar calculado debe ser mayor que 0.")
            
        t_stat = (diff_mean - self.popmean) / std_error
        
        if self.alternative == 'two-sided':
            p_val = 2.0 * float(stats.t.sf(np.abs(t_stat), df=df))
            t_crit = float(stats.t.ppf(1 - self.alpha / 2, df=df))
            ci_lower = diff_mean - t_crit * std_error
            ci_upper = diff_mean + t_crit * std_error
        elif self.alternative == 'greater':
            p_val = float(stats.t.sf(t_stat, df=df))
            t_crit = float(stats.t.ppf(1 - self.alpha, df=df))
            ci_lower = diff_mean - t_crit * std_error
            ci_upper = np.inf
        else:  # less
            p_val = float(stats.t.cdf(t_stat, df=df))
            t_crit = float(stats.t.ppf(1 - self.alpha, df=df))
            ci_lower = -np.inf
            ci_upper = diff_mean + t_crit * std_error

        reject_null = bool(p_val < self.alpha)
        txt = (
            f"Se rechaza H0 en favor de Ha ({self.ha}) ({test_name}, p={p_val:.4e} < alpha={self.alpha}): "
            f"Existe una diferencia significativa entre las medias de los grupos."
            if reject_null
            else f"No se rechaza H0 ({test_name}, Ha: {self.ha}, p={p_val:.4e} >= alpha={self.alpha}): "
            f"No hay suficiente evidencia para afirmar una diferencia significativa de medias."
        )

        res_dict = {
            "Result": reject_null,
            "reject_null": reject_null,
            "statistic": float(t_stat),
            "p_value": float(p_val),
            "df": df,
            "mean1": self.mean1,
            "mean2": self.mean2,
            "diff_mean": diff_mean,
            "std_error": std_error,
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "lb": float(ci_lower),
            "ub": float(ci_upper),
            "equal_var": self.equal_var,
            "method": test_name,
            "alternative": self.alternative,
            "ha": self.ha,
            "homogeneity_test": self.homogeneity_res,
            "group1_assumptions": {
                "shape": self.assumption_report.assessments["shape_group_1"].to_dict(),
                "outliers": self.assumption_report.assessments["outliers_group_1"].to_dict(),
            },
            "group2_assumptions": {
                "shape": self.assumption_report.assessments["shape_group_2"].to_dict(),
                "outliers": self.assumption_report.assessments["outliers_group_2"].to_dict(),
            },
            "assumptions": self.assumption_report.to_dict(),
            "inference_decision": self.inference_decision.to_dict(),
            "txt": txt
        }
        return output_format(data=res_dict)



