from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import warnings
from typing import Any, Dict, Optional, Union
from pyMagicStat.utils.utils import output_format
from pyMagicStat.assumptions import ShapeAssessment
from pyMagicStat._descriptive import sample_descriptives, univariate_sample



#Pendiente ! no esta implementando output_format
#hay que poner bonito el distribution visualization con tablas legibles ,  graficos  q&q
# necesitamos un refactory en el self.type, 
# hay que unificar la llamada al metodo de evaluacion de distribucion para facilitar usabilidad 
# organizar y complementar tabla de estadisticos en visualizacion 
# urgente hay que meter la funcion assing weith a dentro de evaluate normality  
# 

################################# ######
# 1. Clase Principal: Distribution
#######################################
class Distribution:
    """
    Immutable snapshot of one univariate sample and its descriptives.

    ``type`` and :meth:`update_type` remain only for compatibility with legacy
    distribution validators. New code should store structured results in
    ``assessments`` and must not interpret ``type["Normal"]`` as permission or
    prohibition for parametric inference.
    """
    def __init__(self, data: Any, dist_type: Optional[Dict[str, Any]] = None) -> None:
        try:
            snapshot = univariate_sample(data, label="Distribution data").copy()
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError("Error al convertir los datos a numpy array: " + str(e)) from e

        snapshot.flags.writeable = False
        self._data: np.ndarray = snapshot

        descriptive = sample_descriptives(self.data)
        self.type = dict(dist_type) if dist_type is not None else None
        self.assessments: Dict[str, Any] = {}
        self.n = descriptive["n"]
        self.mean = descriptive["mean"]
        self.median = descriptive["median"]
        self.std = descriptive["std"]
        self.var = descriptive["var"]
        self.skewness = descriptive["skewness"]
        self.excess_kurtosis = descriptive["excess_kurtosis"]
        self.mode = stats.mode(self.data)
        self.q1 = descriptive["q1"]
        self.q3 = descriptive["q3"]
        self.iqr = descriptive["iqr"]
        self.min = descriptive["min"]
        self.max = descriptive["max"]
        self.range = descriptive["range"]

    @property
    def data(self) -> np.ndarray:
        """Read-only defensive snapshot used by every assessment layer."""

        return self._data

    @property
    def kurtosis(self) -> float:
        """Deprecated alias for bias-corrected excess kurtosis."""

        warnings.warn(
            "Distribution.kurtosis is deprecated because it is ambiguous; "
            "use Distribution.excess_kurtosis.",
            DeprecationWarning,
            stacklevel=2,
        )
        return float(self.excess_kurtosis)
    
    def __repr__(self):
        return output_format(data=f"""
            Distribution Summary:
            count={self.n},
            type={self.type},
            stats:
            mean={self.mean},
            std={self.std},
            var={self.var},
            skewness={self.skewness},
            excess_kurtosis={self.excess_kurtosis},
            median={self.median},
            mode={self.mode},
            min={self.min},
            max={self.max},
            q1={self.q1},
            q3={self.q3},
            iqr={self.iqr},
            range={self.range}
        """,
        )

    def update_type(self, distribution_name: str, bool_result: bool, static_name: str, value: Any) -> None:
        """
        Update the legacy distribution-type dictionary.

        This compatibility API records goodness-of-fit or exact-distribution
        evidence. It is not an inferential method-selection contract.
        """
        warnings.warn(
            "Distribution.update_type() is deprecated; store structured "
            "assessment results in Distribution.assessments instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.type is None:
            self.type = {}
        self.type.update({distribution_name: bool_result, static_name: value})


#######################################
# 2. Clase Base Abstracta: DistributionValidator
#######################################
class DistributionValidator(ABC):
    """
    Clase base para validadores estadísticos. Permite inicializarse con un objeto Distribution o un numpy array.
    Realiza la validación temprana (validate_data) y rechaza la instancia si ésta falla.
    """
    def __init__(self, data: Union['Distribution', np.ndarray, Any]) -> None:
        # Reuse a canonical snapshot or normalize any compatible array-like.
        if isinstance(data, Distribution):
            self.distribution = data
        else:
            self.distribution = Distribution(data)
        
        # Validación temprana: si validate_data() falla, se rechaza la instancia.
        if not self.validate_data():
            raise ValueError(f"Validación de datos fallida para {self.__class__.__name__}")

    @abstractmethod
    def validate_data(self) -> bool:
        """Valida que los datos sean adecuados para la distribución."""
        pass

    @abstractmethod
    def fit_test(self, *args, **kwargs):
        """
        Método unificado para ejecutar el test de ajuste/validación.
        Se ejecuta solo si la validación de datos ha sido exitosa.
        """
        pass


class ContinuousDistributionValidator(DistributionValidator, ABC):
    @abstractmethod
    def evaluate_normality(self) -> Dict[str, Any]:
        """Realiza los tests de normalidad y retorna los resultados."""
        pass

    @abstractmethod
    def assign_weights(self):
        """Asigna pesos a cada test de normalidad según el tamaño de la muestra."""
        pass

    def fit_test(self):
        """
        En distribuciones continuas, fit_test() ejecuta el test de normalidad.
        Se asume que validate_data() fue exitosa.
        """
        resultados = self.evaluate_normality()
        return resultados


class NormalDistribution(ContinuousDistributionValidator):
    """
    Validador para datos que se espera sean de una distribución normal.
    Ejecuta varios tests de normalidad y actualiza el objeto Distribution.
    """
    def validate_data(self) -> bool:
        data = self.distribution.data
        if not (isinstance(data, np.ndarray) and (np.issubdtype(data.dtype, np.floating) or np.issubdtype(data.dtype, np.integer))):
            warnings.warn("Los datos no son de tipo float o int. La validación falla.")
            return False
        return True

    def evaluate_normality(self) -> Dict[str, Any]:
        data = self.distribution.data
        assessment = ShapeAssessment(alpha=0.05).assess(self.distribution)
        metrics = assessment.metrics
        resultados = {
            "assessment": assessment.to_dict(),
            "Shapiro": {
                "p_value": metrics.get("shapiro_p_value"),
                "alpha": metrics.get("alpha"),
                "rejects_exact_normality": metrics.get(
                    "shapiro_rejects_exact_normality"
                ),
            },
            "D'Agostino": {
                "p_value": metrics.get("dagostino_p_value"),
                "alpha": metrics.get("alpha"),
                "rejects_exact_normality": metrics.get(
                    "dagostino_rejects_exact_normality"
                ),
            },
            "shape": {
                "skewness": metrics.get("skewness"),
                "excess_kurtosis": metrics.get("excess_kurtosis"),
                "departure_magnitude": metrics.get("departure_magnitude"),
            },
            "QQ": self.evaluate_qq(data),
        }
        self.distribution.assessments["normality"] = assessment

        # Compatibility only: this boolean describes whether the available
        # formal tests rejected exact normality. SamplingRobustness and
        # MethodSelector do not consume it.
        exact_rejected = metrics.get("exact_normality_rejected")
        legacy_exact_normality = None if exact_rejected is None else not exact_rejected
        if self.distribution.type is None:
            self.distribution.type = {}
        self.distribution.type.update(
            {
                "Normal": legacy_exact_normality,
                "normality_results": resultados,
            }
        )
        return resultados

    def evaluate_qq(self, data):
        try:
            quantiles_theo, quantiles_emp = stats.probplot(data, dist="norm", fit=False)
            X = sm.add_constant(quantiles_theo)
            modelo = sm.OLS(quantiles_emp, X).fit()
            intercept, slope = modelo.params
            se_intercept, se_slope = modelo.bse
            df = len(quantiles_theo) - 2
            t_slope = (slope - 1) / se_slope
            t_intercept = (intercept - 0) / se_intercept
            p_slope = 2 * (1 - stats.t.cdf(np.abs(t_slope), df=df))
            p_intercept = 2 * (1 - stats.t.cdf(np.abs(t_intercept), df=df))
            return {
                'slope': slope,
                'intercept': intercept,
                'p_value_slope': p_slope,
                'p_value_intercept': p_intercept,
                'r_squared': modelo.rsquared
            }
        except Exception as e:
            warnings.warn("Error en evaluate_qq: " + str(e))
            return {'error': str(e)}

    def assign_weights(self):
        data = self.distribution.data
        try:
            n = len(data)
            if n >= 50:
                pesos = {'KS': 0.2, 'Shapiro': 0.2, "D'Agostino": 0.2, 'Anderson': 0.3, 'QQ': 0.1}
            else:
                pesos = {'KS': 0.25, 'Shapiro': 0.25, "D'Agostino": 0.2, 'Anderson': 0.15, 'QQ': 0.15}
        except Exception as e:
            warnings.warn("Error al asignar pesos: " + str(e))
            pesos = {}
        return pesos


class LognormalDistribution(ContinuousDistributionValidator):
    """
    Validador para datos que se espera sigan una distribución lognormal.
    Se transforma el dataset aplicando el logaritmo (por lo que requiere datos estrictamente positivos).
    """
    def validate_data(self):
        data = self.distribution.data
        if not (isinstance(data, np.ndarray) and np.all(data > 0)):
            warnings.warn("Los datos deben ser positivos para lognormal.")
            return False
        return True

    def evaluate_normality(self):
        data = self.distribution.data
        try:
            log_data = np.log(data)
        except Exception as e:
            warnings.warn("Error al aplicar log: " + str(e))
            self.distribution.update_type('Lognormal', False, 'normality_log_results', {'error': str(e)})
            return {'error': str(e)}
        try:
            # Se utiliza el evaluador normal sobre el logaritmo de los datos.
            evaluator = NormalDistribution(log_data)
            resultados = evaluator.evaluate_normality()
            self.distribution.update_type('Lognormal', resultados is not None, 'normality_log_results', resultados)
            return resultados
        except Exception as e:
            warnings.warn("Error en normalidad lognormal: " + str(e))
            self.distribution.update_type('Lognormal', False, 'normality_log_results', {'error': str(e)})
            return {'error': str(e)}

    def assign_weights(self):
        try:
            log_data = np.log(self.distribution.data)
            evaluator = NormalDistribution(log_data)
            return evaluator.assign_weights()
        except Exception as e:
            warnings.warn("Error al asignar pesos lognormal: " + str(e))
            return {}

    def fit_test(self):
        # En lognormal, se utiliza la prueba de normalidad sobre el logaritmo de los datos.
        return self.evaluate_normality()


#######################################
# 4. Validadores para Distribuciones Discretas
# hay que agregar binomial negativa e hypergeometric
#######################################
class DiscreteDistributionValidator(DistributionValidator, ABC):
    @abstractmethod
    def evaluate_goodness_of_fit(self, *args, **kwargs):
        """Realiza el test de bondad de ajuste para distribuciones discretas."""
        pass

    def fit_test(self, *args, **kwargs):
        """
        En distribuciones discretas, fit_test() ejecuta:
          - Primero, la prueba de bondad de ajuste.
          - Luego, si ésta es exitosa (según .type específico), evalúa la aproximación a la normal.
        """
        gof_results = self.evaluate_goodness_of_fit(*args, **kwargs)
        # Se asume que el test de bondad actualiza el .type correspondiente en Distribution.
        # Si la bondad de ajuste no es exitosa, se interrumpe el proceso.
        dist_name = self.__class__.__name__.replace('Distribution', '')
        if not self.distribution.type.get(dist_name, False):
            raise ValueError(f"Test de bondad de ajuste fallido en {self.__class__.__name__}, no se procede a normalidad.")
        # Se procede a evaluar la aproximación a la normal de forma condicional.
        normal_approx = self.evaluate_normal_approximation()
        return {'goodness_of_fit': gof_results, 'approx_normal': normal_approx}

    @abstractmethod
    def evaluate_normal_approximation(self):
        """
        Evalúa si la distribución discreta se aproxima a una normal.
        Por ejemplo, para Binomial se usa n*p*(1-p) >= 9 y para Poisson lambda >= 9.
        """
        pass


class BinomialDistribution(DiscreteDistributionValidator):
    """
    Validador para datos que se espera sigan una distribución binomial.
    Se ejecuta un test de bondad de ajuste y, si es exitoso, se evalúa la aproximación a la normal.
    """
    def validate_data(self):
        data = self.distribution.data
        # Deben ser enteros, no negativos y no mayores que el máximo observado
        if not (isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer) and np.all(data >= 0)):
            warnings.warn("Los datos deben ser enteros y no negativos para binomial.")
            return False
        return True

    def evaluate_goodness_of_fit(self, n=None, p=None):
        data = self.distribution.data
        try:
            # Si no se pasan n y p, estimar por momentos
            if n is None or p is None:
                params = self.estimate_parameters_moments()
                if 'error' in params:
                    raise ValueError(params['error'])
                n = int(round(params['n']))
                p = params['p']
            # Validar que los datos estén en el rango [0, n]
            if np.any((data < 0) | (data > n)):
                warnings.warn("Algunos datos están fuera del rango [0, n] estimado para binomial.")
            bins = np.arange(0, n + 2)
            observed, _ = np.histogram(data, bins=bins)
            expected = np.array([stats.binom.pmf(k, n, p) * len(data) for k in range(0, n + 1)])
            # Filtrar bins con esperado < 5 (regla estándar chi2)
            mask = expected >= 5
            if not np.any(mask):
                raise ValueError("Todos los bins esperados son menores a 5. No se puede aplicar chi2.")
            observed = observed[mask]
            expected = expected[mask]
            expected = expected * (observed.sum() / expected.sum())  # Ajustar para que sumen lo mism
            chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
            resultados = {'chi2': chi2, 'p_value': p_value, 'n': n, 'p': p}
        except Exception as e:
            warnings.warn("Error en bondad de ajuste binomial: " + str(e))
            resultados = {'error': str(e)}
        self.distribution.update_type('Binomial', resultados.get('p_value', 0) > 0.05, 'goodness_of_fit', resultados)
        return resultados

    def evaluate_normal_approximation(self):
        try:
            params = self.distribution.type.get('goodness_of_fit', {})
            n = params.get('n')
            p = params.get('p')
            if n is None or p is None:
                params = self.estimate_parameters_moments()
                n = params.get('n')
                p = params.get('p')
            var_approx = n * p * (1 - p)
            return bool(var_approx >= 9)
        except Exception as e:
            warnings.warn("Error en test de aproximación normal binomial: " + str(e))
            return False

    def estimate_parameters_moments(self):
        data = self.distribution.data
        try:
            m = np.mean(data)
            v = np.var(data, ddof=1)
            if m == 0:
                raise ValueError("Media cero, no se puede estimar p.")
            p_est = 1 - (v / m)
            if p_est <= 0 or p_est >= 1:
                raise ValueError("p_est fuera de (0,1).")
            n_est = m / p_est
            resultados = {'n': n_est, 'p': p_est}
        except Exception as e:
            warnings.warn("Error en estimación por momentos: " + str(e))
            resultados = {'error': str(e)}
        self.distribution.update_type('Binomial', 'error' not in resultados, 'moments_estimation', resultados)
        return resultados

    
class PoissonDistribution(DiscreteDistributionValidator):
    """
    Validador para datos que se espera sigan una distribución Poisson.
    Se ejecuta el test de bondad de ajuste y, de ser exitoso, se evalúa la aproximación a la normal.
    """
    def validate_data(self):
        data = self.distribution.data
        if not (isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer) and np.all(data >= 0)):
            warnings.warn("Los datos deben ser enteros y no negativos para Poisson.")
            return False
        return True

    def evaluate_goodness_of_fit(self):
        data = self.distribution.data
        try:
            lambda_val = np.mean(data)
            n = len(data)
            bins = np.arange(np.min(data), np.max(data) + 2)
            observed, _ = np.histogram(data, bins=bins)
            expected = np.array([stats.poisson.pmf(k, lambda_val) * n for k in bins[:-1]])
            # Filtrar bins con esperado < 5
            mask = expected >= 5
            if not np.any(mask):
                raise ValueError("Todos los bins esperados son menores a 5. No se puede aplicar chi2.")
            observed = observed[mask]
            expected = expected[mask]
            expected = expected * (observed.sum() / expected.sum())  # Ajustar para que sumen lo mismo
            chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
            resultados = {'chi2': chi2, 'p_value': p_value, 'lambda': lambda_val}
        except Exception as e:
            warnings.warn("Error en bondad de ajuste Poisson: " + str(e))
            resultados = {'error': str(e)}
        self.distribution.update_type('Poisson', resultados.get('p_value', 0) > 0.05, 'goodness_of_fit', resultados)
        return resultados

    def evaluate_normal_approximation(self):
        try:
            params = self.distribution.type.get('goodness_of_fit', {})
            lambda_val = params.get('lambda')
            if lambda_val is None:
                lambda_val = np.mean(self.distribution.data)
            return bool(lambda_val >= 9)
        except Exception as e:
            warnings.warn("Error en test de aproximación normal Poisson: " + str(e))
            return False





################################
#Proximamente distribuciones financieras 
############### empezando con Pareto  
# ################
