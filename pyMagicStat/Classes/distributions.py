from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import warnings
from lib.utils import plot_distribution_summary, output_format


#Pendiente ! no esta implementando output_format
#hay que poner bonito el distribution visualization 
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
    Clase que encapsula los datos y algunos estadísticos básicos.
    Se utiliza para almacenar y actualizar el estado de validación mediante update_type().
    """
    def __init__(self, data, dist_type=None):
        try:
            self.data = np.array(data)
        except Exception as e:
            raise ValueError("Error al convertir los datos a numpy array: " + str(e))
        
        # Estadísticos básicos
        self.type = dist_type
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.var = np.var(self.data)
        self.skewness = stats.skew(self.data)
        self.kurtosis = stats.kurtosis(self.data)
        self.median = np.median(self.data)
        self.mode = stats.mode(self.data)
        self.min = np.min(self.data)
        self.max = np.max(self.data)
        self.q1 = np.percentile(self.data, 25)
        self.q3 = np.percentile(self.data, 75)
        self.iqr = self.q3 - self.q1
        self.range = self.max - self.min

    def update_type(self, distribution_name, bool_result, static_name, value):
        """
        Actualiza el diccionario type con la validación o resultado obtenido.
        """
        if self.type is None:
            self.type = {}
        self.type.update({distribution_name: bool_result, static_name: value})

    def distribution_vizualization(self, title="Resumen de Distribución", x_label="Valor", y_label="Densidad", bins=30):
        """
        Visualiza la distribución de los datos utilizando la función genérica de visualización.
        
        Parámetros:
          title: str, opcional
              Título general de la figura.
          x_label: str, opcional
              Etiqueta del eje X.
          y_label: str, opcional
              Etiqueta del eje Y.
          bins: int, opcional
              Número de bins para el histograma.
        """
        plot_distribution_summary(
            data=self.data,
            distribution_type=self.type,
            title=title,
            x_label=x_label,
            y_label=y_label,
            bins=bins
        )


#######################################
# 2. Clase Base Abstracta: DistributionValidator
#######################################
class DistributionValidator(ABC):
    """
    Clase base para validadores estadísticos. Permite inicializarse con un objeto Distribution o un numpy array.
    Realiza la validación temprana (validate_data) y rechaza la instancia si ésta falla.
    """
    def __init__(self, data):
        # Permite recibir un objeto Distribution o un numpy array.
        if isinstance(data, Distribution):
            self.distribution = data
        elif isinstance(data, np.ndarray):
            self.distribution = Distribution(data)
        else:
            raise ValueError("Los datos deben ser un objeto Distribution o un numpy array.")
        
        # Validación temprana: si validate_data() falla, se rechaza la instancia.
        if not self.validate_data():
            raise ValueError(f"Validación de datos fallida para {self.__class__.__name__}")

    @abstractmethod
    def validate_data(self):
        """Valida que los datos sean adecuados para la distribución."""
        pass

    @abstractmethod
    def fit_test(self, *args, **kwargs):
        """
        Método unificado para ejecutar el test de ajuste/validación.
        Se ejecuta solo si la validación de datos ha sido exitosa.
        """
        pass


#######################################
# 3. Validadores para Distribuciones Continuas
#######################################
class ContinuousDistributionValidator(DistributionValidator, ABC):
    @abstractmethod
    def evaluate_normality(self):
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
    def validate_data(self):
        data = self.distribution.data
        if not (isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating)):
            warnings.warn("Los datos no son de tipo float. La validación falla.")
            return False
        return True

    def evaluate_normality(self):
        data = self.distribution.data
        resultados = {}
        try:
            # Test Kolmogorov-Smirnov
            mu, sigma = np.mean(data), np.std(data, ddof=1)
            stat_ks, p_ks = stats.kstest(data, 'norm', args=(mu, sigma))
            resultados['KS'] = {'statistic': stat_ks, 'p_value': p_ks}
        except Exception as e:
            warnings.warn("Error en KS: " + str(e))
            resultados['KS'] = {'error': str(e)}
        try:
            # Test Shapiro-Wilk
            stat_shapiro, p_shapiro = stats.shapiro(data)
            resultados['Shapiro'] = {'statistic': stat_shapiro, 'p_value': p_shapiro}
        except Exception as e:
            warnings.warn("Error en Shapiro-Wilk: " + str(e))
            resultados['Shapiro'] = {'error': str(e)}
        try:
            # Test D'Agostino-Pearson
            stat_dagostino, p_dagostino = stats.normaltest(data)
            resultados["D'Agostino"] = {'statistic': stat_dagostino, 'p_value': p_dagostino}
        except Exception as e:
            warnings.warn("Error en D'Agostino-Pearson: " + str(e))
            resultados["D'Agostino"] = {'error': str(e)}
        try:
            # Test Anderson-Darling
            ad_result = stats.anderson(data, dist='norm')
            resultados['Anderson'] = {
                'statistic': ad_result.statistic,
                'critical_values': ad_result.critical_values.tolist(),
                'significance_levels': ad_result.significance_level.tolist()
            }
        except Exception as e:
            warnings.warn("Error en Anderson-Darling: " + str(e))
            resultados['Anderson'] = {'error': str(e)}
        try:
            # Análisis Q-Q
            resultados['QQ'] = self.evaluate_qq(data)
        except Exception as e:
            warnings.warn("Error en análisis Q-Q: " + str(e))
            resultados['QQ'] = {'error': str(e)}
        
        # Criterio simple: se asume normal si p_values de KS, Shapiro y D'Agostino son > 0.05.
        try:
            is_normal = (resultados.get('KS', {}).get('p_value', 0) > 0.05 and 
                         resultados.get('Shapiro', {}).get('p_value', 0) > 0.05 and 
                         resultados.get("D'Agostino", {}).get('p_value', 0) > 0.05)
        except Exception as e:
            warnings.warn("Error al evaluar normalidad: " + str(e))
            is_normal = False
        
        # Actualiza el objeto Distribution.
        self.distribution.update_type('Normal', is_normal, 'normality_results', resultados)
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
                'p_value_intercept': p_intercept
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
        if not self.distribution.type.get(self.__class__.__name__, False):
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
        if not (isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer)):
            warnings.warn("Los datos deben ser enteros para binomial.")
            return False
        return True

    def evaluate_goodness_of_fit(self, n=None, p=None):
        data = self.distribution.data
        try:
            if n is not None and p is not None:
                bins = np.arange(0, n + 2)  # Bins de 0 a n
                observed, _ = np.histogram(data, bins=bins)
                expected = np.array([stats.binom.pmf(k, n, p) * len(data) for k in range(0, n + 1)])
                chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected, sum_check=False)
                resultados = {'chi2': chi2, 'p_value': p_value}
            else:
                resultados = {'chi2': 0.0, 'p_value': 1.0}
        except Exception as e:
            warnings.warn("Error en bondad de ajuste binomial: " + str(e))
            resultados = {'error': str(e)}
        # Actualiza el estado en Distribution (usamos el nombre de la clase como key)
        self.distribution.update_type('Binomial', resultados.get('p_value', 0) > 0.05, 'goodness_of_fit', resultados)
        return resultados

    def evaluate_normal_approximation(self):
        # Se evalúa la aproximación normal con la regla n*p*(1-p) >= 9
        try:
            # Si no se pasaron parámetros, se estima por momentos
            if 'moments_estimation' not in self.distribution.type:
                params = self.estimate_parameters_moments()
            else:
                params = self.distribution.type.get('moments_estimation')
            n_est = params['n']
            p_est = params['p']
            var_approx = n_est * p_est * (1 - p_est)
            return var_approx >= 9
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
        if not (isinstance(data, np.ndarray) and 
                np.issubdtype(data.dtype, np.integer) and 
                np.all(data >= 0)):
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
            chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected, sum_check=False)
            resultados = {'chi2': chi2, 'p_value': p_value}
        except Exception as e:
            warnings.warn("Error en bondad de ajuste Poisson: " + str(e))
            resultados = {'error': str(e)}
        self.distribution.update_type('Poisson', resultados.get('p_value', 0) > 0.05, 'goodness_of_fit', resultados)
        return resultados

    def evaluate_normal_approximation(self):
        # Criterio para aproximación normal en Poisson: lambda >= 9.
        try:
            lambda_val = np.mean(self.distribution.data)
            return lambda_val >= 9
        except Exception as e:
            warnings.warn("Error en test de aproximación normal Poisson: " + str(e))
            return False

