from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import warnings
from numbers import Integral, Real
from typing import Any, Dict, Optional, Union
from pyMagicStat.utils.utils import output_format
from pyMagicStat.assumptions import ShapeAssessment
from pyMagicStat.distributions._discrete_gof import (
    pearson_gof_result,
    point_cell,
    unavailable_result,
    upper_tail_cell,
)
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

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore the immutable snapshot contract after Python reconstruction."""

        snapshot = state.get("_data", state.get("data"))
        if not isinstance(snapshot, np.ndarray):
            raise TypeError("Distribution state must contain an ndarray snapshot")
        self.__dict__.update(state)
        self.__dict__.pop("data", None)
        self._data = snapshot
        self._data.flags.writeable = False

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
        if len(data) <= 2:
            return {
                "status": "not_assessed",
                "reason": "Q-Q regression requires at least three observations.",
            }
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
            result = {
                "status": "error",
                "decision": "error",
                "hypothesis": "Exact Gaussianity of log(data)",
                "evaluated_variable": "log(data)",
                "alpha": 0.05,
                "reason": str(e),
            }
            return self._store_lognormality_result(result)
        try:
            # Se utiliza el evaluador normal sobre el logaritmo de los datos.
            evaluator = NormalDistribution(log_data)
            resultados = evaluator.evaluate_normality()
            assessment = evaluator.distribution.assessments["normality"]
            exact_rejected = assessment.metrics.get("exact_normality_rejected")
            if exact_rejected is True:
                decision = "reject"
                reason = "At least one formal test rejects exact Gaussianity of log(data)."
            elif exact_rejected is False:
                decision = "fail_to_reject"
                reason = (
                    "Available formal tests do not reject exact Gaussianity of "
                    "log(data); this does not demonstrate that the original data "
                    "are lognormal."
                )
            else:
                decision = "not_assessed"
                reason = "Exact Gaussianity of log(data) could not be assessed."

            result = dict(resultados)
            result.update(
                {
                    "status": decision,
                    "decision": decision,
                    "hypothesis": "Exact Gaussianity of log(data)",
                    "evaluated_variable": "log(data)",
                    "alpha": float(assessment.metrics.get("alpha", 0.05)),
                    "reason": reason,
                }
            )
            return self._store_lognormality_result(result)
        except Exception as e:
            warnings.warn("Error en normalidad lognormal: " + str(e))
            result = {
                "status": "error",
                "decision": "error",
                "hypothesis": "Exact Gaussianity of log(data)",
                "evaluated_variable": "log(data)",
                "alpha": 0.05,
                "reason": str(e),
            }
            return self._store_lognormality_result(result)

    def _store_lognormality_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        self.distribution.assessments["lognormality"] = result
        legacy_value = {
            "reject": False,
            "fail_to_reject": True,
            "not_assessed": None,
            "error": None,
        }[result["decision"]]
        if self.distribution.type is None:
            self.distribution.type = {}
        self.distribution.type.update(
            {
                "Lognormal": legacy_value,
                "normality_log_results": result,
            }
        )
        return result

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
        En distribuciones discretas, fit_test() ejecuta primero el GOF y sólo
        continúa cuando su decisión estructurada es ``fail_to_reject``.
        """
        gof_results = self.evaluate_goodness_of_fit(*args, **kwargs)
        structured = self.distribution.assessments.get("goodness_of_fit", gof_results)
        if structured.get("decision") != "fail_to_reject":
            reason = structured.get("reason") or (
                "the structured goodness-of-fit decision was "
                f"{structured.get('decision')!r}"
            )
            raise ValueError(
                f"Goodness-of-fit did not permit normal approximation for "
                f"{self.__class__.__name__}: status={structured.get('status')!r}, "
                f"decision={structured.get('decision')!r}; {reason}"
            )
        # Se procede a evaluar la aproximación a la normal de forma condicional.
        normal_approx = self.evaluate_normal_approximation()
        return {'goodness_of_fit': gof_results, 'approx_normal': normal_approx}

    def _store_gof_result(
        self,
        distribution_name: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Store the canonical GOF result and update compatibility mirrors."""

        self.distribution.assessments["goodness_of_fit"] = result
        decision = result.get("decision")
        legacy_value = (
            True
            if decision == "fail_to_reject"
            else False if decision == "reject" else None
        )
        if self.distribution.type is None:
            self.distribution.type = {}
        self.distribution.type.update(
            {
                distribution_name: legacy_value,
                "goodness_of_fit": result,
            }
        )
        return result

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
        alpha = 0.05
        hypothesis = "Pearson chi-square goodness-of-fit for the binomial model"

        if n is None:
            parameters = {
                "n": {"value": None, "source": "required_not_provided"},
                "p": {
                    "value": p,
                    "source": "provided_but_not_used_without_n" if p is not None else "not_provided",
                },
            }
            result = unavailable_result(
                status="not_assessed",
                hypothesis=hypothesis,
                alpha=alpha,
                parameter_count_estimated=0,
                parameters=parameters,
                observed_total=len(data),
                reason=(
                    "n is a required structural parameter for a valid binomial "
                    "chi-square goodness-of-fit assessment."
                ),
                legacy_values={"n": None, "p": p},
            )
            return self._store_gof_result("Binomial", result)

        if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0:
            parameters = {
                "n": {"value": n, "source": "invalid_provided"},
                "p": {"value": p, "source": "provided" if p is not None else "not_provided"},
            }
            result = unavailable_result(
                status="not_assessed",
                hypothesis=hypothesis,
                alpha=alpha,
                parameter_count_estimated=0,
                parameters=parameters,
                observed_total=len(data),
                reason="n must be a positive integer for binomial goodness-of-fit.",
                legacy_values={"n": n, "p": p},
            )
            return self._store_gof_result("Binomial", result)

        n = int(n)
        if np.any(data > n):
            parameters = {
                "n": {"value": n, "source": "provided"},
                "p": {"value": p, "source": "provided" if p is not None else "not_provided"},
            }
            result = unavailable_result(
                status="not_assessed",
                hypothesis=hypothesis,
                alpha=alpha,
                parameter_count_estimated=0,
                parameters=parameters,
                observed_total=len(data),
                reason="All observations must belong to the binomial support [0, n].",
                legacy_values={"n": n, "p": p},
            )
            return self._store_gof_result("Binomial", result)

        parameter_count_estimated = 0
        p_source = "provided"
        if p is None:
            p = float(np.mean(data) / n)
            parameter_count_estimated = 1
            p_source = "estimated_from_sample_mean_given_n"

        if (
            isinstance(p, (bool, np.bool_))
            or not isinstance(p, Real)
            or not np.isfinite(float(p))
            or not 0.0 < float(p) < 1.0
        ):
            parameters = {
                "n": {"value": n, "source": "provided"},
                "p": {"value": p, "source": f"invalid_{p_source}"},
            }
            result = unavailable_result(
                status="not_assessed",
                hypothesis=hypothesis,
                alpha=alpha,
                parameter_count_estimated=parameter_count_estimated,
                parameters=parameters,
                observed_total=len(data),
                reason="p must be strictly between 0 and 1 for binomial goodness-of-fit.",
                legacy_values={"n": n, "p": p},
            )
            return self._store_gof_result("Binomial", result)

        p = float(p)
        parameters = {
            "n": {"value": n, "source": "provided"},
            "p": {"value": p, "source": p_source},
        }
        try:
            support = np.arange(n + 1, dtype=int)
            observed = np.bincount(data.astype(int), minlength=n + 1)
            expected = stats.binom.pmf(support, n, p) * len(data)
            cells = [
                point_cell(k, observed[k], expected[k])
                for k in range(n + 1)
            ]
            result = pearson_gof_result(
                cells=cells,
                hypothesis=hypothesis,
                alpha=alpha,
                parameter_count_estimated=parameter_count_estimated,
                parameters=parameters,
                legacy_values={"n": n, "p": p},
            )
        except Exception as e:
            warnings.warn("Error en bondad de ajuste binomial: " + str(e))
            result = unavailable_result(
                status="error",
                hypothesis=hypothesis,
                alpha=alpha,
                parameter_count_estimated=parameter_count_estimated,
                parameters=parameters,
                observed_total=len(data),
                reason=str(e),
                legacy_values={"n": n, "p": p},
            )
        return self._store_gof_result("Binomial", result)

    def evaluate_normal_approximation(self):
        try:
            assessment = self.distribution.assessments.get("goodness_of_fit", {})
            parameters = assessment.get("parameters", {})
            n = parameters.get("n", {}).get("value")
            p = parameters.get("p", {}).get("value")
            if n is None or p is None:
                raise ValueError("A structured GOF result with n and p is required.")
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
        self.distribution.assessments["moments_estimation"] = resultados
        if self.distribution.type is None:
            self.distribution.type = {}
        self.distribution.type["moments_estimation"] = resultados
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
        alpha = 0.05
        hypothesis = "Pearson chi-square goodness-of-fit for the Poisson model"
        lambda_val = float(np.mean(data))
        parameters = {
            "lambda": {
                "value": lambda_val,
                "source": "estimated_from_sample_mean",
            }
        }
        try:
            sample_size = len(data)
            maximum = int(np.max(data))
            observed = np.bincount(data.astype(int), minlength=maximum + 1)
            support = np.arange(maximum + 1, dtype=int)
            expected = stats.poisson.pmf(support, lambda_val) * sample_size
            cells = [
                point_cell(k, observed[k], expected[k])
                for k in range(maximum + 1)
            ]
            cells.append(
                upper_tail_cell(
                    maximum + 1,
                    0,
                    stats.poisson.sf(maximum, lambda_val) * sample_size,
                )
            )
            result = pearson_gof_result(
                cells=cells,
                hypothesis=hypothesis,
                alpha=alpha,
                parameter_count_estimated=1,
                parameters=parameters,
                legacy_values={"lambda": lambda_val},
            )
        except Exception as e:
            warnings.warn("Error en bondad de ajuste Poisson: " + str(e))
            result = unavailable_result(
                status="error",
                hypothesis=hypothesis,
                alpha=alpha,
                parameter_count_estimated=1,
                parameters=parameters,
                observed_total=len(data),
                reason=str(e),
                legacy_values={"lambda": lambda_val},
            )
        return self._store_gof_result("Poisson", result)

    def evaluate_normal_approximation(self):
        try:
            assessment = self.distribution.assessments.get("goodness_of_fit", {})
            lambda_val = (
                assessment.get("parameters", {})
                .get("lambda", {})
                .get("value")
            )
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
