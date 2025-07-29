import unittest
import numpy as np
import logging
import warnings
from scipy.stats import binom, poisson

logging.basicConfig(filename='test_log.txt', filemode="w", level=logging.INFO, format="%(message)s")

# Importamos el módulo a testear (asegúrate de que distributions.py esté en el path)
import pyMagicStat.Classes.distributions as distributions

class TestDistributions(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", RuntimeWarning)

    def log_result(self, descripcion, exito):
        mensaje = (
            f"Supuesto: {descripcion}\n"
            f"Resultado: {'Aceptado' if exito else 'Rechazado'}\n"
            f"{'-'*50}"
        )
        logging.info(mensaje)

    def test_normal_distribution_valid(self):
        descripcion = (
            "NormalDistribution con datos normales válidos. "
            "Datos generados: 1000 muestras usando np.random.normal(0, 1, 1000). "
            "Se evalúa que los tests de normalidad retornen p_value > 0.05."
        )
        try:
            data = np.random.normal(0, 1, 1000).astype(float)
            validator = distributions.NormalDistribution(data)
            resultados = validator.evaluate_normality()
            is_normal = validator.distribution.type.get('Normal', False)
            self.assertTrue(is_normal, "La distribución no fue detectada como normal.")
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_normal_distribution_invalid(self):
        descripcion = (
            "NormalDistribution con datos no normales (distribución exponencial). "
            "Datos generados: 1000 muestras usando np.random.exponential(1, 1000). "
            "Se evalúa que los tests de normalidad retornen p_value <= 0.05."
        )
        try:
            data = np.random.exponential(1, 1000).astype(float)
            validator = distributions.NormalDistribution(data)
            resultados = validator.evaluate_normality()
            is_normal = validator.distribution.type.get('Normal', False)
            self.assertFalse(is_normal, "La distribución erróneamente fue detectada como normal.")
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_lognormal_distribution_valid(self):
        descripcion = (
            "LognormalDistribution con datos lognormales válidos. "
            "Datos generados: 1000 muestras usando np.random.lognormal(0, 1, 1000). "
            "Se evalúa que al aplicar el logaritmo se detecte la normalidad en los datos transformados."
        )
        try:
            data = np.random.lognormal(0, 1, 1000).astype(float)
            validator = distributions.LognormalDistribution(data)
            resultados = validator.fit_test()  # Se ejecuta el test sobre el logaritmo
            is_lognormal = validator.distribution.type.get('Lognormal', False)
            self.assertTrue(is_lognormal, "La distribución lognormal no fue detectada correctamente.")
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_lognormal_distribution_invalid(self):
        descripcion = (
            "LognormalDistribution con datos inválidos (contiene valores no positivos). "
            "Datos generados: array con valores [-1, 0, 1, 2, 3]. "
            "Se evalúa que la validación falle al tener valores no estrictamente positivos."
        )
        try:
            data = np.array([-1, 0, 1, 2, 3], dtype=float)
            with self.assertRaises(ValueError):
                _ = distributions.LognormalDistribution(data)
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_binomial_distribution_valid(self):
        descripcion = (
            "BinomialDistribution con datos binomiales válidos. "
            "Datos generados: 1000 muestras usando binom.rvs(n=10, p=0.5, size=1000). "
            "Se evalúa el test de bondad de ajuste (se espera p_value > 0.05) y "
            "la estimación de parámetros; la aproximación normal se espera que sea False (ya que n*p*(1-p) < 9)."
        )
        try:
            data = binom.rvs(n=10, p=0.5, size=1000)
            validator = distributions.BinomialDistribution(data)
            gof_results = validator.evaluate_goodness_of_fit(n=10, p=0.5)
            is_binomial = validator.distribution.type.get('Binomial', False)
            self.assertTrue(is_binomial, "La bondad de ajuste para binomial falló.")
            approx_normal = validator.evaluate_normal_approximation()
            self.assertFalse(approx_normal, "La aproximación normal para binomial es incorrecta (se esperaba False).")
            self.log_result(descripcion + f" | Aproximación normal: {approx_normal}", True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_binomial_distribution_invalid(self):
        descripcion = (
            "BinomialDistribution con datos inválidos (valores no enteros). "
            "Datos generados: array de floats [0.5, 1.5, 2.5]. "
            "Se evalúa que la validación falle al requerir datos enteros."
        )
        try:
            data = np.array([0.5, 1.5, 2.5])
            with self.assertRaises(ValueError):
                _ = distributions.BinomialDistribution(data)
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_poisson_distribution_valid(self):
        descripcion = (
            "PoissonDistribution con datos Poisson válidos. "
            "Datos generados: 1000 muestras usando poisson.rvs(mu=10, size=1000). "
            "Se evalúa el test de bondad de ajuste (se espera p_value > 0.05) y "
            "la aproximación normal (se espera True, ya que lambda >= 9)."
        )
        try:
            data = poisson.rvs(mu=10, size=1000)
            validator = distributions.PoissonDistribution(data)
            gof_results = validator.evaluate_goodness_of_fit()
            is_poisson = validator.distribution.type.get('Poisson', False)
            self.assertTrue(is_poisson, "La bondad de ajuste para Poisson falló.")
            approx_normal = validator.evaluate_normal_approximation()
            self.assertTrue(approx_normal, "La aproximación normal para Poisson debería ser True.")
            self.log_result(descripcion + f" | Aproximación normal: {approx_normal}", True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_poisson_distribution_invalid(self):
        descripcion = (
            "PoissonDistribution con datos inválidos (contiene valores negativos). "
            "Datos generados: array con valores [-1, 0, 1, 2, 3]. "
            "Se evalúa que la validación falle al tener valores negativos."
        )
        try:
            data = np.array([-1, 0, 1, 2, 3], dtype=int)
            with self.assertRaises(ValueError):
                _ = distributions.PoissonDistribution(data)
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    # BinomialDistribution absurd/edge cases
    def test_binomial_gof_out_of_range(self):
        descripcion = "BinomialDistribution: datos fuera de rango [0, n], se espera advertencia y posible p_value bajo."
        try:
            data = np.concatenate([binom.rvs(n=8, p=0.5, size=500), np.array([20, 25, 30])])
            validator = distributions.BinomialDistribution(data)
            result = validator.evaluate_goodness_of_fit(n=8, p=0.5)
            self.assertTrue('p_value' in result)
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_binomial_gof_low_expected(self):
        descripcion = "BinomialDistribution: datos con bins esperados < 5, se espera ValueError."
        try:
            data = np.full(100, 0)
            validator = distributions.BinomialDistribution(data)
            with self.assertRaises(ValueError):
                validator.evaluate_goodness_of_fit(n=1, p=0.01)
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_binomial_gof_non_integer(self):
        descripcion = "BinomialDistribution: datos no enteros, se espera ValueError en validación."
        try:
            data = np.array([0.1, 1.2, 2.5])
            with self.assertRaises(ValueError):
                distributions.BinomialDistribution(data)
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    # PoissonDistribution absurd/edge cases
    def test_poisson_gof_low_expected(self):
        descripcion = "PoissonDistribution: datos con bins esperados < 5, se espera ValueError."
        try:
            data = np.full(50, 0)
            validator = distributions.PoissonDistribution(data)
            with self.assertRaises(ValueError):
                validator.evaluate_goodness_of_fit()
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_poisson_gof_negative(self):
        descripcion = "PoissonDistribution: datos negativos, se espera ValueError en validación."
        try:
            data = np.array([-1, 0, 1, 2, 3], dtype=int)
            with self.assertRaises(ValueError):
                distributions.PoissonDistribution(data)
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_poisson_gof_non_integer(self):
        descripcion = "PoissonDistribution: datos no enteros, se espera ValueError en validación."
        try:
            data = np.array([0.5, 1.5, 2.5])
            with self.assertRaises(ValueError):
                distributions.PoissonDistribution(data)
            self.log_result(descripcion, True)
        except Exception as e:
            self.log_result(descripcion + f" - Error: {e}", False)
            raise

    def test_binomial_data_range(self):
        descripcion = "Verifica que los datos binomiales estén en el rango [0, n]."
        n, p = 10, 0.3
        data = binom.rvs(n=n, p=p, size=1000)
        in_range = np.all((data >= 0) & (data <= n))
        self.assertTrue(in_range, "Datos binomiales fuera de rango.")
        self.log_result(descripcion, in_range)

    def test_poisson_data_range(self):
        descripcion = "Verifica que los datos Poisson sean >= 0."
        mu = 5
        data = poisson.rvs(mu=mu, size=1000)
        in_range = np.all(data >= 0)
        self.assertTrue(in_range, "Datos Poisson fuera de rango.")
        self.log_result(descripcion, in_range)

    def test_binomial_mean_variance(self):
        descripcion = "Verifica media y varianza binomial vs esperado."
        n, p = 20, 0.4
        data = binom.rvs(n=n, p=p, size=5000)
        mean_expected = n * p
        var_expected = n * p * (1 - p)
        mean_actual = np.mean(data)
        var_actual = np.var(data)
        mean_close = np.isclose(mean_actual, mean_expected, atol=0.2)
        var_close = np.isclose(var_actual, var_expected, atol=0.2)
        self.assertTrue(mean_close and var_close, "Media/varianza binomial incorrecta.")
        self.log_result(descripcion, mean_close and var_close)

    def test_poisson_mean_variance(self):
        descripcion = "Verifica media y varianza Poisson vs esperado."
        mu = 7
        data = poisson.rvs(mu=mu, size=5000)
        mean_actual = np.mean(data)
        var_actual = np.var(data)
        mean_close = np.isclose(mean_actual, mu, atol=0.2)
        var_close = np.isclose(var_actual, mu, atol=0.2)
        self.assertTrue(mean_close and var_close, "Media/varianza Poisson incorrecta.")
        self.log_result(descripcion, mean_close and var_close)

    def test_binomial_ks_test_consistency(self):
        descripcion = "Consistencia Kolmogorov-Smirnov para binomial."
        n, p = 15, 0.5
        data = binom.rvs(n=n, p=p, size=1000)
        validator = distributions.BinomialDistribution(data)
        gof = validator.evaluate_goodness_of_fit(n=n, p=p)
        p_value = gof.get('p_value', 0)
        self.assertTrue(p_value > 0.05, "KS test inconsistente para binomial.")
        self.log_result(descripcion, p_value > 0.05)

    def test_poisson_ks_test_consistency(self):
        descripcion = "Consistencia Kolmogorov-Smirnov para Poisson."
        mu = 8
        data = poisson.rvs(mu=mu, size=1000)
        validator = distributions.PoissonDistribution(data)
        gof = validator.evaluate_goodness_of_fit()
        p_value = gof.get('p_value', 0)
        self.assertTrue(p_value > 0.05, "KS test inconsistente para Poisson.")
        self.log_result(descripcion, p_value > 0.05)

    def test_combined_lognormal_normal(self):
        descripcion = "Combinación de lognormal y normal, verifica suma de probabilidades."
        data_ln = np.random.lognormal(0, 1, 500)
        data_n = np.random.normal(0, 1, 500)
        combined = np.concatenate([data_ln, data_n])
        validator_ln = distributions.LognormalDistribution(data_ln)
        validator_n = distributions.NormalDistribution(data_n)
        ln_prob = validator_ln.distribution.type.get('Lognormal', 0)
        n_prob = validator_n.distribution.type.get('Normal', 0)
        total_prob = ln_prob + n_prob
        self.assertTrue(0.9 < total_prob <= 1.1, "Suma de probabilidades fuera de rango.")
        self.log_result(descripcion, 0.9 < total_prob <= 1.1)

    def test_custom_binomial_parameters(self):
        descripcion = "Genera binomial con parámetros personalizados y verifica comportamiento."
        n, p = 25, 0.7
        data = binom.rvs(n=n, p=p, size=2000)
        validator = distributions.BinomialDistribution(data)
        gof = validator.evaluate_goodness_of_fit(n=n, p=p)
        mean_actual = np.mean(data)
        mean_expected = n * p
        self.assertTrue(np.isclose(mean_actual, mean_expected, atol=0.5), "Media binomial personalizada incorrecta.")
        self.log_result(descripcion, np.isclose(mean_actual, mean_expected, atol=0.5))

    def test_custom_poisson_parameters(self):
        descripcion = "Genera Poisson con parámetros personalizados y verifica comportamiento."
        mu = 12
        data = poisson.rvs(mu=mu, size=2000)
        validator = distributions.PoissonDistribution(data)
        gof = validator.evaluate_goodness_of_fit()
        mean_actual = np.mean(data)
        self.assertTrue(np.isclose(mean_actual, mu, atol=0.5), "Media Poisson personalizada incorrecta.")
        self.log_result(descripcion, np.isclose(mean_actual, mu, atol=0.5))

if __name__ == "__main__":
    unittest.main()