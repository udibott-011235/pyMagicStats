import unittest
import numpy as np
import logging
import warnings
from scipy.stats import binom, poisson

logging.basicConfig(filename='test_log.txt', filemode="w", level=logging.INFO, format="%(message)s")

# Importamos el módulo a testear (asegúrate de que distributions.py esté en el path)
import pyMagicStat.Classes.distributions as distributions

# Configuración del log: se crea un archivo 'test_log.txt' en modo escritura.

class TestDistributions(unittest.TestCase):
    def setUp(self):
        # Opcional: ignorar ciertos warnings para que no ensucien la salida
        warnings.simplefilter("ignore", RuntimeWarning)

    def log_result(self, descripcion, exito):
        # Función auxiliar para escribir en el log el supuesto evaluado
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
            # Se proveen los parámetros para el test de bondad
            gof_results = validator.evaluate_goodness_of_fit(n=10, p=0.5)
            is_binomial = validator.distribution.type.get('Binomial', False)
            self.assertTrue(is_binomial, "La bondad de ajuste para binomial falló.")
            approx_normal = validator.evaluate_normal_approximation()
            self.assertFalse(approx_normal, "La aproximación normal para binomial es incorrecta (se esperaba False).")
            self.log_result(descripcion, True)
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
            self.log_result(descripcion, True)
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

if __name__ == '__main__':
    unittest.main()
