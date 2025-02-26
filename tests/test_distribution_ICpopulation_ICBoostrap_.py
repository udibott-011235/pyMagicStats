import pytest
import numpy as np
import logging
from pyMagicStat.Classes.nonParametricHTest import (
    kruskalWallisTest,
    BootstrapMeanCI,
    BootstrapMedianCI,
    BootstrapVarianceCI,
    BootstrapProportionCI
)
from pyMagicStat.Classes.orchestrator import ExperimentationIteration

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("test_log.log")]
)

def flush_logs():
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()

# --- Fixture modificada para sample_data ---
@pytest.fixture
def sample_data():
    np.random.seed(42)
    group_1 = np.random.normal(50, 10, 30)
    group_2 = np.random.normal(55, 10, 30)
    group_3 = np.random.normal(60, 10, 30)
    # Se aumenta la diferencia del grupo 4 para favorecer el aumento de R² al remover otros grupos
    group_4 = np.random.normal(90, 10, 30)
    return [group_1, group_2, group_3, group_4]

@pytest.fixture
def extended_bootstrap_data():
    return {
        "normal": np.random.normal(50, 10, 1000),
        "exponential": np.random.exponential(10, 1000),
        "gamma": np.random.gamma(2, 2, 1000),
        "uniform": np.random.uniform(0, 100, 1000),
        "bimodal": np.concatenate([np.random.normal(30, 5, 500), np.random.normal(70, 5, 500)]),
        "lognormal": np.random.lognormal(3, 1, 1000),
        "poisson": np.random.poisson(5, 1000)
    }

# ------------------- Tests para Kruskal-Wallis ------------------- #
def test_kruskal_invalid_groups():
    """Debe lanzar un error si hay menos de 2 grupos."""
    logging.info("Ejecutando test_kruskal_invalid_groups")
    flush_logs()
    with pytest.raises(ValueError):
        kruskalWallisTest(np.random.normal(50, 10, 30))

def test_kruskal_different_means(sample_data):
    """Si los grupos tienen medias distintas, el p-value debería ser bajo."""
    logging.info("Generando datos para test Kruskal-Wallis con diferentes medias")
    for i, group in enumerate(sample_data):
        logging.info(f"Datos del grupo {i+1}: {group}")
    flush_logs()
    
    test = kruskalWallisTest(*sample_data)
    result = test.run_test()
    
    logging.info(f"Resultados: {result}")
    flush_logs()
    assert result['p_value'] < 0.05, "El test debería detectar diferencias."

# ------------------- Tests para Bootstrap CI ------------------- #
def test_extended_bootstrap_ci(extended_bootstrap_data):
    """Prueba Bootstrap para múltiples distribuciones extendidas."""
    for name, data in extended_bootstrap_data.items():
        logging.info(f"Probando Bootstrap CI en distribución {name}")
        flush_logs()
        mean_ci = BootstrapMeanCI(data)
        median_ci = BootstrapMedianCI(data)
        variance_ci = BootstrapVarianceCI(data)
        
        mean_result = mean_ci.calculate_interval()
        median_result = median_ci.calculate_interval()
        variance_result = variance_ci.calculate_interval()
        
        logging.info(f"{name} - Mean CI: {mean_result}")
        logging.info(f"{name} - Median CI: {median_result}")
        logging.info(f"{name} - Variance CI: {variance_result}")
        flush_logs()
        
        # Prueba para la media y la varianza (se asume que output_format retorna escalares)
        assert mean_result['ub'] > mean_result['lb'], "Intervalo de confianza inválido para la media"
        assert variance_result['ub'] > variance_result['lb'], "Intervalo de confianza inválido para la varianza"
        
        # Para la mediana, se verifica si el valor está empaquetado en una tupla y se desempaqueta
        median_lb = median_result['lb'][0] if isinstance(median_result['lb'], (tuple, list)) else median_result['lb']
        median_ub = median_result['ub'][0] if isinstance(median_result['ub'], (tuple, list)) else median_result['ub']
        assert median_ub > median_lb, "Intervalo de confianza inválido para la mediana"

# ------------------- Tests para ExperimentationIteration ------------------- #
def test_experimentation_iteration(sample_data):
    """Validar que el iterador elimina grupos hasta alcanzar el R² deseado."""
    logging.info("Ejecutando iteración experimental para optimizar R²")
    flush_logs()
    test_obj = kruskalWallisTest(*sample_data)
    exp_iter = ExperimentationIteration(test_obj, r2_target=0.75, max_iterations=10)
    history = exp_iter.run()
    logging.info(f"Historial de iteraciones: {history}")
    flush_logs()
    # Se espera que, al finalizar, el R² global sea mayor o se hayan eliminado grupos hasta quedar menos de 2
    assert history[-1]['global_R2'] >= 0.75 or len(history[-1]['groups']) < 2, "No alcanzó el R² esperado."

def test_experimentation_max_iterations(sample_data):
    """Asegurar que la iteración no excede el límite de max_iterations."""
    logging.info("Verificando límite de iteraciones en ExperimentationIteration")
    flush_logs()
    test_obj = kruskalWallisTest(*sample_data)
    max_iter = 5
    exp_iter = ExperimentationIteration(test_obj, r2_target=0.99, max_iterations=max_iter)
    history = exp_iter.run()
    logging.info(f"Historial de iteraciones (max {max_iter}): {history}")
    flush_logs()
    assert len(history) <= max_iter, "Se excedió el límite de iteraciones."
