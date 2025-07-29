import pytest
import numpy as np
import logging

from pyMagicStat.Classes.nonParametricHTest import (
    kruskalWallisTest,
    BootstrapCI
)
from pyMagicStat.Classes.confidence_intervals import (
    PopulationMeanCI,
    PopulationProportionCI,
    PopulationVarianceCI
)
from pyMagicStat.Classes.orchestrator import StatisticalEvaluator, OptimizedExperimentationIteration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("test_CI_K_W_log.log")]
)

def flush_logs():
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()

@pytest.fixture
def sample_data():
    np.random.seed(42)
    group_1 = np.random.normal(50, 10, 30)
    group_2 = np.random.normal(55, 10, 30)
    group_3 = np.random.normal(60, 10, 30)
    group_4 = np.random.normal(90, 10, 30)
    return [group_1, group_2, group_3, group_4]

@pytest.fixture
def extended_data():
    np.random.seed(123)
    return {
        "normal": np.random.normal(50, 10, 1000),
        "exponential": np.random.exponential(10, 1000),
        "gamma": np.random.gamma(2, 2, 1000),
        "uniform": np.random.uniform(0, 100, 1000),
        "bimodal": np.concatenate([np.random.normal(30, 5, 500), np.random.normal(70, 5, 500)]),
        "lognormal": np.random.lognormal(3, 1, 1000),
        "poisson": np.random.poisson(5, 1000)
    }

def test_kruskalwallis_and_statistical_evaluator(sample_data):
    logging.info("Test Kruskal-Wallis y StatisticalEvaluator")
    flush_logs()
    evaluator = StatisticalEvaluator(kruskalWallisTest)
    labels = [f"Group {i+1}" for i in range(len(sample_data))]
    result = evaluator.evaluate(sample_data, labels)
    logging.info(f"Resultados KruskalWallisTest: {result}")
    flush_logs()
    assert result["p_value"] < 0.05, "Kruskal-Wallis debería detectar diferencias significativas"
    assert result["global_R2"] is not None, "global_R2 debe calcularse"

def test_optimized_experimentation_iteration(sample_data):
    logging.info("Test OptimizedExperimentationIteration")
    flush_logs()
    evaluator = StatisticalEvaluator(kruskalWallisTest)
    opt_iter = OptimizedExperimentationIteration(evaluator, sample_data, strategy="greedy", r2_target=0.75, max_iterations=10)
    history = opt_iter.run()
    logging.info(f"Historial de iteraciones: {history}")
    flush_logs()
    assert history[-1]['global_R2'] >= 0.75 or len(history[-1]['groups']) < 2, "No alcanzó el R² esperado."

def test_parametric_confidence_intervals(extended_data):
    logging.info("Test intervalos paramétricos")
    flush_logs()
    for name, data in extended_data.items():
        mean_ci = PopulationMeanCI(data)
        mean_result = mean_ci.calculate_interval()
        var_ci = PopulationVarianceCI(data)
        var_result = var_ci.calculate_interval()
        prop_ci = PopulationProportionCI(data)
        prop_result = prop_ci.calculate_interval()
        logging.info(f"{name} - Mean CI: {mean_result}")
        logging.info(f"{name} - Variance CI: {var_result}")
        logging.info(f"{name} - Proportion CI: {prop_result}")
        flush_logs()
        assert mean_result['ub'] > mean_result['lb'], f"Mean CI inválido para {name}"
        assert var_result['ub'] > var_result['lb'], f"Variance CI inválido para {name}"
        assert prop_result['ub'] > prop_result['lb'], f"Proportion CI inválido para {name}"

def test_bootstrap_confidence_intervals(extended_data):
    logging.info("Test intervalos bootstrap (scipy y numba)")
    flush_logs()
    for name, data in extended_data.items():
        for stat in ['mean', 'median', 'variance', 'proportion']:
            for method in ['scipy', 'numba']:
                ci = BootstrapCI(data, stat=stat, method=method, alpha=0.05, n_resamples=2000)
                result = ci.compute()
                lb = result['lb']
                ub = result['ub']
                logging.info(f"{name} - {stat} ({method}) CI: {result}")
                flush_logs()
                if isinstance(lb, (tuple, list)):
                    lb = lb[0]
                if isinstance(ub, (tuple, list)):
                    ub = ub[0]
                assert ub > lb, f"Bootstrap {stat} CI inválido para {name} con método {method}"

def test_bootstrap_vs_parametric_mean(extended_data):
    logging.info("Comparación Bootstrap vs Paramétrico para la media")
    flush_logs()
    for name, data in extended_data.items():
        param_ci = PopulationMeanCI(data).calculate_interval()
        boot_ci = BootstrapCI(data, stat='mean', method='scipy', alpha=0.05, n_resamples=2000).compute()
        param_width = param_ci['ub'] - param_ci['lb']
        boot_width = boot_ci['ub'] - boot_ci['lb']
        logging.info(f"{name} - Parametric CI width: {param_width}, Bootstrap CI width: {boot_width}")
        flush_logs()
        assert param_width > 0 and boot_width > 0, f"Intervalos inválidos para {name}"
        # Opcional: comparar que los intervalos sean similares en tamaño
        assert abs(param_width - boot_width) < max(param_width, boot_width), f"Intervalos muy diferentes para {name}"
    logging.info(f"Historial de iteraciones (max {max_iter}): {history}")
    flush_logs()
    assert len(history) <= max_iter, "Se excedió el límite de iteraciones."

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__]))
