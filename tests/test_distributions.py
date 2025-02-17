import pytest
import numpy as np
from scipy import stats
from pyMagicStat.Classes.distributions import NormalDistribution, BinomialDistribution, PoissonDistribution, LognormalDistribution

# 📌 Configurar tolerancia de error numérico
TOLERANCE = 1e-6

# 📌 Ruta para guardar el archivo de salida
RESULTS_FILE = "tests/test_results.txt"

def write_results(test_name, result):
    """
    Escribe los resultados de cada test en un archivo de texto.
    """
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{test_name}: {'PASSED' if result else 'FAILED'}\n")

# 🔹 Generar datos de prueba
@pytest.fixture
def sample_data():
    np.random.seed(42)
    return {
        'normal': np.random.normal(loc=10, scale=5, size=5000),
        'binomial': np.random.binomial(n=20, p=0.5, size=5000),
        'poisson': np.random.poisson(lam=5, size=5000),
        'lognormal': np.random.lognormal(mean=2, sigma=0.5, size=5000)
    }

# 🔥 🔹 TESTS PARA NormalDistribution
def test_normal_distribution_fit(sample_data):
    data = sample_data['normal']
    model = NormalDistribution(data)
    result = model.fit_test()

    passed = 'Normal' in model.type and isinstance(model.type['Normal']['Fit'], np.bool_) and model.type['Normal']['Fit'] == True
    write_results("test_normal_distribution_fit", passed)
    assert passed

def test_normal_distribution_approximation(sample_data):
    data = sample_data['normal']
    model = NormalDistribution(data)
    model.fit_test()
    result = model.normal_approximation()

    passed = model.type['Normal']['Normal_approx'] == True
    write_results("test_normal_distribution_approximation", passed)
    assert passed

# 🔥 🔹 TESTS PARA BinomialDistribution
def test_binomial_distribution_fit(sample_data):
    data = sample_data['binomial']
    model = BinomialDistribution(data)
    result = model.fit_test()

    passed = 'Binomial' in model.type and isinstance(model.type['Binomial']['Fit'], np.bool_) and model.type['Binomial']['Fit'] == True
    write_results("test_binomial_distribution_fit", passed)
    assert passed

def test_binomial_distribution_approximation(sample_data):
    data = sample_data['binomial']
    model = BinomialDistribution(data)
    model.fit_test()
    result = model.normal_approximation()

    expected_approx = (model.n * model.p >= 5) and (model.n * (1 - model.p) >= 5)
    passed = model.type['Binomial']['Normal_approx'] == expected_approx
    write_results("test_binomial_distribution_approximation", passed)
    assert passed

# 🔥 🔹 TESTS PARA PoissonDistribution
def test_poisson_distribution_fit(sample_data):
    data = sample_data['poisson']
    model = PoissonDistribution(data)
    result = model.fit_test()

    passed = 'Poisson' in model.type and isinstance(model.type['Poisson']['Fit'], np.bool_) and model.type['Poisson']['Fit'] == True
    write_results("test_poisson_distribution_fit", passed)
    assert passed

def test_poisson_distribution_approximation(sample_data):
    data = sample_data['poisson']
    model = PoissonDistribution(data)
    model.fit_test()
    result = model.normal_approximation()

    expected_approx = model.lam >= 30
    passed = model.type['Poisson']['Normal_approx'] == expected_approx
    write_results("test_poisson_distribution_approximation", passed)
    assert passed

# 🔥 🔹 TESTS PARA LognormalDistribution
def test_lognormal_distribution_fit(sample_data):
    data = sample_data['lognormal']
    model = LognormalDistribution(data)
    result = model.fit_test()

    passed = 'Lognormal' in model.type and isinstance(model.type['Lognormal']['Fit'], np.bool_) and model.type['Lognormal']['Fit'] == True
    write_results("test_lognormal_distribution_fit", passed)
    assert passed

def test_lognormal_distribution_approximation(sample_data):
    data = sample_data['lognormal']
    model = LognormalDistribution(data)
    model.fit_test()
    result = model.normal_approximation()

    log_data = np.log(data)
    skewness = stats.skew(log_data)
    kurtosis = stats.kurtosis(log_data, fisher=True) + 3

    expected_approx = (-2 <= skewness <= 2) and (1.5 <= kurtosis <= 4.5)
    passed = model.type['Lognormal']['Normal_approx'] == expected_approx
    write_results("test_lognormal_distribution_approximation", passed)
    assert passed
