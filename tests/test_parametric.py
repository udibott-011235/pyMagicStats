import pytest
import numpy as np
import scipy.stats as stats
import logging
import time
import warnings

from pyMagicStat.inference.parametric import (
    ParametricMethod,
    PopulationMeanCI,
    PopulationProportionCI,
    PopulationVarianceCI,
    VarianceHomogeneityTest,
    OneSampleTTest,
    PairedTTest,
    TwoSampleTTest
)

# Configurar el logger para trazabilidad de errores
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ParametricTests")

@pytest.fixture
def reproducible_seed():
    """Fija la semilla antes de cada test para garantizar la reproducibilidad."""
    np.random.seed(42)

# ==============================================================================
# 1. Pruebas de Ingeniería y Arquitectura (Manejo de Errores)
# ==============================================================================

def test_nan_inf_handling(reproducible_seed):
    """Evalúa que la clase base rechace datos con NaN o Inf."""
    data_with_nan = np.array([1, 2, np.nan, 4])
    data_with_inf = np.array([1, 2, np.inf, 4])
    
    with pytest.raises(ValueError, match="Data must not contain NaN or Inf values"):
        ParametricMethod(data_with_nan)
        
    with pytest.raises(ValueError, match="Data must not contain NaN or Inf values"):
        ParametricMethod(data_with_inf)

def test_non_normal_small_sample(reproducible_seed):
    """Evalúa que se levante un ValueError si la muestra no es normal y n < 30."""
    # Generar datos exponenciales (no normales) con n=15
    data_small_exp = np.random.exponential(scale=1.0, size=15)
    
    with pytest.raises(ValueError, match="Los datos no siguen una distribución normal y la muestra es menor a 30"):
        ParametricMethod(data_small_exp)

def test_normal_small_sample(reproducible_seed):
    """Evalúa que si la muestra es normal (incluso si n < 30), pase la validación sin usar TLC."""
    # Generar datos normales con n=25 (para evitar warnings de curtosis que requieren n>=20)
    data_small_norm = np.random.normal(loc=0, scale=1, size=25)
    
    # Suprimir warnings por si NormalDistribution arroja algo interno
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = ParametricMethod(data_small_norm)
        
    assert pm.is_normal is True, "La distribución debería haber sido detectada como normal."
    assert pm.tlc_applied is False, "El TLC no debió aplicarse ya que los datos son normales."

# ==============================================================================
# 2. Pruebas de Exactitud Matemática
# ==============================================================================

def test_tlc_mathematical_accuracy(reproducible_seed):
    """Evalúa que al aplicar el TLC, la distribución generada de medias se aproxima a la normal y conserva la media poblacional."""
    # n=50 exponencial (no normal) garantiza que self.is_normal será False y n>=30
    true_mean = 5.0
    data_exp = np.random.exponential(scale=true_mean, size=50)
    
    with pytest.warns(UserWarning, match="Teorema del Límite Central"):
        pm = ParametricMethod(data_exp)
        
    assert pm.tlc_applied is True, "El TLC debió ser aplicado (n>=30 y no normal)."
    assert hasattr(pm, 'tlc_data'), "Se debió generar tlc_data."
    
    # Verificar exactitud: la media de las medias muestrales debe aproximarse a la media original
    original_mean = np.mean(data_exp)
    tlc_mean = np.mean(pm.tlc_data)
    assert np.isclose(original_mean, tlc_mean, rtol=0.05), "La media generada por el TLC difiere significativamente de la media original."
    
    # Comprobar normalidad del tlc_data matemáticamente mediante Shapiro-Wilk
    _, p_value = stats.shapiro(pm.tlc_data)
    assert p_value > 0.05, f"La distribución del TLC no es normal matemáticamente (p-value={p_value})."

def test_mean_ci_accuracy(reproducible_seed):
    """Valida los límites matemáticos del intervalo de confianza para la media poblacional."""
    data_norm = np.random.normal(loc=100, scale=15, size=40)
    mean_ci = PopulationMeanCI(data_norm, alpha=0.05)
    result = mean_ci.calculate_interval()
    
    # Cálculos estrictos a mano
    expected_mean = np.mean(data_norm)
    if mean_ci.tlc_applied:
        expected_std_error = np.std(mean_ci.tlc_data, ddof=1)
    else:
        expected_std_error = np.std(data_norm, ddof=1) / np.sqrt(40)
        
    z_val = stats.norm.ppf(0.975) # Para alpha=0.05 a dos colas
    expected_lb = expected_mean - z_val * expected_std_error
    expected_ub = expected_mean + z_val * expected_std_error
    
    assert result['Result'] == True
    assert np.isclose(result['lb'], expected_lb), "Límite inferior matemáticamente inexacto en PopulationMeanCI"
    assert np.isclose(result['ub'], expected_ub), "Límite superior matemáticamente inexacto en PopulationMeanCI"

def test_proportion_ci_accuracy(reproducible_seed):
    """Valida los límites matemáticos del intervalo de confianza para una proporción."""
    data_prop = np.random.binomial(n=1, p=0.6, size=50).astype(float) # 0s y 1s como floats
    prop_ci = PopulationProportionCI(data_prop, alpha=0.05)
    
    # Evitamos probar normalidad porque PopulationProportionCI hereda de NormalDistConfidenceIntervals
    # y ya realiza sus propios cálculos. Verificaremos que el resultado matemático sea correcto.
    result = prop_ci.calculate_interval()
    
    p_hat = np.mean(data_prop)
    n = len(data_prop)
    expected_std_error = np.sqrt(p_hat * (1 - p_hat) / n)
    z_val = stats.norm.ppf(0.975)
    
    expected_lb = p_hat - z_val * expected_std_error
    expected_ub = p_hat + z_val * expected_std_error
    
    assert np.isclose(result['lb'], expected_lb), "Límite inferior matemáticamente inexacto en PopulationProportionCI"
    assert np.isclose(result['ub'], expected_ub), "Límite superior matemáticamente inexacto en PopulationProportionCI"

def test_variance_ci_accuracy(reproducible_seed):
    """Valida los límites matemáticos del intervalo de confianza para la varianza."""
    data_norm = np.random.normal(loc=50, scale=10, size=50)
    var_ci = PopulationVarianceCI(data_norm, alpha=0.05)
    result = var_ci.calculate_interval()
    
    # Cálculos a mano usando chi-cuadrado
    n = len(data_norm)
    sample_var = np.var(data_norm, ddof=1)
    
    chi2_lower = stats.chi2.ppf(0.975, n - 1)
    chi2_upper = stats.chi2.ppf(0.025, n - 1)
    
    expected_lb = ((n - 1) * sample_var) / chi2_lower
    expected_ub = ((n - 1) * sample_var) / chi2_upper
    
    assert np.isclose(result['lb'], expected_lb), "Límite inferior matemáticamente inexacto en PopulationVarianceCI"
    assert np.isclose(result['ub'], expected_ub), "Límite superior matemáticamente inexacto en PopulationVarianceCI"

# ==============================================================================
# 3. Pruebas de Rendimiento (Performance)
# ==============================================================================

def test_tlc_performance(reproducible_seed):
    """
    Evalúa el rendimiento del algoritmo de remuestreo del TLC.
    Falla si el tiempo de ejecución supera los 0.2 segundos.
    Registra logs detallados para reproducibilidad.
    """
    # Generar un arreglo grande y no normal para forzar un TLC intensivo
    n_size = 500
    num_samples_tlc = 1000
    data_large_exp = np.random.exponential(scale=1.0, size=n_size)
    
    logger.info(f"Iniciando test_tlc_performance. n={n_size}, tlc_iterations={num_samples_tlc}, seed=42")
    
    start_time = time.time()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = ParametricMethod(data_large_exp)
        
    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.info(f"Tiempo de ejecución de inicialización (con TLC si aplica): {execution_time:.4f} segundos")
    
    # Límite estricto de rendimiento sugerido: 0.2 segundos
    MAX_TIME_SECONDS = 0.2
    
    assert pm.tlc_applied is True, "El test de rendimiento requiere que el TLC sea ejecutado."
    assert execution_time < MAX_TIME_SECONDS, (
        f"El rendimiento del módulo degradó. El cálculo del TLC tomó {execution_time:.4f}s, "
        f"lo cual excede el umbral estricto de {MAX_TIME_SECONDS}s."
    )

# ==============================================================================
# 4. Pruebas de Hipótesis Paramétricas (t-tests y Homocedasticidad)
# ==============================================================================

def test_variance_homogeneity_equal(reproducible_seed):
    """Valida la prueba de homogeneidad de varianzas cuando las varianzas son iguales."""
    g1 = np.random.normal(loc=10, scale=2.0, size=50)
    g2 = np.random.normal(loc=15, scale=2.0, size=50)
    
    test = VarianceHomogeneityTest(g1, g2, method='levene')
    res = test.run_test()
    
    scipy_stat, scipy_p = stats.levene(g1, g2, center='median')
    
    assert res['Result'] == True, "Las varianzas deberían identificarse como iguales/homogéneas."
    assert res['equal_variance'] == True
    assert np.isclose(res['statistic'], scipy_stat), "Estadístico de Levene difiere de SciPy."
    assert np.isclose(res['p_value'], scipy_p), "p-value de Levene difiere de SciPy."

def test_variance_homogeneity_unequal(reproducible_seed):
    """Valida la prueba de homogeneidad de varianzas cuando las varianzas son sustancialmente distintas."""
    g1 = np.random.normal(loc=10, scale=1.0, size=50)
    g2 = np.random.normal(loc=10, scale=6.0, size=50)
    
    test = VarianceHomogeneityTest(g1, g2, method='levene')
    res = test.run_test()
    
    assert res['Result'] == False, "Las varianzas deberían identificarse como heterogéneas."
    assert res['equal_variance'] == False

def test_one_sample_ttest_precision(reproducible_seed):
    """Valida la exactitud matemática de la prueba t de 1 muestra contra SciPy."""
    data = np.random.normal(loc=10.5, scale=2.0, size=40)
    popmean = 10.0
    
    one_t = OneSampleTTest(data, popmean=popmean, alpha=0.05, alternative='two-sided')
    res = one_t.run_test()
    
    scipy_stat, scipy_p = stats.ttest_1samp(data, popmean, alternative='two-sided')
    
    assert res['Result'] == (res['p_value'] < 0.05)
    assert np.isclose(res['statistic'], scipy_stat), "Estadístico t difiere de SciPy."
    assert np.isclose(res['p_value'], scipy_p), "p-value de 1 sample t-test difiere de SciPy."
    assert res['df'] == 39

def test_one_sample_ttest_alternatives(reproducible_seed):
    """Valida hipótesis alternativas ('greater' y 'less') en 1 sample t-test."""
    data = np.random.normal(loc=12.0, scale=1.5, size=35)
    popmean = 10.0
    
    one_t_greater = OneSampleTTest(data, popmean=popmean, alternative='greater')
    res_g = one_t_greater.run_test()
    scipy_stat_g, scipy_p_g = stats.ttest_1samp(data, popmean, alternative='greater')
    assert np.isclose(res_g['p_value'], scipy_p_g)

    one_t_less = OneSampleTTest(data, popmean=popmean, alternative='less')
    res_l = one_t_less.run_test()
    scipy_stat_l, scipy_p_l = stats.ttest_1samp(data, popmean, alternative='less')
    assert np.isclose(res_l['p_value'], scipy_p_l)

def test_paired_ttest_precision(reproducible_seed):
    """Valida la exactitud matemática de la prueba t pareada contra SciPy."""
    d1 = np.random.normal(loc=20.0, scale=3.0, size=35)
    d2 = d1 + np.random.normal(loc=1.5, scale=0.5, size=35)
    
    paired_t = PairedTTest(d1, d2, popmean=0.0, alpha=0.05)
    res = paired_t.run_test()
    
    scipy_stat, scipy_p = stats.ttest_rel(d1, d2)
    
    assert np.isclose(res['statistic'], scipy_stat), "Estadístico t pareado difiere de SciPy."
    assert np.isclose(res['p_value'], scipy_p), "p-value t pareado difiere de SciPy."

def test_paired_ttest_length_mismatch():
    """Valida error si las muestras pareadas no tienen la misma longitud."""
    d1 = [1, 2, 3, 4]
    d2 = [1, 2, 3]
    with pytest.raises(ValueError, match="Las muestras pareadas deben tener la misma longitud"):
        PairedTTest(d1, d2)

def test_two_sample_ttest_student_and_welch(reproducible_seed):
    """Valida la prueba t de 2 muestras independientes en modos Student (varianzas iguales) y Welch (varianzas desiguales)."""
    # Grupos con varianzas similares
    g1 = np.random.normal(loc=100, scale=10, size=40)
    g2 = np.random.normal(loc=105, scale=10, size=45)
    
    # Student's t-test (equal_var=True)
    two_t_student = TwoSampleTTest(g1, g2, equal_var=True)
    res_student = two_t_student.run_test()
    scipy_stat_s, scipy_p_s = stats.ttest_ind(g1, g2, equal_var=True)
    
    assert res_student['method'] == "Student's t-test"
    assert np.isclose(res_student['statistic'], scipy_stat_s)
    assert np.isclose(res_student['p_value'], scipy_p_s)
    assert res_student['df'] == (40 + 45 - 2)

    # Grupos con varianzas distintas
    g3 = np.random.normal(loc=100, scale=2, size=40)
    g4 = np.random.normal(loc=110, scale=15, size=45)
    
    # Welch's t-test (equal_var=False)
    two_t_welch = TwoSampleTTest(g3, g4, equal_var=False)
    res_welch = two_t_welch.run_test()
    scipy_stat_w, scipy_p_w = stats.ttest_ind(g3, g4, equal_var=False)
    
    assert res_welch['method'] == "Welch's t-test"
    assert np.isclose(res_welch['statistic'], scipy_stat_w)
    assert np.isclose(res_welch['p_value'], scipy_p_w)

def test_two_sample_ttest_auto_homogeneity(reproducible_seed):
    """Valida la selección automática entre Student y Welch según la prueba de homogeneidad de varianza."""
    # Varianzas iguales -> debe elegir Student
    g1 = np.random.normal(loc=50, scale=5, size=40)
    g2 = np.random.normal(loc=52, scale=5, size=40)
    t_auto_equal = TwoSampleTTest(g1, g2, equal_var=None)
    res_auto_equal = t_auto_equal.run_test()
    assert res_auto_equal['equal_var'] == True
    assert res_auto_equal['method'] == "Student's t-test"

    # Varianzas desiguales -> debe elegir Welch
    g3 = np.random.normal(loc=50, scale=1, size=40)
    g4 = np.random.normal(loc=52, scale=10, size=40)
    t_auto_unequal = TwoSampleTTest(g3, g4, equal_var=None)
    res_auto_unequal = t_auto_unequal.run_test()
    assert res_auto_unequal['equal_var'] == False
    assert res_auto_unequal['method'] == "Welch's t-test"


def test_two_sample_ttest_ha_expressions_and_df_aliases(reproducible_seed):
    """Valida la configuración explícita de Ha ('df2 > df1', 'df1 > df2') y los alias df1, df2."""
    g1 = np.random.normal(loc=10.0, scale=2.0, size=40)
    g2 = np.random.normal(loc=15.0, scale=2.0, size=40)  # g2 mean > g1 mean
    
    # Probando Ha: df2 > df1 (debe mapear a cola 'less' sobre (mean1 - mean2))
    test_ha_g2_greater = TwoSampleTTest(df1=g1, df2=g2, ha="df2 > df1", equal_var=True)
    res1 = test_ha_g2_greater.run_test()
    scipy_stat, scipy_p_less = stats.ttest_ind(g1, g2, alternative='less', equal_var=True)
    
    assert res1['ha'] == "df2 > df1"
    assert res1['alternative'] == 'less'
    assert np.isclose(res1['p_value'], scipy_p_less)
    assert res1['Result'] == True  # Se rechaza H0 en favor de Ha porque g2 > g1
    assert "df2 > df1" in res1['txt']
    
    # Probando Ha: df1 > df2 (debe mapear a cola 'greater')
    test_ha_g1_greater = TwoSampleTTest(df1=g1, df2=g2, ha="df1 > df2", equal_var=True)
    res2 = test_ha_g1_greater.run_test()
    scipy_stat, scipy_p_greater = stats.ttest_ind(g1, g2, alternative='greater', equal_var=True)
    
    assert res2['ha'] == "df1 > df2"
    assert res2['alternative'] == 'greater'
    assert np.isclose(res2['p_value'], scipy_p_greater)
    assert res2['Result'] == False  # No se rechaza porque g1 no es mayor que g2


def test_paired_ttest_ha_and_group_aliases(reproducible_seed):
    """Valida PairedTTest con alias group1/group2 y expresión Ha."""
    d1 = np.random.normal(loc=20.0, scale=3.0, size=35)
    d2 = d1 + np.random.normal(loc=2.0, scale=0.5, size=35)  # d2 > d1
    
    paired_test = PairedTTest(group1=d1, group2=d2, ha="group2 > group1")
    res = paired_test.run_test()
    
    scipy_stat, scipy_p_less = stats.ttest_rel(d1, d2, alternative='less')
    
    assert res['ha'] == "group2 > group1"
    assert res['alternative'] == 'less'
    assert np.isclose(res['p_value'], scipy_p_less)
    assert res['Result'] == True
    assert "group2 > group1" in res['txt']



