import numpy as np
import warnings
from numba import jit
from lib.utils import output_format  

# 🔥 Funciones independientes optimizadas con Numba
@jit(nopython=True, fastmath=True)
def bootstrap_resample_numba(data: np.ndarray, resamples: int, n: int):
    sample_statistics = np.zeros(resamples, dtype=np.float64)
    for i in range(resamples):
        sample = np.take(data, np.random.randint(0, n, n))  # Alternativa a np.random.choice()
        sample_statistics[i] = np.mean(sample)
    return sample_statistics

@jit(nopython=True, fastmath=True)
def bootstrap_resample_median(data: np.ndarray, resamples: int, n: int):
    sample_statistics = np.zeros(resamples, dtype=np.float64)
    for i in range(resamples):
        sample = np.take(data, np.random.randint(0, n, n))
        sample_statistics[i] = np.median(sample)
    return sample_statistics

@jit(nopython=True, fastmath=True)
def bootstrap_resample_variance(data: np.ndarray, resamples: int, n: int):
    sample_statistics = np.zeros(resamples, dtype=np.float64)
    for i in range(resamples):
        sample = np.take(data, np.random.randint(0, n, n))
        sample_statistics[i] = np.var(sample)
    return sample_statistics

class BootstrapConfidenceIntervals:
    def __init__(self, data, alpha=0.05, resamples=10000):
        self.alpha = alpha
        self.resamples = resamples
        self.data = np.asarray(data, dtype=np.float64)  # Asegurar compatibilidad con Numba
        self.n = len(self.data)
        self.sample_statistics = np.zeros(self.resamples, dtype=np.float64)

    def calculate_percentiles(self):
        sorted_samples = np.sort(self.sample_statistics)
        lower_index = int(self.alpha / 2 * self.resamples)
        upper_index = int((1 - self.alpha / 2) * self.resamples)
        return sorted_samples[lower_index], sorted_samples[upper_index]

class BootstrapMeanCI(BootstrapConfidenceIntervals):
    def calculate_interval(self):
        self.sample_statistics = bootstrap_resample_numba(self.data, self.resamples, self.n)
        lb, ub = self.calculate_percentiles()
        return output_format(lb=lb, ub=ub)

class BootstrapMedianCI(BootstrapConfidenceIntervals):
    def calculate_interval(self):
        self.sample_statistics = bootstrap_resample_median(self.data, self.resamples, self.n)
        lb, ub = self.calculate_percentiles()
        return output_format(lb=lb, ub=ub)

class BootstrapVarianceCI(BootstrapConfidenceIntervals):
    def calculate_interval(self):
        self.sample_statistics = bootstrap_resample_variance(self.data, self.resamples, self.n)
        lb, ub = self.calculate_percentiles()
        return output_format(lb=lb, ub=ub)
