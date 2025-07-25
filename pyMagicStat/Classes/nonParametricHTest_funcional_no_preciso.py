import numpy as np
import warnings
from numba import jit
from scipy.stats import  kruskal, mannwhitneyu
from pyMagicStat.lib.utils import output_format 


@jit(nopython=True, fastmath=True)
def bootstrap_resample_numba(data: np.ndarray, resamples: int, n: int):
    sample_statistics = np.zeros(resamples, dtype=np.float64)
    for i in range(resamples):
        sample = np.random.choice(data, size=np.random.randint(int(n*0.7), n), replace=True)  # Alternativa a np.random.choice()
        sample_statistics[i] = np.mean(sample)
    return sample_statistics

@jit(nopython=True, fastmath=True)
def bootstrap_resample_median(data: np.ndarray, resamples: int, n: int):
    sample_statistics = np.zeros(resamples, dtype=np.float64)
    for i in range(resamples):
        sample = np.random.choice(data, size=np.random.randint(int(n*0.70), n))
        sample_statistics[i] = np.median(sample)
    return sample_statistics

@jit(nopython=True, fastmath=True)
def bootstrap_resample_variance(data: np.ndarray, resamples: int, n: int):
    sample_statistics = np.zeros(resamples, dtype=np.float64)
    for i in range(resamples):
        sample = np.random.choice(data, size=np.random.randint(int(n*0.7), n), replace=True)
        mean_sample = np.mean(sample)
        variance_sample = np.sum((sample - mean_sample)**2) / (len(sample) -1)
        sample_statistics[i] = variance_sample
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

class BootstrapProportionCI(BootstrapConfidenceIntervals):
    def __init__(self, data, alpha=0.05, resamples=10000 , p0=None, alternative="two-sided"):
        super().__init__(data, alpha, resamples)
        self.p0 = p0
        self.alternative = alternative 

    def calculate_interval(self):
        p0_adjusted = np.percentile(self.data, min(99, max(1, 100 * (self.p0 - np.min(self.data)) / (np.max(self.data) - np.min(self.data)))))
        
        perturbed_data = self.data + np.random.normal(0, np.std(self.data) * 0.03, size=len(self.data))
        
        self.sample_statistics = np.array([
            np.mean(sample[sample < p0_adjusted]) if np.any(sample < p0_adjusted) else 0
            for sample in (np.random.choice(perturbed_data, size=np.random.randint(int(self.n * 0.7), self.n), replace=True) for _ in range (self.resamples))
        ])
        if np.std(self.sample_statistics) < 1e-3:
            self.sample_statistics += np.random.uniform(0, 0.01, size=self.sample_statistics.shape)
        
        print(f"Valores únicos en sample_statistics: {np.unique(self.sample_statistics)[:10]}")
        print(f"Media del remuestreo: {np.mean(self.sample_statistics)}")
        print(f"Desviación estándar del remuestreo: {np.std(self.sample_statistics)}")
        print(f"Valores Unicos en sample_statistics: {np.unique(self.sample_statistics)[:10]}")
        
        lb, ub = self.calculate_percentiles()
        print(f"Intervalo calculado Li:{lb}, Ls:{ub}")
        return output_format(lb=lb, ub=ub)

    def summary(self):
        ci = self.calculate_interval()
        observed_proportion = np.mean(self.data)
        result = {
            "observed_proportion": observed_proportion,
            "confidence_interval": ci,
            "alpha": self.alpha, 
            "resamples": self.resamples
        }
        if self.p0 is not None:
            result["p0"] = self.p0
            if self.alternative == "two-sided":
               result["significant"] = not (ci["lb"] <= self.p0 <= ci["ub"])
            elif self.alternative == "greater":
               result["significant"] = ci["lb"] > self.p0
            elif self.alternative == "less":
                result["significant"] = ci["ub"] < self.p0
        return result
            
class kruskalWallisTest:
    #### his objective is to explain the variance by studing the groups related to it 
    #### with a low R**2 means the differece bettwen the groups explains the variance or not 
    ######## with a low r**2 pValue will increse pval > Alpha there is no difference within the groups
    #### ver valor H si hay tabla o valor de rechazo  
    #### hay que incluir un metodo vizual 
    #####https://www.ibm.com/docs/es/spss-statistics/beta?topic=tests-kruskal-wallis-test
    def __init__(self, *groups, alpha=0.05, labels=None, alternative="two-sided"):    
        
        
        """ Kruskal-Wallis H test for independent samples , imput can be lists or numpy arrays"""
        if len(groups) < 2:
            raise ValueError("Must provide at least two groups")
        
        self.groups = list(groups)  
        self.alpha = alpha
        self.alternative = alternative

       ## assign lables or generate default one 
        self.labels = labels if labels is not None else [f"Group {i +1}" for i in range(len(groups))]
        if len(self.labels) != len(groups):
            raise ValueError("Number of labels must match number of groups")
        
        #Variance Homogeneity Tst R²
        
        self.ss_total = 0
        self.ss_within = None
        self.r_squared = None
        self._compute_r_squared()
        self.results = {}

     #could we optimice this func with numba ? 
    def _compute_r_squared(self):    
        all_data = np.concatenate(self.groups)
        grand_mean = np.mean(all_data)
        self.ss_total = np.sum((all_data - grand_mean) ** 2)
       
        #compute de sum of squares within ssw
        self.ss_within = [np.sum((group - np.mean(group)) ** 2) for group in self.groups]
       
        #compute de R2 per group
        self.r_squared = [1- (ssw / self.ss_total) if self.ss_total > 0 else 0 for ssw in self.ss_within]
        
    def run_test(self):

        if self.r_squared is None or self.ss_within is None:
            raise ValueError("R² and sum of squares within (SSW) must be computed before running the test.")
        
        H_statistic, p_value = kruskal(*self.groups)
        df = len(self.groups) - 1 # degrees of freedom


        p_values_per_group = []
        for i, group in enumerate(self.groups):
            rest= np.concatenate([self.groups[j] for j in range(len(self.groups)) if j != i ])
            _, p_indiv = mannwhitneyu(group, rest, alternative=self.alternative)
            p_values_per_group.append(p_indiv)
       
        
        self.results = {
            
                "H_statistic": H_statistic,
                "p_value": p_value,
                "df": df,
                "Total_SS": self.ss_total,
                "Groups": [
                    {
                    "Label": label,
                    "SSW": ssw,
                    "R^2": r2,
                    "p_value": p_val
                    }
            for label, ssw, r2, p_val in zip(self.labels, self.ss_within, self.r_squared, p_values_per_group)
            ]
    }

        return output_format(data=self.results)
    
    def remove_group(self, idx):
        removed_label = self.labels[idx]
        del self.groups[idx]
        del self.labels[idx]
        self._compute_r_squared()
        return removed_label
    