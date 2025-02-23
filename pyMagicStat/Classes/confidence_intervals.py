import numpy as np
import scipy.stats as stats
from math import ceil
import warnings
from ..lib.utils import output_format 

# Output format function

class ConfidenceIntervals:
    def __init__(self, data, alpha=0.05):
        self.data = np.array(data)
        self.alpha = alpha
        self.n = len(data)
        if not self.validate_data():
            raise ValueError("Data must not contain NaN values")

    def validate_data(self):
        return np.all(np.isfinite(self.data))

#------------------ Intervalos Paramétricos ------------------#

class NormalDistConfidenceIntervals(ConfidenceIntervals):
    def __init__(self, data, alpha=0.05):
        super().__init__(data, alpha)
        self.mean = np.mean(self.data)
        self.std_dev = np.std(self.data, ddof=1)
        self.variance = self.std_dev ** 2
        self.std_error = self.std_dev / np.sqrt(self.n)

class PopulationMeanCI(NormalDistConfidenceIntervals):
    def calculate_interval(self):
        try:
            z_value = stats.norm.ppf(1 - self.alpha / 2)
            mean = self.data.mean()
            std_dev = self.data.std(ddof=1)
            n = len(self.data)

            if n <= 1:
                return output_format(bool_result=False, txt='Sample is to small , must be > 1')
            if std_dev <= 0:
                return output_format(bool_result=False, txt='Standard Deviation must be > 0')

            margin_of_error = z_value * (std_dev / np.sqrt(n))
            lower_bound = mean - margin_of_error
            upper_bound = mean + margin_of_error

            return output_format(bool_result=True, txt='Confidence Interval for the mean calculated propperly', ub=upper_bound, lb=lower_bound)
        except Exception as e:
            return output_format(bool_result=False, txt=f'Error Calculating Confidence Interval for the Mean')

    def required_sample_size(self, margin_error):
        try:
            z_value = stats.norm.ppf(1 - self.alpha / 2)

            if margin_error <= 2: 
                return output_format(bool_result=False, txt=f"Error must be > 2 , recibed {margin_error}")
            if self.std_dev <= 0:
                return output_format(bool_result=False, txt="Standard Deviation must be > 0")
            n = ceil((z_value * self.std_dev / margin_error) ** 2)
            return output_format(bool_result=True, txt="Sample Size Calculated Fine")
        except Exception as e:
            return output_format(bool_result=False, txt=f'Error calculating the size of the sample {str(e)}')

class PopulationProportionCI(NormalDistConfidenceIntervals):
    def __init__(self, data, alpha=0.05, incidences=None):
        super().__init__(data, alpha)
        if callable(incidences):
            self.incidence_ratio = np.mean([1 if incidences(x) else 0 for x in data])
        else:
            self.incidence_ratio = incidences / self.n if incidences else np.mean(data)
        self.prop_std_dev = np.sqrt(self.incidence_ratio * (1 - self.incidence_ratio) / self.n)

    def calculate_interval(self):
        z_value = stats.norm.ppf(1 - self.alpha / 2)
        margin_of_error = z_value * self.prop_std_dev
        lb = self.incidence_ratio - margin_of_error
        ub = self.incidence_ratio + margin_of_error
        return output_format(lb=lb, ub=ub)

class PopulationVarianceCI(NormalDistConfidenceIntervals):
    def calculate_interval(self):
        chi2_lower = stats.chi2.ppf(1 - self.alpha / 2, self.n - 1)
        chi2_upper = stats.chi2.ppf(self.alpha / 2, self.n - 1)
        lower = ((self.n - 1) * self.variance) / chi2_lower
        upper = ((self.n - 1) * self.variance) / chi2_upper
        return output_format(lb=lower, ub=upper)
    

#------------------ Intervalos No Paramétricos (Bootstrap) ------------------#

# Manejo de advertencias
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    # Inserta aquí el código que podría generar advertencias
