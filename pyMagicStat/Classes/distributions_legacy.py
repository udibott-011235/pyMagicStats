import numpy as np
from scipy import stats
from scipy.stats import kurtosis, skew
import warnings
from ..lib.utils import output_format, validate_non_nan, positive_values_test
from scipy.special import gammaln
from scipy.optimize import minimize

class Distribution:
    def __init__(self, data):
        if isinstance(data, Distribution):
            self.data = data.data  # Maneja el caso en el que `data` ya es una instancia de `Distribution`
        else:
            self.data = np.array(data)
        self.type = {}
        self.stats = self.show_statistics()
        self.aic_table = {}

    def validate_data(self):
        if np.any(np.isnan(self.data)):
            raise ValueError("Data contains NaN values")
        if np.any(np.isinf(self.data)):
            raise ValueError("Data contains infinite values")
        return True

    def show_statistics(self):
        self.validate_data()
        return {
            'min': np.min(self.data),
            'max': np.max(self.data),
            'mean': np.mean(self.data),
            'median': np.median(self.data),
            'standard_deviation': np.std(self.data, ddof=1),
            'kurtosis': self.kurtosis(),
            'skewness': self.skewness()
        }
    
    def kurtosis(self):
        try:
            return stats.kurtosis(self.data)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def skewness(self):
        try:
            return stats.skew(self.data)
        except Exception as e:
            return f"Error: {str(e)}"
        
    def update_type(self, distribution_name, result, normal_approx=None, methods=None):
        """
        Actualiza o establece la información de self.type para la distribución dada.
    
        Parámetros:
        distribution_name: Nombre de la distribución (por ejemplo, 'Binomial')
        result: Resultado global del test de bondad (Fit) (True/False)
        normal_approx: Resultado de la evaluación de aproximación normal (True/False)
        methods: Un diccionario con información adicional (por ejemplo, 'Test' y 'p_value')
        """
        if distribution_name in self.type and isinstance(self.type[distribution_name], dict):
            # Actualizamos los campos existentes sin eliminar otros
            self.type[distribution_name]['Fit'] = result
            if normal_approx is not None:
                self.type[distribution_name]['Normal_approx'] = normal_approx
            if methods is not None:
                self.type[distribution_name]['Methods'] = methods
        else:
            self.type[distribution_name] = {
                'Fit': result,
                'Normal_approx': normal_approx,
                'Methods': methods
            }

class DistributionTest:
    def __init__(self, distribution):
        self.distribution = distribution
        self.data = np.array(distribution.data)
    
    def validate_data(self):
        if np.any(np.isnan(self.data)):
            raise ValueError("Data contains Nan values")
        
        if isinstance(self.distribution, LognormalDistribution) and not positive_values_test(self.data):
            raise ValueError("Data contains non-positive values, which are not allowed")
        
        if isinstance(self.distribution, BinomialDistribution) and not np.all(self.data == np.floor(self.data)):
            raise ValueError("Data contains non-integer values, but integer values are required")
        return True
    
    def calculate_log_likelihood(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def calculate_aic(self):
        log_likelihood = self.calculate_log_likelihood()
        if log_likelihood is None: 
            raise ValueError("Log-likelihood is None, cannot calculate AIC")
        
        k = self.get_num_parameters()
        if k is None:
            raise ValueError("Number of parameters is None, cannot calculate AIC")
        return  2 * k - 2 * log_likelihood
    
    def get_num_parameters(self):
        raise NotImplementedError("This method should be implemented by subclasses")

class NormalDistribution(Distribution, DistributionTest):
    def __init__(self, distribution):
        Distribution.__init__(self, distribution.data)
        DistributionTest.__init__(self, distribution)
        
        self.mean = np.mean(self.data)
        self.std_dev = np.std(self.data, ddof=1)

    def shapiro_test(self):
        if len(self.data) < 3:
            return {"p_value": np.nan, "bool_result": False, "txt": "Sample size too small"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            stat, p_value = stats.shapiro(self.data)
        bool_res = p_value > 0.05
        self.update_type('Normal', bool_res)
        return {"p_value": p_value, "bool_result": bool_res, "txt": "Shapiro test"}

    def anderson_test(self):
        if len(self.data) < 3:
            return {"p_value": np.nan, "bool_result": False, "txt": "Sample size too small"}
        result = stats.anderson(self.data, dist='norm')
        bool_res = result.statistic < result.critical_values[2]
        self.update_type('Normal', bool_res)
        return {"p_value": result.significance_level[2] / 100, "bool_result": bool_res, "txt": "Anderson-Darling test"}

    def dagostino_test(self):
        if len(self.data) < 3:
            return {"p_value": np.nan, "bool_result": False, "txt": "Sample size too small"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            stat, p_value = stats.normaltest(self.data)
        bool_res = p_value > 0.05
        self.update_type('Normal', bool_res)
        return {"p_value": p_value, "bool_result": bool_res, "txt": "D’Agostino-Pearson test"}

    def fit_test(self):
        try:
            self.validate_data()
            if len(self.data) < 3:
                return output_format(bool_result=False, txt='Sample size too small')
            
            #execute the each test
            shapiro_result = self.shapiro_test()
            anderson_result = self.anderson_test()
            dagostino_result = self.dagostino_test()

            p_values = [max(r['p_value'], 1e-16) for r in [shapiro_result, anderson_result, dagostino_result]]
            #Fisher for combined P values 
            fisher_stat = -2 * sum(np.log(p) for p in p_values)
            df = 2 * len(p_values)
            combined_p = 1 - stats.chi2.cdf(fisher_stat, df)
            fit_result = combined_p > 0.05

            method_info = {
                'Tests': 'Fisher combined p-value',
                'p_values': combined_p,
                'fisher_stat': fisher_stat,
                'df': df
                }
            txt = f"Fisher combined p-value: {combined_p:.4f}, Normality: {'Accepted' if fit_result else 'Rejected'}"
            
            self.update_type('Normal', fit_result, methods=method_info)
            
            return output_format(bool_result=fit_result, txt=txt, p_value=combined_p)   
        except Exception as e:
            raise TypeError(f"Error in NormalDistribution fit_test: {str(e)}")

    def normal_approximation(self):
        normal_info = self.type.get('Normal', {})
        if not normal_info.get('Fit', False):
            return output_format(bool_result=False, txt="Not Normal")
        
        is_normal = True
        normal_info['Normal_approx'] = is_normal
        self.type['Normal'] = normal_info
        return output_format(bool_result=is_normal, txt="Can be treated as normal")

    def calculate_log_likelihood(self):
        self.validate_data()
        n = len(self.data)
        log_likelihood = -n / 2 * np.log(2 * np.pi * self.std_dev ** 2) - np.sum((self.data - self.mean) ** 2) /(2 * self.std_dev ** 2)
        return log_likelihood

    def get_num_parameters(self):
        return 2

class BinomialDistribution(Distribution, DistributionTest):
    def __init__(self, distribution, n=None, p=None):
        Distribution.__init__(self, distribution.data)
        DistributionTest.__init__(self, distribution)

        self.data = np.round(distribution.data).astype(int)
        self.validate_data()
        
        if not np.all(self.data >= 0):
            raise ValueError("All values in data must be non-negative integers")

        if n is None:
            #printing for debugging
            print(f"Estimando n con MLE")
            self.n = self.estimate_n_mle()
        else:
            self.n = n 

            #estimate the p parameter using the maximum likelihood estimation
        self.p = p if p is not None else np.mean(self.data) / self.n
        print(f"🔍 Estimación inicial: n = {self.n}, p = {self.p:.4f}")
    def estimate_n_mle(self):
        #estimate the n parameter using the maximum likelihood estimation
        mean_x = np.mean(self.data)

        def neg_log_likelihood(n):
            p_hat = mean_x / n
            if p_hat <= 0 or p_hat >= 1:
                return np.inf # avoid values outside the range (0,1)
            log_likelihood = np.sum(stats.binom.logpmf(self.data, n, p_hat))
            return -log_likelihood
       
        result = minimize(neg_log_likelihood, x0=np.max(self.data), bounds=[(1, None)], method='L-BFGS-B')
        return int(round(result.x[0])) if result.success else np.max(self.data)
    
        #function to group bins , is created in this way to make it easy to move it when the abstraction of chi2 is ready to apply
    @staticmethod 
    def combine_bins(obs, exp, min_expected=5):
        group_obs = []
        group_exp = []
        current_obs = 0
        current_exp = 0
        for o, e in zip(obs, exp):
            current_obs += o
            current_exp += e

            if current_exp >= min_expected:
                group_obs.append(current_obs)
                group_exp.append(current_exp)
                current_obs = 0
                current_exp = 0

        if current_exp > 0:

            if current_exp > 0:
                group_obs[-1] += current_obs
                group_exp[-1] += current_exp
            
            else:
                group_obs.append(current_obs)
                group_exp.append(current_exp)

        return np.array(group_obs), np.array(group_exp)
    
    def fit_test(self):
        try:
            self.validate_data()
            if not np.all((self.data >= 0 ) & (self.data <= self.n) & (self.data == self.data.astype(int))):
                self.update_type('Binomial', False)
                return output_format(bool_result=False, txt="Binomial data check failed")
            
            total_data = len(self.data)
            #printing for debugging 
            print(f"🔍 Debugging BinomialDistribution.fit_test()")
            print(f"  - Total data points: {total_data}")
            print(f"  - Expected n: {self.n}")
            print(f"  - Observed mean: {np.mean(self.data)}")
            print(f"  - Observed sum: {np.sum(self.data)}")

            p_hat = np.mean(self.data) / self.n
            #printing for debugging
            print(f"  - Calculated p_hat: {p_hat:.4f}")

            if p_hat <= 0 or p_hat >= 1:
                return output_format(bool_result=False, txt=f"Invalid p-hat value p_hat={p_hat:.4f}, outside (0,1)")
            
            #create histogram
            bins = np.arange(-0.5, self.n + 1.5, 1)
            observed, _ = np.histogram(self.data, bins=bins)

            #calculate frequencys
            x_vals = np.arange(0, self.n + 1)
            expected = stats.binom.pmf(x_vals, self.n, self.p) * total_data

            grouped_obs, grouped_exp = self.combine_bins(observed, expected, min_expected=5)
            # fredom degrees validation 
            if not np.isclose(grouped_exp.sum(), observed.sum(), rtol=1e-8):
                raise ValueError("Grouped frequencies do not sum to total data count")
            #Nr of groups 
            n_groups = len(grouped_obs)

            # adjust Freedom Degrees 
            ddof = n_groups - 2  
            if ddof <= 0:
                return output_format(bool_result=False, txt="Insufficient degrees of freedom")
            
            #Umbral to concider the sample to big this will help to calibrate the test to use
            LARGE_SAMPLE_THRESHOLD = 5000
            if total_data < LARGE_SAMPLE_THRESHOLD:
                chi_stat, p_value = stats.chisquare(f_obs=grouped_obs, f_exp=grouped_exp, ddof=ddof)
                test_used = "Chi-square test"
            else:

                epsilon = 1e-10
                G2_stat = 2 * np.sum(grouped_obs * np.log((grouped_obs + epsilon) / (grouped_exp + epsilon)))
                p_value = 1 - stats.chi2.cdf(G2_stat, df=ddof)
                chi_stat = G2_stat
                test_used = "Likelihod ratio (G2) test"
            
            result_bool = p_value > 0.05

            self.type = {
            'Binomial': {
                'Fit': np.bool_(result_bool),
                'Normal_approx': None,
                'Methods': {
                    'Test': test_used,
                    'p_value': np.float64(p_value),
                    'chi_stat': np.float64(chi_stat),
                    'p_hat': np.float64(p_hat)  # Agregamos p_hat para depuración
                }
            }
        }

            txt = f"{test_used}: chi2 {chi_stat:.2f}, p = {p_value:.4f} (p_hat = {p_hat:.4f})"
            return output_format(bool_result=result_bool, txt=txt, p_value=p_value)

        except Exception as e:
            raise TypeError(f"Error in BinomialDistribution fit_test: {str(e)}")

    def normal_approximation(self):
        binom_info = self.type.get('Binomial', {})
        if not binom_info.get('Fit', False):
            return output_format(bool_result=False, txt="Not Binomial")
        
        is_normal = self.n * self.p >= 5 and self.n * (1 - self.p) >= 5
        binom_info['Normal_approx'] = is_normal
        self.type['Binomial'] = binom_info
        
        return output_format(
            bool_result=is_normal, txt="Can be treated as normal" if is_normal else "Cannot be treated as normal")
        
    def calculate_log_likelihood(self):
        self.validate_data()
        log_likelihood = np.sum(np.log(stats.binom.pmf(self.data, self.n, self.p)))
        return log_likelihood

    def get_num_parameters(self):
        return 2

class PoissonDistribution(Distribution, DistributionTest):
    
    def __init__(self, distribution, lam=None):
        Distribution.__init__(self, distribution.data)
        DistributionTest.__init__(self, distribution)

        self.validate_data()
        self.lam = lam if lam is not None else np.mean(distribution.data)
    def fit_test(self):
        try:
            self.validate_data()
            if len(self.data) == 0:
                return output_format(bool_result=False, txt="No data available")
            
            if not np.all((self.data >= 0) & np.array([isinstance(x, (int, np.integer)) for x in self.data])):
                self.update_type('Poisson', False)
                return output_format(bool_result=False, txt="Data check failed for Poisson")
            
            total = len(self.data)
            unique, counts = np.unique(self.data, return_counts=True)
            obs = dict(zip(unique, counts))

            k_min = int(np.min(self.data))
            k_max = int(np.max(self.data))
            exp = {}

            for k in range(k_min, k_max + 1):
                expected_prob = stats.poisson.pmf(k, self.lam)
                exp[k] = total * expected_prob
            
            k_vals = np.array(sorted(exp.keys()))
            observed = np.array([obs.get(k, 0) for k in k_vals])
            expected = np.array([exp[k] for k in k_vals])

            expected *= observed.sum() / expected.sum()

            chi_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
            result_bool = p_value > 0.05 
            self.type = {
                'Poisson': {
                    'Fit': np.bool_(result_bool),
                    'Normal_approx': None,
                    'Methods': {
                        'Test': 'Chi-square test',
                        'p_value': p_value,
                        'chi_stat': chi_stat
                    }
                }
            }                 
            
            txt =f"Chi-square test: chi2 {chi_stat:.2f}, p = {p_value:.4f}"
            return output_format(bool_result=result_bool, txt=txt, p_value=p_value)

        except Exception as e: 
            raise TypeError(f"Error in PoissonDistribution fit_test: {str(e)}")

    def normal_approximation(self):
        if not self.type.get('Poisson', False):
            return output_format(bool_result=False, txt="Not Poisson")
        is_normal = self.lam >= 30
        self.type['Poisson']['Normal_approx'] = np.bool_(is_normal)
        
        return output_format(
            bool_result=self.type.get('Normal_approx', False),
            txt="Can be treated as normal" if self.type.get('Normal_approx', False) else "Cannot be treated as normal"
        )

    def calculate_log_likelihood(self):
        self.validate_data()
        log_factorial = np.sum(gammaln(self.data + 1))
        log_likelihood = np.sum(self.data * np.log(self.lam) - self.lam - log_factorial)
        return log_likelihood

    def get_num_parameters(self):
        return 1

class LognormalDistribution(Distribution, DistributionTest):
    def __init__(self, distribution):
        Distribution.__init__(self, distribution.data)
        DistributionTest.__init__(self, distribution)

        self.validate_data()
        if not positive_values_test(self.data):
            self.data = self.data[self.data > 0]  # Filtrar solo valores positivos
            if len(self.data) == 0:
                raise ValueError("Data contains no positive values, which are required for Lognormal distribution")

        self.meanlog = np.mean(np.log(self.data))
        self.sdlog = np.std(np.log(self.data))
    
    def fit_test(self):
        try:
            self.validate_data()
            if len(self.data) == 0:
                return output_format({'bool_result': False, 'txt': 'no data available'})     
            
            if np.any(self.data <= 0):
                return output_format({'bool_result': False, 'txt': 'Data contains non-positive values'})
            
            # Aplicar prueba de noirmalidad a log(data)
            log_data = np.log(self.data)
            if np.any(np.isnan(log_data)) or np.any(np.isinf(log_data)):
                return output_format({'bool_result': False, 'txt': 'Numerical instability in log transformation'}) 
            #apply normality test to log data
            anderson_test = stats.anderson(log_data, dist='norm')
            sharpiro_test = stats.shapiro(log_data)
            ks_test = stats.kstest(log_data, 'norm', args=(np.mean(log_data), np.std(log_data)))
       
            # get the p-values
            anderson_p_value = anderson_test.significance_level[np.argmin(np.abs(anderson_test.statistic - anderson_test.critical_values))] / 100
            shapiro_p_value = sharpiro_test.pvalue
            ks_p_value = ks_test.pvalue

            # Fisher combined p-value
            combined_static = -2 * (np.log(anderson_p_value) + np.log(shapiro_p_value) + np.log(ks_p_value))
            combined_p_value = 1 - stats.chi2.cdf(combined_static, df=6)
            result_bool = combined_p_value > 0.05

            self.type = {
                'Lognormal': {
                    'Fit': np.bool_(result_bool),
                    'Normal_approx': None,
                    'Methods': {
                        'Anderson-Darling': {
                            'Statistic': np.float64(anderson_test.statistic),
                            'p_value': np.float64(anderson_p_value),
                            'Decision': anderson_p_value > 0.05
                        },
                        'Shapiro-Wilk': {
                            'Statistic': np.float64(sharpiro_test.statistic),
                            'p_value': np.float64(shapiro_p_value),
                            'Decision': shapiro_p_value > 0.05
                     },
                        'Kolmogorov-Smirnov': {
                            'Statistic': np.float64(ks_test.statistic),
                            'p_value': np.float64(ks_p_value),
                            'Decision': ks_p_value > 0.05
                     },
                        'Combined': {
                            'Statistic': np.float64(combined_static),
                            'p_value': np.float64(combined_p_value),
                            'Decision': result_bool
                        }
                    }
                }
            }
            
            txt = f"Combined p-value: {combined_p_value:.4f} (Anderson: {anderson_p_value:.4f}, Shapiro: {shapiro_p_value:.4f}, KS: {ks_p_value:.4f})"
            return output_format(bool_result=result_bool, txt=txt)

        except Exception as e:
            raise TypeError(f"Error in LognormalDistribution fit_test: {str(e)}")

    def normal_approximation(self):
        if 'Lognormal' not in self.type or not self.type['Lognormal']['Fit']:
            return output_format(bool_result=False, txt="Not Lognormal")
        
        log_data = np.log(self.data)
        #Normal evaluation of log(X)
        anderson_result = self.type['Lognormal']['Methods']['Anderson-Darling']['Decision']
        shapiro_result = self.type['Lognormal']['Methods']['Shapiro-Wilk']['Decision']
        ks_result = self.type['Lognormal']['Methods']['Kolmogorov-Smirnov']['Decision'] 

        normal_test_passed = anderson_result and shapiro_result and ks_result

        #evaluate skewness & curtosis
        skewness = stats.skew(log_data)
        kurtosis = stats.kurtosis(log_data, fisher = True) + 3

        #set the normal approximation
        is_normal = normal_test_passed and (-2 <= skewness <= 2) and (1.5 <= kurtosis <= 4.5)

        self.type['Lognormal']['Normal_approx'] = np.bool_(is_normal)
        self.type['Lognormal']['Methods']['Normal_approx'] = {
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Decision': is_normal
        }
        return output_format(
            bool_result=is_normal,
            txt=f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}, {'Can be treated as normal' if is_normal else 'Cannot be treated as normal'}"
        )
    
    def calculate_log_likelihood(self):
       self.validate_data()
       n = len(self.data)

       log_likelihood = -n / 2 * np.log(2 * np.pi * self.sdlog ** 2) - np.sum((np.log(self.data) - self.meanlog) **2 ) / (2 * self.sdlog ** 2)  

       return log_likelihood 
    
    def get_num_parameters(self):
        return 2

class GoodnessAndFit:
    DISTRIBUTION_CLASSES = [
        NormalDistribution,
        BinomialDistribution,
        PoissonDistribution,
        LognormalDistribution,
    ]

    def __init__(self, distribution_obj, show_warnings=True):
        """
        Inicializa la clase con el objeto Distribution.

        Parámetros:
            distribution_obj (Distribution): Instancia de la clase `Distribution`.
            show_warnings (bool): Indica si se deben mostrar advertencias.
        """
        if not isinstance(distribution_obj, Distribution):
            raise TypeError("The input must be an instance of the Distribution class.")
        
        if not validate_non_nan(distribution_obj.data):
            raise ValueError("The data in the distribution contains NaN or infinite values.")
        
        # Filtrar datos para distribuciones que necesitan valores positivos
        self.distribution_obj = distribution_obj
        #self.distribution_obj.data = positive_values_test(self.distribution_obj.data)
        self.show_warnings = show_warnings
        self.errors = {}

    def evaluate_fit(self):
        """
        Itera sobre todas las subclases de `Distribution`, evalúa las pruebas necesarias,
        y actualiza el atributo `type` del objeto base.

        Retorna:
            dict: Resumen de distribuciones válidas y errores.
         """
        # Fase 1: Preselección con AIC y Log-Likelihood
        preselected_candidates = {}
        for dist_class in self.DISTRIBUTION_CLASSES:
            try:
                # Crear una instancia temporal de la distribución
                temp_instance = dist_class(self.distribution_obj)
                log_likelihood = temp_instance.calculate_log_likelihood()
                aic = temp_instance.calculate_aic()
            
                preselected_candidates[dist_class.__name__] = {
                    "log_likelihood": log_likelihood,
                    "aic": aic,
                    "instance": temp_instance
                }

            except (TypeError, ValueError) as e:
                # Manejar errores y almacenar los mensajes de error
                self.errors[dist_class.__name__] = f"{type(e).__name__} in {dist_class.__name__}: {str(e)}"
            except Exception as e:
                self.errors[dist_class.__name__] = str(e)

        # Filtrar por el menor AIC (mejor candidato según el criterio de información)
        if preselected_candidates:
            sorted_candidates = sorted(preselected_candidates.items(), key=lambda x: x[1]["aic"])
            # Considerar un umbral, por ejemplo, seleccionar las 3 distribuciones con el AIC más bajo
            top_candidates = sorted_candidates[:3]
        else:
            if self.show_warnings:
                warnings.warn("No valid distributions were found based on AIC and Log-Likelihood.", UserWarning)
            return {
                "type": {},
                "errors": self.errors,
            }

        # Fase 2: Validación con Pruebas de Bondad y Ajuste
        final_candidates = {}
        for candidate_name, candidate_info in top_candidates:
            instance = candidate_info["instance"]
            try:
                fit_result = instance.fit_test()
                if "bool_result" in fit_result and fit_result["bool_result"]:
                    final_candidates[candidate_name] = {
                        "fit_test": fit_result,
                        "log_likelihood": candidate_info["log_likelihood"]
                    }
            except (TypeError, ValueError) as e:
                self.errors[candidate_name] = f"{type(e).__name__} in {candidate_name}: {str(e)}"
            except Exception as e:
                self.errors[candidate_name] = str(e)

        # Seleccionar la mejor distribución según las pruebas de bondad y ajuste
        if final_candidates:
            # Escoger la distribución que pasa todas las pruebas y tiene mayor log-likelihood
            best_fit = max(final_candidates.items(), key=lambda x: x[1]["log_likelihood"])
            best_fit_name = best_fit[0]
            best_fit_instance = best_fit[1]

            # Actualizar el tipo de distribución con la mejor
            self.distribution_obj.type = {
                best_fit_name: best_fit_instance["fit_test"],
                "Normal_approx": best_fit_instance.get("normal_approximation", False)
            }
            self.distribution_obj.aic_table = {best_fit_name: best_fit_instance.get("log_likelihood")}
        else:
            if self.show_warnings:
                warnings.warn("No valid distributions were found after applying goodness of fit tests.", UserWarning)

        return {
            "type": self.distribution_obj.type,
            "errors": self.errors,
        }
