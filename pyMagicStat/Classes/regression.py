# regression.py

import statsmodels.api as sm
from pyMagicStat.Classes.distributions import Distribution, NormalDistribution
from pyMagicStat.lib.utils import output_format, plot_distribution_summary

class RegressionModel:
    """
    Modelo de regresión lineal para análisis explicativo/investigativo y predictivo.
    - Variables categóricas deben predefinirse y envolverlas en C() en la fórmula antes de instanciar.
    - Uso principal: investigación de relaciones y diagnóstico de ajustes.
    - Modo predictivo disponible vía método .predict().

    Este módulo también expone compute_metrics() para su uso en pipelines automatizados,
    permitiendo evaluar condiciones como residuos normalmente distribuidos y umbrales de R².
    """
    def __init__(self, data, formula, target=None):
        """
        Inicializa el modelo:
          data: pandas DataFrame con variables numéricas y/o categóricas.
          formula: string de statsmodels (e.g., 'y ~ x1 + C(cat_var)').
          target: nombre de la variable respuesta (opcional).
        NOTA: Prepara el ajuste y residuos, pero NO evalúa normalidad aquí.
        """
        self.data = data
        self.formula = formula
        self.model = sm.OLS.from_formula(formula, data).fit()
        self.target = target or formula.split('~')[0].strip()

        # Valores ajustados y residuos
        self.fitted_values = self.model.fittedvalues
        self.residuals = self.model.resid

        # Envuelve residuos en Distribution (sin evaluar normalidad automáticamente)
        self.dist = Distribution(self.residuals)

        # Inicializar métricas
        self.r_squared = None
        self.adj_r_squared = None
        self.aic = None
        self.bic = None
        self.residual_normality = None

    def _detect_categorical_bases(self):
        """
        Detecta variables indicadoras creadas por C() y su respectiva categoría base.
        """
        cats = {}
        for name in self.model.params.index:
            if name.startswith("C("):
                var = name.split("[")[0][2:-1]
                cat = name.split("[T.")[1][:-1]
                cats.setdefault(var, []).append(cat)
        lines = []
        for var, present in cats.items():
            levels = sorted(self.data[var].dropna().unique())
            base = next((c for c in levels if c not in present), None)
            lines.append(f"Categoría base para '{var}': {base}")
        return "\n".join(lines)

    def get_equation(self, expand=False):
        """
        Construye la ecuación interpretativa del modelo:
          y = intercept + sum(coef * [var])
        Incluye información de categorías base. Si expand=True, genera ecuaciones por nivel.
        """
        intercept = self.model.params.get("Intercept", 0)
        expr = f"{self.target} = {intercept:.3f}"
        for name, coef in self.model.params.items():
            if name == "Intercept":
                continue
            expr += f" + {coef:.3f}*[{name}]"
        parts = [expr, self._detect_categorical_bases()]
        if expand:
            # Aquí podría implementarse la generación de ecuaciones separadas por nivel
            pass
        return "\n".join(parts)

    def compute_metrics(self):
        """
        Calcula y actualiza métricas del modelo:
          - R² (rsquared) y R² ajustado (rsquared_adj)
          - AIC y BIC
          - Diagnóstico de normalidad de residuos
        Devuelve un dict con estas métricas para uso en pipelines.
        """
        self.r_squared = self.model.rsquared
        self.adj_r_squared = self.model.rsquared_adj
        self.aic = self.model.aic
        self.bic = self.model.bic

        normal_validator = NormalDistribution(self.residuals.to_numpy())
        self.residual_normality = normal_validator.evaluate_normality()

        return {
            'r_squared': self.r_squared,
            'adj_r_squared': self.adj_r_squared,
            'aic': self.aic,
            'bic': self.bic,
            'residual_normality': self.residual_normality
        }

    def summary(self, verbose=True):
        """
        Genera un resumen investigativo:
          - Ecuación interpretativa
          - Estadísticas de ajuste (R², AIC, etc.) si verbose=True
          - Diagnóstico de normalidad de residuos (evaluate_normality)
          - Visualización de la distribución de residuos
        También actualiza métricas internas para uso programático.
        Devuelve los resultados formateados con output_format().
        """
        # Actualizar métricas
        metrics = self.compute_metrics()

        # Ecuación interpretativa
        eq = self.get_equation()
        fit_stats = self.model.summary() if verbose else None

        # Visualización de la distribución de residuos
        plot_distribution_summary(
            data=self.residuals,
            stats=self.residual_normality,
            distribution_type="Residuals"
        )

        # Preparar resultado combinado
        result = {
            'equation': eq,
            'fit_stats': str(fit_stats) if verbose else None,
            **metrics
        }
        return output_format(data=result)

    def predict(self, new_data):
        """
        Predice la variable respuesta para un nuevo DataFrame new_data.
        new_data debe incluir las mismas variables (con dummies si corresponde).
        """
        return self.model.predict(new_data)
