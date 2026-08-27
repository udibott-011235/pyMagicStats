import statsmodels.api as sm
import pandas as pd
from typing import Any, Dict, Optional, Union
from pyMagicStat.distributions.distributions import Distribution, NormalDistribution
from pyMagicStat.utils.utils import output_format
from pyMagicStat.viz.plots import plot_distribution_summary

class RegressionModel:
    """
    Modelo de regresión lineal para análisis explicativo/investigativo y predictivo.

    Variables categóricas deben predefinirse y envolverlas en C() en la fórmula antes de instanciar.
    Uso principal: investigación de relaciones y diagnóstico de ajustes.
    Modo predictivo disponible vía método .predict().

    Este módulo también expone compute_metrics() para su uso en pipelines automatizados,
    permitiendo evaluar condiciones como residuos normalmente distribuidos y umbrales de R².

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame de pandas con variables numéricas y/o categóricas.
    formula : str
        Fórmula en formato de R/statsmodels (e.g., 'y ~ x1 + C(cat_var)').
    target : str, optional
        Nombre de la variable respuesta. Si no se provee, se infiere de la fórmula.
    """
    def __init__(self, data: pd.DataFrame, formula: str, target: Optional[str] = None) -> None:
        self.data: pd.DataFrame = data
        self.formula: str = formula
        self.model = sm.OLS.from_formula(formula, data).fit()
        self.target: str = target or formula.split('~')[0].strip()

        # Valores ajustados y residuos
        self.fitted_values: pd.Series = self.model.fittedvalues
        self.residuals: pd.Series = self.model.resid

        # Envuelve residuos en Distribution (sin evaluar normalidad automáticamente)
        self.dist: Distribution = Distribution(self.residuals.to_numpy())

        # Inicializar métricas
        self.r_squared: Optional[float] = None
        self.adj_r_squared: Optional[float] = None
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.residual_normality: Optional[Dict[str, Any]] = None

    def _detect_categorical_bases(self) -> str:
        """
        Detecta variables indicadoras creadas por C() y su respectiva categoría base.

        Returns
        -------
        str
            Cadena de texto multilinea con la información de la categoría base de cada variable.
        """
        cats: Dict[str, list] = {}
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

    def get_equation(self, expand: bool = False) -> str:
        """
        Construye la ecuación interpretativa del modelo:
          y = intercept + sum(coef * [var])
        Incluye información de categorías base.

        Parameters
        ----------
        expand : bool, default=False
            Si True, genera ecuaciones por nivel (no implementado completamente).

        Returns
        -------
        str
            Ecuación resultante del modelo.
        """
        intercept = float(self.model.params.get("Intercept", 0))
        expr = f"{self.target} = {intercept:.3f}"
        for name, coef in self.model.params.items():
            if name == "Intercept":
                continue
            expr += f" + {float(coef):.3f}*[{name}]"
        parts = [expr, self._detect_categorical_bases()]
        if expand:
            pass
        return "\n".join(parts)

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Calcula y actualiza métricas del modelo (R², AIC, BIC, normalidad de residuos).

        Returns
        -------
        Dict[str, Any]
            Diccionario con las métricas para uso en pipelines.
        """
        self.r_squared = float(self.model.rsquared)
        self.adj_r_squared = float(self.model.rsquared_adj)
        self.aic = float(self.model.aic)
        self.bic = float(self.model.bic)

        normal_validator = NormalDistribution(self.residuals.to_numpy())
        self.residual_normality = normal_validator.evaluate_normality()

        return {
            'r_squared': self.r_squared,
            'adj_r_squared': self.adj_r_squared,
            'aic': self.aic,
            'bic': self.bic,
            'residual_normality': self.residual_normality
        }

    def summary(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Genera un resumen investigativo del modelo y renderiza gráficos diagnósticos.

        Parameters
        ----------
        verbose : bool, default=True
            Si True, incluye las estadísticas completas de ajuste en la salida.

        Returns
        -------
        Dict[str, Any]
            Resultados formateados mediante `output_format()`.
        """
        metrics = self.compute_metrics()
        eq = self.get_equation()
        fit_stats = str(self.model.summary()) if verbose else None

        plot_distribution_summary(
            data=self.residuals.to_numpy(),
            stats=self.residual_normality,
            distribution_type="Residuals"
        )

        result: Dict[str, Any] = {
            'equation': eq,
            'fit_stats': fit_stats,
            **metrics
        }
        return output_format(data=result)

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        """
        Predice la variable respuesta para un nuevo DataFrame.

        Parameters
        ----------
        new_data : pd.DataFrame
            Nuevos datos. Deben contener las mismas columnas utilizadas en la fórmula.

        Returns
        -------
        pd.Series
            Valores predichos.
        """
        return self.model.predict(new_data)
