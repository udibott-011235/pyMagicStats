from pyMagicStat.distributionOrchestrator import GoodnessAndFit
from pyMagicStat.confidence_intervals import (
    BootstrapConfidenceIntervals,
    NormalDistConfidenceIntervals,
    PopulationMeanCI,
    PopulationProportionCI,
    PopulationVarianceCI,
)

class StatisticDecision:
    """
    Clase para decidir qué método estadístico aplicar basado en la distribución y el estadístico solicitado.
    """

    def __init__(self, distribution_obj):
        """
        Inicializa la clase con un objeto de distribución.

        Parámetros:
            distribution_obj (Distribution): Objeto de la clase `Distribution`.
        """
        self.distribution_obj = distribution_obj
        self.method = None
        self.result = None

    def _evaluate_distribution(self):
        """
        Evalúa la distribución usando GoodnessAndFit si type está vacío.
        """
        if not self.distribution_obj.type:
            goodness = GoodnessAndFit(self.distribution_obj)
            goodness.test()

    def decide_method(self, statistic):
        """
        Decide qué método aplicar basado en la distribución y el estadístico solicitado.

        Parámetros:
            statistic (str): Estadístico solicitado (ej., 'mean', 'proportion').

        Retorna:
            str: Tipo de método seleccionado ('parametric', 'non-parametric', 'bootstrap').
        """
        self._evaluate_distribution()

        # Determinar si se pueden aplicar métodos paramétricos
        is_parametric = any(
            key.endswith("NormalApprox") or key == "NormalDistribution"
            for key in self.distribution_obj.type
        )

        if is_parametric:
            if statistic == "mean":
                self.method = PopulationMeanCI(self.distribution_obj).calculate
                return "parametric"
            elif statistic == "proportion":
                self.method = PopulationProportionCI(self.distribution_obj).calculate
                return "parametric"
            elif statistic == "variance":
                self.method = PopulationVarianceCI(self.distribution_obj).calculate
                return "parametric"
            else:
                raise ValueError(f"Estadístico '{statistic}' no soportado en métodos paramétricos.")
        else:
            # Si no es paramétrico, usar Bootstrap como método alternativo
            self.method = BootstrapConfidenceIntervals(self.distribution_obj).calculate
            return "bootstrap"

    def calculate(self, statistic):
        """
        Calcula el estadístico solicitado usando el método determinado.

        Parámetros:
            statistic (str): Estadístico a calcular (ej., 'mean', 'variance').

        Retorna:
            dict: Resultado del cálculo.
        """
        if not self.method:
            self.decide_method(statistic)

        # Ejecutar el método seleccionado
        self.result = self.method()
        return self.result
