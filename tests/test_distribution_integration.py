import warnings

import numpy as np
import pandas as pd

from pyMagicStat.distributions.distributions import NormalDistribution
from pyMagicStat.models.regression import RegressionModel


def test_normal_distribution_uses_structured_shape_assessment_without_ks_warning():
    data = np.random.default_rng(42).normal(size=100)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        evaluator = NormalDistribution(data)
        result = evaluator.evaluate_normality()

    assert not caught
    assert result["assessment"]["status"] in {"pass", "warn"}
    assert "skewness" in result["shape"]
    assert "KS" not in result
    assert evaluator.distribution.type["Normal"] is True


def test_regression_residuals_use_the_shared_shape_contract():
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 10.0, 80)
    data = pd.DataFrame({"x": x, "y": 3.0 + 2.0 * x + rng.normal(size=x.size)})

    metrics = RegressionModel(data, "y ~ x").compute_metrics()

    residual_shape = metrics["residual_normality"]
    assert residual_shape["name"] == "shape_residuals"
    assert residual_shape["status"] in {"pass", "warn"}
    assert "shapiro_p_value" in residual_shape["metrics"]
