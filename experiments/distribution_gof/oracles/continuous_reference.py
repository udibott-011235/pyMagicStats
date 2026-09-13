"""External SciPy references where its simple-null contract coincides exactly."""

from __future__ import annotations

import numpy as np
from scipy import stats


def continuous_simple_null_reference(sample, bound, statistic: str) -> float:
    values = np.asarray(sample, dtype=np.float64)
    backend = bound.family._backend(bound.parameters)
    if statistic == "CVM":
        return float(stats.cramervonmises(values, backend.cdf).statistic)
    if statistic == "KS":
        return float(stats.kstest(values, backend.cdf, method="exact").statistic)
    raise ValueError("external continuous reference is defined only for CVM and KS")
