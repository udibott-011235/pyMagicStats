"""Independent reference implementations for CP05-B software gates."""

from .continuous_reference import continuous_simple_null_reference
from .negative_binomial_high_precision import high_precision_nb_statistic

__all__ = ["continuous_simple_null_reference", "high_precision_nb_statistic"]
