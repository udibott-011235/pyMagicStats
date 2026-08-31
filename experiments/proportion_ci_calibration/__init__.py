"""CP-06 calibration harness for one-sample proportion intervals."""

from .harness import (
    ALPHAS,
    CANDIDATE_SHA,
    METHODS,
    calibrate_n,
    coverage_from_intervals,
    expected_width_matrix,
    production_interval_grid,
)

__all__ = [
    "ALPHAS",
    "CANDIDATE_SHA",
    "METHODS",
    "calibrate_n",
    "coverage_from_intervals",
    "expected_width_matrix",
    "production_interval_grid",
]
