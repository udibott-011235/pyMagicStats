"""Paired Monte Carlo calibration harness for Student t versus raw EL."""

from .scenarios import (
    HOLDOUT_POLICY_VERSION,
    HoldoutViolation,
    canonical_cells,
    scenario_registry,
)

__all__ = [
    "HOLDOUT_POLICY_VERSION",
    "HoldoutViolation",
    "canonical_cells",
    "scenario_registry",
]
