"""Authoritative non-holdout scenario registry for the EL-versus-t study.

The distribution objects and cell matrix come directly from the existing
adversarial one-sample robustness calibration.  This module adds only a
fail-closed holdout boundary and stable registry metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Iterable, Mapping

from experiments.adversarial_robustness_calibration import (
    Scenario,
    calibration_plan as existing_calibration_plan,
    scenario_catalog as existing_scenario_catalog,
)


HOLDOUT_POLICY_VERSION = "sealed-blind-holdout-v1"
RESERVED_SAMPLE_SIZES = frozenset({6, 12, 25, 35, 45, 65, 90, 150, 350, 1000, 5000})
RESERVED_LOGNORMAL_SIGMA = (0.35, 0.75, 1.25)
RESERVED_STUDENT_T_DF = (4.0, 7.0, 15.0)
RESERVED_CONTAMINATION_EPSILON = (0.003, 0.015, 0.04)
RESERVED_FAMILIES = frozenset({"laplace", "weibull", "pareto", "beta"})
_FLOAT_ABS_TOLERANCE = 1e-12


class HoldoutViolation(ValueError):
    """Raised before generation when a request intersects the sealed holdout."""


@dataclass(frozen=True)
class ExperimentCell:
    """One canonical scenario/sample-size calibration cell."""

    scenario: Scenario
    n: int
    evidence_tier: str

    @property
    def cell_id(self) -> str:
        return f"{self.scenario.name}__n_{self.n}"

    def to_metadata(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "scenario_id": self.scenario.name,
            "family": self.scenario.family,
            "parameters": dict(self.scenario.parameters),
            "population_mean": self.scenario.population_mean,
            "n": self.n,
            "source_evidence_tier": self.evidence_tier,
        }


def active_holdout_policy() -> dict[str, object]:
    """Return the complete, machine-readable exclusion policy."""

    return {
        "policy_version": HOLDOUT_POLICY_VERSION,
        "mode": "fail_closed_before_generation",
        "reserved_sample_sizes": sorted(RESERVED_SAMPLE_SIZES),
        "reserved_lognormal_sigma": list(RESERVED_LOGNORMAL_SIGMA),
        "reserved_student_t_df": [int(value) for value in RESERVED_STUDENT_T_DF],
        "reserved_contamination_epsilon": list(RESERVED_CONTAMINATION_EPSILON),
        "reserved_families": sorted(RESERVED_FAMILIES),
    }


def _matches_reserved(value: object, reserved: Iterable[float]) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return any(
        math.isclose(number, candidate, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOLERANCE)
        for candidate in reserved
    )


def validate_sample_size(n: int) -> None:
    """Fail closed for a reserved sample size."""

    if isinstance(n, bool) or int(n) != n or int(n) < 2:
        raise ValueError("sample size must be an integer of at least 2")
    if int(n) in RESERVED_SAMPLE_SIZES:
        raise HoldoutViolation(f"sample size {n} is reserved by {HOLDOUT_POLICY_VERSION}")


def validate_scenario_definition(
    *,
    family: str,
    parameters: Mapping[str, object],
    scenario_id: str = "<requested>",
) -> None:
    """Fail closed when a scenario definition intersects any reserved axis."""

    normalized_family = str(family).strip().casefold()
    if normalized_family in RESERVED_FAMILIES:
        raise HoldoutViolation(
            f"scenario {scenario_id!r} uses reserved family {family!r}"
        )
    lowered = {str(key).casefold(): value for key, value in parameters.items()}
    if normalized_family == "lognormal" and _matches_reserved(
        lowered.get("sigma"), RESERVED_LOGNORMAL_SIGMA
    ):
        raise HoldoutViolation(f"scenario {scenario_id!r} uses reserved lognormal sigma")
    if normalized_family == "student_t" and _matches_reserved(
        lowered.get("df"), RESERVED_STUDENT_T_DF
    ):
        raise HoldoutViolation(f"scenario {scenario_id!r} uses reserved Student-t df")
    if "contamination" in normalized_family and _matches_reserved(
        lowered.get("epsilon"), RESERVED_CONTAMINATION_EPSILON
    ):
        raise HoldoutViolation(
            f"scenario {scenario_id!r} uses reserved contamination epsilon"
        )


def validate_scenario(scenario: Scenario) -> None:
    validate_scenario_definition(
        family=scenario.family,
        parameters=scenario.parameters,
        scenario_id=scenario.name,
    )


def _number_from_name(pattern: str, scenario_id: str) -> float | None:
    match = re.search(pattern, scenario_id.casefold())
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))


def validate_requested_scenario_id(scenario_id: str) -> None:
    """Protect the CLI even when a reserved name is not in the registry."""

    normalized = scenario_id.strip().casefold()
    family_token = normalized.split("_", 1)[0]
    if family_token in RESERVED_FAMILIES:
        raise HoldoutViolation(f"requested scenario uses reserved family {family_token!r}")
    sigma = _number_from_name(r"lognormal(?:_sigma)?_([0-9]+(?:[p.][0-9]+)?)", normalized)
    if sigma is not None and _matches_reserved(sigma, RESERVED_LOGNORMAL_SIGMA):
        raise HoldoutViolation("requested scenario uses reserved lognormal sigma")
    df = _number_from_name(r"student_t_df_([0-9]+(?:[p.][0-9]+)?)", normalized)
    if df is not None and _matches_reserved(df, RESERVED_STUDENT_T_DF):
        raise HoldoutViolation("requested scenario uses reserved Student-t df")
    epsilon = _number_from_name(r"(?:eps|epsilon)_([0-9]+(?:[p.][0-9]+)?)", normalized)
    if epsilon is not None and _matches_reserved(
        epsilon, RESERVED_CONTAMINATION_EPSILON
    ):
        raise HoldoutViolation("requested scenario uses reserved contamination epsilon")


def scenario_registry() -> tuple[Scenario, ...]:
    """Return the canonical existing catalog after enforcing the holdout guard."""

    registry = tuple(existing_scenario_catalog())
    for scenario in registry:
        validate_scenario(scenario)
    return registry


def canonical_cells() -> tuple[ExperimentCell, ...]:
    """Reuse the complete existing calibration matrix with no new cells."""

    cells = tuple(
        ExperimentCell(cell.scenario, int(cell.n), cell.evidence_tier)
        for cell in existing_calibration_plan()
    )
    for cell in cells:
        validate_scenario(cell.scenario)
        validate_sample_size(cell.n)
    return cells


def select_cells(
    scenario_ids: Iterable[str] | None = None,
    sample_sizes: Iterable[int] | None = None,
) -> tuple[ExperimentCell, ...]:
    """Select only canonical cells, validating requested axes before lookup."""

    requested_scenarios = None
    if scenario_ids is not None:
        requested_scenarios = {str(value) for value in scenario_ids}
        for scenario_id in requested_scenarios:
            validate_requested_scenario_id(scenario_id)
    requested_sizes = None
    if sample_sizes is not None:
        requested_sizes = {int(value) for value in sample_sizes}
        for n in requested_sizes:
            validate_sample_size(n)

    registry_ids = {scenario.name for scenario in scenario_registry()}
    if requested_scenarios is not None:
        unknown = requested_scenarios - registry_ids
        if unknown:
            raise ValueError(f"unknown canonical scenario(s): {sorted(unknown)}")

    selected = tuple(
        cell
        for cell in canonical_cells()
        if (requested_scenarios is None or cell.scenario.name in requested_scenarios)
        and (requested_sizes is None or cell.n in requested_sizes)
    )
    if not selected:
        raise ValueError("the requested filters select no canonical calibration cells")
    return selected


def registry_digest(cells: Iterable[ExperimentCell]) -> str:
    """Stable digest used to reject incompatible shards."""

    payload = [cell.to_metadata() for cell in cells]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
