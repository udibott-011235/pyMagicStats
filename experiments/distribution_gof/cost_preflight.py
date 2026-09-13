"""Mechanical, non-calibratory cost projection for CP05 runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .manifest import ExperimentManifest
from .runner import owned_raw_indices


@dataclass(frozen=True, slots=True)
class CostProjection:
    claim_status: str
    outer_units: int
    expected_fitting_calls: int
    expected_inner_attempts: int
    projected_runtime_seconds: float
    estimated_peak_sample_bytes_per_worker: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_cost(
    manifests: Iterable[ExperimentManifest],
    *,
    seconds_per_fit: float,
    seconds_per_statistic: float,
    expected_nb_inner_eligibility: float = 1.0,
) -> CostProjection:
    if not math.isfinite(seconds_per_fit) or seconds_per_fit < 0:
        raise ValueError("seconds_per_fit must be finite and non-negative")
    if not math.isfinite(seconds_per_statistic) or seconds_per_statistic < 0:
        raise ValueError("seconds_per_statistic must be finite and non-negative")
    if not math.isfinite(expected_nb_inner_eligibility) or not 0 < expected_nb_inner_eligibility <= 1:
        raise ValueError("expected NB inner eligibility must be in (0, 1]")

    outer_units = fitting_calls = inner_attempts = statistic_calls = peak_bytes = 0
    for manifest in manifests:
        units = len(owned_raw_indices(manifest))
        outer_units += units
        attempts_per_outer = manifest.B
        if manifest.family == "negative_binomial" and manifest.null_type == "composite":
            attempts_per_outer = min(
                100 * manifest.B,
                math.ceil(manifest.B / expected_nb_inner_eligibility),
            )
        inner_attempts += units * attempts_per_outer
        statistic_calls += units * (1 + attempts_per_outer)
        if manifest.null_type == "composite":
            fitting_calls += units * (1 + attempts_per_outer)
        item_size = 8
        peak_bytes = max(
            peak_bytes,
            manifest.worker_spec["workers"] * manifest.n * item_size * 3,
        )
    runtime = fitting_calls * seconds_per_fit + statistic_calls * seconds_per_statistic
    return CostProjection(
        claim_status="NON_CALIBRATION_NON_CLAIMING",
        outer_units=outer_units,
        expected_fitting_calls=fitting_calls,
        expected_inner_attempts=inner_attempts,
        projected_runtime_seconds=float(runtime),
        estimated_peak_sample_bytes_per_worker=peak_bytes,
    )
