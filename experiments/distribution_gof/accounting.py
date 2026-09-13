"""Typed terminal states and audit records for CP05-B."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any


class AssessmentStatus(str, Enum):
    ASSESSED = "ASSESSED"
    NOT_ASSESSED = "NOT_ASSESSED"
    FAILED = "FAILED"


class ReasonCode(str, Enum):
    ASSESSMENT_COMPLETE = "ASSESSMENT_COMPLETE"
    MATHEMATICAL_INELIGIBILITY = "MATHEMATICAL_INELIGIBILITY"
    RETRY_CAP_EXHAUSTED = "RETRY_CAP_EXHAUSTED"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    FIT_NUMERICAL_FAILURE = "FIT_NUMERICAL_FAILURE"
    FIT_BACKEND_FAILURE = "FIT_BACKEND_FAILURE"
    STATISTIC_FAILURE = "STATISTIC_FAILURE"
    TAIL_CERTIFICATION_FAILURE = "TAIL_CERTIFICATION_FAILURE"
    ORACLE_MISMATCH = "ORACLE_MISMATCH"
    ARTIFACT_INTEGRITY_FAILURE = "ARTIFACT_INTEGRITY_FAILURE"
    COMPARATOR_NOT_ASSESSABLE = "COMPARATOR_NOT_ASSESSABLE"


@dataclass(slots=True)
class OuterResult:
    schema_version: str
    canonical_cell_id: str
    raw_outer_index: int
    eligible_outer_index: int | None
    seed_identity: str
    status: str
    reason_code: str
    observed_fit_status: str
    statistic_value: float | None
    T_obs: float | None
    exceedance_count: int | None
    p_mc: float | None
    reject: bool | None
    inner_attempts: int
    inner_eligible: int
    inner_ineligible: int
    failure_class: str | None
    observed_mle_eligible: bool | None
    outer_ineligibility_reason: str | None
    inner_ineligibility_reason_counts: dict[str, int]
    tail_remainder_bound: float | None
    oracle_value: float | None
    oracle_abs_error: float | None
    oracle_relative_scaled_error: float | None
    observed_fit_provenance: dict[str, Any] | None
    observed_fit_calls: int
    replicate_fit_calls: int
    raw_inner_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            float64_value=self.statistic_value,
            absolute_error=self.oracle_abs_error,
            relative_scaled_error=self.oracle_relative_scaled_error,
            remainder_bound=self.tail_remainder_bound,
        )
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        payload["output_digest"] = hashlib.sha256(encoded).hexdigest()
        return payload


@dataclass(frozen=True, slots=True)
class InnerAttempt:
    canonical_cell_id: str
    raw_outer_index: int
    raw_inner_index: int
    seed_identity: str
    eligible: bool
    reason_code: str
    fit_calls: int
    statistic_value: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
