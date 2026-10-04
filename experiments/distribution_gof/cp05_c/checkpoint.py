"""Atomic index plus immutable outer segments; resume only committed prefixes."""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import re

from ..accounting import OuterResult, InnerAttempt
from ..artifacts import (ArtifactIntegrityError, write_json_atomic, _sha256_file,
                         _checkpoint_record_data)
from ..manifest import canonical_json
from ..seed_derivation import derive_seed
from .manifest import OUTPUT_SCHEMA

CHECKPOINT_SCHEMA = "cp05-c-prefix-checkpoint-v1"


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def initial_state(manifest, target, fixture):
    return {"schema_version": CHECKPOINT_SCHEMA, "manifest_identity": manifest.resume_identity,
            "target": target, "software_fixture": fixture, "segments": [], "status": "RUNNING",
            "failure_reason": None, "cell_execution_started": False, "execution_wall_seconds": 0.0,
            "shadow": {"entry_gate": "NOT_APPLICABLE", "periodic_gate": "NOT_APPLICABLE",
                       "checks": [], "shadow_wall_seconds": 0.0,
                       "canonical_shadow_fit_count": 0, "failure_reason": None}}


def save_state(directory, state):
    write_json_atomic(Path(directory) / "checkpoint.json", {**state, "checkpoint_digest": _digest(state)})


def save_segment(directory, result, attempts, wall_seconds):
    name = f"checkpoints/outer-{result.raw_outer_index:06d}.json"
    payload = {"outer": result.to_dict(), "inner": [a.to_dict() for a in attempts],
               "primary_wall_seconds": wall_seconds}
    write_json_atomic(Path(directory) / name, payload)
    return {"raw_outer_index": result.raw_outer_index, "path": name,
            "sha256": _sha256_file(Path(directory) / name)}


def validate_unit(manifest, result, attempts, expected_index, eligible_index):
    def require(condition, message):
        if not condition:
            raise ArtifactIntegrityError(message)
    require(result.schema_version == OUTPUT_SCHEMA, "outer schema mismatch")
    require(result.canonical_cell_id == manifest.canonical_cell_id, "outer cell mismatch")
    require(result.raw_outer_index == expected_index, "outer raw prefix mismatch")
    require(type(result.raw_outer_index) is int and (result.eligible_outer_index is None
            or type(result.eligible_outer_index) is int), "outer index type mismatch")
    require(result.seed_identity == derive_seed(manifest.canonical_cell_id, expected_index,
                                                "outer_observed").digest_hex, "outer seed mismatch")
    require(result.status in ("ASSESSED", "NOT_ASSESSED", "FAILED"), "invalid outer status")
    require(result.eligible_outer_index == (eligible_index if result.status == "ASSESSED" else None),
            "eligible outer index mismatch")
    require(result.inner_attempts == len(attempts) and result.raw_inner_indices == list(range(len(attempts))),
            "inner attempt indices mismatch")
    require(len(attempts) <= manifest.B * (100 if manifest.nb_composite else 1), "inner cap exceeded")
    require(result.inner_eligible == sum(a.eligible for a in attempts), "inner eligible count mismatch")
    require(result.replicate_fit_calls == sum(a.fit_calls for a in attempts), "inner fit count mismatch")
    for index, attempt in enumerate(attempts):
        require(attempt.canonical_cell_id == manifest.canonical_cell_id
                and attempt.raw_outer_index == expected_index and attempt.raw_inner_index == index,
                "inner identity mismatch")
        require(attempt.seed_identity == derive_seed(manifest.canonical_cell_id, expected_index,
                                                     "inner_bootstrap", index).digest_hex,
                "inner seed mismatch")
        require(type(attempt.eligible) is bool and attempt.fit_calls in (0, 1), "inner schema mismatch")
    if manifest.null_type == "simple":
        require(result.observed_fit_calls == result.replicate_fit_calls == 0, "simple null fit contamination")
    if result.status == "NOT_ASSESSED":
        require(manifest.nb_composite and result.reason_code == "MATHEMATICAL_INELIGIBILITY"
                and result.observed_mle_eligible is False and not attempts,
                "silent replacement of non-mathematical outer failure")
    if result.status == "ASSESSED":
        require(type(result.reject) is bool and type(result.p_mc) is float
                and type(result.exceedance_count) is int, "assessed Monte Carlo scalar types mismatch")
        require(type(result.T_obs) in (int, float) and math.isfinite(result.T_obs), "nonfinite observed statistic")
        require(result.inner_eligible == manifest.B and result.observed_mle_eligible is True,
                "incomplete assessed unit")
        eligible_values = [a.statistic_value for a in attempts if a.eligible]
        require(all(type(v) in (int, float) and math.isfinite(v) for v in eligible_values), "nonfinite inner statistic")
        exceedances = sum(v >= result.T_obs for v in eligible_values)
        require(result.exceedance_count == exceedances
                and result.p_mc == (exceedances + 1) / (manifest.B + 1)
                and result.reject == (result.p_mc <= manifest.alpha), "Monte Carlo result mismatch")
    else:
        require(result.p_mc is result.reject is result.exceedance_count is None, "unassessed Monte Carlo contamination")


def load_state(directory, manifest, target, fixture):
    path = Path(directory) / "checkpoint.json"
    if not path.exists():
        return initial_state(manifest, target, fixture), [], [], []
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        digest = state.pop("checkpoint_digest")
        if digest != _digest(state):
            raise ArtifactIntegrityError("checkpoint digest mismatch")
        if (state["schema_version"] != CHECKPOINT_SCHEMA
                or state["manifest_identity"] != manifest.resume_identity
                or state["target"] != target or state["software_fixture"] is not fixture):
            raise ArtifactIntegrityError("checkpoint scientific identity mismatch")
        rows, inner, timings, eligible = [], [], [], 0
        for index, segment in enumerate(state["segments"]):
            if (segment["raw_outer_index"] != index
                    or segment["path"] != f"checkpoints/outer-{index:06d}.json"):
                raise ArtifactIntegrityError("duplicate, missing or unsafe checkpoint segment")
            segment_path = Path(directory) / segment["path"]
            if _sha256_file(segment_path) != segment["sha256"]:
                raise ArtifactIntegrityError("checkpoint segment digest mismatch")
            payload = json.loads(segment_path.read_text(encoding="utf-8"))
            data = _checkpoint_record_data(payload["outer"])
            if set(data) != {f.name for f in fields(OuterResult)}:
                raise ArtifactIntegrityError("checkpoint outer schema mismatch")
            result = OuterResult(**data)
            attempts = tuple(InnerAttempt(**a) for a in payload["inner"])
            validate_unit(manifest, result, attempts, index, eligible)
            if any(r.status == "FAILED" for r in rows):
                raise ArtifactIntegrityError("outer execution continued after failure")
            eligible += result.status == "ASSESSED"
            rows.append(result)
            inner.extend(attempts)
            wall = payload["primary_wall_seconds"]
            if type(wall) not in (int, float) or not 0 <= wall < float("inf"):
                raise ArtifactIntegrityError("invalid primary timing")
            timings.append(wall)
        if eligible > target or len(rows) > target * (100 if manifest.nb_composite else 1):
            raise ArtifactIntegrityError("checkpoint exceeds outer target/cap")
        validate_shadow_state(manifest, state, len(rows))
        if state["status"] == "COMPLETE" and eligible != target:
            raise ArtifactIntegrityError("false completed checkpoint")
        if eligible == target and state["status"] != "COMPLETE":
            raise ArtifactIntegrityError("completed prefix has inconsistent terminal status")
        if len(rows) == target * (100 if manifest.nb_composite else 1) and eligible < target and state["status"] != "FAILED":
            raise ArtifactIntegrityError("exhausted outer cap has inconsistent terminal status")
        if rows and rows[-1].status == "FAILED" and state["status"] != "FAILED":
            raise ArtifactIntegrityError("failed unit in nonfailed checkpoint")
        if state["status"] not in ("RUNNING", "PAUSED", "COMPLETE", "FAILED", "FAILED_SHADOW_ORACLE"):
            raise ArtifactIntegrityError("unknown checkpoint status")
        return state, rows, inner, timings
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError("checkpoint is missing or invalid") from exc


def validate_shadow_state(manifest, state, raw_count):
    shadow = state["shadow"]
    if not manifest.nb_composite:
        if shadow["checks"] or shadow["entry_gate"] != "NOT_APPLICABLE":
            raise ArtifactIntegrityError("shadow contamination of non-NB-composite cell")
        return
    if shadow["canonical_shadow_fit_count"] != len(shadow["checks"]):
        raise ArtifactIntegrityError("shadow fit count mismatch")
    if shadow["entry_gate"] not in ("PASS", "FAIL") or shadow["periodic_gate"] not in ("PASS", "FAIL"):
        raise ArtifactIntegrityError("invalid shadow status")
    if shadow.get("NB_TRANSFORM_ATOL") != 1e-8 or shadow.get("NB_OBJECTIVE_RTOL") != 1e-9:
        raise ArtifactIntegrityError("shadow fitting thresholds changed")
    if not 0 <= shadow["shadow_wall_seconds"] < float("inf"):
        raise ArtifactIntegrityError("invalid shadow timing")
    seen = set()
    for row in shadow["checks"]:
        outer, inner, role = row["raw_outer_index"], row["raw_inner_index"], row["role"]
        key = (role, outer, inner)
        if key in seen or row["canonical_cell_id"] != manifest.canonical_cell_id:
            raise ArtifactIntegrityError("duplicate/wrong-cell shadow check")
        seen.add(key)
        if (not isinstance(row["sample_digest"], str) or not re.fullmatch("[0-9a-f]{64}", row["sample_digest"])
                or row["sample_shape"] != [manifest.n]):
            raise ArtifactIntegrityError("invalid shadow sample identity")
        from .shadow import validate_evidence_row
        validate_evidence_row(row)
        if (role == "observed" and (inner is not None or not 0 <= outer < 64)
                or role == "inner" and (type(inner) is not int or not 0 <= inner < 64)
                or role == "periodic_observed" and (inner is not None or outer % 500 or outer > raw_count)
                or role not in ("observed", "inner", "periodic_observed")):
            raise ArtifactIntegrityError("invalid shadow schedule")
        purpose = "outer_observed" if inner is None else "inner_bootstrap"
        if row["seed_identity"] != derive_seed(manifest.canonical_cell_id, outer, purpose, inner).digest_hex:
            raise ArtifactIntegrityError("shadow seed mismatch")
        if row["passed"] and not (row["classification_gate"] and row["convergence_gate"]):
            raise ArtifactIntegrityError("inconsistent shadow gates")
        if row["passed"] and row["classification_fast"] == "ELIGIBLE":
            if not (row["parameter_gate"] and row["objective_gate"]
                    and row["failure_reason_fast"] is row["failure_reason_canonical"] is None):
                raise ArtifactIntegrityError("inconsistent eligible shadow evidence")
    if shadow["entry_gate"] == "PASS":
        observed = [r for r in shadow["checks"] if r["role"] == "observed"]
        inner = [r for r in shadow["checks"] if r["role"] == "inner"]
        if (not observed or not inner or any(not r["passed"] for r in observed + inner)
                or [r["raw_outer_index"] for r in observed] != list(range(len(observed)))
                or [r["raw_inner_index"] for r in inner] != list(range(len(inner)))
                or observed[-1]["classification_fast"] != "ELIGIBLE"
                or inner[-1]["classification_fast"] != "ELIGIBLE"
                or any(r["classification_fast"] == "ELIGIBLE" for r in observed[:-1] + inner[:-1])
                or any(r["raw_outer_index"] != observed[-1]["raw_outer_index"] for r in inner)):
            raise ArtifactIntegrityError("entry canary certification is incomplete")
    elif raw_count or state["cell_execution_started"]:
        raise ArtifactIntegrityError("campaign started without entry certification")
    observed_indices = {r["raw_outer_index"] for r in shadow["checks"] if r["role"] == "observed"}
    periodic_indices = {r["raw_outer_index"] for r in shadow["checks"] if r["role"] == "periodic_observed"}
    required = set(range(0, raw_count, 500)) - observed_indices
    if not required.issubset(periodic_indices) or observed_indices.intersection(periodic_indices):
        raise ArtifactIntegrityError("missing or duplicate periodic shadow checks")
    if any(not r["passed"] for r in shadow["checks"]) and state["status"] != "FAILED_SHADOW_ORACLE":
        raise ArtifactIntegrityError("shadow failure bypassed")
    if shadow["periodic_gate"] == "FAIL" and state["status"] != "FAILED_SHADOW_ORACLE":
        raise ArtifactIntegrityError("periodic failure bypassed")
