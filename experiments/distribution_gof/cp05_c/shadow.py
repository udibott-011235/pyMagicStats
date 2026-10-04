"""Stateless entry and periodic fitting canaries, with strict PERF-01 gates."""
from __future__ import annotations

import hashlib
import math
import time
import numpy as np

from ..generators import bind_family, family_for, generate_sample
from ..seed_derivation import derive_seed, numpy_rng
from ..cuda_calibration.nb_fitting_perf01 import fast_cpu
from ..cuda_calibration.nb_fitting_perf01.engines import canonical_cpu
from ..cuda_calibration.nb_fitting_perf01.benchmark import compare_record
from ..cuda_calibration.equivalence_preregistration import NB_TRANSFORM_ATOL, NB_OBJECTIVE_RTOL
from .fitting import operational_fit
from .manifest import SHADOW_OUTER_RAW_CAP, SHADOW_INNER_RAW_CAP, SHADOW_PERIOD

INELIGIBLE = ("ALL_ZERO_NON_IDENTIFYING", "VARIANCE_NOT_GREATER_THAN_MEAN")


def sample_identity(sample):
    x = np.asarray(sample)
    dtype, shape = x.dtype.str, list(x.shape)
    digest = hashlib.sha256(x.tobytes(order="C")).hexdigest()
    return digest, dtype, shape


def _safe_number(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Inf" if value > 0 else "-Inf")
    return value


def compare(manifest, role, raw_outer_index, raw_inner_index, sample, seed, fast, canonical):
    digest, dtype, shape = sample_identity(sample)
    descriptor = {"identity": seed.digest_hex, "sample_identity": seed.digest_hex,
                  "sample_digest": digest, "payload_dtype": dtype, "payload_shape": shape,
                  "seed_identity": seed.digest_hex, "cell_id": manifest.canonical_cell_id,
                  "raw_outer_index": raw_outer_index, "raw_inner_index": raw_inner_index,
                  "accepted_ordinal": None}
    if not isinstance(fast, dict):
        fast = fast_cpu.failed("malformed fast outcome")
    if not isinstance(canonical, dict):
        canonical = fast_cpu.failed("malformed canonical outcome")
    classification_match = (fast.get("classification") == canonical.get("classification")
                            and fast.get("classification") in ("ELIGIBLE", *INELIGIBLE))
    convergence_match = (type(fast.get("converged")) is bool
                         and type(canonical.get("converged")) is bool
                         and fast["converged"] == canonical["converged"])
    parameter_gate = objective_gate = None
    failure_reason = None
    if classification_match and fast["classification"] in INELIGIBLE:
        # Ineligible fits have no parameter/objective gates. Convergence and
        # canonical CP04 exception classification must still agree exactly.
        canonical_error = ("FitIdentifiabilityError:" if fast["classification"] == INELIGIBLE[0]
                           else "NoFiniteMLEError:")
        passed = (convergence_match and fast["converged"] is False
                  and fast.get("failure_reason") == fast["classification"]
                  and isinstance(canonical.get("failure_reason"), str)
                  and canonical["failure_reason"].startswith(canonical_error))
        if not passed:
            failure_reason = "invalid mathematical-ineligibility evidence"
    else:
        gates = compare_record({**descriptor, "result": canonical}, {**descriptor, "result": fast})
        parameter_gate, objective_gate = gates["parameter_gate"], gates["log_likelihood_gate"]
        passed, failure_reason = gates["scientifically_equivalent"], gates["failure_reason"]
    if not classification_match or not convergence_match:
        passed = False
        failure_reason = failure_reason or "classification/convergence mismatch"
    row = {"canonical_cell_id": manifest.canonical_cell_id, "role": role,
           "raw_outer_index": raw_outer_index, "raw_inner_index": raw_inner_index,
           "sample_digest": digest, "sample_dtype": dtype, "sample_shape": shape,
           "seed_identity": seed.digest_hex, "classification_fast": fast.get("classification"),
           "classification_canonical": canonical.get("classification"),
           "converged_fast": fast.get("converged"), "converged_canonical": canonical.get("converged"),
           "classification_gate": classification_match, "convergence_gate": convergence_match,
           "parameter_gate": parameter_gate, "objective_gate": objective_gate,
           "passed": bool(passed), "failure_reason": failure_reason,
           "failure_reason_fast": fast.get("failure_reason"),
           "failure_reason_canonical": canonical.get("failure_reason")}
    for label, outcome in (("fast", fast), ("canonical", canonical)):
        params = outcome.get("parameters")
        params = params if isinstance(params, dict) else {}
        for name in ("r", "p"):
            row[name + "_" + label] = _safe_number(params.get(name))
        row["ll_" + label] = _safe_number(outcome.get("log_likelihood"))
    return row


def validate_evidence_row(row):
    """Reapply fitting gates to stored evidence without regenerating any sample."""
    from ..artifacts import ArtifactIntegrityError
    fast, canonical = {}, {}
    for label, outcome in (("fast", fast), ("canonical", canonical)):
        outcome.update(classification=row["classification_" + label], converged=row["converged_" + label],
                       parameters={"r": row["r_" + label], "p": row["p_" + label]},
                       log_likelihood=row["ll_" + label], failure_reason=row["failure_reason_" + label])
    classification = (fast["classification"] == canonical["classification"]
                      and fast["classification"] in ("ELIGIBLE", *INELIGIBLE))
    convergence = (type(fast["converged"]) is bool and type(canonical["converged"]) is bool
                   and fast["converged"] == canonical["converged"])
    if row["classification_gate"] != classification or row["convergence_gate"] != convergence:
        raise ArtifactIntegrityError("stored shadow categorical gate mismatch")
    if classification and fast["classification"] in INELIGIBLE:
        prefix = "FitIdentifiabilityError:" if fast["classification"] == INELIGIBLE[0] else "NoFiniteMLEError:"
        passed = (convergence and fast["converged"] is False and fast["failure_reason"] == fast["classification"]
                  and isinstance(canonical["failure_reason"], str) and canonical["failure_reason"].startswith(prefix))
        parameter = objective = None
    else:
        descriptor = {"identity": row["seed_identity"], "sample_identity": row["seed_identity"],
                      "sample_digest": row["sample_digest"], "payload_dtype": row["sample_dtype"],
                      "payload_shape": row["sample_shape"], "seed_identity": row["seed_identity"],
                      "cell_id": row["canonical_cell_id"], "raw_outer_index": row["raw_outer_index"],
                      "raw_inner_index": row["raw_inner_index"], "accepted_ordinal": None}
        gates = compare_record({**descriptor, "result": canonical}, {**descriptor, "result": fast})
        parameter, objective, passed = gates["parameter_gate"], gates["log_likelihood_gate"], gates["scientifically_equivalent"]
    if (row["parameter_gate"] is not parameter or row["objective_gate"] is not objective
            or row["passed"] is not bool(passed)):
        raise ArtifactIntegrityError("stored shadow numerical gate mismatch")


class ShadowOracle:
    def __init__(self, manifest, *, fast=None, canonical=None, generate=generate_sample):
        if not manifest.nb_composite:
            raise ValueError("shadow oracle only supports NB composite")
        self.manifest = manifest
        self.fast = fast or fast_cpu.fit_negative_binomial
        self.canonical = canonical or canonical_cpu()
        self.generate = generate

    def check(self, role, outer, inner, bound=None):
        purpose = "outer_observed" if inner is None else "inner_bootstrap"
        seed = derive_seed(self.manifest.canonical_cell_id, outer, purpose, inner)
        self.last_identity = {"role": role, "raw_outer_index": outer, "raw_inner_index": inner,
                              "seed_identity": seed.digest_hex}
        bound = bound or bind_family(self.manifest.family, self.manifest.canonical_parameters)
        sample = self.generate(bound, self.manifest.n, numpy_rng(seed))
        def invoke(fitter):
            try:
                return fitter(sample.copy())
            except Exception as exc:
                return fast_cpu.failed(type(exc).__name__ + ": " + str(exc))
        fast, canonical = invoke(self.fast), invoke(self.canonical)
        return compare(self.manifest, role, outer, inner, sample, seed, fast, canonical), fast, sample

    def entry(self):
        start, rows = time.perf_counter(), []
        error, passed = None, False
        try:
            first = None
            for outer in range(SHADOW_OUTER_RAW_CAP):
                row, fast, sample = self.check("observed", outer, None)
                rows.append(row)
                if not row["passed"]:
                    error = "observed shadow mismatch"
                    break
                if row["classification_fast"] == "ELIGIBLE":
                    first = (outer, operational_fit(family_for(self.manifest.family), sample, fast))
                    break
            if first is None and error is None:
                error = "SHADOW_OUTER_RAW_CAP_EXHAUSTED"
            if first is not None:
                outer, fit = first
                for inner in range(SHADOW_INNER_RAW_CAP):
                    row, _, _ = self.check("inner", outer, inner, fit.fitted_distribution)
                    rows.append(row)
                    if not row["passed"]:
                        error = "inner shadow mismatch"
                        break
                    if row["classification_fast"] == "ELIGIBLE":
                        passed = True
                        break
                if not passed and error is None:
                    error = "SHADOW_INNER_RAW_CAP_EXHAUSTED"
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        return {"entry_gate": "PASS" if passed else "FAIL", "checks": rows,
                "shadow_wall_seconds": time.perf_counter() - start, "failure_reason": error,
                "canonical_shadow_fit_count": len(rows), "periodic_gate": "PASS",
                "failure_identity": getattr(self, "last_identity", None) if error else None,
                "NB_TRANSFORM_ATOL": NB_TRANSFORM_ATOL, "NB_OBJECTIVE_RTOL": NB_OBJECTIVE_RTOL}

    def periodic(self, index):
        start = time.perf_counter()
        try:
            row, _, _ = self.check("periodic_observed", index, None)
        except Exception as exc:
            return None, time.perf_counter() - start, type(exc).__name__ + ": " + str(exc)
        return row, time.perf_counter() - start, None


def periodic_due(index, checks):
    return index % SHADOW_PERIOD == 0 and not any(
        r["role"] in ("observed", "periodic_observed") and r["raw_outer_index"] == index for r in checks)
