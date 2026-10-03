"""Complete logical B=199 surfaces, never scientific samples or trusted counts."""
from __future__ import annotations

import hashlib
import math

from .adjudication import adjudicate_outer
from .contract import B_R11, GATES, QUANTITIES

SCENARIOS = (
    ("equal_b8", 8, True), ("equal_b9", 9, True),
    ("equal_b10", 10, True), ("equal_b11", 11, True),
    ("certified_b9_b8", 9, True), ("certified_b10_b9", 10, False),
    ("opposite_crossing", 8, False), ("nextafter_neighbor", 9, False),
    ("one_ulp_unequal", 9, False), ("isclose_only", 9, False),
    ("cancelling_mismatches", 9, False), ("unexplained_same_reject", 8, False),
)


def logical_record(expected, cpu, cuda):
    """Logical numerical gates model already-passing delegated comparisons."""
    values = [{"quantity": q, "evaluation_point": 0.0, "cpu_value": 0.5,
               "cuda_value": 0.5, "abs_error": 0.0, "allowed_tolerance": 5e-13,
               "passed": True} for q in QUANTITIES["exponential"]]
    return {
        **expected, "family": "exponential",
        "cpu_classification": "ELIGIBLE", "cuda_classification": "ELIGIBLE",
        "cpu_parameters": {"scale": 1.0}, "cuda_parameters": {"scale": 1.0},
        "cpu_log_likelihood": None, "cuda_log_likelihood": None,
        "cpu_statistic": cpu, "cuda_statistic": cuda,
        **{g: True for g in GATES}, "distribution_evidence": values,
        "flat_objective_used": None, "flat_objective_diagnostic": None,
        "cuda_failure_reason": None, "cuda_solver_converged": True,
        "cpu_input_identity_pass": True, "cuda_input_identity_pass": True,
        "cpu_sample_digest": expected["sample_digest"],
        "cuda_sample_digest": expected["sample_digest"],
        "evaluation_points": [0.0], "cpu_evaluation_points": [0.0],
        "cuda_evaluation_points": [0.0],
        "raw_cpu_result": {"classification": "ELIGIBLE", "statistic": cpu},
        "raw_cuda_result": {"classification": "ELIGIBLE", "statistic": cuda},
    }


def materialize(name):
    scenario = next((s for s in SCENARIOS if s[0] == name), None)
    if scenario is None:
        raise ValueError("unknown boundary fixture")
    _, count, expected_pass = scenario
    cell = f"exponential|LOGICAL_FIXTURE={name}"
    outer = cell + "|raw_outer=0"
    expected = []
    for index in range(B_R11 + 1):
        raw = None if index == 0 else index - 1
        identity = outer if raw is None else f"{outer}|raw_inner={raw}"
        expected.append({
            "identity": identity, "record_type": "observed" if raw is None else "bootstrap",
            "cell_id": cell, "raw_outer_index": 0, "raw_inner_index": raw,
            "accepted_ordinal": raw, "seed_identity": index,
            "sample_digest": hashlib.sha256(identity.encode()).hexdigest(),
            "payload_dtype": "<f8", "payload_shape": [1],
        })
    records = [logical_record(expected[0], 1.0, 1.0)]
    records.extend(logical_record(e, 2.0 if j < count else 0.0,
                                  2.0 if j < count else 0.0)
                   for j, e in enumerate(expected[1:]))
    # Alter statistic pairs, never feed trusted b/p/reject to the adjudicator.
    if name.startswith("certified_"):
        records[count]["cpu_statistic"] = 1.0
        records[count]["cuda_statistic"] = math.nextafter(1.0, 0.0)
    elif name == "opposite_crossing":
        records[count + 1]["cpu_statistic"] = 0.0
        records[count + 1]["cuda_statistic"] = 2.0
    elif name in ("nextafter_neighbor", "one_ulp_unequal", "isclose_only"):
        records[count]["cpu_statistic"] = (
            math.nextafter(1.0, math.inf) if name == "nextafter_neighbor"
            else 1.0 + 2.0 ** -52 if name == "one_ulp_unequal" else 1.0 + 1e-12)
        records[count]["cuda_statistic"] = math.nextafter(1.0, 0.0)
    elif name == "cancelling_mismatches":
        records[count]["cpu_statistic"] = 1.0
        records[count]["cuda_statistic"] = math.nextafter(1.0, 0.0)
        records[count + 1]["cpu_statistic"] = 0.0
        records[count + 1]["cuda_statistic"] = 2.0
    elif name == "unexplained_same_reject":
        records[count]["cpu_statistic"] = 2.0
        records[count]["cuda_statistic"] = 0.0
    for record in records:
        record["raw_cpu_result"]["statistic"] = record["cpu_statistic"]
        record["raw_cuda_result"]["statistic"] = record["cuda_statistic"]
    return outer, records, expected, expected_pass


def evaluate_fixtures():
    results = []
    for name, _, _ in SCENARIOS:
        outer, records, expected, disposition = materialize(name)
        result = adjudicate_outer(records, expected, outer)
        results.append({"name": name, "logical_fixture": True,
                        "expected_outer_pass": disposition,
                        "fixture_pass": result["R11_OUTER_PASS"] is disposition,
                        "adjudication": result})
    return results
