"""Read-only B=199 aggregation and DEC-024 certification; no numerical fitting."""
from __future__ import annotations

import math

from ..r11_reference_workload.codec import require
from .contract import (ALPHA, B_R11, GATES, IDENTITY_FIELDS, PAYLOAD_FIELDS,
                       QUANTITIES, RECORD_FIELDS, SOURCE_OUTERS, TOTAL_RECORDS)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def exact(left, right):
    return type(left) is type(right) and left == right


def distribution_structure(record):
    """Check the delegated evidence universe, without recomputing tolerances."""
    points = record.get("evaluation_points")
    quantities = QUANTITIES.get(record.get("family"))
    rows = record.get("distribution_evidence")
    if (not quantities or type(points) is not list or not points
            or type(rows) is not list
            or record.get("cpu_evaluation_points") != points
            or record.get("cuda_evaluation_points") != points):
        return False
    pairs = [(q, x) for q in quantities for x in points]
    fields = {"quantity", "evaluation_point", "cpu_value", "cuda_value",
              "abs_error", "allowed_tolerance", "passed"}
    return len(rows) == len(pairs) and all(
        type(row) is dict and fields <= row.keys()
        and (row["quantity"], row["evaluation_point"]) == pair
        and row.get("failure_reason") is None and type(row["passed"]) is bool
        for row, pair in zip(rows, pairs))


def record_checks(record, expected):
    schema = type(record) is dict and RECORD_FIELDS <= record.keys()
    identity = all(k in record and k in expected and exact(record[k], expected[k])
                   for k in IDENTITY_FIELDS)
    payload = all(k in record and k in expected and exact(record[k], expected[k])
                  for k in PAYLOAD_FIELDS)
    digest = expected.get("sample_digest")
    digest_valid = (type(digest) is str and len(digest) == 64
                    and all(c in "0123456789abcdef" for c in digest))
    sample = (payload and digest_valid
              and record.get("cpu_sample_digest") == digest
              and record.get("cuda_sample_digest") == digest
              and record.get("cpu_input_identity_pass") is True
              and record.get("cuda_input_identity_pass") is True)
    structure = not distribution_structure(record) or bool(record.get("structural_failure"))
    cuda_failure = (record.get("cuda_failure_reason", "MISSING") is not None
                    or record.get("cuda_classification") != "ELIGIBLE")
    nonconvergence = record.get("cuda_solver_converged") is not True
    checks = {
        "schema_complete": schema, "identity_match": identity,
        "sample_digest_match": sample,
        "statistics_finite": all(finite(record.get(k)) for k in ("cpu_statistic", "cuda_statistic")),
        "eligible_classifications": all(record.get(k) == "ELIGIBLE"
                                        for k in ("cpu_classification", "cuda_classification")),
        "structural_failure": structure, "cuda_failure": cuda_failure,
        "cuda_non_convergence": nonconvergence,
        "distribution_rows_pass": (not structure and all(
            row["passed"] is True for row in record["distribution_evidence"])),
        **{gate: record.get(gate) is True for gate in GATES},
    }
    checks["valid"] = (
        all(checks[k] for k in ("schema_complete", "identity_match", "sample_digest_match",
                               "statistics_finite", "eligible_classifications",
                               "distribution_rows_pass", *GATES))
        and not (structure or cuda_failure or nonconvergence))
    return checks


def adjudicate_outer(records, expected_records, outer_identity):
    """Both raw counts and decisions are derived outputs, never caller inputs."""
    require(len(records) == len(expected_records) == B_R11 + 1,
            "exactly observed + 199 bootstraps required")
    require(len({r["identity"] for r in expected_records}) == B_R11 + 1,
            "duplicate frozen identity")
    observed_expected = expected_records[0]
    require(observed_expected["identity"] == outer_identity
            and observed_expected["record_type"] == "observed"
            and observed_expected["raw_inner_index"] is None
            and observed_expected["accepted_ordinal"] is None, "observed order mismatch")
    previous = -1
    for ordinal, expected in enumerate(expected_records[1:]):
        raw = expected["raw_inner_index"]
        require(type(raw) is int and raw > previous
                and expected["accepted_ordinal"] == ordinal
                and type(expected["accepted_ordinal"]) is int
                and expected["record_type"] == "bootstrap"
                and expected["identity"] == f"{outer_identity}|raw_inner={raw}"
                and exact(expected["raw_outer_index"], observed_expected["raw_outer_index"])
                and expected["cell_id"] == observed_expected["cell_id"],
                "frozen bootstrap order mismatch")
        previous = raw
    checks = [record_checks(r, e) for r, e in zip(records, expected_records)]
    observed = records[0]
    indicators = []
    for index, record in enumerate(records[1:], 1):
        cpu_obs, cuda_obs = observed.get("cpu_statistic"), observed.get("cuda_statistic")
        cpu_boot, cuda_boot = record.get("cpu_statistic"), record.get("cuda_statistic")
        cpu_finite = finite(cpu_boot) and finite(cpu_obs)
        cuda_finite = finite(cuda_boot) and finite(cuda_obs)
        cpu_indicator = bool(cpu_boot >= cpu_obs) if cpu_finite else None
        cuda_indicator = bool(cuda_boot >= cuda_obs) if cuda_finite else None
        tie = cpu_finite and cpu_boot == cpu_obs
        mismatch = cpu_finite and cuda_finite and cpu_indicator != cuda_indicator
        certified = (mismatch and tie and cpu_indicator is True and cuda_indicator is False
                     and checks[0]["valid"] and checks[index]["valid"])
        indicators.append({
            "identity": record.get("identity"), "accepted_ordinal": index - 1,
            "T_cpu_obs": cpu_obs, "T_cuda_obs": cuda_obs,
            "T_cpu_boot": cpu_boot, "T_cuda_boot": cuda_boot,
            "cpu_indicator": cpu_indicator, "cuda_indicator": cuda_indicator,
            "cpu_reference_exact_tie": tie, "indicator_mismatch": mismatch,
            "certified_exact_tie_crossing": certified,
            "observed_provenance": dict(checks[0]), "bootstrap_provenance": dict(checks[index]),
        })
    raw = {}
    for engine in ("cpu", "cuda"):
        values = [r[engine + "_indicator"] for r in indicators]
        b = sum(values) if all(type(x) is bool for x in values) else None
        p = (b + 1) / (B_R11 + 1) if b is not None else None
        raw.update({f"raw_b_{engine}": b, f"raw_p_{engine}": p,
                    f"raw_reject_{engine}": p <= ALPHA if p is not None else None})
    mismatches = sum(row["indicator_mismatch"] for row in indicators)
    certified = sum(row["certified_exact_tie_crossing"] for row in indicators)
    unexplained = mismatches - certified  # DEC-024: unmatched crossings, never a product.
    signed = (raw["raw_b_cpu"] is not None and raw["raw_b_cuda"] is not None
              and raw["raw_b_cpu"] - raw["raw_b_cuda"] == certified)
    reject_match = (type(raw["raw_reject_cpu"]) is bool
                    and type(raw["raw_reject_cuda"]) is bool
                    and raw["raw_reject_cpu"] == raw["raw_reject_cuda"])
    failures = []
    if not all(check["valid"] for check in checks):
        failures.append("RECORD_GATE_OR_PROVENANCE")
    if any(raw["raw_b_" + e] is None for e in ("cpu", "cuda")):
        failures.append("MC_NOT_EVALUABLE")
    if unexplained:
        failures.append("UNEXPLAINED_INDICATOR_MISMATCH")
    if not signed:
        failures.append("SIGNED_ACCOUNTING")
    if not reject_match:
        failures.append("REJECT_DECISION_MISMATCH")
    return {
        "identity": outer_identity, **raw, "records_evaluated": len(records),
        "bootstrap_count": len(indicators), "record_checks": checks,
        "indicator_adjudication": indicators, "INDICATOR_MISMATCH_COUNT": mismatches,
        "CERTIFIED_EXACT_TIE_CROSSING_COUNT": certified,
        "MC_UNEXPLAINED_INDICATOR_MISMATCH": unexplained,
        "MC_BOOTSTRAP_IDENTITY_MISMATCH": sum(
            not c["identity_match"] or not c["sample_digest_match"] for c in checks[1:]),
        "DEC024_SIGNED_ACCOUNTING": signed,
        "RAW_REJECT_DECISION_MATCH": reject_match,
        "R11_OUTER_PASS": not failures, "failures": failures,
    }


def summarize(outers, records, expected, fixtures, *, accepted, consumed, failure=None):
    """Strict complete-run gate. The boundary observation count is diagnostic."""
    checks = [record_checks(r, e) for r, e in zip(records, expected)]
    counters = {
        "STRUCTURAL_FAILURE": sum(c["structural_failure"] or not c["schema_complete"] for c in checks),
        "CUDA_FAILURE": sum(c["cuda_failure"] for c in checks),
        "CUDA_NON_CONVERGENCE": sum(c["cuda_non_convergence"] for c in checks),
        "MC_BOOTSTRAP_IDENTITY_MISMATCH": sum(
            not c["identity_match"] or not c["sample_digest_match"]
            for c, e in zip(checks, expected) if e["record_type"] == "bootstrap"),
        "MC_UNEXPLAINED_INDICATOR_MISMATCH": sum(
            o["MC_UNEXPLAINED_INDICATOR_MISMATCH"] for o in outers),
    }
    expected_ids = [e["identity"] for e in expected if e["record_type"] == "observed"]
    complete = (
        len(expected) == len(records) == TOTAL_RECORDS
        and len(expected_ids) == len(set(expected_ids)) == SOURCE_OUTERS
        and [o["identity"] for o in outers] == expected_ids
        and all(o["records_evaluated"] == B_R11 + 1 and o["bootstrap_count"] == B_R11
                for o in outers))
    fixture_pass = (len(fixtures) == 12 and len({f["name"] for f in fixtures}) == 12
                    and all(f["fixture_pass"] is True for f in fixtures))
    passed = (
        accepted is True and consumed is True and failure is None and complete
        and all(c["valid"] for c in checks) and all(v == 0 for v in counters.values())
        and all(o["R11_OUTER_PASS"] is True and o["DEC024_SIGNED_ACCOUNTING"] is True
                and o["RAW_REJECT_DECISION_MATCH"] is True for o in outers)
        and fixture_pass)
    return {
        "R11_GLOBAL_PASS": passed, "R11_REFERENCE_WORKLOAD_ACCEPTED": accepted,
        "EXECUTION_AUTHORIZATION_CONSUMED": consumed, "failure": failure,
        "outer_count": len(outers), "records_evaluated": sum(c["schema_complete"] for c in checks),
        "outer_pass_count": sum(o["R11_OUTER_PASS"] is True for o in outers),
        "reject_agreement_count": sum(o["RAW_REJECT_DECISION_MATCH"] is True for o in outers),
        "complete_traversal": complete, "boundary_fixtures_pass": fixture_pass,
        "SCIENTIFIC_BOUNDARY_NEIGHBORHOOD_OBSERVED": sum(
            o["raw_b_cpu"] in (8, 9, 10, 11) for o in outers),
        **counters, **{gate: all(c[gate] for c in checks) and complete for gate in GATES},
        "AUTO_RERUN": False, "AUTO_RESUME": False, "CHECKPOINT_RESUME": False,
    }
