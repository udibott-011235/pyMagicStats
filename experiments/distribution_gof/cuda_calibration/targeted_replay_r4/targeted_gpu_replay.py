#!/usr/bin/env python3
"""DEC-025 R4 harness: static validation or separately authorized fresh execution.

Import and --validate-static use only the standard library. No checkout, repair,
resume, rerun, full-campaign dispatch, historical-result oracle, or CPU fallback.
Run this external support file against an explicitly supplied scientific checkout;
the checkout must be d2abd57..., NOT the documentary/harness commit.
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import datetime as dt
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

WORK_ITEM = "CP05-C2C-R10-A-TARGETED-GPU-REPLAY-R4-HARNESS"
PREREGISTRATION_SHA = "d481125e6981d43bbb529c8ed102c034438cbddd"
REPLAY_EXECUTION_SHA = "d2abd57e65bb7433eff81872a4f6510144d7c267"
REPLAY_EXECUTION_TREE = "70b20d1f9cbc20d6cd77de4c25b3017a6413932c"
R9_EXECUTION_SHA = "01c9c0759d3ffec1bbd95cb5aa3f67793aab544b"
IDENTITY_MANIFEST_SHA256 = "3f300ff5a678e3eac4956fe10514bf7634ec0ea824f0bfef56f4688a86adb286"
IDENTITY_MANIFEST_SIZE = 118681
IDENTITY_MANIFEST_LF = 2979
IDENTITY_MANIFEST_CRLF = 0
ORDERED_PAYLOAD_SHA256 = "3fe47fef69fc2dfac152dd48eac75841fe25a107b0da6194c06a5d14a7be1395"
SCHEMA_VERSION = "cp05-c2c-r10a-targeted-replay-v4"
DECISION = "DEC-025"
REPLAY_VERSION = "R4"
NAMESPACE = "CP05-C2C"
R_EQ = 8
B_EQ = 15
WORKLOAD_A_EXPECTED = 134
WORKLOAD_A_OBSERVED_EXPECTED = 10
WORKLOAD_A_BOOTSTRAP_EXPECTED = 124
WORKLOAD_B_OUTER_EXPECTED = 12
WORKLOAD_B_BOOTSTRAPS_PER_OUTER = 15
MANIFEST_PATH = Path(__file__).resolve().with_name("frozen_identity_manifest.json")
QUANTITIES = ("pmf", "logPMF", "cdf", "sf", "logCDF", "logSF")
STRUCTURAL_REASONS = (
    "EVALUATION_POINT_IDENTITY_MISMATCH", "DISTRIBUTION_VALUE_LENGTH_MISMATCH",
    "MISSING_CPU_DISTRIBUTION_QUANTITY", "MISSING_CUDA_DISTRIBUTION_QUANTITY",
    "MISSING_BOTH_DISTRIBUTION_QUANTITY", "UNEXPECTED_DISTRIBUTION_QUANTITY",
)
GATES = {
    "classification_gate_pass": "CLASSIFICATION_MISMATCH",
    "fit_gate_pass": "FIT_GATE_FAILURE",
    "distribution_value_gate_pass": "DISTRIBUTION_VALUE_GATE_FAILURE",
    "statistic_gate_pass": "STATISTIC_GATE_FAILURE",
}
REQUIRED_COUNTERS = {
    "CLASSIFICATION_MISMATCH": 0, "CUDA_NONCONVERGENCE": 0,
    "FIT_GATE_FAILURE": 0, "DISTRIBUTION_VALUE_GATE_FAILURE": 0,
    "STATISTIC_GATE_FAILURE": 0, "UNEXPLAINED_DISCREPANCY": 0,
    "MC_BOOTSTRAP_IDENTITY_MISMATCH": 0, "MC_OUTERS_RECONSTRUCTED": 12,
    "MC_UNEXPLAINED_INDICATOR_MISMATCH": 0, "MC_REJECT_DECISION_MISMATCH": 0,
    "MC_OUTERS_ADJUDICATED": 12,
}
DIAGNOSTIC_COUNTERS = ("MC_RAW_EXCEEDANCE_COUNT_MISMATCH_OUTERS",
                       "MC_REFERENCE_EXACT_TIE_CROSSINGS")
MC_LIMITATION = {
    "B_EQ": B_EQ, "ALPHA": 0.05, "P_MIN": 1 / (B_EQ + 1),
    "MC_REJECT_BOUNDARY_INFORMATIVE": "NO",
    "MC_REJECT_BOUNDARY_REASON": "B_EQ=15 implies p_min=0.0625 > alpha=0.05",
}
PROJECT_ROOTS = {"experiments", "pyMagicStat", "pyMagicStats", "pymagicstats"}
MODULES = {
    "runner": "cp05_c2c_equivalence_runner",
    "preregistration": "equivalence_preregistration",
    "engine": "cp05_cuda_engine",
    "nb_support": "nb_support",
    "cuda_candidate": "cuda_candidate",
}
MODULE_PREFIX = "experiments.distribution_gof.cuda_calibration."
RECORD_FIELDS = {
    "identity", "record_type", "cell_id", "family", "raw_outer_index",
    "raw_inner_index", "sample_digest", "cpu_classification", "cuda_classification",
    "cpu_parameters", "cuda_parameters", "cpu_log_likelihood", "cuda_log_likelihood",
    "cpu_statistic", "cuda_statistic", "statistic_abs_error",
    "statistic_allowed_tolerance", "distribution_evidence", "cuda_failure_reason",
    "flat_objective_used", "flat_objective_diagnostic", *GATES,
}


class HarnessContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise HarnessContractError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise HarnessContractError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=constant)
    except (ValueError, UnicodeError) as exc:
        raise HarnessContractError("invalid strict UTF-8 JSON") from exc


def integer(value, upper=None):
    return type(value) is int and value >= 0 and (upper is None or value < upper)


def validate_manifest(manifest):
    require(isinstance(manifest, dict), "manifest must be an object")
    expected = {
        "schema_version": SCHEMA_VERSION, "decision": DECISION,
        "replay_version": REPLAY_VERSION, "replay_execution_sha": REPLAY_EXECUTION_SHA,
        "replay_execution_tree": REPLAY_EXECUTION_TREE, "r9_execution_sha": R9_EXECUTION_SHA,
        "namespace": NAMESPACE, "R_EQ": R_EQ, "B_EQ": B_EQ,
        "selection_semantics": "R9 persisted failure union C/F/S/H",
        "persisted_failure_record_count": WORKLOAD_A_EXPECTED,
        "mc_failed_outer_count": WORKLOAD_B_OUTER_EXPECTED,
    }
    for key, value in expected.items():
        require(type(manifest.get(key)) is type(value) and manifest[key] == value,
                f"manifest metadata mismatch: {key}")
    a = manifest.get("persisted_failure_record_identities")
    b = manifest.get("mc_failed_outers")
    require(isinstance(a, list) and len(a) == WORKLOAD_A_EXPECTED, "Workload A count")
    require(isinstance(b, list) and len(b) == WORKLOAD_B_OUTER_EXPECTED, "Workload B count")
    ids, outer_ids = [], []
    observed = bootstrap = 0
    for row in a:
        require(isinstance(row, dict), "Workload A row")
        cell, outer = row.get("cell_id"), row.get("raw_outer_index")
        require(isinstance(cell, str) and cell.startswith("negative_binomial|")
                and row.get("family") == "negative_binomial", "Workload A family")
        require(integer(outer, R_EQ), "Workload A raw_outer")
        identity = f"{cell}|raw_outer={outer}"
        inner = row.get("raw_inner_index")
        if row.get("record_type") == "observed":
            observed += 1
            require(inner is None, "observed raw_inner must be null")
        else:
            bootstrap += 1
            require(row.get("record_type") == "bootstrap" and integer(inner),
                    "bootstrap raw_inner/record_type")
            identity += f"|raw_inner={inner}"
        require(row.get("identity") == identity, "Workload A identity recomposition")
        ids.append(identity)
    require(len(set(ids)) == len(ids), "duplicate Workload A identity")
    require((observed, bootstrap) == (WORKLOAD_A_OBSERVED_EXPECTED, WORKLOAD_A_BOOTSTRAP_EXPECTED),
            "Workload A observed/bootstrap cardinality")
    for row in b:
        require(isinstance(row, dict), "Workload B row")
        cell, outer = row.get("cell_id"), row.get("raw_outer_index")
        require(isinstance(cell, str) and cell.startswith("negative_binomial|")
                and integer(outer, R_EQ), "Workload B cell/raw_outer")
        identity = f"{cell}|raw_outer={outer}"
        require(row.get("outer_identity") == row.get("observed_identity") == identity,
                "Workload B observed identity")
        outer_ids.append(identity)
        boots = row.get("bootstrap_identities")
        require(isinstance(boots, list) and len(boots) == WORKLOAD_B_BOOTSTRAPS_PER_OUTER,
                "Workload B bootstrap count")
        pairs = []
        for item in boots:
            require(isinstance(item, dict) and integer(item.get("raw_inner_index")),
                    "Workload B raw_inner")
            inner = item["raw_inner_index"]
            require(item.get("identity") == f"{identity}|raw_inner={inner}",
                    "Workload B bootstrap identity recomposition")
            pairs.append((item["identity"], inner))
        require(len(set(pairs)) == len(pairs), "duplicate Workload B bootstrap")
    require(len(set(outer_ids)) == len(outer_ids), "duplicate Workload B outer")
    # No sorting/renumbering of any list: the digest fixes every field and list position.
    payload = {"persisted_failure_record_identities": a, "mc_failed_outers": b}
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False).encode("ascii"))
    require(digest == ORDERED_PAYLOAD_SHA256, "ordered payload digest mismatch")
    return {
        "WORKLOAD_A_IDENTITIES": len(a), "WORKLOAD_A_OBSERVED": observed,
        "WORKLOAD_A_BOOTSTRAP": bootstrap, "WORKLOAD_A_DUPLICATES": 0,
        "WORKLOAD_B_OUTERS": len(b), "WORKLOAD_B_BOOTSTRAPS_PER_OUTER": 15,
        "WORKLOAD_B_OUTER_DUPLICATES": 0, "WORKLOAD_B_BOOTSTRAP_DUPLICATES": 0,
        "ORDERED_PAYLOAD_SHA256": digest, "IDENTITY_RECOMPOSITION": "PASS",
        "RAW_INNER_GAPS_PRESERVED": "YES", "ORDER_PRESERVED": "YES",
    }


def load_manifest():
    data = MANIFEST_PATH.read_bytes()
    require(sha256(data) == IDENTITY_MANIFEST_SHA256, "manifest SHA256 mismatch")
    require(len(data) == IDENTITY_MANIFEST_SIZE, "manifest byte size mismatch")
    require(data.count(b"\n") == IDENTITY_MANIFEST_LF, "manifest LF mismatch")
    require(data.count(b"\r\n") == IDENTITY_MANIFEST_CRLF, "manifest CRLF mismatch")
    require(data.isascii(), "manifest must be ASCII")
    manifest = strict_json(data)
    return manifest, validate_manifest(manifest)


def validate_static():
    _, report = load_manifest()
    return {
        "WORK_ITEM": WORK_ITEM, "STATIC_VALIDATION": "PASS",
        "PREREGISTRATION_SHA": PREREGISTRATION_SHA, "DECISION": DECISION,
        "REPLAY_VERSION": REPLAY_VERSION, "REPLAY_EXECUTION_SHA": REPLAY_EXECUTION_SHA,
        "REPLAY_EXECUTION_TREE": REPLAY_EXECUTION_TREE,
        "MANIFEST_SHA256": IDENTITY_MANIFEST_SHA256, "MANIFEST_SIZE_BYTES": IDENTITY_MANIFEST_SIZE,
        "MANIFEST_LF": IDENTITY_MANIFEST_LF, "MANIFEST_CRLF": IDENTITY_MANIFEST_CRLF,
        **report,
    }


def inside(path, root):
    return path.resolve().is_relative_to(root.resolve())


def git(repo, *args):
    executable = os.environ.get("GIT_EXE") or shutil.which("git")
    require(executable is not None, "Git unavailable")
    result = subprocess.run(
        [executable, "--no-optional-locks", "-C", str(repo), "-c", "core.fsmonitor=false", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True, encoding="utf-8",
    )
    require(result.returncode == 0, f"Git read failed: {result.stderr.strip()}")
    return result.stdout.strip()


def preflight_repo(repo):
    require(repo.is_dir(), "scientific repo does not exist")
    require(Path(git(repo, "rev-parse", "--show-toplevel")).resolve() == repo.resolve(),
            "explicit scientific repository root required")
    head = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    require(head == REPLAY_EXECUTION_SHA, f"wrong scientific HEAD: {head}")
    require(tree == REPLAY_EXECUTION_TREE, f"wrong scientific tree: {tree}")
    require(not git(repo, "status", "--porcelain=v1", "--untracked-files=all"),
            "scientific worktree is dirty")
    return {"scientific_repo_path": str(repo), "scientific_head": head,
            "scientific_tree": tree, "scientific_worktree_clean": True}


def project_name(name):
    return name.split(".")[0] in PROJECT_ROOTS


def verify_loaded_scientific_modules(repo):
    locations = {}
    for name, module in tuple(sys.modules.items()):
        if not project_name(name):
            continue
        file = getattr(module, "__file__", None)
        paths = list(getattr(module, "__path__", []))
        require(file is not None or bool(paths), f"project module has no origin: {name}")
        for path in ([file] if file else []) + paths:
            require(inside(Path(path), repo), f"project module outside checkout: {name}: {path}")
        locations[name] = str(Path(file).resolve()) if file else [str(Path(p).resolve()) for p in paths]
    return locations


class CheckoutImportGuard(importlib.abc.MetaPathFinder):
    """Reject foreign project specs before their code executes, including namespace packages."""
    def __init__(self, repo):
        self.repo = repo

    def find_spec(self, fullname, path=None, target=None):
        if not project_name(fullname):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        require(spec is not None, f"missing scientific module: {fullname}")
        paths = list(spec.submodule_search_locations or [])
        if spec.origin:
            paths.append(spec.origin)
        require(bool(paths), f"scientific module without physical origin: {fullname}")
        require(all(inside(Path(p), self.repo) for p in paths),
                f"scientific import shadow outside checkout: {fullname}")
        return spec


def load_frozen_runtime(repo):
    verify_loaded_scientific_modules(repo)
    sys.dont_write_bytecode = True  # Never modify the scientific checkout with import caches.
    sys.path.insert(0, str(repo))
    importlib.invalidate_caches()
    guard = CheckoutImportGuard(repo)
    sys.meta_path.insert(0, guard)
    try:
        modules = {key: importlib.import_module(MODULE_PREFIX + name) for key, name in MODULES.items()}
        modules["numpy"] = importlib.import_module("numpy")
        modules["scipy"] = importlib.import_module("scipy")
        for key, name in MODULES.items():
            expected = repo / "experiments/distribution_gof/cuda_calibration" / (name + ".py")
            actual = Path(modules[key].__file__).resolve()
            require(actual == expected.resolve() and inside(actual, repo),
                    f"wrong scientific module file: {name}")
        paths = verify_loaded_scientific_modules(repo)
    finally:
        sys.meta_path.remove(guard)
    return SimpleNamespace(**modules, loaded_module_paths=paths)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_safe(value):
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    raise HarnessContractError(f"unsupported evidence type: {type(value).__name__}")


def write_json(path, payload):
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def append_jsonl(path, payload):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(json_safe(payload), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def write_digests(output):
    write_json(output / "digests.json", {
        p.relative_to(output).as_posix(): sha256(p.read_bytes())
        for p in sorted(output.rglob("*")) if p.is_file() and p != output / "digests.json"
    })


def initial_state():
    return {
        "preflight_passed": False, "authorization_consumed": False,
        "environment_preflight": {},
        "counters": {key: 0 for key in (*REQUIRED_COUNTERS, *DIAGNOSTIC_COUNTERS)},
        "STRUCTURAL_REASON_ITEM_COUNTS": {key: 0 for key in STRUCTURAL_REASONS},
        "STRUCTURAL_REASON_RECORD_COUNTS": {key: 0 for key in STRUCTURAL_REASONS},
        "_structural_seen": {key: set() for key in STRUCTURAL_REASONS},
        "WORKLOAD_A_RECORDS_EXECUTED": 0, "WORKLOAD_B_RECORDS_EXECUTED": 0,
        "mc_results": [], "context": {"phase": "preflight", "identity": None,
        "record_type": None, "cell_id": None, "raw_outer": None, "raw_inner": None,
        "failed_gates": [], "structural_failure_reasons": []},
    }


def context(state, phase, row=None):
    row = row or {}
    state["context"] = {
        "phase": phase, "identity": row.get("identity", row.get("outer_identity")),
        "record_type": row.get("record_type", "outer" if row else None),
        "cell_id": row.get("cell_id"), "raw_outer": row.get("raw_outer_index"),
        "raw_inner": row.get("raw_inner_index"), "failed_gates": [],
        "structural_failure_reasons": [],
    }


def build_cell_index(runtime, manifest):
    # Enumerate fixture definitions only; never traverse primary outers/campaigns.
    index = {c.canonical_id: c for c in runtime.runner.primary_fixture_matrix()}
    selected = {r["cell_id"] for r in manifest["persisted_failure_record_identities"]}
    selected.update(r["cell_id"] for r in manifest["mc_failed_outers"])
    require(selected <= index.keys(), "manifest cell missing from frozen fixtures")
    return {key: index[key] for key in selected}


def evaluate_record(runtime, cell, item, sample):
    runner = runtime.runner
    digest = sha256(sample.tobytes())
    maximum = int(max(sample))
    if item["record_type"] == "observed":
        cpu = runner._cpu_nb_classification(sample)
        cuda = runner._cuda_nb_classification(sample)
        if cpu != "ELIGIBLE" or cuda != "ELIGIBLE":
            result = runner._ineligible_observed_outer(
                cell, item["raw_outer_index"], sample, {"sample_digest_sha256": digest}, cpu, cuda)
            return {**result["records"][0], "sample_max": maximum, "evaluation_points": [],
                    "certified_support_stop": None,
                    "support_evidence_note": "No support certification for ineligible observed record."}
    bound = runner.reference_fit(cell.family, sample)["bound"]
    support = runtime.nb_support.certify_nb_support(sample, bound, cell.statistic)
    captures = {}

    def reference_adapter(c, x):
        require(x is sample and sha256(x.tobytes()) == digest, "CPU sample identity changed")
        result = runner.evaluate_reference_record(c, x)
        captures["cpu_evaluation_points"] = list(result.get("evaluation_points", ()))
        require(sha256(x.tobytes()) == digest, "CPU mutated sample")
        return result

    def cuda_adapter(c, x):
        require(x is sample and sha256(x.tobytes()) == digest, "CUDA sample identity changed")
        result = runner.evaluate_cuda_record(c, x, certified_support=support.indices,
                                             remainder_bound=support.remainder_bound)
        captures["cuda_evaluation_points"] = list(result.get("evaluation_points", ()))
        require(sha256(x.tobytes()) == digest, "CUDA mutated sample")
        return result

    record = runner.evaluate_fixed_record(
        identity=item["identity"], record_type=item["record_type"], cell=cell,
        raw_outer_index=item["raw_outer_index"], raw_inner_index=item["raw_inner_index"],
        sample=sample, reference_adapter=reference_adapter, cuda_adapter=cuda_adapter)
    require(record["sample_digest"] == digest, "evaluator sample digest drift")
    return {**record, **captures, "sample_max": maximum,
            "evaluation_points": list(runner.canonical_distribution_value_points(cell, sample)),
            "certified_support_stop": support.support_stop,
            "statistic_support_evidence": dataclasses.asdict(support),
            "statistic_support_size": len(support.indices)}


def workload_a_record(runtime, cell, item):
    runner = runtime.runner
    observed, meta = runner.fixed_observed(cell, item["raw_outer_index"], NAMESPACE)
    sample, seed = observed, meta["seed_identity"]
    if item["record_type"] == "bootstrap":
        fit = runner.reference_fit(cell.family, observed)
        seed = runtime.engine.derive_seed(NAMESPACE, cell.canonical_id, item["raw_outer_index"],
                                          "inner_bootstrap", item["raw_inner_index"])
        sample = runtime.engine._generate(cell.family, fit["parameters"], cell.n, seed)
        runner.reference_fit(cell.family, sample)  # Canonical eligibility, no substituted raw index.
    return {**evaluate_record(runtime, cell, item, sample), "seed_identity": seed,
            "r9_provenance": {k: v for k, v in item.items()
                              if k.startswith("r9_") or k in {"cpu_classification", "cuda_classification"}}}


def numeric_evidence_schema_complete(record):
    """Check evidence shape, not agreement; a boolean False is a valid numerical result."""
    points = record.get("evaluation_points")
    maximum = record.get("sample_max")
    if not integer(maximum) or points != list(range(maximum + 1)):
        return False
    if any(record.get(k) != points for k in ("cpu_evaluation_points", "cuda_evaluation_points")):
        return False
    evidence = record.get("distribution_evidence")
    if not isinstance(evidence, list) or len(evidence) != len(QUANTITIES) * len(points):
        return False
    expected = [(q, point) for q in QUANTITIES for point in points]
    fields = {"quantity", "evaluation_point", "cpu_value", "cuda_value",
              "abs_error", "allowed_tolerance", "passed"}
    return all(
        isinstance(row, dict) and fields <= row.keys() and not row.get("failure_reason")
        and (row["quantity"], row["evaluation_point"]) == pair and type(row["passed"]) is bool
        for row, pair in zip(evidence, expected)
    )


def consume_record(state, output, phase, expected, record):
    counts = state["counters"]
    before = counts.copy()
    failed = []
    if not isinstance(record, dict):
        raise HarnessContractError("evaluator did not return a record")
    if not RECORD_FIELDS <= record.keys():
        counts["UNEXPLAINED_DISCREPANCY"] += 1
        failed.append("record_schema")
    for key in ("identity", "record_type", "cell_id", "raw_outer_index", "raw_inner_index"):
        if record.get(key) != expected.get(key):
            counts["UNEXPLAINED_DISCREPANCY"] += 1
            failed.append("record_identity")
            break
    evidence = record.get("distribution_evidence")
    evidence = evidence if isinstance(evidence, list) else []
    reasons = []
    for item in evidence:
        reason = item.get("failure_reason") if isinstance(item, dict) else None
        if reason in STRUCTURAL_REASONS:
            reasons.append(reason)
            state["STRUCTURAL_REASON_ITEM_COUNTS"][reason] += 1
            state["_structural_seen"][reason].add(expected["identity"])
            state["STRUCTURAL_REASON_RECORD_COUNTS"][reason] = len(state["_structural_seen"][reason])
        elif reason:
            counts["UNEXPLAINED_DISCREPANCY"] += 1
            failed.append("unknown_structural_reason")
    # Preserve the runner's explicit unassessed-observed contract; never waive an eligible gate.
    unassessed = (
        phase == "workload_a" and record.get("record_type") == "observed"
        and record.get("observed_ineligible") is True
        and record.get("cpu_classification") == record.get("cuda_classification")
        and record.get("cpu_classification") in {"ALL_ZERO_NON_IDENTIFYING", "VARIANCE_NOT_GREATER_THAN_MEAN"}
    )
    for gate, counter in GATES.items():
        value = record.get(gate)
        if value is False:
            counts[counter] += 1
            failed.append(gate)
        elif value is not True and not (unassessed and gate != "classification_gate_pass" and value is None):
            counts["UNEXPLAINED_DISCREPANCY"] += 1
            failed.append(gate)
    failure = record.get("cuda_failure_reason")
    if failure:
        counter = "CUDA_NONCONVERGENCE" if failure == "CUDA solver non-convergence" else "UNEXPLAINED_DISCREPANCY"
        counts[counter] += 1
        failed.append("cuda_failure_reason")
    if not unassessed:
        eligible = all(record.get(k) == "ELIGIBLE" for k in ("cpu_classification", "cuda_classification"))
        if not eligible and record.get("classification_gate_pass") is not False:
            counts["UNEXPLAINED_DISCREPANCY"] += 1
            failed.append("unexplained_classification")
        schema_complete = numeric_evidence_schema_complete(record)
        # Structural evidence replaces numerical rows in the frozen runner. A known
        # CUDA failure can likewise prevent them; neither is a second unexplained fault.
        schema_explained = bool(reasons) or "unknown_structural_reason" in failed or (
            failure == "CUDA solver non-convergence"
            and record.get("distribution_value_gate_pass") is False
        )
        if not schema_complete and not schema_explained:
            counts["UNEXPLAINED_DISCREPANCY"] += 1
            failed.append("required_quantity_evidence_schema")
        if schema_complete and any(row["passed"] is False for row in evidence) and record.get("distribution_value_gate_pass") is not False:
            counts["UNEXPLAINED_DISCREPANCY"] += 1
            failed.append("distribution_agreement_gate_inconsistency")
        for key in ("cpu_statistic", "cuda_statistic"):
            value = record.get(key)
            if not isinstance(value, (int, float)):
                counts["UNEXPLAINED_DISCREPANCY"] += 1
                failed.append("statistic_schema")
            elif not math.isfinite(value) and record.get("statistic_gate_pass") is not False:
                counts["UNEXPLAINED_DISCREPANCY"] += 1
                failed.append("finite_statistics")
    elif evidence or any(record.get(k) is not None for k in ("cpu_statistic", "cuda_statistic")):
        counts["UNEXPLAINED_DISCREPANCY"] += 1
        failed.append("unassessed_schema")
    state["context"]["failed_gates"] = list(dict.fromkeys(failed))
    state["context"]["structural_failure_reasons"] = list(dict.fromkeys(reasons))
    append_jsonl(output / (phase + "_records.jsonl"), record)
    state[phase.upper() + "_RECORDS_EXECUTED"] += 1
    if reasons or counts != before:
        raise HarnessContractError(f"record failed: {expected['identity']}")


def run_workload_a(runtime, manifest, cells, output, state):
    for item in manifest["persisted_failure_record_identities"]:
        context(state, "workload_a", item)
        begin_scientific_execution(state, output)
        record = workload_a_record(runtime, cells[item["cell_id"]], item)
        consume_record(state, output, "workload_a", item, record)
    require(state["WORKLOAD_A_RECORDS_EXECUTED"] == WORKLOAD_A_EXPECTED, "incomplete Workload A")


def finite_statistic(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def record_adjudication_checks(record, expected, frozen, outer, observed=False):
    """Validate reconstructed provenance and original record gates, never repair."""
    def exact(a, b):
        return type(a) is type(b) and a == b

    seed = expected.get("expected_seed_identity")
    digest = expected.get("expected_sample_digest")
    identity = frozen["observed_identity"] if observed else frozen["identity"]
    inner = None if observed else frozen["raw_inner_index"]
    reasons = [row.get("failure_reason") for row in record.get("distribution_evidence", [])
               if isinstance(row, dict) and row.get("failure_reason")]
    result = {
        **expected,
        "identity_match": (
            record.get("identity") == expected.get("expected_identity") == identity
            and record.get("cell_id") == outer["cell_id"]
            and record.get("record_type") == ("observed" if observed else "bootstrap")
            and exact(record.get("raw_outer_index"), outer["raw_outer_index"])
            and exact(record.get("raw_inner_index"), inner)),
        "seed_identity_match": integer(seed) and exact(record.get("seed_identity"), seed),
        "sample_digest_match": (
            isinstance(digest, str) and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest)
            and record.get("sample_digest") == digest),
        "statistics_finite": all(finite_statistic(record.get(k)) for k in ("cpu_statistic", "cuda_statistic")),
        "structural_failure_present": any(reason in STRUCTURAL_REASONS for reason in reasons),
        "unknown_structural_failure_present": any(reason not in STRUCTURAL_REASONS for reason in reasons),
        "cuda_failure_present": record.get("cuda_failure_reason", "MISSING") is not None,
        "eligible_classifications": all(record.get(k) == "ELIGIBLE"
                                       for k in ("cpu_classification", "cuda_classification")),
        "record_schema_complete": RECORD_FIELDS <= record.keys() and numeric_evidence_schema_complete(record),
        "distribution_rows_pass": all(row.get("passed") is True for row in record.get("distribution_evidence", [])
                                      if isinstance(row, dict)),
        **{gate: record.get(gate) is True for gate in GATES},
    }
    result["valid"] = (
        all(result[k] for k in ("identity_match", "seed_identity_match", "sample_digest_match",
                               "statistics_finite", "eligible_classifications", "record_schema_complete",
                               "distribution_rows_pass", *GATES))
        and not any(result[k] for k in ("structural_failure_present", "unknown_structural_failure_present",
                                       "cuda_failure_present")))
    return result


def adjudicate_mc(records, expected_records, outer, aggregate):
    """Separate DEC-024 layer. Read-only inputs; raw aggregate is never corrected."""
    require(len(records) == len(expected_records) == B_EQ + 1, "complete 16-record MC outer required")
    require(len(outer["bootstrap_identities"]) == B_EQ, "15 frozen bootstrap identities required")
    require(isinstance(aggregate, dict), "raw aggregate must be an object")
    observed = records[0]
    obs = record_adjudication_checks(observed, expected_records[0], outer, outer, observed=True)
    checks, evidence = [obs], []
    for record, expected, frozen in zip(records[1:], expected_records[1:], outer["bootstrap_identities"]):
        boot = record_adjudication_checks(record, expected, frozen, outer)
        checks.append(boot)
        T_cpu_boot, T_cpu_obs = record.get("cpu_statistic"), observed.get("cpu_statistic")
        T_cuda_boot, T_cuda_obs = record.get("cuda_statistic"), observed.get("cuda_statistic")
        cpu_finite = finite_statistic(T_cpu_boot) and finite_statistic(T_cpu_obs)
        cuda_finite = finite_statistic(T_cuda_boot) and finite_statistic(T_cuda_obs)
        cpu_exceedance = bool(T_cpu_boot >= T_cpu_obs) if cpu_finite else None
        cuda_exceedance = bool(T_cuda_boot >= T_cuda_obs) if cuda_finite else None
        cpu_reference_exact_tie = (
            cpu_finite and math.isfinite(T_cpu_boot) and math.isfinite(T_cpu_obs)
            and T_cpu_boot == T_cpu_obs)
        mismatch = cpu_finite and cuda_finite and cpu_exceedance != cuda_exceedance
        certified = (
            cpu_reference_exact_tie and mismatch and obs["valid"] and boot["valid"]
            and cpu_exceedance is True and cuda_exceedance is False)
        evidence.append({
            "identity": record.get("identity"), "raw_inner_index": record.get("raw_inner_index"),
            "T_cpu_obs": T_cpu_obs, "T_cuda_obs": T_cuda_obs,
            "T_cpu_boot": T_cpu_boot, "T_cuda_boot": T_cuda_boot,
            "cpu_exceedance": cpu_exceedance, "cuda_exceedance": cuda_exceedance,
            "indicator_mismatch": mismatch, "cpu_reference_exact_tie": cpu_reference_exact_tie,
            "identity_match": obs["identity_match"] and boot["identity_match"],
            "seed_identity_match": obs["seed_identity_match"] and boot["seed_identity_match"],
            "sample_digest_match": obs["sample_digest_match"] and boot["sample_digest_match"],
            **{"observed_" + gate: obs[gate] for gate in GATES},
            **{"bootstrap_" + gate: boot[gate] for gate in GATES},
            "statistics_finite": cpu_finite and cuda_finite,
            "structural_failure_present": obs["structural_failure_present"] or boot["structural_failure_present"],
            "cuda_failure_present": obs["cuda_failure_present"] or boot["cuda_failure_present"],
            "observed_provenance": obs, "bootstrap_provenance": boot,
            "certified_reference_exact_tie_crossing": certified,
        })
    failures = []
    if not all(check["valid"] for check in checks):
        failures.append("RECORD_GATE_OR_PROVENANCE")
    recomputed = {}
    for engine in ("cpu", "cuda"):
        indicators = [row[engine + "_exceedance"] for row in evidence]
        count = sum(indicators) if all(type(x) is bool for x in indicators) else None
        p = (count + 1) / (B_EQ + 1) if count is not None else None
        reject = p <= 0.05 if p is not None else None
        recomputed.update({f"recomputed_b_{engine}": count, f"recomputed_p_{engine}": p,
                           f"recomputed_reject_{engine}": reject})
        b_raw, p_raw, r_raw = (aggregate.get(k + "_" + engine) for k in ("b", "p", "reject"))
        schema = integer(b_raw, B_EQ + 1) and finite_statistic(p_raw) and type(r_raw) is bool
        if not schema or count is None or (b_raw, p_raw, r_raw) != (count, p, reject):
            failures.append("RAW_AGGREGATE_RECOMPUTATION_" + engine.upper())
    raw = {"raw_" + k: aggregate.get(k) for k in
           ("b_cpu", "b_cuda", "p_cpu", "p_cuda", "reject_cpu", "reject_cuda")}
    count_match = (integer(raw["raw_b_cpu"], B_EQ + 1) and integer(raw["raw_b_cuda"], B_EQ + 1)
                   and raw["raw_b_cpu"] == raw["raw_b_cuda"])
    reject_match = (type(raw["raw_reject_cpu"]) is bool and type(raw["raw_reject_cuda"]) is bool
                    and raw["raw_reject_cpu"] == raw["raw_reject_cuda"])
    mismatches = sum(row["indicator_mismatch"] for row in evidence)
    certified = sum(row["certified_reference_exact_tie_crossing"] for row in evidence)
    unexplained = mismatches - certified
    require(unexplained >= 0, "negative unexplained indicator count")
    signed = (integer(raw["raw_b_cpu"], B_EQ + 1) and integer(raw["raw_b_cuda"], B_EQ + 1)
              and raw["raw_b_cpu"] - raw["raw_b_cuda"] == certified)
    if unexplained:
        failures.append("UNEXPLAINED_INDICATOR_MISMATCH")
    if not signed or (count_match and mismatches):
        failures.append("SIGNED_ACCOUNTING")
    if not reject_match:
        failures.append("REJECT_DECISION_MISMATCH")
    if aggregate.get("mc_evaluable") is not True:
        failures.append("MC_NOT_EVALUABLE")
    if aggregate.get("cuda_mc_unavailable_records") != []:
        failures.append("CUDA_MC_UNAVAILABLE")
    # The immutable frozen aggregate's gates mean RAW equality. Both false is
    # expected on a certified raw count mismatch, never rewritten to true.
    raw_gate_expected = count_match and reject_match
    for gate in ("mc_gate_pass", "outer_gate_pass"):
        if aggregate.get(gate) is not raw_gate_expected:
            failures.append("RAW_AGGREGATE_GATE_" + gate)
    return {
        **raw, **recomputed, "bootstrap_adjudication": evidence,
        "indicator_mismatch_count": mismatches,
        "certified_reference_exact_tie_crossing_count": certified,
        "unexplained_indicator_mismatch_count": unexplained,
        "MC_RAW_EXCEEDANCE_COUNT_MATCH": count_match,
        "MC_REFERENCE_EXACT_TIE_CROSSING_COUNT": certified,
        "MC_UNEXPLAINED_INDICATOR_MISMATCH": unexplained,
        "MC_REJECT_DECISION_MATCH": reject_match,
        "MC_EQUIVALENCE_ADJUDICATED": not failures,
        "failures": failures,
    }


def check_mc(state, adjudication):
    counts = state["counters"]
    counts["MC_RAW_EXCEEDANCE_COUNT_MISMATCH_OUTERS"] += int(not adjudication["MC_RAW_EXCEEDANCE_COUNT_MATCH"])
    counts["MC_REFERENCE_EXACT_TIE_CROSSINGS"] += adjudication["MC_REFERENCE_EXACT_TIE_CROSSING_COUNT"]
    counts["MC_UNEXPLAINED_INDICATOR_MISMATCH"] += adjudication["MC_UNEXPLAINED_INDICATOR_MISMATCH"]
    counts["MC_REJECT_DECISION_MISMATCH"] += int(not adjudication["MC_REJECT_DECISION_MATCH"])
    failed = adjudication["failures"]
    specific = {"UNEXPLAINED_INDICATOR_MISMATCH", "SIGNED_ACCOUNTING", "REJECT_DECISION_MISMATCH"}
    if any(reason not in specific for reason in failed):
        counts["UNEXPLAINED_DISCREPANCY"] += 1
    state["context"]["failed_gates"] = list(dict.fromkeys(failed))
    require(adjudication["MC_EQUIVALENCE_ADJUDICATED"] is True, "MC adjudication failed")
    counts["MC_OUTERS_ADJUDICATED"] += 1


def run_workload_b(runtime, manifest, cells, output, state):
    for outer in manifest["mc_failed_outers"]:
        context(state, "workload_b", outer)
        cell = cells[outer["cell_id"]]
        observed, meta = runtime.runner.fixed_observed(cell, outer["raw_outer_index"], NAMESPACE)
        _, attempts, eligible = runtime.runner.fixed_bootstraps(cell, observed, outer["raw_outer_index"], NAMESPACE)
        actual = [(f"{cell.canonical_id}|raw_outer={outer['raw_outer_index']}|raw_inner={r['raw_inner_index']}",
                   r["raw_inner_index"]) for r in eligible]
        expected = [(r["identity"], r["raw_inner_index"]) for r in outer["bootstrap_identities"]]
        if actual != expected:
            state["counters"]["MC_BOOTSTRAP_IDENTITY_MISMATCH"] += 1
            state["context"]["failed_gates"] = ["MC_BOOTSTRAP_IDENTITY_MISMATCH"]
            append_jsonl(output / "workload_b_records.jsonl",
                         {"outer_identity": outer["outer_identity"], "expected": expected,
                          "reconstructed": actual, "failure_reason": "MC_BOOTSTRAP_IDENTITY_MISMATCH"})
            raise HarnessContractError("Workload B reconstructed identity/order/gaps mismatch")
        observed_item = {"identity": outer["observed_identity"], "record_type": "observed",
                         "cell_id": cell.canonical_id, "raw_outer_index": outer["raw_outer_index"],
                         "raw_inner_index": None}
        records = []
        expected_records = []
        sequence = [(observed_item, observed, meta["seed_identity"], meta["sample_digest_sha256"])]
        for frozen, reconstructed in zip(outer["bootstrap_identities"], eligible):
            item = {**observed_item, **frozen, "record_type": "bootstrap"}
            sequence.append((item, reconstructed["sample"], reconstructed["seed_identity"], reconstructed["sample_digest"]))
        for item, sample, seed, digest in sequence:
            context(state, "workload_b", item)
            require(sha256(sample.tobytes()) == digest, "reconstructed sample digest mismatch")
            record = {**evaluate_record(runtime, cell, item, sample), "seed_identity": seed}
            consume_record(state, output, "workload_b", item, record)
            records.append(record)
            expected_records.append({"expected_identity": item["identity"],
                                     "expected_seed_identity": seed,
                                     "expected_sample_digest": digest})
        context(state, "workload_b_mc", outer)
        aggregate = runtime.runner.aggregate_outer(records[0], records[1:])
        state["counters"]["MC_OUTERS_RECONSTRUCTED"] += 1
        adjudication = adjudicate_mc(records, expected_records, outer, aggregate)
        result = {"outer_identity": outer["outer_identity"], "bootstrap_identities": outer["bootstrap_identities"],
                  "raw_bootstrap_attempt_count": len(attempts), **aggregate,
                  "raw_aggregate": aggregate, "adjudication": adjudication,
                  "r9_provenance": {k: v for k, v in outer.items() if k.startswith("r9_")}}
        state["mc_results"].append(result)
        write_json(output / "mc_results.json", state["mc_results"])
        check_mc(state, adjudication)


def final_summary(state, failure=False):
    passed = (
        not failure and {key: state["counters"].get(key) for key in REQUIRED_COUNTERS} == REQUIRED_COUNTERS
        and integer(state["counters"].get(DIAGNOSTIC_COUNTERS[0]), WORKLOAD_B_OUTER_EXPECTED + 1)
        and integer(state["counters"].get(DIAGNOSTIC_COUNTERS[1]), WORKLOAD_B_OUTER_EXPECTED * B_EQ + 1)
        and state["WORKLOAD_A_RECORDS_EXECUTED"] == WORKLOAD_A_EXPECTED
        and state["WORKLOAD_B_RECORDS_EXECUTED"] == WORKLOAD_B_OUTER_EXPECTED * (B_EQ + 1)
        and all(value == 0 for view in ("STRUCTURAL_REASON_ITEM_COUNTS", "STRUCTURAL_REASON_RECORD_COUNTS")
                for value in state[view].values())
        and len(state["mc_results"]) == WORKLOAD_B_OUTER_EXPECTED
        and all(row.get("adjudication", {}).get("MC_EQUIVALENCE_ADJUDICATED") is True
                for row in state["mc_results"])
    )
    return {
        "WORK_ITEM": WORK_ITEM, "DECISION": DECISION, "REPLAY_VERSION": REPLAY_VERSION,
        "TARGETED_REPLAY_PASS": "YES" if passed else "NO",
        "execution_state": ("COMPLETE" if passed else
                            "SCIENTIFIC_FAILURE" if state["authorization_consumed"] else
                            "PRE_IDENTITY_FAILURE" if state["preflight_passed"] else "PREFLIGHT_FAILURE"),
        **authorization_status(state),
        **MC_LIMITATION,
        **state["counters"],
        **{key: state[key] for key in ("STRUCTURAL_REASON_ITEM_COUNTS", "STRUCTURAL_REASON_RECORD_COUNTS",
                                       "WORKLOAD_A_RECORDS_EXECUTED", "WORKLOAD_B_RECORDS_EXECUTED")},
        "claim": ("CPU/CUDA targeted equivalence passed on the frozen R10-A workload, "
                  "with any raw MC count differences limited to certified canonical "
                  "CPU-reference exact-tie crossings and with identical reject decisions.") if passed else None,
    }


def gpu_environment(runtime):
    cp = runtime.cuda_candidate.require_cuda()
    count = int(cp.cuda.runtime.getDeviceCount())
    require(count > 0, "no CUDA GPU")
    device = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(device)
    name = properties.get("name", properties.get(b"name"))
    require(name is not None, "GPU name unavailable")
    name = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
    try:
        driver, driver_error = int(cp.cuda.runtime.driverGetVersion()), None
    except Exception as exc:
        driver, driver_error = None, str(exc)
    return {
        "cupy_version": str(cp.__version__), "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "gpu_count": count, "gpu_device_id": device, "gpu_name": name,
        "cuda_driver_version": driver, "cuda_driver_query_error": driver_error,
    }


def gpu_smoke(runtime):
    cp = runtime.cuda_candidate.require_cuda()
    values = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float64)
    result = float(cp.asnumpy(cp.sum(values)))
    cp.cuda.runtime.deviceSynchronize()
    require(result == 6.0, "minimal CUDA smoke failed")
    return {"float64_sum": result, "passed": True}


def authorization_status(state):
    consumed = state["authorization_consumed"]
    return {
        "AUTHORIZATION_CONSUMED": "YES" if consumed else "NO",
        "AUTHORIZATION_NOT_CONSUMED": "NO" if consumed else "YES",
        "EXECUTION_STARTED": "YES" if consumed else "NO",
        "EXECUTION_NOT_STARTED": "NO" if consumed else "YES",
        "AUTO_RERUN": "NO", "AUTO_RESUME": "NO",
    }


def begin_scientific_execution(state, output):
    if state["authorization_consumed"]:
        return
    require(state["preflight_passed"], "scientific execution before readiness")
    # Persist the boundary before evaluating the first identity. If persistence
    # fails, no evaluator is called and the in-memory authorization stays unused.
    started = {**state, "authorization_consumed": True}
    write_json(output / "authorization.json", {
        **authorization_status(started), "timestamp": now(), **state["context"],
    })
    state["authorization_consumed"] = True


def load_cuda_library(soname, environment):
    # Try the effective dynamic loader first, then explicitly recorded wheel/
    # toolkit locations. Do not mutate loader paths, install or repair anything.
    candidates = [soname]
    directories = [Path(p) for p in (environment["LD_LIBRARY_PATH"] or "").split(os.pathsep) if p]
    for key in ("CUDA_WHEEL_ROOT", "CUDA_PATH"):
        if environment[key]:
            root = Path(environment[key])
            directories.extend((root / "lib", root / "lib64", root / "targets/x86_64-linux/lib"))
    candidates.extend(str(directory / soname) for directory in directories)
    errors = []
    for candidate in dict.fromkeys(candidates):
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise HarnessContractError(f"cannot load {soname}: " + "; ".join(errors))


def strong_environment_preflight(runtime, report, requested, repo):
    """Readiness only: deterministic microchecks, no fixtures or scientific RNG."""
    report.update({key: os.environ.get(key) for key in
                   ("CUDA_WHEEL_ROOT", "CUDA_PATH", "LD_LIBRARY_PATH")})
    report["checks"] = checks = {}

    def check(name, operation):
        checks[name] = "RUNNING"
        try:
            result = operation()
        except BaseException:
            checks[name] = "FAIL"
            raise
        checks[name] = "PASS"
        return result

    # All CuPy smoke caches live outside the scientific checkout and requested
    # output. A unique source also defeats CuPy's per-device in-memory memoizer.
    temp_root = Path(tempfile.gettempdir()).resolve()
    require(not inside(temp_root, requested) and not inside(temp_root, repo),
            "preflight cache root must be outside scientific output/checkout")
    previous = {key: os.environ.get(key) for key in ("CUPY_CACHE_DIR", "CUPY_CACHE_IN_MEMORY")}
    try:
        with tempfile.TemporaryDirectory(prefix="r3-nvrtc-", dir=temp_root) as cache:
            cache_path = Path(cache)
            require(not any(cache_path.iterdir()), "preflight cache must be fresh")
            os.environ["CUPY_CACHE_DIR"] = cache
            os.environ["CUPY_CACHE_IN_MEMORY"] = "0"
            report["rawkernel_cache"] = {"fresh": True, "temporary": True, "path": cache,
                                         "removed": False, "source_unique": True, "backend": "nvrtc"}
            report.update(check("CUDA_DEVICE_COUNT", lambda: gpu_environment(runtime)))
            require(report["gpu_count"] >= 1, "no CUDA GPU")
            report["smoke"] = check("FLOAT64_BASIC_SMOKE", lambda: gpu_smoke(runtime))
            cp = runtime.cuda_candidate.require_cuda()
            require(not cp.cuda.runtime.is_hip, "NVRTC requires CUDA, not HIP")
            require(report["cuda_driver_version"] is not None, "CUDA driver version unavailable")
            cudart, cudart_path = check("LIBCUDART_SO_13_LOAD",
                                       lambda: load_cuda_library("libcudart.so.13", report))
            nvrtc, nvrtc_path = check("LIBNVRTC_SO_13_LOAD",
                                     lambda: load_cuda_library("libnvrtc.so.13", report))
            report["library_paths"] = {"libcudart.so.13": cudart_path, "libnvrtc.so.13": nvrtc_path}

            def version_check():
                major, minor = ctypes.c_int(), ctypes.c_int()
                nvrtc.nvrtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
                nvrtc.nvrtcVersion.restype = ctypes.c_int
                require(nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) == 0,
                        "nvrtcVersion failed")
                version = (major.value, minor.value)
                report["nvrtc_version"] = version
                report["cupy_nvrtc_version"] = tuple(cp.cuda.nvrtc.getVersion())
                require(version == report["cupy_nvrtc_version"] == (13, 0), "NVRTC_VERSION must be (13,0)")
            check("NVRTC_VERSION", version_check)

            # See CuPy v13.6.0 cupy/_core/raw.pyx _get_raw_module and
            # cupy/cuda/compiler.py _compile_with_cache_cuda: source participates
            # in both cache keys. A new cache path makes this source unique,
            # without sampling any RNG used by the scientific procedure.
            token = sha256(cache.encode("utf-8"))
            code = ('// fresh preflight ' + token + '\n'
                    'extern "C" __global__ void r3_smoke(const double* x, double* y) {'
                    ' int i = threadIdx.x; if (i < 3) y[i] = 2.0 * x[i] + 1.0; }')
            kernel = cp.RawKernel(code, "r3_smoke", backend="nvrtc")
            require(kernel.backend == "nvrtc", "RawKernel backend must be NVRTC")
            check("RAWKERNEL_NVRTC_COMPILE", kernel.compile)
            values = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float64)
            result = cp.empty_like(values)

            def launch():
                kernel((1,), (3,), (values, result))
                cp.cuda.runtime.deviceSynchronize()
            check("RAWKERNEL_NVRTC_LAUNCH", launch)
            check("RAWKERNEL_NUMERICAL_RESULT",
                  lambda: require(cp.asnumpy(result).tolist() == [3.0, 5.0, 7.0],
                                  "RawKernel float64 result mismatch"))

            def elementwise():
                result = cp.sum(values * values + 1.0)
                cp.cuda.runtime.deviceSynchronize()
                require(float(cp.asnumpy(result)) == 17.0, "CuPy elementwise/reduction failed")
            check("CUPY_ELEMENTWISE_REDUCTION", elementwise)
            special = importlib.import_module("cupyx.scipy.special")

            def special_check(name, args, expected):
                result = getattr(special, name)(*args)
                cp.cuda.runtime.deviceSynchronize()
                actual = float(cp.asnumpy(result))
                require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                        f"cupyx.scipy.{name} failed")
            check("CUPYX_SCIPY_GAMMALN",
                  lambda: special_check("gammaln", (cp.asarray(5.0, dtype=cp.float64),), math.log(24.0)))
            check("CUPYX_SCIPY_BETAINC",
                  lambda: special_check("betainc", (cp.asarray(2.0, dtype=cp.float64),
                                                   cp.asarray(3.0, dtype=cp.float64),
                                                   cp.asarray(0.5, dtype=cp.float64)), 0.6875))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if "rawkernel_cache" in report:
            report["rawkernel_cache"]["removed"] = not Path(report["rawkernel_cache"]["path"]).exists()
    return report


def preserve_failure(output, state, exc):
    payload = {**state["context"], "exception_type": type(exc).__name__,
               "exception": str(exc), "traceback": traceback.format_exc(),
               "PRESERVE_EVIDENCE": True, **authorization_status(state),
               "OUTPUT_NOT_CREATED": "YES" if output is None else "NO",
               "execution_state": final_summary(state, failure=True)["execution_state"],
               "environment_preflight": state["environment_preflight"],
               "scientific_counters": state["counters"]}
    print(json.dumps(json_safe(payload)), file=sys.stderr)
    if output is None:
        return
    # Independent attempts: one failed write must not suppress all other available evidence.
    operations = (
        lambda: write_json(output / "authorization.json", {**authorization_status(state), **state["context"]}),
        lambda: write_json(output / "failure.json", payload),
        lambda: write_json(output / "summary.json", final_summary(state, failure=True)),
        lambda: write_json(output / "mc_results.json", state["mc_results"]),
        lambda: write_digests(output),
    )
    for operation in operations:
        try:
            operation()
        except Exception as error:
            print(f"additional preservation failure: {error}", file=sys.stderr)


def execute(args):
    state, output = initial_state(), None
    try:
        context(state, "static_validation")
        validate_static()
        manifest, _ = load_manifest()
        context(state, "repo_preflight")
        repo = Path(args.repo).resolve()
        repo_state = preflight_repo(repo)
        requested = Path(args.output)
        require(not requested.is_symlink() and not requested.exists(), "fresh output already exists")
        requested = requested.resolve()
        require(not inside(requested, repo), "output must be outside scientific checkout")
        context(state, "scientific_imports")
        runtime = load_frozen_runtime(repo)
        context(state, "module_origins")
        environment = {"platform": platform.platform(), "python": sys.version,
                       "loaded_module_paths": verify_loaded_scientific_modules(repo),
                       "numpy_version": str(runtime.numpy.__version__),
                       "scipy_version": str(runtime.scipy.__version__)}
        context(state, "strong_environment_preflight")
        strong_environment_preflight(runtime, state["environment_preflight"], requested, repo)
        environment.update(state["environment_preflight"])
        state["preflight_passed"] = True
        context(state, "fresh_output")
        requested.mkdir(parents=True, exist_ok=False)
        output = requested
        context(state, "provenance")
        provenance = {
            "timestamp": now(), **repo_state, "preregistration_sha": PREREGISTRATION_SHA,
            "harness_sha256": sha256(Path(__file__).read_bytes()),
            "manifest_sha256": IDENTITY_MANIFEST_SHA256, "ordered_payload_sha256": ORDERED_PAYLOAD_SHA256,
            "decision": DECISION, "replay_version": REPLAY_VERSION, "namespace": NAMESPACE,
            "R_EQ": R_EQ, "B_EQ": B_EQ, "FRESH_OUTPUT_REQUIRED": True,
            **MC_LIMITATION,
            "CHECKPOINT_REUSE": False, "PRIOR_FAILED_OUTPUT_REUSE": False,
            **authorization_status(state), "authorization_state_file": "authorization.json",
        }
        write_json(output / "authorization.json", authorization_status(state))
        write_json(output / "execution_manifest.json", provenance)
        write_json(output / "environment.json", {**provenance, **environment})
        for name in ("workload_a_records.jsonl", "workload_b_records.jsonl"):
            (output / name).touch(exist_ok=False)
        write_json(output / "mc_results.json", [])
        context(state, "fixture_selection")
        cells = build_cell_index(runtime, manifest)
        run_workload_a(runtime, manifest, cells, output, state)
        run_workload_b(runtime, manifest, cells, output, state)
        context(state, "finalize")
        verify_loaded_scientific_modules(repo)
        require(final_summary(state)["TARGETED_REPLAY_PASS"] == "YES", "final acceptance failed")
        write_json(output / "summary.json", final_summary(state))
        write_digests(output)
        return 0
    except BaseException as exc:
        if (state["authorization_consumed"] and not state["context"]["failed_gates"]
                and not state["context"]["structural_failure_reasons"]):
            state["counters"]["UNEXPLAINED_DISCREPANCY"] += 1
        preserve_failure(output, state, exc)
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-static", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--repo", help="explicit exact scientific checkout root")
    parser.add_argument("--output", help="fresh output directory outside scientific checkout")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.validate_static:
        if args.repo is not None or args.output is not None:
            parser.error("static mode does not accept execution paths")
        try:
            print(json.dumps(validate_static(), indent=2, sort_keys=True))
            return 0
        except Exception as exc:
            print(f"STATIC_VALIDATION=FAIL: {exc}", file=sys.stderr)
            return 1
    if not args.repo or not args.output:
        parser.error("--execute requires --repo and --output")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
