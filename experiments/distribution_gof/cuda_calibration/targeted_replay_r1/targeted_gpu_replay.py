#!/usr/bin/env python3
"""External, fail-closed harness for the preregistered R10-A targeted replay.

Static validation is deliberately standard-library-only.  Scientific modules
from pyMagicStats are imported lazily, and only after ``--execute`` has passed
the exact-repository preflight and created a fresh evidence directory.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback
import types
from pathlib import Path
from typing import Any


WORK_ITEM = "CP05-C2C-R10-A-TARGETED-GPU-REPLAY-HARNESS"
PREREGISTRATION_SHA = "b8255f4fdebe59bf534003b2fdcdaac687bd6543"
REPLAY_EXECUTION_SHA = "649ca296c57ab237f3e93ea35ac9870e1d1bfc9d"
R9_EXECUTION_SHA = "01c9c0759d3ffec1bbd95cb5aa3f67793aab544b"
IDENTITY_MANIFEST_SHA256 = "efb550af870213160ed450c222043a0c73cc415a7aecb633c62d61118d72b36d"
SCHEMA_VERSION = "cp05-c2c-r10a-targeted-replay-v1"
DECISION = "DEC-019"
NAMESPACE = "CP05-C2C"
R_EQ = 8
B_EQ = 15
WORKLOAD_A_EXPECTED = 134
WORKLOAD_B_OUTER_EXPECTED = 12
MANIFEST_PATH = Path(__file__).resolve().with_name("frozen_identity_manifest.json")

REQUIRED_COUNTERS = {
    "CLASSIFICATION_MISMATCH": 0,
    "CUDA_NONCONVERGENCE": 0,
    "FIT_GATE_FAILURE": 0,
    "DISTRIBUTION_VALUE_GATE_FAILURE": 0,
    "STATISTIC_GATE_FAILURE": 0,
    "UNEXPLAINED_DISCREPANCY": 0,
    "MC_BOOTSTRAP_IDENTITY_MISMATCH": 0,
    "MC_OUTERS_RECONSTRUCTED": WORKLOAD_B_OUTER_EXPECTED,
    "MC_EXCEEDANCE_COUNT_MISMATCH": 0,
    "MC_REJECT_DECISION_MISMATCH": 0,
}

PROJECT_MODULES = {
    "runner": "experiments.distribution_gof.cuda_calibration.cp05_c2c_equivalence_runner",
    "preregistration": "experiments.distribution_gof.cuda_calibration.equivalence_preregistration",
    "engine": "experiments.distribution_gof.cuda_calibration.cp05_cuda_engine",
    "nb_support": "experiments.distribution_gof.cuda_calibration.nb_support",
    "cuda_candidate": "experiments.distribution_gof.cuda_calibration.cuda_candidate",
}

EXPECTED_MODULE_FILES = {
    "runner": Path("experiments/distribution_gof/cuda_calibration/cp05_c2c_equivalence_runner.py"),
    "preregistration": Path("experiments/distribution_gof/cuda_calibration/equivalence_preregistration.py"),
    "engine": Path("experiments/distribution_gof/cuda_calibration/cp05_cuda_engine.py"),
    "nb_support": Path("experiments/distribution_gof/cuda_calibration/nb_support.py"),
    "cuda_candidate": Path("experiments/distribution_gof/cuda_calibration/cuda_candidate.py"),
}


class HarnessContractError(RuntimeError):
    """A frozen identity, environment, or replay contract was violated."""


@dataclasses.dataclass(frozen=True)
class FrozenRuntime:
    runner: Any
    preregistration: Any
    engine: Any
    nb_support: Any
    cuda_candidate: Any
    loaded_module_paths: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def load_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    if not MANIFEST_PATH.is_file():
        raise HarnessContractError(f"missing frozen manifest: {MANIFEST_PATH}")
    digest = sha256_file(MANIFEST_PATH)
    if digest != IDENTITY_MANIFEST_SHA256:
        raise HarnessContractError(
            f"manifest SHA256 mismatch: expected {IDENTITY_MANIFEST_SHA256}, got {digest}"
        )
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HarnessContractError("frozen manifest is not valid UTF-8 JSON") from exc
    report = validate_manifest(manifest)
    return manifest, report


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "r9_execution_sha": R9_EXECUTION_SHA,
        "replay_execution_sha": REPLAY_EXECUTION_SHA,
        "namespace": NAMESPACE,
        "R_EQ": R_EQ,
        "B_EQ": B_EQ,
        "persisted_failure_record_count": WORKLOAD_A_EXPECTED,
        "mc_failed_outer_count": WORKLOAD_B_OUTER_EXPECTED,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise HarnessContractError(f"frozen manifest constants mismatch: {mismatches}")

    workload_a = manifest.get("persisted_failure_record_identities")
    if not isinstance(workload_a, list) or len(workload_a) != WORKLOAD_A_EXPECTED:
        raise HarnessContractError("Workload A must contain exactly 134 records")
    identities = [item.get("identity") for item in workload_a]
    if None in identities or len(set(identities)) != WORKLOAD_A_EXPECTED:
        raise HarnessContractError("Workload A identities must be present and unique")
    observed = 0
    bootstrap = 0
    families: set[str] = set()
    for item in workload_a:
        identity = item.get("identity")
        cell_id = item.get("cell_id")
        family = item.get("family")
        record_type = item.get("record_type")
        raw_outer = item.get("raw_outer_index")
        raw_inner = item.get("raw_inner_index")
        if not isinstance(cell_id, str) or not isinstance(identity, str):
            raise HarnessContractError("Workload A identity/cell_id must be strings")
        if not isinstance(raw_outer, int) or not 0 <= raw_outer < R_EQ:
            raise HarnessContractError(f"invalid Workload A raw_outer_index: {identity}")
        families.add(family)
        outer_identity = f"{cell_id}|raw_outer={raw_outer}"
        if record_type == "observed":
            observed += 1
            if raw_inner is not None or identity != outer_identity:
                raise HarnessContractError(f"invalid observed identity: {identity}")
        elif record_type == "bootstrap":
            bootstrap += 1
            if not isinstance(raw_inner, int) or raw_inner < 0:
                raise HarnessContractError(f"invalid bootstrap raw_inner_index: {identity}")
            if identity != f"{outer_identity}|raw_inner={raw_inner}":
                raise HarnessContractError(f"inconsistent bootstrap identity: {identity}")
        else:
            raise HarnessContractError(f"unknown Workload A record_type: {record_type}")
    if observed != 10 or bootstrap != 124 or families != {"negative_binomial"}:
        raise HarnessContractError(
            f"Workload A topology mismatch: observed={observed}, bootstrap={bootstrap}, families={families}"
        )

    workload_b = manifest.get("mc_failed_outers")
    if not isinstance(workload_b, list) or len(workload_b) != WORKLOAD_B_OUTER_EXPECTED:
        raise HarnessContractError("Workload B must contain exactly 12 outers")
    outer_ids = [item.get("outer_identity") for item in workload_b]
    if None in outer_ids or len(set(outer_ids)) != WORKLOAD_B_OUTER_EXPECTED:
        raise HarnessContractError("Workload B outer identities must be present and unique")
    all_bootstrap_ids: list[str] = []
    raw_inner_order_valid = True
    gaps_preserved = False
    for outer in workload_b:
        cell_id = outer.get("cell_id")
        raw_outer = outer.get("raw_outer_index")
        outer_identity = outer.get("outer_identity")
        if not isinstance(cell_id, str) or not isinstance(raw_outer, int) or not 0 <= raw_outer < R_EQ:
            raise HarnessContractError(f"invalid Workload B outer: {outer_identity}")
        canonical_outer = f"{cell_id}|raw_outer={raw_outer}"
        if outer_identity != canonical_outer or outer.get("observed_identity") != canonical_outer:
            raise HarnessContractError(f"inconsistent Workload B observed identity: {outer_identity}")
        bootstraps = outer.get("bootstrap_identities")
        if not isinstance(bootstraps, list) or len(bootstraps) != B_EQ:
            raise HarnessContractError(f"Workload B outer does not have 15 bootstraps: {outer_identity}")
        indices = [item.get("raw_inner_index") for item in bootstraps]
        ids = [item.get("identity") for item in bootstraps]
        if any(not isinstance(index, int) or index < 0 for index in indices):
            raise HarnessContractError(f"invalid Workload B raw_inner_index: {outer_identity}")
        if any(not isinstance(identity, str) for identity in ids):
            raise HarnessContractError(f"invalid Workload B bootstrap identity: {outer_identity}")
        if len(set(indices)) != B_EQ or len(set(ids)) != B_EQ:
            raise HarnessContractError(f"duplicate Workload B bootstrap identity: {outer_identity}")
        if any(right <= left for left, right in zip(indices, indices[1:])):
            raw_inner_order_valid = False
        if any(right - left > 1 for left, right in zip(indices, indices[1:])):
            gaps_preserved = True
        for identity, index in zip(ids, indices):
            if identity != f"{canonical_outer}|raw_inner={index}":
                raise HarnessContractError(f"inconsistent Workload B bootstrap identity: {identity}")
        all_bootstrap_ids.extend(ids)
    if not raw_inner_order_valid:
        raise HarnessContractError("Workload B raw_inner_index order is not strictly increasing")
    if len(set(all_bootstrap_ids)) != WORKLOAD_B_OUTER_EXPECTED * B_EQ:
        raise HarnessContractError("duplicate bootstrap identity across Workload B outers")
    if not gaps_preserved:
        raise HarnessContractError("Workload B manifest unexpectedly contains no raw_inner_index gaps")

    return {
        "manifest_sha256": IDENTITY_MANIFEST_SHA256,
        "workload_a_count": len(workload_a),
        "workload_a_unique": len(set(identities)),
        "workload_a_observed": observed,
        "workload_a_bootstrap": bootstrap,
        "workload_a_families": sorted(families),
        "workload_b_outers": len(workload_b),
        "workload_b_all_have_15_bootstraps": True,
        "workload_b_raw_inner_order_valid": raw_inner_order_valid,
        "workload_b_gaps_preserved": gaps_preserved,
    }


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def static_ast_audit(source_path: Path) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    allowed_import_roots = {
        "__future__", "argparse", "ast", "dataclasses", "datetime", "hashlib",
        "importlib", "json", "math", "os", "platform", "shutil", "subprocess",
        "sys", "traceback", "types", "pathlib", "typing",
    }
    nonstdlib_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            nonstdlib_imports.extend(alias.name for alias in node.names if alias.name.split(".")[0] not in allowed_import_roots)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in allowed_import_roots:
                nonstdlib_imports.append(node.module or "")
    if nonstdlib_imports:
        raise HarnessContractError(f"static mode has non-stdlib imports: {nonstdlib_imports}")

    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def enclosing_function(node: ast.AST) -> str | None:
        current = parent.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current.name
            current = parent.get(current)
        return None

    lazy_import_violations: list[str] = []
    forbidden_calls: list[str] = []
    forbidden_names = {"traverse_primary", "_official_dispatch", "_execute_equivalence"}
    subprocess_forbidden: list[str] = []
    full_matrix_loops = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func) or ""
            if name.endswith("import_module") and enclosing_function(node) != "load_frozen_runtime":
                lazy_import_violations.append(f"line {node.lineno}: {name}")
            leaf = name.rsplit(".", 1)[-1]
            if leaf in forbidden_names or (leaf == "main" and "." in name):
                forbidden_calls.append(f"line {node.lineno}: {name}")
            if name in {"subprocess.run", "subprocess.Popen", "subprocess.check_call", "subprocess.check_output"}:
                literals = [item.value for item in ast.walk(node) if isinstance(item, ast.Constant) and isinstance(item.value, str)]
                joined = " ".join(literals)
                if "cp05_c2c_equivalence_runner" in joined or "--mode equivalence" in joined:
                    subprocess_forbidden.append(f"line {node.lineno}: {joined}")
        if isinstance(node, ast.For):
            iterator = _call_name(node.iter.func) if isinstance(node.iter, ast.Call) else None
            if iterator and iterator.endswith("primary_fixture_matrix"):
                if any(
                    isinstance(child, ast.For)
                    and isinstance(child.iter, ast.Call)
                    and (_call_name(child.iter.func) or "").endswith("range")
                    for child in ast.walk(node)
                ):
                    full_matrix_loops += 1
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            names = [
                _call_name(gen.iter.func) if isinstance(gen.iter, ast.Call) else None
                for gen in node.generators
            ]
            if any(name and name.endswith("primary_fixture_matrix") for name in names) and any(
                name and name.endswith("range") for name in names
            ):
                full_matrix_loops += 1

    validate_node = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_static"),
        None,
    )
    if validate_node is None:
        raise HarnessContractError("validate_static function is missing")
    if any(
        isinstance(node, ast.Call) and (_call_name(node.func) or "").endswith("load_frozen_runtime")
        for node in ast.walk(validate_node)
    ):
        lazy_import_violations.append("validate_static calls load_frozen_runtime")

    violations = lazy_import_violations + forbidden_calls + subprocess_forbidden
    if violations or full_matrix_loops:
        raise HarnessContractError(
            f"static AST audit failed: violations={violations}, full_matrix_loops={full_matrix_loops}"
        )
    return {
        "lazy_runtime_imports": "PASS",
        "static_mode_stdlib_only": "PASS",
        "forbidden_full_runner_calls": 0,
        "forbidden_runner_subprocesses": 0,
        "full_matrix_loop_present": False,
    }


def validate_static() -> dict[str, Any]:
    _, manifest_report = load_manifest()
    ast_report = static_ast_audit(Path(__file__).resolve())
    return {
        "work_item": WORK_ITEM,
        "validation": "PASS",
        "preregistration_sha": PREREGISTRATION_SHA,
        "replay_execution_sha": REPLAY_EXECUTION_SHA,
        "namespace": NAMESPACE,
        "R_EQ": R_EQ,
        "B_EQ": B_EQ,
        **manifest_report,
        **ast_report,
        "resume_supported": False,
        "overwrite_supported": False,
        "cuda_imported": False,
        "gpu_runtime_queried": False,
        "scientific_work_executed": False,
    }


def _git(repo_root: Path, *arguments: str) -> str:
    executable = os.environ.get("GIT_EXE") or shutil.which("git")
    if not executable:
        raise HarnessContractError("git executable is unavailable")
    completed = subprocess.run(
        [executable, "-C", str(repo_root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise HarnessContractError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def preflight_repo(repo_root: Path) -> dict[str, Any]:
    if not repo_root.is_dir():
        raise HarnessContractError(f"repo is not a directory: {repo_root}")
    head = _git(repo_root, "rev-parse", "HEAD")
    if head != REPLAY_EXECUTION_SHA:
        raise HarnessContractError(f"wrong replay checkout: expected {REPLAY_EXECUTION_SHA}, got {head}")
    status = _git(repo_root, "status", "--short")
    if status:
        raise HarnessContractError("replay checkout is not clean")
    return {"repo_head": head, "repo_clean": True}


def verify_loaded_scientific_modules(repo_root: Path) -> dict[str, str]:
    module_paths: dict[str, str] = {}
    for name, module in tuple(sys.modules.items()):
        if name == "experiments" or name.startswith("experiments."):
            location = getattr(module, "__file__", None)
            if location:
                resolved = Path(location).resolve()
                if not _inside(resolved, repo_root):
                    raise HarnessContractError(
                        f"loaded scientific module escaped checkout: {name} -> {resolved}"
                    )
                module_paths[name] = str(resolved)
    return dict(sorted(module_paths.items()))


def load_frozen_runtime(repo_root: Path) -> FrozenRuntime:
    """Import frozen project modules only for an authorized future execution."""
    repo_root = repo_root.resolve()
    for key, relative in EXPECTED_MODULE_FILES.items():
        expected = (repo_root / relative).resolve()
        if not expected.is_file():
            raise HarnessContractError(f"missing frozen module {key}: {expected}")

    for name, module in tuple(sys.modules.items()):
        if name == "experiments" or name.startswith("experiments."):
            location = getattr(module, "__file__", None)
            if location and not _inside(Path(location), repo_root):
                raise HarnessContractError(f"preloaded module shadows frozen checkout: {name} -> {location}")

    repo_text = str(repo_root)
    sys.path[:] = [entry for entry in sys.path if entry != repo_text]
    sys.path.insert(0, repo_text)
    importlib.invalidate_caches()

    loaded: dict[str, Any] = {}
    for key, module_name in PROJECT_MODULES.items():
        loaded[key] = importlib.import_module(module_name)
    module_paths: dict[str, str] = {}
    for key, module in loaded.items():
        location = getattr(module, "__file__", None)
        if not location:
            raise HarnessContractError(f"scientific module has no __file__: {PROJECT_MODULES[key]}")
        resolved = Path(location).resolve()
        expected = (repo_root / EXPECTED_MODULE_FILES[key]).resolve()
        if resolved != expected or not _inside(resolved, repo_root):
            raise HarnessContractError(
                f"import shadowing detected: {PROJECT_MODULES[key]} -> {resolved}; expected {expected}"
            )
        module_paths[PROJECT_MODULES[key]] = str(resolved)

    module_paths.update(verify_loaded_scientific_modules(repo_root))

    return FrozenRuntime(
        runner=loaded["runner"],
        preregistration=loaded["preregistration"],
        engine=loaded["engine"],
        nb_support=loaded["nb_support"],
        cuda_candidate=loaded["cuda_candidate"],
        loaded_module_paths=dict(sorted(module_paths.items())),
    )


def build_cell_index(runtime: FrozenRuntime, manifest: dict[str, Any]) -> dict[str, Any]:
    cells = runtime.runner.primary_fixture_matrix()
    index: dict[str, Any] = {}
    for cell in cells:
        if cell.canonical_id in index:
            raise HarnessContractError(f"duplicate frozen cell: {cell.canonical_id}")
        index[cell.canonical_id] = cell
    selected = {
        item["cell_id"] for item in manifest["persisted_failure_record_identities"]
    } | {item["cell_id"] for item in manifest["mc_failed_outers"]}
    missing = sorted(selected - set(index))
    if missing:
        raise HarnessContractError(f"unknown frozen cells: {missing}")
    return {cell_id: index[cell_id] for cell_id in selected}


def evaluate_eligible_record(
    runtime: FrozenRuntime,
    *,
    identity: str,
    record_type: str,
    cell: Any,
    raw_outer_index: int,
    raw_inner_index: int | None,
    sample: Any,
) -> dict[str, Any]:
    runner = runtime.runner
    support = None
    if cell.family == "negative_binomial":
        bound = runner.reference_fit(cell.family, sample)["bound"]
        support = runtime.nb_support.certify_nb_support(sample, bound, cell.statistic)

    def cuda_adapter(adapter_cell: Any, adapter_sample: Any) -> dict[str, Any]:
        return runner.evaluate_cuda_record(
            adapter_cell,
            adapter_sample,
            certified_support=support.indices if support else None,
            remainder_bound=support.remainder_bound if support else None,
        )

    return runner.evaluate_fixed_record(
        identity=identity,
        record_type=record_type,
        cell=cell,
        raw_outer_index=raw_outer_index,
        raw_inner_index=raw_inner_index,
        sample=sample,
        reference_adapter=runner.evaluate_reference_record,
        cuda_adapter=cuda_adapter,
    )


def initial_state() -> dict[str, int]:
    return {
        "CLASSIFICATION_MISMATCH": 0,
        "CUDA_NONCONVERGENCE": 0,
        "FIT_GATE_FAILURE": 0,
        "DISTRIBUTION_VALUE_GATE_FAILURE": 0,
        "STATISTIC_GATE_FAILURE": 0,
        "UNEXPLAINED_DISCREPANCY": 0,
        "MC_BOOTSTRAP_IDENTITY_MISMATCH": 0,
        "MC_OUTERS_RECONSTRUCTED": 0,
        "MC_EXCEEDANCE_COUNT_MISMATCH": 0,
        "MC_REJECT_DECISION_MISMATCH": 0,
        "WORKLOAD_A_RECORDS_EXECUTED": 0,
        "WORKLOAD_B_RECORDS_EXECUTED": 0,
    }


def update_record_counters(state: dict[str, int], record: dict[str, Any]) -> None:
    named = {
        "classification_gate_pass": "CLASSIFICATION_MISMATCH",
        "fit_gate_pass": "FIT_GATE_FAILURE",
        "distribution_value_gate_pass": "DISTRIBUTION_VALUE_GATE_FAILURE",
        "statistic_gate_pass": "STATISTIC_GATE_FAILURE",
    }
    for field, counter in named.items():
        if record.get(field) is False:
            state[counter] += 1
    failure_reason = record.get("cuda_failure_reason")
    if failure_reason == "CUDA solver non-convergence":
        state["CUDA_NONCONVERGENCE"] += 1
    elif failure_reason:
        state["UNEXPLAINED_DISCREPANCY"] += 1
    expected_eligible = not bool(record.get("observed_ineligible", False))
    if expected_eligible and any(record.get(field) is None for field in named):
        state["UNEXPLAINED_DISCREPANCY"] += 1


def workload_a_record(
    runtime: FrozenRuntime,
    cell: Any,
    item: dict[str, Any],
) -> dict[str, Any]:
    runner = runtime.runner
    raw_outer = item["raw_outer_index"]
    observed, observed_meta = runner.fixed_observed(cell, raw_outer, NAMESPACE)
    if item["record_type"] == "observed":
        cpu_classification = runner._cpu_nb_classification(observed)
        cuda_classification = runner._cuda_nb_classification(observed)
        if cpu_classification != "ELIGIBLE" or cuda_classification != "ELIGIBLE":
            result = runner._ineligible_observed_outer(
                cell,
                raw_outer,
                observed,
                observed_meta,
                cpu_classification,
                cuda_classification,
            )
            record = result["records"][0]
        else:
            record = evaluate_eligible_record(
                runtime,
                identity=item["identity"],
                record_type="observed",
                cell=cell,
                raw_outer_index=raw_outer,
                raw_inner_index=None,
                sample=observed,
            )
    else:
        observed_fit = runner.reference_fit(cell.family, observed)
        raw_inner = item["raw_inner_index"]
        seed = runtime.engine.derive_seed(
            NAMESPACE,
            cell.canonical_id,
            raw_outer,
            "inner_bootstrap",
            raw_inner,
        )
        sample = runtime.engine._generate(
            cell.family,
            observed_fit["parameters"],
            cell.n,
            seed,
        )
        try:
            runner.reference_fit(cell.family, sample)
        except runtime.engine.EngineContractError as exc:
            raise HarnessContractError(
                f"manifest Workload A bootstrap is not canonically eligible: {item['identity']}: {exc}"
            ) from exc
        record = evaluate_eligible_record(
            runtime,
            identity=item["identity"],
            record_type="bootstrap",
            cell=cell,
            raw_outer_index=raw_outer,
            raw_inner_index=raw_inner,
            sample=sample,
        )
        record["seed_identity"] = seed
    if record.get("identity") != item["identity"]:
        raise HarnessContractError(f"record identity drift: {item['identity']}")
    record["r9_provenance"] = {
        key: value
        for key, value in item.items()
        if key.startswith("r9_") or key in {"cpu_classification", "cuda_classification"}
    }
    return record


def record_counter_delta(state: dict[str, int], record: dict[str, Any]) -> dict[str, int]:
    """Update frozen counters and isolate discrepancies caused by this record."""
    counters = (
        "CLASSIFICATION_MISMATCH", "CUDA_NONCONVERGENCE", "FIT_GATE_FAILURE",
        "DISTRIBUTION_VALUE_GATE_FAILURE", "STATISTIC_GATE_FAILURE",
        "UNEXPLAINED_DISCREPANCY",
    )
    before = {key: state[key] for key in counters}
    update_record_counters(state, record)
    return {key: state[key] - before[key] for key in counters if state[key] > before[key]}


def run_workload_a(
    runtime: FrozenRuntime,
    manifest: dict[str, Any],
    cells: dict[str, Any],
    output: Path,
    state: dict[str, int],
) -> None:
    target = output / "workload_a_records.jsonl"
    for item in manifest["persisted_failure_record_identities"]:
        record = workload_a_record(runtime, cells[item["cell_id"]], item)
        discrepancy = record_counter_delta(state, record)
        append_jsonl(target, record)
        state["WORKLOAD_A_RECORDS_EXECUTED"] += 1
        if discrepancy:
            raise HarnessContractError(f"Workload A record discrepancy: {record['identity']}: {discrepancy}")
    if state["WORKLOAD_A_RECORDS_EXECUTED"] != WORKLOAD_A_EXPECTED:
        raise HarnessContractError("Workload A did not execute exactly 134 records")


def _expected_bootstrap_pairs(outer: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        (item["identity"], item["raw_inner_index"])
        for item in outer["bootstrap_identities"]
    ]


def _reconstructed_bootstrap_rows(outer_identity: str, eligible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "identity": f"{outer_identity}|raw_inner={item['raw_inner_index']}",
            "raw_inner_index": item["raw_inner_index"],
            "seed_identity": item["seed_identity"],
            "sample_digest": item["sample_digest"],
        }
        for item in eligible
    ]


def run_workload_b(
    runtime: FrozenRuntime,
    manifest: dict[str, Any],
    cells: dict[str, Any],
    output: Path,
    state: dict[str, int],
) -> None:
    runner = runtime.runner
    target = output / "workload_b_outers.jsonl"
    for frozen_outer in manifest["mc_failed_outers"]:
        cell = cells[frozen_outer["cell_id"]]
        raw_outer = frozen_outer["raw_outer_index"]
        outer_identity = frozen_outer["outer_identity"]
        observed, observed_meta = runner.fixed_observed(cell, raw_outer, NAMESPACE)
        _, attempts, eligible = runner.fixed_bootstraps(cell, observed, raw_outer, NAMESPACE)
        reconstructed = _reconstructed_bootstrap_rows(outer_identity, eligible)
        expected_pairs = _expected_bootstrap_pairs(frozen_outer)
        reconstructed_pairs = [
            (item["identity"], item["raw_inner_index"]) for item in reconstructed
        ]
        identity_match = expected_pairs == reconstructed_pairs
        base_outer = {
            "outer_identity": outer_identity,
            "cell_id": frozen_outer["cell_id"],
            "raw_outer_index": raw_outer,
            "expected_bootstrap_identities": frozen_outer["bootstrap_identities"],
            "reconstructed_bootstrap_identities": reconstructed,
            "identity_match": identity_match,
            "raw_bootstrap_attempt_count": len(attempts),
            "r9_provenance": {
                key: value for key, value in frozen_outer.items() if key.startswith("r9_")
            },
        }
        if not identity_match:
            state["MC_BOOTSTRAP_IDENTITY_MISMATCH"] += 1
            append_jsonl(target, {**base_outer, "record_level_results": []})
            raise HarnessContractError(
                f"MC_BOOTSTRAP_IDENTITY_MISMATCH: {outer_identity}"
            )

        cpu_classification = runner._cpu_nb_classification(observed)
        cuda_classification = runner._cuda_nb_classification(observed)
        if cpu_classification != "ELIGIBLE" or cuda_classification != "ELIGIBLE":
            result = runner._ineligible_observed_outer(
                cell,
                raw_outer,
                observed,
                observed_meta,
                cpu_classification,
                cuda_classification,
            )
            observed_record = result["records"][0]
            update_record_counters(state, observed_record)
            state["WORKLOAD_B_RECORDS_EXECUTED"] += 1
            append_jsonl(
                target,
                {**base_outer, "record_level_results": [observed_record], **result["outer"]},
            )
            raise HarnessContractError(f"Workload B observed record is not jointly eligible: {outer_identity}")

        observed_record = evaluate_eligible_record(
            runtime,
            identity=outer_identity,
            record_type="observed",
            cell=cell,
            raw_outer_index=raw_outer,
            raw_inner_index=None,
            sample=observed,
        )
        discrepancy = record_counter_delta(state, observed_record)
        state["WORKLOAD_B_RECORDS_EXECUTED"] += 1
        if discrepancy:
            append_jsonl(target, {**base_outer, "record_level_results": [observed_record]})
            raise HarnessContractError(f"Workload B observed discrepancy: {outer_identity}: {discrepancy}")
        bootstrap_records: list[dict[str, Any]] = []
        for eligible_item, expected_item in zip(eligible, frozen_outer["bootstrap_identities"]):
            record = evaluate_eligible_record(
                runtime,
                identity=expected_item["identity"],
                record_type="bootstrap",
                cell=cell,
                raw_outer_index=raw_outer,
                raw_inner_index=eligible_item["raw_inner_index"],
                sample=eligible_item["sample"],
            )
            record["seed_identity"] = eligible_item["seed_identity"]
            bootstrap_records.append(record)
            discrepancy = record_counter_delta(state, record)
            state["WORKLOAD_B_RECORDS_EXECUTED"] += 1
            if discrepancy:
                append_jsonl(
                    target,
                    {**base_outer, "record_level_results": [observed_record, *bootstrap_records]},
                )
                raise HarnessContractError(f"Workload B bootstrap discrepancy: {record['identity']}: {discrepancy}")

        records = [observed_record, *bootstrap_records]
        aggregate = runner.aggregate_outer(observed_record, bootstrap_records)
        state["MC_OUTERS_RECONSTRUCTED"] += 1
        exceedance_match = aggregate["b_cpu"] == aggregate["b_cuda"]
        reject_match = aggregate["reject_cpu"] == aggregate["reject_cuda"]
        append_jsonl(
            target,
            {
                **base_outer,
                "record_level_results": records,
                "b_cpu": aggregate["b_cpu"],
                "b_cuda": aggregate["b_cuda"],
                "p_cpu": aggregate["p_cpu"],
                "p_cuda": aggregate["p_cuda"],
                "reject_cpu": aggregate["reject_cpu"],
                "reject_cuda": aggregate["reject_cuda"],
                "mc_exceedance_match": exceedance_match,
                "mc_reject_match": reject_match,
                "mc_evaluable": aggregate["mc_evaluable"],
                "cuda_mc_unavailable_records": aggregate["cuda_mc_unavailable_records"],
                "outer_gate_pass": aggregate["outer_gate_pass"],
            },
        )
        if not exceedance_match:
            state["MC_EXCEEDANCE_COUNT_MISMATCH"] += 1
        if not reject_match:
            state["MC_REJECT_DECISION_MISMATCH"] += 1
        if not exceedance_match or not reject_match:
            raise HarnessContractError(
                f"Workload B MC discrepancy: {outer_identity}: "
                f"mc_exceedance_match={exceedance_match}, mc_reject_match={reject_match}"
            )


def write_digests(output: Path) -> None:
    digests = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "digests.json"
    }
    write_json(output / "digests.json", digests)


def final_summary(state: dict[str, int], *, failure: bool = False) -> dict[str, Any]:
    gate_pass = not failure and all(state.get(key) == expected for key, expected in REQUIRED_COUNTERS.items())
    return {
        "execution_state": "INCOMPLETE_UNEXPECTED_FAILURE" if failure else "COMPLETE",
        **state,
        "TARGETED_GPU_REPLAY": "PASS" if gate_pass else "FAIL",
        "calibration_claim": False,
        "performance_claim": False,
        "production_claim": False,
        "full_equivalence_claim": False,
    }


def collect_gpu_environment(runtime: FrozenRuntime) -> dict[str, Any]:
    cp = runtime.cuda_candidate.require_cuda()
    device_id = int(cp.cuda.runtime.getDevice())
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name") if isinstance(properties, dict) else None
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return {
        "loaded_module_paths": runtime.loaded_module_paths,
        "cupy_version": str(cp.__version__),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "gpu_device_name": str(name),
        "gpu_device_id": device_id,
    }


def preserve_failure(output: Path, state: dict[str, int], exc: BaseException) -> None:
    failure = {
        "execution_state": "INCOMPLETE_UNEXPECTED_FAILURE",
        "TARGETED_GPU_REPLAY": "FAIL",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "preservation_policy": {
            "PRESERVE_EVIDENCE": True,
            "NO_AUTOMATIC_RERUN": True,
            "NO_DELETE": True,
            "NO_RESUME": True,
            "NO_THRESHOLD_CHANGE": True,
            "NO_SEED_CHANGE": True,
            "NO_RNG_CHANGE": True,
            "NO_RETRY_POLICY_CHANGE": True,
            "NO_FIXTURE_CHANGE": True,
        },
    }
    try:
        write_json(output / "failure.json", failure)
        write_json(output / "summary.json", final_summary(state, failure=True))
        write_digests(output)
    except Exception as preservation_error:
        print(f"evidence preservation encountered an additional error: {preservation_error}", file=sys.stderr)


def execute(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    repo_root = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    harness_path = Path(__file__).resolve()
    if _inside(output, repo_root):
        raise HarnessContractError("output directory must be outside the repository")
    repo_state = preflight_repo(repo_root)
    if output.exists():
        raise HarnessContractError(f"fresh-only output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    state = initial_state()
    start = utc_timestamp()
    harness_digest = sha256_file(harness_path)
    runtime: FrozenRuntime | None = None
    environment: dict[str, Any] = {}
    try:
        shutil.copyfile(harness_path, output / "harness.py")
        shutil.copyfile(MANIFEST_PATH, output / "frozen_identity_manifest.json")
        execution_manifest = {
            "work_item": WORK_ITEM,
            "preregistration_sha": PREREGISTRATION_SHA,
            "replay_execution_sha": REPLAY_EXECUTION_SHA,
            "identity_manifest_sha256": IDENTITY_MANIFEST_SHA256,
            "harness_sha256": harness_digest,
            "namespace": NAMESPACE,
            "R_EQ": R_EQ,
            "B_EQ": B_EQ,
            "workload_a_expected": WORKLOAD_A_EXPECTED,
            "workload_b_outer_expected": WORKLOAD_B_OUTER_EXPECTED,
            "start_timestamp": start,
            "execution_mode": "targeted-r10a-replay",
            "calibration_claim": False,
            "performance_claim": False,
            "production_claim": False,
            "full_equivalence_claim": False,
        }
        write_json(output / "execution_manifest.json", execution_manifest)
        environment = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "repo_absolute_path": str(repo_root),
            "repo_head": repo_state["repo_head"],
            "repo_clean_state": repo_state["repo_clean"],
            "harness_absolute_path": str(harness_path),
            "harness_sha256": harness_digest,
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "loaded_module_paths": {},
            "cupy_version": None,
            "cuda_runtime_version": None,
            "gpu_device_name": None,
        }
        write_json(output / "environment.json", environment)

        runtime = load_frozen_runtime(repo_root)
        environment.update(collect_gpu_environment(runtime))
        write_json(output / "environment.json", environment)
        cells = build_cell_index(runtime, manifest)
        run_workload_a(runtime, manifest, cells, output, state)
        environment["loaded_module_paths"] = verify_loaded_scientific_modules(repo_root)
        write_json(output / "environment.json", environment)
        run_workload_b(runtime, manifest, cells, output, state)
        environment["loaded_module_paths"] = verify_loaded_scientific_modules(repo_root)
        write_json(output / "environment.json", environment)
        summary = final_summary(state)
        write_json(output / "summary.json", summary)
        write_digests(output)
        return 0 if summary["TARGETED_GPU_REPLAY"] == "PASS" else 1
    except BaseException as exc:
        if runtime is not None and environment:
            try:
                environment["loaded_module_paths"] = verify_loaded_scientific_modules(repo_root)
                write_json(output / "environment.json", environment)
            except Exception:
                pass
        preserve_failure(output, state, exc)
        print("PRESERVE_EVIDENCE", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-static", action="store_true", help="validate without project/CUDA imports")
    modes.add_argument("--execute", action="store_true", help="run the future targeted GPU replay")
    parser.add_argument("--require-gpu", action="store_true", help="mandatory explicit GPU execution guard")
    parser.add_argument("--repo", help="exact clean checkout at the frozen replay SHA")
    parser.add_argument("--output", help="new, non-existing evidence directory outside the repository")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.validate_static:
        if args.require_gpu or args.repo is not None or args.output is not None:
            parser.error("--validate-static does not accept --require-gpu, --repo, or --output")
        report = validate_static()
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    if not args.require_gpu or not args.repo or not args.output:
        parser.error("--execute requires --require-gpu --repo <checkout> --output <new-directory>")
    manifest, _ = load_manifest()
    return execute(args, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
