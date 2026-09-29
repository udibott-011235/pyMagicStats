"""No-GPU contract tests for DEC-025. All project/NumPy/SciPy/CuPy imports are blocked.

Synthetic --execute orchestration is tested via monkeypatched boundaries. These
tests never run the scientific runner, query a CUDA runtime, or execute a replay.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock

import pytest

BLOCKED = {"numpy", "scipy", "cupy", "cupyx", "experiments",
           "pyMagicStat", "pyMagicStats", "pymagicstats"}


class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise AssertionError("scientific/GPU import forbidden: " + fullname)
        return None


guard = NoScience()
sys.meta_path.insert(0, guard)
try:
    spec = importlib.util.spec_from_file_location("harness_r4_under_test",
                                                 Path(__file__).with_name("targeted_gpu_replay.py"))
    h = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = h
    spec.loader.exec_module(h)
finally:
    sys.meta_path.remove(guard)


@pytest.fixture(autouse=True)
def no_science():
    blocker = NoScience()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


@pytest.fixture
def manifest():
    return copy.deepcopy(h.load_manifest()[0])


def test_import_and_static_isolation_fresh_process():
    # A fresh interpreter catches imports even if a pytest plugin preloaded a library.
    script = r'''
import importlib.abc, importlib.util, json, sys
blocked = {"numpy","scipy","cupy","cupyx","experiments","pyMagicStat","pyMagicStats","pymagicstats"}
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split(".")[0] in blocked: raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
spec=importlib.util.spec_from_file_location("isolated_r4", sys.argv[1])
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
def forbidden(*args, **kwargs): raise AssertionError("runtime boundary reached")
module.load_frozen_runtime=forbidden
module.gpu_environment=forbidden
module.gpu_smoke=forbidden
module.strong_environment_preflight=forbidden
assert module.main(["--validate-static"]) == 0
assert not any(name.split(".")[0] in blocked for name in sys.modules)
print("STATIC_SCIENTIFIC_IMPORTS=0")
print("STATIC_CUPY_IMPORTS=0")
print("STATIC_GPU_RUNTIME_QUERIES=0")
'''
    result = subprocess.run([sys.executable, "-B", "-I", "-c", script, h.__file__],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "STATIC_GPU_RUNTIME_QUERIES=0" in result.stdout


def test_frozen_bytes_and_static_pass():
    data = h.MANIFEST_PATH.read_bytes()
    assert len(data) == 118681
    assert data.count(b"\n") == 2979 and data.count(b"\r\n") == 0
    assert h.sha256(data) == h.IDENTITY_MANIFEST_SHA256
    report = h.validate_static()
    assert report["STATIC_VALIDATION"] == "PASS"
    assert report["WORKLOAD_A_OBSERVED"] == 10
    assert report["WORKLOAD_A_BOOTSTRAP"] == 124


def test_one_byte_manifest_mutation_fails(tmp_path, monkeypatch):
    path = tmp_path / "mutated.json"
    data = h.MANIFEST_PATH.read_bytes()
    path.write_bytes(data.replace(b"DEC-025", b"DEC-026", 1))
    monkeypatch.setattr(h, "MANIFEST_PATH", path)
    with pytest.raises(h.HarnessContractError, match="SHA256"):
        h.load_manifest()


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_nonfinite(constant):
    with pytest.raises(h.HarnessContractError):
        h.strict_json(('{"x":' + constant + '}').encode())


def test_strict_json_rejects_duplicate_nested_key():
    with pytest.raises(h.HarnessContractError, match="duplicate"):
        h.strict_json(b'{"x":{"a":1,"a":2}}')


@pytest.mark.parametrize("key,value", [
    ("schema_version", "v1"), ("decision", "DEC-019"), ("replay_version", "R1"),
    ("replay_execution_sha", "bad"), ("replay_execution_tree", "bad"),
    ("r9_execution_sha", "bad"), ("namespace", "OTHER"), ("R_EQ", 9),
    ("R_EQ", True), ("B_EQ", 14), ("selection_semantics", "new discovery"),
])
def test_manifest_metadata_fails(manifest, key, value):
    manifest[key] = value
    with pytest.raises(h.HarnessContractError, match="metadata"):
        h.validate_manifest(manifest)


@pytest.mark.parametrize("mutation,match", [
    ("a_duplicate", "duplicate Workload A"), ("a_observed_inner", "observed raw_inner"),
    ("a_bootstrap_null", "bootstrap raw_inner"), ("a_identity", "identity recomposition"),
    ("a_family", "family"), ("a_count", "Workload A count"),
    ("b_duplicate", "duplicate Workload B outer"), ("b_boot_duplicate", "duplicate Workload B bootstrap"),
    ("b_boot_count", "bootstrap count"), ("b_observed", "observed identity"),
    ("b_order", "ordered payload"), ("b_gaps", "ordered payload"),
    ("a_order", "ordered payload"), ("historical_flag", "ordered payload"),
])
def test_manifest_mutations(manifest, mutation, match):
    a, b = manifest["persisted_failure_record_identities"], manifest["mc_failed_outers"]
    if mutation == "a_duplicate": a[1] = copy.deepcopy(a[0])
    elif mutation == "a_observed_inner": next(r for r in a if r["record_type"] == "observed")["raw_inner_index"] = 0
    elif mutation == "a_bootstrap_null": a[0]["raw_inner_index"] = None
    elif mutation == "a_identity": a[0]["identity"] += "bad"
    elif mutation == "a_family": a[0]["family"] = "gamma"
    elif mutation == "a_count": a.pop()
    elif mutation == "b_duplicate": b[1] = copy.deepcopy(b[0])
    elif mutation == "b_boot_duplicate": b[0]["bootstrap_identities"][1] = copy.deepcopy(b[0]["bootstrap_identities"][0])
    elif mutation == "b_boot_count": b[0]["bootstrap_identities"].pop()
    elif mutation == "b_observed": b[0]["observed_identity"] += "bad"
    elif mutation == "b_order": b[0]["bootstrap_identities"].reverse()
    elif mutation == "a_order": a.reverse()
    elif mutation == "historical_flag": a[0]["r9_fit_gate_pass"] = True
    elif mutation == "b_gaps":
        for i, item in enumerate(b[1]["bootstrap_identities"]):
            item.update(raw_inner_index=i, identity=b[1]["outer_identity"] + f"|raw_inner={i}")
    with pytest.raises(h.HarnessContractError, match=match):
        h.validate_manifest(manifest)


def test_wrong_ordered_payload_digest(manifest, monkeypatch):
    monkeypatch.setattr(h, "ORDERED_PAYLOAD_SHA256", "0" * 64)
    with pytest.raises(h.HarnessContractError, match="ordered payload"):
        h.validate_manifest(manifest)


@pytest.mark.parametrize("fault", ["head", "tree", "dirty", "not_git", "missing"])
def test_repo_preflight_stops_before_import_and_output(tmp_path, monkeypatch, fault):
    repo = tmp_path / "repo"
    if fault != "missing": repo.mkdir()
    def fake_git(root, *args):
        if fault == "not_git": raise h.HarnessContractError("not Git")
        if args == ("rev-parse", "--show-toplevel"): return str(repo)
        if args == ("rev-parse", "HEAD"): return "bad" if fault == "head" else h.REPLAY_EXECUTION_SHA
        if args == ("rev-parse", "HEAD^{tree}"): return "bad" if fault == "tree" else h.REPLAY_EXECUTION_TREE
        if args[0] == "status": return " M scientific.py" if fault == "dirty" else ""
        raise AssertionError(args)
    monkeypatch.setattr(h, "git", fake_git)
    imports, gpu = Mock(side_effect=AssertionError("import")), Mock(side_effect=AssertionError("GPU"))
    monkeypatch.setattr(h, "load_frozen_runtime", imports)
    monkeypatch.setattr(h, "gpu_environment", gpu)
    output = tmp_path / "out"
    assert h.execute(SimpleNamespace(repo=str(repo), output=str(output))) == 1
    assert not output.exists()
    imports.assert_not_called()
    gpu.assert_not_called()


@pytest.mark.parametrize("root", ["experiments", "pyMagicStat"])
def test_preloaded_module_outside_repo_fails(tmp_path, monkeypatch, root):
    repo = tmp_path / "repo"
    repo.mkdir()
    module = ModuleType(root + ".foreign")
    module.__file__ = str(tmp_path / "site-packages/foreign.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(h.HarnessContractError, match="outside checkout"):
        h.verify_loaded_scientific_modules(repo)


def test_namespace_package_outside_repo_fails(tmp_path, monkeypatch):
    module = ModuleType("experiments.foreign")
    module.__path__ = [str(tmp_path / "another-clone")]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(h.HarnessContractError, match="outside checkout"):
        h.verify_loaded_scientific_modules(tmp_path / "repo")


def test_import_guard_rejects_foreign_spec_before_execution(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("experiments.foreign", tmp_path / "other/foreign.py")
    monkeypatch.setattr(h.importlib.machinery.PathFinder, "find_spec", Mock(return_value=spec))
    with pytest.raises(h.HarnessContractError, match="shadow outside"):
        h.CheckoutImportGuard(tmp_path / "repo").find_spec("experiments.foreign")


class Sample(list):
    def tobytes(self):
        return b"".join(struct.pack("<d", float(x)) for x in self)


def passing_record(item):
    return {
        **{k: item[k] for k in ("identity", "cell_id", "record_type", "raw_outer_index", "raw_inner_index")},
        "family": "negative_binomial", "sample_digest": h.sha256(Sample([0, 1]).tobytes()),
        "sample_max": 1, "evaluation_points": [0, 1],
        "cpu_evaluation_points": [0, 1], "cuda_evaluation_points": [0, 1],
        "cpu_classification": "ELIGIBLE", "cuda_classification": "ELIGIBLE",
        "cpu_parameters": {"r": 1.0, "p": 0.5}, "cuda_parameters": {"r": 1.0, "p": 0.5},
        "cpu_log_likelihood": -2.0, "cuda_log_likelihood": -2.0,
        "cpu_statistic": 1.0, "cuda_statistic": 1.0,
        "statistic_abs_error": 0.0, "statistic_allowed_tolerance": 2e-11,
        **{key: True for key in h.GATES},
        "distribution_evidence": [
            {"quantity": q, "evaluation_point": p, "cpu_value": 0.5, "cuda_value": 0.5,
             "abs_error": 0.0, "allowed_tolerance": 5e-11, "passed": True}
            for q in h.QUANTITIES for p in [0, 1]],
        "cuda_failure_reason": None, "flat_objective_used": False, "flat_objective_diagnostic": None,
        "certified_support_stop": 9, "statistic_support_size": 10,
    }


def synthetic_aggregate(observed, boots):
    result = {}
    for engine in ("cpu", "cuda"):
        b = sum(r[engine + "_statistic"] >= observed[engine + "_statistic"] for r in boots)
        p = (b + 1) / 16
        result.update({f"b_{engine}": b, f"p_{engine}": p, f"reject_{engine}": p <= 0.05})
    equal = result['b_cpu'] == result['b_cuda'] and result['reject_cpu'] == result['reject_cuda']
    result.update(mc_evaluable=True, mc_gate_pass=equal, outer_gate_pass=equal,
                  cuda_mc_unavailable_records=[])
    return result


@pytest.fixture
def synthetic(tmp_path, monkeypatch, manifest):
    repo = tmp_path / "scientific"
    repo.mkdir()
    output = tmp_path / "evidence"
    cells = {r["cell_id"]: SimpleNamespace(canonical_id=r["cell_id"])
             for r in manifest["persisted_failure_record_identities"] + manifest["mc_failed_outers"]}
    sample = Sample([0, 1])
    meta = {"seed_identity": 123, "sample_digest_sha256": h.sha256(sample.tobytes())}
    runner = SimpleNamespace(fixed_observed=Mock(return_value=(sample, meta)),
                             aggregate_outer=Mock(side_effect=synthetic_aggregate))
    def bootstraps(cell, observed, raw_outer, namespace):
        outer = next(o for o in manifest["mc_failed_outers"]
                     if o["cell_id"] == cell.canonical_id and o["raw_outer_index"] == raw_outer)
        rows = [{**r, "sample": sample, "sample_digest": meta["sample_digest_sha256"], "seed_identity": 123}
                for r in outer["bootstrap_identities"]]
        return {}, rows, rows
    runner.fixed_bootstraps = Mock(side_effect=bootstraps)
    runtime = SimpleNamespace(runner=runner, numpy=SimpleNamespace(__version__="synthetic"),
                              scipy=SimpleNamespace(__version__="synthetic"))
    monkeypatch.setattr(h, "preflight_repo", Mock(return_value={
        "scientific_repo_path": str(repo), "scientific_head": h.REPLAY_EXECUTION_SHA,
        "scientific_tree": h.REPLAY_EXECUTION_TREE, "scientific_worktree_clean": True}))
    monkeypatch.setattr(h, "load_frozen_runtime", Mock(return_value=runtime))
    monkeypatch.setattr(h, "verify_loaded_scientific_modules", Mock(return_value={}))
    monkeypatch.setattr(h, "gpu_environment", Mock(return_value={"synthetic": True}))
    monkeypatch.setattr(h, "gpu_smoke", Mock(return_value={"synthetic": True}))
    monkeypatch.setattr(h, "strong_environment_preflight", Mock(return_value={"synthetic": True}))
    monkeypatch.setattr(h, "build_cell_index", Mock(return_value=cells))
    a_eval = Mock(side_effect=lambda runtime, cell, item: passing_record(item))
    b_eval = Mock(side_effect=lambda runtime, cell, item, sample: passing_record(item))
    monkeypatch.setattr(h, "workload_a_record", a_eval)
    monkeypatch.setattr(h, "evaluate_record", b_eval)
    return SimpleNamespace(args=SimpleNamespace(repo=str(repo), output=str(output)),
                           output=output, runtime=runtime, a_eval=a_eval, b_eval=b_eval,
                           manifest=manifest, sample=sample)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assert_digests(output):
    digests = read(output / "digests.json")
    assert set(digests) == {p.name for p in output.iterdir() if p.name != "digests.json"}
    assert all(h.sha256((output / name).read_bytes()) == digest for name, digest in digests.items())


def test_complete_synthetic_success_claims_and_historical_oracle_not_used(synthetic):
    s = synthetic
    assert h.execute(s.args) == 0
    assert s.a_eval.call_count == 134 and s.b_eval.call_count == 192
    summary = read(s.output / "summary.json")
    assert summary["TARGETED_REPLAY_PASS"] == "YES"
    assert summary["MC_OUTERS_RECONSTRUCTED"] == 12
    assert set(h.REQUIRED_COUNTERS) <= summary.keys()
    assert summary["claim"] == ("CPU/CUDA targeted equivalence passed on the frozen R10-A workload, "
                                "with any raw MC count differences limited to certified canonical "
                                "CPU-reference exact-tie crossings and with identical reject decisions.")
    for forbidden in ("full GPU equivalence", "full 1152 equivalence", "Type-I calibration",
                      "power", "performance qualification", "production readiness", "CP05-D readiness"):
        assert forbidden not in json.dumps(summary)
    assert not (s.output / "failure.json").exists()
    results = read(s.output / "mc_results.json")
    assert all(r["b_cpu"] == r["b_cuda"] == 15 and r["p_cpu"] == 1.0 for r in results)
    assert any(r["r9_provenance"]["r9_b_cpu"] != r["b_cpu"] for r in results)
    assert len(rows(s.output / "workload_a_records.jsonl")) == 134
    assert len(rows(s.output / "workload_b_records.jsonl")) == 192
    assert_digests(s.output)


def test_existing_output_never_overwritten_and_no_resume(synthetic):
    s = synthetic
    s.output.mkdir()
    marker = s.output / "marker"
    marker.write_bytes(b"preserve")
    assert h.execute(s.args) == 1
    assert marker.read_bytes() == b"preserve"
    assert list(s.output.iterdir()) == [marker]
    h.load_frozen_runtime.assert_not_called()
    h.gpu_environment.assert_not_called()


@pytest.mark.parametrize("flag", ["--resume", "--rerun", "--continue", "--full-campaign",
                                 "--ignore-failure", "--force", "--exec"])
def test_forbidden_modes(flag):
    with pytest.raises(SystemExit) as error:
        h.main([flag])
    assert error.value.code != 0


@pytest.mark.parametrize("reason", h.STRUCTURAL_REASONS)
def test_structural_reason_item_and_record_views_force_failure(synthetic, reason):
    s = synthetic
    bad = passing_record(s.manifest["persisted_failure_record_identities"][0])
    structural = {"quantity": "cdf", "evaluation_point": None, "cpu_value": None,
                  "cuda_value": None, "abs_error": float("inf"), "allowed_tolerance": None,
                  "passed": False, "failure_reason": reason}
    bad["distribution_evidence"] += [structural, dict(structural)]
    # Aggregate booleans maliciously remain true.
    s.a_eval.side_effect = lambda *args: copy.deepcopy(bad)
    assert h.execute(s.args) == 1
    assert s.a_eval.call_count == 1 and s.b_eval.call_count == 0
    summary = read(s.output / "summary.json")
    assert summary["TARGETED_REPLAY_PASS"] == "NO"
    assert summary["STRUCTURAL_REASON_ITEM_COUNTS"][reason] == 2
    assert summary["STRUCTURAL_REASON_RECORD_COUNTS"][reason] == 1
    assert summary["UNEXPLAINED_DISCREPANCY"] == 0
    failure = read(s.output / "failure.json")
    assert failure["identity"] == bad["identity"] and failure["phase"] == "workload_a"
    assert failure["structural_failure_reasons"] == [reason]
    assert len(rows(s.output / "workload_a_records.jsonl")) == 1
    assert_digests(s.output)


@pytest.mark.parametrize("quantity", h.QUANTITIES)
def test_missing_quantity_fails_even_if_all_gates_true(synthetic, quantity):
    s = synthetic
    bad = passing_record(s.manifest["persisted_failure_record_identities"][0])
    bad["distribution_evidence"] = [e for e in bad["distribution_evidence"] if e["quantity"] != quantity]
    s.a_eval.side_effect = lambda *args: copy.deepcopy(bad)
    assert h.execute(s.args) == 1
    summary = read(s.output / "summary.json")
    assert summary["UNEXPLAINED_DISCREPANCY"] > 0 and summary["TARGETED_REPLAY_PASS"] == "NO"
    assert s.a_eval.call_count == 1 and s.b_eval.call_count == 0


@pytest.mark.parametrize("fault", ["extra", "duplicate", "point", "length", "empty", "cpu_grid", "sample_max"])
def test_numeric_evidence_schema_fails(synthetic, fault):
    s = synthetic
    bad = passing_record(s.manifest["persisted_failure_record_identities"][0])
    ev = bad["distribution_evidence"]
    if fault == "extra": ev.append({**ev[0], "quantity": "surprise"})
    if fault == "duplicate": ev[-1] = dict(ev[0])
    if fault == "point": ev[0]["evaluation_point"] = 9
    if fault == "length": ev.pop()
    if fault == "empty": ev.clear()
    if fault == "cpu_grid": bad["cpu_evaluation_points"] = [0, 2]
    if fault == "sample_max": bad["sample_max"] = 9
    s.a_eval.side_effect = lambda *args: copy.deepcopy(bad)
    assert h.execute(s.args) == 1
    assert read(s.output / "summary.json")["UNEXPLAINED_DISCREPANCY"] > 0


@pytest.mark.parametrize("gate", list(h.GATES))
def test_a_gate_fail_fast_persists_before_remaining_work(synthetic, gate):
    s = synthetic
    bad = passing_record(s.manifest["persisted_failure_record_identities"][0])
    bad[gate] = False
    s.a_eval.side_effect = lambda *args: bad
    assert h.execute(s.args) == 1
    assert s.a_eval.call_count == 1 and s.b_eval.call_count == 0
    assert s.runtime.runner.fixed_bootstraps.call_count == 0
    assert read(s.output / "summary.json")[h.GATES[gate]] == 1
    assert read(s.output / "summary.json")["UNEXPLAINED_DISCREPANCY"] == 0
    failure = read(s.output / "failure.json")
    assert gate in failure["failed_gates"]
    assert all(key in failure for key in ("phase", "identity", "record_type", "cell_id",
                                          "raw_outer", "raw_inner", "exception"))
    assert rows(s.output / "workload_a_records.jsonl")[0][gate] is False


@pytest.mark.parametrize("position", range(16))
def test_b_first_failed_record_stops_at_each_position(synthetic, position):
    s = synthetic
    def evaluate(runtime, cell, item, sample):
        row = passing_record(item)
        if s.b_eval.call_count == position + 1: row["fit_gate_pass"] = False
        return row
    s.b_eval.side_effect = evaluate
    assert h.execute(s.args) == 1
    assert s.b_eval.call_count == position + 1
    assert s.runtime.runner.fixed_bootstraps.call_count == 1
    s.runtime.runner.aggregate_outer.assert_not_called()
    assert len(rows(s.output / "workload_b_records.jsonl")) == position + 1
    assert read(s.output / "mc_results.json") == []
    assert read(s.output / "failure.json")["phase"] == "workload_b"
    assert_digests(s.output)


@pytest.mark.parametrize("fault", ["identity", "order", "gaps", "count"])
def test_b_identity_sequence_mismatch_precedes_evaluation_and_mc(synthetic, fault):
    s = synthetic
    original = s.runtime.runner.fixed_bootstraps.side_effect
    def boots(*args):
        cpu, attempts, eligible = original(*args)
        eligible = copy.deepcopy(eligible)
        if fault == "order": eligible.reverse()
        elif fault == "count": eligible.pop()
        elif fault == "gaps": eligible[0]["raw_inner_index"] = 99
        else: eligible[0]["raw_inner_index"] = 1
        return cpu, attempts, eligible
    s.runtime.runner.fixed_bootstraps.side_effect = boots
    assert h.execute(s.args) == 1
    assert s.b_eval.call_count == 0
    s.runtime.runner.aggregate_outer.assert_not_called()
    assert read(s.output / "summary.json")["MC_BOOTSTRAP_IDENTITY_MISMATCH"] == 1
    assert read(s.output / "mc_results.json") == []
    assert "reconstructed" in rows(s.output / "workload_b_records.jsonl")[0]


@pytest.mark.parametrize("fault,counter", [
    ("b", "MC_RAW_EXCEEDANCE_COUNT_MISMATCH_OUTERS"), ("reject", "MC_REJECT_DECISION_MISMATCH"),
    ("p", "UNEXPLAINED_DISCREPANCY"),
])
def test_mc_mismatch_fails_and_preserves_partial_results(synthetic, fault, counter):
    s = synthetic
    def aggregate(obs, boots):
        result = synthetic_aggregate(obs, boots)
        result[fault + "_cuda"] = {"b": 14, "p": 0.9375, "reject": True}[fault]
        return result
    s.runtime.runner.aggregate_outer.side_effect = aggregate
    assert h.execute(s.args) == 1
    assert s.b_eval.call_count == 16
    assert read(s.output / "summary.json")[counter] > 0
    assert len(read(s.output / "mc_results.json")) == 1
    assert read(s.output / "failure.json")["phase"] == "workload_b_mc"
    assert_digests(s.output)


@pytest.mark.parametrize("boundary,phase", [
    ("load_frozen_runtime", "scientific_imports"), ("verify_loaded_scientific_modules", "module_origins"),
    ("strong_environment_preflight", "strong_environment_preflight"),
])
def test_runtime_boundary_exception_preserves_evidence_and_stops(synthetic, monkeypatch, capsys, boundary, phase):
    s = synthetic
    monkeypatch.setattr(h, boundary, Mock(side_effect=RuntimeError("synthetic boundary fault")))
    assert h.execute(s.args) == 1
    assert s.a_eval.call_count == s.b_eval.call_count == 0
    assert not s.output.exists()
    failure = json.loads(capsys.readouterr().err.splitlines()[0])
    assert failure["phase"] == phase
    assert failure["EXECUTION_NOT_STARTED"] == "YES"
    assert failure["AUTHORIZATION_NOT_CONSUMED"] == "YES"
    assert failure["OUTPUT_NOT_CREATED"] == "YES"
    assert all(value == 0 for value in failure["scientific_counters"].values())
    if phase in {"scientific_imports", "module_origins"}:
        h.gpu_environment.assert_not_called()


def test_a_reconstruction_uses_frozen_seed_raw_inner_and_same_sample(manifest, monkeypatch):
    item = manifest["persisted_failure_record_identities"][0]
    sample, observed = Sample([0, 1]), Sample([0, 2])
    runner = SimpleNamespace(fixed_observed=Mock(return_value=(observed, {"seed_identity": 11})),
                             reference_fit=Mock(return_value={"parameters": {"r": 2, "p": 0.5}}))
    engine = SimpleNamespace(derive_seed=Mock(return_value=123), _generate=Mock(return_value=sample))
    runtime = SimpleNamespace(runner=runner, engine=engine)
    cell = SimpleNamespace(canonical_id=item["cell_id"], family="negative_binomial", n=20)
    evaluate = Mock(return_value=passing_record(item))
    monkeypatch.setattr(h, "evaluate_record", evaluate)
    row = h.workload_a_record(runtime, cell, item)
    engine.derive_seed.assert_called_once_with(h.NAMESPACE, cell.canonical_id, 2, "inner_bootstrap", 7)
    assert evaluate.call_args.args[-1] is sample
    assert row["seed_identity"] == 123


def test_adapter_uses_same_sample_separate_value_grid_and_full_support(manifest):
    item = manifest["persisted_failure_record_identities"][0]
    sample = Sample([0, 9])
    cell = SimpleNamespace(canonical_id=item["cell_id"], family="negative_binomial", statistic="AD")
    @dataclasses.dataclass
    class Support:
        support_stop: int = 169
        remainder_bound: float = 1e-15
        @property
        def indices(self): return tuple(range(self.support_stop + 1))
    support = Support()
    seen = []
    def cpu(c, x):
        assert x is sample
        seen.append("cpu")
        return {"evaluation_points": tuple(range(10))}
    def cuda(c, x, **kw):
        assert x is sample
        assert kw["certified_support"] == tuple(range(170))
        seen.append("cuda")
        return {"evaluation_points": tuple(range(10))}
    def fixed(**kwargs):
        kwargs["reference_adapter"](cell, kwargs["sample"])
        kwargs["cuda_adapter"](cell, kwargs["sample"])
        return {"sample_digest": h.sha256(sample.tobytes())}
    runner = SimpleNamespace(reference_fit=Mock(return_value={"bound": object()}),
                             evaluate_reference_record=cpu, evaluate_cuda_record=cuda,
                             evaluate_fixed_record=fixed,
                             canonical_distribution_value_points=Mock(return_value=tuple(range(10))))
    runtime = SimpleNamespace(runner=runner, nb_support=SimpleNamespace(certify_nb_support=Mock(return_value=support)))
    row = h.evaluate_record(runtime, cell, item, sample)
    assert seen == ["cpu", "cuda"]
    assert row["sample_max"] == 9
    assert row["evaluation_points"] == list(range(10))
    assert row["certified_support_stop"] == 169 and row["statistic_support_size"] == 170


def test_record_count_view_deduplicates_identity_across_workloads(manifest, tmp_path):
    item = manifest["persisted_failure_record_identities"][0]
    row = passing_record(item)
    reason = h.STRUCTURAL_REASONS[0]
    row["distribution_evidence"] = [{"failure_reason": reason}]
    state = h.initial_state()
    for phase in ("workload_a", "workload_b"):
        h.context(state, phase, item)
        with pytest.raises(h.HarnessContractError):
            h.consume_record(state, tmp_path, phase, item, row)
    assert state["STRUCTURAL_REASON_ITEM_COUNTS"][reason] == 2
    assert state["STRUCTURAL_REASON_RECORD_COUNTS"][reason] == 1


def test_pure_distribution_numeric_failure_has_complete_schema(synthetic):
    s = synthetic
    bad = passing_record(s.manifest["persisted_failure_record_identities"][0])
    bad["distribution_evidence"][0].update(cuda_value=0.6, abs_error=0.1, passed=False)
    bad["distribution_value_gate_pass"] = False
    assert h.numeric_evidence_schema_complete(bad)
    s.a_eval.side_effect = lambda *args: copy.deepcopy(bad)
    assert h.execute(s.args) == 1
    summary = read(s.output / "summary.json")
    assert summary["DISTRIBUTION_VALUE_GATE_FAILURE"] == 1
    assert summary["UNEXPLAINED_DISCREPANCY"] == 0
    assert s.a_eval.call_count == 1 and s.b_eval.call_count == 0
    assert rows(s.output / "workload_a_records.jsonl")[0]["distribution_evidence"][0]["passed"] is False
    assert_digests(s.output)


@pytest.mark.parametrize("gate,counter", [
    ("fit_gate_pass", "FIT_GATE_FAILURE"),
    ("statistic_gate_pass", "STATISTIC_GATE_FAILURE"),
])
def test_pure_fit_or_finite_statistic_failure_is_explained(synthetic, gate, counter):
    s = synthetic
    bad = passing_record(s.manifest["persisted_failure_record_identities"][0])
    bad[gate] = False
    if gate == "fit_gate_pass":
        bad["cuda_parameters"] = {"r": 2.0, "p": 0.5}
    else:
        bad.update(cuda_statistic=2.0, statistic_abs_error=1.0)
    s.a_eval.side_effect = lambda *args: copy.deepcopy(bad)
    assert h.execute(s.args) == 1
    summary = read(s.output / "summary.json")
    assert summary[counter] == 1 and summary["UNEXPLAINED_DISCREPANCY"] == 0
    assert all(summary[key] == int(key == counter) for key in h.GATES.values())


@pytest.mark.parametrize("with_structural_reason", [False, True])
def test_known_cuda_nonconvergence_and_consequent_gates_are_explained(synthetic, with_structural_reason):
    s = synthetic
    bad = passing_record(s.manifest["persisted_failure_record_identities"][0])
    bad.update(cuda_classification="FAILED", cuda_failure_reason="CUDA solver non-convergence",
               cuda_parameters={}, cuda_log_likelihood=None, cuda_statistic=float("nan"),
               cuda_evaluation_points=[], distribution_evidence=[],
               **{gate: False for gate in h.GATES})
    if with_structural_reason:
        bad["distribution_evidence"] = [{"failure_reason": reason} for reason in (
            "EVALUATION_POINT_IDENTITY_MISMATCH", "MISSING_CUDA_DISTRIBUTION_QUANTITY")]
    s.a_eval.side_effect = lambda *args: copy.deepcopy(bad)
    assert h.execute(s.args) == 1
    summary = read(s.output / "summary.json")
    assert summary["CUDA_NONCONVERGENCE"] == 1
    assert all(summary[key] == 1 for key in h.GATES.values())
    assert summary["UNEXPLAINED_DISCREPANCY"] == 0
    assert s.a_eval.call_count == 1 and s.b_eval.call_count == 0
    assert_digests(s.output)


def test_classification_only_failure_is_explained(synthetic):
    s = synthetic
    bad = passing_record(s.manifest["persisted_failure_record_identities"][0])
    bad.update(cuda_classification="FAILED", classification_gate_pass=False)
    s.a_eval.side_effect = lambda *args: copy.deepcopy(bad)
    assert h.execute(s.args) == 1
    summary = read(s.output / "summary.json")
    assert summary["CLASSIFICATION_MISMATCH"] == 1
    assert summary["UNEXPLAINED_DISCREPANCY"] == 0


@pytest.mark.parametrize("fault", ["identity", "missing_field", "missing_quantity", "passed_type"])
def test_known_gate_does_not_mask_independent_schema_inconsistency(synthetic, fault):
    s = synthetic
    bad = passing_record(s.manifest["persisted_failure_record_identities"][0])
    bad["fit_gate_pass"] = False
    if fault == "identity": bad["identity"] += "wrong"
    elif fault == "missing_field": del bad["cpu_parameters"]
    elif fault == "missing_quantity": bad["distribution_evidence"] = bad["distribution_evidence"][2:]
    else: bad["distribution_evidence"][0]["passed"] = 1
    s.a_eval.side_effect = lambda *args: copy.deepcopy(bad)
    assert h.execute(s.args) == 1
    summary = read(s.output / "summary.json")
    assert summary["FIT_GATE_FAILURE"] == 1 and summary["UNEXPLAINED_DISCREPANCY"] == 1


def test_failed_numeric_row_with_true_gate_remains_fail_closed(synthetic):
    s = synthetic
    bad = passing_record(s.manifest["persisted_failure_record_identities"][0])
    bad["distribution_evidence"][0]["passed"] = False
    assert h.numeric_evidence_schema_complete(bad)
    s.a_eval.side_effect = lambda *args: copy.deepcopy(bad)
    assert h.execute(s.args) == 1
    summary = read(s.output / "summary.json")
    assert summary["DISTRIBUTION_VALUE_GATE_FAILURE"] == 0
    assert summary["UNEXPLAINED_DISCREPANCY"] == 1


@pytest.mark.parametrize("reason_kind", ["structural", "cuda"])
def test_unknown_failure_reason_remains_unexplained(synthetic, reason_kind):
    s = synthetic
    bad = passing_record(s.manifest["persisted_failure_record_identities"][0])
    if reason_kind == "structural":
        bad["distribution_evidence"] = [{"failure_reason": "UNKNOWN_REASON"}]
    else:
        bad["cuda_failure_reason"] = "unknown CUDA failure"
    s.a_eval.side_effect = lambda *args: copy.deepcopy(bad)
    assert h.execute(s.args) == 1
    assert read(s.output / "summary.json")["UNEXPLAINED_DISCREPANCY"] == 1


def test_mc_coherent_plus_one_values_contradicting_records_fail(synthetic):
    s = synthetic
    def aggregate(obs, boots):
        result = synthetic_aggregate(obs, boots)
        result.update(b_cuda=14, p_cuda=15 / 16, mc_gate_pass=False, outer_gate_pass=False)
        return result
    s.runtime.runner.aggregate_outer.side_effect = aggregate
    assert h.execute(s.args) == 1
    summary = read(s.output / "summary.json")
    assert summary["MC_RAW_EXCEEDANCE_COUNT_MISMATCH_OUTERS"] == 1
    assert summary["UNEXPLAINED_DISCREPANCY"] == 1
    assert s.b_eval.call_count == 16
    result = read(s.output / "mc_results.json")[0]
    assert result["p_cpu"] == (result["b_cpu"] + 1) / 16
    assert result["p_cuda"] == (result["b_cuda"] + 1) / 16
    assert_digests(s.output)


def test_mc_reject_mismatch_also_fails_recomputation(mc_case):
    aggregate = dict(b_cpu=15, b_cuda=15, p_cpu=1.0, p_cuda=1.0,
                     reject_cpu=False, reject_cuda=True,
                     mc_evaluable=True, mc_gate_pass=False, outer_gate_pass=False,
                     cuda_mc_unavailable_records=[])
    state = h.initial_state()
    with pytest.raises(h.HarnessContractError):
        h.check_mc(state, adjudicate_case(mc_case, aggregate))
    assert state["counters"]["MC_REJECT_DECISION_MISMATCH"] == 1
    assert state["counters"]["UNEXPLAINED_DISCREPANCY"] == 1


@pytest.mark.parametrize("fault", ["different_p_equal_b", "same_invalid_p", "different_b_invalid_p",
                                  "aggregate_gate", "not_evaluable", "both_wrong_reject", "reject_type"])
def test_true_mc_inconsistency_counts_once(fault, mc_case):
    aggregate = dict(b_cpu=15, b_cuda=15, p_cpu=1.0, p_cuda=1.0,
                     reject_cpu=False, reject_cuda=False,
                     mc_evaluable=True, mc_gate_pass=True, outer_gate_pass=True,
                     cuda_mc_unavailable_records=[])
    if fault == "different_p_equal_b": aggregate["p_cuda"] = 15 / 16
    elif fault == "same_invalid_p": aggregate.update(p_cpu=15 / 16, p_cuda=15 / 16)
    elif fault == "different_b_invalid_p": aggregate.update(b_cuda=14, p_cuda=1.0)
    elif fault == "aggregate_gate": aggregate["outer_gate_pass"] = False
    elif fault == "not_evaluable": aggregate["mc_evaluable"] = False
    elif fault == "both_wrong_reject": aggregate.update(reject_cpu=True, reject_cuda=True)
    else: aggregate["reject_cuda"] = 0
    state = h.initial_state()
    with pytest.raises(h.HarnessContractError):
        h.check_mc(state, adjudicate_case(mc_case, aggregate))
    assert state["counters"]["UNEXPLAINED_DISCREPANCY"] == 1


# Keep the production boundaries before fixtures monkeypatch them. No GPU module
# is imported: the fake below implements only deterministic smoke operations.
REAL_STRONG_PREFLIGHT = h.strong_environment_preflight
REAL_GPU_ENVIRONMENT = h.gpu_environment
REAL_GPU_SMOKE = h.gpu_smoke


class FakeArray:
    def __init__(self, value):
        self.value = value

    def __mul__(self, other):
        return FakeArray([a * b for a, b in zip(self.value, other.value)])

    def __add__(self, other):
        return FakeArray([a + other for a in self.value])

    def tolist(self):
        return self.value

    def __float__(self):
        return float(self.value)


@pytest.fixture
def fake_gpu(tmp_path, monkeypatch):
    events, caches, sources = [], [], []
    control = SimpleNamespace(fault=None)

    def fail(name):
        events.append(name)
        if control.fault == name:
            raise RuntimeError("mock " + name)

    def synchronize():
        fail("synchronize")
        if events[-2:] == ["launch", "synchronize"] and control.fault == "launch_sync":
            raise RuntimeError("mock asynchronous launch failure")

    def raw_kernel(code, name, *, backend):
        assert backend == "nvrtc" and name == "r3_smoke"
        assert "double*" in code and "2.0 * x[i] + 1.0" in code
        cache = Path(h.os.environ["CUPY_CACHE_DIR"])
        assert cache.is_dir() and not any(cache.iterdir())
        assert h.os.environ["CUPY_CACHE_IN_MEMORY"] == "0"
        caches.append(cache)
        sources.append(code)
        def compile_kernel():
            fail("compile")
        def launch(grid, block, args):
            assert grid == (1,) and block == (3,)
            fail("launch")
            x, y = args
            y.value = [2 * a + 1 for a in x.value]
            if control.fault == "raw_result": y.value[0] = 0.
        kernel = Mock(side_effect=launch)
        kernel.backend = "nvcc" if control.fault == "backend" else backend
        kernel.compile = Mock(side_effect=compile_kernel)
        return kernel

    def sum_array(values):
        is_basic = values.value == [1., 2., 3.]
        fault = "basic_result" if is_basic else "elementwise_result"
        fail("basic" if is_basic else "elementwise")
        return FakeArray(-1. if control.fault == fault else sum(values.value))

    cp = SimpleNamespace(
        __version__="13.6.0-mocked", float64="float64",
        asarray=lambda x, dtype: FakeArray(x),
        asnumpy=lambda x: x, sum=sum_array,
        empty_like=lambda x: FakeArray([float("nan")] * len(x.value)),
        RawKernel=Mock(side_effect=raw_kernel),
        cuda=SimpleNamespace(runtime=SimpleNamespace(
            is_hip=False, getDeviceCount=lambda: 0 if control.fault == "device_count" else 1,
            getDevice=lambda: 0, getDeviceProperties=lambda _: {"name": b"mock GPU"},
            runtimeGetVersion=lambda: 13000, driverGetVersion=lambda: 13000,
            deviceSynchronize=Mock(side_effect=synchronize)),
            nvrtc=SimpleNamespace(getVersion=lambda: (12, 0) if control.fault == "cupy_nvrtc" else (13, 0))))
    runtime = SimpleNamespace(cuda_candidate=SimpleNamespace(require_cuda=Mock(return_value=cp)))

    def version(major, minor):
        major._obj.value = 12 if control.fault == "nvrtc_major" else 13
        minor._obj.value = 1 if control.fault == "nvrtc_minor" else 0
        return 1 if control.fault == "nvrtc_query" else 0
    library = SimpleNamespace(nvrtcVersion=Mock(side_effect=version))
    def load_library(name):
        fail(name)
        return library
    monkeypatch.setattr(h.ctypes, "CDLL", Mock(side_effect=load_library))
    def special(name, value):
        def operation(*args):
            fail(name)
            if control.fault == name + "_wrong": return FakeArray(0.)
            if control.fault == name + "_nan": return FakeArray(float("nan"))
            return FakeArray(value)
        return operation
    special_module = SimpleNamespace(gammaln=special("gammaln", h.math.log(24.)),
                                      betainc=special("betainc", 0.6875))
    original_import = h.importlib.import_module
    def import_module(name, *args, **kwargs):
        if name == "cupyx.scipy.special":
            fail("special_import")
            return special_module
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(h.importlib, "import_module", import_module)
    monkeypatch.setattr(h, "gpu_environment", REAL_GPU_ENVIRONMENT)
    monkeypatch.setattr(h, "gpu_smoke", REAL_GPU_SMOKE)
    monkeypatch.setattr(h.tempfile, "gettempdir", lambda: str(tmp_path))
    # No real loader paths or Owner environment are used by these mocked tests.
    for name in ("CUDA_WHEEL_ROOT", "CUDA_PATH", "LD_LIBRARY_PATH"):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(runtime=runtime, cp=cp, control=control, events=events,
                           caches=caches, sources=sources, root=tmp_path)


PREFLIGHT_CHECKS = {
    "CUDA_DEVICE_COUNT", "FLOAT64_BASIC_SMOKE", "LIBCUDART_SO_13_LOAD",
    "LIBNVRTC_SO_13_LOAD", "NVRTC_VERSION", "RAWKERNEL_NVRTC_COMPILE",
    "RAWKERNEL_NVRTC_LAUNCH", "RAWKERNEL_NUMERICAL_RESULT",
    "CUPY_ELEMENTWISE_REDUCTION", "CUPYX_SCIPY_GAMMALN", "CUPYX_SCIPY_BETAINC",
}


def test_strong_preflight_all_checks_versions_and_fresh_nvrtc_cache(fake_gpu, monkeypatch):
    g = fake_gpu
    monkeypatch.setenv("CUPY_CACHE_DIR", "existing-owner-cache")
    monkeypatch.setenv("CUPY_CACHE_IN_MEMORY", "1")
    monkeypatch.setenv("CUDA_WHEEL_ROOT", "/custom/not-hardcoded/cu13")
    monkeypatch.setenv("CUDA_PATH", "/custom/toolkit")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/custom/libraries")
    for _ in range(2):
        report = {}
        REAL_STRONG_PREFLIGHT(g.runtime, report, g.root / "output", g.root / "repo")
        assert report["checks"] == dict.fromkeys(PREFLIGHT_CHECKS, "PASS")
        assert report["nvrtc_version"] == report["cupy_nvrtc_version"] == (13, 0)
        assert report["cuda_runtime_version"] == report["cuda_driver_version"] == 13000
        assert report["gpu_name"] == "mock GPU" and report["gpu_count"] == 1
        assert report["CUDA_WHEEL_ROOT"] == "/custom/not-hardcoded/cu13"
        assert report["CUDA_PATH"] == "/custom/toolkit"
        assert report["LD_LIBRARY_PATH"] == "/custom/libraries"
        assert report["rawkernel_cache"]["removed"] is True
        assert report["rawkernel_cache"]["fresh"] is True
        assert h.os.environ["CUPY_CACHE_DIR"] == "existing-owner-cache"
        assert h.os.environ["CUPY_CACHE_IN_MEMORY"] == "1"
    assert len(set(g.sources)) == len(set(g.caches)) == 2
    assert all(not p.exists() for p in g.caches)
    assert g.events.index("compile") < g.events.index("launch")
    assert g.events[g.events.index("launch") + 1] == "synchronize"
    assert not (g.root / "output").exists()

@pytest.fixture
def mc_case(manifest):
    outer = manifest["mc_failed_outers"][0]
    observed = dict(identity=outer["observed_identity"], cell_id=outer["cell_id"],
                    record_type="observed", raw_outer_index=outer["raw_outer_index"], raw_inner_index=None)
    items = [observed] + [{**observed, **boot, "record_type": "bootstrap"}
                          for boot in outer["bootstrap_identities"]]
    records = [dict(passing_record(item), seed_identity=1000 + i) for i, item in enumerate(items)]
    expected = [dict(expected_identity=r["identity"], expected_seed_identity=r["seed_identity"],
                     expected_sample_digest=r["sample_digest"]) for r in records]
    return SimpleNamespace(records=records, expected=expected, outer=outer)


def adjudicate_case(case, aggregate=None):
    if aggregate is None:
        aggregate = synthetic_aggregate(case.records[0], case.records[1:])
    return h.adjudicate_mc(case.records, case.expected, case.outer, aggregate)


def crossing(case, index=1):
    case.records[index]["cuda_statistic"] = h.math.nextafter(1.0, 0.0)


def test_exact_tie_literal_and_immutable_raw_aggregate(mc_case):
    crossing(mc_case)
    raw = synthetic_aggregate(mc_case.records[0], mc_case.records[1:])
    before = copy.deepcopy((mc_case.records, mc_case.expected, mc_case.outer, raw))
    result = adjudicate_case(mc_case, raw)
    assert result["MC_EQUIVALENCE_ADJUDICATED"] is True
    assert (mc_case.records, mc_case.expected, mc_case.outer, raw) == before
    assert raw["mc_gate_pass"] is raw["outer_gate_pass"] is False
    assert result["raw_b_cpu"] == 15 and result["raw_b_cuda"] == 14
    assert result["raw_p_cpu"] == 1 and result["raw_p_cuda"] == 15 / 16
    row = result["bootstrap_adjudication"][0]
    assert row["cpu_reference_exact_tie"] and row["certified_reference_exact_tie_crossing"]
    assert row["cpu_exceedance"] is True and row["cuda_exceedance"] is False
    assert row["T_cpu_boot"] == row["T_cpu_obs"] == 1
    assert not any(key in result for key in (
        "corrected_b_cuda", "corrected_p_cuda", "adjusted_p_cuda", "canonicalized_cuda_statistic"))


@pytest.mark.parametrize("method,direction", [
    ("nextafter", -1), ("nextafter", 1), ("ulp", -1), ("ulp", 1),
    ("within_statistic_tolerance", 1), ("isclose", -1)])
def test_unequal_cpu_reference_never_certified(mc_case, method, direction):
    if method == "nextafter":
        value = h.math.nextafter(1.0, 0.0 if direction < 0 else 2.0)
    elif method == "ulp":
        value = 1.0 + direction * h.math.ulp(1.0)
    else:
        value = 1.0 + direction * 1e-12
        assert h.math.isclose(value, 1.0)
        assert abs(value - 1.0) < mc_case.records[1]["statistic_allowed_tolerance"]
    mc_case.records[1].update(cpu_statistic=value,
                              cuda_statistic=1.0 if direction < 0 else h.math.nextafter(1.0, 0.0))
    result = adjudicate_case(mc_case)
    row = result["bootstrap_adjudication"][0]
    assert row["indicator_mismatch"] is True
    assert row["cpu_reference_exact_tie"] is False
    assert row["certified_reference_exact_tie_crossing"] is False
    assert result["MC_UNEXPLAINED_INDICATOR_MISMATCH"] == 1
    assert result["MC_EQUIVALENCE_ADJUDICATED"] is False


@pytest.mark.parametrize("forward,reverse", [(0, 1), (1, 1), (2, 1)])
def test_opposite_cancelling_and_three_net_one_mismatches_fail(mc_case, forward, reverse):
    for i in range(1, forward + 1):
        crossing(mc_case, i)
    for i in range(forward + 1, forward + reverse + 1):
        mc_case.records[i]["cpu_statistic"] = h.math.nextafter(1.0, 0.0)
    result = adjudicate_case(mc_case)
    assert result["indicator_mismatch_count"] == forward + reverse
    assert result["certified_reference_exact_tie_crossing_count"] == forward
    assert result["unexplained_indicator_mismatch_count"] == reverse
    assert result["raw_b_cpu"] - result["raw_b_cuda"] == forward - reverse
    if forward == reverse:
        assert result["MC_RAW_EXCEEDANCE_COUNT_MATCH"] is True
    assert result["MC_EQUIVALENCE_ADJUDICATED"] is False
    assert "SIGNED_ACCOUNTING" in result["failures"]


@pytest.mark.parametrize("index", [0, 1], ids=["observed", "bootstrap"])
@pytest.mark.parametrize("gate", tuple(h.GATES))
@pytest.mark.parametrize("fault", ["missing", "false", "null"])
def test_each_observed_bootstrap_gate_required(mc_case, index, gate, fault):
    crossing(mc_case)
    if fault == "missing":
        del mc_case.records[index][gate]
    else:
        mc_case.records[index][gate] = False if fault == "false" else None
    result = adjudicate_case(mc_case)
    assert result["MC_EQUIVALENCE_ADJUDICATED"] is False
    assert result["bootstrap_adjudication"][0]["certified_reference_exact_tie_crossing"] is False


@pytest.mark.parametrize("index", [0, 1], ids=["observed", "bootstrap"])
@pytest.mark.parametrize("field,value", [
    ("identity", "wrong"), ("seed_identity", 42), ("sample_digest", "0" * 64),
    ("raw_inner_index", -1), ("raw_outer_index", True), ("cell_id", "wrong"),
    ("record_type", "wrong"), ("seed_identity", None), ("sample_digest", None)])
def test_produced_provenance_must_match_reconstructed_and_frozen(mc_case, index, field, value):
    crossing(mc_case)
    mc_case.records[index][field] = value
    result = adjudicate_case(mc_case)
    assert not result["MC_EQUIVALENCE_ADJUDICATED"]
    assert not result["bootstrap_adjudication"][0]["certified_reference_exact_tie_crossing"]


@pytest.mark.parametrize("field", ["expected_identity", "expected_seed_identity", "expected_sample_digest"])
@pytest.mark.parametrize("index", [0, 1])
def test_missing_reconstructed_provenance_is_not_a_match(mc_case, field, index):
    crossing(mc_case)
    del mc_case.expected[index][field]
    assert not adjudicate_case(mc_case)["MC_EQUIVALENCE_ADJUDICATED"]


@pytest.mark.parametrize("reason", (*h.STRUCTURAL_REASONS, "UNKNOWN_REASON"))
@pytest.mark.parametrize("index", [0, 1])
def test_structural_reason_blocks_certification_even_with_true_gates(mc_case, reason, index):
    crossing(mc_case)
    mc_case.records[index]["distribution_evidence"][0]["failure_reason"] = reason
    result = adjudicate_case(mc_case)
    assert not result["MC_EQUIVALENCE_ADJUDICATED"]
    row = result["bootstrap_adjudication"][0]
    assert row["structural_failure_present"] == (reason in h.STRUCTURAL_REASONS)
    assert not row["certified_reference_exact_tie_crossing"]


@pytest.mark.parametrize("reason", ["non-convergence", "unknown", "", 0, False])
@pytest.mark.parametrize("index", [0, 1])
def test_any_non_none_cuda_failure_blocks_certification(mc_case, reason, index):
    crossing(mc_case)
    mc_case.records[index]["cuda_failure_reason"] = reason
    result = adjudicate_case(mc_case)
    assert not result["MC_EQUIVALENCE_ADJUDICATED"]
    assert result["bootstrap_adjudication"][0]["cuda_failure_present"]
    assert not result["bootstrap_adjudication"][0]["certified_reference_exact_tie_crossing"]


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("engine", ["cpu", "cuda"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None])
def test_nonfinite_statistic_cannot_certify(mc_case, index, engine, value):
    crossing(mc_case)
    raw = synthetic_aggregate(mc_case.records[0], mc_case.records[1:])
    mc_case.records[index][engine + "_statistic"] = value
    result = adjudicate_case(mc_case, raw)
    assert not result["MC_EQUIVALENCE_ADJUDICATED"]
    assert not result["bootstrap_adjudication"][0]["statistics_finite"]
    assert not result["bootstrap_adjudication"][0]["certified_reference_exact_tie_crossing"]


@pytest.mark.parametrize("field,value", [
    ("b_cpu", 14), ("b_cuda", 14), ("b_cpu", True), ("b_cuda", 15.0),
    ("p_cpu", 15 / 16), ("p_cuda", 15 / 16), ("p_cuda", float("nan")),
    ("reject_cpu", True), ("reject_cuda", True), ("reject_cuda", 0),
    ("mc_evaluable", False), ("mc_evaluable", 1),
    ("cuda_mc_unavailable_records", ["failure"]), ("cuda_mc_unavailable_records", None),
    ("mc_gate_pass", False), ("outer_gate_pass", False),
    ("mc_gate_pass", None), ("outer_gate_pass", 1)])
def test_raw_aggregate_inconsistency_fails_closed(mc_case, field, value):
    raw = synthetic_aggregate(mc_case.records[0], mc_case.records[1:])
    raw[field] = value
    assert not adjudicate_case(mc_case, raw)["MC_EQUIVALENCE_ADJUDICATED"]


@pytest.mark.parametrize("count", [0, 1, 2, 7, 15])
def test_arbitrary_certified_crossing_counts_not_historical_oracle(mc_case, count):
    for i in range(1, count + 1):
        crossing(mc_case, i)
    raw = synthetic_aggregate(mc_case.records[0], mc_case.records[1:])
    result = adjudicate_case(mc_case, raw)
    assert result["MC_EQUIVALENCE_ADJUDICATED"] is True
    assert result["indicator_mismatch_count"] == result["MC_REFERENCE_EXACT_TIE_CROSSING_COUNT"] == count
    assert result["MC_UNEXPLAINED_INDICATOR_MISMATCH"] == 0
    assert result["raw_b_cpu"] - result["raw_b_cuda"] == count
    assert raw["mc_gate_pass"] is raw["outer_gate_pass"] is (count == 0)
    assert result["raw_reject_cpu"] is result["raw_reject_cuda"] is False
    state = h.initial_state()
    h.check_mc(state, result)
    assert state["counters"]["MC_OUTERS_ADJUDICATED"] == 1
    assert state["counters"]["MC_REFERENCE_EXACT_TIE_CROSSINGS"] == count
    assert state["counters"]["MC_RAW_EXCEEDANCE_COUNT_MISMATCH_OUTERS"] == int(count > 0)


def test_all_fifteen_bootstrap_evidence_fields_are_auditable(mc_case):
    result = adjudicate_case(mc_case)
    required = {
        "identity", "raw_inner_index", "T_cpu_obs", "T_cuda_obs", "T_cpu_boot", "T_cuda_boot",
        "cpu_exceedance", "cuda_exceedance", "indicator_mismatch", "cpu_reference_exact_tie",
        "identity_match", "seed_identity_match", "sample_digest_match", "statistics_finite",
        "structural_failure_present", "cuda_failure_present", "certified_reference_exact_tie_crossing"}
    required |= {prefix + gate for prefix in ("observed_", "bootstrap_") for gate in h.GATES}
    evidence = result["bootstrap_adjudication"]
    assert len(evidence) == 15
    for i, row in enumerate(evidence, 1):
        assert required <= row.keys()
        for prefix, index in (("observed", 0), ("bootstrap", i)):
            provenance = row[prefix + "_provenance"]
            assert all(provenance[k] == v for k, v in mc_case.expected[index].items())
            assert provenance["valid"] is True
        assert row["identity"] == mc_case.outer["bootstrap_identities"][i - 1]["identity"]


@pytest.mark.parametrize("count", [0, 15, 17])
def test_incomplete_or_extra_record_sequence_fails(mc_case, count):
    records = (mc_case.records * 2)[:count]
    with pytest.raises(h.HarnessContractError, match="16-record"):
        h.adjudicate_mc(records, mc_case.expected, mc_case.outer, {})


@pytest.mark.parametrize("fault", ["a_count", "b_count", "reconstructed", "adjudicated",
                                  "zero_counter", "structural_item", "structural_record",
                                  "missing_outer", "failed_outer", "failure",
                                  "negative_diagnostic", "outer_diagnostic_overflow",
                                  "crossing_diagnostic_overflow"])
def test_partial_or_inconsistent_traversal_never_global_pass(mc_case, fault):
    state = h.initial_state()
    state["counters"].update(h.REQUIRED_COUNTERS)
    state.update(WORKLOAD_A_RECORDS_EXECUTED=134, WORKLOAD_B_RECORDS_EXECUTED=192)
    state["mc_results"] = [{"adjudication": adjudicate_case(mc_case)} for _ in range(12)]
    assert h.final_summary(state)["TARGETED_REPLAY_PASS"] == "YES"
    if fault == "a_count": state["WORKLOAD_A_RECORDS_EXECUTED"] -= 1
    elif fault == "b_count": state["WORKLOAD_B_RECORDS_EXECUTED"] -= 1
    elif fault == "reconstructed": state["counters"]["MC_OUTERS_RECONSTRUCTED"] -= 1
    elif fault == "adjudicated": state["counters"]["MC_OUTERS_ADJUDICATED"] -= 1
    elif fault == "zero_counter": state["counters"]["MC_UNEXPLAINED_INDICATOR_MISMATCH"] = 1
    elif fault == "structural_item": state["STRUCTURAL_REASON_ITEM_COUNTS"][h.STRUCTURAL_REASONS[0]] = 1
    elif fault == "structural_record": state["STRUCTURAL_REASON_RECORD_COUNTS"][h.STRUCTURAL_REASONS[0]] = 1
    elif fault == "missing_outer": state["mc_results"].pop()
    elif fault == "failed_outer": state["mc_results"][0]["adjudication"]["MC_EQUIVALENCE_ADJUDICATED"] = False
    elif fault == "negative_diagnostic": state["counters"][h.DIAGNOSTIC_COUNTERS[0]] = -1
    elif fault == "outer_diagnostic_overflow": state["counters"][h.DIAGNOSTIC_COUNTERS[0]] = 13
    elif fault == "crossing_diagnostic_overflow": state["counters"][h.DIAGNOSTIC_COUNTERS[1]] = 181
    assert h.final_summary(state, failure=fault == "failure")["TARGETED_REPLAY_PASS"] == "NO"


def test_synthetic_full_path_persists_certified_raw_failures_and_limitation(synthetic):
    s = synthetic
    def evaluate(runtime, cell, item, sample):
        record = passing_record(item)
        if item["record_type"] == "bootstrap":
            record["cuda_statistic"] = h.math.nextafter(1.0, 0.0)
        return record
    s.b_eval.side_effect = evaluate
    assert h.execute(s.args) == 0
    summary = read(s.output / "summary.json")
    assert summary["TARGETED_REPLAY_PASS"] == "YES"
    assert summary["MC_OUTERS_ADJUDICATED"] == 12
    assert summary["MC_REFERENCE_EXACT_TIE_CROSSINGS"] == 180
    assert summary["MC_RAW_EXCEEDANCE_COUNT_MISMATCH_OUTERS"] == 12
    assert summary["MC_UNEXPLAINED_INDICATOR_MISMATCH"] == 0
    for result in read(s.output / "mc_results.json"):
        raw, adj = result["raw_aggregate"], result["adjudication"]
        assert raw["b_cpu"] == 15 and raw["b_cuda"] == 0
        assert raw["p_cpu"] == 1 and raw["p_cuda"] == 1 / 16
        assert raw["mc_gate_pass"] is raw["outer_gate_pass"] is False
        assert adj["MC_EQUIVALENCE_ADJUDICATED"] is True
        assert all(result[k] == v for k, v in raw.items())
    for payload in (summary, read(s.output / "environment.json")):
        assert payload["B_EQ"] == 15 and payload["ALPHA"] == 0.05 and payload["P_MIN"] == 0.0625
        assert payload["MC_REJECT_BOUNDARY_INFORMATIVE"] == "NO"
        assert payload["MC_REJECT_BOUNDARY_REASON"] == "B_EQ=15 implies p_min=0.0625 > alpha=0.05"
    assert all((b + 1) / 16 > 0.05 for b in range(16))
    assert_digests(s.output)


def test_r4_manifest_r3_arrays_identical_without_r3_acceptance_oracle():
    r4 = h.load_manifest()[0]
    source = h.MANIFEST_PATH.parent.parent / "targeted_replay_r3/frozen_identity_manifest.json"
    r3 = h.strict_json(source.read_bytes())
    for key in ("persisted_failure_record_identities", "mc_failed_outers"):
        assert r4[key] == r3[key]


def test_exact_tie_predicate_has_no_fuzzy_operations():
    import ast
    tree = ast.parse(Path(h.__file__).read_bytes())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "adjudicate_mc")
    tie = next(n.value for n in ast.walk(function) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "cpu_reference_exact_tie" for t in n.targets))
    calls = {ast.unparse(n.func) for n in ast.walk(tie) if isinstance(n, ast.Call)}
    assert calls == {"math.isfinite"}
    assert any(isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.Eq)
               and ast.unparse(n) == "T_cpu_boot == T_cpu_obs" for n in ast.walk(tie))



@pytest.mark.parametrize("fault,check", [
    ("device_count", "CUDA_DEVICE_COUNT"),
    ("basic", "FLOAT64_BASIC_SMOKE"), ("basic_result", "FLOAT64_BASIC_SMOKE"),
    ("libcudart.so.13", "LIBCUDART_SO_13_LOAD"), ("libnvrtc.so.13", "LIBNVRTC_SO_13_LOAD"),
    ("nvrtc_query", "NVRTC_VERSION"), ("nvrtc_major", "NVRTC_VERSION"),
    ("nvrtc_minor", "NVRTC_VERSION"), ("cupy_nvrtc", "NVRTC_VERSION"),
    ("compile", "RAWKERNEL_NVRTC_COMPILE"), ("launch", "RAWKERNEL_NVRTC_LAUNCH"),
    ("launch_sync", "RAWKERNEL_NVRTC_LAUNCH"), ("raw_result", "RAWKERNEL_NUMERICAL_RESULT"),
    ("elementwise", "CUPY_ELEMENTWISE_REDUCTION"),
    ("elementwise_result", "CUPY_ELEMENTWISE_REDUCTION"),
    ("gammaln", "CUPYX_SCIPY_GAMMALN"), ("gammaln_wrong", "CUPYX_SCIPY_GAMMALN"),
    ("gammaln_nan", "CUPYX_SCIPY_GAMMALN"), ("betainc", "CUPYX_SCIPY_BETAINC"),
    ("betainc_wrong", "CUPYX_SCIPY_BETAINC"), ("betainc_nan", "CUPYX_SCIPY_BETAINC"),
])
def test_each_strong_preflight_failure_no_output_no_authorization_no_science(
        synthetic, fake_gpu, monkeypatch, capsys, fault, check):
    s, g = synthetic, fake_gpu
    monkeypatch.setenv("CUPY_CACHE_DIR", "owner-cache-must-be-restored")
    monkeypatch.setenv("CUPY_CACHE_IN_MEMORY", "1")
    s.runtime.cuda_candidate = g.runtime.cuda_candidate
    g.control.fault = fault
    monkeypatch.setattr(h, "strong_environment_preflight", REAL_STRONG_PREFLIGHT)
    state = h.initial_state()
    monkeypatch.setattr(h, "initial_state", lambda: state)
    assert h.execute(s.args) == 1
    assert not s.output.exists()
    s.a_eval.assert_not_called()
    s.b_eval.assert_not_called()
    s.runtime.runner.fixed_bootstraps.assert_not_called()
    h.build_cell_index.assert_not_called()
    assert not state["preflight_passed"] and not state["authorization_consumed"]
    assert all(v == 0 for v in state["counters"].values())
    assert all(v == 0 for key in ("STRUCTURAL_REASON_ITEM_COUNTS", "STRUCTURAL_REASON_RECORD_COUNTS")
               for v in state[key].values())
    failure = json.loads(capsys.readouterr().err.splitlines()[0])
    assert failure["execution_state"] == "PREFLIGHT_FAILURE"
    assert failure["environment_preflight"]["checks"][check] == "FAIL"
    for key in ("EXECUTION_NOT_STARTED", "AUTHORIZATION_NOT_CONSUMED", "OUTPUT_NOT_CREATED"):
        assert failure[key] == "YES"
    for key in ("AUTO_RERUN", "AUTO_RESUME"):
        assert failure[key] == "NO"
    assert all(not p.exists() for p in g.caches)
    assert h.os.environ["CUPY_CACHE_DIR"] == "owner-cache-must-be-restored"
    assert h.os.environ["CUPY_CACHE_IN_MEMORY"] == "1"


@pytest.mark.parametrize("fault", ["backend", "special_import", "synchronize"])
def test_other_readiness_failures_also_stop_before_output(synthetic, fake_gpu, monkeypatch, fault):
    s, g = synthetic, fake_gpu
    s.runtime.cuda_candidate = g.runtime.cuda_candidate
    g.control.fault = fault
    monkeypatch.setattr(h, "strong_environment_preflight", REAL_STRONG_PREFLIGHT)
    assert h.execute(s.args) == 1 and not s.output.exists()
    s.a_eval.assert_not_called()
    s.b_eval.assert_not_called()


def test_output_provenance_before_first_identity_authorization(synthetic, fake_gpu, monkeypatch):
    s, g = synthetic, fake_gpu
    s.runtime.cuda_candidate = g.runtime.cuda_candidate
    def preflight(*args):
        assert not s.output.exists()
        result = REAL_STRONG_PREFLIGHT(*args)
        assert not s.output.exists()
        return result
    monkeypatch.setattr(h, "strong_environment_preflight", preflight)
    cells = h.build_cell_index.return_value
    def build_cells(*args):
        assert s.output.is_dir()
        assert read(s.output / "authorization.json")["AUTHORIZATION_CONSUMED"] == "NO"
        assert read(s.output / "environment.json")["checks"] == dict.fromkeys(PREFLIGHT_CHECKS, "PASS")
        return cells
    monkeypatch.setattr(h, "build_cell_index", build_cells)
    def first_identity(runtime, cell, item):
        assert read(s.output / "authorization.json")["AUTHORIZATION_CONSUMED"] == "YES"
        assert read(s.output / "authorization.json")["EXECUTION_STARTED"] == "YES"
        assert read(s.output / "authorization.json")["identity"] == s.manifest["persisted_failure_record_identities"][0]["identity"]
        return passing_record(item)
    s.a_eval.side_effect = first_identity
    assert h.execute(s.args) == 0
    environment = read(s.output / "environment.json")
    assert environment["decision"] == "DEC-025" and environment["replay_version"] == "R4"
    assert environment["preregistration_sha"] == "d481125e6981d43bbb529c8ed102c034438cbddd"
    assert environment["scientific_head"] == h.REPLAY_EXECUTION_SHA
    assert environment["scientific_tree"] == h.REPLAY_EXECUTION_TREE
    assert all(key in environment for key in ("CUDA_WHEEL_ROOT", "CUDA_PATH", "LD_LIBRARY_PATH",
                                              "cupy_version", "nvrtc_version", "gpu_name"))
    summary = read(s.output / "summary.json")
    assert summary["execution_state"] == "COMPLETE"
    assert summary["AUTHORIZATION_CONSUMED"] == summary["EXECUTION_STARTED"] == "YES"
    assert_digests(s.output)


@pytest.mark.parametrize("fault", ["provenance", "fixture_selection", "authorization_write"])
def test_post_preflight_pre_identity_failure_never_scientific_discrepancy(synthetic, monkeypatch, fault):
    s = synthetic
    original = h.write_json
    def write(path, payload):
        if ((fault == "provenance" and path.name == "environment.json") or
                (fault == "authorization_write" and path.name == "authorization.json"
                 and payload["AUTHORIZATION_CONSUMED"] == "YES")):
            raise OSError("mock persistence failure")
        return original(path, payload)
    monkeypatch.setattr(h, "write_json", write)
    if fault == "fixture_selection":
        monkeypatch.setattr(h, "build_cell_index", Mock(side_effect=RuntimeError("selection failure")))
    assert h.execute(s.args) == 1
    s.a_eval.assert_not_called()
    s.b_eval.assert_not_called()
    summary = read(s.output / "summary.json")
    assert summary["execution_state"] == "PRE_IDENTITY_FAILURE"
    assert summary["AUTHORIZATION_CONSUMED"] == "NO" and summary["UNEXPLAINED_DISCREPANCY"] == 0
    assert read(s.output / "authorization.json")["AUTHORIZATION_CONSUMED"] == "NO"
    assert_digests(s.output)


def test_first_scientific_failure_consumes_once_and_preserves_boundary(synthetic):
    s = synthetic
    s.a_eval.side_effect = RuntimeError("first scientific evaluation failure")
    assert h.execute(s.args) == 1
    assert s.a_eval.call_count == 1 and s.b_eval.call_count == 0
    for name in ("summary.json", "failure.json", "authorization.json"):
        payload = read(s.output / name)
        assert payload["AUTHORIZATION_CONSUMED"] == payload["EXECUTION_STARTED"] == "YES"
        assert payload["AUTO_RERUN"] == payload["AUTO_RESUME"] == "NO"
    assert read(s.output / "summary.json")["execution_state"] == "SCIENTIFIC_FAILURE"
    assert read(s.output / "summary.json")["UNEXPLAINED_DISCREPANCY"] == 1


def test_library_resolution_records_effective_paths_without_environment_repair(monkeypatch):
    environment = {"CUDA_WHEEL_ROOT": "/wheel", "CUDA_PATH": "/toolkit", "LD_LIBRARY_PATH": "/loader"}
    expected = str(Path("/wheel") / "lib" / "libnvrtc.so.13")
    handle = object()
    def load(path):
        if path == expected: return handle
        raise OSError("not found")
    monkeypatch.setattr(h.ctypes, "CDLL", Mock(side_effect=load))
    before = dict(h.os.environ)
    actual, location = h.load_cuda_library("libnvrtc.so.13", environment)
    assert actual is handle and location == expected
    assert dict(h.os.environ) == before
    assert h.ctypes.CDLL.call_args_list[0].args == ("libnvrtc.so.13",)


def test_cache_inside_requested_output_fails_before_creation(fake_gpu):
    g = fake_gpu
    with pytest.raises(h.HarnessContractError, match="cache root"):
        REAL_STRONG_PREFLIGHT(g.runtime, {}, g.root, g.root / "repo")
    assert not g.caches and not g.cp.RawKernel.called


def test_r3_r4_scientific_logic_and_contracts_unchanged():
    import ast
    r3 = Path(__file__).resolve().parent.parent / "targeted_replay_r3/targeted_gpu_replay.py"
    r3_bytes = r3.read_bytes()
    assert h.sha256(r3_bytes) == "5285b3e608d5de1d3b7d2b25d08fdad3bd6cda1e7e0f9f8d2fb0cf5d13ba9604"
    old, new = ast.parse(r3_bytes), ast.parse(Path(h.__file__).read_bytes())
    def functions(tree):
        return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    old_funcs, new_funcs = functions(old), functions(new)
    assert {name for name in old_funcs if ast.dump(old_funcs[name]) != ast.dump(new_funcs[name])} == {
        "load_manifest", "initial_state", "check_mc", "run_workload_b", "final_summary", "execute"}
    assert set(new_funcs) - set(old_funcs) == {
        "finite_statistic", "record_adjudication_checks", "adjudicate_mc"}
    # Every other function (including A, evaluator, seed/retry reconstruction
    # helpers, strong preflight and authorization boundary) is identical.
    # B adds provenance/evidence only: its scientific calls remain identical.
    def scientific_calls(node):
        return [ast.dump(n) for n in ast.walk(node) if isinstance(n, ast.Call)
                and (ast.unparse(n.func).startswith("runtime.")
                     or ast.unparse(n.func) in {"evaluate_record", "consume_record"})]
    assert scientific_calls(old_funcs["run_workload_b"]) == scientific_calls(new_funcs["run_workload_b"])
    def assignments(tree):
        return {ast.unparse(n.targets[0]): ast.dump(n.value) for n in tree.body if isinstance(n, ast.Assign)}
    old_constants, new_constants = assignments(old), assignments(new)
    assert set(new_constants) - set(old_constants) == {"DIAGNOSTIC_COUNTERS", "MC_LIMITATION"}
    assert {key for key in old_constants if old_constants[key] != new_constants[key]} == {
        "WORK_ITEM", "PREREGISTRATION_SHA", "IDENTITY_MANIFEST_SHA256",
        "IDENTITY_MANIFEST_SIZE", "IDENTITY_MANIFEST_LF", "SCHEMA_VERSION", "DECISION", "REPLAY_VERSION",
        "REQUIRED_COUNTERS"}


def test_r3_manifest_blob_copy_and_r2_ordered_payload_unchanged():
    # R3 canonical Git bytes have LF. The working-tree documentary copy can have
    # Windows EOL translation; byte identity against Git is additionally checked
    # during materialization/commit, outside this portable no-Git suite.
    frozen = h.MANIFEST_PATH.read_bytes()
    assert h.sha256(frozen) == "3f300ff5a678e3eac4956fe10514bf7634ec0ea824f0bfef56f4688a86adb286"
    assert (len(frozen), frozen.count(b"\n"), frozen.count(b"\r\n")) == (118681, 2979, 0)
    r3 = h.strict_json(frozen)
    r2 = h.strict_json((h.MANIFEST_PATH.parent.parent / "targeted_replay_r2/frozen_identity_manifest.json").read_bytes())
    for key in ("persisted_failure_record_identities", "mc_failed_outers"):
        assert r3[key] == r2[key]
    assert h.validate_manifest(r3)["ORDERED_PAYLOAD_SHA256"] == "3fe47fef69fc2dfac152dd48eac75841fe25a107b0da6194c06a5d14a7be1395"


def test_failed_output_creation_after_readiness_does_not_consume_authorization(synthetic, monkeypatch, capsys):
    s = synthetic
    original = Path.mkdir
    def mkdir(path, *args, **kwargs):
        if path == s.output:
            raise OSError("output creation forbidden by filesystem")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    assert h.execute(s.args) == 1 and not s.output.exists()
    s.a_eval.assert_not_called()
    s.b_eval.assert_not_called()
    failure = json.loads(capsys.readouterr().err.splitlines()[0])
    assert failure["execution_state"] == "PRE_IDENTITY_FAILURE"
    assert failure["OUTPUT_NOT_CREATED"] == failure["AUTHORIZATION_NOT_CONSUMED"] == "YES"
    assert all(value == 0 for value in failure["scientific_counters"].values())


def test_failed_library_search_is_diagnostic_only(monkeypatch):
    before = dict(h.os.environ)
    monkeypatch.setattr(h.ctypes, "CDLL", Mock(side_effect=OSError("loader unavailable")))
    environment = {"CUDA_WHEEL_ROOT": "/wheel", "CUDA_PATH": "/toolkit", "LD_LIBRARY_PATH": "/loader"}
    with pytest.raises(h.HarnessContractError, match="cannot load libnvrtc.so.13"):
        h.load_cuda_library("libnvrtc.so.13", environment)
    assert dict(h.os.environ) == before
    attempts = [call.args[0] for call in h.ctypes.CDLL.call_args_list]
    assert attempts[0] == "libnvrtc.so.13"
    assert str(Path("/loader") / "libnvrtc.so.13") in attempts
    assert str(Path("/wheel") / "lib" / "libnvrtc.so.13") in attempts
    assert str(Path("/toolkit") / "lib64" / "libnvrtc.so.13") in attempts


def test_nonfresh_preflight_cache_is_rejected(fake_gpu, monkeypatch):
    g = fake_gpu
    cache = g.root / "already-populated-cache"
    cache.mkdir()
    sentinel = cache / "prior.cubin"
    sentinel.write_bytes(b"not acceptable evidence")
    class BadCache:
        def __enter__(self): return str(cache)
        def __exit__(self, *args): return False
    monkeypatch.setattr(h.tempfile, "TemporaryDirectory", lambda **kwargs: BadCache())
    with pytest.raises(h.HarnessContractError, match="cache must be fresh"):
        REAL_STRONG_PREFLIGHT(g.runtime, {}, g.root / "output", g.root / "repo")
    assert sentinel.read_bytes() == b"not acceptable evidence"
    g.cp.RawKernel.assert_not_called()
    assert not (g.root / "output").exists()
