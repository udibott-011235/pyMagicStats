"""No-GPU contract tests for DEC-022. All project/NumPy/SciPy/CuPy imports are blocked.

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
    spec = importlib.util.spec_from_file_location("harness_r2_under_test",
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
spec=importlib.util.spec_from_file_location("isolated_r2", sys.argv[1])
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
def forbidden(*args, **kwargs): raise AssertionError("runtime boundary reached")
module.load_frozen_runtime=forbidden
module.gpu_environment=forbidden
module.gpu_smoke=forbidden
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
    assert len(data) == 117577
    assert data.count(b"\n") == 2963 and data.count(b"\r\n") == 0
    assert h.sha256(data) == h.IDENTITY_MANIFEST_SHA256
    report = h.validate_static()
    assert report["STATIC_VALIDATION"] == "PASS"
    assert report["WORKLOAD_A_OBSERVED"] == 10
    assert report["WORKLOAD_A_BOOTSTRAP"] == 124


def test_one_byte_manifest_mutation_fails(tmp_path, monkeypatch):
    path = tmp_path / "mutated.json"
    data = h.MANIFEST_PATH.read_bytes()
    path.write_bytes(data.replace(b"DEC-022", b"DEC-023", 1))
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
    result.update(mc_evaluable=True, mc_gate_pass=True, outer_gate_pass=True)
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
    assert summary["claim"] == ("The preregistered R10-A targeted GPU replay R2 passed "
                                "for the frozen directed historical workload.")
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
    ("b", "MC_EXCEEDANCE_COUNT_MISMATCH"), ("reject", "MC_REJECT_DECISION_MISMATCH"),
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
    ("gpu_environment", "gpu_query"), ("gpu_smoke", "gpu_smoke"),
])
def test_runtime_boundary_exception_preserves_evidence_and_stops(synthetic, monkeypatch, boundary, phase):
    s = synthetic
    monkeypatch.setattr(h, boundary, Mock(side_effect=RuntimeError("synthetic boundary fault")))
    assert h.execute(s.args) == 1
    assert s.a_eval.call_count == s.b_eval.call_count == 0
    assert read(s.output / "failure.json")["phase"] == phase
    assert read(s.output / "summary.json")["TARGETED_REPLAY_PASS"] == "NO"
    if phase in {"scientific_imports", "module_origins"}:
        h.gpu_environment.assert_not_called()
    assert_digests(s.output)


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


def test_mc_count_mismatch_with_own_plus_one_values_is_explained(synthetic):
    s = synthetic
    def aggregate(obs, boots):
        result = synthetic_aggregate(obs, boots)
        result.update(b_cuda=14, p_cuda=15 / 16, mc_gate_pass=False, outer_gate_pass=False)
        return result
    s.runtime.runner.aggregate_outer.side_effect = aggregate
    assert h.execute(s.args) == 1
    summary = read(s.output / "summary.json")
    assert summary["MC_EXCEEDANCE_COUNT_MISMATCH"] == 1
    assert summary["UNEXPLAINED_DISCREPANCY"] == 0
    assert s.b_eval.call_count == 16
    result = read(s.output / "mc_results.json")[0]
    assert result["p_cpu"] == (result["b_cpu"] + 1) / 16
    assert result["p_cuda"] == (result["b_cuda"] + 1) / 16
    assert_digests(s.output)


def test_mc_reject_mismatch_is_specific_not_generic_unexplained():
    aggregate = dict(b_cpu=15, b_cuda=15, p_cpu=1.0, p_cuda=1.0,
                     reject_cpu=False, reject_cuda=True,
                     mc_evaluable=True, mc_gate_pass=False, outer_gate_pass=False)
    state = h.initial_state()
    with pytest.raises(h.HarnessContractError):
        h.check_mc(state, aggregate)
    assert state["counters"]["MC_REJECT_DECISION_MISMATCH"] == 1
    assert state["counters"]["UNEXPLAINED_DISCREPANCY"] == 0


@pytest.mark.parametrize("fault", ["different_p_equal_b", "same_invalid_p", "different_b_invalid_p",
                                  "aggregate_gate", "not_evaluable", "both_wrong_reject", "reject_type"])
def test_true_mc_inconsistency_counts_once(fault):
    aggregate = dict(b_cpu=15, b_cuda=15, p_cpu=1.0, p_cuda=1.0,
                     reject_cpu=False, reject_cuda=False,
                     mc_evaluable=True, mc_gate_pass=True, outer_gate_pass=True)
    if fault == "different_p_equal_b": aggregate["p_cuda"] = 15 / 16
    elif fault == "same_invalid_p": aggregate.update(p_cpu=15 / 16, p_cuda=15 / 16)
    elif fault == "different_b_invalid_p": aggregate.update(b_cuda=14, p_cuda=1.0)
    elif fault == "aggregate_gate": aggregate["outer_gate_pass"] = False
    elif fault == "not_evaluable": aggregate["mc_evaluable"] = False
    elif fault == "both_wrong_reject": aggregate.update(reject_cpu=True, reject_cuda=True)
    else: aggregate["reject_cuda"] = 0
    state = h.initial_state()
    with pytest.raises(h.HarnessContractError):
        h.check_mc(state, aggregate)
    assert state["counters"]["UNEXPLAINED_DISCREPANCY"] == 1
