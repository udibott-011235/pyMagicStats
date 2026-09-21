"""A2 CLI and artifact wiring tests; these are not CUDA equivalence evidence."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.distribution_gof.cuda_calibration import cp05_c2c_equivalence_runner as runner
from experiments.distribution_gof.cuda_calibration.a2_artifacts import NAMES, load_fixtures, run_adversarial_fixture, to_json_safe
from experiments.distribution_gof.cuda_calibration.artifact_validation import ArtifactValidationError, validate_bundle, validate_summary
from experiments.distribution_gof.cuda_calibration.artifact_writers import write_fixture_manifest
from experiments.distribution_gof.cuda_calibration.execution_checkpoint import CheckpointError, ExecutionCheckpoint
from experiments.distribution_gof.cuda_calibration.equivalence_preregistration import REQUIRED_ARTIFACTS


def _adversarial(passed=True):
    return [{"fixture_name": name, "overall_fixture_pass": passed} for name in NAMES]


def _rng():
    return [{"canonical_cell_id": "cell", "raw_outer_index": 0,
             "purpose": "outer_observed", "raw_inner_index": None}]


def _outer(passed=True, count=runner.PRIMARY_OUTER_TARGET):
    return [{"identity": f"cell|raw_outer={index}", "outer_gate_pass": passed} for index in range(count)]


def _generator(passed=True, count=7):
    return [{"identity": f"generator-{index}", "passed": passed} for index in range(count)]


def _publish(tmp_path, *, outer=True, adversarial=True, batch=True, generator=False, mode="equivalence", generator_count=7):
    return runner.publish_execution_bundle(
        tmp_path / "bundle", mode=mode, git_sha=runner.BASELINE_SHA, records=[],
        outer_results=_outer(outer), adversarial=_adversarial(adversarial),
        batch_passed=batch, rng_identities=_rng(), generator_sanity_passed=generator,
        generator_results=_generator(generator, generator_count) if mode in {"generator-sanity", "all"} else None,
    )


def test_valid_structural_bundle_is_published_and_has_exactly_eleven_artifacts(tmp_path):
    summary = _publish(tmp_path)
    bundle = tmp_path / "bundle"
    assert summary["equivalence_gate_passed"] is True
    assert set(path.name for path in bundle.iterdir()) == set(REQUIRED_ARTIFACTS)
    assert validate_bundle(bundle) is True


def test_scientific_failure_is_a_valid_published_evidence_bundle(tmp_path):
    summary = _publish(tmp_path, adversarial=False)
    assert summary["equivalence_gate_passed"] is False
    assert summary["overall_pass"] is False
    assert validate_bundle(tmp_path / "bundle") is True


def test_invalid_software_bundle_is_not_published(monkeypatch, tmp_path):
    def corrupt(*args, **kwargs):
        raise ArtifactValidationError("simulated parquet corruption")
    monkeypatch.setattr(runner, "write_fit_comparison", corrupt)
    with pytest.raises(ArtifactValidationError):
        _publish(tmp_path)
    assert not (tmp_path / "bundle").exists()


def test_digest_corruption_is_detected(tmp_path):
    _publish(tmp_path)
    target = tmp_path / "bundle" / "summary.json"
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="digest mismatch"):
        validate_bundle(tmp_path / "bundle")


@pytest.mark.parametrize("mode", ["equivalence", "generator-sanity", "all"])
def test_cli_modes_dispatch_only_after_gpu_gate(monkeypatch, tmp_path, mode):
    seen = []
    monkeypatch.setattr(runner, "_official_dispatch", lambda args: seen.append(args.mode) or 0)
    assert runner.main(["--output", str(tmp_path / mode), "--require-gpu", "--namespace", "test", "--mode", mode]) == 0
    assert seen == [mode]


def test_cli_fails_closed_for_missing_gpu_output_unknown_mode_and_calibration(tmp_path):
    with pytest.raises(SystemExit, match="require-gpu"):
        runner.main(["--output", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        runner.main(["--require-gpu"])
    with pytest.raises(SystemExit):
        runner.main(["--output", str(tmp_path / "x"), "--require-gpu", "--namespace", "test", "--mode", "calibration"])


def test_gpu_unavailable_fails_closed_without_publication(tmp_path):
    with pytest.raises(SystemExit, match="CUDA_CANDIDATE_UNIMPLEMENTED"):
        runner.main(["--output", str(tmp_path / "x"), "--require-gpu", "--namespace", "test"])
    assert not (tmp_path / "x").exists()


def test_all_mode_requires_both_gate_results(tmp_path):
    summary = _publish(tmp_path, mode="all", generator=False)
    assert summary["equivalence_gate_passed"] is True
    assert summary["generator_sanity_passed"] is False
    assert summary["overall_pass"] is False


def _contract():
    return {"execution_sha": "a" * 40, "schema": "cp05-c2c-v1", "dec016": "DEC-016",
            "mode": "equivalence", "namespace": "fixture", "R_EQ": 8, "B_EQ": 15,
            "primary_cell_ids": ["cell"], "fixture_digest": "f" * 64, "float_precision": "float64"}


def test_checkpoint_is_atomic_identity_guarded_and_resumable(tmp_path):
    output = tmp_path / "output"
    checkpoint = ExecutionCheckpoint.open(output, _contract(), resume=False)
    checkpoint.store_primary("cell|raw_outer=0", {"identity": "cell|raw_outer=0", "observed_sample_digest": "x", "raw_bootstrap_attempts": [], "eligible_bootstrap_identities": [], "records": [], "outer": {}})
    recovered = ExecutionCheckpoint.open(output, _contract(), resume=True)
    assert recovered.has_primary("cell|raw_outer=0")
    with pytest.raises(CheckpointError, match="duplicate"):
        recovered.store_primary("cell|raw_outer=0", {})
    with pytest.raises(CheckpointError, match="RESUME_IDENTITY_MISMATCH"):
        ExecutionCheckpoint.open(output, dict(_contract(), R_EQ=9), resume=True)
    with pytest.raises(CheckpointError, match="RESUME_IDENTITY_MISMATCH"):
        ExecutionCheckpoint.open(output, dict(_contract(), execution_sha="b" * 40), resume=True)
    with pytest.raises(CheckpointError, match="RESUME_IDENTITY_MISMATCH"):
        ExecutionCheckpoint.open(output, dict(_contract(), fixture_digest="0" * 64), resume=True)
    (output.with_name(output.name + ".checkpoint") / "checkpoint.json").write_text("{", encoding="utf-8")
    with pytest.raises(CheckpointError, match="corrupt"):
        ExecutionCheckpoint.open(output, _contract(), resume=True)


def test_official_dispatch_invokes_real_workloads_and_finalizes_only_when_complete(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(runner, "_require_cuda", lambda: seen.append("cuda"))
    monkeypatch.setattr(runner, "_execution_sha", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_execute_equivalence", lambda args, cp: ([], _outer(), _adversarial(), True, _rng()))
    monkeypatch.setattr(runner, "_execute_generator_sanity", lambda args, cp: (_generator(), True))
    assert runner.main(["--output", str(tmp_path / "bundle"), "--require-gpu", "--namespace", "fixture", "--mode", "all"]) == 0
    assert seen == ["cuda"]
    assert (tmp_path / "bundle" / "summary.json").exists()
    assert not (tmp_path / "bundle.checkpoint").exists()


def test_resume_skips_completed_primary_and_preserves_scientific_failure(monkeypatch, tmp_path):
    output = tmp_path / "resume"
    checkpoint = ExecutionCheckpoint.open(output, _contract(), resume=False)
    cell = runner.primary_fixture_matrix()[0]
    first = f"{cell.canonical_id}|raw_outer=0"
    checkpoint.store_primary(first, {"identity": first, "cell_id": cell.canonical_id, "raw_outer_index": 0,
                                     "observed_sample_digest": "first", "raw_bootstrap_attempts": [],
                                     "eligible_bootstrap_identities": [], "records": [],
                                     "outer": {"identity": first, "outer_gate_pass": False}})
    calls = []
    def fake_outer(cell, index, namespace):
        calls.append((cell.canonical_id, index, namespace))
        identity = f"{cell.canonical_id}|raw_outer={index}"
        return {"identity": identity, "cell_id": cell.canonical_id, "raw_outer_index": index,
                "observed_sample_digest": str(index), "raw_bootstrap_attempts": [{"raw_inner_index": 0}],
                "eligible_bootstrap_identities": [{"raw_inner_index": 0}], "records": [],
                "outer": {"identity": identity, "outer_gate_pass": True}}
    monkeypatch.setattr(runner, "execute_primary_outer", fake_outer)
    monkeypatch.setattr(runner, "primary_fixture_matrix", lambda: (cell,))
    monkeypatch.setattr(runner, "run_adversarial_fixture", lambda name, fixture, digest: {"fixture_name": name, "overall_fixture_pass": False})
    args = type("Args", (), {"namespace": "fixture"})()
    resumed = ExecutionCheckpoint.open(output, _contract(), resume=True)
    _, outer, _, _, _ = runner._execute_equivalence(args, resumed)
    assert calls[0][1] == 1 and all(index != 0 for _, index, _ in calls)
    assert outer[0]["outer_gate_pass"] is False


def test_adversarial_and_generator_resume_skip_completed_identities(monkeypatch, tmp_path):
    output = tmp_path / "resume"
    checkpoint = ExecutionCheckpoint.open(output, _contract(), resume=False)
    checkpoint.store_adversarial(NAMES[0], {"fixture_name": NAMES[0], "overall_fixture_pass": False})
    monkeypatch.setattr(runner, "primary_fixture_matrix", lambda: ())
    seen = []
    monkeypatch.setattr(runner, "run_adversarial_fixture", lambda name, fixture, digest: seen.append(name) or {"fixture_name": name, "overall_fixture_pass": False})
    runner._execute_equivalence(type("Args", (), {"namespace": "fixture"})(), checkpoint)
    assert NAMES[0] not in seen and len(seen) == len(NAMES) - 1
    identity = "exponential|{\"scale\": 1.0}"
    checkpoint.store_generator(identity, {"identity": identity, "passed": True})
    monkeypatch.setattr(runner, "GENERATOR_SANITY_CASES", (("exponential", {"scale": 1.0}),))
    values, passed = runner._execute_generator_sanity(type("Args", (), {"namespace": "fixture"})(), checkpoint)
    assert passed is True and values[0]["identity"] == identity


def test_software_failure_leaves_checkpoint_but_never_publishes_final_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_require_cuda", lambda: None)
    monkeypatch.setattr(runner, "_execution_sha", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_execute_equivalence", lambda args, cp: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(SystemExit, match="crash"):
        runner.main(["--output", str(tmp_path / "bundle"), "--require-gpu", "--namespace", "fixture"])
    assert not (tmp_path / "bundle").exists()
    assert (tmp_path / "bundle.checkpoint" / "checkpoint.json").exists()


def test_runner_has_no_interactive_tty_dependency():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "input(" not in source and "isatty(" not in source


@pytest.mark.parametrize("name", ["nb_very_sparse", "nb_heavy_tail"])
def test_real_adversarial_records_are_checkpoint_and_json_safe(name, tmp_path):
    fixtures, digest = load_fixtures()
    record = run_adversarial_fixture(name, fixtures[name], digest)
    assert json.loads(json.dumps(record)) == record
    checkpoint = ExecutionCheckpoint.open(tmp_path / name, _contract(), resume=False)
    checkpoint.store_adversarial(name, record)
    write_fixture_manifest(tmp_path / f"{name}.json", [], [], [record])
    assert (tmp_path / f"{name}.json").exists()


def test_non_json_safe_adversarial_payload_is_rejected():
    with pytest.raises(Exception, match="non-JSON-safe"):
        to_json_safe({"fixture_name": "x", "bad": object()})


def test_workload_cardinality_and_unique_identity_gates(tmp_path):
    complete = _outer()
    runner.validate_required_workload_complete(mode="equivalence", outer_results=complete, adversarial=_adversarial(), batch_passed=True, generator_results=None)
    for count in (1151, 1153):
        with pytest.raises(runner.C2CError, match="primary"):
            runner.validate_required_workload_complete(mode="equivalence", outer_results=_outer(count=count), adversarial=_adversarial(), batch_passed=True, generator_results=None)
    duplicate = _outer(); duplicate[-1] = dict(duplicate[0])
    with pytest.raises(runner.C2CError, match="duplicate"):
        runner.validate_required_workload_complete(mode="equivalence", outer_results=duplicate, adversarial=_adversarial(), batch_passed=True, generator_results=None)
    with pytest.raises(runner.C2CError, match="primary"):
        runner.publish_execution_bundle(tmp_path / "incomplete", mode="equivalence", git_sha=runner.BASELINE_SHA, records=[], outer_results=_outer(count=1151), adversarial=_adversarial(), batch_passed=True, rng_identities=_rng())
    assert not (tmp_path / "incomplete").exists()


def test_generator_and_mode_specific_completeness_gates():
    runner.validate_required_workload_complete(mode="generator-sanity", outer_results=[], adversarial=_adversarial(), batch_passed=False, generator_results=_generator())
    for count in (6, 8):
        with pytest.raises(runner.C2CError, match="generator"):
            runner.validate_required_workload_complete(mode="generator-sanity", outer_results=[], adversarial=_adversarial(), batch_passed=False, generator_results=_generator(count=count))
    runner.validate_required_workload_complete(mode="all", outer_results=_outer(), adversarial=_adversarial(), batch_passed=True, generator_results=_generator())
    with pytest.raises(runner.C2CError, match="primary"):
        runner.validate_required_workload_complete(mode="all", outer_results=[], adversarial=_adversarial(), batch_passed=True, generator_results=_generator())


def test_summary_expected_count_validation_rejects_wrong_counts():
    generator = {"case_count": 7, "results": _generator()}
    valid = {"execution_mode": "equivalence", "primary_outer_expected": 1152, "primary_outer_observed": 1152,
             "adversarial_fixture_expected": 14, "adversarial_fixture_observed": 14, "calibration_claim": False}
    assert validate_summary(valid, generator=generator) is True
    with pytest.raises(ArtifactValidationError, match="expected count"):
        validate_summary(dict(valid, primary_outer_expected=1151), generator=generator)
    with pytest.raises(ArtifactValidationError, match="generator"):
        validate_summary(dict(valid, execution_mode="generator-sanity", primary_outer_observed=0, adversarial_fixture_observed=0), generator={"case_count": 6, "results": _generator(6)})


def _mc_record(*, cuda_statistic=1.0, cuda_classification="ELIGIBLE", failure_reason=None, inner=None):
    return {"identity": f"row-{inner}", "record_type": "observed" if inner is None else "bootstrap",
            "raw_inner_index": inner, "cuda_classification": cuda_classification,
            "cuda_failure_reason": failure_reason, "cpu_statistic": 1.0,
            "cuda_statistic": cuda_statistic, "classification_gate_pass": cuda_classification == "ELIGIBLE",
            "fit_gate_pass": cuda_classification == "ELIGIBLE", "distribution_value_gate_pass": cuda_classification == "ELIGIBLE",
            "statistic_gate_pass": cuda_classification == "ELIGIBLE"}


def test_observed_cuda_failure_is_scientific_evidence_not_mc_crash():
    observed = _mc_record(cuda_statistic=float("nan"), cuda_classification="FAILED", failure_reason="solver non-convergence")
    aggregate = runner.aggregate_outer(observed, [_mc_record(inner=index) for index in range(15)])
    assert aggregate["b_cpu"] == 15 and aggregate["p_cpu"] == 1.0
    assert aggregate["b_cuda"] is aggregate["p_cuda"] is aggregate["reject_cuda"] is None
    assert aggregate["mc_evaluable"] is False and aggregate["mc_gate_pass"] is False and aggregate["outer_gate_pass"] is False
    assert aggregate["cuda_mc_unavailable_records"][0]["cuda_failure_reason"] == "solver non-convergence"


def test_bootstrap_cuda_failure_preserves_cpu_mc_and_finite_case_semantics():
    observed = _mc_record()
    bootstraps = [_mc_record(inner=index) for index in range(15)]
    bootstraps[4] = _mc_record(inner=4, cuda_statistic=float("nan"), cuda_classification="FAILED", failure_reason="candidate statistic nonfinite")
    aggregate = runner.aggregate_outer(observed, bootstraps)
    assert aggregate["b_cpu"] == 15 and aggregate["p_cpu"] == 1.0
    assert aggregate["b_cuda"] is None and aggregate["mc_evaluable"] is False
    finite = runner.aggregate_outer(_mc_record(), [_mc_record(inner=index) for index in range(15)])
    assert finite["mc_evaluable"] is True and finite["b_cuda"] == 15 and finite["reject_cuda"] is False


def test_failed_gamma_outer_is_checkpointable_without_nan_mc(monkeypatch, tmp_path):
    cell = runner.primary_fixture_matrix()[0]
    def failed_cuda(*args, **kwargs):
        return {"classification": "FAILED", "parameters": {}, "log_likelihood": None,
                "statistic": float("nan"), "evaluation_points": [], "distribution_values": {},
                "solver_converged": False, "iterations": 0, "dtype": "float64",
                "failure_reason": "quantum regression failure"}
    completed = runner.execute_primary_outer(cell, 0, "fixture", cuda_adapter=failed_cuda)
    assert completed["outer"]["outer_gate_pass"] is False
    assert completed["outer"]["mc_evaluable"] is False
    checkpoint = ExecutionCheckpoint.open(tmp_path / "checkpointable", _contract(), resume=False)
    checkpoint.store_primary(completed["identity"], completed)
    assert ExecutionCheckpoint.open(tmp_path / "checkpointable", _contract(), resume=True).has_primary(completed["identity"])


def test_campaign_continues_after_scientific_failure(monkeypatch, tmp_path):
    cell = runner.primary_fixture_matrix()[0]
    checkpoint = ExecutionCheckpoint.open(tmp_path / "resume", _contract(), resume=False)
    seen = []
    def failed_outer(cell, index, namespace):
        seen.append(index); identity = f"{cell.canonical_id}|raw_outer={index}"
        return {"identity": identity, "cell_id": cell.canonical_id, "raw_outer_index": index,
                "observed_sample_digest": str(index), "raw_bootstrap_attempts": [], "eligible_bootstrap_identities": [],
                "records": [], "outer": {"identity": identity, "outer_gate_pass": False, "mc_evaluable": False}}
    monkeypatch.setattr(runner, "primary_fixture_matrix", lambda: (cell,))
    monkeypatch.setattr(runner, "execute_primary_outer", failed_outer)
    monkeypatch.setattr(runner, "run_adversarial_fixture", lambda name, fixture, digest: {"fixture_name": name, "overall_fixture_pass": False})
    runner._execute_equivalence(type("Args", (), {"namespace": "fixture"})(), checkpoint)
    assert seen == list(range(8))


def test_complete_scientific_failure_bundle_is_publishable(tmp_path):
    summary = _publish(tmp_path, outer=False, adversarial=False, batch=False)
    assert summary["equivalence_gate_passed"] is False and summary["overall_pass"] is False
    assert validate_bundle(tmp_path / "bundle") is True


def _nb_cell():
    return SimpleNamespace(family="negative_binomial", statistic="AD", canonical_id="negative_binomial|fixture|n=20|AD|composite", n=20)


def _ineligible_outer(monkeypatch, sample, cpu_classification, cuda_classification):
    cell = _nb_cell()
    meta = {"sample_digest_sha256": "synthetic"}
    monkeypatch.setattr(runner, "fixed_observed", lambda *args: (np.asarray(sample, dtype=np.int64), meta))
    def no_bootstrap(*args, **kwargs):
        raise AssertionError("ineligible observed NB must not bootstrap")
    monkeypatch.setattr(runner, "fixed_bootstraps", no_bootstrap)
    return runner.execute_primary_outer(cell, 0, "CP05-C2C", cpu_nb_classifier=lambda _: cpu_classification,
                                        cuda_nb_classifier=lambda _: cuda_classification)


def test_quantum_nb_variance_case_is_completed_without_fit_or_bootstrap(monkeypatch, tmp_path):
    cell = next(item for item in runner.primary_fixture_matrix() if item.family == "negative_binomial" and dict(item.parameters) == {"r": .25, "p": .5} and item.n == 20 and item.statistic == "AD")
    sample, _ = runner.fixed_observed(cell, 0, "CP05-C2C")
    assert runner._cpu_nb_classification(sample) == "VARIANCE_NOT_GREATER_THAN_MEAN"
    completed = _ineligible_outer(monkeypatch, sample, "VARIANCE_NOT_GREATER_THAN_MEAN", "VARIANCE_NOT_GREATER_THAN_MEAN")
    outer = completed["outer"]
    assert outer["outer_gate_pass"] is True and outer["mc_evaluable"] is False
    assert outer["b_cpu"] is outer["b_cuda"] is None
    checkpoint = ExecutionCheckpoint.open(tmp_path / "nb", _contract(), resume=False)
    checkpoint.store_primary(completed["identity"], completed)
    assert ExecutionCheckpoint.open(tmp_path / "nb", _contract(), resume=True).has_primary(completed["identity"])


def test_all_zero_and_classification_disagreement_are_completed_categorical_outers(monkeypatch):
    matching = _ineligible_outer(monkeypatch, [0] * 20, "ALL_ZERO_NON_IDENTIFYING", "ALL_ZERO_NON_IDENTIFYING")
    assert matching["outer"]["outer_gate_pass"] is True
    assert matching["records"][0]["fit_gate_pass"] is None
    assert matching["records"][0]["distribution_value_gate_pass"] is None
    assert matching["records"][0]["statistic_gate_pass"] is None
    mismatch = _ineligible_outer(monkeypatch, [0, 1] * 10, "VARIANCE_NOT_GREATER_THAN_MEAN", "ALL_ZERO_NON_IDENTIFYING")
    assert mismatch["records"][0]["classification_gate_pass"] is False
    assert mismatch["outer"]["outer_gate_pass"] is False


def test_eligible_nb_path_reaches_existing_bootstrap_pipeline(monkeypatch):
    cell = _nb_cell(); sample = np.asarray([0] * 19 + [100], dtype=np.int64)
    meta = {"sample_digest_sha256": "eligible"}; seen = []
    monkeypatch.setattr(runner, "fixed_observed", lambda *args: (sample, meta))
    monkeypatch.setattr(runner, "reference_fit", lambda *args: {"bound": object()})
    monkeypatch.setattr(runner, "certify_nb_support", lambda *args: SimpleNamespace(indices=[0], remainder_bound=0.0))
    def fake_bootstrap(*args):
        seen.append("bootstrap")
        rows = [{"raw_inner_index": index, "seed_identity": index, "sample_digest": str(index), "sample": sample,
                 "canonical_status": "ELIGIBLE"} for index in range(15)]
        return None, rows, rows
    monkeypatch.setattr(runner, "fixed_bootstraps", fake_bootstrap)
    def reference(candidate_cell, value):
        return {"classification": "ELIGIBLE", "parameters": {"r": 1.0, "p": .5}, "log_likelihood": -1.0,
                "reference_log_likelihood": lambda _: -1.0, "statistic": 1.0, "evaluation_points": [0.0],
                "distribution_values": {"pmf": [.5], "logPMF": [-.7], "cdf": [.5], "sf": [.5], "logCDF": [-.7], "logSF": [-.7]}}
    def cuda_from_reference(candidate_cell, value, **kwargs):
        cpu = reference(candidate_cell, value)
        return {"classification": "ELIGIBLE", "parameters": cpu["parameters"], "log_likelihood": cpu["log_likelihood"], "statistic": cpu["statistic"], "evaluation_points": cpu["evaluation_points"], "distribution_values": cpu["distribution_values"], "solver_converged": True, "iterations": 1, "dtype": "float64", "failure_reason": None}
    completed = runner.execute_primary_outer(cell, 0, "fixture", reference_adapter=reference, cuda_adapter=cuda_from_reference,
                                             cpu_nb_classifier=lambda _: "ELIGIBLE", cuda_nb_classifier=lambda _: "ELIGIBLE")
    assert seen == ["bootstrap"] and len(completed["records"]) == 16


def test_complete_bundle_with_ineligible_outer_identities_is_publishable(tmp_path):
    summary = _publish(tmp_path, outer=True, adversarial=True, batch=True)
    assert summary["equivalence_gate_passed"] is True


FROZEN_QUANTUM_NB_SAMPLE = np.asarray([28, 34, 54, 29, 40, 75, 27, 52, 44, 47,
                                        46, 17, 85, 60, 12, 23, 35, 24, 30, 19], dtype=np.int64)


def test_frozen_quantum_nb_probe_fixture_and_cpu_reference_are_stable():
    assert hashlib.sha256(FROZEN_QUANTUM_NB_SAMPLE.tobytes()).hexdigest() == "36e6c8d854094357aaabc466a8ef14917380d576df9d330cc62e0de25c07d4bb"
    assert np.mean(FROZEN_QUANTUM_NB_SAMPLE) == 39.05
    assert np.var(FROZEN_QUANTUM_NB_SAMPLE, ddof=1) == pytest.approx(366.68157894736834)
    cell = SimpleNamespace(family="negative_binomial", statistic="AD", canonical_id="quantum-nb", n=20)
    reference = runner.evaluate_reference_record(cell, FROZEN_QUANTUM_NB_SAMPLE)
    assert reference["classification"] == "ELIGIBLE"
    assert all(math.isfinite(value) for value in reference["parameters"].values())
    assert math.isfinite(reference["log_likelihood"])


def test_trigamma_order_is_device_native_and_has_no_cpu_fallback():
    source = Path("experiments/distribution_gof/cuda_calibration/cuda_candidate.py").read_text(encoding="utf-8")
    assert "trigamma_order=cp.asarray(1,dtype=cp.int32)" in source
    assert "csp.polygamma(trigamma_order,a+r[...,None])" in source
    assert "csp.polygamma(trigamma_order,shape)" in source
    assert source.count("trigamma_order=cp.asarray(1,dtype=cp.int32)") == 2
    assert "scipy.special.polygamma" not in source
    assert "cp.asnumpy" not in source
