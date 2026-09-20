"""A2 CLI and artifact wiring tests; these are not CUDA equivalence evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.distribution_gof.cuda_calibration import cp05_c2c_equivalence_runner as runner
from experiments.distribution_gof.cuda_calibration.a2_artifacts import NAMES
from experiments.distribution_gof.cuda_calibration.artifact_validation import ArtifactValidationError, validate_bundle
from experiments.distribution_gof.cuda_calibration.execution_checkpoint import CheckpointError, ExecutionCheckpoint
from experiments.distribution_gof.cuda_calibration.equivalence_preregistration import REQUIRED_ARTIFACTS


def _adversarial(passed=True):
    return [{"fixture_name": name, "overall_fixture_pass": passed} for name in NAMES]


def _rng():
    return [{"canonical_cell_id": "cell", "raw_outer_index": 0,
             "purpose": "outer_observed", "raw_inner_index": None}]


def _outer(passed=True):
    return [{"identity": "cell|raw_outer=0", "outer_gate_pass": passed}]


def _publish(tmp_path, *, outer=True, adversarial=True, batch=True, generator=False, mode="equivalence"):
    return runner.publish_execution_bundle(
        tmp_path / "bundle", mode=mode, git_sha=runner.BASELINE_SHA, records=[],
        outer_results=_outer(outer), adversarial=_adversarial(adversarial),
        batch_passed=batch, rng_identities=_rng(), generator_sanity_passed=generator,
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
    monkeypatch.setattr(runner, "_execute_generator_sanity", lambda args, cp: ([{"identity": "g", "passed": True}], True))
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
