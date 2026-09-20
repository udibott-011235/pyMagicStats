"""A2 CLI and artifact wiring tests; these are not CUDA equivalence evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.distribution_gof.cuda_calibration import cp05_c2c_equivalence_runner as runner
from experiments.distribution_gof.cuda_calibration.a2_artifacts import NAMES
from experiments.distribution_gof.cuda_calibration.artifact_validation import ArtifactValidationError, validate_bundle
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
    assert runner.main(["--output", str(tmp_path / mode), "--require-gpu", "--mode", mode]) == 0
    assert seen == [mode]


def test_cli_fails_closed_for_missing_gpu_output_unknown_mode_and_calibration(tmp_path):
    with pytest.raises(SystemExit, match="require-gpu"):
        runner.main(["--output", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        runner.main(["--require-gpu"])
    with pytest.raises(SystemExit):
        runner.main(["--output", str(tmp_path / "x"), "--require-gpu", "--mode", "calibration"])


def test_gpu_unavailable_fails_closed_without_publication(tmp_path):
    with pytest.raises(SystemExit, match="CUDA_CANDIDATE_UNIMPLEMENTED"):
        runner.main(["--output", str(tmp_path / "x"), "--require-gpu"])
    assert not (tmp_path / "x").exists()


def test_all_mode_requires_both_gate_results(tmp_path):
    summary = _publish(tmp_path, mode="all", generator=False)
    assert summary["equivalence_gate_passed"] is True
    assert summary["generator_sanity_passed"] is False
    assert summary["overall_pass"] is False
