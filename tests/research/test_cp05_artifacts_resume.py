from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from experiments.distribution_gof.artifacts import (
    ArtifactIntegrityError,
    read_checkpoint,
    resume_manifest,
    validate_artifact_digests,
    validate_auditable_artifacts,
    write_run_artifacts,
)
from experiments.distribution_gof.cost_preflight import estimate_cost
from experiments.distribution_gof.manifest import canonical_json
from experiments.distribution_gof.runner import RunnerHooks, run_manifest, run_outer_unit
from experiments.distribution_gof.statistics import StatisticEvaluation

from ._helpers import manifest


def test_atomic_artifact_surface_provenance_and_digests(tmp_path):
    configured = manifest(raw_outer_range=(0, 1))
    output = run_manifest(configured)
    digests = write_run_artifacts(
        tmp_path,
        configured,
        output,
        table_format="jsonl",
        command="python  -m   experiments.distribution_gof.runner",
        elapsed_seconds=1.25,
    )
    expected = {
        "manifest.json",
        "environment.json",
        "outer_results.jsonl",
        "inner_accounting.jsonl",
        "cell_summary.json",
        "run_metadata.json",
        "digests.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert validate_artifact_digests(tmp_path) == digests
    validate_auditable_artifacts(tmp_path)
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    environment = json.loads((tmp_path / "environment.json").read_text())
    assert metadata["normalized_command"] == "python -m experiments.distribution_gof.runner"
    assert metadata["manifest_digest"] == configured.digest
    assert metadata["claim_status"] == "NON_CLAIMING"
    assert environment["repository"] == "udibott-011235/pyMagicStats"
    assert environment["source_sha"] == configured.source_sha
    assert environment["numpy_version"] and environment["scipy_version"]
    assert not list(tmp_path.glob(".*.tmp"))


def test_artifact_digest_tamper_fails_closed(tmp_path):
    configured = manifest(raw_outer_range=(0, 1))
    write_run_artifacts(
        tmp_path,
        configured,
        run_manifest(configured),
        table_format="jsonl",
        command="test",
        elapsed_seconds=0.0,
    )
    with (tmp_path / "manifest.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ArtifactIntegrityError, match="manifest.json"):
        validate_artifact_digests(tmp_path)


def test_missing_critical_provenance_fails_even_with_recomputed_file_digest(tmp_path):
    configured = manifest(raw_outer_range=(0, 1))
    write_run_artifacts(
        tmp_path,
        configured,
        run_manifest(configured),
        table_format="jsonl",
        command="test",
        elapsed_seconds=0.0,
    )
    environment_path = tmp_path / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment.pop("python_version")
    environment_path.write_text(json.dumps(environment, sort_keys=True) + "\n")
    digests_path = tmp_path / "digests.json"
    digests = json.loads(digests_path.read_text())
    digests["files"]["environment.json"] = hashlib.sha256(
        environment_path.read_bytes()
    ).hexdigest()
    digests_path.write_text(json.dumps(digests, sort_keys=True) + "\n")
    with pytest.raises(ArtifactIntegrityError, match="provenance fields"):
        validate_auditable_artifacts(tmp_path)


def test_interrupted_resume_is_logically_identical_to_uninterrupted_run(tmp_path):
    configured = manifest(raw_outer_range=(0, 3), statistic="KS")
    uninterrupted = run_manifest(configured)
    checkpoint = tmp_path / "checkpoint.json"
    partial, complete = resume_manifest(configured, checkpoint, max_new_units=1)
    assert not complete
    assert [item.raw_outer_index for item in partial.outer_results] == [0]
    resumed, complete = resume_manifest(
        configured, checkpoint, execution_order=(2, 1)
    )
    assert complete
    assert resumed.logical_results() == uninterrupted.logical_results()
    assert [item.to_dict() for item in resumed.inner_accounting] == [
        item.to_dict() for item in uninterrupted.inner_accounting
    ]
    assert not list(tmp_path.glob(".*.tmp"))


def test_resume_manifest_and_source_identity_mismatch_fails_closed(tmp_path):
    configured = manifest(raw_outer_range=(0, 2))
    checkpoint = tmp_path / "checkpoint.json"
    resume_manifest(configured, checkpoint, max_new_units=1)
    changed_source = manifest(raw_outer_range=(0, 2), source_sha="0" * 40)
    with pytest.raises(ArtifactIntegrityError, match="manifest identity"):
        read_checkpoint(checkpoint, changed_source)


def test_checkpoint_digest_and_completed_index_tamper_fail_closed(tmp_path):
    configured = manifest(raw_outer_range=(0, 2))
    checkpoint = tmp_path / "checkpoint.json"
    resume_manifest(configured, checkpoint, max_new_units=1)
    payload = json.loads(checkpoint.read_text())
    payload["completed_raw_indices"] = [1]
    body = dict(payload)
    body.pop("checkpoint_digest")
    payload["checkpoint_digest"] = hashlib.sha256(
        canonical_json(body).encode()
    ).hexdigest()
    checkpoint.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(ArtifactIntegrityError, match="completed raw indices"):
        read_checkpoint(checkpoint, configured)


def test_cost_preflight_is_mechanical_and_non_claiming():
    simple = manifest(raw_outer_range=(0, 2))
    composite = manifest(
        raw_outer_range=(0, 2), null_type="composite", statistic="KS"
    )
    projection = estimate_cost(
        [simple, composite], seconds_per_fit=0.01, seconds_per_statistic=0.001
    )
    assert projection.claim_status == "NON_CALIBRATION_NON_CLAIMING"
    assert projection.outer_units == 4
    assert projection.expected_inner_attempts == 4 * 199
    assert projection.expected_fitting_calls == 2 * 200
    assert projection.projected_runtime_seconds > 0
    assert projection.estimated_peak_sample_bytes_per_worker > 0


def test_minimal_e2e_control_covers_nulls_nb_redraw_artifacts_and_resume(tmp_path):
    simple_manifest = manifest(raw_outer_range=(0, 1), statistic="KS")
    composite_manifest = manifest(
        raw_outer_range=(0, 1), null_type="composite", statistic="KS"
    )
    simple = run_manifest(simple_manifest)
    composite = run_manifest(composite_manifest)
    assert simple.outer_results[0].observed_fit_calls == 0
    assert composite.outer_results[0].observed_fit_calls == 1
    assert composite.outer_results[0].replicate_fit_calls == 199

    eligible = np.array([0, 0, 1, 1, 10], dtype=np.int64)
    ineligible = np.array([0, 1, 1, 1, 1], dtype=np.int64)
    calls = 0

    def one_redraw(bound, n, rng):
        nonlocal calls
        calls += 1
        return ineligible.copy() if calls == 2 else eligible.copy()

    nb_manifest = manifest(
        null_type="composite",
        family="negative_binomial",
        statistic="KS",
        n=5,
        canonical_parameters={"r": "2", "p": "0.5"},
        raw_outer_range=(0, 1),
    )
    nb, attempts = run_outer_unit(
        nb_manifest,
        0,
        hooks=RunnerHooks(
            generate=one_redraw,
            statistic=lambda sample, *args, **kwargs: StatisticEvaluation(float(np.mean(sample))),
        ),
    )
    assert nb.inner_ineligible == 1 and nb.inner_eligible == 199
    assert len({attempt.seed_identity for attempt in attempts}) == 200

    artifacts = tmp_path / "artifacts"
    write_run_artifacts(
        artifacts,
        simple_manifest,
        simple,
        table_format="jsonl",
        command="CP05-B E2E NON_CALIBRATION",
        elapsed_seconds=0.0,
    )
    validate_auditable_artifacts(artifacts)
    checkpoint = tmp_path / "e2e-checkpoint.json"
    partial, complete = resume_manifest(
        manifest(raw_outer_range=(0, 2), statistic="KS"),
        checkpoint,
        max_new_units=1,
    )
    assert not complete and len(partial.outer_results) == 1
    resumed, complete = resume_manifest(
        manifest(raw_outer_range=(0, 2), statistic="KS"), checkpoint
    )
    assert complete and len(resumed.outer_results) == 2
