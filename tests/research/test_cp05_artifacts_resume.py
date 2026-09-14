from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.distribution_gof.artifacts import (
    ArtifactIntegrityError,
    read_parquet_rows,
    read_checkpoint,
    resume_manifest,
    validate_artifact_digests,
    validate_auditable_artifacts,
    write_run_artifacts,
)
from experiments.distribution_gof.cost_preflight import estimate_cost
from experiments.distribution_gof.manifest import canonical_json
from experiments.distribution_gof.runner import (
    RunOutput,
    RunnerHooks,
    run_manifest,
    run_outer_unit,
)
from experiments.distribution_gof.statistics import StatisticEvaluation

from ._helpers import manifest


@pytest.fixture(scope="module")
def parquet_source_output():
    return run_manifest(manifest(raw_outer_range=(0, 1)))


def _logical_jsonl_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize(
    (
        "family",
        "null_type",
        "canonical_parameters",
        "reason_counts",
        "fit_provenance",
    ),
    [
        pytest.param(
            "exponential", "simple", {"scale": "1"}, {}, None, id="continuous-simple"
        ),
        pytest.param(
            "exponential",
            "composite",
            {"scale": "1"},
            {},
            {
                "estimation_method": "maximum_likelihood",
                "estimator_id": "exponential-mle-v1",
            },
            id="continuous-composite",
        ),
        pytest.param(
            "negative_binomial",
            "composite",
            {"r": "2", "p": "0.5"},
            {},
            {"estimation_method": "maximum_likelihood", "estimator_id": "nb-mle-v1"},
            id="nb-clean",
        ),
        pytest.param(
            "negative_binomial",
            "composite",
            {"r": "2", "p": "0.5"},
            {"NoFiniteMLEError": 3},
            {"estimation_method": "maximum_likelihood", "estimator_id": "nb-mle-v1"},
            id="nb-one-reason",
        ),
        pytest.param(
            "negative_binomial",
            "composite",
            {"r": "2", "p": "0.5"},
            {"FitIdentifiabilityError": 2, "NoFiniteMLEError": 3},
            {"estimation_method": "maximum_likelihood", "estimator_id": "nb-mle-v1"},
            id="nb-multiple-reasons",
        ),
    ],
)
def test_parquet_structured_fields_round_trip_as_exact_logical_rows(
    tmp_path,
    parquet_source_output,
    family,
    null_type,
    canonical_parameters,
    reason_counts,
    fit_provenance,
):
    configured = manifest(
        family=family,
        null_type=null_type,
        canonical_parameters=canonical_parameters,
        raw_outer_range=(0, 1),
    )
    result = replace(
        parquet_source_output.outer_results[0],
        canonical_cell_id=configured.canonical_cell_id,
        inner_ineligibility_reason_counts=reason_counts,
        observed_fit_provenance=fit_provenance,
        observed_fit_calls=1 if null_type == "composite" else 0,
        replicate_fit_calls=configured.B if null_type == "composite" else 0,
    )
    inner = tuple(
        replace(row, canonical_cell_id=configured.canonical_cell_id)
        for row in parquet_source_output.inner_accounting
    )
    output = RunOutput(outer_results=(result,), inner_accounting=inner)
    parquet_directory = tmp_path / "parquet"
    jsonl_directory = tmp_path / "jsonl"

    parquet_digests = write_run_artifacts(
        parquet_directory,
        configured,
        output,
        command="CP05-B Parquet round trip",
        elapsed_seconds=0.0,
    )
    write_run_artifacts(
        jsonl_directory,
        configured,
        output,
        table_format="jsonl",
        command="CP05-B Parquet round trip",
        elapsed_seconds=0.0,
    )

    parquet_outer = read_parquet_rows(parquet_directory / "outer_results.parquet")
    parquet_inner = read_parquet_rows(parquet_directory / "inner_accounting.parquet")
    assert parquet_outer == _logical_jsonl_rows(jsonl_directory / "outer_results.jsonl")
    assert parquet_inner == _logical_jsonl_rows(jsonl_directory / "inner_accounting.jsonl")
    assert parquet_outer[0]["inner_ineligibility_reason_counts"] == reason_counts

    import pyarrow.parquet as pq

    encoded = pq.read_table(parquet_directory / "outer_results.parquet").to_pylist()[0]
    assert encoded["inner_ineligibility_reason_counts"] == canonical_json(reason_counts)
    assert encoded["canonical_parameters"] == canonical_json(
        dict(configured.canonical_parameters)
    )
    assert encoded["observed_fit_provenance"] == canonical_json(fit_provenance)
    assert encoded["raw_inner_indices"] == canonical_json(result.raw_inner_indices)
    assert "outer_results.parquet" in parquet_digests
    validate_auditable_artifacts(parquet_directory)
    assert not list(parquet_directory.glob(".*.tmp"))


def test_parquet_failure_closes_handle_cleans_temp_and_delays_digest(
    tmp_path, monkeypatch, parquet_source_output
):
    configured = manifest(raw_outer_range=(0, 1))
    captured_temporary: Path | None = None
    injected_error = RuntimeError("injected Parquet write failure")

    def fail_after_partial_write(table, destination, **kwargs):
        del table, kwargs
        nonlocal captured_temporary
        captured_temporary = Path(destination.name)
        destination.write(b"partial parquet")
        destination.flush()
        raise injected_error

    import pyarrow.parquet as pq

    with monkeypatch.context() as patch:
        patch.setattr(pq, "write_table", fail_after_partial_write)
        with pytest.raises(RuntimeError, match="injected Parquet write failure") as raised:
            write_run_artifacts(
                tmp_path,
                configured,
                parquet_source_output,
                command="CP05-B injected Parquet failure",
                elapsed_seconds=0.0,
            )

    assert raised.value is injected_error
    assert captured_temporary is not None
    assert not captured_temporary.exists()
    assert not (tmp_path / "outer_results.parquet").exists()
    assert not (tmp_path / "digests.json").exists()
    assert not list(tmp_path.glob(".*.tmp"))
    captured_temporary.write_bytes(b"unlocked")
    captured_temporary.unlink()

    digests = write_run_artifacts(
        tmp_path,
        configured,
        parquet_source_output,
        command="CP05-B successful Parquet retry",
        elapsed_seconds=0.0,
    )
    assert (tmp_path / "digests.json").is_file()
    assert validate_artifact_digests(tmp_path) == digests
    assert not list(tmp_path.glob(".*.tmp"))


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
