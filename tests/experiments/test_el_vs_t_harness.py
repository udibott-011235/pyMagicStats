from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

from experiments.el_vs_t import backends, metrics
from experiments.el_vs_t.aggregate import (
    aggregate_calibration,
    summarize_cell,
    validate_replicate_ids,
    validate_shard_manifests,
)
from experiments.el_vs_t.backends import SampleBackend
from experiments.el_vs_t.metrics import MethodExecutor, REPLICATE_COLUMNS, evaluate_batch
from experiments.el_vs_t.runner import RunConfig, run_shard
from experiments.el_vs_t.scenarios import (
    HOLDOUT_POLICY_VERSION,
    HoldoutViolation,
    scenario_registry,
    select_cells,
    validate_requested_scenario_id,
    validate_sample_size,
    validate_scenario_definition,
)
from experiments.el_vs_t.seeds import (
    derive_seed,
    owned_replicate_ids,
)
from experiments.el_vs_t.storage import read_json
from pyMagicStat.assumptions.models import Estimand, InferenceDesign
from pyMagicStat.inference.capabilities import capability_for
from pyMagicStat.inference.selector import MethodSelector


def _normal_cell():
    return select_cells(("normal",), (5,))[0]


def _record_for_sample(sample: np.ndarray) -> dict[str, object]:
    cell = _normal_cell()
    values = np.asarray(sample, dtype=float).reshape(1, -1)
    seed = derive_seed(321, cell.scenario.name, 0)
    backend = SampleBackend("cpu")
    diagnostics = backend.diagnostics(backends.NativeBatch(values, "cpu"))
    with MethodExecutor(1) as executor:
        records, _ = evaluate_batch(
            values,
            diagnostics,
            cell.scenario,
            (0,),
            (seed,),
            shard_id=0,
            num_shards=1,
            alpha=0.05,
            confidence_level=0.95,
            generation_backend="cpu",
            executor=executor,
        )
    return records[0]


def _evaluate_one(monkeypatch: pytest.MonkeyPatch):
    cell = _normal_cell()
    seed = derive_seed(123, cell.scenario.name, 0)
    backend = SampleBackend("cpu")
    native = backend.generate_native(cell.scenario, cell.n, (seed,))
    diagnostics = backend.diagnostics(native)
    samples = backend.to_cpu(native)
    seen: dict[str, np.ndarray] = {}
    original_t = metrics._student_t_task
    original_el_test = metrics.empirical_likelihood_mean_test
    original_el_ci = metrics.empirical_likelihood_mean_ci

    def t_wrapper(task):
        seen["t"] = np.array(task[0], copy=True)
        return original_t(task)

    def el_test_wrapper(sample, mu):
        seen["el_test"] = np.array(sample, copy=True)
        return original_el_test(sample, mu)

    def el_ci_wrapper(sample, confidence):
        seen["el_ci"] = np.array(sample, copy=True)
        return original_el_ci(sample, confidence)

    monkeypatch.setattr(metrics, "_student_t_task", t_wrapper)
    monkeypatch.setattr(metrics, "empirical_likelihood_mean_test", el_test_wrapper)
    monkeypatch.setattr(metrics, "empirical_likelihood_mean_ci", el_ci_wrapper)
    with MethodExecutor(1) as executor:
        records, _ = evaluate_batch(
            samples,
            diagnostics,
            cell.scenario,
            (0,),
            (seed,),
            shard_id=0,
            num_shards=1,
            alpha=0.05,
            confidence_level=0.95,
            generation_backend="cpu",
            executor=executor,
        )
    return samples[0], seen, records[0]


def test_same_replicate_is_supplied_to_student_t_and_both_el_operations(monkeypatch):
    sample, seen, record = _evaluate_one(monkeypatch)

    assert set(seen) == {"t", "el_test", "el_ci"}
    for received in seen.values():
        np.testing.assert_array_equal(received, sample)
    assert record["paired_sample_fingerprint"]


def test_hull_outside_has_unconditional_noncoverage_and_rejection_with_diagnostics():
    record = _record_for_sample(np.array([1.0, 2.0, 4.0]))
    summary, _ = summarize_cell(pd.DataFrame([record]))

    assert record["mu0_hull_location"] == "outside"
    assert record["el_hull_outside"] == 1
    assert record["el_regular"] == 0
    assert record["el_ci_available"] == 1
    assert record["el_test_numerical_failure"] == 0
    assert record["el_solver_failure"] == 0
    assert record["el_ci_covers_mu0_unconditional"] == 0
    assert record["el_coverage_unconditional_eligible"] == 1
    assert record["el_reject_unconditional"] == 1
    assert record["el_type1_unconditional_eligible"] == 1
    assert math.isnan(record["el_ci_covers_mu0_regular"])
    assert math.isnan(record["el_reject_regular"])
    assert summary["el_coverage_unconditional_denominator"] == 1
    assert summary["el_coverage_unconditional"] == 0.0
    assert summary["el_coverage_regular_denominator"] == 0
    assert summary["el_type1_unconditional_denominator"] == 1
    assert summary["el_type1_unconditional"] == 1.0
    assert summary["el_type1_regular_denominator"] == 0
    assert summary["el_hull_outside_rate"] == 1.0
    assert summary["el_nonregular_rate"] == 1.0


def test_numerical_failures_remain_missing_from_unconditional_metrics(monkeypatch):
    def fail_numerically(*args, **kwargs):
        raise FloatingPointError("synthetic solver failure")

    monkeypatch.setattr(metrics, "empirical_likelihood_mean_test", fail_numerically)
    monkeypatch.setattr(metrics, "empirical_likelihood_mean_ci", fail_numerically)
    record = _record_for_sample(np.array([-1.0, 0.5, 2.0]))
    summary, _ = summarize_cell(pd.DataFrame([record]))

    assert record["el_hull_outside"] == 0
    assert record["el_test_numerical_failure"] == 1
    assert record["el_ci_numerical_failure"] == 1
    assert record["el_solver_failure"] == 1
    assert record["el_type1_unconditional_eligible"] == 0
    assert math.isnan(record["el_reject_unconditional"])
    assert record["el_coverage_unconditional_eligible"] == 0
    assert math.isnan(record["el_ci_covers_mu0_unconditional"])
    assert summary["el_type1_unconditional_denominator"] == 0
    assert summary["el_coverage_unconditional_denominator"] == 0
    assert summary["el_solver_failure_rate"] == 1.0


def test_seed_derivation_is_stable_and_coordinate_sensitive():
    first = derive_seed(20260829, "normal", 17)
    second = derive_seed(20260829, "normal", 17)

    assert first == second
    assert first != derive_seed(20260829, "normal", 18)
    assert first != derive_seed(20260829, "student_t_df_5", 17)


def test_shards_are_disjoint_and_cover_global_replicate_ids():
    owned = [set(owned_replicate_ids(23, shard, 4)) for shard in range(4)]

    assert set.union(*owned) == set(range(23))
    assert all(owned[left].isdisjoint(owned[right]) for left in range(4) for right in range(left + 1, 4))


def test_same_shard_rerun_has_identical_ids_and_seed_identities():
    ids_first = tuple(owned_replicate_ids(31, 2, 5))
    ids_second = tuple(owned_replicate_ids(31, 2, 5))
    seeds_first = [derive_seed(9, "normal", value).identity for value in ids_first]
    seeds_second = [derive_seed(9, "normal", value).identity for value in ids_second]

    assert ids_first == ids_second
    assert seeds_first == seeds_second


def test_seed_and_cpu_sample_are_invariant_to_shard_count():
    cell = _normal_cell()
    replicate_id = 17
    backend = SampleBackend("cpu")
    identities = []
    samples = []
    for num_shards in (1, 2, 7, 20):
        shard_id = replicate_id % num_shards
        assert replicate_id in owned_replicate_ids(40, shard_id, num_shards)
        seed = derive_seed(20260829, cell.scenario.name, replicate_id)
        identities.append(seed.identity)
        batch = backend.generate_native(cell.scenario, cell.n, (seed,))
        samples.append(backend.to_cpu(batch)[0])

    assert len(set(identities)) == 1
    for sample in samples[1:]:
        np.testing.assert_array_equal(sample, samples[0])


@pytest.mark.parametrize("sample_size", [6, 12, 25, 35, 45, 65, 90, 150, 350, 1000, 5000])
def test_reserved_sample_size_fails_closed(sample_size):
    with pytest.raises(HoldoutViolation, match=HOLDOUT_POLICY_VERSION):
        validate_sample_size(sample_size)


@pytest.mark.parametrize("sigma", [0.35, 0.75, 1.25])
def test_reserved_lognormal_sigma_fails_closed(sigma):
    with pytest.raises(HoldoutViolation, match="lognormal sigma"):
        validate_scenario_definition(family="lognormal", parameters={"sigma": sigma})


@pytest.mark.parametrize("df", [4, 7, 15])
def test_reserved_student_t_df_fails_closed(df):
    with pytest.raises(HoldoutViolation, match="Student-t df"):
        validate_scenario_definition(family="student_t", parameters={"df": df})


@pytest.mark.parametrize("epsilon", [0.003, 0.015, 0.04])
def test_reserved_contamination_epsilon_fails_closed(epsilon):
    with pytest.raises(HoldoutViolation, match="contamination epsilon"):
        validate_scenario_definition(
            family="normal_contamination_asymmetric",
            parameters={"epsilon": epsilon},
        )


@pytest.mark.parametrize("family", ["Laplace", "WEIBULL", "pareto", "Beta"])
def test_reserved_family_fails_closed_case_insensitively(family):
    with pytest.raises(HoldoutViolation, match="reserved family"):
        validate_scenario_definition(family=family, parameters={})


@pytest.mark.parametrize(
    "scenario_id",
    ["lognormal_sigma_0.35", "student_t_df_7", "contamination_eps_0p015", "Laplace"],
)
def test_reserved_cli_style_scenario_name_fails_before_registry_lookup(scenario_id):
    with pytest.raises(HoldoutViolation):
        validate_requested_scenario_id(scenario_id)


def test_canonical_registry_contains_no_reserved_holdout_definition():
    assert scenario_registry()


def test_aggregation_detects_duplicate_replicate_ids():
    frame = pd.DataFrame(
        {"replicate_id": [0, 0, 1], "shard_id": [0, 0, 1]}
    )
    with pytest.raises(ValueError, match="duplicate replicate IDs"):
        validate_replicate_ids(frame, replicates_per_cell=3, num_shards=2)


def _compatible_manifests():
    methods = {"t": {"version": "1"}, "el": {"version": "1"}}
    run = {
        "num_shards": 1,
        "run_id": "run",
        "repository_sha": "abc",
        "alpha": 0.05,
        "confidence_level": 0.95,
        "scenario_registry_digest": "registry",
        "method_versions": methods,
        "seed_derivation_scheme": "seed-v2",
        "seed_namespace": "experiment-v1",
        "el_accounting_version": "accounting-v2",
        "storage_format": "csv.gz",
        "holdout_used": False,
    }
    shard = {
        "shard_id": 0,
        "status": "complete",
        "holdout_used": False,
        "run_id": "run",
        "repository_sha": "abc",
        "alpha": 0.05,
        "confidence_level": 0.95,
        "scenario_registry_digest": "registry",
        "method_versions_digest": __import__("hashlib").sha256(
            __import__("json").dumps(methods, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "seed_derivation_scheme": "seed-v2",
        "seed_namespace": "experiment-v1",
        "el_accounting_version": "accounting-v2",
        "num_shards": 1,
        "storage_format": "csv.gz",
    }
    return run, shard


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("repository_sha", "different"),
        ("alpha", 0.01),
        ("confidence_level", 0.90),
        ("scenario_registry_digest", "different"),
        ("method_versions_digest", "different"),
        ("seed_derivation_scheme", "different"),
        ("seed_namespace", "different"),
        ("el_accounting_version", "different"),
    ],
)
def test_aggregation_detects_incompatible_metadata(field, bad_value):
    run, shard = _compatible_manifests()
    shard[field] = bad_value
    with pytest.raises(ValueError, match="incompatible shard"):
        validate_shard_manifests(run, [shard])


def test_aggregation_detects_missing_shard():
    run, shard = _compatible_manifests()
    run["num_shards"] = 2
    with pytest.raises(ValueError, match="missing or unexpected shards"):
        validate_shard_manifests(run, [shard])


def test_resume_skips_complete_blocks_and_force_recomputes(tmp_path, monkeypatch):
    def routing_is_forbidden(*args, **kwargs):
        raise AssertionError("MethodSelector must never be invoked")

    monkeypatch.setattr(MethodSelector, "select", routing_is_forbidden)
    base = RunConfig(
        output=tmp_path / "run",
        replicates_per_cell=3,
        backend="cpu",
        batch_size=2,
        scenario_ids=("normal",),
        sample_sizes=(5,),
        storage_format="csv.gz",
    )
    first = run_shard(base)
    second = run_shard(base)
    forced = run_shard(RunConfig(**{**base.__dict__, "force": True}))

    assert first.computed_blocks == 2
    assert second.computed_blocks == 0
    assert second.skipped_blocks == 2
    assert forced.computed_blocks == 2
    manifest = read_json(base.output / "run_manifest.json")
    assert manifest["holdout_used"] is False
    assert manifest["holdout_exclusion_policy"]["mode"] == "fail_closed_before_generation"


def test_cpu_backend_requires_no_gpu_library():
    backend = SampleBackend("cpu")
    cell = _normal_cell()
    seeds = (derive_seed(1, "normal", 0), derive_seed(1, "normal", 1))
    batch = backend.generate_native(cell.scenario, cell.n, seeds)

    assert batch.engine == "cpu"
    assert backend.to_cpu(batch).shape == (2, 5)
    assert np.isfinite(backend.diagnostics(batch)).all()


def test_auto_backend_falls_back_safely_without_cupy(monkeypatch):
    monkeypatch.setattr(backends, "_load_cupy", lambda: (None, "not installed", None))
    backend = backends.resolve_backend("auto")

    assert backend.info.resolved == "cpu-fallback"
    assert backend.info.gpu_available is False


def test_gpu_backend_schema_and_inputs_when_available():
    try:
        backend = SampleBackend("gpu")
    except RuntimeError:
        pytest.skip("CuPy/CUDA is not available in this environment")
    cell = _normal_cell()
    seeds = (derive_seed(5, "normal", 0),)
    native = backend.generate_native(cell.scenario, cell.n, seeds)
    samples = backend.to_cpu(native)

    assert native.engine == "gpu"
    assert samples.shape == (1, 5)
    assert np.isfinite(samples).all()
    assert np.isfinite(backend.diagnostics(native)).all()


def test_method_outputs_are_backend_label_invariant(monkeypatch):
    sample, _, first = _evaluate_one(monkeypatch)
    cell = _normal_cell()
    seed = derive_seed(123, "normal", 0)
    diagnostics = SampleBackend("cpu").diagnostics(
        backends.NativeBatch(sample.reshape(1, -1), "cpu")
    )
    with MethodExecutor(1) as executor:
        rows, _ = evaluate_batch(
            sample.reshape(1, -1),
            diagnostics,
            cell.scenario,
            (0,),
            (seed,),
            shard_id=0,
            num_shards=1,
            alpha=0.05,
            confidence_level=0.95,
            generation_backend="gpu",
            executor=executor,
        )
    second = rows[0]
    first = {key: value for key, value in first.items() if key != "generation_backend"}
    second = {key: value for key, value in second.items() if key != "generation_backend"}
    for key in first:
        if isinstance(first[key], float) and math.isnan(first[key]):
            assert math.isnan(second[key])
        else:
            assert second[key] == first[key]


def test_capability_remains_uncalibrated_and_not_automatically_selectable():
    capability = capability_for(
        "empirical_likelihood", InferenceDesign.ONE_SAMPLE, Estimand.MEAN
    )

    assert capability is not None
    assert capability.calibrated is False
    assert capability.automatic_selection_allowed is False


def test_summary_uses_relevant_denominators_and_mcse():
    frame = pd.DataFrame(
        {
            "scenario_id": ["normal"] * 4,
            "family": ["normal"] * 4,
            "parameters_json": ["{}"] * 4,
            "n": [5] * 4,
            "t_reject": [1.0, 0.0, 1.0, 0.0],
            "el_reject_unconditional": [1.0, 0.0, 1.0, 0.0],
            "el_reject_regular": [1.0, 0.0, math.nan, 0.0],
            "el_type1_unconditional_eligible": [1, 1, 1, 1],
            "t_ci_covers_mu0": [1.0, 1.0, 0.0, 0.0],
            "el_ci_covers_mu0_unconditional": [1.0, 0.0, 0.0, 0.0],
            "el_ci_covers_mu0_regular": [1.0, 0.0, math.nan, 0.0],
            "el_coverage_unconditional_eligible": [1, 1, 1, 1],
            "t_test_numerical_failure": [0, 0, 0, 0],
            "el_test_numerical_failure": [0, 0, 1, 0],
            "t_ci_numerical_failure": [0, 0, 0, 0],
            "el_ci_numerical_failure": [0, 0, 1, 0],
            "t_ci_width": [2.0, 2.0, 2.0, 2.0],
            "el_ci_width": [1.0, 2.0, math.nan, 4.0],
            "mu0_hull_location": ["inside", "inside", "outside", "inside"],
            "el_hull_outside": [0, 0, 1, 0],
            "el_boundary": [0] * 4,
            "el_regular": [1, 1, 0, 1],
            "el_ci_available": [1, 1, 0, 1],
            "el_solver_failure": [0, 0, 0, 0],
        }
    )
    summary, disagreement = summarize_cell(frame)

    assert summary["t_type1_denominator"] == 4
    assert summary["t_type1"] == 0.5
    assert summary["t_type1_mcse"] == 0.25
    assert summary["el_type1_unconditional_denominator"] == 4
    assert summary["el_type1_unconditional"] == 0.5
    assert summary["el_type1_regular_denominator"] == 3
    assert summary["el_type1_regular_mcse"] == pytest.approx(math.sqrt((1 / 3) * (2 / 3) / 3))
    assert summary["el_coverage_unconditional_denominator"] == 4
    assert summary["el_coverage_regular_denominator"] == 3
    assert disagreement["rejection_unconditional_pair_denominator"] == 4
    assert disagreement["width_pair_denominator"] == 3


def test_two_shard_smoke_aggregates_without_simulation_rerun(tmp_path):
    root = tmp_path / "calibration"
    for shard_id in (0, 1):
        run_shard(
            RunConfig(
                output=root,
                replicates_per_cell=4,
                backend="cpu",
                batch_size=2,
                shard_id=shard_id,
                num_shards=2,
                scenario_ids=("normal",),
                sample_sizes=(5,),
                storage_format="csv.gz",
            )
        )

    metadata = aggregate_calibration(root, root / "summary")

    assert metadata["paired_replicate_rows"] == 4
    assert metadata["holdout_used"] is False
    assert (root / "summary" / "el_vs_t_summary.csv").is_file()
    assert (root / "summary" / "el_vs_t_disagreement.csv").is_file()
    assert "POLICY — NOT DETERMINED" in (root / "summary" / "el_vs_t_report.md").read_text(encoding="utf-8")
    summary = pd.read_csv(root / "summary" / "el_vs_t_summary.csv")
    assert summary.loc[0, "R"] == 4


def test_replicate_schema_is_explicit_and_complete(monkeypatch):
    _, _, record = _evaluate_one(monkeypatch)

    assert tuple(record) == REPLICATE_COLUMNS
