"""Software fixtures only: no R_C=2000 campaign, power or holdout samples."""
from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pyMagicStat.distributions.families import (
    NegativeBinomialFamily, GammaFamily, ExponentialFamily, FitNumericalError,
    FitIdentifiabilityError, NoFiniteMLEError,
)
from experiments.distribution_gof.runner import RunnerHooks
from experiments.distribution_gof.statistics import StatisticEvaluation, evaluate_statistic
from experiments.distribution_gof.seed_derivation import derive_seed, numpy_rng
from experiments.distribution_gof.artifacts import ArtifactIntegrityError, read_parquet_rows
from experiments.distribution_gof.cp05_c import manifest as contract
from experiments.distribution_gof.cp05_c.manifest import DevelopmentManifest, null_matrix
from experiments.distribution_gof.cp05_c import fitting, shadow, aggregate as aggregation
from experiments.distribution_gof.cp05_c.runner import run_cell, run_cells, _execute_outer, wilson95
from experiments.distribution_gof.cp05_c.artifacts import validate_bundle, REQUIRED_FILES
from experiments.distribution_gof.cp05_c.checkpoint import _digest, load_state, save_state


SAMPLE = np.array([0] * 12 + [1] * 4 + [5] * 3 + [20], dtype=np.int64)
GOOD = {"classification": "ELIGIBLE", "converged": True, "parameters": {"r": 1., "p": .5},
        "log_likelihood": -100., "failure_reason": None}


def cell(**changes):
    values = dict(family="negative_binomial", statistic="AD", n=20, canonical_parameters={"r": "1", "p": "0.5"},
                  null_type="composite", B=199)
    values.update(changes)
    return DevelopmentManifest(**values)


def sample_generator(bound, n, rng):
    return SAMPLE.copy() if n == 20 else np.resize(SAMPLE, n)


def cheap_stat(sample, bound, family, statistic, *, parameter_count_estimated):
    return StatisticEvaluation(float(np.mean(sample)))


def fixture_fit(family, sample):
    return fitting.operational_fit(family, sample, copy.deepcopy(GOOD))


HOOKS = RunnerHooks(generate=sample_generator, fit=fixture_fit, statistic=cheap_stat)


def fixture_oracle(manifest):
    return shadow.ShadowOracle(manifest, fast=lambda x: copy.deepcopy(GOOD),
                               canonical=lambda x: copy.deepcopy(GOOD), generate=sample_generator)


def fixture_run(manifest, directory, **kwargs):
    return run_cell(manifest, directory, fixture_target=kwargs.pop("fixture_target", 1),
                    hooks=kwargs.pop("hooks", HOOKS), shadow_factory=kwargs.pop("shadow_factory", fixture_oracle), **kwargs)


def test_independently_materialized_cartesian_matrix():
    # Independent literal DEC-014 grid, not the implementation's count constants.
    expected = set()
    for family, parameters, statistics in (
        ("gamma", [{"shape": s, "scale": "1"} for s in ("0.25", "0.5", "1", "2", "10")], ("AD", "CVM", "KS")),
        ("exponential", [{"scale": "1"}], ("AD", "CVM", "KS")),
        ("negative_binomial", [{"r": r, "p": p} for r in ("0.25", "1", "5", "20") for p in ("0.1", "0.5", "0.9")],
         ("AD", "CVM", "KS", "PEARSON")),
    ):
        for params in parameters:
            for n in (20, 50, 100, 250):
                for statistic in statistics:
                    for null in ("simple", "composite"):
                        for B in (199, 999):
                            expected.add((family, tuple(sorted(params.items())), n, statistic, null, B))
    matrix = null_matrix()
    actual = {(m.family, tuple(sorted(m.canonical_parameters.items())), m.n, m.statistic, m.null_type, m.B) for m in matrix}
    assert actual == expected and len(actual) == len(matrix) == 1056
    assert sum(row[3] in ("AD", "CVM") for row in expected) == sum(m.primary for m in matrix) == 576
    assert sum(row[3] not in ("AD", "CVM") for row in expected) == 480
    assert len(expected) * 2000 == contract.ELIGIBLE_OUTER_TARGET == 2112000
    assert all(m.phase == "CP05-C" and m.namespace_id == "pyMagicStats/STAGE-DIST-FAMILIES-001/CP05-C/v1"
               and m.R_C == 2000 and m.alpha == .05 for m in matrix)
    assert {m.B for m in matrix} == {199, 999}
    assert len({m.canonical_cell_id for m in matrix}) == 1056


@pytest.mark.parametrize("changes", [dict(phase="CP05-B_NON_CALIBRATION"), dict(phase="CP05-D"),
    dict(namespace_id="private"), dict(B=15), dict(B=True), dict(n=21), dict(R_C=1), dict(alpha=.1),
    dict(canonical_parameters={"r": "2", "p": ".5"}), dict(null_type="power"), dict(statistic="POWER"), dict(workers=0)])
def test_manifest_rejects_contract_drift(changes):
    with pytest.raises((ValueError, TypeError)):
        cell(**changes)


def test_canonical_decimal_identity_seed_and_topology():
    a = cell(canonical_parameters={"r": "1.000", "p": "0.5000"})
    b = replace(a, workers=4, batch_size=13)
    assert a.canonical_cell_id == b.canonical_cell_id
    assert json.loads(a.canonical_cell_id) == a.cell_identity
    assert set(a.cell_identity) == {"phase", "null_type", "family", "statistic", "n", "canonical_parameters", "B"}
    for outer in (0, 1, 500):
        for purpose, inner in (("outer_observed", None), ("inner_bootstrap", 0), ("inner_bootstrap", 198)):
            seed = derive_seed(a.canonical_cell_id, outer, purpose, inner)
            expected = hashlib.sha256("\0".join((contract.NAMESPACE, a.canonical_cell_id, str(outer), purpose,
                                                "" if inner is None else str(inner))).encode()).hexdigest()
            assert seed.digest_hex == expected == derive_seed(b.canonical_cell_id, outer, purpose, inner).digest_hex
    with pytest.raises(ValueError):
        derive_seed(a.canonical_cell_id, 0, "outer_observed", namespace="CP05-D")


def test_real_fast_adapter_and_canonical_fixture_agree():
    family = NegativeBinomialFamily()
    fast = fitting.fast_fit(family, SAMPLE)
    canonical = family.fit(SAMPLE)
    assert abs(np.log(fast.fitted_distribution.parameters.r) - np.log(canonical.fitted_distribution.parameters.r)) <= 1e-8
    assert fast.metadata["estimator_id"] == fitting.ESTIMATOR_ID
    assert fast.metadata["estimator_id"] != canonical.metadata["estimator_id"]
    assert fast.backend == fitting.ENGINE and fast.metadata["PRODUCTION_BACKEND"] == "NO"
    assert fast.fitted_distribution.cdf(1) == family.bind(**vars_parameters(fast)).cdf(1)


def test_real_entry_canaries_use_canonical_decimal_and_normal_seeds():
    m = cell()
    result = shadow.ShadowOracle(m).entry()
    assert result["entry_gate"] == "PASS" and all(r["passed"] for r in result["checks"])
    for row in result["checks"]:
        shadow.validate_evidence_row(row)
        if row["role"] == "observed":
            bound = NegativeBinomialFamily().bind(r=1, p=.5)
            sample = bound.rvs(size=m.n, rng=numpy_rng(derive_seed(m.canonical_cell_id, row["raw_outer_index"], "outer_observed")))
            assert shadow.sample_identity(sample)[0] == row["sample_digest"]


@pytest.mark.parametrize("statistic", ["AD", "CVM", "KS", "PEARSON"])
def test_fast_fitted_operations_work_with_real_cp05_statistics(statistic):
    # Pearson needs enough retained categories for two estimated parameters.
    sample = np.resize(SAMPLE, 250) if statistic == "PEARSON" else SAMPLE
    fit = fitting.fast_fit(NegativeBinomialFamily(), sample)
    evaluation = evaluate_statistic(sample, fit.fitted_distribution, "negative_binomial", statistic,
                                   parameter_count_estimated=2)
    assert np.isfinite(evaluation.value) and evaluation.value >= 0


def test_real_fast_bootstrap_core_integration(tmp_path):
    result = run_cell(cell(statistic="KS"), tmp_path, fixture_target=1,
                      hooks=RunnerHooks(generate=sample_generator), shadow_factory=fixture_oracle)
    assert result["cell_status"] == "COMPLETE" and result["fast_fit_count"] == 200
    validate_bundle(tmp_path)


def test_small_sample_pearson_nonassessment_blocks_without_replacement(tmp_path):
    result = run_cell(cell(statistic="PEARSON"), tmp_path, fixture_target=2,
                      hooks=RunnerHooks(generate=sample_generator), shadow_factory=fixture_oracle)
    assert result["cell_status"] == "FAILED" and result["failure_reason"] == "COMPARATOR_NOT_ASSESSABLE"
    assert result["raw_outer_attempts"] == result["fast_fit_count"] == 1
    assert result["eligible_outer_count"] == result["inner_attempts"] == 0
    assert result["Wilson95_upper"] is None and not result["primary"]
    validate_bundle(tmp_path)


def vars_parameters(fit):
    params = fit.fitted_distribution.parameters
    return {"r": params.r, "p": params.p}


@pytest.mark.parametrize("sample,error", [(np.zeros(20, dtype=int), FitIdentifiabilityError),
                                          (np.ones(20, dtype=int), NoFiniteMLEError)])
def test_adapter_maps_exact_cp04_ineligibility(sample, error):
    with pytest.raises(error):
        fitting.fast_fit(NegativeBinomialFamily(), sample)


@pytest.mark.parametrize("manifest", [cell(null_type="simple"), cell(family="gamma", canonical_parameters={"shape": ".25", "scale": 1}),
                                      cell(family="exponential", canonical_parameters={"scale": 1})])
def test_fast_hook_only_nb_composite(manifest):
    with pytest.raises(ValueError):
        fitting.fit_hook(manifest)


def test_simple_nb_never_fits(tmp_path):
    def bomb(*args):
        raise AssertionError("simple null fitter invoked")
    result = fixture_run(cell(null_type="simple"), tmp_path, hooks=replace(HOOKS, fit=bomb))
    assert result["eligible_outer_count"] == 1 and result["fast_fit_count"] == result["canonical_shadow_fit_count"] == 0
    row = read_parquet_rows(tmp_path / "outer_results.parquet")[0]
    assert row["observed_fit_calls"] == row["replicate_fit_calls"] == 0


@pytest.mark.parametrize("family,params,cls,estimator", [
    ("gamma", {"shape": 2, "scale": 1}, GammaFamily, "scipy-gamma-fixed-loc-mle-v1"),
    ("exponential", {"scale": 1}, ExponentialFamily, "pymagicstats-exponential-closed-form-mle-v1")])
def test_continuous_composite_uses_canonical(tmp_path, monkeypatch, family, params, cls, estimator):
    original, calls = cls.fit, []
    def tracked(self, sample):
        calls.append(1)
        return original(self, sample)
    monkeypatch.setattr(cls, "fit", tracked)
    monkeypatch.setattr(fitting.fast_cpu, "fit_negative_binomial", lambda x: pytest.fail("fast routed to continuous"))
    result = run_cell(cell(family=family, canonical_parameters=params), tmp_path, fixture_target=1,
                      hooks=RunnerHooks(statistic=cheap_stat))
    assert result["cell_status"] == "COMPLETE" and len(calls) == 200
    row = read_parquet_rows(tmp_path / "outer_results.parquet")[0]
    assert row["observed_fit_provenance"]["estimator_id"] == estimator


def test_fast_failure_has_no_canonical_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(fitting.fast_cpu, "fit_negative_binomial", lambda x: {**GOOD, "failure_reason": "backend failed"})
    monkeypatch.setattr(NegativeBinomialFamily, "fit", lambda *a: pytest.fail("canonical fallback"))
    result = fixture_run(cell(), tmp_path, hooks=replace(HOOKS, fit=None))
    assert result["cell_status"] == "FAILED" and result["eligible_outer_count"] == 0
    assert result["numerical_failure_count"] == 1
    again = fixture_run(cell(), tmp_path)
    assert again["raw_outer_attempts"] == 1 and again["cell_status"] == "FAILED"


@pytest.mark.parametrize("outcome", [None, {**GOOD, "converged": False}, {**GOOD, "classification": "FAILED"},
    {**GOOD, "parameters": {"r": float("nan"), "p": .5}}, {**GOOD, "log_likelihood": float("inf")},
    {**GOOD, "parameters": {"r": 1, "p": 1}},
    {"classification": "ALL_ZERO_NON_IDENTIFYING", "converged": False, "failure_reason": "backend"}])
def test_adapter_fail_closed_on_bad_evidence(outcome):
    with pytest.raises(FitNumericalError):
        fitting.operational_fit(NegativeBinomialFamily(), SAMPLE, outcome)


@pytest.mark.parametrize("mutate", [
    lambda o: o.update(classification="ALL_ZERO_NON_IDENTIFYING", converged=False, failure_reason="ALL_ZERO_NON_IDENTIFYING"),
    lambda o: o.update(converged=False), lambda o: o["parameters"].update(r=1.0001),
    lambda o: o["parameters"].update(p=.5001), lambda o: o.update(log_likelihood=-100.001),
    lambda o: o["parameters"].update(r=float("nan")), lambda o: o["parameters"].update(p=float("inf")),
    lambda o: o.update(log_likelihood=float("inf")), lambda o: o.update(failure_reason="numerical")])
def test_shadow_mismatch_blocks_before_campaign(tmp_path, mutate):
    bad = copy.deepcopy(GOOD)
    mutate(bad)
    def oracle(m):
        return shadow.ShadowOracle(m, fast=lambda x: bad, canonical=lambda x: GOOD, generate=sample_generator)
    def bomb(*args):
        pytest.fail("campaign started after entry mismatch")
    result = fixture_run(cell(), tmp_path, shadow_factory=oracle, hooks=replace(HOOKS, generate=bomb))
    assert result["cell_status"] == "FAILED_SHADOW_ORACLE" and result["CELL_EXECUTION_STARTED"] == "NO"
    assert result["eligible_outer_count"] == result["raw_outer_attempts"] == 0
    validate_bundle(tmp_path)


def test_ineligible_classification_mismatch_blocks():
    fast = {"classification": "ALL_ZERO_NON_IDENTIFYING", "converged": False, "failure_reason": "ALL_ZERO_NON_IDENTIFYING"}
    canonical = {"classification": "VARIANCE_NOT_GREATER_THAN_MEAN", "converged": False, "failure_reason": "NoFiniteMLEError: fixture"}
    result = shadow.ShadowOracle(cell(), fast=lambda x: fast, canonical=lambda x: canonical, generate=sample_generator).entry()
    assert result["entry_gate"] == "FAIL" and not result["checks"][0]["classification_gate"]


def test_shadow_ineligible_prefix_then_inner_certificate():
    calls = []
    def gen(bound, n, rng):
        calls.append(rng.bit_generator.state)
        return np.zeros(n, dtype=int) if len(calls) in (1, 3) else SAMPLE.copy()
    def b(x):
        return copy.deepcopy(GOOD) if x.any() else {"classification": "ALL_ZERO_NON_IDENTIFYING", "converged": False,
                                                   "failure_reason": "ALL_ZERO_NON_IDENTIFYING"}
    def a(x):
        return copy.deepcopy(GOOD) if x.any() else {"classification": "ALL_ZERO_NON_IDENTIFYING", "converged": False,
                                                   "failure_reason": "FitIdentifiabilityError: fixture"}
    result = shadow.ShadowOracle(cell(), fast=b, canonical=a, generate=gen).entry()
    assert result["entry_gate"] == "PASS"
    assert [(r["role"], r["raw_outer_index"], r["raw_inner_index"]) for r in result["checks"]] == [
        ("observed", 0, None), ("observed", 1, None), ("inner", 1, 0), ("inner", 1, 1)]
    assert all(r["passed"] for r in result["checks"])


@pytest.mark.parametrize("inner_cap", [False, True])
def test_shadow_64_caps_fail_closed(inner_cap):
    calls = []
    def b(x):
        calls.append(1)
        if inner_cap and len(calls) == 1:
            return copy.deepcopy(GOOD)
        return {"classification": "ALL_ZERO_NON_IDENTIFYING", "converged": False, "failure_reason": "ALL_ZERO_NON_IDENTIFYING"}
    def a(x):
        if inner_cap and len(calls) == 1:
            return copy.deepcopy(GOOD)
        return {"classification": "ALL_ZERO_NON_IDENTIFYING", "converged": False, "failure_reason": "FitIdentifiabilityError: fixture"}
    result = shadow.ShadowOracle(cell(), fast=b, canonical=a, generate=sample_generator).entry()
    assert result["entry_gate"] == "FAIL" and len(result["checks"]) == (65 if inner_cap else 64)
    assert result["failure_reason"] == ("SHADOW_INNER_RAW_CAP_EXHAUSTED" if inner_cap else "SHADOW_OUTER_RAW_CAP_EXHAUSTED")


def test_no_flat_objective_waiver():
    seed = derive_seed(cell().canonical_cell_id, 0, "outer_observed")
    b = {**GOOD, "parameters": {"r": 1.001, "p": .5}}
    row = shadow.compare(cell(), "observed", 0, None, SAMPLE, seed, b, GOOD)
    assert row["objective_gate"] and not row["parameter_gate"] and not row["passed"]


def test_shadow_does_not_contaminate_primary_units_or_global_rng(tmp_path):
    before = copy.deepcopy(np.random.get_state())
    m = cell()
    without, attempts = _execute_outer(m, 0, HOOKS, 0)
    result = fixture_run(m, tmp_path)
    main = read_parquet_rows(tmp_path / "outer_results.parquet")[0]
    logical = without.to_dict()
    assert all(main[k] == v for k, v in logical.items() if k != "output_digest")
    assert read_parquet_rows(tmp_path / "inner_accounting.parquet") == [a.to_dict() for a in attempts]
    assert result["eligible_outer_count"] == 1 and result["rejection_count"] == 0
    assert result["Wilson95_upper"] == wilson95(0, 1)[1] and main["p_mc"] == 1
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    checks = json.loads((tmp_path / "shadow_oracle.json").read_text())["checks"]
    assert checks[0]["seed_identity"] == main["seed_identity"]
    assert checks[1]["seed_identity"] == attempts[0].seed_identity
    assert result["fast_fit_count"] == 200 and result["canonical_shadow_fit_count"] == 2


def test_periodic_schedule_exact_raw_indices_without_zero_duplicate():
    checks = fixture_oracle(cell()).entry()["checks"]
    assert [i for i in range(2001) if shadow.periodic_due(i, checks)] == [500, 1000, 1500, 2000]
    assert shadow.periodic_due(0, [])


def test_periodic_observed_mismatch_stops_at_raw_500(tmp_path):
    def oracle(m):
        value = fixture_oracle(m)
        original = value.periodic
        def periodic(index):
            assert index == 500
            value.canonical = lambda x: {**GOOD, "parameters": {"r": 1.01, "p": .5}}
            return original(index)
        value.periodic = periodic
        return value
    def ineligible(*args):
        raise NoFiniteMLEError("fixture")
    result = fixture_run(cell(), tmp_path, fixture_target=6, hooks=replace(HOOKS, fit=ineligible), shadow_factory=oracle)
    assert result["raw_outer_attempts"] == 500 and result["eligible_outer_count"] == 0
    assert result["cell_status"] == "FAILED_SHADOW_ORACLE" and result["canonical_shadow_fit_count"] == 3
    evidence = json.loads((tmp_path / "shadow_oracle.json").read_text())["checks"][-1]
    assert evidence["role"] == "periodic_observed" and evidence["raw_outer_index"] == 500
    assert evidence["raw_inner_index"] is None and not evidence["passed"]
    validate_bundle(tmp_path)
    resumed = fixture_run(cell(), tmp_path, fixture_target=6)
    assert resumed["raw_outer_attempts"] == 500 and resumed["cell_status"] == "FAILED_SHADOW_ORACLE"


def test_raw_eligible_distinction_and_outer_cap(tmp_path):
    calls = []
    def fit(family, x):
        calls.append(1)
        if len(calls) <= 2:
            raise FitIdentifiabilityError("fixture")
        return fixture_fit(family, x)
    result = fixture_run(cell(), tmp_path / "eligible", hooks=replace(HOOKS, fit=fit))
    assert result["raw_outer_attempts"] == 3 and result["eligible_outer_count"] == 1
    rows = read_parquet_rows(tmp_path / "eligible" / "outer_results.parquet")
    assert [(r["raw_outer_index"], r["eligible_outer_index"], r["status"]) for r in rows] == [
        (0, None, "NOT_ASSESSED"), (1, None, "NOT_ASSESSED"), (2, 0, "ASSESSED")]
    assert result["applicability_rate"] == 1/3 and result["outer_ineligibility_reason_counts"] == {"FitIdentifiabilityError": 2}
    def ineligible(*args):
        raise NoFiniteMLEError("fixture")
    failed = fixture_run(cell(), tmp_path / "cap", hooks=replace(HOOKS, fit=ineligible))
    assert failed["raw_outer_attempts"] == 100 and failed["eligible_outer_count"] == 0
    assert failed["cell_status"] == "FAILED" and failed["failure_reason"] == "OUTER_RETRY_CAP_EXHAUSTED"
    assert failed["Wilson95_upper"] is None


def test_nb_inner_redraw_cap_is_100_B_and_not_outer_replacement(tmp_path):
    calls = []
    def fit(family, x):
        calls.append(1)
        if len(calls) == 1:
            return fixture_fit(family, x)
        raise NoFiniteMLEError("fixture")
    result = fixture_run(cell(), tmp_path, hooks=replace(HOOKS, fit=fit))
    assert result["inner_attempts"] == result["inner_fit_attempts"] == 100 * 199
    assert len(calls) == 1 + 100 * 199 and result["raw_outer_attempts"] == 1
    assert result["cell_status"] == "FAILED" and result["failure_reason"] == "RETRY_CAP_EXHAUSTED"
    assert result["eligible_outer_count"] == 0


def test_inner_mathematical_redraw_and_every_refit(tmp_path):
    calls = []
    def fit(family, sample):
        calls.append(1)
        if len(calls) in (2, 4):
            raise NoFiniteMLEError("fixture")
        return fixture_fit(family, sample)
    result = fixture_run(cell(), tmp_path, hooks=replace(HOOKS, fit=fit))
    assert result["inner_attempts"] == 201 and result["fast_fit_count"] == 202
    assert result["inner_ineligibility_histogram"] == {"NoFiniteMLEError": 2}
    validate_bundle(tmp_path)


def test_other_families_fail_without_replacement(tmp_path):
    def fit(*args):
        raise FitIdentifiabilityError("fixture")
    m = cell(family="gamma", canonical_parameters={"shape": 2, "scale": 1})
    result = run_cell(m, tmp_path, fixture_target=2, hooks=replace(HOOKS, fit=fit))
    assert result["cell_status"] == "FAILED" and result["raw_outer_attempts"] == 1


def test_generation_backend_failure_blocks_and_is_accounted(tmp_path):
    def failed_generation(*args):
        raise FloatingPointError("fixture backend overflow")
    result = fixture_run(cell(null_type="simple"), tmp_path, fixture_target=2,
                         hooks=replace(HOOKS, generate=failed_generation))
    assert result["cell_status"] == "FAILED" and result["raw_outer_attempts"] == 1
    assert result["eligible_outer_count"] == 0 and result["numerical_failure_count"] == 1
    assert result["failure_reason_counts"] == {"GENERATION_FAILURE": 1}
    validate_bundle(tmp_path)


def test_resume_exact_prefix_without_duplication_or_entry_recheck(tmp_path):
    m = cell()
    first = fixture_run(m, tmp_path, fixture_target=3, max_new_units=1)
    assert first["cell_status"] == "PAUSED"
    def no_entry(m):
        oracle = fixture_oracle(m)
        oracle.entry = lambda: pytest.fail("entry rechecked on resume")
        return oracle
    final = fixture_run(replace(m, workers=3, batch_size=4), tmp_path, fixture_target=3, shadow_factory=no_entry)
    assert final["cell_status"] == "COMPLETE" and final["raw_outer_attempts"] == final["eligible_outer_count"] == 3
    rows = read_parquet_rows(tmp_path / "outer_results.parquet")
    assert [r["raw_outer_index"] for r in rows] == [0, 1, 2]
    assert len(read_parquet_rows(tmp_path / "inner_accounting.parquet")) == 3 * 199
    validate_bundle(tmp_path)


@pytest.mark.parametrize("corruption", ["digest", "duplicate", "identity", "segment"])
def test_checkpoint_resume_fail_closed(tmp_path, corruption):
    fixture_run(cell(), tmp_path, fixture_target=2, max_new_units=1)
    path = tmp_path / "checkpoint.json"
    payload = json.loads(path.read_text())
    if corruption == "digest":
        payload["target"] = 10
    elif corruption == "duplicate":
        payload["segments"].append(payload["segments"][0])
    elif corruption == "identity":
        payload["manifest_identity"]["phase"] = "CP05-D"
    else:
        segment = tmp_path / payload["segments"][0]["path"]
        segment.write_text(segment.read_text() + " ")
    if corruption in ("duplicate", "identity"):
        payload.pop("checkpoint_digest")
        payload["checkpoint_digest"] = _digest(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ArtifactIntegrityError):
        fixture_run(cell(), tmp_path, fixture_target=2)


def test_topology_and_configuration_order_invariance(tmp_path):
    cells = (cell(null_type="simple"), cell(null_type="simple", statistic="CVM"), cell(null_type="simple", B=999))
    options = dict(fixture_target=2, hooks=HOOKS)
    run_cells(cells, tmp_path / "serial", workers=1, batch_size=1, **options)
    run_cells(tuple(reversed(cells)), tmp_path / "parallel", workers=3, batch_size=3, **options)
    for m in cells:
        a = read_parquet_rows(tmp_path / "serial" / m.directory_name / "outer_results.parquet")
        b = read_parquet_rows(tmp_path / "parallel" / m.directory_name / "outer_results.parquet")
        provenance = {"source_sha", "config_digest", "environment_digest", "output_digest"}
        assert [{k: v for k, v in r.items() if k not in provenance} for r in a] == [
            {k: v for k, v in r.items() if k not in provenance} for r in b]


def test_artifact_digest_inventory_and_corruption(tmp_path):
    fixture_run(cell(), tmp_path)
    validate_bundle(tmp_path)
    assert set(REQUIRED_FILES).issubset({p.name for p in tmp_path.iterdir()})
    for name in REQUIRED_FILES:
        path = tmp_path / name
        original = path.read_bytes()
        path.write_bytes(original + b"tamper")
        with pytest.raises(ArtifactIntegrityError):
            validate_bundle(tmp_path)
        path.write_bytes(original)
    assert json.loads((tmp_path / "shadow_oracle.json").read_text())["FLAT_OBJECTIVE_WAIVER_USED"] == "NO"


@pytest.mark.parametrize("change", ["parameter", "objective", "gate", "schedule", "threshold"])
def test_redigested_shadow_evidence_still_fails_closed(tmp_path, change):
    fixture_run(cell(), tmp_path)
    payload = json.loads((tmp_path / "checkpoint.json").read_text())
    row = payload["shadow"]["checks"][0]
    if change == "parameter":
        row["r_fast"] = 1.01
    elif change == "objective":
        row["ll_fast"] = -110.
    elif change == "gate":
        row["passed"] = False
    elif change == "schedule":
        row["raw_outer_index"] = 500
    else:
        payload["shadow"]["NB_TRANSFORM_ATOL"] = 1e-2
    payload.pop("checkpoint_digest")
    payload["checkpoint_digest"] = _digest(payload)
    (tmp_path / "checkpoint.json").write_text(json.dumps(payload))
    with pytest.raises(ArtifactIntegrityError):
        load_state(tmp_path, cell(), 1, True)


def fake_scientific_bundle(directory):
    m = directory
    return m, {"canonical_cell_id": m.canonical_cell_id, "software_fixture": False, "primary": m.primary,
               "cell_status": "COMPLETE", "CELL_ACCEPTED": True, "SHADOW_GATE": "PASS" if m.nb_composite else "NOT_APPLICABLE",
               "eligible_outer_count": 2000, "raw_outer_attempts": 2000, "inner_attempts": 2000*m.B}


def test_aggregator_exact_inventory_and_per_cell_gates(monkeypatch):
    monkeypatch.setattr(aggregation, "validate_bundle", fake_scientific_bundle)
    matrix = null_matrix()
    result = aggregation.aggregate(matrix)
    assert result["EXPECTED_CONFIGURATIONS"] == result["COMPLETE_CONFIGURATIONS"] == 1056
    assert result["PRIMARY_CONFIGURATIONS"] == 576 and result["COMPARATOR_CONFIGURATIONS"] == 480
    assert result["TOTAL_ELIGIBLE_OUTERS"] == result["TOTAL_RAW_OUTERS"] == 2112000
    assert result["ALL_PRIMARY_NULL_CELLS_WILSON_PASS"] and result["SHADOW_GATE_PASS"]
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        aggregation.aggregate(matrix[:-1])
    with pytest.raises(ArtifactIntegrityError, match="duplicate"):
        aggregation.aggregate(matrix + matrix[:1])
    def fail(directory):
        m, data = fake_scientific_bundle(directory)
        if m == matrix[0]:
            data["CELL_ACCEPTED"] = False
        return m, data
    monkeypatch.setattr(aggregation, "validate_bundle", fail)
    assert not aggregation.aggregate(matrix)["ALL_PRIMARY_NULL_CELLS_WILSON_PASS"]


def test_aggregator_excludes_fixture_bundles(tmp_path):
    fixture_run(cell(), tmp_path)
    with pytest.raises(ArtifactIntegrityError, match="software fixtures"):
        aggregation.aggregate([tmp_path])


def test_wilson_unrounded_cell_gate():
    assert wilson95(100, 2000)[1] < .065
    assert wilson95(110, 2000)[1] > .065
    assert wilson95(0, 0) is None


def test_real_runs_reject_custom_scientific_hooks(tmp_path):
    with pytest.raises(ValueError, match="software fixtures"):
        run_cell(cell(), tmp_path, hooks=HOOKS)
    with pytest.raises(ValueError):
        fixture_run(cell(), tmp_path, fixture_target=2000)


def test_scientific_preflight_rejects_wrong_source_before_sampling(tmp_path):
    with pytest.raises(ValueError, match="source_sha"):
        run_cell(cell(source_sha="1"*40), tmp_path, max_new_units=0)
    assert not tmp_path.joinpath("checkpoint.json").exists()


def test_cli_manifest_generates_no_samples_or_campaign(tmp_path, monkeypatch):
    from experiments.distribution_gof.cp05_c import __main__ as cli
    monkeypatch.setattr(cli, "run_cells", lambda *a, **k: pytest.fail("manifest started campaign"))
    cli.main(["manifest", "--root", str(tmp_path)])
    data = json.loads((tmp_path / "null_matrix.json").read_text())
    assert len(data["cells"]) == 1056 and data["POWER_STAGE_IMPLEMENTED"] == "NO"


@pytest.mark.parametrize("action", ["power", "holdout", "select-method", "select-B"])
def test_cli_has_no_power_holdout_or_selection_entrypoint(tmp_path, action):
    from experiments.distribution_gof.cp05_c.__main__ import main
    with pytest.raises(SystemExit):
        main([action, "--root", str(tmp_path)])


def test_power_and_holdout_execution_absent_and_production_untouched():
    root = Path(__file__).resolve().parents[2]
    for name in ("Weibull", "Lognormal", "R_POWER", "holdout_secret", "sealed_matrix"):
        assert not any(name in p.read_text() for p in (root / "experiments/distribution_gof/cp05_c").glob("*.py"))
    assert contract.BLOCKED_STAGES == {"POWER_STAGE_IMPLEMENTED": "NO", "POWER_PREREG_SPEC_COMPLETE": "NO",
        "POWER_EXECUTION_AUTHORIZED": "NO", "METHOD_SELECTION_PERFORMED": "NO", "CP05_C_COMPLETE": "NO",
        "CP05_D_ACCESSED": "NO", "CP05_D_EXECUTED": "NO"}
    # Baseline object comparison also works before this candidate is committed.
    import subprocess
    paths = ["pyMagicStat", "experiments/distribution_gof/manifest.py", "experiments/distribution_gof/runner.py",
             "experiments/distribution_gof/bootstrap.py", "experiments/distribution_gof/seed_derivation.py",
             "experiments/distribution_gof/statistics.py", "experiments/distribution_gof/cuda_calibration/nb_fitting_perf01",
             "experiments/distribution_gof/cuda_calibration/cuda_candidate.py"]
    result = subprocess.run(["git", "diff", "--exit-code", contract.BASE_SHA, "--", *paths], cwd=root, capture_output=True)
    assert result.returncode == 0, result.stdout.decode()
