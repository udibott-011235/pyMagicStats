"""DEC-020 routing and fail-closed regressions; no CUDA execution/equivalence claim."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.distribution_gof.cuda_calibration import cp05_c2c_equivalence_runner as runner


CELL_ID = "negative_binomial|r=0.25,p=0.1|n=20|AD|composite"
IDENTITY = CELL_ID + "|raw_outer=2|raw_inner=7"
DIGEST = "bca89c41ebd5857f10a8b9908766c6dd9486eb0bd0bbb6813304823a83c15a0b"
QUANTITIES = ("pmf", "logPMF", "cdf", "sf", "logCDF", "logSF")


@pytest.fixture(scope="module")
def exact_failure():
    cell = next(c for c in runner.primary_fixture_matrix() if c.canonical_id == CELL_ID)
    observed, _ = runner.fixed_observed(cell, 2, "CP05-C2C")
    observed_fit = runner.reference_fit(cell.family, observed)
    seed = runner.derive_seed("CP05-C2C", cell.canonical_id, 2, "inner_bootstrap", 7)
    sample = runner._generate(cell.family, observed_fit["parameters"], cell.n, seed)
    reference = runner.evaluate_reference_record(cell, sample)
    support = runner.certify_nb_support(sample, reference["bound"], cell.statistic)
    return cell, sample, reference, support


def mock_candidate(monkeypatch):
    """Spy only on the candidate surface, with no reference fit/result sharing."""
    calls = {}
    monkeypatch.setattr(runner.cuda_candidate, "cp", SimpleNamespace(
        all=np.all, asnumpy=np.asarray, asarray=np.asarray))

    def fit(family, sample):
        calls["fit"] = (family, sample)
        return {"r": 2.0, "p": 0.4, "shape": 2.0, "scale": 3.0,
                "converged": True, "iterations": 1, "log_likelihood": -10.0}

    def values(family, points, params):
        calls["value_points"] = points
        names = QUANTITIES if family == "negative_binomial" else QUANTITIES[2:]
        return {name: np.full(len(points), 0.25) for name in names}

    def statistic(family, sample, params, statistic, *, support, remainder_bound):
        calls["support"] = support
        calls["remainder_bound"] = remainder_bound
        return 123.0

    monkeypatch.setattr(runner.cuda_candidate, "fit", fit)
    monkeypatch.setattr(runner.cuda_candidate, "distribution_values", values)
    monkeypatch.setattr(runner.cuda_candidate, "candidate_statistic", statistic)
    return calls


def test_exact_failure_reconstruction_and_extended_tail(exact_failure):
    cell, sample, reference, support = exact_failure
    assert cell.canonical_id + "|raw_outer=2|raw_inner=7" == IDENTITY
    assert hashlib.sha256(sample.tobytes()).hexdigest() == DIGEST
    assert max(sample) == 9
    assert support.support_stop == 169 > max(sample)
    assert support.certified
    points = tuple(range(int(max(sample)) + 1))
    assert reference["evaluation_points"] == points
    assert set(points) < set(support.indices)
    assert set(reference["distribution_values"]) == set(QUANTITIES)
    assert all(len(v) == len(points) for v in reference["distribution_values"].values())


def test_exact_failure_cuda_routes_value_grid_and_full_support_separately(exact_failure, monkeypatch):
    cell, sample, reference, support = exact_failure
    calls = mock_candidate(monkeypatch)
    # A spy proves both adapters independently invoke the same frozen rule.
    helper = runner.canonical_distribution_value_points
    grid_calls = []

    def grid(c, s):
        grid_calls.append((c, s))
        return helper(c, s)

    monkeypatch.setattr(runner, "canonical_distribution_value_points", grid)
    cpu = runner.evaluate_reference_record(cell, sample)
    full_support = support.indices
    cuda = runner.evaluate_cuda_record(cell, sample, certified_support=full_support,
                                       remainder_bound=support.remainder_bound)
    assert len(grid_calls) == 2
    assert all(c is cell and s is sample for c, s in grid_calls)
    assert calls["fit"][1] is sample
    assert cuda["classification"] == "ELIGIBLE"
    assert cpu["evaluation_points"] == cuda["evaluation_points"] == tuple(range(10))
    assert calls["value_points"] == reference["evaluation_points"]
    assert calls["support"] is full_support
    assert calls["support"] != calls["value_points"]
    assert calls["support"] is not calls["value_points"]
    assert calls["support"][-1] == 169
    assert calls["remainder_bound"] == support.remainder_bound
    assert cuda["statistic"] == 123.0
    assert all(len(v) == 10 for v in cuda["distribution_values"].values())


@pytest.mark.parametrize("stop", [10, 25, 169])
def test_tail_extension_does_not_change_value_grid(exact_failure, monkeypatch, stop):
    cell, sample, _, support = exact_failure
    calls = mock_candidate(monkeypatch)
    indices = tuple(range(stop + 1))
    record = runner.evaluate_cuda_record(cell, sample, certified_support=indices,
                                         remainder_bound=support.remainder_bound)
    assert record["classification"] == "ELIGIBLE"
    assert calls["value_points"] == tuple(range(10))
    assert calls["support"] is indices


@pytest.mark.parametrize("support", [None, ()])
def test_missing_statistic_support_still_fails_closed(exact_failure, monkeypatch, support):
    cell, sample, _, _ = exact_failure
    calls = mock_candidate(monkeypatch)
    record = runner.evaluate_cuda_record(cell, sample, certified_support=support)
    assert record["classification"] == "FAILED"
    assert record["failure_reason"] == "uncertified NB tail"
    assert "value_points" not in calls and "support" not in calls


@pytest.mark.parametrize("family", ["gamma", "exponential"])
def test_continuous_sorted_unique_semantics_preserved(family, monkeypatch):
    cell = SimpleNamespace(family=family, statistic="AD")
    sample = np.asarray([3.0, 1.0, 2.0, 1.0, 4.0])
    calls = mock_candidate(monkeypatch)
    cpu = runner.evaluate_reference_record(cell, sample)
    cuda = runner.evaluate_cuda_record(cell, sample)
    assert cuda["classification"] == "ELIGIBLE"
    assert cpu["evaluation_points"] == cuda["evaluation_points"] == (1., 2., 3., 4.)
    assert calls["value_points"] == (1., 2., 3., 4.)
    assert calls["support"] is None


def fake_record(points):
    return {"classification": "ELIGIBLE", "parameters": {"r": 2.0, "p": .4},
            "log_likelihood": -10., "reference_log_likelihood": lambda _: -10.,
            "statistic": .2, "evaluation_points": points,
            "distribution_values": {name: [.25] * len(points) for name in QUANTITIES}}


def compare(cpu, cuda):
    cell = SimpleNamespace(family="negative_binomial", canonical_id=CELL_ID, n=20, statistic="AD")
    return runner.evaluate_fixed_record(identity=IDENTITY, record_type="bootstrap", cell=cell,
        raw_outer_index=2, raw_inner_index=7, sample=np.asarray([0, 1, 2]),
        reference_adapter=lambda *_: cpu, cuda_adapter=lambda *_: cuda)


@pytest.mark.parametrize("cuda_points", [(0, 1, 3), (2, 1, 0), (0, 1, 2, 3)])
def test_different_point_vectors_fail_before_numeric_comparison(monkeypatch, cuda_points):
    def forbidden(*args):
        pytest.fail("mismatched grids must not reach numerical comparison")
    monkeypatch.setattr(runner, "distribution_value_agreement", forbidden)
    record = compare(fake_record((0, 1, 2)), fake_record(cuda_points))
    assert not record["distribution_value_gate_pass"]
    assert len(record["distribution_evidence"]) == 6
    for item in record["distribution_evidence"]:
        assert item["failure_reason"] == "EVALUATION_POINT_IDENTITY_MISMATCH"
        assert item["evaluation_point"] is item["cpu_value"] is item["cuda_value"] is None
        assert item["abs_error"] == float("inf") and not item["passed"]


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_each_quantity_gate_is_required_and_numeric_failure_is_distinct(quantity):
    cpu, cuda = fake_record((0, 1, 2)), fake_record((0, 1, 2))
    cuda["distribution_values"][quantity][1] += 1e-8
    record = compare(cpu, cuda)
    assert not record["distribution_value_gate_pass"]
    failures = [e for e in record["distribution_evidence"] if not e["passed"]]
    assert len(failures) == 1
    assert failures[0]["quantity"] == quantity and failures[0]["evaluation_point"] == 1
    assert failures[0].get("failure_reason") is None
    assert failures[0]["allowed_tolerance"] == max(5e-13, 5e-11 * .25)


@pytest.mark.parametrize("left", [0., 1e-4, .25, -20.])
@pytest.mark.parametrize("factor,passed", [(0.5, True), (2., False)])
def test_dec016_value_tolerance_unchanged(left, factor, passed):
    cpu, cuda = fake_record((0,)), fake_record((0,))
    tolerance = max(5e-13, 5e-11 * abs(left))
    cpu["distribution_values"]["cdf"] = [left]
    cuda["distribution_values"]["cdf"] = [left + factor * tolerance]
    record = compare(cpu, cuda)
    assert record["distribution_value_gate_pass"] is passed
    item = next(e for e in record["distribution_evidence"] if e["quantity"] == "cdf")
    assert item["allowed_tolerance"] == tolerance


@pytest.mark.parametrize("length", [0, 2, 4])
def test_values_must_cover_exact_point_vector_without_zip_truncation(length):
    cpu, cuda = fake_record((0, 1, 2)), fake_record((0, 1, 2))
    cpu["distribution_values"]["cdf"] = [.25] * length
    cuda["distribution_values"]["cdf"] = [.25] * length
    record = compare(cpu, cuda)
    assert not record["distribution_value_gate_pass"]
    item = next(e for e in record["distribution_evidence"] if e["quantity"] == "cdf")
    assert item["failure_reason"] == "DISTRIBUTION_VALUE_LENGTH_MISMATCH"


def test_identical_points_compare_all_six_quantities_and_empty_evidence_fails():
    record = compare(fake_record((0, 1, 2)), fake_record([0, 1, 2]))
    assert record["distribution_value_gate_pass"]
    assert len(record["distribution_evidence"]) == 18
    assert {e["quantity"] for e in record["distribution_evidence"]} == set(QUANTITIES)
    assert not compare(fake_record(()), fake_record(()))["distribution_value_gate_pass"]
