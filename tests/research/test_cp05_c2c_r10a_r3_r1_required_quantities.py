"""Independent DEC-021 completeness probes; no CUDA execution or QA verdict."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.distribution_gof.cuda_calibration import cp05_c2c_equivalence_runner as runner


NB_REQUIRED = ("pmf", "logPMF", "cdf", "sf", "logCDF", "logSF")
CONTINUOUS_REQUIRED = ("cdf", "sf", "logCDF", "logSF")
ENGINES = ("CPU", "CUDA", "BOTH")


def record(family):
    # Expected contract is independently stated, not obtained from the helper.
    quantities = NB_REQUIRED if family == "negative_binomial" else CONTINUOUS_REQUIRED
    parameters = {"r": 2., "p": .4} if family == "negative_binomial" else {"shape": 2., "scale": 1.}
    return {"classification": "ELIGIBLE", "parameters": parameters,
            "log_likelihood": -10., "reference_log_likelihood": lambda _: -10.,
            "statistic": .2, "evaluation_points": (0, 1, 2),
            "distribution_values": {q: [.25, .5, .75] for q in quantities}}


def selected_records(cpu, cuda, engine):
    return [cpu, cuda] if engine == "BOTH" else [cpu if engine == "CPU" else cuda]


def compare(family, cpu, cuda):
    cell = SimpleNamespace(family=family, canonical_id="DEC021-probe", n=3, statistic="AD")
    return runner.evaluate_fixed_record(identity="DEC021-probe", record_type="observed",
        cell=cell, raw_outer_index=0, raw_inner_index=None, sample=np.asarray([0, 1, 2]),
        reference_adapter=lambda *_: cpu, cuda_adapter=lambda *_: cuda)


def forbid_numerical_comparison(monkeypatch):
    def forbidden(*args):
        pytest.fail("structural failure must precede every numerical comparison")
    monkeypatch.setattr(runner, "distribution_value_agreement", forbidden)


def assert_structural(item, reason):
    assert item["failure_reason"] == reason
    assert item["passed"] is False
    assert item["evaluation_point"] is None
    assert item["cpu_value"] is None and item["cuda_value"] is None
    assert item["allowed_tolerance"] is None
    assert item["abs_error"] == float("inf")


@pytest.mark.parametrize("family,expected", [
    ("negative_binomial", NB_REQUIRED),
    ("gamma", CONTINUOUS_REQUIRED),
    ("exponential", CONTINUOUS_REQUIRED),
])
def test_canonical_required_quantities_exact(family, expected):
    assert runner.required_distribution_quantities(family) == expected
    assert isinstance(runner.required_distribution_quantities(family), tuple)


def test_unknown_family_is_not_inferred_from_adapter_keys():
    with pytest.raises(runner.C2CError, match="unknown distribution-value family"):
        compare("unknown", record("gamma"), record("gamma"))


@pytest.mark.parametrize("quantity", NB_REQUIRED)
@pytest.mark.parametrize("engine", ENGINES)
def test_each_nb_required_quantity_missing_fails_closed(quantity, engine, monkeypatch):
    cpu, cuda = record("negative_binomial"), record("negative_binomial")
    for item in selected_records(cpu, cuda, engine):
        del item["distribution_values"][quantity]
    forbid_numerical_comparison(monkeypatch)
    result = compare("negative_binomial", cpu, cuda)
    assert result["distribution_value_gate_pass"] is False
    evidence = result["distribution_evidence"]
    assert len(evidence) == 1 and evidence[0]["quantity"] == quantity
    assert_structural(evidence[0], f"MISSING_{engine}_DISTRIBUTION_QUANTITY")


@pytest.mark.parametrize("family", ["gamma", "exponential"])
@pytest.mark.parametrize("quantity", CONTINUOUS_REQUIRED)
@pytest.mark.parametrize("engine", ENGINES)
def test_each_continuous_required_quantity_missing_fails_closed(family, quantity, engine, monkeypatch):
    cpu, cuda = record(family), record(family)
    for item in selected_records(cpu, cuda, engine):
        del item["distribution_values"][quantity]
    forbid_numerical_comparison(monkeypatch)
    result = compare(family, cpu, cuda)
    assert result["distribution_value_gate_pass"] is False
    assert_structural(result["distribution_evidence"][0], f"MISSING_{engine}_DISTRIBUTION_QUANTITY")


@pytest.mark.parametrize("family", ["negative_binomial", "gamma", "exponential"])
@pytest.mark.parametrize("engine", ENGINES)
def test_unexpected_quantity_fails_closed_with_engine_evidence(family, engine, monkeypatch):
    cpu, cuda = record(family), record(family)
    for item in selected_records(cpu, cuda, engine):
        item["distribution_values"]["foo"] = [.25, .5, .75]
    forbid_numerical_comparison(monkeypatch)
    result = compare(family, cpu, cuda)
    assert result["distribution_value_gate_pass"] is False
    evidence = result["distribution_evidence"]
    assert len(evidence) == 1 and evidence[0]["quantity"] == "foo"
    assert_structural(evidence[0], "UNEXPECTED_DISTRIBUTION_QUANTITY")
    assert evidence[0]["engines"] == (["CPU", "CUDA"] if engine == "BOTH" else [engine])


@pytest.mark.parametrize("family", ["negative_binomial", "gamma", "exponential"])
@pytest.mark.parametrize("engine", ENGINES)
def test_absent_adapter_mapping_reports_every_required_quantity(family, engine, monkeypatch):
    cpu, cuda = record(family), record(family)
    for item in selected_records(cpu, cuda, engine):
        del item["distribution_values"]
    forbid_numerical_comparison(monkeypatch)
    result = compare(family, cpu, cuda)
    required = NB_REQUIRED if family == "negative_binomial" else CONTINUOUS_REQUIRED
    assert result["distribution_value_gate_pass"] is False
    assert tuple(e["quantity"] for e in result["distribution_evidence"]) == required
    for item in result["distribution_evidence"]:
        assert_structural(item, f"MISSING_{engine}_DISTRIBUTION_QUANTITY")


@pytest.mark.parametrize("family", ["negative_binomial", "gamma", "exponential"])
def test_complete_quantities_compare_in_canonical_not_adapter_order(family):
    cpu, cuda = record(family), record(family)
    cpu["distribution_values"] = dict(reversed(tuple(cpu["distribution_values"].items())))
    result = compare(family, cpu, cuda)
    required = NB_REQUIRED if family == "negative_binomial" else CONTINUOUS_REQUIRED
    assert result["distribution_value_gate_pass"] is True
    assert tuple(e["quantity"] for e in result["distribution_evidence"]) == tuple(q for q in required for _ in range(3))


@pytest.mark.parametrize("failure", ["quantity", "points", "length", "numeric", None])
def test_nonempty_evidence_is_insufficient_without_every_gate(failure, monkeypatch):
    cpu, cuda = record("negative_binomial"), record("negative_binomial")
    if failure == "quantity":
        del cpu["distribution_values"]["logSF"]
    elif failure == "points":
        cuda["evaluation_points"] = (0, 1, 3)
    elif failure == "length":
        cuda["distribution_values"]["logSF"] = [.25, .5]
    elif failure == "numeric":
        cuda["distribution_values"]["logSF"][2] = .8
    if failure in {"quantity", "points", "length"}:
        forbid_numerical_comparison(monkeypatch)
    result = compare("negative_binomial", cpu, cuda)
    assert len(result["distribution_evidence"]) > 0
    assert result["distribution_value_gate_pass"] is (failure is None)


def test_point_failure_precedes_and_preserves_completeness_evidence(monkeypatch):
    cpu, cuda = record("negative_binomial"), record("negative_binomial")
    cuda["evaluation_points"] = (2, 1, 0)
    del cpu["distribution_values"]["pmf"]
    del cuda["distribution_values"]["pmf"]
    cuda["distribution_values"]["foo"] = [.25, .5, .75]
    forbid_numerical_comparison(monkeypatch)
    result = compare("negative_binomial", cpu, cuda)
    assert result["distribution_value_gate_pass"] is False
    evidence = result["distribution_evidence"]
    assert [e["failure_reason"] for e in evidence[:6]] == ["EVALUATION_POINT_IDENTITY_MISMATCH"] * 6
    assert_structural(evidence[6], "MISSING_BOTH_DISTRIBUTION_QUANTITY")
    assert_structural(evidence[7], "UNEXPECTED_DISTRIBUTION_QUANTITY")
