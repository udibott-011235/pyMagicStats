"""Small deterministic CP05-C2A developer fixtures; these are not equivalence QA."""

import hashlib
import json
import math

import numpy as np
import pytest

from experiments.distribution_gof.cuda_calibration.cp05_cuda_engine import (
    ARTIFACT_SCHEMA_VERSION,
    ENGINE_ID,
    EngineContractError,
    EngineRequest,
    artifact_metadata,
    build_parser,
    continuous_ad,
    continuous_cvm,
    derive_seed,
    mc_pvalue,
    nb_discrete_ad,
    nb_discrete_cvm,
    nb_eligibility,
    run_composite_batches,
    write_artifact_bundle,
)


def test_seed_identity_is_deterministic_and_index_sensitive():
    first = derive_seed("ns", "cell", 2, "bootstrap", 3)
    assert first == derive_seed("ns", "cell", 2, "bootstrap", 3)
    assert first != derive_seed("ns", "cell", 2, "bootstrap", 4)


def test_continuous_primitives_match_hand_formula_without_clipping():
    u = np.array([0.2, 0.6, 0.9])
    ranks = (2 * np.arange(1, 4) - 1) / 6
    assert continuous_cvm(u) == pytest.approx(1 / 36 + np.sum((u - ranks) ** 2))
    expected = -3 - np.sum((2 * np.arange(1, 4) - 1) * (np.log(u) + np.log1p(-u[::-1]))) / 3
    assert continuous_ad(np.log(u), np.log1p(-u[::-1])) == pytest.approx(expected)
    with pytest.raises(EngineContractError):
        continuous_ad(np.array([0.0]), np.array([-math.inf]))


def test_nb_primitives_and_eligibility_contract():
    p = np.array([0.4, 0.3, 0.2])
    h = np.cumsum(p)
    sample = np.array([0, 1, 1])
    z = np.array([1 - 3 * h[0], 3 - 3 * h[1], 3 - 3 * h[2]])
    assert nb_discrete_cvm(p, h, sample) == pytest.approx(np.sum(z ** 2 * p) / 3)
    assert nb_discrete_ad(p, h, sample) == pytest.approx(np.sum(z ** 2 * p / (h * (1 - h))) / 3)
    assert nb_eligibility([0, 0, 0]) == (False, "ALL_ZERO_NON_IDENTIFYING")
    assert nb_eligibility([1, 2, 3]) == (False, "VARIANCE_NOT_GREATER_THAN_MEAN")


def test_nb_statistics_accept_leading_batch_dimensions():
    p = np.array([0.4, 0.3, 0.2])
    h = np.cumsum(p)
    samples = np.array([[0, 1, 1], [0, 0, 2]])
    cvm = nb_discrete_cvm(p, h, samples)
    ad = nb_discrete_ad(p, h, samples)
    assert cvm.shape == (2,)
    assert ad.shape == (2,)
    assert cvm[0] == pytest.approx(nb_discrete_cvm(p, h, samples[0]))
    assert ad[1] == pytest.approx(nb_discrete_ad(p, h, samples[1]))


def test_mc_plus_one_ties_and_metadata_are_non_claiming():
    assert mc_pvalue(2.0, [1.0, 2.0, 3.0]) == (2, 0.75)
    request = EngineRequest("exponential", {"scale": 1.0}, 8, "AD", "composite", 9, 1, 1, 3, "ns")
    metadata = artifact_metadata(request)
    assert metadata["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert metadata["engine"] == ENGINE_ID
    assert metadata["production_engine"] is False
    assert metadata["equivalence_gate_passed"] is False
    assert metadata["calibration_claim"] is False


def test_artifact_bundle_schema_and_digests_when_parquet_backend_available(tmp_path):
    pytest.importorskip("pyarrow")
    request = EngineRequest("exponential", {"scale": 1.0}, 8, "AD", "composite", 9, 1, 1, 3, "ns")
    output = tmp_path / "bundle"
    write_artifact_bundle(output, request, [{"raw_outer_index": 0, "p_mc": 0.5}])
    expected = {"manifest.json", "results.parquet", "accounting.json", "environment.json", "summary.json", "digests.json"}
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine"] == ENGINE_ID
    assert manifest["production_engine"] is False
    digests = json.loads((output / "digests.json").read_text(encoding="utf-8"))
    for name, digest in digests.items():
        assert digest == hashlib.sha256((output / name).read_bytes()).hexdigest()


def test_cli_defaults_are_small_and_output_is_required():
    parser = build_parser()
    args = parser.parse_args(["--output", "out"])
    assert (args.B, args.R, args.outer_batch_size, args.bootstrap_batch_size) == (9, 1, 1, 3)
    assert args.null_type == "composite"
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_composite_refits_every_bootstrap_and_is_batch_invariant(monkeypatch):
    calls = []

    class Bound:
        def __init__(self, marker):
            self.parameters = type("Parameters", (), {"scale": marker})()

    class Fit:
        def __init__(self, marker):
            self.fitted_distribution = Bound(marker)

    def generate(family, parameters, n, seed):
        return np.array([seed % 7 + 1.0] * n)

    def fit(family, sample):
        calls.append(tuple(sample))
        return Fit(float(sample[0]))

    def statistic(sample, bound, family, statistic):
        return float(sample[0] + bound.parameters.scale)

    monkeypatch.setattr("experiments.distribution_gof.cuda_calibration.cp05_cuda_engine._generate", generate)
    monkeypatch.setattr("experiments.distribution_gof.cuda_calibration.cp05_cuda_engine._cp04_fit", fit)
    monkeypatch.setattr("experiments.distribution_gof.cuda_calibration.cp05_cuda_engine._cp04_statistic", statistic)
    monkeypatch.setattr("experiments.distribution_gof.cuda_calibration.cp05_cuda_engine._parameters", lambda bound: {"scale": bound.parameters.scale})
    common = dict(family="exponential", parameters={"scale": 1.0}, n=2, statistic="AD", null_type="composite", B=3, R=3, seed_namespace="ns")
    one = run_composite_batches(EngineRequest(**common, outer_batch_size=1, bootstrap_batch_size=1), "cell")
    many = run_composite_batches(EngineRequest(**common, outer_batch_size=3, bootstrap_batch_size=3), "cell")
    assert one == many
    assert [item.raw_outer_index for item in one] == [0, 1, 2]
    assert all(item.observed_fit_calls == 1 and item.bootstrap_fit_calls == 3 for item in one)
    assert len(calls) == 2 * 3 * (1 + 3)
