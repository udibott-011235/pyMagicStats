"""Small deterministic CP05-C2A developer fixtures; these are not equivalence QA."""

import math

import numpy as np
import pytest

from experiments.distribution_gof.cuda_calibration.cp05_cuda_engine import (
    ARTIFACT_SCHEMA_VERSION,
    ENGINE_ID,
    EngineContractError,
    EngineRequest,
    artifact_metadata,
    continuous_ad,
    continuous_cvm,
    derive_seed,
    mc_pvalue,
    nb_discrete_ad,
    nb_discrete_cvm,
    nb_eligibility,
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


def test_mc_plus_one_ties_and_metadata_are_non_claiming():
    assert mc_pvalue(2.0, [1.0, 2.0, 3.0]) == (2, 0.75)
    request = EngineRequest("exponential", {"scale": 1.0}, 8, "AD", "composite", 9, 1, 1, 3, "ns")
    metadata = artifact_metadata(request)
    assert metadata["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert metadata["engine"] == ENGINE_ID
    assert metadata["production_engine"] is False
    assert metadata["equivalence_gate_passed"] is False
    assert metadata["calibration_claim"] is False
