from __future__ import annotations

import pytest

from experiments.distribution_gof.manifest import ExperimentManifest
from experiments.distribution_gof.seed_derivation import derive_seed, numpy_rng

from ._helpers import manifest


def test_canonical_cell_id_is_decimal_stable_and_excludes_topology():
    baseline = manifest()
    changed_topology = manifest(
        canonical_parameters={"scale": 1},
        shard_spec={"shard_id": 3, "shard_count": 7},
        batch_spec={"batch_size": 31},
        worker_spec={"workers": 9},
    )
    assert baseline.canonical_cell_id == changed_topology.canonical_cell_id
    assert baseline.canonical_cell_id == (
        '{"B":199,"canonical_parameters":{"scale":"1"},'
        '"family":"exponential","n":8,"null_type":"simple",'
        '"phase":"CP05-B_NON_CALIBRATION","statistic":"CVM"}'
    )


def test_seed_known_answer_uses_first_128_bits_big_endian():
    seed = derive_seed(manifest().canonical_cell_id, 7, "inner_bootstrap", 3)
    assert seed.digest_hex == "125c7f84c0555bc6da4f0f21af617cbe6879b7255dfbf18b9a0e556cfd58bea8"
    assert seed.entropy == 24406381618775119483031309464042044606
    assert numpy_rng(seed).integers(0, 2**31) == 349986056


@pytest.mark.parametrize(
    "override",
    [
        {"B": 200},
        {"alpha": 0.051},
        {"family": "normal"},
        {"statistic": "SHAPIRO"},
        {"null_type": "unknown"},
        {"namespace_id": "secret"},
        {"phase": "CP05-D"},
        {"claim_status": "CALIBRATION"},
    ],
)
def test_manifest_rejects_non_contract_inputs(override):
    with pytest.raises(ValueError):
        manifest(**override)


def test_manifest_round_trip_and_extra_fields_fail_closed():
    original = manifest()
    assert ExperimentManifest.from_dict(original.to_dict()) == original
    payload = original.to_dict()
    payload["unregistered"] = True
    with pytest.raises(ValueError, match="fields"):
        ExperimentManifest.from_dict(payload)


def test_seed_validation_never_accepts_secret_namespace_or_negative_indices():
    cell = manifest().canonical_cell_id
    with pytest.raises(ValueError, match="development namespace"):
        derive_seed(cell, 0, "outer", namespace="CP05-D-secret")
    with pytest.raises(ValueError, match="raw_outer_index"):
        derive_seed(cell, -1, "outer")
    with pytest.raises(ValueError, match="raw_inner_index"):
        derive_seed(cell, 0, "inner", -1)
