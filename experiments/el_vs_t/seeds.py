"""Stable seed derivation and deterministic shard ownership."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterator

import numpy as np


SEED_DERIVATION_SCHEME = "blake2b-128-json-shard-independent-v2"
SEED_NAMESPACE = "pyMagicStats-el-vs-t-calibration-v1"


@dataclass(frozen=True)
class SeedMaterial:
    identity: str
    uint64: int


def derive_seed(
    master_seed: int,
    scenario_id: str,
    replicate_id: int,
) -> SeedMaterial:
    """Derive a persistent statistical seed independent of execution topology."""

    if min(int(master_seed), int(replicate_id)) < 0:
        raise ValueError("master_seed and replicate_id must be non-negative")
    payload = {
        "experiment_namespace": SEED_NAMESPACE,
        "master_seed": int(master_seed),
        "replicate_id": int(replicate_id),
        "scenario_id": str(scenario_id),
        "scheme": SEED_DERIVATION_SCHEME,
    }
    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        digest_size=16,
        person=b"pyMagicStat-ELt",
    ).digest()
    return SeedMaterial(identity=digest.hex(), uint64=int.from_bytes(digest[:8], "little"))


def numpy_rng(seed: SeedMaterial) -> np.random.Generator:
    return np.random.default_rng(seed.uint64)


def owned_replicate_ids(
    replicates_per_cell: int,
    shard_id: int,
    num_shards: int,
) -> range:
    """Return global IDs ``r`` where ``r % num_shards == shard_id``."""

    if int(replicates_per_cell) < 1:
        raise ValueError("replicates_per_cell must be positive")
    if int(num_shards) < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= int(shard_id) < int(num_shards):
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    return range(int(shard_id), int(replicates_per_cell), int(num_shards))


def replicate_blocks(replicate_ids: range, batch_size: int) -> Iterator[tuple[int, ...]]:
    """Yield bounded checkpoint blocks without materializing all IDs."""

    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    block: list[int] = []
    for replicate_id in replicate_ids:
        block.append(int(replicate_id))
        if len(block) == int(batch_size):
            yield tuple(block)
            block.clear()
    if block:
        yield tuple(block)
