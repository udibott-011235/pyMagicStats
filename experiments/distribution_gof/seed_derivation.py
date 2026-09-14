"""CP05 deterministic seed ownership independent of execution topology."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .manifest import DEVELOPMENT_NAMESPACE


SEED_DERIVATION_VERSION = "sha256-nul-utf8-first128-big-endian-v1"


@dataclass(frozen=True, slots=True)
class SeedIdentity:
    digest_hex: str
    entropy: int
    raw_outer_index: int
    purpose: str
    raw_inner_index: int | None


def derive_seed(
    canonical_cell_id: str,
    raw_outer_index: int,
    purpose: str,
    raw_inner_index: int | None = None,
    *,
    namespace: str = DEVELOPMENT_NAMESPACE,
) -> SeedIdentity:
    if namespace != DEVELOPMENT_NAMESPACE:
        raise ValueError("only the CP05 public development namespace is allowed")
    if type(raw_outer_index) is not int or raw_outer_index < 0:
        raise ValueError("raw_outer_index must be a non-negative Python int")
    if raw_inner_index is not None and (
        type(raw_inner_index) is not int or raw_inner_index < 0
    ):
        raise ValueError("raw_inner_index must be None or a non-negative Python int")
    if type(canonical_cell_id) is not str or not canonical_cell_id:
        raise ValueError("canonical_cell_id must be a nonempty string")
    if type(purpose) is not str or not purpose or "\x00" in purpose:
        raise ValueError("purpose must be a nonempty NUL-free string")
    fields = (
        namespace,
        canonical_cell_id,
        str(raw_outer_index),
        purpose,
        "" if raw_inner_index is None else str(raw_inner_index),
    )
    if any("\x00" in field for field in fields):
        raise ValueError("seed fields must not contain NUL")
    digest = hashlib.sha256("\x00".join(fields).encode("utf-8")).digest()
    return SeedIdentity(
        digest_hex=digest.hex(),
        entropy=int.from_bytes(digest[:16], "big", signed=False),
        raw_outer_index=raw_outer_index,
        purpose=purpose,
        raw_inner_index=raw_inner_index,
    )


def numpy_rng(seed: SeedIdentity) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed.entropy))
