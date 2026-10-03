"""Canonical JSON and exact ndarray payload transport, without generation."""
from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ContractError(ValueError):
    """Fail closed before any downstream use."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def strict_json(data: bytes, *, canonical=False):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ContractError("non-finite JSON constant: " + value)

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=constant)
        # Also rejects overflowed float literals such as 1e999.
        encoded = canonical_json(value)
        require(not canonical or encoded == data, "non-canonical JSON bytes")
        return value
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def keys(value, expected):
    require(type(value) is dict and set(value) == set(expected), "schema keys mismatch")


def integer(value, *, minimum=0):
    require(type(value) is int and value >= minimum, "invalid integer")
    return value


def text_identity(value):
    require(type(value) is str and bool(value) and "\0" not in value,
            "invalid identity")
    return value


def hex_digest(value, length=64):
    require(type(value) is str and len(value) == length
            and all(c in "0123456789abcdef" for c in value), "invalid digest")
    return value


def canonicalize_sample(sample: np.ndarray) -> np.ndarray:
    # Refuse array protocols / device arrays that might transfer data implicitly.
    require(type(sample) is np.ndarray, "CPU NumPy ndarray required")
    require(not sample.dtype.hasobject and sample.dtype.fields is None
            and sample.dtype.subdtype is None and sample.dtype.itemsize > 0,
            "unsupported payload dtype")
    return np.ascontiguousarray(sample)


def sample_bytes(sample: np.ndarray) -> bytes:
    return canonicalize_sample(sample).tobytes(order="C")


def sample_digest(sample: np.ndarray) -> str:
    return sha256(sample_bytes(sample))


@dataclass(frozen=True)
class SamplePayload:
    dtype_str: str
    shape: tuple[int, ...]
    raw_bytes: bytes
    sample_digest: str

    @classmethod
    def from_sample(cls, sample):
        array = canonicalize_sample(sample)
        raw = array.tobytes(order="C")
        # ascontiguousarray promotes scalars to shape (1,); retain the original
        # ndarray interpretation while hashing exactly its canonical C bytes.
        return cls(array.dtype.str, tuple(sample.shape), raw, sha256(raw))

    def verify(self):
        require(type(self.raw_bytes) is bytes and type(self.shape) is tuple,
                "payload must be immutable bytes/shape")
        require(type(self.dtype_str) is str, "invalid dtype")
        try:
            dtype = np.dtype(self.dtype_str)
        except (TypeError, ValueError) as exc:
            raise ContractError("invalid dtype") from exc
        require(dtype.str == self.dtype_str and not dtype.hasobject
                and dtype.fields is None and dtype.subdtype is None
                and dtype.itemsize > 0, "non-canonical dtype")
        for dimension in self.shape:
            integer(dimension)
        require(len(self.raw_bytes) == math.prod(self.shape) * dtype.itemsize,
                "shape/dtype/payload length mismatch")
        hex_digest(self.sample_digest)
        require(sha256(self.raw_bytes) == self.sample_digest, "sample digest mismatch")

    def deserialize(self) -> np.ndarray:
        self.verify()
        array = np.frombuffer(self.raw_bytes, dtype=np.dtype(self.dtype_str)).reshape(self.shape)
        require(sample_digest(array) == self.sample_digest, "loaded sample digest mismatch")
        return array  # bytes-backed, read-only

    def serialize(self) -> dict:
        self.verify()
        return {"dtype_str": self.dtype_str, "shape": list(self.shape),
                "payload_base64": base64.b64encode(self.raw_bytes).decode("ascii"),
                "sample_digest": self.sample_digest}

    @classmethod
    def from_serialized(cls, value):
        keys(value, ("dtype_str", "shape", "payload_base64", "sample_digest"))
        require(type(value["shape"]) is list, "invalid shape")
        require(type(value["payload_base64"]) is str, "invalid base64")
        try:
            raw = base64.b64decode(value["payload_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ContractError("invalid base64") from exc
        require(base64.b64encode(raw).decode("ascii") == value["payload_base64"],
                "non-canonical base64")
        result = cls(value["dtype_str"], tuple(value["shape"]), raw, value["sample_digest"])
        result.verify()
        return result


def write_once(path: Path, data: bytes) -> str:
    """Exclusive create; never overwrite, repair, rerun, or resume."""
    with Path(path).open("xb") as stream:
        stream.write(data)
    return sha256(data)
