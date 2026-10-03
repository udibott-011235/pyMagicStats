"""Exclusive evidence publication and immutable raw-result snapshots."""
from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

from ..r11_reference_workload.codec import canonical_json, require, sha256, write_once
from .contract import EVIDENCE_NAMES


def json_safe(value):
    """Retain nonfinite numbers explicitly instead of emitting invalid JSON."""
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": value.hex()}
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), "non-string evidence key")
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    # Only host NumPy evidence may use scalar/array conversion; never device protocols.
    if type(value).__module__.split(".")[0] == "numpy":
        return json_safe(value.tolist())
    if callable(value):
        return {"opaque_callable": getattr(value, "__qualname__", type(value).__qualname__)}
    return {"opaque_type": type(value).__module__ + "." + type(value).__qualname__}


def encode(value):
    return canonical_json(json_safe(value))


def snapshot(value):
    """Detached immutable bytes, captured before any subsequent engine runs."""
    return encode(value)


class EvidenceBundle:
    """A fresh directory and exclusively-created streams; no resume/overwrite."""
    def __init__(self, output, repository):
        self.output = Path(output).resolve()
        repository = Path(repository).resolve()
        require(not self.output.is_relative_to(repository), "evidence must be outside repository")
        require(not self.output.exists(), "output exists; rerun/resume/overwrite prohibited")
        self.output.mkdir(parents=True, exist_ok=False)
        self._streams = {}
        self._published = set()
        try:
            for name in ("records.jsonl", "indicator_adjudication.jsonl"):
                self._streams[name] = (self.output / name).open("xb")
                self._published.add(name)
        except BaseException:
            self.close()
            raise

    def append(self, name, data):
        require(type(data) is bytes and b"\n" not in data, "canonical JSONL bytes required")
        stream = self._streams[name]
        stream.write(data + b"\n")
        stream.flush()
        os.fsync(stream.fileno())

    def publish(self, name, value, *, raw=False):
        require(name not in self._published and Path(name).name == name, "duplicate evidence artifact")
        data = value if raw else encode(value)
        require(type(data) is bytes, "evidence bytes required")
        write_once(self.output / name, data)
        self._published.add(name)

    def close(self):
        for stream in self._streams.values():
            if not stream.closed:
                stream.close()

    def finish_digests(self):
        self.close()
        require(set(EVIDENCE_NAMES) <= self._published, "incomplete evidence publication")
        entries = {name: {"sha256": sha256((self.output / name).read_bytes()),
                          "bytes": (self.output / name).stat().st_size}
                   for name in sorted(self._published)}
        # A digest manifest cannot contain its own hash. It covers every other artifact.
        self.publish("digests.json", {"artifacts": entries, "self_hash_excluded": "digests.json"})
