"""Atomic, identity-guarded runtime checkpoints for the CP05-C2C runner."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


class CheckpointError(RuntimeError):
    pass


SCHEMA_VERSION = "cp05-c2c-checkpoint-v1"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


class ExecutionCheckpoint:
    """A single self-contained file updated after every canonical work identity."""

    def __init__(self, directory: Path, contract: dict, data: dict):
        self.directory = directory
        self.contract = contract
        self.data = data

    @property
    def path(self) -> Path:
        return self.directory / "checkpoint.json"

    @classmethod
    def open(cls, output: Path, contract: dict, *, resume: bool) -> "ExecutionCheckpoint":
        directory = output.with_name(output.name + ".checkpoint")
        if output.exists():
            raise CheckpointError("output already exists; refusing overwrite")
        if directory.exists() and not resume:
            raise CheckpointError("checkpoint exists; use --resume")
        if not directory.exists():
            if resume:
                raise CheckpointError("checkpoint missing for --resume")
            data = {"schema_version": SCHEMA_VERSION, "contract": contract,
                    "primary": {}, "adversarial": {}, "generator": {}, "batch_invariance": None}
            result = cls(directory, contract, data)
            result.save()
            return result
        try:
            data = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError("corrupt checkpoint") from exc
        if data.get("schema_version") != SCHEMA_VERSION or data.get("contract") != contract:
            raise CheckpointError("RESUME_IDENTITY_MISMATCH")
        return cls(directory, contract, data)

    def save(self) -> None:
        _atomic_json(self.path, self.data)

    def has_primary(self, identity: str) -> bool:
        return identity in self.data["primary"]

    def store_primary(self, identity: str, record: dict) -> None:
        if identity in self.data["primary"]:
            raise CheckpointError("duplicate checkpoint primary identity")
        self.data["primary"][identity] = record
        self.save()

    def store_adversarial(self, name: str, record: dict) -> None:
        if name not in self.data["adversarial"]:
            self.data["adversarial"][name] = record
            self.save()

    def store_generator(self, identity: str, record: dict) -> None:
        if identity not in self.data["generator"]:
            self.data["generator"][identity] = record
            self.save()

    def store_batch(self, record: dict) -> None:
        self.data["batch_invariance"] = record
        self.save()

    def close_after_publication(self) -> None:
        shutil.rmtree(self.directory)
