"""Atomic block storage with Parquet preference and CSV.gz fallback."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import pandas as pd

from .metrics import REPLICATE_COLUMNS, REPLICATE_SCHEMA_VERSION


FORMAT_EXTENSIONS = {"parquet": "parquet", "csv.gz": "csv.gz"}


def parquet_engine() -> str | None:
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return engine
        except ImportError:
            continue
    return None


def resolve_storage_format(requested: str) -> tuple[str, str | None]:
    normalized = requested.casefold()
    if normalized not in {"auto", "parquet", "csv.gz"}:
        raise ValueError("format must be one of: auto, parquet, csv.gz")
    engine = parquet_engine()
    if normalized == "parquet" and engine is None:
        raise RuntimeError("Parquet requested but neither pyarrow nor fastparquet is installed")
    if normalized == "auto":
        return ("parquet", engine) if engine is not None else ("csv.gz", None)
    return normalized, engine if normalized == "parquet" else None


def validate_replicate_frame(frame: pd.DataFrame) -> None:
    missing = set(REPLICATE_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(REPLICATE_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"replicate schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if list(frame.columns) != list(REPLICATE_COLUMNS):
        raise ValueError("replicate columns are not in the canonical order")
    if frame.empty:
        raise ValueError("replicate block must not be empty")
    if set(frame["schema_version"].astype(str)) != {REPLICATE_SCHEMA_VERSION}:
        raise ValueError("replicate schema version mismatch")


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def _fsync_file(path: Path) -> None:
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_frame_atomic(
    path: Path,
    frame: pd.DataFrame,
    storage_format: str,
    parquet_backend: str | None,
) -> None:
    validate_replicate_frame(frame)
    temporary = _temporary_path(path)
    try:
        if storage_format == "parquet":
            frame.to_parquet(temporary, index=False, engine=parquet_backend)
        elif storage_format == "csv.gz":
            with temporary.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                        frame.to_csv(text, index=False, lineterminator="\n")
        else:
            raise ValueError(f"unsupported storage format: {storage_format}")
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_frame(path: Path, storage_format: str, parquet_backend: str | None) -> pd.DataFrame:
    if storage_format == "parquet":
        frame = pd.read_parquet(path, engine=parquet_backend or "auto")
    elif storage_format == "csv.gz":
        frame = pd.read_csv(path, compression="gzip", keep_default_na=True)
        for column in (
            "t_test_failure_reason",
            "el_test_failure_reason",
            "el_ci_failure_reason",
        ):
            frame[column] = frame[column].fillna("")
    else:
        raise ValueError(f"unsupported storage format: {storage_format}")
    frame = frame.loc[:, list(REPLICATE_COLUMNS)]
    validate_replicate_frame(frame)
    return frame


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return result
