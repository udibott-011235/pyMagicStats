"""Shared provenance and persistence primitives for CP06-G/H.

G/H are deliberately additive to the frozen C-F harness.  Every executable
entry point calls :func:`verify_frozen_sources` before reading evidence or
writing an artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Iterable

import mpmath
import numpy
import pandas
import pyarrow
import scipy
import statsmodels


REPO = "udibott-011235/pyMagicStats"
SOURCE_CF_HARNESS_SHA = "c87c6126135e300958e13d088aaef0643b28d645"
PRODUCTION_CANDIDATE_SHA = "fb3ecc6252e8c631596b7b975e683360dcde4ae4"
CP04_DOCUMENT_SHA = "63eaaed6842e2f82473bfa857524645123f95218"

GH_EXPERIMENT_VERSION = "proportion-ci-cp06-gh-v3"
GH_SCHEMA_VERSION = "cp06-gh-schema-v3"
G_SELECTION_SCHEMA_VERSION = "cp06-g-selection-schema-v3"
G_MC_SCHEMA_VERSION = "cp06-g-mc-schema-v3"
H_DESIGN_SCHEMA_VERSION = "cp06-h-design-schema-v3"
H_EVALUATION_SCHEMA_VERSION = "cp06-h-evaluation-schema-v3"

SOURCE_FILES = (
    "experiments/proportion_ci_calibration/harness.py",
    "experiments/proportion_ci_calibration/run.py",
    "experiments/proportion_ci_calibration/high_precision.py",
)
GH_EXECUTABLE_FILES = (
    "experiments/proportion_ci_calibration/acceptance.py",
    "experiments/proportion_ci_calibration/gh.py",
    "experiments/proportion_ci_calibration/gh_common.py",
    "experiments/proportion_ci_calibration/holdout.py",
    "experiments/proportion_ci_calibration/shadow_mc.py",
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class SourceIntegrityError(RuntimeError):
    """Raised when G/H is not running over the frozen C-F source."""


def _git(repo_root: Path, *arguments: str, text: bool = True) -> str | bytes:
    command = ["git", "-c", f"safe.directory={repo_root.as_posix()}", "-C", str(repo_root)]
    command.extend(arguments)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=text,
    )
    if completed.returncode:
        stderr = completed.stderr.strip() if text else completed.stderr.decode(errors="replace").strip()
        raise SourceIntegrityError(f"git {' '.join(arguments)} failed: {stderr}")
    return completed.stdout.strip() if text else completed.stdout


def runtime_head(repo_root: Path = REPOSITORY_ROOT) -> str:
    return str(_git(Path(repo_root), "rev-parse", "HEAD"))


def verify_frozen_sources(repo_root: Path = REPOSITORY_ROOT) -> dict[str, object]:
    """Fail closed unless C-F sources and production are exactly frozen.

    The Git objects at ``HEAD`` and the checked-out bytes must both match the
    source commit.  Production paths must still match the frozen production
    candidate, including staged and unstaged state.
    """

    repo_root = Path(repo_root).resolve()
    head = runtime_head(repo_root)
    ancestor = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.as_posix()}",
            "-C",
            str(repo_root),
            "merge-base",
            "--is-ancestor",
            SOURCE_CF_HARNESS_SHA,
            head,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode:
        raise SourceIntegrityError(
            f"runtime HEAD {head} is not descended from {SOURCE_CF_HARNESS_SHA}"
        )

    source_hashes: dict[str, str] = {}
    for path in SOURCE_FILES:
        expected = str(_git(repo_root, "rev-parse", f"{SOURCE_CF_HARNESS_SHA}:{path}"))
        head_blob = str(_git(repo_root, "rev-parse", f"HEAD:{path}"))
        worktree_blob = str(_git(repo_root, "hash-object", "--", path))
        if head_blob != expected or worktree_blob != expected:
            raise SourceIntegrityError(
                f"frozen C-F source mismatch for {path}: "
                f"source={expected}, HEAD={head_blob}, worktree={worktree_blob}"
            )
        source_hashes[path] = expected

    production_delta = str(
        _git(
            repo_root,
            "diff",
            "--name-only",
            f"{PRODUCTION_CANDIDATE_SHA}..HEAD",
            "--",
            "pyMagicStat",
        )
    )
    dirty_production = str(
        _git(repo_root, "status", "--porcelain", "--", "pyMagicStat")
    )
    if production_delta or dirty_production:
        raise SourceIntegrityError(
            "production paths differ from the frozen production candidate"
        )
    dirty_gh = str(
        _git(repo_root, "status", "--porcelain", "--", *GH_EXECUTABLE_FILES)
    )
    if dirty_gh:
        raise SourceIntegrityError(
            "G/H executable sources are dirty; commit and regenerate phase artifacts"
        )
    runtime_hashes: dict[str, str] = {}
    for path in GH_EXECUTABLE_FILES:
        head_blob = str(_git(repo_root, "rev-parse", f"HEAD:{path}"))
        worktree_blob = str(_git(repo_root, "hash-object", "--", path))
        if head_blob != worktree_blob:
            raise SourceIntegrityError(f"G/H executable source mismatch for {path}")
        runtime_hashes[path] = head_blob
    return {
        "runtime_head": head,
        "source_cf_harness_sha": SOURCE_CF_HARNESS_SHA,
        "production_candidate_sha": PRODUCTION_CANDIDATE_SHA,
        "cp04_document_sha": CP04_DOCUMENT_SHA,
        "source_blob_sha1": source_hashes,
        "runtime_executable_blob_sha1": runtime_hashes,
        "production_unchanged": True,
    }


def canonical_cell_id(method: str, n: int, alpha: float, p: float) -> str:
    return "|".join(
        (str(method), str(int(n)), float(alpha).hex(), float(p).hex())
    )


def stable_digest(*parts: object) -> bytes:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def stable_rank(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pandas.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def metadata_base(
    *,
    command: Iterable[str],
    workers: int,
    master_seed: str | None,
    output_counts: dict[str, int],
    exclusions: list[str] | None = None,
    errors: list[str] | None = None,
    repo_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    integrity = verify_frozen_sources(repo_root)
    return {
        "repo": REPO,
        **integrity,
        "experiment_version": GH_EXPERIMENT_VERSION,
        "schema_version": GH_SCHEMA_VERSION,
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "mpmath": mpmath.__version__,
            "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "os": platform.platform(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "backend": "CPU/NumPy/SciPy float64 with mpmath HP audit",
        "exact_command": " ".join(str(part) for part in command),
        "workers": int(workers),
        "master_seed": master_seed,
        "output_counts": {key: int(value) for key, value in output_counts.items()},
        "exclusions": list(exclusions or []),
        "errors": list(errors or []),
    }


def metadata_content_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def seal_metadata(payload: dict[str, object]) -> dict[str, object]:
    sealed = dict(payload)
    sealed.pop("metadata_content_sha256", None)
    sealed["metadata_content_sha256"] = metadata_content_hash(sealed)
    return sealed


def verify_metadata_content(payload: dict[str, object]) -> None:
    claimed = payload.get("metadata_content_sha256")
    unsealed = dict(payload)
    unsealed.pop("metadata_content_sha256", None)
    if not isinstance(claimed, str) or claimed != metadata_content_hash(unsealed):
        raise ValueError("metadata content SHA-256 does not match")
