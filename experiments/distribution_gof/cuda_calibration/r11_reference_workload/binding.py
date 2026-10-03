"""Explicit builder identity binding; never silently replace an existing binding."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .builder import artifact_document
from .codec import require
from .contract import SCIENTIFIC_SHA, SCIENTIFIC_TREE
from .schema import load_artifact, validate_binding


def binding_from_git(repository: Path):
    """Read-only identity preflight for a future clean, committed builder."""
    def git(*args):
        result = subprocess.run(["git", *args], cwd=repository, check=True,
                                capture_output=True, text=True, encoding="utf-8")
        return result.stdout.strip()
    require(not git("status", "--porcelain=v1"), "builder worktree must be clean")
    require(git("rev-parse", SCIENTIFIC_SHA + "^{tree}") == SCIENTIFIC_TREE,
            "scientific tree mismatch")
    original = set(git("ls-tree", "-r", "--name-only", SCIENTIFIC_SHA,
                       "--", "experiments", "pyMagicStat").splitlines())
    changed = set(git("diff", "--name-only", SCIENTIFIC_SHA, "HEAD",
                      "--", "experiments", "pyMagicStat").splitlines())
    require(not original.intersection(changed), "preexisting scientific code changed")
    binding = {"sha": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}")}
    validate_binding(binding)
    return binding


def bind_artifact(artifact_bytes: bytes, binding: dict, *, allow_synthetic=False):
    """Create new canonical bytes from an unbound artifact; no in-place edits."""
    validate_binding(binding)
    workload = load_artifact(artifact_bytes, allow_synthetic=allow_synthetic,
                             require_binding=False, require_complete=False)
    require(workload["builder_binding"] is None, "existing builder binding is immutable")
    workload["builder_binding"] = dict(binding)
    return artifact_document(workload)
