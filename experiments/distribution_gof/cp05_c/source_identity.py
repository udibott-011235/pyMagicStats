"""Committed research source identity for future, separately authorized runs."""
from __future__ import annotations

from pathlib import Path
import subprocess

from .manifest import BASE_SHA, BASE_TREE


ROOT = Path(__file__).resolve().parents[3]


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise ValueError("CP05-C source identity Git check failed: " + " ".join(args))
    return result.stdout.strip()


def current_source_sha():
    return git("rev-parse", "HEAD")


def repository_identity(source_sha):
    if source_sha != current_source_sha():
        raise ValueError("scientific manifest source_sha must identify the running committed checkout")
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise ValueError("CP05-C checkout root mismatch")
    if git("rev-parse", BASE_SHA + "^{tree}") != BASE_TREE:
        raise ValueError("CP05-C base tree mismatch")
    git("merge-base", "--is-ancestor", BASE_SHA, "HEAD")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("scientific execution requires a clean committed checkout")
    for line in git("diff", "--no-renames", "--name-status", BASE_SHA, "HEAD").splitlines():
        status, path = line.split("\t")
        if status != "A" or not (path.startswith("experiments/distribution_gof/cp05_c/")
                                  or path.startswith("tests/research/")):
            raise ValueError("source change outside additive CP05-C research scope: " + path)
    git("ls-files", "--error-unmatch", "experiments/distribution_gof/cp05_c/source_identity.py")
    return {"BASE_SHA": BASE_SHA, "BASE_TREE": BASE_TREE, "CANDIDATE_SHA": source_sha,
            "CANDIDATE_TREE": git("rev-parse", "HEAD^{tree}")}
