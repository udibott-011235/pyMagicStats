"""Read-only Git and accepted-workload preflight. No numerical evaluation."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..r11_reference_workload import oracle
from ..r11_reference_workload.codec import canonical_json, require, sha256, strict_json
from .contract import (ARTIFACT_HASHES, BUILDER_PATH, BUILDER_SHA, BUILDER_TREE,
                       PACKAGE_PATH, PROJECT_ROOTS, REFERENCE_WORKLOAD_SHA256,
                       SCIENTIFIC_SHA, SCIENTIFIC_TREE, SOURCE_PATH, SCHEMA_VERSION)


def git(repository, *args):
    executable = shutil.which("git")
    require(executable is not None, "Git unavailable")
    result = subprocess.run(
        [executable, "--no-optional-locks", "-c", "core.fsmonitor=false", *args],
        cwd=repository, capture_output=True, text=True, encoding="utf-8")
    require(result.returncode == 0, "Git identity read failed: " + result.stderr.strip())
    return result.stdout.strip()


def repository_identity(repository):
    repository = Path(repository).resolve()
    require(Path(git(repository, "rev-parse", "--show-toplevel")).resolve() == repository,
            "repository root required")
    require(not git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
            "harness worktree must be clean")
    require(git(repository, "rev-parse", SCIENTIFIC_SHA + "^{tree}") == SCIENTIFIC_TREE,
            "scientific tree mismatch")
    require(git(repository, "rev-parse", BUILDER_SHA + "^{tree}") == BUILDER_TREE,
            "builder base tree mismatch")
    original = set(git(repository, "ls-tree", "-r", "--name-only", SCIENTIFIC_SHA,
                       "--", *PROJECT_ROOTS).splitlines())
    changed = set(git(repository, "diff", "--no-renames", "--name-only", SCIENTIFIC_SHA,
                      "HEAD", "--", *PROJECT_ROOTS).splitlines())
    require(not original.intersection(changed), "preexisting scientific code changed")
    require(not git(repository, "diff", "--no-renames", "--name-only", BUILDER_SHA,
                    "HEAD", "--", BUILDER_PATH), "R11 builder surface changed")
    baseline = set(git(repository, "ls-tree", "-r", "--name-only", BUILDER_SHA).splitlines())
    delta = set(git(repository, "diff", "--no-renames", "--name-only", BUILDER_SHA, "HEAD").splitlines())
    test_path = "tests/research/test_cp05_c2c_r11_decision_equivalence.py"
    require(not baseline.intersection(delta) and all(
        name.startswith(PACKAGE_PATH + "/") or name == test_path for name in delta),
        "harness diff outside authorized isolated scope")
    return {
        "R11_HARNESS_SHA": git(repository, "rev-parse", "HEAD"),
        "R11_HARNESS_TREE": git(repository, "rev-parse", "HEAD^{tree}"),
        "R11_SCIENTIFIC_BASE_SHA": SCIENTIFIC_SHA, "R11_SCIENTIFIC_BASE_TREE": SCIENTIFIC_TREE,
        "R11_BUILDER_SHA": BUILDER_SHA, "R11_BUILDER_TREE": BUILDER_TREE,
        "executing_repository": str(repository), "worktree_clean": True,
        "scientific_files_unchanged": True, "builder_surface_unchanged": True,
    }


def verify_workload(data, manifest, archive, crossings):
    require(type(data) is bytes and sha256(data) == REFERENCE_WORKLOAD_SHA256,
            "accepted R11 workload SHA256 mismatch")
    require(type(crossings) is bytes, "R4 crossings required")
    # The existing loader verifies frozen schema, every payload, source projection,
    # archive/crossings/extracted historical hashes, and the ordered R4 prefix.
    workload = oracle.load_for_use(data, manifest, archive, crossings_bytes=crossings)
    require(workload["builder_binding"] == {"sha": BUILDER_SHA, "tree": BUILDER_TREE},
            "accepted builder SHA/tree mismatch")
    return workload


@dataclass(frozen=True)
class PreparedWorkload:
    repository: Path
    workload_bytes: bytes
    body_bytes: bytes
    identity_bytes: bytes

    def workload(self):
        return strict_json(self.body_bytes, canonical=True)

    def identity(self):
        return strict_json(self.identity_bytes, canonical=True)


def prepare(workload_path, archive_path, crossings_path):
    repository = Path(__file__).resolve().parents[4]
    identity = repository_identity(repository)
    data = Path(workload_path).read_bytes()
    workload = verify_workload(
        data, (repository / SOURCE_PATH).read_bytes(), Path(archive_path).read_bytes(),
        Path(crossings_path).read_bytes())
    identity.update({
        "schema_version": SCHEMA_VERSION,
        "R11_REFERENCE_WORKLOAD_SHA256": REFERENCE_WORKLOAD_SHA256,
        "R11_REFERENCE_WORKLOAD_ACCEPTED": True, "R4_RUNTIME_ORACLE_VERIFIED": True,
        "source_manifest_sha256": workload["source_manifest_hash"],
        "source_projection_sha256": workload["source_projection_hash"],
        "R4_artifact_sha256": dict(ARTIFACT_HASHES),
    })
    return PreparedWorkload(repository, data, canonical_json(workload), canonical_json(identity))


def descriptor(outer, item, ordinal=None):
    observed = ordinal is None
    payload = item["payload"]
    return {
        "identity": item["identity"], "record_type": "observed" if observed else "bootstrap",
        "cell_id": outer["cell_id"], "raw_outer_index": outer["raw_outer_index"],
        "raw_inner_index": None if observed else item["raw_inner_index"],
        "accepted_ordinal": ordinal, "seed_identity": item["seed_identity"],
        "sample_digest": payload["sample_digest"], "payload_dtype": payload["dtype_str"],
        "payload_shape": list(payload["shape"]),
    }


def ordered_records(workload):
    """Yield only stored accepted payloads; prospective attempts are never evaluated."""
    for outer in workload["outers"]:
        yield outer, descriptor(outer, outer["observed"]), outer["observed"]["payload"]
        for ordinal, item in enumerate(outer["accepted"]):
            yield outer, descriptor(outer, item, ordinal), item["payload"]
