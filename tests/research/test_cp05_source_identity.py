from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from experiments.distribution_gof import cp05_c1_preflight
from experiments.distribution_gof.cp05_c1_preflight import (
    AUTHORIZED_BRANCH,
    CELLS,
    GitEnvironmentError,
    SourceIdentityError,
    _manifest,
    _plan_payload,
    _provenance_payload,
    _verify_source_identity,
)


GIT = shutil.which("git")


def _git(repository: Path, *args: str) -> str:
    if GIT is None:
        pytest.skip("git is required for source-identity integration tests")
    return subprocess.run(
        [GIT, *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    ).stdout.strip()


def _commit(repository: Path, text: str) -> str:
    tracked = repository / "tracked.txt"
    tracked.write_text(text, encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", text.strip())
    return _git(repository, "rev-parse", "HEAD")


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "CP05 test")
    _git(repository, "config", "user.email", "cp05-test@example.invalid")
    baseline_sha = _commit(repository, "baseline\n")
    return repository, baseline_sha


def test_baseline_head_on_authorized_branch_is_valid(tmp_path):
    repository, baseline_sha = _repository(tmp_path)
    _git(repository, "switch", "-c", AUTHORIZED_BRANCH)

    identity = _verify_source_identity(repository, baseline_sha=baseline_sha)

    assert identity["actual_head_sha"] == baseline_sha
    assert identity["baseline_ancestor_of_head"] is True
    assert identity["merge_base"] == baseline_sha
    assert identity["branch_or_detached_state"] == "AUTHORIZED_BRANCH"
    assert identity["checkout_mode"] == "AUTHORIZED_BRANCH"


def test_candidate_head_on_authorized_branch_is_valid(tmp_path):
    repository, baseline_sha = _repository(tmp_path)
    _git(repository, "switch", "-c", AUTHORIZED_BRANCH)
    candidate_sha = _commit(repository, "candidate\n")

    identity = _verify_source_identity(repository, baseline_sha=baseline_sha)

    assert identity["authorized_candidate_sha"] == candidate_sha
    assert identity["actual_head_sha"] == candidate_sha
    assert identity["baseline_merge_base"] == baseline_sha


def test_detached_candidate_reachable_from_authorized_ref_is_valid(tmp_path):
    repository, baseline_sha = _repository(tmp_path)
    _git(repository, "switch", "-c", AUTHORIZED_BRANCH)
    candidate_sha = _commit(repository, "candidate\n")
    _git(repository, "switch", "--detach", candidate_sha)

    identity = _verify_source_identity(repository, baseline_sha=baseline_sha)

    assert identity["actual_head_sha"] == candidate_sha
    assert identity["branch"] == "DETACHED"
    assert identity["branch_or_detached_state"] == "DETACHED_HEAD"
    assert identity["checkout_mode"] == "DETACHED_HEAD"
    assert identity["authorized_refs"] == [
        {"ref": f"refs/heads/{AUTHORIZED_BRANCH}", "sha": candidate_sha}
    ]


def test_baseline_not_ancestor_fails_closed(tmp_path):
    repository, baseline_sha = _repository(tmp_path)
    _git(repository, "checkout", "--orphan", "unrelated-root")
    _git(repository, "rm", "-f", "tracked.txt")
    _commit(repository, "unrelated\n")
    _git(repository, "branch", "-M", AUTHORIZED_BRANCH)

    with pytest.raises(SourceIdentityError, match="baseline is not an ancestor"):
        _verify_source_identity(repository, baseline_sha=baseline_sha)


def test_unrelated_active_branch_fails_closed(tmp_path):
    repository, baseline_sha = _repository(tmp_path)
    _git(repository, "switch", "-c", "experiments/unrelated")
    _commit(repository, "unrelated branch\n")

    with pytest.raises(SourceIdentityError, match="not the authorized work line"):
        _verify_source_identity(repository, baseline_sha=baseline_sha)


def test_detached_head_not_reachable_from_authorized_ref_fails_closed(tmp_path):
    repository, baseline_sha = _repository(tmp_path)
    _git(repository, "switch", "-c", AUTHORIZED_BRANCH)
    _git(repository, "switch", "--detach", baseline_sha)
    detached_sha = _commit(repository, "detached candidate\n")

    with pytest.raises(SourceIdentityError, match="not reachable"):
        _verify_source_identity(repository, baseline_sha=baseline_sha)

    assert detached_sha != baseline_sha


def test_detached_head_without_authorized_refs_reports_limitation(tmp_path):
    repository, baseline_sha = _repository(tmp_path)
    _git(repository, "switch", "-c", AUTHORIZED_BRANCH)
    candidate_sha = _commit(repository, "candidate\n")
    _git(repository, "switch", "--detach", candidate_sha)
    _git(repository, "branch", "-D", AUTHORIZED_BRANCH)

    with pytest.raises(
        SourceIdentityError,
        match="limitation=AUTHORIZED_BRANCH_REF_UNAVAILABLE",
    ):
        _verify_source_identity(repository, baseline_sha=baseline_sha)


def test_missing_git_is_explicitly_blocked(monkeypatch):
    monkeypatch.setattr(cp05_c1_preflight.shutil, "which", lambda name: None)

    with pytest.raises(GitEnvironmentError, match="BLOCKED_ENVIRONMENT"):
        _verify_source_identity()


def test_plan_and_manifest_record_actual_head_separately_from_baseline(tmp_path):
    repository, baseline_sha = _repository(tmp_path)
    _git(repository, "switch", "-c", AUTHORIZED_BRANCH)
    candidate_sha = _commit(repository, "evidence candidate\n")
    identity = _verify_source_identity(repository, baseline_sha=baseline_sha)

    plan = _plan_payload(identity)
    provenance = _provenance_payload(
        identity,
        16,
        executable="python",
        command="python -m experiments.distribution_gof.cp05_c1_preflight",
    )
    manifest = _manifest(CELLS[0], "AD", 199, identity["actual_head_sha"])

    assert plan["source_sha"] == candidate_sha
    assert plan["execution_source_sha"] == candidate_sha
    assert plan["actual_head_sha"] == candidate_sha
    assert plan["authorized_candidate_sha"] == candidate_sha
    assert manifest.source_sha == candidate_sha
    assert provenance["source_sha"] == candidate_sha
    assert provenance["execution_source_sha"] == candidate_sha
    assert provenance["actual_head_sha"] == candidate_sha
    assert provenance["authorized_candidate_sha"] == candidate_sha
    assert plan["baseline_main_sha"] == baseline_sha
    assert plan["baseline_sha"] == baseline_sha
    assert provenance["baseline_main_sha"] == baseline_sha
    assert provenance["baseline_sha"] == baseline_sha
    assert identity["baseline_main_sha"] == baseline_sha
