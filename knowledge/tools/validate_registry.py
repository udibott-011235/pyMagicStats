"""Validate the pyMagicStats knowledge registry using only the standard library."""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "knowledge" / "registry.json"

ALLOWED_KINDS = {
    "theory",
    "paper",
    "dataset",
    "experiment",
    "evidence",
    "decision",
    "debt",
    "role-note",
    "branch",
}
ALLOWED_STATUSES = {
    "proposed",
    "under_review",
    "accepted",
    "validated_with_limits",
    "open",
    "blocked",
    "rejected",
    "superseded",
    "archived",
}
ALLOWED_BRANCH_RELATIONS = {
    "canonical",
    "same_head",
    "fully_contained",
    "contains_main",
    "diverged",
}
ALLOWED_INTEGRATION_STATES = {
    "not_applicable",
    "pending",
    "merge_candidate",
    "merged",
    "not_planned",
}
REQUIRED_RECORD_KEYS = {
    "id",
    "kind",
    "title",
    "status",
    "scope",
    "path",
    "evidence_paths",
    "owner_role",
    "reviewer_roles",
    "supersedes",
    "tags",
}
REQUIRED_BRANCH_KEYS = {
    "name",
    "head_sha_at_decision",
    "observed_at",
    "relation_to_main",
    "integration_state",
    "ahead_of_main",
    "behind_main",
    "unique_commits",
    "decision_reason",
}
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
ID_RE = re.compile(r"^[A-Z]+-[0-9]{3}$")


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _supersession_cycles(records_by_id: dict[str, dict[str, Any]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(record_id: str) -> None:
        if record_id in active_set:
            start = active.index(record_id)
            cycles.append([*active[start:], record_id])
            return
        if record_id in visited:
            return
        visited.add(record_id)
        active.append(record_id)
        active_set.add(record_id)
        for target in records_by_id[record_id].get("supersedes", []):
            if target in records_by_id:
                visit(target)
        active.pop()
        active_set.remove(record_id)

    for record_id in records_by_id:
        visit(record_id)
    return cycles


def validate_registry(registry: dict[str, Any], root: Path = ROOT) -> list[str]:
    """Return all registry consistency errors without mutating the input."""

    errors: list[str] = []
    if registry.get("schema_version") != "1.1.0":
        errors.append("schema_version must be 1.1.0")
    if registry.get("knowledge_base_version") != "1.2.0":
        errors.append("knowledge_base_version must be 1.2.0")
    if registry.get("canonical_branch") != "main":
        errors.append("canonical_branch must be main")

    roles_value = registry.get("roles", [])
    if not isinstance(roles_value, list) or not roles_value:
        errors.append("roles must be a non-empty list")
        roles: set[str] = set()
    else:
        roles = set(roles_value)

    records = registry.get("records", [])
    if not isinstance(records, list):
        return errors + ["records must be a list"]

    ids: set[str] = set()
    branch_names: set[str] = set()
    records_by_id: dict[str, dict[str, Any]] = {}
    canonical_branch_records: list[str] = []

    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            errors.append(f"record #{position}: record must be an object")
            continue
        label = record.get("id", f"record #{position}")
        missing = REQUIRED_RECORD_KEYS - set(record)
        if missing:
            errors.append(f"{label}: missing keys {sorted(missing)}")
            continue

        if not isinstance(label, str) or ID_RE.fullmatch(label) is None:
            errors.append(f"{label}: invalid id")
        if label in ids:
            errors.append(f"{label}: duplicate id")
        ids.add(label)
        records_by_id[label] = record

        kind = record["kind"]
        status = record["status"]
        if kind not in ALLOWED_KINDS:
            errors.append(f"{label}: invalid kind {kind}")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{label}: invalid status {status}")
        if record["owner_role"] not in roles:
            errors.append(f"{label}: unknown owner role {record['owner_role']}")

        reviewers = record["reviewer_roles"]
        if not isinstance(reviewers, list):
            errors.append(f"{label}: reviewer_roles must be a list")
            reviewers = []
        unknown_reviewers = set(reviewers) - roles
        if unknown_reviewers:
            errors.append(f"{label}: unknown reviewer roles {sorted(unknown_reviewers)}")
        if status in {"accepted", "validated_with_limits"} and not reviewers:
            errors.append(f"{label}: accepted/validated records require a reviewer")
        if record["owner_role"] in reviewers and len(set(reviewers)) == 1:
            errors.append(f"{label}: the owner cannot be the only reviewer")

        relative_paths = [record["path"], *record["evidence_paths"]]
        for relative_path in relative_paths:
            if not isinstance(relative_path, str):
                errors.append(f"{label}: repository path must be a string")
                continue
            if not (root / relative_path).exists():
                errors.append(f"{label}: missing repository path {relative_path}")

        has_branch = "branch" in record
        if kind == "branch" and not has_branch:
            errors.append(f"{label}: branch object is required for kind branch")
        if kind != "branch" and has_branch:
            errors.append(f"{label}: branch object is forbidden for kind {kind}")
        if kind != "branch" or not has_branch:
            continue

        branch = record["branch"]
        if not isinstance(branch, dict):
            errors.append(f"{label}: branch must be an object")
            continue
        missing_branch_keys = REQUIRED_BRANCH_KEYS - set(branch)
        if missing_branch_keys:
            errors.append(f"{label}: branch missing keys {sorted(missing_branch_keys)}")
            continue

        branch_name = branch["name"]
        if not isinstance(branch_name, str) or not branch_name:
            errors.append(f"{label}: invalid branch name")
        elif branch_name in branch_names:
            errors.append(f"{label}: duplicate branch name {branch_name}")
        else:
            branch_names.add(branch_name)

        if not _valid_sha(branch["head_sha_at_decision"]):
            errors.append(f"{label}: invalid head_sha_at_decision")
        for optional_sha in ("merge_base", "parent_sha"):
            if optional_sha in branch and not _valid_sha(branch[optional_sha]):
                errors.append(f"{label}: invalid {optional_sha}")

        try:
            date.fromisoformat(branch["observed_at"])
        except (TypeError, ValueError):
            errors.append(f"{label}: observed_at must be an ISO date")
        if branch["relation_to_main"] not in ALLOWED_BRANCH_RELATIONS:
            errors.append(f"{label}: invalid relation_to_main {branch['relation_to_main']}")
        if branch["integration_state"] not in ALLOWED_INTEGRATION_STATES:
            errors.append(f"{label}: invalid integration_state {branch['integration_state']}")

        for count_key in ("ahead_of_main", "behind_main"):
            if not _nonnegative_integer(branch[count_key]):
                errors.append(f"{label}: {count_key} must be a nonnegative integer")
        unique_commits = branch["unique_commits"]
        if not isinstance(unique_commits, list):
            errors.append(f"{label}: unique_commits must be a list")
        else:
            if len(unique_commits) != len(set(unique_commits)):
                errors.append(f"{label}: unique_commits must be unique")
            for commit in unique_commits:
                if not _valid_sha(commit):
                    errors.append(f"{label}: invalid unique commit SHA {commit}")
            if _nonnegative_integer(branch["ahead_of_main"]) and len(unique_commits) != branch["ahead_of_main"]:
                errors.append(f"{label}: unique_commits count must equal ahead_of_main")

        if not isinstance(branch["decision_reason"], str) or not branch["decision_reason"]:
            errors.append(f"{label}: decision_reason must be non-empty")
        if "pr_number" in branch and (
            not isinstance(branch["pr_number"], int)
            or isinstance(branch["pr_number"], bool)
            or branch["pr_number"] < 1
        ):
            errors.append(f"{label}: pr_number must be a positive integer")

        integration_state = branch["integration_state"]
        if integration_state == "merged" and status == "under_review":
            errors.append(f"{label}: merged integration cannot be under_review")
        if integration_state == "merge_candidate" and status in {"archived", "superseded"}:
            errors.append(f"{label}: merge_candidate cannot be {status}")
        if branch["relation_to_main"] == "canonical":
            canonical_branch_records.append(label)
            if branch_name != "main":
                errors.append(f"{label}: canonical relation must belong to main")

    for record_id, record in records_by_id.items():
        supersedes = record.get("supersedes", [])
        if not isinstance(supersedes, list):
            errors.append(f"{record_id}: supersedes must be a list")
            continue
        if record_id in supersedes:
            errors.append(f"{record_id}: record cannot supersede itself")
        for superseded_id in supersedes:
            if superseded_id not in ids:
                errors.append(f"{record_id}: supersedes unknown id {superseded_id}")

    for cycle in _supersession_cycles(records_by_id):
        errors.append(f"supersedes cycle detected: {' -> '.join(cycle)}")

    if len(canonical_branch_records) != 1:
        errors.append("exactly one branch record must have canonical relation")

    readme = (root / "README.md").read_text(encoding="utf-8")
    if "knowledge/README.md" not in readme:
        errors.append("root README must link to knowledge/README.md")
    return errors


def validate() -> list[str]:
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read registry: {exc}"]
    return validate_registry(registry)


def main() -> int:
    errors = validate()
    if errors:
        print("Knowledge registry validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Knowledge registry validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
