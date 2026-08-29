"""Validate the pyMagicStats knowledge registry using only the standard library."""

from __future__ import annotations

import json
import sys
from pathlib import Path


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


def validate() -> list[str]:
    errors: list[str] = []
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read registry: {exc}"]

    roles = set(registry.get("roles", []))
    records = registry.get("records", [])
    if registry.get("canonical_branch") != "main":
        errors.append("canonical_branch must be main")
    if not roles:
        errors.append("roles must not be empty")
    if not isinstance(records, list):
        return errors + ["records must be a list"]

    ids: set[str] = set()
    for position, record in enumerate(records, start=1):
        label = record.get("id", f"record #{position}")
        missing = REQUIRED_RECORD_KEYS - set(record)
        if missing:
            errors.append(f"{label}: missing keys {sorted(missing)}")
            continue
        if label in ids:
            errors.append(f"{label}: duplicate id")
        ids.add(label)
        if record["kind"] not in ALLOWED_KINDS:
            errors.append(f"{label}: invalid kind {record['kind']}")
        if record["status"] not in ALLOWED_STATUSES:
            errors.append(f"{label}: invalid status {record['status']}")
        if record["owner_role"] not in roles:
            errors.append(f"{label}: unknown owner role {record['owner_role']}")
        reviewers = record["reviewer_roles"]
        unknown_reviewers = set(reviewers) - roles
        if unknown_reviewers:
            errors.append(f"{label}: unknown reviewer roles {sorted(unknown_reviewers)}")
        if record["status"] in {"accepted", "validated_with_limits"} and not reviewers:
            errors.append(f"{label}: accepted/validated records require a reviewer")
        if record["owner_role"] in reviewers and len(set(reviewers)) == 1:
            errors.append(f"{label}: the owner cannot be the only reviewer")
        for relative_path in [record["path"], *record["evidence_paths"]]:
            path = ROOT / relative_path
            if not path.exists():
                errors.append(f"{label}: missing repository path {relative_path}")

    for record in records:
        for superseded_id in record.get("supersedes", []):
            if superseded_id not in ids:
                errors.append(f"{record['id']}: supersedes unknown id {superseded_id}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "knowledge/README.md" not in readme:
        errors.append("root README must link to knowledge/README.md")
    return errors


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

