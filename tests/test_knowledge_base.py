import copy
import json
from pathlib import Path

from knowledge.tools.validate_registry import validate, validate_registry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "knowledge" / "registry.json"
SCHEMA_PATH = ROOT / "knowledge" / "schema" / "registry.schema.json"

EXPECTED_BRANCH_NAMES = {
    "main",
    "audit/global-main-a0881c4",
    "docs/project-knowledge-base",
    "experiments/el-vs-t-calibration-harness",
    "feature/anova-engine",
    "feature/empirical-likelihood-mean",
    "fix/el-ci-numerical-convergence",
    "fix/el-vs-t-calibration-accounting",
    "fix/el-vs-t-cupy-generator-compatibility",
    "fix/gate2-major-remediation",
    "fix/gate2-distribution-gof-remediation",
    "fix/gate2-adversarial-remediation",
    "refactor/distribution-shape-contract",
    "refactor/inference-capability-routing",
    "refactor/inference-engine",
    "refactor/sampling-robustness-v3",
}

EXPECTED_LIFECYCLE = {
    "BR-001": ("accepted", "canonical", "not_applicable"),
    "BR-002": ("archived", "fully_contained", "not_applicable"),
    "BR-003": ("archived", "fully_contained", "merged"),
    "BR-004": ("archived", "fully_contained", "merged"),
    "BR-005": ("under_review", "diverged", "pending"),
    "BR-006": ("archived", "fully_contained", "merged"),
    "BR-007": ("archived", "fully_contained", "merged"),
    "BR-008": ("archived", "fully_contained", "merged"),
    "BR-009": ("archived", "fully_contained", "merged"),
    "BR-010": ("superseded", "fully_contained", "not_planned"),
    "BR-011": ("superseded", "fully_contained", "merged"),
    "BR-012": ("archived", "fully_contained", "merged"),
    "BR-013": ("archived", "fully_contained", "merged"),
    "BR-014": ("archived", "fully_contained", "merged"),
    "BR-015": ("archived", "fully_contained", "merged"),
    "BR-016": ("archived", "fully_contained", "merged"),
}


def _registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _record(registry, record_id):
    return next(record for record in registry["records"] if record["id"] == record_id)


def test_knowledge_registry_is_consistent():
    assert validate() == []


def test_schema_and_registry_versions_define_conditional_branch_records():
    registry = _registry()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    item_schema = schema["properties"]["records"]["items"]

    assert registry["schema_version"] == "1.1.0"
    assert registry["knowledge_base_version"] == "1.2.0"
    assert schema["properties"]["schema_version"]["const"] == "1.1.0"
    assert schema["properties"]["knowledge_base_version"]["const"] == "1.2.0"
    assert "branch" in item_schema["properties"]["kind"]["enum"]
    assert item_schema["allOf"][0]["then"] == {"required": ["branch"]}
    assert item_schema["allOf"][0]["else"] == {"not": {"required": ["branch"]}}


def test_registry_has_unique_ids_and_exactly_the_governed_branches():
    registry = _registry()
    ids = [record["id"] for record in registry["records"]]
    branches = [record for record in registry["records"] if record["kind"] == "branch"]
    branch_names = [record["branch"]["name"] for record in branches]

    assert len(ids) == len(set(ids))
    assert len(branches) == 16
    assert len(branch_names) == len(set(branch_names))
    assert set(branch_names) == EXPECTED_BRANCH_NAMES
    assert all("branch" not in record for record in registry["records"] if record["kind"] != "branch")


def test_all_owner_architecture_lifecycle_decisions_are_exact():
    registry = _registry()

    for record_id, expected in EXPECTED_LIFECYCLE.items():
        record = _record(registry, record_id)
        observed = (
            record["status"],
            record["branch"]["relation_to_main"],
            record["branch"]["integration_state"],
        )
        assert observed == expected


def test_lifecycle_decisions_and_gate2_supersession_are_materialized_exactly():
    registry = _registry()
    main = _record(registry, "BR-001")
    knowledge = _record(registry, "BR-003")
    anova = _record(registry, "BR-005")
    gate2_placeholder = _record(registry, "BR-010")
    gate2_distribution = _record(registry, "BR-011")
    gate2_adversarial = _record(registry, "BR-012")

    assert registry["canonical_branch"] == "main"
    assert main["status"] == "accepted"
    assert main["branch"]["relation_to_main"] == "canonical"
    assert main["branch"]["head_sha_at_decision"] == "f1725ebdfebcb667c053420e4cb4c1e35048f9e0"
    assert knowledge["status"] == "archived"
    assert knowledge["branch"]["integration_state"] == "merged"
    assert knowledge["branch"]["merged_via"] == "PR #1"
    assert knowledge["branch"]["pr_number"] == 1
    assert anova["status"] == "under_review"
    assert anova["branch"]["integration_state"] == "pending"
    assert gate2_placeholder["status"] == "superseded"
    assert gate2_distribution["supersedes"] == ["BR-010"]
    assert gate2_distribution["branch"]["integration_state"] == "merged"
    assert gate2_adversarial["supersedes"] == ["BR-011"]
    assert gate2_adversarial["status"] == "archived"
    assert gate2_adversarial["branch"]["integration_state"] == "merged"
    assert gate2_adversarial["branch"]["merged_via"] == "PR #3"
    assert gate2_adversarial["branch"]["head_sha_at_decision"] == (
        "9a87c5d48dba8b8a172b5386d7318e7f37ec98fe"
    )
    assert gate2_adversarial["branch"]["ahead_of_main"] == 0


def test_validator_rejects_missing_or_forbidden_branch_objects():
    missing = _registry()
    _record(missing, "BR-002").pop("branch")
    assert "BR-002: branch object is required for kind branch" in validate_registry(missing)

    forbidden = _registry()
    _record(forbidden, "TH-001")["branch"] = copy.deepcopy(_record(forbidden, "BR-001")["branch"])
    assert "TH-001: branch object is forbidden for kind theory" in validate_registry(forbidden)


def test_validator_rejects_duplicate_names_invalid_shas_and_enums():
    registry = _registry()
    duplicate = _record(registry, "BR-002")
    duplicate["branch"]["name"] = "main"
    duplicate["branch"]["head_sha_at_decision"] = "not-a-sha"
    duplicate["branch"]["relation_to_main"] = "unknown"
    duplicate["branch"]["integration_state"] = "unknown"
    errors = validate_registry(registry)

    assert "BR-002: duplicate branch name main" in errors
    assert "BR-002: invalid head_sha_at_decision" in errors
    assert "BR-002: invalid relation_to_main unknown" in errors
    assert "BR-002: invalid integration_state unknown" in errors


def test_validator_rejects_unknown_self_and_cyclic_supersession():
    unknown = _registry()
    _record(unknown, "BR-001")["supersedes"] = ["BR-999"]
    assert "BR-001: supersedes unknown id BR-999" in validate_registry(unknown)

    self_supersession = _registry()
    _record(self_supersession, "BR-010")["supersedes"] = ["BR-010"]
    assert "BR-010: record cannot supersede itself" in validate_registry(self_supersession)

    cyclic = _registry()
    _record(cyclic, "BR-010")["supersedes"] = ["BR-012"]
    assert any(error.startswith("supersedes cycle detected:") for error in validate_registry(cyclic))


def test_validator_enforces_canonical_and_integration_constraints():
    multiple_canonical = _registry()
    _record(multiple_canonical, "BR-002")["branch"]["relation_to_main"] = "canonical"
    errors = validate_registry(multiple_canonical)
    assert "BR-002: canonical relation must belong to main" in errors
    assert "exactly one branch record must have canonical relation" in errors

    merged_under_review = _registry()
    _record(merged_under_review, "BR-005")["branch"]["integration_state"] = "merged"
    assert "BR-005: merged integration cannot be under_review" in validate_registry(merged_under_review)

    candidate_archived = _registry()
    _record(candidate_archived, "BR-002")["branch"]["integration_state"] = "merge_candidate"
    assert "BR-002: merge_candidate cannot be archived" in validate_registry(candidate_archived)

    candidate_superseded = _registry()
    _record(candidate_superseded, "BR-010")["branch"]["integration_state"] = "merge_candidate"
    assert "BR-010: merge_candidate cannot be superseded" in validate_registry(candidate_superseded)
