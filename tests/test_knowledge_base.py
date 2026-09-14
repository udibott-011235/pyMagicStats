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
    "feature/distribution-family-framework-cp01",
    "docs/distribution-family-framework-cp01-post-merge",
    "feature/distribution-family-framework-cp02-continuous-core",
    "docs/distribution-family-framework-cp02-post-merge",
    "feature/distribution-family-framework-cp03-discrete-core",
    "docs/distribution-family-framework-cp03-post-merge",
    "feature/distribution-family-framework-cp04-wave1-fitting",
    "docs/distribution-family-framework-cp04-post-merge",
    "feature/distribution-family-framework-cp05-gof-calibration",
    "docs/distribution-family-framework-cp05-a-post-merge",
    "feature/distribution-family-framework-cp05-b-harness",
    "docs/close-cp05-b-governance",
    "docs/cp05-c-holdout-commitment",
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
    "BR-017": ("archived", "fully_contained", "merged"),
    "BR-018": ("under_review", "same_head", "pending"),
    "BR-019": ("archived", "fully_contained", "merged"),
    "BR-020": ("archived", "fully_contained", "merged"),
    "BR-021": ("archived", "fully_contained", "merged"),
    "BR-022": ("archived", "fully_contained", "merged"),
    "BR-023": ("archived", "fully_contained", "merged"),
    "BR-024": ("archived", "fully_contained", "merged"),
    "BR-025": ("archived", "fully_contained", "merged"),
    "BR-026": ("archived", "fully_contained", "merged"),
    "BR-027": ("archived", "fully_contained", "merged"),
    "BR-028": ("archived", "fully_contained", "merged"),
    "BR-029": ("under_review", "same_head", "pending"),
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
    assert registry["knowledge_base_version"] == "1.3.0"
    assert schema["properties"]["schema_version"]["const"] == "1.1.0"
    assert schema["properties"]["knowledge_base_version"]["const"] == "1.3.0"
    assert "branch" in item_schema["properties"]["kind"]["enum"]
    assert item_schema["allOf"][0]["then"] == {"required": ["branch"]}
    assert item_schema["allOf"][0]["else"] == {"not": {"required": ["branch"]}}


def test_registry_has_unique_ids_and_exactly_the_governed_branches():
    registry = _registry()
    ids = [record["id"] for record in registry["records"]]
    branches = [record for record in registry["records"] if record["kind"] == "branch"]
    branch_names = [record["branch"]["name"] for record in branches]

    assert len(ids) == len(set(ids))
    assert len(branches) == len(EXPECTED_BRANCH_NAMES)
    assert {record["id"] for record in branches} == set(EXPECTED_LIFECYCLE)
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
    assert main["branch"]["head_sha_at_decision"] == "c8df1bdab55aabf10e048e31aed61fd0d09cb5f6"
    assert main["branch"]["merge_base"] == main["branch"]["head_sha_at_decision"]
    assert (main["branch"]["ahead_of_main"], main["branch"]["behind_main"]) == (0, 0)
    evidence_path = "knowledge/evidence/distribution-family-framework-cp04-evidence.md"
    assert _record(registry, "EV-012")["status"] == "accepted"
    assert _record(registry, "EV-012")["path"] == evidence_path
    for record_id in ("DEC-009", "DEC-013", "BR-001"):
        assert evidence_path in _record(registry, record_id)["evidence_paths"]
    cp04 = _record(registry, "BR-023")["branch"]
    assert cp04["head_sha_at_decision"] == "6e92ef20aca375878964321596ba525539433f79"
    assert cp04["pr_number"] == 12
    assert cp04["merged_via"] == "PR #12 / merge commit 2b6e1263b8489592030b0838cd3851f193fbfd7f"
    assert (cp04["ahead_of_main"], cp04["behind_main"]) == (0, 1)
    closure = _record(registry, "BR-024")["branch"]
    assert closure["parent_branch"] == "main"
    assert closure["parent_sha"] == "2b6e1263b8489592030b0838cd3851f193fbfd7f"
    assert closure["head_sha_at_decision"] == "9da985d1770ac2ec6bb542e2233d6e882e56d1c2"
    assert closure["merge_base"] == closure["head_sha_at_decision"]
    assert (closure["ahead_of_main"], closure["behind_main"]) == (0, 1)
    assert closure["unique_commits"] == []
    assert closure["pr_number"] == 13
    assert closure["merged_via"] == "PR #13 / merge commit 6409717ebdfdd34d41d41c36983dda82de935e6b"
    cp05a = _record(registry, "BR-025")["branch"]
    assert cp05a["parent_branch"] == "main"
    assert cp05a["parent_sha"] == "6409717ebdfdd34d41d41c36983dda82de935e6b"
    assert cp05a["head_sha_at_decision"] == "2caf234cf1bfa8c66dd0317986803ff443ca3194"
    assert cp05a["merge_base"] == cp05a["head_sha_at_decision"]
    assert (cp05a["ahead_of_main"], cp05a["behind_main"]) == (0, 1)
    assert cp05a["unique_commits"] == []
    assert cp05a["pr_number"] == 14
    assert cp05a["merged_via"] == "PR #14 / merge commit 3d9db61cf7414ce7fe3d94819b5f9e005fff527f"
    cp05a_closure = _record(registry, "BR-026")["branch"]
    assert cp05a_closure["parent_branch"] == "main"
    assert cp05a_closure["parent_sha"] == "3d9db61cf7414ce7fe3d94819b5f9e005fff527f"
    assert cp05a_closure["head_sha_at_decision"] == "ded24ff19bc0ab515bdd9a3442879d005356f867"
    assert cp05a_closure["merge_base"] == cp05a_closure["head_sha_at_decision"]
    assert (cp05a_closure["ahead_of_main"], cp05a_closure["behind_main"]) == (0, 1)
    assert cp05a_closure["unique_commits"] == []
    assert cp05a_closure["pr_number"] == 15
    assert cp05a_closure["merged_via"] == "PR #15 / merge commit 5eb179be578594aa900a29bf5ae2f5540e05ffa2"
    cp05b = _record(registry, "BR-027")["branch"]
    assert cp05b["parent_branch"] == "main"
    assert cp05b["parent_sha"] == "5eb179be578594aa900a29bf5ae2f5540e05ffa2"
    assert cp05b["head_sha_at_decision"] == "75529e4415558c1abef6166432ebbafafd00a812"
    assert cp05b["merge_base"] == cp05b["head_sha_at_decision"]
    assert (cp05b["ahead_of_main"], cp05b["behind_main"]) == (0, 1)
    assert cp05b["unique_commits"] == []
    assert cp05b["pr_number"] == 16
    assert cp05b["merged_via"] == "PR #16 / merge commit 9fac41a38ed6583356b0e305a856dca7a3096530"
    cp05b_closure = _record(registry, "BR-028")["branch"]
    assert cp05b_closure["parent_branch"] == "main"
    assert cp05b_closure["parent_sha"] == "9fac41a38ed6583356b0e305a856dca7a3096530"
    assert cp05b_closure["head_sha_at_decision"] == "00c48edcb107089a314a43e55f31c50b68bf303e"
    assert cp05b_closure["merge_base"] == cp05b_closure["head_sha_at_decision"]
    assert (cp05b_closure["ahead_of_main"], cp05b_closure["behind_main"]) == (0, 1)
    assert cp05b_closure["unique_commits"] == []
    assert cp05b_closure["merged_via"] == (
        "PR #17 / merge commit c8df1bdab55aabf10e048e31aed61fd0d09cb5f6"
    )
    assert cp05b_closure["pr_number"] == 17
    cp05c0 = _record(registry, "BR-029")["branch"]
    assert cp05c0["parent_branch"] == "main"
    assert cp05c0["parent_sha"] == main["branch"]["head_sha_at_decision"]
    assert cp05c0["head_sha_at_decision"] == cp05c0["parent_sha"]
    assert cp05c0["merge_base"] == cp05c0["parent_sha"]
    assert (cp05c0["ahead_of_main"], cp05c0["behind_main"]) == (0, 0)
    assert cp05c0["unique_commits"] == []
    assert "merged_via" not in cp05c0
    assert "pr_number" not in cp05c0
    assert _record(registry, "DEC-014")["status"] == "accepted"
    assert _record(registry, "EV-013")["status"] == "accepted"
    assert _record(registry, "EV-014")["status"] == "accepted"
    assert _record(registry, "EV-015")["status"] == "accepted"
    assert _record(registry, "DEC-014")["path"] == (
        "knowledge/decisions/distribution-family-framework-cp05-contract.md"
    )
    assert _record(registry, "EV-013")["path"] == (
        "knowledge/evidence/distribution-family-framework-cp05-preregistration.md"
    )
    assert _record(registry, "EV-014")["path"] == (
        "knowledge/evidence/distribution-family-framework-cp05-preregistration.md"
    )
    decision = _record(registry, "DEC-014")
    evidence = _record(registry, "EV-013")
    assert decision["owner_role"] == "statistical-software-architecture"
    assert evidence["owner_role"] == "implementation-engineering"
    assert "adversarial-statistical-qa" in decision["reviewer_roles"]
    assert "adversarial-statistical-qa" in evidence["reviewer_roles"]
    assert "ROLE_DRIFT" in evidence["scope"]
    assert "17bf06639ad84a18c26865b46cdecf26dc3ab9ed" in evidence["scope"]
    assert "draft excluded from candidate genealogy" in evidence["scope"]
    assert "BR-029" in cp05b_closure["next_action"]
    assert "separate authorization" in cp05c0["next_action"]
    assert "No push, PR, merge, simulation, calibration, method selection or holdout access" in cp05c0["next_action"]
    evidence_text = (ROOT / evidence["path"]).read_text(encoding="utf-8")
    assert "noncanonical_draft_in_genealogy = false" in evidence_text
    assert "TECHNICAL_DESIGN_DISPOSITION=USABLE" in evidence_text
    assert "VERDICT=FAIL_DO_NOT_MERGE" in evidence_text
    assert "CP05-A = IN_PROGRESS" in evidence_text
    for checkpoint in ("CP05-B", "CP05-C", "CP05-D"):
        assert f"{checkpoint} = NOT_STARTED" in evidence_text
    closure = evidence_text.split("## CP05-B software harness closure", 1)[1]
    for state in (
        "CP05_A_INTEGRATION=COMPLETE",
        "CP05_A_GOVERNANCE=CLOSED", "CP05_A_OVERALL=COMPLETE",
        "CP05_B_INTEGRATION=COMPLETE", "CP05_B_GOVERNANCE=CLOSED",
        "CP05_B_OVERALL=COMPLETE", "CP05_OVERALL=IN_PROGRESS",
        "CP05_C=NOT_STARTED", "CP05_D=NOT_STARTED", "CP06_CP08=NOT_STARTED",
        "CP05_B_CLAIM_SCOPE=SOFTWARE_CORRECTNESS_ONLY",
        "CALIBRATION_VALIDATED=NO", "TYPE_I_CONTROL_VALIDATED=NO",
        "POWER_VALIDATED=NO", "METHOD_SELECTED=NO",
        "PRODUCTION_SUITABILITY_VALIDATED=NO", "HOLDOUT_ACCESSED=NO",
        "HOLDOUT_SECRET_GENERATED=NO",
    ):
        assert state in closure
    for evidence_item in (
        "CP05_B_CANDIDATE_SHA=75529e4415558c1abef6166432ebbafafd00a812",
        "CP05_B_MERGE_SHA=9fac41a38ed6583356b0e305a856dca7a3096530",
        "CP05_B_MERGE_TREE=71f7b72fedee0fd4dbcdd1e41f208d53056fdb2f",
        "ADVERSARIAL_AUDIT=PASS", "FINDING_CP05B_001=CLOSED",
        "RESEARCH_TESTS=43 passed", "CP04_REGRESSION=313 passed",
        "DISTRIBUTION_REGRESSION=888 passed",
        "PRODUCTION_REGRESSION=1128 passed, 3 inherited skips",
        "REGISTRY_VALIDATOR=PASS", "COMPILEALL=PASS", "DIFF_CHECK=PASS",
    ):
        assert evidence_item in closure
    commitment = evidence_text.split("## CP05-C0 holdout commitment registration", 1)[1]
    for state in (
        "CP05_D_NAMESPACE_COMMITMENT_SHA256=0d15aa19ff174fba06e3b06817e288b78e6168d4a775061cd4766e94c3c1896b",
        "COMMITMENT_STATUS=DEPOSITED", "HOLDOUT_COMMITMENT=DEPOSITED",
        "SECRET_STORED_OFF_REPO=YES",
        "SECRET_DISCLOSED=NO", "SECRET_ACCESSED_BY_CORTEX=NO",
        "HOLDOUT_SECRET_ACCESSED=NO", "HOLDOUT_SECRET_DISCLOSED=NO",
        "HOLDOUT_EXECUTED=NO", "CP05_C=AUTHORIZED_TO_START",
        "CP05_C_EXECUTION=NOT_STARTED", "CP05_D=NOT_STARTED",
        "CP05_OVERALL=IN_PROGRESS", "R_PREFLIGHT_EXECUTED=NO",
        "CP05_C_SIMULATION_EXECUTED=NO", "CALIBRATION_EXECUTED=NO",
        "POWER_ANALYSIS_EXECUTED=NO", "METHOD_SELECTED=NO",
    ):
        assert state in commitment
    namespace_assignments = [
        line for line in commitment.splitlines() if line.startswith("CP05_D_NAMESPACE")
    ]
    assert namespace_assignments == [
        "CP05_D_NAMESPACE_COMMITMENT_SHA256=0d15aa19ff174fba06e3b06817e288b78e6168d4a775061cd4766e94c3c1896b"
    ]
    assert "PENDING_OWNER" not in commitment
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
