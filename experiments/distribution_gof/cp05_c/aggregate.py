"""Per-cell null aggregation only; no method, B, power or holdout selection."""
from __future__ import annotations

from ..artifacts import ArtifactIntegrityError
from .artifacts import validate_bundle
from .manifest import null_matrix, BASE_SHA, BLOCKED_STAGES


def aggregate(directories, *, source_sha=BASE_SHA):
    expected = {m.canonical_cell_id: m for m in null_matrix(source_sha=source_sha)}
    actual = {}
    for directory in directories:
        manifest, cell = validate_bundle(directory)
        identity = manifest.canonical_cell_id
        if identity in actual:
            raise ArtifactIntegrityError("duplicate null cell")
        if identity not in expected or manifest.source_sha != source_sha:
            raise ArtifactIntegrityError("unexpected null cell/source SHA")
        if cell["software_fixture"]:
            raise ArtifactIntegrityError("software fixtures are excluded from scientific aggregation")
        actual[identity] = cell
    if set(actual) != set(expected):
        raise ArtifactIntegrityError(f"missing null cells: {len(set(expected) - set(actual))}")
    cells = list(actual.values())
    primary = [c for c in cells if c["primary"]]
    complete = sum(c["cell_status"] == "COMPLETE" for c in cells)
    failed = sum(c["cell_status"] in ("FAILED", "FAILED_SHADOW_ORACLE") for c in cells)
    return {"EXPECTED_CONFIGURATIONS": len(expected), "COMPLETE_CONFIGURATIONS": complete,
            "FAILED_CONFIGURATIONS": failed, "INCOMPLETE_CONFIGURATIONS": len(cells) - complete - failed,
            "PRIMARY_CONFIGURATIONS": len(primary), "COMPARATOR_CONFIGURATIONS": len(cells) - len(primary),
            "ALL_PRIMARY_NULL_CELLS_WILSON_PASS": all(c["cell_status"] == "COMPLETE" and c["CELL_ACCEPTED"] for c in primary),
            "SHADOW_GATE_PASS": all(c["SHADOW_GATE"] == "PASS" for c in cells if expected[c["canonical_cell_id"]].nb_composite),
            "TOTAL_ELIGIBLE_OUTERS": sum(c["eligible_outer_count"] for c in cells),
            "TOTAL_RAW_OUTERS": sum(c["raw_outer_attempts"] for c in cells),
            "TOTAL_INNER_ATTEMPTS": sum(c["inner_attempts"] for c in cells), **BLOCKED_STAGES}
