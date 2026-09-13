from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from experiments.proportion_ci_calibration import acceptance, gh, gh_common
from experiments.proportion_ci_calibration.acceptance import (
    AcceptanceLocalization,
    NumericalCoverageError,
    high_precision_recheck,
    interval_for_cell,
    localize_acceptance,
    stable_binomial_coverage,
)
from experiments.proportion_ci_calibration.gh_common import (
    H_DESIGN_SCHEMA_VERSION,
    CP04_DOCUMENT_SHA,
    G_MC_SCHEMA_VERSION,
    G_SELECTION_SCHEMA_VERSION,
    GH_EXPERIMENT_VERSION,
    GH_SCHEMA_VERSION,
    PRODUCTION_CANDIDATE_SHA,
    SOURCE_CF_HARNESS_SHA,
    SourceIntegrityError,
    canonical_cell_id,
    runtime_head,
    seal_metadata,
    sha256_file,
    verify_frozen_sources,
)
from experiments.proportion_ci_calibration import holdout
from experiments.proportion_ci_calibration.holdout import (
    H_ALPHA_BASE_QUOTA,
    H_CELLS_PER_METHOD,
    H_METHODS,
    H_N_QUOTAS,
    H_P_QUOTAS,
    H_TOTAL_CELLS,
    canonical_design_hash,
    evaluate_holdout_frame,
    generate_fixture_holdout_design,
    log_uniform_integer,
    production_holdout_quota_plan,
    validate_holdout_design,
    verify_design_artifact,
)
from experiments.proportion_ci_calibration.harness import ALPHAS, PRODUCTION_METHODS
from experiments.proportion_ci_calibration import shadow_mc
from experiments.proportion_ci_calibration.shadow_mc import (
    BROAD_CELLS,
    BROAD_REPS,
    CRITICAL_CELLS,
    CRITICAL_PER_METHOD,
    CRITICAL_REPS,
    N_STRATA,
    TOTAL_DRAWS,
    DeterministicMappingError,
    attach_cf_float64_authority,
    build_g_selection,
    canonical_selection_hash,
    cell_seed_128,
    mc_gate,
    simulate_shadow_cells,
    validate_g_selection,
)
from pyMagicStat.inference import PopulationProportionCI


FIXTURE_G_SEED = "FIXTURE_ONLY:CP06-G-UNIT-v1"
FIXTURE_H_SEED = "FIXTURE_ONLY:CP06-H-UNIT-v1"


def _e_minima_fixture() -> pd.DataFrame:
    n_values = [1, 6, 11, 21, 31, 51, 101, 251, 501, 1001, 2001, 7500]
    rows = []
    index = 0
    for method in PRODUCTION_METHODS:
        for alpha in ALPHAS:
            for n in n_values:
                p = float((index % 997 + 1) / 1000.0)
                coverage = 1.0 - alpha - (index % 29) / 10_000.0
                rows.append(
                    {
                        "method": method,
                        "n": n,
                        "alpha": alpha,
                        "p": p,
                        "coverage": coverage,
                        "nominal": 1.0 - alpha,
                        "acceptance_kind": "contiguous_integer_range",
                        "first_x": 0,
                        "last_x": n,
                        "acceptance_runs": json.dumps([[0, n]]),
                    }
                )
                index += 1
    return pd.DataFrame(rows)


def _brute_coverage(method: str, n: int, alpha: float, p: float) -> tuple[float, list[int]]:
    selected = []
    for x in range(n + 1):
        lower, upper = interval_for_cell(method, n, x, alpha)
        if lower <= p <= upper:
            selected.append(x)
    probabilities = __import__("scipy").stats.binom.pmf(selected, n, p)
    return float(np.sum(probabilities)), selected


def _wald_boundary_probabilities() -> list[float]:
    lower = [
        0.0,
        float(np.nextafter(0.0, 1.0)),
        float(1e-323),
        1e-300,
        1e-12,
        1e-11,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
    ]
    values = lower + [1.0 - value for value in lower]
    values.extend([float(np.nextafter(1.0, 0.0)), 1.0])
    return sorted({value for value in values if 0.0 <= value <= 1.0})


def _acceptance_from_intervals(
    intervals: list[tuple[float, float]], p: float
) -> list[int]:
    return [
        x for x, (lower, upper) in enumerate(intervals) if lower <= p <= upper
    ]


def _selection_with_authority(*, include_f: bool = True) -> pd.DataFrame:
    selection = build_g_selection(_e_minima_fixture())
    lookup = {
        canonical_cell_id(row.method, row.n, row.alpha, row.p): float(row.coverage)
        for row in selection.itertuples(index=False)
        if row.selection_kind == "critical"
    }

    def frozen_value(method, n, alpha, p):
        identity = canonical_cell_id(method, n, alpha, p)
        return lookup.get(identity, 0.875)

    result = attach_cf_float64_authority(
        selection,
        cf_provider=frozen_value,
        independent_provider=frozen_value,
    )
    if not include_f:
        return result
    for column in (
        "coverage_hp_float",
        "resolved",
        "classification",
        "acceptance_changed",
        "acceptance_runs_float64",
        "acceptance_runs_hp",
    ):
        result[column] = None
    wilson = (result["selection_kind"] == "critical") & (
        result["method"] == "wilson"
    )
    result.loc[wilson, "coverage_hp_float"] = result.loc[
        wilson, "coverage_cf_float64"
    ]
    result.loc[wilson, "resolved"] = True
    result.loc[wilson, "classification"] = "confirmed_no_shortfall_at_audited_cell"
    result.loc[wilson, "acceptance_changed"] = False
    result.loc[wilson, "acceptance_runs_float64"] = result.loc[
        wilson, "acceptance_runs"
    ]
    result.loc[wilson, "acceptance_runs_hp"] = result.loc[
        wilson, "acceptance_runs"
    ]
    validate_g_selection(result, require_authority=True, require_f=True)
    return result


def _write_selection_artifact(
    tmp_path: Path,
    selection: pd.DataFrame,
) -> tuple[Path, Path, dict[str, object]]:
    selection_path = tmp_path / gh.G_SELECTION_NAME
    selection.to_parquet(selection_path, index=False)
    metadata = {
        "runtime_head": runtime_head(),
        "selection_schema_version": G_SELECTION_SCHEMA_VERSION,
        "experiment_version": GH_EXPERIMENT_VERSION,
        "schema_version": GH_SCHEMA_VERSION,
        "mc_schema_version": G_MC_SCHEMA_VERSION,
        "cf_harness_schema_version": "cp06-harness-schema-v4",
        "cf_shard_schema_version": "cp06-shard-schema-v4",
        "source_cf_harness_sha": SOURCE_CF_HARNESS_SHA,
        "production_candidate_sha": PRODUCTION_CANDIDATE_SHA,
        "cp04_document_sha": CP04_DOCUMENT_SHA,
        "critical_cells": CRITICAL_CELLS,
        "broad_cells": BROAD_CELLS,
        "implied_draws": TOTAL_DRAWS,
        "selection_artifact_sha256": sha256_file(selection_path),
        "canonical_selection_sha256": canonical_selection_hash(selection),
        "e_artifact_sha256": "1" * 64,
        "e_metadata_sha256": "2" * 64,
        "f_audit_sha256": "3" * 64,
        "f_metadata_sha256": "4" * 64,
    }
    metadata = seal_metadata(metadata)
    metadata_path = tmp_path / gh.G_SELECTION_METADATA_NAME
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return selection_path, metadata_path, metadata


def _write_f_audit(
    tmp_path: Path,
    selection: pd.DataFrame,
    *,
    omit_last: bool = False,
    unresolved: bool = False,
    metadata_updates: dict[str, object] | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    wilson = selection.loc[
        (selection["selection_kind"] == "critical")
        & (selection["method"] == "wilson")
    ]
    if omit_last:
        wilson = wilson.iloc[:-1]
    for source in wilson.itertuples(index=False):
        localization = localize_acceptance(
            str(source.method), int(source.n), float(source.alpha), float(source.p)
        )
        runs = json.dumps(localization.runs, separators=(",", ":"))
        rows.append(
            {
                "audit_kind": "coverage",
                "method": source.method,
                "n": int(source.n),
                "alpha": float(source.alpha),
                "p": float(source.p),
                "coverage_hp_float": float(source.coverage_cf_float64),
                "resolved": not unresolved,
                "classification": "confirmed_no_shortfall_at_audited_cell",
                "acceptance_changed": False,
                "acceptance_runs_float64": runs,
                "acceptance_runs_hp": runs,
            }
        )
    audit = pd.DataFrame(rows)
    audit_path = tmp_path / "proportion_ci_cp06_f_high_precision_audit.parquet"
    audit.to_parquet(audit_path, index=False)
    metadata = {
        "candidate_sha": PRODUCTION_CANDIDATE_SHA,
        "cp04_document_sha": CP04_DOCUMENT_SHA,
        "experiment_version": "proportion-ci-cp06-v3",
        "harness_schema_version": "cp06-harness-schema-v4",
        "source_shard_schema_version": "cp06-shard-schema-v4",
        "checkpoint": "F",
        "source_checkpoints": ["C", "D", "E"],
        "digits": 80,
        "queue_rows": len(audit),
        "audit_rows": len(audit),
        "hashes": {audit_path.name: sha256_file(audit_path)},
    }
    metadata.update(metadata_updates or {})
    metadata_path = tmp_path / "proportion_ci_cp06_f_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return audit_path, metadata_path, audit


def _write_design_artifact(
    tmp_path: Path,
    design: pd.DataFrame,
) -> tuple[Path, Path, dict[str, object]]:
    design_path = tmp_path / "design.parquet"
    design.to_parquet(design_path, index=False)
    metadata = {
        "runtime_head": runtime_head(),
        "source_cf_harness_sha": SOURCE_CF_HARNESS_SHA,
        "production_candidate_sha": PRODUCTION_CANDIDATE_SHA,
        "cp04_document_sha": CP04_DOCUMENT_SHA,
        "design_schema_version": H_DESIGN_SCHEMA_VERSION,
        "design_artifact_sha256": sha256_file(design_path),
        "canonical_design_sha256": canonical_design_hash(design),
        "cell_count": len(design),
        "master_seed": FIXTURE_H_SEED,
    }
    metadata = seal_metadata(metadata)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return design_path, metadata_path, metadata


def test_source_integrity_is_bound_to_exact_cf_commit_and_production_candidate():
    result = verify_frozen_sources()
    assert result["source_cf_harness_sha"] == gh_common.SOURCE_CF_HARNESS_SHA
    assert result["production_candidate_sha"] == gh_common.PRODUCTION_CANDIDATE_SHA
    assert result["production_unchanged"] is True
    assert set(result["source_blob_sha1"]) == set(gh_common.SOURCE_FILES)


def test_source_integrity_fails_closed_on_head_or_worktree_mismatch(monkeypatch):
    original = gh_common._git

    def mismatched(repo_root, *arguments, **kwargs):
        value = original(repo_root, *arguments, **kwargs)
        if arguments[:2] == ("hash-object", "--"):
            return "0" * 40
        return value

    monkeypatch.setattr(gh_common, "_git", mismatched)
    with pytest.raises(SourceIntegrityError, match="frozen C-F source mismatch"):
        verify_frozen_sources()


def test_source_integrity_fails_closed_on_head_blob_mismatch(monkeypatch):
    original = gh_common._git

    def mismatched(repo_root, *arguments, **kwargs):
        value = original(repo_root, *arguments, **kwargs)
        if arguments[:1] == ("rev-parse",) and str(arguments[1]).startswith("HEAD:"):
            return "f" * 40
        return value

    monkeypatch.setattr(gh_common, "_git", mismatched)
    with pytest.raises(SourceIntegrityError, match="frozen C-F source mismatch"):
        verify_frozen_sources()


def test_source_integrity_rejects_dirty_gh_executable_source(monkeypatch):
    original = gh_common._git

    def dirty(repo_root, *arguments, **kwargs):
        if arguments[:3] == ("status", "--porcelain", "--") and any(
            str(path).endswith("shadow_mc.py") for path in arguments[3:]
        ):
            return " M experiments/proportion_ci_calibration/shadow_mc.py"
        return original(repo_root, *arguments, **kwargs)

    monkeypatch.setattr(gh_common, "_git", dirty)
    with pytest.raises(SourceIntegrityError, match="executable sources are dirty"):
        verify_frozen_sources()


def test_canonical_cell_id_uses_exact_float_hex_components():
    identity = canonical_cell_id("wilson", 27, 0.05, np.nextafter(0.2, 1.0))
    assert identity == "|".join(
        ("wilson", "27", float(0.05).hex(), float(np.nextafter(0.2, 1.0)).hex())
    )


def test_g_selection_has_exact_counts_draws_and_required_coverage():
    selection = build_g_selection(_e_minima_fixture())
    validate_g_selection(selection)
    assert selection["selection_kind"].value_counts().to_dict() == {
        "broad": BROAD_CELLS,
        "critical": CRITICAL_CELLS,
    }
    assert int(selection["reps"].sum()) == TOTAL_DRAWS == 256_000_000
    critical = selection[selection["selection_kind"] == "critical"]
    for method in PRODUCTION_METHODS:
        method_rows = critical[critical["method"] == method]
        assert len(method_rows) >= CRITICAL_PER_METHOD
        assert set(method_rows["alpha"]) == set(ALPHAS)
        assert set(method_rows["n_stratum"]) == {item[0] for item in N_STRATA}
    broad = selection[selection["selection_kind"] == "broad"]
    grouped = broad.groupby(["method", "alpha", "n_stratum"]).size()
    assert len(grouped) == 3 * 7 * 12
    assert int((grouped == 3).sum()) == 8
    assert int((grouped == 2).sum()) == 244


def test_g_selection_is_invariant_to_evidence_row_order():
    source = _e_minima_fixture()
    left = build_g_selection(source)
    right = build_g_selection(source.sample(frac=1.0, random_state=701))
    assert left["canonical_cell_id"].tolist() == right["canonical_cell_id"].tolist()


def test_g_repetition_constants_imply_exact_preregistered_total():
    assert CRITICAL_CELLS * CRITICAL_REPS + BROAD_CELLS * BROAD_REPS == 256_000_000


def test_per_cell_seed_is_stable_128_bit_and_cell_specific():
    identity = canonical_cell_id("wilson", 3, 0.05, 0.1)
    one = cell_seed_128(FIXTURE_G_SEED, identity)
    two = cell_seed_128(FIXTURE_G_SEED, canonical_cell_id("wilson", 3, 0.05, 0.2))
    expected = int.from_bytes(
        hashlib.sha256(
            f"CP06-G-MC-v1|{FIXTURE_G_SEED}|{identity}".encode("utf-8")
        ).digest()[:16],
        byteorder="big",
        signed=False,
    )
    assert one == expected == cell_seed_128(FIXTURE_G_SEED, identity)
    assert 0 <= one < 2**128
    assert one != two


def test_g_select_cli_requires_both_f_inputs():
    with pytest.raises(SystemExit):
        gh.main(
            [
                "g-select",
                "--e-minima",
                "e.parquet",
                "--e-metadata",
                "e.json",
                "--output-dir",
                "out",
            ]
        )


def test_g_authority_uses_frozen_cf_for_broad_and_retains_e_for_critical():
    selection = build_g_selection(_e_minima_fixture())
    selection.loc[selection["selection_kind"] == "critical", "coverage"] = 0.91
    attached = attach_cf_float64_authority(
        selection,
        cf_provider=lambda *args: 0.91,
        independent_provider=lambda *args: 0.91,
    )
    broad = attached["selection_kind"] == "broad"
    critical = ~broad
    assert (attached.loc[broad, "coverage_cf_float64"] == 0.91).all()
    assert attached.loc[broad, "coverage_e_float64"].isna().all()
    assert (attached.loc[critical, "coverage_e_float64"] == 0.91).all()
    assert attached["cf_vs_independent_consistent"].all()
    assert attached.loc[critical, "e_vs_cf_consistent"].all()


def test_g_wrong_independent_localization_is_explicitly_blocking():
    selection = build_g_selection(_e_minima_fixture())
    selection.loc[selection["selection_kind"] == "critical", "coverage"] = 0.91
    with pytest.raises(DeterministicMappingError, match="blocking deterministic mapping"):
        attach_cf_float64_authority(
            selection,
            cf_provider=lambda *args: 0.91,
            independent_provider=lambda *args: 0.81,
        )


def test_g_critical_e_disagreement_is_explicitly_blocking():
    selection = build_g_selection(_e_minima_fixture())
    with pytest.raises(DeterministicMappingError, match="blocking deterministic mapping"):
        attach_cf_float64_authority(
            selection,
            cf_provider=lambda *args: 0.91,
            independent_provider=lambda *args: 0.91,
        )


def test_f_metadata_and_all_selected_critical_wilson_evidence_are_required(tmp_path):
    selection = _selection_with_authority(include_f=False)
    audit_path, metadata_path, audit = _write_f_audit(tmp_path, selection)
    attached = gh._attach_f_audit(selection, audit_path, metadata_path)
    selected = attached.loc[
        (attached["selection_kind"] == "critical")
        & (attached["method"] == "wilson")
    ]
    assert len(selected) == len(audit)
    assert selected["coverage_hp_float"].notna().all()
    assert selected["resolved"].astype(bool).all()

    _, wrong_metadata, _ = _write_f_audit(
        tmp_path,
        selection,
        metadata_updates={"candidate_sha": "0" * 40},
    )
    with pytest.raises(ValueError, match="F metadata mismatch for candidate_sha"):
        gh._attach_f_audit(selection, audit_path, wrong_metadata)


def test_unresolved_or_incomplete_f_audit_is_rejected(tmp_path):
    selection = _selection_with_authority(include_f=False)
    unresolved_path, unresolved_metadata, _ = _write_f_audit(
        tmp_path / "unresolved", selection, unresolved=True
    )
    with pytest.raises(ValueError, match="unresolved rows"):
        gh._attach_f_audit(selection, unresolved_path, unresolved_metadata)

    incomplete_path, incomplete_metadata, _ = _write_f_audit(
        tmp_path / "incomplete", selection, omit_last=True
    )
    with pytest.raises(ValueError, match="every selected critical Wilson cell"):
        gh._attach_f_audit(selection, incomplete_path, incomplete_metadata)


def test_float64_and_hp_mc_use_distinct_acceptance_runs_on_same_draws(monkeypatch):
    localization = AcceptanceLocalization(
        first_x=0,
        last_x=0,
        acceptance_kind="contiguous_integer_range",
        localization_method="synthetic",
        boundary_validation="synthetic",
    )
    monkeypatch.setattr(shadow_mc, "localize_acceptance", lambda *args: localization)
    identity = canonical_cell_id("wilson", 1, 0.05, 0.25)
    selection = pd.DataFrame(
        [
            {
                "method": "wilson",
                "n": 1,
                "alpha": 0.05,
                "p": 0.25,
                "selection_kind": "critical",
                "canonical_cell_id": identity,
                "coverage_cf_float64": 0.75,
                "coverage_independent_localization": 0.75,
                "coverage_hp_float": 0.25,
                "resolved": True,
                "classification": "float64_boundary_artifact",
                "acceptance_changed": True,
                "acceptance_runs_float64": "[[0,0]]",
                "acceptance_runs_hp": "[[1,1]]",
            }
        ]
    )
    result = simulate_shadow_cells(
        selection,
        master_seed=FIXTURE_G_SEED,
        workers=1,
        fixture_reps=20_000,
        verify_source=False,
    ).iloc[0]
    assert result["covered_float64"] + result["covered_hp"] == result["reps"]
    assert result["coverage_mc_float64"] != result["coverage_mc_hp"]
    assert result["float64_mc_gate_pass"]
    assert result["hp_mc_gate_pass"]


def test_g_selection_tampering_canonical_tampering_and_runtime_mismatch_are_rejected(
    tmp_path, monkeypatch
):
    selection = _selection_with_authority()
    selection_path, metadata_path, metadata = _write_selection_artifact(
        tmp_path, selection
    )
    loaded, _ = gh._verify_g_selection_artifact(selection_path, metadata_path)
    assert len(loaded) == len(selection)

    tampered = selection.copy()
    tampered.loc[tampered.index[0], "p"] = np.nextafter(
        float(tampered.loc[tampered.index[0], "p"]), 1.0
    )
    with pytest.raises(ValueError, match="canonical-cell tampering"):
        validate_g_selection(tampered)

    artifact_tampered = selection.copy()
    artifact_tampered.loc[artifact_tampered.index[0], "probe_count"] = 999
    artifact_tampered.to_parquet(selection_path, index=False)
    with pytest.raises(ValueError, match="artifact SHA-256"):
        gh._verify_g_selection_artifact(selection_path, metadata_path)

    selection.to_parquet(selection_path, index=False)
    runtime_mismatch = dict(metadata)
    runtime_mismatch["runtime_head"] = "f" * 40
    runtime_mismatch = seal_metadata(runtime_mismatch)
    metadata_path.write_text(json.dumps(runtime_mismatch), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime HEAD"):
        gh._verify_g_selection_artifact(selection_path, metadata_path)


def test_mc_gate_uses_five_se_or_point001():
    result = mc_gate(0.95, 0.949, 1_000_000)
    expected_se = math.sqrt(0.95 * 0.05 / 1_000_000)
    assert result["mc_standard_error"] == pytest.approx(expected_se)
    assert result["mc_tolerance"] == pytest.approx(max(5 * expected_se, 0.001))
    assert result["mc_gate_pass"] is True


def test_hp_gate_accounting_counts_only_explicit_governing_boolean_failures():
    passing = pd.DataFrame(
        {
            "hp_governs": [True, True, False],
            "hp_mc_gate_pass": pd.Series([True, True, None], dtype=object),
        }
    )
    failing = pd.DataFrame(
        {
            "hp_governs": [True, True, False],
            "hp_mc_gate_pass": pd.Series([True, False, None], dtype=object),
        }
    )
    assert gh._count_hp_gate_failures(passing) == 0
    assert gh._count_hp_gate_failures(failing) == 1


@pytest.mark.parametrize(
    "invalid_gate",
    [
        pytest.param(None, id="none"),
        pytest.param(np.nan, id="nan"),
        pytest.param(1, id="integer-one"),
        pytest.param(1.0, id="float-one"),
        pytest.param("True", id="string-true"),
    ],
)
def test_hp_gate_accounting_rejects_nonboolean_governing_values(invalid_gate):
    results = pd.DataFrame(
        {
            "hp_governs": [True],
            "hp_mc_gate_pass": pd.Series([invalid_gate], dtype=object),
        }
    )
    with pytest.raises(ValueError, match="explicit boolean MC gate"):
        gh._count_hp_gate_failures(results)


def test_old_hp_gate_expression_reproduces_object_dtype_inversion_defect():
    results = pd.DataFrame(
        {
            "hp_governs": [True, True, False],
            "hp_mc_gate_pass": pd.Series([True, True, None], dtype=object),
        }
    )
    filled = results["hp_mc_gate_pass"].fillna(True)
    assert filled.dtype == object
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert (~filled).tolist() == [-2, -2, -2]
        assert int((results["hp_governs"] & ~filled).sum()) == 2
    assert gh._count_hp_gate_failures(results) == 0


def _invoke_fixture_g_run(tmp_path, monkeypatch, results: pd.DataFrame) -> Path:
    selection_path = tmp_path / "selection.parquet"
    selection_metadata_path = tmp_path / "selection-metadata.json"
    selection_path.write_bytes(b"fixture selection")
    selection_metadata_path.write_text("{}", encoding="utf-8")
    selection_metadata = {
        "runtime_head": "fixture-runtime",
        "selection_artifact_sha256": "1" * 64,
        "canonical_selection_sha256": "2" * 64,
    }
    monkeypatch.setattr(gh, "verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        gh,
        "_verify_g_selection_artifact",
        lambda *args: (pd.DataFrame(), selection_metadata),
    )
    monkeypatch.setattr(gh, "simulate_shadow_cells", lambda *args, **kwargs: results)
    monkeypatch.setattr(
        gh,
        "metadata_base",
        lambda **kwargs: {"output_counts": kwargs["output_counts"]},
    )
    output_dir = tmp_path / "out"
    arguments = __import__("argparse").Namespace(
        selection=selection_path,
        selection_metadata=selection_metadata_path,
        master_seed="FIXTURE_ONLY:ACCOUNTING-v1",
        workers=1,
        output_dir=output_dir,
    )
    gh.g_run(arguments, ["fixture-g-run"])
    return output_dir


def test_g_run_metadata_records_zero_for_observed_all_true_hp_gates(
    tmp_path, monkeypatch
):
    results = pd.DataFrame(
        {
            "hp_governs": [True, True, False],
            "hp_mc_gate_pass": pd.Series([True, True, None], dtype=object),
            "float64_mc_gate_pass": [True, True, True],
        }
    )
    output_dir = _invoke_fixture_g_run(tmp_path, monkeypatch, results)
    metadata = json.loads((output_dir / gh.G_METADATA_NAME).read_text(encoding="utf-8"))
    assert metadata["output_counts"]["hp_gate_failures"] == 0


def test_g_run_fails_before_writing_metadata_for_invalid_governing_hp_gate(
    tmp_path, monkeypatch
):
    results = pd.DataFrame(
        {
            "hp_governs": [True],
            "hp_mc_gate_pass": pd.Series([None], dtype=object),
            "float64_mc_gate_pass": [True],
        }
    )
    with pytest.raises(ValueError, match="explicit boolean MC gate"):
        _invoke_fixture_g_run(tmp_path, monkeypatch, results)
    assert not (tmp_path / "out" / gh.G_METADATA_NAME).exists()


def test_shadow_rng_is_invariant_to_worker_count_and_input_order():
    cells = pd.DataFrame(
        [
            {"method": "wilson", "n": 27, "alpha": 0.05, "p": 0.2, "selection_kind": "critical"},
            {"method": "clopper_pearson", "n": 12, "alpha": 0.1, "p": 0.01, "selection_kind": "broad"},
            {"method": "wald", "n": 10, "alpha": 0.2, "p": 0.5, "selection_kind": "broad"},
        ]
    )
    cells["canonical_cell_id"] = [
        canonical_cell_id(row.method, row.n, row.alpha, row.p)
        for row in cells.itertuples()
    ]
    serial = simulate_shadow_cells(
        cells, master_seed=FIXTURE_G_SEED, workers=1, fixture_reps=2_000, verify_source=False
    )
    parallel = simulate_shadow_cells(
        cells.iloc[::-1], master_seed=FIXTURE_G_SEED, workers=2, fixture_reps=2_000, verify_source=False
    )
    pd.testing.assert_frame_equal(serial, parallel)


def test_holdout_quota_plan_is_exact_without_generating_production_design():
    plan = production_holdout_quota_plan()
    assert plan["total"] == H_TOTAL_CELLS == 10_000
    assert plan["methods"] == {method: H_CELLS_PER_METHOD for method in H_METHODS}
    assert sum(H_N_QUOTAS.values()) == H_CELLS_PER_METHOD
    assert sum(H_P_QUOTAS.values()) == H_CELLS_PER_METHOD
    assert H_ALPHA_BASE_QUOTA * len(ALPHAS) + 1 == H_CELLS_PER_METHOD


def test_production_alpha_vector_has_357_each_plus_one_seed_selected_extra():
    rng = np.random.Generator(np.random.PCG64DXSM(4004))
    values = holdout._alpha_vector(rng)
    counts = pd.Series(values).value_counts().to_dict()
    assert len(values) == H_CELLS_PER_METHOD
    assert sorted(counts.values()) == [357, 357, 357, 357, 357, 357, 358]


def test_synthetic_fixture_proves_exact_h_output_quotas(monkeypatch):
    """Exercise 10k quota assembly without revealing a real seeded design."""

    counter = iter(range(1, H_TOTAL_CELLS + 1))
    fixed_n = {
        "n_1_5000": 7,
        "n_5001_100000": 7_500,
        "n_100001_1000000": 250_000,
    }
    monkeypatch.setattr(holdout, "_sample_n", lambda rng, band: fixed_n[band])

    def synthetic_probability(rng, family, method, n, alpha):
        return next(counter) / (H_TOTAL_CELLS + 1), {"mirror": False}

    monkeypatch.setattr(holdout, "_sample_probability", synthetic_probability)
    design = holdout.generate_holdout_design("FIXTURE_ONLY:SYNTHETIC-QUOTAS-v1")
    assert len(design) == H_TOTAL_CELLS
    assert design["canonical_cell_id"].is_unique
    assert design.groupby("method").size().to_dict() == {
        method: H_CELLS_PER_METHOD for method in H_METHODS
    }
    for method in H_METHODS:
        selected = design[design["method"] == method]
        assert selected["n_band"].value_counts().to_dict() == H_N_QUOTAS
        assert selected["p_family"].value_counts().to_dict() == H_P_QUOTAS
        assert sorted(selected["alpha"].value_counts().tolist()) == [
            357,
            357,
            357,
            357,
            357,
            357,
            358,
        ]
    altered = design.copy()
    altered.loc[altered.index[0], "n_band"] = "n_5001_100000"
    with pytest.raises(ValueError, match="n quotas|n-band labels"):
        validate_holdout_design(altered)


@pytest.mark.parametrize("lower,upper", [(1, 5_000), (5_001, 100_000), (100_001, 1_000_000)])
def test_log_uniform_integer_stays_inside_inclusive_bounds(lower, upper):
    rng = np.random.Generator(np.random.PCG64DXSM(9917 + lower))
    values = [log_uniform_integer(rng, lower, upper) for _ in range(1_000)]
    assert min(values) >= lower
    assert max(values) <= upper


def test_probability_families_have_support_and_mirror_provenance():
    rng = np.random.Generator(np.random.PCG64DXSM(801))
    for family in ("uniform", "log_boundary", "event_scale"):
        values = [holdout._sample_probability(rng, family, "wilson", 100, 0.05) for _ in range(100)]
        assert all(0.0 <= p <= 1.0 for p, _ in values)
        if family != "uniform":
            assert {bool(meta["mirror"]) for _, meta in values} == {False, True}


@pytest.mark.parametrize("method", H_METHODS)
def test_endpoint_family_uses_own_method_and_preserves_provenance_without_clipping(method):
    rng = np.random.Generator(np.random.PCG64DXSM(1234))
    p, provenance = holdout._endpoint_probability(rng, method, 27, 0.05)
    lower, upper = interval_for_cell(method, 27, provenance["endpoint_source_x"], 0.05)
    expected = lower if provenance["endpoint_kind"] == "lower" else upper
    assert provenance["endpoint_value"] == expected
    assert provenance["endpoint_value_hex"] == float(expected).hex()
    assert provenance["endpoint_variant"] in {"exact", "nextafter_0", "nextafter_1"}
    assert math.isfinite(p) and 0.0 <= p <= 1.0


def test_duplicate_cells_are_retried_deterministically(monkeypatch):
    probabilities = iter([(0.2, {}), (0.2, {}), (0.3, {})])
    monkeypatch.setattr(holdout, "_sample_probability", lambda *args, **kwargs: next(probabilities))
    monkeypatch.setattr(holdout, "_sample_n", lambda *args, **kwargs: 10)
    rows = holdout._generate_method_design(
        FIXTURE_H_SEED,
        "wilson",
        n_quotas={"n_1_5000": 2},
        p_quotas={"uniform": 2},
        alpha_values=np.asarray([0.05, 0.05]),
    )
    assert [row["p"] for row in rows] == [0.2, 0.3]
    assert [row["duplicate_retry"] for row in rows] == [0, 1]


def test_fixture_design_is_unique_balanced_and_clearly_nonproduction():
    design = generate_fixture_holdout_design(FIXTURE_H_SEED)
    assert len(design) == 32
    assert design.groupby("method").size().to_dict() == {method: 8 for method in H_METHODS}
    assert design["canonical_cell_id"].is_unique
    assert set(design["p_family"]) == set(H_P_QUOTAS)
    with pytest.raises(ValueError, match="FIXTURE_ONLY"):
        generate_fixture_holdout_design("looks-like-production")


def test_design_artifact_hash_is_verified_before_evaluation(tmp_path):
    design = generate_fixture_holdout_design(FIXTURE_H_SEED)
    design_path, metadata_path, metadata = _write_design_artifact(tmp_path, design)
    loaded, _ = verify_design_artifact(
        design_path, metadata_path, require_production=False
    )
    assert len(loaded) == len(design)
    metadata["design_artifact_sha256"] = "0" * 64
    metadata = seal_metadata(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact SHA-256"):
        verify_design_artifact(
            design_path, metadata_path, require_production=False
        )


def test_h_design_metadata_content_and_runtime_mismatch_are_rejected(tmp_path):
    design = generate_fixture_holdout_design(FIXTURE_H_SEED)
    design_path, metadata_path, metadata = _write_design_artifact(tmp_path, design)

    content_tampered = dict(metadata)
    content_tampered["cell_count"] = len(design) + 1
    metadata_path.write_text(json.dumps(content_tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata content SHA-256"):
        verify_design_artifact(
            design_path, metadata_path, require_production=False
        )

    runtime_tampered = dict(metadata)
    runtime_tampered["runtime_head"] = "e" * 40
    runtime_tampered = seal_metadata(runtime_tampered)
    metadata_path.write_text(json.dumps(runtime_tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime_head"):
        verify_design_artifact(
            design_path, metadata_path, require_production=False
        )


@pytest.mark.parametrize("n", [1, 3, 27, 1_000_000])
@pytest.mark.parametrize("alpha", [0.001, 0.05, 0.2])
def test_wilson_boundary_localization_preserves_exact_x0_xn_identities(n, alpha):
    zero = localize_acceptance("wilson", n, alpha, 0.0)
    one = localize_acceptance("wilson", n, alpha, 1.0)
    assert (zero.first_x, zero.last_x) == (0, 0)
    assert (one.first_x, one.last_x) == (n, n)
    assert stable_binomial_coverage(n, 0.0, zero.first_x, zero.last_x) == 1.0
    assert stable_binomial_coverage(n, 1.0, one.first_x, one.last_x) == 1.0


@pytest.mark.parametrize("method", H_METHODS)
@pytest.mark.parametrize("p", [0.0, 1e-6, 0.1, 0.5, 0.9, 1.0])
def test_independent_acceptance_localization_matches_small_brute_force(method, p):
    n = 18
    alpha = 0.05
    localization = localize_acceptance(method, n, alpha, p)
    coverage = stable_binomial_coverage(n, p, localization.first_x, localization.last_x)
    brute, selected = _brute_coverage(method, n, alpha, p)
    expected = [] if localization.is_empty else list(range(localization.first_x, localization.last_x + 1))
    assert selected == expected
    assert coverage == pytest.approx(brute, abs=5e-15)


def test_clopper_pearson_small_fixture_never_shortfalls_nominal():
    for n in (1, 2, 5, 10, 25):
        for alpha in (0.001, 0.05, 0.2):
            for p in np.linspace(0.0, 1.0, 41):
                localized = localize_acceptance("clopper_pearson", n, alpha, float(p))
                coverage = stable_binomial_coverage(n, float(p), localized.first_x, localized.last_x)
                assert coverage >= 1.0 - alpha - 1e-12


def test_wald_localization_follows_unclipped_production_formula():
    localization = localize_acceptance("wald", 10, 0.05, 0.01)
    brute, selected = _brute_coverage("wald", 10, 0.05, 0.01)
    assert selected == list(range(localization.first_x, localization.last_x + 1))
    assert stable_binomial_coverage(10, 0.01, localization.first_x, localization.last_x) == pytest.approx(brute)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        interval = PopulationProportionCI.from_counts(1, 10, alpha=0.05, method="wald").calculate_interval()
    assert interval["lb"] < 0.0


def test_wald_maj01_small_n_boundary_grid_matches_production_brute_force():
    empty_cases: list[tuple[int, float, float]] = []
    for n in (1, 2):
        for alpha in ALPHAS:
            intervals = [interval_for_cell("wald", n, x, alpha) for x in range(n + 1)]
            for p in _wald_boundary_probabilities():
                expected = _acceptance_from_intervals(intervals, p)
                localized = localize_acceptance("wald", n, float(alpha), p)
                actual = (
                    []
                    if localized.is_empty
                    else list(range(localized.first_x, localized.last_x + 1))
                )
                assert actual == expected, (n, alpha, p.hex(), actual, expected)
                coverage = stable_binomial_coverage(
                    n, p, localized.first_x, localized.last_x
                )
                brute = float(__import__("scipy").stats.binom.pmf(expected, n, p).sum())
                assert coverage == pytest.approx(brute, abs=1e-12)
                if not expected:
                    empty_cases.append((n, float(alpha), p))
                    assert localized.acceptance_kind == "empty"
                    assert localized.first_x > localized.last_x
                    assert coverage == 0.0
    assert empty_cases


def test_wald_adversarial_n1_to_n100_grid_matches_exact_production_membership():
    probabilities = sorted(
        set(_wald_boundary_probabilities())
        | {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}
    )
    for n in range(1, 101):
        for alpha in ALPHAS:
            intervals = [interval_for_cell("wald", n, x, alpha) for x in range(n + 1)]
            for p in probabilities:
                expected = _acceptance_from_intervals(intervals, p)
                if expected:
                    assert expected == list(range(expected[0], expected[-1] + 1))
                localized = localize_acceptance("wald", n, float(alpha), p)
                actual = (
                    []
                    if localized.is_empty
                    else list(range(localized.first_x, localized.last_x + 1))
                )
                assert actual == expected, (n, alpha, p.hex(), actual, expected)
                exact = float(__import__("scipy").stats.binom.pmf(expected, n, p).sum())
                independent = stable_binomial_coverage(
                    n, p, localized.first_x, localized.last_x
                )
                assert abs(independent - exact) <= 1e-12


def test_wald_acceptance_has_representable_complement_symmetry():
    probabilities = sorted(
        set(_wald_boundary_probabilities()) | {0.01, 0.1, 0.25, 0.5}
    )
    for n in (1, 2, 3, 10, 27, 100):
        for alpha in ALPHAS:
            for p in probabilities:
                complement = 1.0 - p
                if 1.0 - complement != p:
                    continue
                left = localize_acceptance("wald", n, float(alpha), p)
                right = localize_acceptance("wald", n, float(alpha), complement)
                left_set = (
                    [] if left.is_empty else list(range(left.first_x, left.last_x + 1))
                )
                right_set = (
                    [] if right.is_empty else list(range(right.first_x, right.last_x + 1))
                )
                assert right_set == sorted(n - x for x in left_set)


def test_wald_subnormal_guards_exclude_impossible_boundary_outcomes():
    positive_subnormal = float(np.nextafter(0.0, 1.0))
    below_one = float(np.nextafter(1.0, 0.0))
    for n in (1, 2):
        for alpha in ALPHAS:
            low = localize_acceptance("wald", n, float(alpha), positive_subnormal)
            high = localize_acceptance("wald", n, float(alpha), below_one)
            low_actual = (
                [] if low.is_empty else list(range(low.first_x, low.last_x + 1))
            )
            high_actual = (
                [] if high.is_empty else list(range(high.first_x, high.last_x + 1))
            )
            low_brute = _brute_coverage("wald", n, float(alpha), positive_subnormal)[1]
            high_brute = _brute_coverage("wald", n, float(alpha), below_one)[1]
            assert low_actual == low_brute
            assert high_actual == high_brute
            assert 0 not in low_actual
            assert n not in high_actual


def test_g_wald_boundary_fixture_retains_cf_truth_and_simulates_empty_membership():
    p = float(np.nextafter(0.0, 1.0))
    rows = []
    truth: dict[str, float] = {}
    for n in (1, 2):
        identity = canonical_cell_id("wald", n, 0.05, p)
        coverage, _ = _brute_coverage("wald", n, 0.05, p)
        truth[identity] = coverage
        rows.append(
            {
                "method": "wald",
                "n": n,
                "alpha": 0.05,
                "p": p,
                "selection_kind": "broad",
                "canonical_cell_id": identity,
            }
        )
    selection = attach_cf_float64_authority(
        pd.DataFrame(rows),
        cf_provider=lambda method, n, alpha, value: truth[
            canonical_cell_id(method, n, alpha, value)
        ],
    )
    result = simulate_shadow_cells(
        selection,
        master_seed=FIXTURE_G_SEED,
        workers=1,
        fixture_reps=2_000,
        verify_source=False,
    )
    n1 = result.loc[result["n"] == 1].iloc[0]
    assert n1["coverage_cf_float64"] == 0.0
    assert n1["coverage_independent_localization"] == 0.0
    assert n1["acceptance_kind"] == "empty"
    assert n1["first_x"] > n1["last_x"]
    assert n1["covered_float64"] == 0
    assert n1["float64_mc_gate_pass"]
    assert result["cf_vs_independent_consistent"].all()


def test_h_wald_maj01_empty_region_is_statistical_evidence_not_implementation_failure():
    p = float(np.nextafter(0.0, 1.0))
    row = {
        "method": "wald",
        "n": 1,
        "alpha": 0.05,
        "p": p,
        "p_family": "fixture_boundary",
        "canonical_cell_id": canonical_cell_id("wald", 1, 0.05, p),
    }
    summary, audit, failure = holdout._evaluate_one(row)
    assert summary["coverage_float64"] == 0.0
    assert summary["classification"] == "observed_statistical_shortfall"
    assert summary["resolved"] is True
    assert audit is None
    assert failure is None


def test_only_g_mc_schema_advances_for_metadata_accounting_remediation():
    assert GH_EXPERIMENT_VERSION == "proportion-ci-cp06-gh-v3"
    assert GH_SCHEMA_VERSION == "cp06-gh-schema-v3"
    assert G_SELECTION_SCHEMA_VERSION == "cp06-g-selection-schema-v3"
    assert G_MC_SCHEMA_VERSION == "cp06-g-mc-schema-v4"
    assert H_DESIGN_SCHEMA_VERSION == "cp06-h-design-schema-v3"
    assert gh_common.H_EVALUATION_SCHEMA_VERSION == "cp06-h-evaluation-schema-v3"


def test_f_resolved_validation_emits_no_futurewarning_and_accepts_only_true(tmp_path):
    selection = _selection_with_authority(include_f=False)
    audit_path, metadata_path, _ = _write_f_audit(tmp_path, selection)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        attached = gh._attach_f_audit(selection, audit_path, metadata_path)
    assert attached.loc[
        (attached["selection_kind"] == "critical")
        & (attached["method"] == "wilson"),
        "resolved",
    ].eq(True).all()

    unresolved_path, unresolved_metadata, _ = _write_f_audit(
        tmp_path / "false", selection, unresolved=True
    )
    with warnings.catch_warnings(), pytest.raises(ValueError, match="unresolved rows"):
        warnings.simplefilter("error", FutureWarning)
        gh._attach_f_audit(selection, unresolved_path, unresolved_metadata)

    missing = _selection_with_authority()
    wilson_index = missing.index[
        (missing["selection_kind"] == "critical") & (missing["method"] == "wilson")
    ][0]
    missing.loc[wilson_index, "resolved"] = np.nan
    with warnings.catch_warnings(), pytest.raises(
        ValueError, match="resolved F coverage"
    ):
        warnings.simplefilter("error", FutureWarning)
        validate_g_selection(missing, require_authority=True, require_f=True)


def test_jeffreys_evaluation_remains_bayesian_comparator():
    design = pd.DataFrame(
        [{"method": "jeffreys", "n": 20, "alpha": 0.05, "p": 0.4, "p_family": "uniform", "canonical_cell_id": canonical_cell_id("jeffreys", 20, 0.05, 0.4)}]
    )
    result = evaluate_holdout_frame(design, workers=1)["summary"].iloc[0]
    assert result["interval_kind"] == "bayesian_comparator"
    assert result["classification"] == "bayesian_comparator"


def test_cp_shortfall_escalates_to_hp_and_unresolved_is_blocking(monkeypatch):
    identity = canonical_cell_id("clopper_pearson", 10, 0.05, 0.5)
    design = pd.DataFrame(
        [{"method": "clopper_pearson", "n": 10, "alpha": 0.05, "p": 0.5, "p_family": "uniform", "canonical_cell_id": identity}]
    )
    monkeypatch.setattr(holdout, "stable_binomial_coverage", lambda *args: 0.8)
    monkeypatch.setattr(
        holdout,
        "high_precision_recheck",
        lambda *args, **kwargs: {
            "coverage_hp_float": 0.8,
            "classification": "unresolved",
            "resolved": False,
            "notes": "fixture inconsistency",
        },
    )
    evaluated = evaluate_holdout_frame(design, workers=1)
    assert evaluated["summary"].iloc[0]["hp_governs"]
    assert not evaluated["summary"].iloc[0]["resolved"]
    assert evaluated["failures"].iloc[0]["blocking"]


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_raw_binomial_coverage_fails_closed(monkeypatch, bad_value):
    def bad_cdf(k, *args, **kwargs):
        return 0.0 if float(k) < 0 else bad_value

    monkeypatch.setattr(acceptance.stats.binom, "cdf", bad_cdf)
    with pytest.raises(NumericalCoverageError, match="not finite"):
        stable_binomial_coverage(10, 0.5, 0, 2)


def test_out_of_range_raw_binomial_coverage_fails_closed(monkeypatch):
    monkeypatch.setattr(
        acceptance.stats.binom,
        "cdf",
        lambda k, *args, **kwargs: 0.0 if float(k) < 0 else 1.01,
    )
    with pytest.raises(NumericalCoverageError, match=r"outside \[0,1\]"):
        stable_binomial_coverage(10, 0.5, 0, 2)


def test_numerical_coverage_error_becomes_blocking_h_finding(monkeypatch):
    identity = canonical_cell_id("wilson", 10, 0.05, 0.5)
    row = {
        "method": "wilson",
        "n": 10,
        "alpha": 0.05,
        "p": 0.5,
        "p_family": "uniform",
        "canonical_cell_id": identity,
    }
    monkeypatch.setattr(
        holdout,
        "stable_binomial_coverage",
        lambda *args: (_ for _ in ()).throw(NumericalCoverageError("synthetic NaN")),
    )
    summary, audit, failure = holdout._evaluate_one(row)
    assert audit is None
    assert summary["resolved"] is False
    assert failure["blocking"] is True
    assert "NumericalCoverageError" in failure["details"]


def test_high_precision_boundary_recheck_is_resolved_for_canonical_wilson_boundaries():
    for p, expected in ((0.0, (0, 0)), (1.0, (27, 27))):
        localization = localize_acceptance("wilson", 27, 0.2, p)
        audit = high_precision_recheck(
            "wilson", 27, 0.2, p, 1.0, localization, digits=80
        )
        assert json.loads(audit["acceptance_runs_hp"]) == [list(expected)]
        assert audit["coverage_hp_float"] == 1.0
        assert audit["resolved"] is True


def test_no_production_routing_metadata_or_capability_files_changed():
    result = verify_frozen_sources()
    assert result["production_unchanged"] is True
