from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from experiments.proportion_ci_calibration import gh_common
from experiments.proportion_ci_calibration.acceptance import (
    high_precision_recheck,
    interval_for_cell,
    localize_acceptance,
    stable_binomial_coverage,
)
from experiments.proportion_ci_calibration.gh_common import (
    H_DESIGN_SCHEMA_VERSION,
    SourceIntegrityError,
    canonical_cell_id,
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
    build_g_selection,
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
    one = cell_seed_128(FIXTURE_G_SEED, canonical_cell_id("wilson", 3, 0.05, 0.1))
    two = cell_seed_128(FIXTURE_G_SEED, canonical_cell_id("wilson", 3, 0.05, 0.2))
    assert one == cell_seed_128(FIXTURE_G_SEED, canonical_cell_id("wilson", 3, 0.05, 0.1))
    assert 0 <= one < 2**128
    assert one != two


def test_mc_gate_uses_five_se_or_point001():
    result = mc_gate(0.95, 0.949, 1_000_000)
    expected_se = math.sqrt(0.95 * 0.05 / 1_000_000)
    assert result["mc_standard_error"] == pytest.approx(expected_se)
    assert result["mc_tolerance"] == pytest.approx(max(5 * expected_se, 0.001))
    assert result["mc_gate_pass"] is True


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
    design_path = tmp_path / "design.parquet"
    design.to_parquet(design_path, index=False)
    metadata = {
        "design_schema_version": H_DESIGN_SCHEMA_VERSION,
        "design_artifact_sha256": sha256_file(design_path),
        "canonical_design_sha256": canonical_design_hash(design),
        "cell_count": len(design),
        "master_seed": FIXTURE_H_SEED,
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    loaded, _ = verify_design_artifact(design_path, metadata_path)
    assert len(loaded) == len(design)
    metadata["design_artifact_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact SHA-256"):
        verify_design_artifact(design_path, metadata_path)


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
