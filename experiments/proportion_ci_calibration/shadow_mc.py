"""CP06-G deterministic selection and shadow Monte Carlo."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.proportion_ci_calibration.acceptance import (
    localize_acceptance,
    stable_binomial_coverage,
)
from experiments.proportion_ci_calibration.gh_common import (
    canonical_cell_id,
    stable_digest,
    stable_rank,
    verify_frozen_sources,
)
from experiments.proportion_ci_calibration.harness import (
    ALPHAS,
    EVENT_LAMBDAS,
    FIXED_ANCHORS,
    DEFAULT_ENDPOINT_CACHE_ROOT,
    EndpointGridCache,
    PRODUCTION_METHODS,
    STRESS_N,
    evaluate_coverage,
    production_interval_grid,
)


CRITICAL_CELLS = 128
BROAD_CELLS = 512
CRITICAL_PER_METHOD = 42
CRITICAL_REPS = 1_000_000
BROAD_REPS = 250_000
TOTAL_DRAWS = CRITICAL_CELLS * CRITICAL_REPS + BROAD_CELLS * BROAD_REPS
RNG_ALGORITHM = "numpy.random.Generator(PCG64DXSM)"
RNG_DERIVATION_LABEL = "CP06-G-MC-v1"
_INTERNAL_DRAW_CHUNK = 250_000


class DeterministicMappingError(RuntimeError):
    """Raised when C-F and independent G/H mappings disagree."""

N_STRATA = (
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-30", 21, 30),
    ("31-50", 31, 50),
    ("51-100", 51, 100),
    ("101-250", 101, 250),
    ("251-500", 251, 500),
    ("501-1000", 501, 1_000),
    ("1001-2000", 1_001, 2_000),
    ("2001-5000", 2_001, 5_000),
    (">5000", 5_001, 1_000_000),
)


def n_stratum(n: int) -> str:
    for label, lower, upper in N_STRATA:
        if lower <= int(n) <= upper:
            return label
    raise ValueError(f"n={n} is outside the CP-04 domain")


def _required_e_columns() -> set[str]:
    return {"method", "n", "alpha", "p", "coverage", "nominal"}


def _ranked_e_minima(frame: pd.DataFrame) -> pd.DataFrame:
    missing = _required_e_columns() - set(frame.columns)
    if missing:
        raise ValueError(f"E minima are missing columns: {sorted(missing)}")
    selected = frame.loc[frame["method"].isin(PRODUCTION_METHODS)].copy()
    selected["n"] = selected["n"].astype(int)
    selected["alpha"] = selected["alpha"].astype(float)
    selected["p"] = selected["p"].astype(float)
    selected["coverage"] = selected["coverage"].astype(float)
    selected["nominal"] = selected["nominal"].astype(float)
    selected["signed_margin"] = selected["coverage"] - selected["nominal"]
    selected["n_stratum"] = selected["n"].map(n_stratum)
    selected["canonical_cell_id"] = [
        canonical_cell_id(method, n, alpha, p)
        for method, n, alpha, p in zip(
            selected["method"], selected["n"], selected["alpha"], selected["p"]
        )
    ]
    selected = selected.drop_duplicates("canonical_cell_id", keep="first")
    return selected.sort_values(
        ["signed_margin", "canonical_cell_id"], kind="mergesort"
    ).reset_index(drop=True)


def select_critical_cells(e_minima: pd.DataFrame) -> pd.DataFrame:
    ranked = _ranked_e_minima(e_minima)
    chosen_ids: set[str] = set()
    chosen_rows: list[pd.Series] = []

    def choose(row: pd.Series) -> None:
        identity = str(row["canonical_cell_id"])
        if identity not in chosen_ids:
            chosen_ids.add(identity)
            chosen_rows.append(row)

    for method in PRODUCTION_METHODS:
        method_rows = ranked.loc[ranked["method"] == method]
        if method_rows.empty:
            raise ValueError(f"E minima contain no rows for {method}")
        before = len(chosen_rows)
        for alpha in ALPHAS:
            candidates = method_rows.loc[method_rows["alpha"] == float(alpha)]
            if candidates.empty:
                raise ValueError(f"E minima lack {method}, alpha={alpha}")
            choose(candidates.iloc[0])
        for label, _, _ in N_STRATA:
            candidates = method_rows.loc[method_rows["n_stratum"] == label]
            if candidates.empty:
                raise ValueError(f"E minima lack {method}, n stratum={label}")
            choose(candidates.iloc[0])
        target = before + CRITICAL_PER_METHOD
        for _, row in method_rows.iterrows():
            if len(chosen_rows) >= target:
                break
            choose(row)
        if len(chosen_rows) != target:
            raise ValueError(f"E minima cannot fill 42 unique critical cells for {method}")

    for _, row in ranked.iterrows():
        if len(chosen_rows) == CRITICAL_CELLS:
            break
        choose(row)
    if len(chosen_rows) != CRITICAL_CELLS:
        raise ValueError("E minima cannot fill the two globally worst unselected cells")

    result = pd.DataFrame(chosen_rows).reset_index(drop=True)
    result["selection_kind"] = "critical"
    result["reps"] = CRITICAL_REPS
    result["selection_rank"] = np.arange(1, len(result) + 1, dtype=np.int64)
    return result


def _n_values_for_stratum(label: str) -> tuple[int, ...] | range:
    for candidate, lower, upper in N_STRATA:
        if candidate != label:
            continue
        if label == ">5000":
            return tuple(int(value) for value in STRESS_N)
        return range(lower, upper + 1)
    raise ValueError(label)


def _stable_n(method: str, alpha: float, stratum: str, probe: int) -> int:
    domain = _n_values_for_stratum(stratum)
    digest = stable_digest("CP06-G-BROAD-N-v1", method, float(alpha).hex(), stratum, probe)
    return int(domain[int.from_bytes(digest[:8], "big") % len(domain)])


def _stable_probability(method: str, n: int, alpha: float, stratum: str, probe: int) -> tuple[float, str]:
    digest = stable_digest(
        "CP06-G-BROAD-P-v1", method, n, float(alpha).hex(), stratum, probe
    )
    family = int.from_bytes(digest[:2], "big") % 3
    side = int.from_bytes(digest[2:4], "big") % 2
    index = int.from_bytes(digest[4:12], "big")
    if family == 0:
        anchors = np.unique(np.concatenate((FIXED_ANCHORS, 1.0 - FIXED_ANCHORS)))
        p = float(anchors[index % len(anchors)])
        origin = "cp04_fixed_anchor"
    elif family == 1:
        p = float((index % 9_999 + 1) / 10_000.0)
        origin = "cp04_linear_interior"
    else:
        eligible = EVENT_LAMBDAS[EVENT_LAMBDAS / n <= 0.5]
        q = float(eligible[index % len(eligible)] / n)
        p = q if side == 0 else 1.0 - q
        origin = "cp04_event_scale"
    return p, origin


def select_broad_cells(critical: pd.DataFrame) -> pd.DataFrame:
    critical_ids = set(str(value) for value in critical["canonical_cell_id"])
    selected_ids = set(critical_ids)
    strata = [
        (method, float(alpha), label)
        for method in PRODUCTION_METHODS
        for alpha in ALPHAS
        for label, _, _ in N_STRATA
    ]
    extra_strata = set(
        sorted(
            strata,
            key=lambda item: stable_rank(
                "CP06-G-BROAD-EXTRA-v1", item[0], float(item[1]).hex(), item[2]
            ),
        )[:8]
    )
    rows: list[dict[str, object]] = []
    for method, alpha, stratum in strata:
        quota = 3 if (method, alpha, stratum) in extra_strata else 2
        accepted = 0
        probe = 0
        while accepted < quota:
            n = _stable_n(method, alpha, stratum, probe)
            p, origin = _stable_probability(method, n, alpha, stratum, probe)
            identity = canonical_cell_id(method, n, alpha, p)
            probe += 1
            if identity in selected_ids:
                continue
            selected_ids.add(identity)
            rows.append(
                {
                    "method": method,
                    "n": n,
                    "alpha": alpha,
                    "p": p,
                    "n_stratum": stratum,
                    "canonical_cell_id": identity,
                    "selection_kind": "broad",
                    "selection_origin": origin,
                    "probe_count": probe,
                    "reps": BROAD_REPS,
                }
            )
            accepted += 1
    result = pd.DataFrame(rows)
    if len(result) != BROAD_CELLS:
        raise AssertionError(f"expected 512 broad cells, got {len(result)}")
    result["selection_rank"] = np.arange(1, len(result) + 1, dtype=np.int64)
    return result


def build_g_selection(e_minima: pd.DataFrame) -> pd.DataFrame:
    critical = select_critical_cells(e_minima)
    broad = select_broad_cells(critical)
    columns = sorted(set(critical.columns) | set(broad.columns))
    result = pd.concat(
        [critical.reindex(columns=columns), broad.reindex(columns=columns)],
        ignore_index=True,
    )
    if result["canonical_cell_id"].duplicated().any():
        raise AssertionError("G selection contains duplicate canonical cells")
    return result


def numerical_tolerance(n: int) -> float:
    return 1e-12 if int(n) <= 5_000 else 1e-10


def recompute_canonical_ids(selection: pd.DataFrame) -> pd.Series:
    required = {"method", "n", "alpha", "p", "canonical_cell_id"}
    missing = required - set(selection.columns)
    if missing:
        raise ValueError(f"G selection is missing identity columns: {sorted(missing)}")
    reconstructed = pd.Series(
        [
            canonical_cell_id(method, n, alpha, p)
            for method, n, alpha, p in zip(
                selection["method"],
                selection["n"],
                selection["alpha"],
                selection["p"],
            )
        ],
        index=selection.index,
        dtype=object,
    )
    mismatched = reconstructed != selection["canonical_cell_id"].astype(str)
    if bool(mismatched.any()):
        raise ValueError(
            "G selection canonical-cell tampering detected: "
            f"{selection.loc[mismatched, 'canonical_cell_id'].tolist()[:3]}"
        )
    return reconstructed


def canonical_selection_hash(selection: pd.DataFrame) -> str:
    identities = recompute_canonical_ids(selection)
    digest = hashlib.sha256()
    for identity in sorted(identities.tolist()):
        digest.update((identity + "\n").encode("utf-8"))
    return digest.hexdigest()


def _cf_coverage_values(
    selection: pd.DataFrame,
    *,
    endpoint_cache: EndpointGridCache,
) -> np.ndarray:
    values = np.empty(len(selection), dtype=np.float64)
    indexed = selection.reset_index(drop=True)
    for (method, n, alpha), group in indexed.groupby(
        ["method", "n", "alpha"], sort=True
    ):
        n = int(n)
        alpha = float(alpha)
        method = str(method)
        grid = endpoint_cache.get_or_create(
            n,
            alpha,
            method,
            lambda n=n, alpha=alpha, method=method: production_interval_grid(
                n, alpha, method
            ),
        )
        evaluation = evaluate_coverage(
            n,
            group["p"].to_numpy(dtype=np.float64),
            grid.lower,
            grid.upper,
        )
        values[group.index.to_numpy()] = evaluation.coverage
    return values


def attach_cf_float64_authority(
    selection: pd.DataFrame,
    *,
    endpoint_cache: EndpointGridCache | None = None,
    cf_provider=None,
    independent_provider=None,
    fail_on_inconsistency: bool = True,
) -> pd.DataFrame:
    """Attach frozen C-F truth and the independent MC-side reconstruction."""

    result = selection.reset_index(drop=True).copy()
    recompute_canonical_ids(result)
    if cf_provider is None:
        cache = endpoint_cache or EndpointGridCache(DEFAULT_ENDPOINT_CACHE_ROOT)
        cf_values = _cf_coverage_values(result, endpoint_cache=cache)
    else:
        cf_values = np.asarray(
            [
                cf_provider(str(row.method), int(row.n), float(row.alpha), float(row.p))
                for row in result.itertuples(index=False)
            ],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(cf_values)):
        raise DeterministicMappingError("C-F produced non-finite deterministic coverage")

    independent_values: list[float] = []
    for row in result.itertuples(index=False):
        if independent_provider is None:
            localized = localize_acceptance(
                str(row.method), int(row.n), float(row.alpha), float(row.p)
            )
            value = stable_binomial_coverage(
                int(row.n), float(row.p), localized.first_x, localized.last_x
            )
        else:
            value = independent_provider(
                str(row.method), int(row.n), float(row.alpha), float(row.p)
            )
        independent_values.append(float(value))
    independent = np.asarray(independent_values, dtype=np.float64)
    if not np.all(np.isfinite(independent)):
        raise DeterministicMappingError(
            "independent G/H localization produced non-finite coverage"
        )

    result["coverage_cf_float64"] = cf_values
    result["coverage_independent_localization"] = independent
    result["cf_vs_independent_difference"] = np.abs(cf_values - independent)
    tolerances = result["n"].map(numerical_tolerance).to_numpy(dtype=np.float64)
    result["cf_vs_independent_consistent"] = (
        result["cf_vs_independent_difference"].to_numpy() <= tolerances
    )
    critical = result["selection_kind"] == "critical"
    e_values = pd.to_numeric(result.get("coverage"), errors="coerce")
    result["coverage_e_float64"] = np.where(critical, e_values, np.nan)
    result["e_vs_cf_difference"] = np.where(
        critical, np.abs(e_values - cf_values), np.nan
    )
    result["e_vs_cf_consistent"] = np.where(
        critical, result["e_vs_cf_difference"] <= tolerances, True
    )
    inconsistent = ~result["cf_vs_independent_consistent"] | ~result[
        "e_vs_cf_consistent"
    ].astype(bool)
    if fail_on_inconsistency and bool(inconsistent.any()):
        rows = result.loc[
            inconsistent,
            [
                "canonical_cell_id",
                "coverage_cf_float64",
                "coverage_independent_localization",
                "cf_vs_independent_difference",
                "coverage_e_float64",
                "e_vs_cf_difference",
            ],
        ]
        raise DeterministicMappingError(
            "blocking deterministic mapping inconsistency: "
            + rows.head(10).to_json(orient="records")
        )
    return result


def validate_g_selection(
    selection: pd.DataFrame,
    *,
    require_authority: bool = False,
    require_f: bool = False,
) -> None:
    recompute_canonical_ids(selection)
    counts = selection["selection_kind"].value_counts().to_dict()
    if counts != {"broad": BROAD_CELLS, "critical": CRITICAL_CELLS}:
        raise ValueError(f"invalid G selection counts: {counts}")
    expected_reps = selection["selection_kind"].map(
        {"critical": CRITICAL_REPS, "broad": BROAD_REPS}
    )
    if not np.array_equal(selection["reps"].to_numpy(), expected_reps.to_numpy()):
        raise ValueError("G selection repetitions do not match the preregistration")
    if int(expected_reps.sum()) != TOTAL_DRAWS:
        raise ValueError("G selection does not imply exactly 256,000,000 draws")
    if selection["canonical_cell_id"].duplicated().any():
        raise ValueError("G selection contains duplicate canonical cells")
    for method in PRODUCTION_METHODS:
        critical = selection.loc[
            (selection["selection_kind"] == "critical")
            & (selection["method"] == method)
        ]
        if len(critical) < CRITICAL_PER_METHOD:
            raise ValueError(f"critical selection underfills {method}")
        if set(critical["alpha"].astype(float)) != set(ALPHAS):
            raise ValueError(f"critical selection does not cover every alpha for {method}")
        if set(critical["n_stratum"]) != {row[0] for row in N_STRATA}:
            raise ValueError(f"critical selection does not cover every n stratum for {method}")
    if require_authority:
        required = {
            "coverage_cf_float64",
            "coverage_independent_localization",
            "cf_vs_independent_difference",
            "cf_vs_independent_consistent",
            "coverage_e_float64",
            "e_vs_cf_difference",
            "e_vs_cf_consistent",
        }
        missing = required - set(selection.columns)
        if missing:
            raise ValueError(f"G selection lacks C-F authority fields: {sorted(missing)}")
        if not bool(selection["cf_vs_independent_consistent"].astype(bool).all()):
            raise DeterministicMappingError("G selection contains a blocking C-F mapping mismatch")
        critical = selection["selection_kind"] == "critical"
        if not bool(selection.loc[critical, "e_vs_cf_consistent"].astype(bool).all()):
            raise DeterministicMappingError("critical E and reconstructed C-F coverage disagree")
    if require_f:
        required = {
            "coverage_hp_float",
            "resolved",
            "classification",
            "acceptance_changed",
            "acceptance_runs_float64",
            "acceptance_runs_hp",
        }
        missing = required - set(selection.columns)
        if missing:
            raise ValueError(f"G selection lacks F evidence fields: {sorted(missing)}")
        wilson_critical = (selection["selection_kind"] == "critical") & (
            selection["method"] == "wilson"
        )
        evidence = selection.loc[wilson_critical]
        required_values = [
            "coverage_hp_float",
            "classification",
            "acceptance_changed",
            "acceptance_runs_float64",
            "acceptance_runs_hp",
        ]
        if evidence[required_values].isna().any().any() or not bool(
            evidence["resolved"].fillna(False).astype(bool).all()
        ):
            raise ValueError(
                "every selected critical Wilson cell requires resolved F coverage evidence"
            )
        for row in evidence.itertuples(index=False):
            if not isinstance(row.resolved, (bool, np.bool_)) or not bool(row.resolved):
                raise ValueError("resolved F coverage must carry an explicit true boolean")
            if not isinstance(row.acceptance_changed, (bool, np.bool_)):
                raise ValueError("F acceptance_changed must be boolean")
            if not isinstance(row.classification, str) or not row.classification:
                raise ValueError("F classification must be a non-empty string")
            float_runs = _parse_runs(
                row.acceptance_runs_float64, "acceptance_runs_float64"
            )
            hp_runs = _parse_runs(row.acceptance_runs_hp, "acceptance_runs_hp")
            if bool(row.acceptance_changed) != (float_runs != hp_runs):
                raise ValueError("F acceptance_changed disagrees with the recorded runs")


def cell_seed_128(master_seed: str, canonical_id: str) -> int:
    if not isinstance(master_seed, str) or not master_seed:
        raise ValueError("a non-empty production master seed is required at runtime")
    digest = stable_digest(RNG_DERIVATION_LABEL, master_seed, canonical_id)[:16]
    return int.from_bytes(digest, byteorder="big", signed=False)


def mc_gate(exact: float, mc: float, reps: int) -> dict[str, float | bool]:
    standard_error = math.sqrt(max(0.0, exact * (1.0 - exact)) / int(reps))
    tolerance = max(5.0 * standard_error, 0.001)
    difference = abs(float(mc) - float(exact))
    return {
        "mc_standard_error": standard_error,
        "mc_tolerance": tolerance,
        "mc_absolute_difference": difference,
        "mc_gate_pass": bool(difference <= tolerance),
    }


def _parse_runs(value: object, field: str) -> list[tuple[int, int]]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing {field}")
    try:
        runs = [tuple(int(endpoint) for endpoint in item) for item in json.loads(value)]
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {field}") from error
    if any(len(run) != 2 or run[0] > run[1] for run in runs):
        raise ValueError(f"invalid {field}")
    return runs


def _count_runs(draws: np.ndarray, runs: list[tuple[int, int]]) -> int:
    selected = np.zeros(draws.shape, dtype=bool)
    for first, last in runs:
        selected |= (draws >= first) & (draws <= last)
    return int(np.count_nonzero(selected))


def _simulate_one(arguments: tuple[dict[str, object], str, int | None]) -> dict[str, object]:
    row, master_seed, fixture_reps = arguments
    method = str(row["method"])
    n = int(row["n"])
    alpha = float(row["alpha"])
    p = float(row["p"])
    identity = canonical_cell_id(method, n, alpha, p)
    reps = int(fixture_reps) if fixture_reps is not None else (
        CRITICAL_REPS if row["selection_kind"] == "critical" else BROAD_REPS
    )
    localization = localize_acceptance(method, n, alpha, p)
    localized_coverage = stable_binomial_coverage(
        n, p, localization.first_x, localization.last_x
    )
    deterministic = float(row["coverage_cf_float64"])
    tolerance = numerical_tolerance(n)
    if abs(localized_coverage - float(row["coverage_independent_localization"])) > tolerance:
        raise DeterministicMappingError(
            f"runtime independent mapping changed for {identity}"
        )
    if abs(deterministic - localized_coverage) > tolerance:
        raise DeterministicMappingError(
            f"blocking C-F versus independent mapping mismatch for {identity}"
        )

    float_runs = localization.runs
    hp_value = row.get("coverage_hp_float")
    hp_resolved = row.get("resolved")
    hp_governs = hp_value is not None and not pd.isna(hp_value)
    hp_runs: list[tuple[int, int]] = []
    if hp_governs:
        if hp_resolved is not True and hp_resolved != np.bool_(True):
            raise ValueError(f"unresolved F evidence for selected cell {identity}")
        hp_runs = _parse_runs(row.get("acceptance_runs_hp"), "acceptance_runs_hp")
        recorded_float_runs = _parse_runs(
            row.get("acceptance_runs_float64"), "acceptance_runs_float64"
        )
        if recorded_float_runs != float_runs:
            raise DeterministicMappingError(
                f"F float64 acceptance differs from GH independent mapping for {identity}"
            )
        acceptance_changed = bool(row.get("acceptance_changed"))
        if acceptance_changed != (recorded_float_runs != hp_runs):
            raise ValueError(f"F acceptance_changed is inconsistent for {identity}")

    rng = np.random.Generator(np.random.PCG64DXSM(cell_seed_128(master_seed, identity)))
    covered_float64 = 0
    covered_hp = 0
    remaining = reps
    while remaining:
        draw_count = min(_INTERNAL_DRAW_CHUNK, remaining)
        draws = rng.binomial(n, p, size=draw_count)
        covered_float64 += _count_runs(draws, float_runs)
        if hp_governs:
            covered_hp += _count_runs(draws, hp_runs)
        remaining -= draw_count
    mc_float64 = covered_float64 / reps
    mc_hp = covered_hp / reps if hp_governs else math.nan
    float_gate = mc_gate(deterministic, mc_float64, reps)
    output = {
        "canonical_cell_id": identity,
        "method": method,
        "n": n,
        "alpha": alpha,
        "p": p,
        "selection_kind": row["selection_kind"],
        "reps": reps,
        "covered": covered_float64,
        "covered_float64": covered_float64,
        "coverage_mc": mc_float64,
        "coverage_mc_float64": mc_float64,
        "coverage_cf_float64": deterministic,
        "coverage_independent_localization": localized_coverage,
        "cf_vs_independent_difference": abs(deterministic - localized_coverage),
        "cf_vs_independent_consistent": True,
        "first_x": localization.first_x,
        "last_x": localization.last_x,
        "acceptance_kind": localization.acceptance_kind,
        "localization_method": localization.localization_method,
        "boundary_validation": localization.boundary_validation,
        "rng_algorithm": RNG_ALGORITHM,
        "rng_derivation": RNG_DERIVATION_LABEL,
        **{f"float64_{key}": value for key, value in float_gate.items()},
        "acceptance_runs_float64": json.dumps(float_runs, separators=(",", ":")),
        "hp_governs": hp_governs,
        "hp_resolved": bool(hp_resolved) if hp_governs else None,
        "coverage_hp": float(hp_value) if hp_governs else math.nan,
        "covered_hp": covered_hp if hp_governs else None,
        "coverage_mc_hp": mc_hp,
        "acceptance_runs_hp": (
            json.dumps(hp_runs, separators=(",", ":")) if hp_governs else None
        ),
        "acceptance_changed": bool(row.get("acceptance_changed")) if hp_governs else None,
        "hp_mc_gate_pass": None,
    }
    if hp_governs:
        hp = float(hp_value)
        hp_gate = mc_gate(hp, mc_hp, reps)
        output.update(
            {
                **{f"hp_{key}": value for key, value in hp_gate.items()},
            }
        )
    return output


def simulate_shadow_cells(
    selection: pd.DataFrame,
    *,
    master_seed: str,
    workers: int,
    fixture_reps: int | None = None,
    verify_source: bool = True,
) -> pd.DataFrame:
    """Run G; ``fixture_reps`` exists only for explicitly small unit fixtures."""

    if verify_source:
        verify_frozen_sources()
    if fixture_reps is None:
        validate_g_selection(selection, require_authority=True, require_f=True)
    elif fixture_reps < 1:
        raise ValueError("fixture_reps must be positive")
    elif "coverage_cf_float64" not in selection.columns:
        def fixture_coverage(method: str, n: int, alpha: float, p: float) -> float:
            localized = localize_acceptance(method, n, alpha, p)
            return stable_binomial_coverage(
                n, p, localized.first_x, localized.last_x
            )

        selection = attach_cf_float64_authority(
            selection,
            cf_provider=fixture_coverage,
            fail_on_inconsistency=True,
        )
    records = selection.sort_values("canonical_cell_id", kind="mergesort").to_dict("records")
    arguments = [(row, master_seed, fixture_reps) for row in records]
    if workers == 1:
        rows = [_simulate_one(item) for item in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_simulate_one, arguments, chunksize=1))
    return pd.DataFrame(rows).sort_values("canonical_cell_id", kind="mergesort").reset_index(drop=True)
