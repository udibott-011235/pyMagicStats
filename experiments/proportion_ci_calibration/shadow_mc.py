"""CP06-G deterministic selection and shadow Monte Carlo."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
from typing import Iterable

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
    PRODUCTION_METHODS,
    STRESS_N,
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


def validate_g_selection(selection: pd.DataFrame) -> None:
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


def cell_seed_128(master_seed: str, canonical_id: str) -> int:
    if not isinstance(master_seed, str) or not master_seed:
        raise ValueError("a non-empty production master seed is required at runtime")
    digest = np.frombuffer(
        __import__("hashlib").sha256(
            (RNG_DERIVATION_LABEL + master_seed + canonical_id).encode("utf-8")
        ).digest()[:16],
        dtype="<u8",
    )
    return int(digest[0]) | (int(digest[1]) << 64)


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
    evidence_coverage = row.get("coverage")
    deterministic = (
        localized_coverage
        if evidence_coverage is None or pd.isna(evidence_coverage)
        else float(evidence_coverage)
    )
    rng = np.random.Generator(np.random.PCG64DXSM(cell_seed_128(master_seed, identity)))
    covered = 0
    remaining = reps
    while remaining:
        draw_count = min(_INTERNAL_DRAW_CHUNK, remaining)
        draws = rng.binomial(n, p, size=draw_count)
        covered += int(
            np.count_nonzero(
                (draws >= localization.first_x) & (draws <= localization.last_x)
            )
        )
        remaining -= draw_count
    mc = covered / reps
    float_gate = mc_gate(deterministic, mc, reps)
    output = {
        "canonical_cell_id": identity,
        "method": method,
        "n": n,
        "alpha": alpha,
        "p": p,
        "selection_kind": row["selection_kind"],
        "reps": reps,
        "covered": covered,
        "coverage_mc": mc,
        "coverage_deterministic_float64": deterministic,
        "coverage_independent_localization": localized_coverage,
        "cf_e_coverage_available": evidence_coverage is not None
        and not pd.isna(evidence_coverage),
        "cf_e_vs_independent_difference": (
            abs(deterministic - localized_coverage)
            if evidence_coverage is not None and not pd.isna(evidence_coverage)
            else math.nan
        ),
        "first_x": localization.first_x,
        "last_x": localization.last_x,
        "acceptance_kind": localization.acceptance_kind,
        "localization_method": localization.localization_method,
        "boundary_validation": localization.boundary_validation,
        "rng_algorithm": RNG_ALGORITHM,
        "rng_derivation": RNG_DERIVATION_LABEL,
        **{f"float64_{key}": value for key, value in float_gate.items()},
        "hp_governs": False,
        "hp_resolved": None,
        "coverage_hp": math.nan,
        "hp_mc_gate_pass": None,
    }
    hp_value = row.get("coverage_hp_float")
    hp_resolved = row.get("resolved")
    if hp_value is not None and not pd.isna(hp_value):
        if hp_resolved is not True and hp_resolved != np.bool_(True):
            raise ValueError(f"unresolved F evidence for selected cell {identity}")
        hp = float(hp_value)
        hp_gate = mc_gate(hp, mc, reps)
        output.update(
            {
                "hp_governs": True,
                "hp_resolved": True,
                "coverage_hp": hp,
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
        validate_g_selection(selection)
    elif fixture_reps < 1:
        raise ValueError("fixture_reps must be positive")
    records = selection.sort_values("canonical_cell_id", kind="mergesort").to_dict("records")
    arguments = [(row, master_seed, fixture_reps) for row in records]
    if workers == 1:
        rows = [_simulate_one(item) for item in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_simulate_one, arguments, chunksize=1))
    return pd.DataFrame(rows).sort_values("canonical_cell_id", kind="mergesort").reset_index(drop=True)
