"""CP06-H seeded holdout generation and independent evaluation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.proportion_ci_calibration.acceptance import (
    AcceptanceLocalizationError,
    endpoint_proximity_for_localization,
    high_precision_recheck,
    interval_for_cell,
    localize_acceptance,
    stable_binomial_coverage,
)
from experiments.proportion_ci_calibration.gh_common import (
    CP04_DOCUMENT_SHA,
    H_DESIGN_SCHEMA_VERSION,
    PRODUCTION_CANDIDATE_SHA,
    SOURCE_CF_HARNESS_SHA,
    canonical_cell_id,
    runtime_head,
    sha256_file,
    stable_digest,
    verify_metadata_content,
    verify_frozen_sources,
)
from experiments.proportion_ci_calibration.harness import ALPHAS, undercoverage_tier


H_METHODS = ("wilson", "clopper_pearson", "wald", "jeffreys")
H_CELLS_PER_METHOD = 2_500
H_TOTAL_CELLS = 10_000
H_N_QUOTAS = {
    "n_1_5000": 1_500,
    "n_5001_100000": 750,
    "n_100001_1000000": 250,
}
H_P_QUOTAS = {
    "uniform": 750,
    "log_boundary": 750,
    "event_scale": 750,
    "endpoint_neighbor": 250,
}
H_ALPHA_BASE_QUOTA = 357
H_RNG_ALGORITHM = "numpy.random.Generator(PCG64DXSM)"
H_RNG_DERIVATION_LABEL = "CP06-H-DESIGN-v1"


def production_holdout_quota_plan() -> dict[str, object]:
    return {
        "methods": {method: H_CELLS_PER_METHOD for method in H_METHODS},
        "n_per_method": dict(H_N_QUOTAS),
        "p_per_method": dict(H_P_QUOTAS),
        "alpha_per_method": {
            "base_per_alpha": H_ALPHA_BASE_QUOTA,
            "seed_selected_extra": 1,
            "total": H_ALPHA_BASE_QUOTA * len(ALPHAS) + 1,
        },
        "total": H_TOTAL_CELLS,
    }


def log_uniform_integer(rng: np.random.Generator, lower: int, upper: int) -> int:
    if lower < 1 or upper < lower:
        raise ValueError("invalid log-uniform integer bounds")
    value = math.floor(math.exp(rng.uniform(math.log(lower), math.log(upper + 1))))
    return min(upper, max(lower, value))


def _method_rng(master_seed: str, method: str) -> np.random.Generator:
    if not isinstance(master_seed, str) or not master_seed:
        raise ValueError("a non-empty Project Owner master seed is required")
    seed = int.from_bytes(
        stable_digest(H_RNG_DERIVATION_LABEL, master_seed, method)[:16], "big"
    )
    return np.random.Generator(np.random.PCG64DXSM(seed))


def _alpha_vector(rng: np.random.Generator) -> np.ndarray:
    values = [float(alpha) for alpha in ALPHAS for _ in range(H_ALPHA_BASE_QUOTA)]
    values.append(float(ALPHAS[int(rng.integers(0, len(ALPHAS)))]))
    result = np.asarray(values, dtype=np.float64)
    rng.shuffle(result)
    return result


def _quota_vector(rng: np.random.Generator, quotas: dict[str, int]) -> np.ndarray:
    values = np.asarray(
        [label for label, count in quotas.items() for _ in range(count)],
        dtype=object,
    )
    rng.shuffle(values)
    return values


def _sample_n(rng: np.random.Generator, band: str) -> int:
    if band == "fixture_n_small":
        return log_uniform_integer(rng, 1, 30)
    if band == "fixture_n_around100":
        return log_uniform_integer(rng, 91, 110)
    if band == "n_1_5000":
        return log_uniform_integer(rng, 1, 5_000)
    if band == "n_5001_100000":
        return log_uniform_integer(rng, 5_001, 100_000)
    if band == "n_100001_1000000":
        return log_uniform_integer(rng, 100_001, 1_000_000)
    raise ValueError(f"unknown n band: {band}")


def _mirrored(rng: np.random.Generator, q: float) -> tuple[float, bool]:
    mirror = bool(rng.integers(0, 2))
    return (1.0 - q if mirror else q), mirror


def _endpoint_probability(
    rng: np.random.Generator,
    method: str,
    n: int,
    alpha: float,
) -> tuple[float, dict[str, object]]:
    """Draw an own-method endpoint or neighbor; never clip invalid Wald."""

    for retry in range(10_000):
        x = int(rng.integers(0, n + 1))
        lower, upper = interval_for_cell(method, n, x, alpha)
        endpoint_kind = "lower" if int(rng.integers(0, 2)) == 0 else "upper"
        endpoint = lower if endpoint_kind == "lower" else upper
        variant_code = int(rng.integers(0, 3))
        variant = ("exact", "nextafter_0", "nextafter_1")[variant_code]
        if variant == "exact":
            p = endpoint
        elif variant == "nextafter_0":
            p = float(np.nextafter(endpoint, 0.0))
        else:
            p = float(np.nextafter(endpoint, 1.0))
        if not math.isfinite(endpoint) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
            continue
        return p, {
            "endpoint_source_x": x,
            "endpoint_kind": endpoint_kind,
            "endpoint_value": float(endpoint),
            "endpoint_value_hex": float(endpoint).hex(),
            "endpoint_variant": variant,
            "endpoint_retry": retry,
        }
    raise RuntimeError(
        f"could not draw a finite in-domain own-method endpoint for {method}, n={n}"
    )


def _sample_probability(
    rng: np.random.Generator,
    family: str,
    method: str,
    n: int,
    alpha: float,
) -> tuple[float, dict[str, object]]:
    if family == "uniform":
        return float(rng.random()), {"mirror": False}
    if family == "log_boundary":
        q = float(math.exp(rng.uniform(math.log(1e-12), math.log(0.5))))
        p, mirror = _mirrored(rng, q)
        return p, {"q": q, "mirror": mirror}
    if family == "event_scale":
        lambda_max = min(100.0, n / 2.0)
        event_lambda = float(
            math.exp(rng.uniform(math.log(1e-6), math.log(lambda_max)))
        )
        q = event_lambda / n
        p, mirror = _mirrored(rng, q)
        return p, {"event_lambda": event_lambda, "q": q, "mirror": mirror}
    if family == "endpoint_neighbor":
        return _endpoint_probability(rng, method, n, alpha)
    raise ValueError(f"unknown p family: {family}")


def _generate_method_design(
    master_seed: str,
    method: str,
    *,
    n_quotas: dict[str, int],
    p_quotas: dict[str, int],
    alpha_values: np.ndarray,
) -> list[dict[str, object]]:
    rng = _method_rng(master_seed, method)
    n_bands = _quota_vector(rng, n_quotas)
    p_families = _quota_vector(rng, p_quotas)
    alpha_values = np.asarray(alpha_values, dtype=np.float64).copy()
    rng.shuffle(alpha_values)
    if not len(n_bands) == len(p_families) == len(alpha_values):
        raise ValueError("holdout quota vectors must have the same length")
    occupied: set[str] = set()
    rows: list[dict[str, object]] = []
    for index, (band, family, alpha) in enumerate(
        zip(n_bands, p_families, alpha_values)
    ):
        n = _sample_n(rng, str(band))
        for duplicate_retry in range(10_000):
            p, provenance = _sample_probability(
                rng, str(family), method, n, float(alpha)
            )
            identity = canonical_cell_id(method, n, float(alpha), p)
            if identity not in occupied:
                break
        else:
            raise RuntimeError(f"duplicate retry exhausted for {method} row {index}")
        occupied.add(identity)
        rows.append(
            {
                "design_index_within_method": index,
                "method": method,
                "interval_kind": (
                    "bayesian_comparator" if method == "jeffreys" else "frequentist"
                ),
                "n": n,
                "n_band": str(band),
                "alpha": float(alpha),
                "p": p,
                "p_hex": p.hex(),
                "p_family": str(family),
                "canonical_cell_id": identity,
                "duplicate_retry": duplicate_retry,
                "endpoint_source_x": provenance.get("endpoint_source_x"),
                "endpoint_kind": provenance.get("endpoint_kind"),
                "endpoint_value": provenance.get("endpoint_value"),
                "endpoint_value_hex": provenance.get("endpoint_value_hex"),
                "endpoint_variant": provenance.get("endpoint_variant"),
                "endpoint_retry": provenance.get("endpoint_retry"),
                "event_lambda": provenance.get("event_lambda"),
                "q": provenance.get("q"),
                "mirror": bool(provenance.get("mirror", False)),
            }
        )
    return rows


def generate_holdout_design(master_seed: str) -> pd.DataFrame:
    """Generate the exact 10,000-cell production holdout after owner seeding."""

    verify_frozen_sources()
    rows: list[dict[str, object]] = []
    for method in H_METHODS:
        alpha_rng = _method_rng(master_seed + "|alpha", method)
        rows.extend(
            _generate_method_design(
                master_seed,
                method,
                n_quotas=H_N_QUOTAS,
                p_quotas=H_P_QUOTAS,
                alpha_values=_alpha_vector(alpha_rng),
            )
        )
    frame = pd.DataFrame(rows)
    if len(frame) != H_TOTAL_CELLS or frame["canonical_cell_id"].duplicated().any():
        raise AssertionError("production holdout must contain 10,000 unique method cells")
    validate_holdout_design(frame, require_production=True)
    return frame


def generate_fixture_holdout_design(master_seed: str) -> pd.DataFrame:
    """Small smoke design; accepts only conspicuously non-production seeds."""

    if not master_seed.startswith("FIXTURE_ONLY:"):
        raise ValueError("fixture seeds must start with 'FIXTURE_ONLY:'")
    rows: list[dict[str, object]] = []
    n_quotas = {"fixture_n_small": 4, "fixture_n_around100": 4}
    p_quotas = {family: 2 for family in H_P_QUOTAS}
    fixture_alphas = np.asarray(
        [float(ALPHAS[index % len(ALPHAS)]) for index in range(8)], dtype=np.float64
    )
    for method in H_METHODS:
        rows.extend(
            _generate_method_design(
                master_seed,
                method,
                n_quotas=n_quotas,
                p_quotas=p_quotas,
                alpha_values=fixture_alphas,
            )
        )
    frame = pd.DataFrame(rows)
    if frame["canonical_cell_id"].duplicated().any():
        raise AssertionError("fixture holdout contains duplicates")
    validate_holdout_design(frame, require_production=False)
    return frame


def canonical_design_hash(frame: pd.DataFrame) -> str:
    required = {"canonical_cell_id", "method", "n", "alpha", "p", "p_hex"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"design missing identity columns: {sorted(missing)}")
    values = frame.sort_values("canonical_cell_id", kind="mergesort")
    digest = hashlib.sha256()
    for row in values.itertuples(index=False):
        p = float(row.p)
        if row.p_hex != p.hex():
            raise ValueError(f"design p_hex mismatch: {row.canonical_cell_id}")
        identity = canonical_cell_id(row.method, int(row.n), float(row.alpha), p)
        if identity != row.canonical_cell_id:
            raise ValueError(f"design identity mismatch: {row.canonical_cell_id}")
        digest.update((identity + "\n").encode("utf-8"))
    return digest.hexdigest()


def _expected_n_band(n: int) -> str:
    if 1 <= n <= 5_000:
        return "n_1_5000"
    if 5_001 <= n <= 100_000:
        return "n_5001_100000"
    if 100_001 <= n <= 1_000_000:
        return "n_100001_1000000"
    raise ValueError(f"holdout n={n} is outside the preregistered domain")


def validate_holdout_design(frame: pd.DataFrame, *, require_production: bool = True) -> None:
    canonical_design_hash(frame)
    if frame["canonical_cell_id"].duplicated().any():
        raise ValueError("holdout design contains duplicate canonical cells")
    if not np.isfinite(frame["p"].to_numpy(dtype=np.float64)).all() or not bool(
        frame["p"].between(0.0, 1.0, inclusive="both").all()
    ):
        raise ValueError("holdout design contains invalid probabilities")
    if not require_production:
        return
    if len(frame) != H_TOTAL_CELLS:
        raise ValueError("production holdout must contain exactly 10,000 cells")
    method_counts = frame["method"].value_counts().to_dict()
    if method_counts != {method: H_CELLS_PER_METHOD for method in H_METHODS}:
        raise ValueError("holdout method quotas do not match the preregistration")
    expected_alphas = set(float(alpha) for alpha in ALPHAS)
    for method in H_METHODS:
        selected = frame.loc[frame["method"] == method]
        if selected["n_band"].value_counts().to_dict() != H_N_QUOTAS:
            raise ValueError(f"holdout n quotas are invalid for {method}")
        actual_bands = selected["n"].astype(int).map(_expected_n_band)
        if not np.array_equal(actual_bands.to_numpy(), selected["n_band"].to_numpy()):
            raise ValueError(f"holdout n-band labels are invalid for {method}")
        if selected["p_family"].value_counts().to_dict() != H_P_QUOTAS:
            raise ValueError(f"holdout p-family quotas are invalid for {method}")
        alpha_counts = selected["alpha"].astype(float).value_counts().to_dict()
        if set(alpha_counts) != expected_alphas or sorted(alpha_counts.values()) != [
            357,
            357,
            357,
            357,
            357,
            357,
            358,
        ]:
            raise ValueError(f"holdout alpha quotas are invalid for {method}")


def verify_design_artifact(
    design_path: Path,
    metadata_path: Path,
    *,
    require_production: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    verify_metadata_content(metadata)
    if metadata.get("design_schema_version") != H_DESIGN_SCHEMA_VERSION:
        raise ValueError("holdout design schema is incompatible")
    identities = {
        "runtime_head": runtime_head(),
        "source_cf_harness_sha": SOURCE_CF_HARNESS_SHA,
        "production_candidate_sha": PRODUCTION_CANDIDATE_SHA,
        "cp04_document_sha": CP04_DOCUMENT_SHA,
    }
    for key, expected in identities.items():
        if metadata.get(key) != expected:
            raise ValueError(f"holdout design metadata mismatch for {key}")
    artifact_hash = sha256_file(Path(design_path))
    if artifact_hash != metadata.get("design_artifact_sha256"):
        raise ValueError("holdout design artifact SHA-256 does not match metadata")
    frame = pd.read_parquet(design_path)
    validate_holdout_design(frame, require_production=require_production)
    if canonical_design_hash(frame) != metadata.get("canonical_design_sha256"):
        raise ValueError("holdout canonical design SHA-256 does not match metadata")
    expected_count = H_TOTAL_CELLS if require_production else len(frame)
    if int(metadata.get("cell_count", -1)) != expected_count or len(frame) != expected_count:
        raise ValueError("holdout design row count does not match metadata")
    return frame, metadata


def _evaluate_one(row: dict[str, object]) -> tuple[dict[str, object], dict[str, object] | None, dict[str, object] | None]:
    method = str(row["method"])
    n = int(row["n"])
    alpha = float(row["alpha"])
    p = float(row["p"])
    identity = canonical_cell_id(method, n, alpha, p)
    common = {
        "canonical_cell_id": identity,
        "method": method,
        "n": n,
        "alpha": alpha,
        "p": p,
        "p_hex": p.hex(),
        "p_family": row.get("p_family"),
        "interval_kind": "bayesian_comparator" if method == "jeffreys" else "frequentist",
    }
    try:
        localization = localize_acceptance(method, n, alpha, p)
        coverage = stable_binomial_coverage(
            n, p, localization.first_x, localization.last_x
        )
        nominal = 1.0 - alpha
        deficit = max(0.0, nominal - coverage)
        proximity = endpoint_proximity_for_localization(
            method, n, alpha, p, localization
        )
        trigger: str | None = None
        if method == "clopper_pearson" and coverage < nominal - 1e-12:
            trigger = "cp_exact_shortfall"
        elif method == "wilson" and deficit > 0.030 and proximity["is_near"]:
            trigger = "severe_or_critical_wilson_and_endpoint_proximity"
        elif method == "wilson" and deficit > 0.030:
            trigger = "severe_or_critical_wilson"
        elif method in {"wilson", "jeffreys"} and proximity["is_near"]:
            trigger = "endpoint_proximity"
        summary = {
            **common,
            "coverage_float64": coverage,
            "nominal": nominal,
            "deficit_float64": deficit,
            "undercoverage_tier": undercoverage_tier(deficit),
            "first_x": localization.first_x,
            "last_x": localization.last_x,
            "acceptance_kind": localization.acceptance_kind,
            "acceptance_runs": json.dumps(localization.runs, separators=(",", ":")),
            "localization_method": localization.localization_method,
            "boundary_validation": localization.boundary_validation,
            "endpoint_near": bool(proximity["is_near"]),
            "nearest_endpoint_distance": float(proximity["nearest_distance"]),
            "hp_trigger": trigger,
            "hp_governs": False,
            "coverage_governing": coverage,
            "classification": (
                "bayesian_comparator"
                if method == "jeffreys"
                else (
                    "observed_statistical_shortfall"
                    if deficit > 0.0
                    else "no_shortfall_float64"
                )
            ),
            "resolved": True,
        }
        audit = None
        failure = None
        if trigger is not None:
            audit = high_precision_recheck(
                method, n, alpha, p, coverage, localization, digits=80
            )
            summary.update(
                {
                    "hp_governs": True,
                    "coverage_governing": float(audit["coverage_hp_float"]),
                    "classification": audit["classification"],
                    "resolved": bool(audit["resolved"]),
                }
            )
            hp_cp_shortfall = (
                method == "clopper_pearson"
                and float(audit["coverage_hp_float"]) < nominal - 1e-12
            )
            if not audit["resolved"] or hp_cp_shortfall:
                failure = {
                    **common,
                    "failure_kind": (
                        "confirmed_cp_exact_shortfall"
                        if hp_cp_shortfall
                        else "unresolved_high_precision_audit"
                    ),
                    "blocking": True,
                    "details": audit.get("notes"),
                }
        return summary, audit, failure
    except Exception as error:
        summary = {
            **common,
            "coverage_float64": math.nan,
            "nominal": 1.0 - alpha,
            "deficit_float64": math.nan,
            "undercoverage_tier": "unresolved",
            "classification": "implementation_or_localization_error",
            "resolved": False,
        }
        failure = {
            **common,
            "failure_kind": "implementation_or_localization_error",
            "blocking": True,
            "details": f"{type(error).__name__}: {error}",
        }
        return summary, None, failure


def evaluate_holdout_frame(design: pd.DataFrame, *, workers: int) -> dict[str, pd.DataFrame]:
    verify_frozen_sources()
    records = design.sort_values("canonical_cell_id", kind="mergesort").to_dict("records")
    if workers == 1:
        evaluated = [_evaluate_one(row) for row in records]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            evaluated = list(executor.map(_evaluate_one, records, chunksize=1))
    summaries = pd.DataFrame([item[0] for item in evaluated])
    audits = pd.DataFrame([item[1] for item in evaluated if item[1] is not None])
    failures = pd.DataFrame([item[2] for item in evaluated if item[2] is not None])
    return {
        "summary": summaries,
        "high_precision_audit": audits,
        "failures": failures,
    }
