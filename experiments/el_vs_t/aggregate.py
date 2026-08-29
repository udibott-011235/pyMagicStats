"""Strict, simulation-free aggregation of completed calibration shards."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .scenarios import active_holdout_policy
from .seeds import derive_seed
from .storage import (
    parquet_engine,
    read_frame,
    read_json,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)


def _stable_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_shard_manifests(
    run_manifest: Mapping[str, object],
    shard_manifests: Sequence[Mapping[str, object]],
) -> None:
    """Reject missing, duplicate, unfinished, holdout, or incompatible shards."""

    expected_count = int(run_manifest["num_shards"])
    ids = [int(manifest["shard_id"]) for manifest in shard_manifests]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate shard IDs detected")
    expected_ids = set(range(expected_count))
    if set(ids) != expected_ids:
        raise ValueError(
            f"missing or unexpected shards: expected={sorted(expected_ids)}, found={sorted(ids)}"
        )
    method_digest = _stable_digest(run_manifest["method_versions"])
    compatibility = {
        "run_id": run_manifest["run_id"],
        "repository_sha": run_manifest["repository_sha"],
        "alpha": run_manifest["alpha"],
        "confidence_level": run_manifest["confidence_level"],
        "scenario_registry_digest": run_manifest["scenario_registry_digest"],
        "method_versions_digest": method_digest,
        "seed_derivation_scheme": run_manifest["seed_derivation_scheme"],
        "seed_namespace": run_manifest["seed_namespace"],
        "el_accounting_version": run_manifest["el_accounting_version"],
        "num_shards": run_manifest["num_shards"],
        "storage_format": run_manifest["storage_format"],
    }
    if run_manifest.get("holdout_used") is not False:
        raise ValueError("root manifest does not certify holdout_used=false")
    for manifest in shard_manifests:
        if manifest.get("status") != "complete":
            raise ValueError(f"shard {manifest.get('shard_id')} is not complete")
        if manifest.get("holdout_used") is not False:
            raise ValueError(f"shard {manifest.get('shard_id')} used or obscured holdout status")
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in compatibility.items()
            if manifest.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"incompatible shard {manifest.get('shard_id')} metadata: {mismatches}"
            )


def validate_replicate_ids(
    frame: pd.DataFrame,
    *,
    replicates_per_cell: int,
    num_shards: int,
) -> None:
    """Require each global replicate ID exactly once with valid ownership."""

    ids = frame["replicate_id"].astype(int)
    duplicates = sorted(ids[ids.duplicated(keep=False)].unique().tolist())
    if duplicates:
        raise ValueError(f"duplicate replicate IDs detected: {duplicates[:10]}")
    expected = set(range(int(replicates_per_cell)))
    found = set(ids.tolist())
    if found != expected:
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        raise ValueError(
            f"replicate ID coverage mismatch; missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    invalid_owner = frame.loc[
        ids % int(num_shards) != frame["shard_id"].astype(int),
        ["replicate_id", "shard_id"],
    ]
    if not invalid_owner.empty:
        raise ValueError("replicate IDs violate deterministic shard ownership")


def _proportion(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    denominator = int(numeric.size)
    if denominator == 0:
        return {"denominator": 0, "successes": 0, "rate": math.nan, "mcse": math.nan}
    successes = int(numeric.sum())
    rate = successes / denominator
    return {
        "denominator": denominator,
        "successes": successes,
        "rate": rate,
        "mcse": math.sqrt(rate * (1.0 - rate) / denominator),
    }


def _quantiles(values: pd.Series, prefix: str) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return {
            f"{prefix}_denominator": 0,
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            **{f"{prefix}_q{int(q * 100):02d}": math.nan for q in (0.10, 0.25, 0.50, 0.75, 0.90)},
        }
    result: dict[str, float | int] = {
        f"{prefix}_denominator": int(numeric.size),
        f"{prefix}_mean": float(numeric.mean()),
        f"{prefix}_median": float(numeric.median()),
    }
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        result[f"{prefix}_q{int(q * 100):02d}"] = float(numeric.quantile(q))
    return result


def summarize_cell(frame: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    """Compute per-cell method and paired metrics with explicit denominators."""

    first = frame.iloc[0]
    summary: dict[str, object] = {
        "scenario": str(first["scenario_id"]),
        "family": str(first["family"]),
        "parameters": str(first["parameters_json"]),
        "n": int(first["n"]),
        "R": int(len(frame)),
    }
    t_type1 = _proportion(frame["t_reject"])
    t_coverage = _proportion(frame["t_ci_covers_mu0"])
    summary.update(
        {
            "t_type1_denominator": t_type1["denominator"],
            "t_type1_rejections": t_type1["successes"],
            "t_type1": t_type1["rate"],
            "t_type1_mcse": t_type1["mcse"],
            "t_coverage_denominator": t_coverage["denominator"],
            "t_coverage_successes": t_coverage["successes"],
            "t_coverage": t_coverage["rate"],
            "t_coverage_mcse": t_coverage["mcse"],
            "t_test_numerical_failure_rate": float(
                frame["t_test_numerical_failure"].astype(float).mean()
            ),
            "t_ci_numerical_failure_rate": float(
                frame["t_ci_numerical_failure"].astype(float).mean()
            ),
        }
    )
    t_widths = _quantiles(frame["t_ci_width"], "t_ci_width")
    summary.update(t_widths)
    summary["t_mean_width"] = t_widths["t_ci_width_mean"]
    summary["t_median_width"] = t_widths["t_ci_width_median"]

    el_type1_unconditional = _proportion(frame["el_reject_unconditional"])
    el_type1_regular = _proportion(frame["el_reject_regular"])
    el_coverage_unconditional = _proportion(
        frame["el_ci_covers_mu0_unconditional"]
    )
    el_coverage_regular = _proportion(frame["el_ci_covers_mu0_regular"])
    expected_type1_denominator = int(
        frame["el_type1_unconditional_eligible"].astype(int).sum()
    )
    expected_coverage_denominator = int(
        frame["el_coverage_unconditional_eligible"].astype(int).sum()
    )
    if el_type1_unconditional["denominator"] != expected_type1_denominator:
        raise ValueError("EL unconditional Type-I eligibility/denominator mismatch")
    if el_coverage_unconditional["denominator"] != expected_coverage_denominator:
        raise ValueError("EL unconditional coverage eligibility/denominator mismatch")
    summary.update(
        {
            "el_type1_unconditional_denominator": el_type1_unconditional["denominator"],
            "el_type1_unconditional_rejections": el_type1_unconditional["successes"],
            "el_type1_unconditional": el_type1_unconditional["rate"],
            "el_type1_unconditional_mcse": el_type1_unconditional["mcse"],
            "el_type1_regular_denominator": el_type1_regular["denominator"],
            "el_type1_regular_rejections": el_type1_regular["successes"],
            "el_type1_regular": el_type1_regular["rate"],
            "el_type1_regular_mcse": el_type1_regular["mcse"],
            "el_coverage_unconditional_denominator": el_coverage_unconditional["denominator"],
            "el_coverage_unconditional_successes": el_coverage_unconditional["successes"],
            "el_coverage_unconditional": el_coverage_unconditional["rate"],
            "el_coverage_unconditional_mcse": el_coverage_unconditional["mcse"],
            "el_coverage_regular_denominator": el_coverage_regular["denominator"],
            "el_coverage_regular_successes": el_coverage_regular["successes"],
            "el_coverage_regular": el_coverage_regular["rate"],
            "el_coverage_regular_mcse": el_coverage_regular["mcse"],
            "el_test_numerical_failure_rate": float(
                frame["el_test_numerical_failure"].astype(float).mean()
            ),
            "el_ci_numerical_failure_rate": float(
                frame["el_ci_numerical_failure"].astype(float).mean()
            ),
        }
    )
    el_widths = _quantiles(frame["el_ci_width"], "el_ci_width")
    summary.update(el_widths)
    summary["el_mean_width"] = el_widths["el_ci_width_mean"]
    summary["el_median_width"] = el_widths["el_ci_width_median"]

    summary.update(
        {
            "el_regular_rate": float(frame["el_regular"].astype(float).mean()),
            "el_ci_available_rate": float(frame["el_ci_available"].astype(float).mean()),
            "el_hull_outside_rate": float(frame["el_hull_outside"].astype(float).mean()),
            "el_boundary_rate": float(frame["el_boundary"].astype(float).mean()),
            "el_nonregular_rate": float((frame["el_regular"].astype(int) == 0).mean()),
            "el_solver_failure_rate": float(frame["el_solver_failure"].astype(float).mean()),
        }
    )

    disagreement: dict[str, object] = {
        "scenario": summary["scenario"],
        "family": summary["family"],
        "parameters": summary["parameters"],
        "n": summary["n"],
        "R": summary["R"],
    }
    rejection_valid = frame["t_reject"].notna() & frame["el_reject_unconditional"].notna()
    t_reject = frame.loc[rejection_valid, "t_reject"].astype(int)
    el_reject = frame.loc[rejection_valid, "el_reject_unconditional"].astype(int)
    rejection_categories = {
        "both_reject_unconditional": (t_reject == 1) & (el_reject == 1),
        "t_only_reject_unconditional": (t_reject == 1) & (el_reject == 0),
        "el_only_reject_unconditional": (t_reject == 0) & (el_reject == 1),
        "neither_reject_unconditional": (t_reject == 0) & (el_reject == 0),
    }
    disagreement["rejection_unconditional_pair_denominator"] = int(rejection_valid.sum())
    for name, mask in rejection_categories.items():
        count = int(mask.sum())
        disagreement[f"{name}_count"] = count
        disagreement[f"{name}_rate"] = (
            count / int(rejection_valid.sum()) if rejection_valid.any() else math.nan
        )
    disagreement["rejection_unconditional_rate_difference_el_minus_t"] = (
        float((el_reject - t_reject).mean()) if rejection_valid.any() else math.nan
    )

    coverage_valid = frame["t_ci_covers_mu0"].notna() & frame[
        "el_ci_covers_mu0_unconditional"
    ].notna()
    t_cover = frame.loc[coverage_valid, "t_ci_covers_mu0"].astype(int)
    el_cover = frame.loc[coverage_valid, "el_ci_covers_mu0_unconditional"].astype(int)
    coverage_categories = {
        "both_cover_unconditional": (t_cover == 1) & (el_cover == 1),
        "t_only_cover_unconditional": (t_cover == 1) & (el_cover == 0),
        "el_only_cover_unconditional": (t_cover == 0) & (el_cover == 1),
        "neither_cover_unconditional": (t_cover == 0) & (el_cover == 0),
    }
    disagreement["coverage_unconditional_pair_denominator"] = int(coverage_valid.sum())
    for name, mask in coverage_categories.items():
        count = int(mask.sum())
        disagreement[f"{name}_count"] = count
        disagreement[f"{name}_rate"] = count / int(coverage_valid.sum()) if coverage_valid.any() else math.nan
    disagreement["coverage_unconditional_rate_difference_el_minus_t"] = (
        float((el_cover - t_cover).mean()) if coverage_valid.any() else math.nan
    )

    t_width = pd.to_numeric(frame["t_ci_width"], errors="coerce")
    el_width = pd.to_numeric(frame["el_ci_width"], errors="coerce")
    width_valid = np.isfinite(t_width) & np.isfinite(el_width) & (t_width > 0.0)
    ratios = el_width[width_valid] / t_width[width_valid]
    differences = el_width[width_valid] - t_width[width_valid]
    disagreement.update(_quantiles(ratios, "width_ratio"))
    disagreement["width_pair_denominator"] = int(width_valid.sum())
    disagreement["width_difference_el_minus_t_mean"] = (
        float(differences.mean()) if width_valid.any() else math.nan
    )
    disagreement["width_difference_el_minus_t_median"] = (
        float(differences.median()) if width_valid.any() else math.nan
    )
    summary["width_ratio_median"] = disagreement["width_ratio_median"]
    return summary, disagreement


def _load_cell(
    input_dir: Path,
    run_manifest: Mapping[str, object],
    cell: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, float]]:
    frames: list[pd.DataFrame] = []
    timings: dict[str, float] = {}
    storage_format = str(run_manifest["storage_format"])
    parquet_backend = parquet_engine() if storage_format == "parquet" else None
    for shard_id in range(int(run_manifest["num_shards"])):
        cell_dir = (
            input_dir
            / "shards"
            / f"shard_{shard_id:03d}"
            / str(cell["scenario_id"])
            / f"n_{int(cell['n']):05d}"
        )
        success_path = cell_dir / "_SUCCESS.json"
        if not success_path.is_file():
            raise ValueError(f"missing cell completion marker: {success_path}")
        success = read_json(success_path)
        if success.get("run_id") != run_manifest["run_id"] or success.get("holdout_used") is not False:
            raise ValueError(f"incompatible cell completion marker: {success_path}")
        markers = sorted(cell_dir.glob("block_*.complete.json"))
        if len(markers) != int(success["expected_blocks"]):
            raise ValueError(f"missing block completion marker(s) in {cell_dir}")
        for marker_path in markers:
            marker = read_json(marker_path)
            data_path = marker_path.parent / str(marker["data_file"])
            if marker.get("run_id") != run_manifest["run_id"]:
                raise ValueError(f"block belongs to a different run: {marker_path}")
            if marker.get("holdout_used") is not False:
                raise ValueError(f"block does not certify holdout_used=false: {marker_path}")
            if not data_path.is_file() or sha256_file(data_path) != marker.get("sha256"):
                raise ValueError(f"missing or corrupted completed block: {data_path}")
            frame = read_frame(
                data_path,
                storage_format,
                str(parquet_backend) if parquet_backend is not None else None,
            )
            if len(frame) != int(marker["rows"]):
                raise ValueError(f"row count mismatch: {data_path}")
            frames.append(frame)
            for key, value in dict(marker["stage_timings_seconds"]).items():
                timings[key] = timings.get(key, 0.0) + float(value)
    if not frames:
        raise ValueError(f"calibration cell has no replicate data: {cell['cell_id']}")
    return pd.concat(frames, ignore_index=True), timings


def _validate_cell_frame(
    frame: pd.DataFrame,
    run_manifest: Mapping[str, object],
    cell: Mapping[str, object],
) -> None:
    if set(frame["scenario_id"].astype(str)) != {str(cell["scenario_id"])}:
        raise ValueError("scenario IDs do not match the cell manifest")
    if set(frame["family"].astype(str)) != {str(cell["family"])}:
        raise ValueError("distribution families do not match the cell manifest")
    expected_parameters = json.dumps(
        cell["parameters"], sort_keys=True, separators=(",", ":")
    )
    if set(frame["parameters_json"].astype(str)) != {expected_parameters}:
        raise ValueError("distribution parameters do not match the cell manifest")
    if set(frame["n"].astype(int)) != {int(cell["n"])}:
        raise ValueError("sample sizes do not match the cell manifest")
    mu0 = pd.to_numeric(frame["mu0"], errors="coerce").to_numpy(dtype=float)
    if not np.all(mu0 == float(cell["population_mean"])):
        raise ValueError("population means do not match the cell manifest")
    if set(frame["num_shards"].astype(int)) != {int(run_manifest["num_shards"])}:
        raise ValueError("replicate rows report incompatible num_shards")
    validate_replicate_ids(
        frame,
        replicates_per_cell=int(run_manifest["replicates_per_cell"]),
        num_shards=int(run_manifest["num_shards"]),
    )
    for row in frame.loc[:, ["scenario_id", "replicate_id", "seed_identity"]].itertuples(index=False):
        expected = derive_seed(
            int(run_manifest["master_seed"]),
            str(row.scenario_id),
            int(row.replicate_id),
        ).identity
        if str(row.seed_identity) != expected:
            raise ValueError(
                f"seed identity mismatch for {row.scenario_id} replicate {row.replicate_id}"
            )


def _report(summary: pd.DataFrame, metadata_name: str) -> str:
    total = int(summary["R"].sum())
    return "\n".join(
        (
            "# Student t versus uncorrected empirical likelihood calibration",
            "",
            "This report was generated mechanically from validated shard outputs; no LLM or external API was used.",
            "",
            "## OBSERVATION",
            "",
            f"Validated {len(summary)} canonical calibration cells containing {total} paired replicates.",
            "Each replicate applied Student t and raw empirical likelihood to the same generated sample.",
            "PRIMARY EL metrics are `el_type1_unconditional` and `el_coverage_unconditional` with explicit eligible-replicate denominators.",
            "A null mean outside the sample hull is an unconditional EL rejection and an unconditional CI noncoverage.",
            "DIAGNOSTIC metrics `el_type1_regular` and `el_coverage_regular` condition on a regular converged EL evaluation and regular available CI.",
            "Numerical failures are exposed separately and reduce a denominator only when no hull-based outcome is independently determined.",
            "Width quantiles, failure rates, hull evidence, and regular/available rates are in `el_vs_t_summary.csv`.",
            "Paired unconditional rejection/coverage and regular finite interval-width outcomes are in `el_vs_t_disagreement.csv`.",
            f"Provenance and compatibility checks are recorded in `{metadata_name}`.",
            "",
            "## INTERPRETATION",
            "",
            "The outputs quantify Monte Carlo behavior under the named true simulation scenarios.",
            "Differences are descriptive calibration evidence and must be interpreted with their MCSEs, relevant denominators, and numerical-failure rates.",
            "",
            "## POLICY — NOT DETERMINED",
            "",
            "This experiment does not define routing rules, sample-shape thresholds, or automatic method selection.",
            "Empirical likelihood remains uncalibrated for automatic routing.",
            "",
        )
    )


def aggregate_calibration(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Validate every fragment and aggregate without generating any samples."""

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    run_manifest = read_json(input_path / "run_manifest.json")
    shard_root = input_path / "shards"
    discovered = {path.name for path in shard_root.glob("shard_*") if path.is_dir()}
    expected_directories = {
        f"shard_{shard_id:03d}" for shard_id in range(int(run_manifest["num_shards"]))
    }
    if discovered != expected_directories:
        raise ValueError(
            "missing or unexpected shard directories: "
            f"expected={sorted(expected_directories)}, found={sorted(discovered)}"
        )
    shard_manifests = []
    for shard_id in range(int(run_manifest["num_shards"])):
        path = input_path / "shards" / f"shard_{shard_id:03d}" / "shard_manifest.json"
        if not path.is_file():
            raise ValueError(f"missing shard manifest: {path}")
        shard_manifests.append(read_json(path))
    validate_shard_manifests(run_manifest, shard_manifests)

    summaries: list[dict[str, object]] = []
    disagreements: list[dict[str, object]] = []
    aggregate_timings: dict[str, float] = {}
    for cell in run_manifest["cells"]:
        frame, timings = _load_cell(input_path, run_manifest, cell)
        _validate_cell_frame(frame, run_manifest, cell)
        summary, disagreement = summarize_cell(frame)
        summaries.append(summary)
        disagreements.append(disagreement)
        for key, value in timings.items():
            aggregate_timings[key] = aggregate_timings.get(key, 0.0) + value

    summary_frame = pd.DataFrame(summaries).sort_values(["scenario", "n"])
    disagreement_frame = pd.DataFrame(disagreements).sort_values(["scenario", "n"])
    output_path.mkdir(parents=True, exist_ok=True)
    summary_name = "el_vs_t_summary.csv"
    disagreement_name = "el_vs_t_disagreement.csv"
    metadata_name = "el_vs_t_metadata.json"
    write_text_atomic(output_path / summary_name, summary_frame.to_csv(index=False, lineterminator="\n"))
    write_text_atomic(
        output_path / disagreement_name,
        disagreement_frame.to_csv(index=False, lineterminator="\n"),
    )
    metadata = {
        "schema_version": "el-vs-t-aggregate-v1",
        "run_id": run_manifest["run_id"],
        "repository_sha": run_manifest["repository_sha"],
        "alpha": run_manifest["alpha"],
        "confidence_level": run_manifest["confidence_level"],
        "scenario_registry_digest": run_manifest["scenario_registry_digest"],
        "el_accounting_version": run_manifest["el_accounting_version"],
        "el_accounting_semantics": {
            "primary": ["el_type1_unconditional", "el_coverage_unconditional"],
            "diagnostic": ["el_type1_regular", "el_coverage_regular"],
            "hull_outside_type1": "rejection",
            "hull_outside_coverage": "noncoverage",
            "numerical_failures": (
                "reported separately; denominator exclusion occurs only when no "
                "independent hull-based outcome is determined"
            ),
        },
        "method_versions": run_manifest["method_versions"],
        "replicates_per_cell": run_manifest["replicates_per_cell"],
        "num_shards": run_manifest["num_shards"],
        "cell_count": len(summary_frame),
        "paired_replicate_rows": int(summary_frame["R"].sum()),
        "backend_metadata_by_shard": [manifest["backend"] for manifest in shard_manifests],
        "storage_metadata_by_shard": [
            {
                "pandas_version": manifest.get("pandas_version"),
                "parquet_engine_used": manifest.get("parquet_engine_used"),
            }
            for manifest in shard_manifests
        ],
        "stage_timings_seconds": aggregate_timings,
        "holdout_used": False,
        "holdout_exclusion_policy": active_holdout_policy(),
        "network_used": False,
        "llm_or_external_api_used": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [summary_name, disagreement_name, metadata_name, "el_vs_t_report.md"],
    }
    write_json_atomic(output_path / metadata_name, metadata)
    write_text_atomic(output_path / "el_vs_t_report.md", _report(summary_frame, metadata_name))
    return metadata
