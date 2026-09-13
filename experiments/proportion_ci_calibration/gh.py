"""Command line interface for the frozen CP06-G/H implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from experiments.proportion_ci_calibration.gh_common import (
    G_MC_SCHEMA_VERSION,
    G_SELECTION_SCHEMA_VERSION,
    H_DESIGN_SCHEMA_VERSION,
    H_EVALUATION_SCHEMA_VERSION,
    CP04_DOCUMENT_SHA,
    GH_EXPERIMENT_VERSION,
    GH_SCHEMA_VERSION,
    PRODUCTION_CANDIDATE_SHA,
    SOURCE_CF_HARNESS_SHA,
    atomic_json,
    atomic_parquet,
    canonical_cell_id,
    metadata_base,
    runtime_head,
    seal_metadata,
    sha256_file,
    verify_metadata_content,
    verify_frozen_sources,
)
from experiments.proportion_ci_calibration.holdout import (
    H_RNG_ALGORITHM,
    canonical_design_hash,
    evaluate_holdout_frame,
    generate_fixture_holdout_design,
    generate_holdout_design,
    production_holdout_quota_plan,
    verify_design_artifact,
)
from experiments.proportion_ci_calibration.shadow_mc import (
    BROAD_CELLS,
    CRITICAL_CELLS,
    RNG_ALGORITHM,
    TOTAL_DRAWS,
    attach_cf_float64_authority,
    build_g_selection,
    canonical_selection_hash,
    simulate_shadow_cells,
    validate_g_selection,
)


G_SELECTION_NAME = "proportion_ci_cp06_g_selection.parquet"
G_SELECTION_METADATA_NAME = "proportion_ci_cp06_g_selection_metadata.json"
G_MC_NAME = "proportion_ci_cp06_g_mc_shadow.parquet"
G_METADATA_NAME = "proportion_ci_cp06_g_metadata.json"
H_DESIGN_NAME = "proportion_ci_cp06_h_design.parquet"
H_DESIGN_METADATA_NAME = "proportion_ci_cp06_h_design_metadata.json"
H_SUMMARY_NAME = "proportion_ci_cp06_h_holdout_summary.parquet"
H_HP_NAME = "proportion_ci_cp06_h_high_precision_audit.parquet"
H_FAILURES_NAME = "proportion_ci_cp06_h_failures.parquet"
H_METADATA_NAME = "proportion_ci_cp06_h_metadata.json"


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _verify_e_evidence(path: Path, metadata_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "candidate_sha": PRODUCTION_CANDIDATE_SHA,
        "cp04_document_sha": CP04_DOCUMENT_SHA,
        "experiment_version": "proportion-ci-cp06-v3",
        "harness_schema_version": "cp06-harness-schema-v4",
        "shard_schema_version": "cp06-shard-schema-v4",
        "checkpoint": "E",
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"E metadata mismatch for {key}")
    expected = metadata.get("hashes", {}).get(path.name)
    if expected is None or sha256_file(path) != expected:
        raise ValueError("E adversarial minima hash does not match its metadata")
    return pd.read_parquet(path), metadata


def _attach_f_audit(
    selection: pd.DataFrame,
    path: Path,
    metadata_path: Path,
) -> pd.DataFrame:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "candidate_sha": PRODUCTION_CANDIDATE_SHA,
        "cp04_document_sha": CP04_DOCUMENT_SHA,
        "experiment_version": "proportion-ci-cp06-v3",
        "harness_schema_version": "cp06-harness-schema-v4",
        "source_shard_schema_version": "cp06-shard-schema-v4",
        "checkpoint": "F",
        "source_checkpoints": ["C", "D", "E"],
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"F metadata mismatch for {key}: expected {expected_value!r}, "
                f"got {metadata.get(key)!r}"
            )
    if int(metadata.get("digits", 0)) < 80:
        raise ValueError("F metadata records fewer than 80 digits")
    expected = metadata.get("hashes", {}).get(path.name)
    if expected is None or sha256_file(path) != expected:
        raise ValueError("F audit hash does not match its metadata")
    audit = pd.read_parquet(path).copy()
    if int(metadata.get("queue_rows", -1)) != len(audit) or int(
        metadata.get("audit_rows", -1)
    ) != len(audit):
        raise ValueError("F queue/audit row counts do not match the audit artifact")
    if (
        "resolved" not in audit
        or not pd.api.types.is_bool_dtype(audit["resolved"].dtype)
        or not bool(audit["resolved"].fillna(False).all())
    ):
        raise ValueError("F audit contains unresolved rows")
    needed = {
        "audit_kind",
        "method",
        "n",
        "alpha",
        "p",
        "coverage_hp_float",
        "resolved",
        "classification",
        "acceptance_changed",
        "acceptance_runs_float64",
        "acceptance_runs_hp",
    }
    missing = needed - set(audit.columns)
    if missing:
        raise ValueError(f"F audit is missing columns: {sorted(missing)}")
    audit = audit.loc[audit["audit_kind"] == "coverage"].copy()
    audit["canonical_cell_id"] = [
        canonical_cell_id(method, n, alpha, p)
        for method, n, alpha, p in zip(
            audit["method"], audit["n"], audit["alpha"], audit["p"]
        )
    ]
    if audit["canonical_cell_id"].duplicated().any():
        raise ValueError("F contains duplicate coverage evidence for a canonical cell")
    columns = [
        "canonical_cell_id",
        "coverage_hp_float",
        "resolved",
        "classification",
        "acceptance_changed",
        "acceptance_runs_float64",
        "acceptance_runs_hp",
    ]
    result = selection.merge(audit[columns], on="canonical_cell_id", how="left")
    validate_g_selection(result, require_f=True)
    return result


def _verify_g_selection_artifact(
    selection_path: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    verify_metadata_content(metadata)
    expected = {
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
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"G selection metadata mismatch for {key}")
    if metadata.get("runtime_head") != runtime_head():
        raise ValueError("G runtime HEAD differs from selection runtime HEAD")
    if sha256_file(selection_path) != metadata.get("selection_artifact_sha256"):
        raise ValueError("G selection artifact SHA-256 does not match metadata")
    selection = pd.read_parquet(selection_path)
    if canonical_selection_hash(selection) != metadata.get(
        "canonical_selection_sha256"
    ):
        raise ValueError("G canonical selection SHA-256 does not match metadata")
    validate_g_selection(selection, require_authority=True, require_f=True)
    for field in (
        "e_artifact_sha256",
        "e_metadata_sha256",
        "f_audit_sha256",
        "f_metadata_sha256",
    ):
        value = metadata.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"G selection metadata lacks valid {field}")
    return selection, metadata


def g_select(args: argparse.Namespace, command: list[str]) -> None:
    integrity = verify_frozen_sources()
    minima, e_metadata = _verify_e_evidence(args.e_minima, args.e_metadata)
    selection = attach_cf_float64_authority(build_g_selection(minima))
    selection = _attach_f_audit(
        selection, args.f_audit, args.f_metadata
    )
    validate_g_selection(selection, require_authority=True, require_f=True)
    selection["source_cf_harness_sha"] = SOURCE_CF_HARNESS_SHA
    selection["source_e_artifact_sha256"] = sha256_file(args.e_minima)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / G_SELECTION_NAME
    metadata_path = args.output_dir / G_SELECTION_METADATA_NAME
    atomic_parquet(selection, path)
    metadata = metadata_base(
        command=command,
        workers=1,
        master_seed=None,
        output_counts={
            "critical_cells": CRITICAL_CELLS,
            "broad_cells": BROAD_CELLS,
            "selected_cells": len(selection),
            "implied_draws": TOTAL_DRAWS,
        },
    )
    metadata.update(
        {
            "checkpoint": "G-select",
            "selection_schema_version": G_SELECTION_SCHEMA_VERSION,
            "mc_schema_version": G_MC_SCHEMA_VERSION,
            "cf_harness_schema_version": "cp06-harness-schema-v4",
            "cf_shard_schema_version": "cp06-shard-schema-v4",
            "selection_artifact_sha256": sha256_file(path),
            "canonical_selection_sha256": canonical_selection_hash(selection),
            "e_artifact_sha256": sha256_file(args.e_minima),
            "e_metadata_sha256": sha256_file(args.e_metadata),
            "f_audit_sha256": sha256_file(args.f_audit),
            "f_metadata_sha256": sha256_file(args.f_metadata),
            "critical_cells": CRITICAL_CELLS,
            "broad_cells": BROAD_CELLS,
            "implied_draws": TOTAL_DRAWS,
            "source_e_checkpoint_spec_sha256": e_metadata.get(
                "checkpoint_spec_sha256"
            ),
        }
    )
    metadata = seal_metadata(metadata)
    atomic_json(metadata, metadata_path)
    print(
        json.dumps(
            {
                **integrity,
                "selection_schema_version": G_SELECTION_SCHEMA_VERSION,
                "selection_path": str(path),
                "selection_metadata_path": str(metadata_path),
                "selection_sha256": metadata["selection_artifact_sha256"],
                "canonical_selection_sha256": metadata[
                    "canonical_selection_sha256"
                ],
                "critical_cells": CRITICAL_CELLS,
                "broad_cells": BROAD_CELLS,
                "implied_draws": TOTAL_DRAWS,
                "source_e_checkpoint_spec_sha256": e_metadata.get(
                    "checkpoint_spec_sha256"
                ),
            },
            sort_keys=True,
        )
    )


def g_run(args: argparse.Namespace, command: list[str]) -> None:
    verify_frozen_sources()
    selection, selection_metadata = _verify_g_selection_artifact(
        args.selection, args.selection_metadata
    )
    results = simulate_shadow_cells(
        selection,
        master_seed=args.master_seed,
        workers=args.workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mc_path = args.output_dir / G_MC_NAME
    atomic_parquet(results, mc_path)
    hashes = {
        args.selection.name: sha256_file(args.selection),
        args.selection_metadata.name: sha256_file(args.selection_metadata),
        G_MC_NAME: sha256_file(mc_path),
    }
    metadata = metadata_base(
        command=command,
        workers=args.workers,
        master_seed=args.master_seed,
        output_counts={
            "critical_cells": CRITICAL_CELLS,
            "broad_cells": BROAD_CELLS,
            "total_cells": len(results),
            "total_draws": TOTAL_DRAWS,
            "float64_gate_failures": int((~results["float64_mc_gate_pass"]).sum()),
            "hp_gate_failures": int(
                (results["hp_governs"] & ~results["hp_mc_gate_pass"].fillna(True)).sum()
            ),
        },
    )
    metadata.update(
        {
            "checkpoint": "G",
            "selection_schema_version": G_SELECTION_SCHEMA_VERSION,
            "mc_schema_version": G_MC_SCHEMA_VERSION,
            "rng_algorithm": RNG_ALGORITHM,
            "implied_draws": TOTAL_DRAWS,
            "selection_runtime_head": selection_metadata["runtime_head"],
            "selection_artifact_sha256": selection_metadata[
                "selection_artifact_sha256"
            ],
            "canonical_selection_sha256": selection_metadata[
                "canonical_selection_sha256"
            ],
            "artifact_sha256": hashes,
        }
    )
    metadata = seal_metadata(metadata)
    atomic_json(metadata, args.output_dir / G_METADATA_NAME)


def h_generate(args: argparse.Namespace, command: list[str]) -> None:
    verify_frozen_sources()
    design = generate_holdout_design(args.master_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / H_DESIGN_NAME
    atomic_parquet(design, path)
    metadata = metadata_base(
        command=command,
        workers=1,
        master_seed=args.master_seed,
        output_counts={"design_cells": len(design)},
    )
    metadata.update(
        {
            "checkpoint": "H-generate",
            "design_schema_version": H_DESIGN_SCHEMA_VERSION,
            "rng_algorithm": H_RNG_ALGORITHM,
            "quota_plan": production_holdout_quota_plan(),
            "cell_count": len(design),
            "design_artifact_sha256": sha256_file(path),
            "canonical_design_sha256": canonical_design_hash(design),
        }
    )
    metadata = seal_metadata(metadata)
    atomic_json(metadata, args.output_dir / H_DESIGN_METADATA_NAME)


def h_evaluate(args: argparse.Namespace, command: list[str]) -> None:
    verify_frozen_sources()
    design, design_metadata = verify_design_artifact(args.design, args.design_metadata)
    frames = evaluate_holdout_frame(design, workers=args.workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": args.output_dir / H_SUMMARY_NAME,
        "high_precision_audit": args.output_dir / H_HP_NAME,
        "failures": args.output_dir / H_FAILURES_NAME,
    }
    for name, path in paths.items():
        atomic_parquet(frames[name], path)
    hashes = {
        path.name: sha256_file(path) for path in paths.values()
    }
    hashes[args.design.name] = sha256_file(args.design)
    hashes[args.design_metadata.name] = sha256_file(args.design_metadata)
    metadata = metadata_base(
        command=command,
        workers=args.workers,
        master_seed=str(design_metadata["master_seed"]),
        output_counts={
            "design_cells": len(design),
            "evaluated_cells": len(frames["summary"]),
            "high_precision_rechecks": len(frames["high_precision_audit"]),
            "failures": len(frames["failures"]),
            "blocking_failures": int(
                frames["failures"].get("blocking", pd.Series(dtype=bool)).sum()
            ),
        },
    )
    metadata.update(
        {
            "checkpoint": "H-evaluate",
            "design_schema_version": H_DESIGN_SCHEMA_VERSION,
            "evaluation_schema_version": H_EVALUATION_SCHEMA_VERSION,
            "rng_algorithm": design_metadata.get("rng_algorithm"),
            "design_artifact_sha256": design_metadata["design_artifact_sha256"],
            "canonical_design_sha256": design_metadata["canonical_design_sha256"],
            "artifact_sha256": hashes,
        }
    )
    metadata = seal_metadata(metadata)
    atomic_json(metadata, args.output_dir / H_METADATA_NAME)


def fixture_smoke(args: argparse.Namespace, command: list[str]) -> None:
    seed = "FIXTURE_ONLY:CP06-GH-SMOKE-v1"
    design = generate_fixture_holdout_design(seed)
    evaluated = evaluate_holdout_frame(design, workers=args.workers)
    g_fixture = pd.DataFrame(
        [
            {
                "method": "wilson",
                "n": 27,
                "alpha": 0.05,
                "p": 0.2,
                "selection_kind": "critical",
                "canonical_cell_id": canonical_cell_id("wilson", 27, 0.05, 0.2),
            },
            {
                "method": "clopper_pearson",
                "n": 100,
                "alpha": 0.1,
                "p": 0.01,
                "selection_kind": "broad",
                "canonical_cell_id": canonical_cell_id(
                    "clopper_pearson", 100, 0.1, 0.01
                ),
            },
            {
                "method": "wald",
                "n": 10,
                "alpha": 0.2,
                "p": 0.5,
                "selection_kind": "broad",
                "canonical_cell_id": canonical_cell_id("wald", 10, 0.2, 0.5),
            },
        ]
    )
    shadow = simulate_shadow_cells(
        g_fixture,
        master_seed="FIXTURE_ONLY:CP06-G-SMOKE-v1",
        workers=args.workers,
        fixture_reps=5_000,
    )
    print(
        json.dumps(
            {
                "fixture_only": True,
                "h_design_cells": len(design),
                "h_evaluated_cells": len(evaluated["summary"]),
                "h_failures": len(evaluated["failures"]),
                "g_cells": len(shadow),
                "g_draws": int(shadow["reps"].sum()),
                "g_gate_passes": int(shadow["float64_mc_gate_pass"].sum()),
            },
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("g-select")
    select.add_argument("--e-minima", type=Path, required=True)
    select.add_argument("--e-metadata", type=Path, required=True)
    select.add_argument("--f-audit", type=Path, required=True)
    select.add_argument("--f-metadata", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.set_defaults(handler=g_select)

    run = subparsers.add_parser("g-run")
    run.add_argument("--selection", type=Path, required=True)
    run.add_argument("--selection-metadata", type=Path, required=True)
    run.add_argument("--master-seed", required=True)
    run.add_argument("--workers", type=_positive, default=1)
    run.add_argument("--output-dir", type=Path, required=True)
    run.set_defaults(handler=g_run)

    generate = subparsers.add_parser("h-generate")
    generate.add_argument("--master-seed", required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.set_defaults(handler=h_generate)

    evaluate = subparsers.add_parser("h-evaluate")
    evaluate.add_argument("--design", type=Path, required=True)
    evaluate.add_argument("--design-metadata", type=Path, required=True)
    evaluate.add_argument("--workers", type=_positive, default=1)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.set_defaults(handler=h_evaluate)

    smoke = subparsers.add_parser("fixture-smoke")
    smoke.add_argument("--workers", type=_positive, default=1)
    smoke.set_defaults(handler=fixture_smoke)

    args = parser.parse_args(argv)
    command = [sys.executable, "-m", __name__, *(argv if argv is not None else sys.argv[1:])]
    args.handler(args, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
