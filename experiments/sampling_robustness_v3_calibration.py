"""Audit SamplingRobustnessV3 on the existing calibration set only.

This script must not receive a holdout or validation dataset.  It derives the
candidate transition anchors from the existing confirmatory calibration rows,
verifies the constants embedded in v3, and compares v2 with two v3 views:

``v3_default``
    No external model or process knowledge (the API default).

``v3_oracle_simulation_truth``
    Oracle sensitivity context supplied from the true simulation generator.
    It is excluded from headline real-world safety claims because callers do
    not ordinarily know their data-generating process.

The per-replicate input is reproduced by
``experiments/adversarial_robustness_calibration.py`` and is intentionally
ignored by Git because it is large.  No production code loads this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy

from experiments.adversarial_robustness_calibration import wilson_interval
from pyMagicStat.assumptions import (
    Assessment,
    AssessmentStatus,
    AssumptionProvenance,
    AssumptionReport,
    CalibrationAnchors,
    Estimand,
    InferenceDesign,
    ProcessUncertainty,
    SamplingRobustness,
    SamplingRobustnessV3,
)
from pyMagicStat.assumptions.robustness_v3 import DEFAULT_CALIBRATION_ANCHORS


SAFE_REFERENCE_CELLS = (
    ("normal", 30),
    ("normal", 100),
    ("normal", 750),
    ("normal", 2000),
    ("normal", 10000),
    ("student_t_df_5", 20),
    ("bimodal_symmetric", 300),
)
ADVERSE_REFERENCE_CELLS = (
    ("lognormal_sigma_1.00", 30),
    ("contamination_asymmetric_eps_0p01", 100),
    ("contamination_asymmetric_eps_0p025", 100),
)
SPECIAL_COMPARISON_CELLS = (
    ("normal", 3),
    ("normal", 30),
    ("normal", 10000),
    ("student_t_df_5", 20),
    ("bimodal_symmetric", 300),
    ("lognormal_sigma_0.25", 20),
    ("lognormal_sigma_0.50", 50),
    ("lognormal_sigma_1.00", 30),
    ("contamination_asymmetric_eps_0p005", 100),
    ("contamination_asymmetric_eps_0p01", 100),
    ("contamination_asymmetric_eps_0p025", 100),
    ("contamination_symmetric_eps_0p1", 100),
)
V3_DEFAULT_PROFILE = "v3_default"
V3_ORACLE_PROFILE = "v3_oracle_simulation_truth"


def _cell_mask(frame: pd.DataFrame, cells: Sequence[tuple[str, int]]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for scenario, n in cells:
        mask |= (frame["scenario"] == scenario) & (frame["n"] == n)
    return mask


def derive_calibration_anchors(replicates: pd.DataFrame) -> CalibrationAnchors:
    """Derive transition anchors without a fitted weighted risk formula.

    The compatible endpoint covers 90% of pooled observations from confirmed
    target-conforming cells.  The adverse endpoint is the first quartile of
    pooled observations from clearly deficient cells.  The interval between
    endpoints is deliberately CAUTION rather than an optimized hard separator.
    """

    safe = replicates.loc[_cell_mask(replicates, SAFE_REFERENCE_CELLS)]
    adverse = replicates.loc[_cell_mask(replicates, ADVERSE_REFERENCE_CELLS)]
    if len(safe) != 45_000 or len(adverse) != 20_000:
        raise ValueError(
            "The calibration reference cells are incomplete; expected 45,000 safe "
            "and 20,000 adverse replicate rows"
        )

    def features(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        return (
            frame["skewness"].abs(),
            frame["excess_kurtosis"].clip(lower=0.0),
            frame["influence_ratio"],
        )

    safe_skew, safe_kurtosis, safe_influence = features(safe)
    adverse_skew, adverse_kurtosis, adverse_influence = features(adverse)
    return CalibrationAnchors(
        skewness_compatible=round(float(safe_skew.quantile(0.90)), 6),
        skewness_adverse=round(float(adverse_skew.quantile(0.25)), 6),
        positive_kurtosis_compatible=round(float(safe_kurtosis.quantile(0.90)), 6),
        positive_kurtosis_adverse=round(float(adverse_kurtosis.quantile(0.25)), 6),
        influence_compatible=round(float(safe_influence.quantile(0.90)), 6),
        influence_elevated=round(float(adverse_influence.quantile(0.25)), 6),
    )


def _assessment_report(row: object) -> AssumptionReport:
    shape_status = AssessmentStatus(str(row.shape_status))
    outlier_status = AssessmentStatus(str(row.outlier_status))
    exact_rejected = row.exact_normality_rejected
    if pd.isna(exact_rejected):
        exact_rejected = None
    else:
        exact_rejected = bool(exact_rejected)
    return AssumptionReport(
        design=InferenceDesign.ONE_SAMPLE,
        estimand=Estimand.MEAN,
        assessments={
            "data_quality": Assessment(
                "data_quality_sample",
                AssessmentStatus.PASS,
                {"n": int(row.n)},
            ),
            "shape": Assessment(
                "shape_sample",
                shape_status,
                {
                    "n": int(row.n),
                    "skewness": float(row.skewness),
                    "excess_kurtosis": float(row.excess_kurtosis),
                    "departure_magnitude": str(row.departure_magnitude),
                    "exact_normality_rejected": exact_rejected,
                },
            ),
            "outliers": Assessment(
                "outliers_sample",
                outlier_status,
                {
                    "count": int(row.extreme_count),
                    "fraction": float(row.extreme_fraction),
                    "influence_ratio": float(row.influence_ratio),
                },
            ),
            "independence": Assessment(
                "independence",
                AssessmentStatus.PASS,
                {"independence": "assumed"},
            ),
        },
    )


def _oracle_policy(scenario: str, family: str) -> SamplingRobustnessV3:
    """Supply simulation-generator truth for an oracle sensitivity profile."""

    process_elevated = (
        family == "lognormal"
        or family == "normal_contamination_asymmetric"
        or scenario in {"bimodal_asymmetric", "mixture_distinct_means"}
    )
    return SamplingRobustnessV3(
        model_provenance=AssumptionProvenance.EXTERNAL,
        process_uncertainty=(
            ProcessUncertainty.ELEVATED if process_elevated else ProcessUncertainty.LOW
        ),
    )


def classify_replicates(replicates: pd.DataFrame) -> pd.DataFrame:
    """Return rows plus legacy, sample-default, and oracle classifications."""

    output = replicates.copy()
    output["v2"] = output["sampling_robustness_level"]
    default_policy = SamplingRobustnessV3()
    scenario_policies: dict[tuple[str, str], SamplingRobustnessV3] = {}
    default_levels: list[str] = []
    contextual_levels: list[str] = []
    for row in output.itertuples(index=False):
        report = _assessment_report(row)
        default_levels.append(default_policy.evaluate(report).level.value)
        key = (str(row.scenario), str(row.distribution_family))
        policy = scenario_policies.get(key)
        if policy is None:
            policy = _oracle_policy(*key)
            scenario_policies[key] = policy
        contextual_levels.append(policy.evaluate(report).level.value)
    output[V3_DEFAULT_PROFILE] = default_levels
    output[V3_ORACLE_PROFILE] = contextual_levels
    return output


def summarize_classifications(classified: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    policies = ("v2", V3_DEFAULT_PROFILE, V3_ORACLE_PROFILE)
    for (scenario, family, n), cell in classified.groupby(
        ["scenario", "distribution_family", "n"],
        sort=True,
    ):
        total = len(cell)
        for policy in policies:
            for level in ("all", "acceptable", "caution", "insufficient"):
                selected = cell if level == "all" else cell[cell[policy] == level]
                denominator = len(selected)
                classification_count = total if level == "all" else denominator
                rejections = int(selected["reject_h0"].sum())
                covered = int(selected["ci_contains_true_mean"].sum())
                class_low, class_high = wilson_interval(classification_count, total)
                type_low, type_high = wilson_interval(rejections, denominator)
                coverage_low, coverage_high = wilson_interval(covered, denominator)
                rows.append(
                    {
                        "policy": policy,
                        "scenario": scenario,
                        "distribution_family": family,
                        "n": int(n),
                        "level": level,
                        "total_replications": total,
                        "conditional_denominator": denominator,
                        "classification_rate": classification_count / total,
                        "classification_ci95_lower": class_low,
                        "classification_ci95_upper": class_high,
                        "type_i_error": rejections / denominator if denominator else np.nan,
                        "type_i_ci95_lower": type_low,
                        "type_i_ci95_upper": type_high,
                        "ci_coverage": covered / denominator if denominator else np.nan,
                        "coverage_ci95_lower": coverage_low,
                        "coverage_ci95_upper": coverage_high,
                    }
                )
    return rows


def flag_operating_regions(summary: pd.DataFrame) -> pd.DataFrame:
    """Label false-safe/false-insufficient proxies using provisional targets."""

    conditional = summary[summary["level"] != "all"].copy()
    conditional["denominator_sufficient"] = conditional["conditional_denominator"] >= 200
    conditional["confirmatory_cell"] = conditional["total_replications"] >= 5_000
    policy_totals = (
        summary[summary["level"] == "all"]
        .groupby("policy")["total_replications"]
        .sum()
    )
    acceptable_totals = (
        conditional[conditional["level"] == "acceptable"]
        .groupby("policy")["conditional_denominator"]
        .sum()
    )
    conditional["policy_total_replications"] = conditional["policy"].map(policy_totals)
    conditional["acceptable_denominator"] = (
        conditional["policy"].map(acceptable_totals).fillna(0).astype(int)
    )
    conditional["overall_acceptable_rate"] = (
        conditional["acceptable_denominator"]
        / conditional["policy_total_replications"]
    )
    meets_target = (
        (conditional["type_i_error"] <= SamplingRobustnessV3.CALIBRATION_TYPE_I_TARGET)
        & (conditional["ci_coverage"] >= SamplingRobustnessV3.CALIBRATION_COVERAGE_TARGET)
    )
    conditional["operating_region"] = "unclassified"
    conditional["false_safe_evaluation"] = "not_applicable"
    acceptable = conditional["level"] == "acceptable"
    vacuous = acceptable & (conditional["conditional_denominator"] == 0)
    underpowered = (
        acceptable
        & (conditional["conditional_denominator"] > 0)
        & ~conditional["denominator_sufficient"]
    )
    evaluable = acceptable & conditional["denominator_sufficient"]
    conditional.loc[
        vacuous,
        ["operating_region", "false_safe_evaluation"],
    ] = ["not_evaluable_vacuous", "NOT EVALUABLE / VACUOUS"]
    conditional.loc[
        underpowered,
        ["operating_region", "false_safe_evaluation"],
    ] = [
        "not_evaluable_insufficient_denominator",
        "NOT EVALUABLE / INSUFFICIENT DENOMINATOR",
    ]
    conditional.loc[evaluable, "false_safe_evaluation"] = "EVALUATED"
    conditional.loc[
        evaluable & ~meets_target,
        "operating_region",
    ] = "false_safe"
    conditional.loc[
        (conditional["level"] == "insufficient") & meets_target,
        "operating_region",
    ] = "false_insufficient"
    return conditional[conditional["operating_region"] != "unclassified"]


def summarize_false_safe_metrics(summary: pd.DataFrame) -> list[dict[str, object]]:
    """Report false-safe evidence with exposure and headline eligibility."""

    flagged = flag_operating_regions(summary)
    total_rows = summary[summary["level"] == "all"]
    acceptable_rows = summary[summary["level"] == "acceptable"]
    rows: list[dict[str, object]] = []
    for policy in ("v2", V3_DEFAULT_PROFILE, V3_ORACLE_PROFILE):
        total_replications = int(
            total_rows.loc[
                total_rows["policy"] == policy,
                "total_replications",
            ].sum()
        )
        acceptable_denominator = int(
            acceptable_rows.loc[
                acceptable_rows["policy"] == policy,
                "conditional_denominator",
            ].sum()
        )
        acceptable_rate = (
            acceptable_denominator / total_replications
            if total_replications
            else 0.0
        )
        is_oracle = policy == V3_ORACLE_PROFILE
        if acceptable_denominator == 0:
            false_safe_regions: int | None = None
            evaluation = "NOT EVALUABLE / VACUOUS"
        else:
            reliable_false_safe = flagged[
                (flagged["policy"] == policy)
                & (flagged["operating_region"] == "false_safe")
                & flagged["denominator_sufficient"]
                & flagged["confirmatory_cell"]
            ]
            false_safe_regions = int(len(reliable_false_safe))
            evaluation = "EVALUATED"
        rows.append(
            {
                "policy": policy,
                "profile_kind": (
                    "oracle_simulation_truth"
                    if is_oracle
                    else "sample_default"
                    if policy == V3_DEFAULT_PROFILE
                    else "legacy"
                ),
                "headline_eligible": not is_oracle,
                "total_replications": total_replications,
                "acceptable_denominator": acceptable_denominator,
                "overall_acceptable_rate": acceptable_rate,
                "confirmatory_false_safe_regions": false_safe_regions,
                "false_safe_evaluation": evaluation,
                "supports_zero_false_safe_claim": (
                    not is_oracle
                    and acceptable_denominator > 0
                    and false_safe_regions == 0
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def run(input_path: Path, output_dir: Path) -> dict[str, object]:
    expected_input = Path(
        "experiments/results/robustness_calibration_replicates.csv.gz"
    ).resolve()
    if input_path.resolve() != expected_input:
        raise ValueError(
            "Only experiments/results/robustness_calibration_replicates.csv.gz "
            "from the existing calibration is allowed"
        )
    if not input_path.is_file():
        raise FileNotFoundError(
            f"{input_path} is missing; reproduce it with "
            "python -m experiments.adversarial_robustness_calibration --profile calibration"
        )
    replicates = pd.read_csv(input_path)
    anchors = derive_calibration_anchors(replicates)
    if asdict(anchors) != asdict(DEFAULT_CALIBRATION_ANCHORS):
        raise AssertionError(
            f"Embedded v3 anchors do not match calibration derivation: {anchors!r}"
        )

    classified = classify_replicates(replicates)
    summary_rows = summarize_classifications(classified)
    summary = pd.DataFrame(summary_rows)
    flagged = flag_operating_regions(summary)
    false_safe_metrics = summarize_false_safe_metrics(summary)
    special_mask = pd.Series(False, index=summary.index)
    for scenario, n in SPECIAL_COMPARISON_CELLS:
        special_mask |= (summary["scenario"] == scenario) & (summary["n"] == n)
    special = summary[special_mask & summary["level"].isin(["acceptable", "caution", "insufficient"])]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "sampling_robustness_v3_comparison.csv", index=False)
    flagged.to_csv(output_dir / "sampling_robustness_v3_flagged_regions.csv", index=False)
    pd.DataFrame(false_safe_metrics).to_csv(
        output_dir / "sampling_robustness_v3_false_safe_metrics.csv",
        index=False,
    )
    special.to_csv(output_dir / "sampling_robustness_v3_special_cells.csv", index=False)

    metadata = {
        "candidate_policy": SamplingRobustnessV3.POLICY_VERSION,
        "legacy_policy": SamplingRobustness.POLICY_VERSION,
        "calibration_input": str(input_path.as_posix()),
        "holdout_used": False,
        "calibration_only": True,
        "anchors": asdict(anchors),
        "anchor_derivation": {
            "compatible_quantile": 0.90,
            "adverse_quantile": 0.25,
            "safe_reference_cells": [list(item) for item in SAFE_REFERENCE_CELLS],
            "adverse_reference_cells": [list(item) for item in ADVERSE_REFERENCE_CELLS],
        },
        "provisional_targets": {
            "acceptable_type_i_max": SamplingRobustnessV3.CALIBRATION_TYPE_I_TARGET,
            "acceptable_coverage_min": SamplingRobustnessV3.CALIBRATION_COVERAGE_TARGET,
        },
        "false_safe_metrics": false_safe_metrics,
        "replications": len(replicates),
        "cell_count": int(replicates.groupby(["scenario", "n"]).ngroups),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": {
            V3_DEFAULT_PROFILE: {
                "profile_kind": "sample_default",
                "headline_eligible": True,
                "model_provenance": "unknown",
                "process_uncertainty": "unknown",
            },
            "realistic_external_context": {
                "available": False,
                "reason": (
                    "No external study metadata independent of simulation-generator "
                    "truth is available in this calibration artifact."
                ),
            },
            V3_ORACLE_PROFILE: {
                "profile_kind": "oracle_simulation_truth",
                "headline_eligible": False,
                "uses_true_generator_knowledge": True,
                "description": (
                    "Sensitivity profile using true generator family/scenario to set "
                    "external provenance and process uncertainty."
                ),
            },
        },
    }
    (output_dir / "sampling_robustness_v3_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "replications": len(replicates),
                "summary_rows": len(summary),
                "flagged_rows": len(flagged),
                "holdout_used": False,
            }
        )
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/results/robustness_calibration_replicates.csv.gz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
    )
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
