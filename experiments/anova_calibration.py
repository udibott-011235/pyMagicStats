"""Reproducible calibration matrix for independent one-way mean inference."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from pyMagicStat.assumptions import InferenceValidator, OneWayRobustness, RobustnessLevel
from pyMagicStat.inference.anova import (
    _classical_anova_statistics,
    _welch_anova_statistics,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    size_multipliers: Tuple[float, ...] = (1.0, 1.0, 1.0)
    scale_multipliers: Tuple[float, ...] = (1.0, 1.0, 1.0)
    parameter: float = 0.0


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("normal_equal_balanced", "normal"),
    Scenario("normal_equal_unbalanced", "normal", (0.5, 1.0, 2.0)),
    Scenario("normal_unequal_balanced", "normal", scale_multipliers=(1.0, 2.0, 4.0)),
    Scenario(
        "normal_small_group_high_variance",
        "normal",
        (0.5, 1.0, 2.0),
        (4.0, 2.0, 1.0),
    ),
    Scenario(
        "normal_large_group_high_variance",
        "normal",
        (0.5, 1.0, 2.0),
        (1.0, 2.0, 4.0),
    ),
    Scenario(
        "normal_equal_five_groups",
        "normal",
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, 1.0),
    ),
    Scenario("gamma_moderate", "gamma", parameter=4.0),
    Scenario("exponential_severe", "gamma", parameter=1.0),
    Scenario("lognormal_moderate", "lognormal", parameter=0.5),
    Scenario("lognormal_severe", "lognormal", parameter=1.2),
    Scenario("student_t_df3", "student_t", parameter=3.0),
    Scenario("laplace", "laplace"),
    Scenario("mixture_symmetric", "mixture_symmetric"),
    Scenario("mixture_skewed", "mixture_skewed"),
    Scenario("outlier_contamination_5pct", "contaminated", parameter=0.05),
)


def _standardized_draw(
    rng: np.random.Generator,
    family: str,
    size: int,
    parameter: float,
) -> np.ndarray:
    if family == "normal":
        return rng.normal(size=size)
    if family == "gamma":
        shape = parameter
        return (rng.gamma(shape=shape, size=size) - shape) / np.sqrt(shape)
    if family == "lognormal":
        sigma = parameter
        mean = np.exp(0.5 * sigma**2)
        variance = (np.exp(sigma**2) - 1.0) * np.exp(sigma**2)
        return (rng.lognormal(sigma=sigma, size=size) - mean) / np.sqrt(variance)
    if family == "student_t":
        df = parameter
        return rng.standard_t(df=df, size=size) * np.sqrt((df - 2.0) / df)
    if family == "laplace":
        return rng.laplace(scale=1.0 / np.sqrt(2.0), size=size)
    if family == "mixture_symmetric":
        broad = rng.random(size) < 0.05
        values = rng.normal(size=size)
        values[broad] = rng.normal(scale=6.0, size=int(np.sum(broad)))
        return values / np.sqrt(0.95 + 0.05 * 36.0)
    if family == "mixture_skewed":
        upper = rng.random(size) < 0.10
        values = rng.normal(loc=-0.5, size=size)
        values[upper] = rng.normal(loc=4.5, size=int(np.sum(upper)))
        return values / np.sqrt(3.25)
    if family == "contaminated":
        probability = parameter
        contaminated = rng.random(size) < probability
        values = rng.normal(size=size)
        values[contaminated] = rng.normal(
            loc=10.0,
            size=int(np.sum(contaminated)),
        )
        mean = 10.0 * probability
        variance = 1.0 + probability * (1.0 - probability) * 100.0
        return (values - mean) / np.sqrt(variance)
    raise ValueError(f"Unknown family: {family}")


def generate_groups(
    scenario: Scenario,
    nominal_size: int,
    effect_size: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, ...]:
    sizes = tuple(
        max(2, int(round(nominal_size * multiplier)))
        for multiplier in scenario.size_multipliers
    )
    offsets = np.linspace(-effect_size / 2.0, effect_size / 2.0, len(sizes))
    return tuple(
        _standardized_draw(
            rng,
            scenario.family,
            size,
            scenario.parameter,
        )
        * scale
        + offset
        for size, scale, offset in zip(
            sizes,
            scenario.scale_multipliers,
            offsets,
        )
    )


def _rate(values: Sequence[bool]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _conditional_rate(outcomes: Sequence[bool], mask: Sequence[bool]) -> float:
    selected = [outcome for outcome, include in zip(outcomes, mask) if include]
    return _rate(selected)


def _nullable_rate(value: float):
    return float(value) if np.isfinite(value) else None


def simulate_cell(
    scenario: Scenario,
    scenario_index: int,
    nominal_size: int,
    effect_size: float,
    effect_index: int,
    seed: int,
    replications: int,
    alpha: float = 0.05,
) -> Dict[str, object]:
    policy = OneWayRobustness()
    levels: List[RobustnessLevel] = []
    classical_rejections: List[bool] = []
    welch_rejections: List[bool] = []
    max_skewness: List[float] = []
    max_kurtosis: List[float] = []
    max_outlier_fraction: List[float] = []
    variance_ratios: List[float] = []
    classical_eligible: List[bool] = []

    for replication in range(replications):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, scenario_index, nominal_size, effect_index, replication]
            )
        )
        groups = generate_groups(scenario, nominal_size, effect_size, rng)
        validation = InferenceValidator(alpha=alpha).validate_one_way(
            *groups,
            independence="assumed",
        )
        report = validation.report
        robustness = policy.evaluate(report)
        levels.append(robustness.level)

        classical = _classical_anova_statistics(validation.samples)
        welch = _welch_anova_statistics(validation.samples)
        classical_rejections.append(bool(classical["p_value"] < alpha))
        welch_rejections.append(bool(welch["p_value"] < alpha))

        group_shapes = [
            item
            for name, item in report.assessments.items()
            if name.startswith("shape_group_")
        ]
        group_outliers = [
            item
            for name, item in report.assessments.items()
            if name.startswith("outliers_group_")
        ]
        max_skewness.append(
            max(abs(float(item.metrics["skewness"])) for item in group_shapes)
        )
        max_kurtosis.append(
            max(abs(float(item.metrics["excess_kurtosis"])) for item in group_shapes)
        )
        max_outlier_fraction.append(
            max(float(item.metrics["fraction"]) for item in group_outliers)
        )
        variance = report.assessments["variance"]
        variance_ratios.append(float(variance.metrics["variance_ratio"]))
        classical_eligible.append(
            robustness.level is not RobustnessLevel.INSUFFICIENT
            and variance.status.value == "pass"
            and not bool(variance.metrics["small_group_large_variance"])
        )

    supported = [level is not RobustnessLevel.INSUFFICIENT for level in levels]
    acceptable = [level is RobustnessLevel.ACCEPTABLE for level in levels]
    caution = [level is RobustnessLevel.CAUTION for level in levels]
    selected_rejections = [
        rejected and is_supported
        for rejected, is_supported in zip(welch_rejections, supported)
    ]
    acceptable_rejection_rate = _conditional_rate(welch_rejections, acceptable)
    caution_rejection_rate = _conditional_rate(welch_rejections, caution)
    conditional_selected_rate = _conditional_rate(welch_rejections, supported)
    under_null = effect_size == 0.0
    false_acceptable_rate = 0.0
    if (
        under_null
        and np.isfinite(acceptable_rejection_rate)
        and not 0.025 <= acceptable_rejection_rate <= 0.075
    ):
        false_acceptable_rate = _rate(acceptable)

    return {
        "scenario": scenario.name,
        "family": scenario.family,
        "nominal_group_n": nominal_size,
        "group_sizes": ";".join(
            str(max(2, int(round(nominal_size * value))))
            for value in scenario.size_multipliers
        ),
        "scale_multipliers": ";".join(str(value) for value in scenario.scale_multipliers),
        "effect_size": effect_size,
        "hypothesis": "H0" if under_null else "H1",
        "seed": seed,
        "replications": replications,
        "classical_rejection_rate": _rate(classical_rejections),
        "welch_rejection_rate": _rate(welch_rejections),
        "selected_unconditional_rejection_rate": _rate(selected_rejections),
        "selected_conditional_rejection_rate": _nullable_rate(conditional_selected_rate),
        "acceptable_rejection_rate": _nullable_rate(acceptable_rejection_rate),
        "caution_rejection_rate": _nullable_rate(caution_rejection_rate),
        "acceptable_rate": _rate(acceptable),
        "caution_rate": _rate(caution),
        "insufficient_rate": _rate(
            [level is RobustnessLevel.INSUFFICIENT for level in levels]
        ),
        "welch_selection_rate": _rate(supported),
        "classical_selection_rate": 0.0,
        "classical_eligible_rate": _rate(classical_eligible),
        "false_acceptable_rate": false_acceptable_rate,
        "excessive_conservatism": bool(
            under_null
            and np.isfinite(conditional_selected_rate)
            and conditional_selected_rate < 0.025
            and _rate(supported) >= 0.20
        ),
        "median_max_abs_skewness": float(np.median(max_skewness)),
        "median_max_abs_excess_kurtosis": float(np.median(max_kurtosis)),
        "median_max_outlier_fraction": float(np.median(max_outlier_fraction)),
        "median_variance_ratio": float(np.median(variance_ratios)),
        "policy_version": policy.POLICY_VERSION,
    }


def run_matrix(
    *,
    replications: int,
    nominal_sizes: Sequence[int],
    effect_sizes: Sequence[float],
    seeds: Sequence[int],
    scenarios: Sequence[Scenario] = SCENARIOS,
    alpha: float = 0.05,
) -> List[Dict[str, object]]:
    rows = []
    for scenario_index, scenario in enumerate(scenarios):
        for nominal_size in nominal_sizes:
            for effect_index, effect_size in enumerate(effect_sizes):
                for seed in seeds:
                    rows.append(
                        simulate_cell(
                            scenario,
                            scenario_index,
                            nominal_size,
                            effect_size,
                            effect_index,
                            seed,
                            replications,
                            alpha,
                        )
                    )
    return rows


def write_results(
    rows: Sequence[Dict[str, object]],
    output_dir: Path,
    configuration: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "anova_calibration_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        **configuration,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(SCENARIOS),
        "cell_count": len(rows),
        "total_replications": int(sum(int(row["replications"]) for row in rows)),
        "policy_version": OneWayRobustness.POLICY_VERSION,
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
    }
    (output_dir / "anova_calibration_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=250)
    parser.add_argument("--nominal-sizes", type=int, nargs="+", default=[10, 25, 60])
    parser.add_argument("--effect-sizes", type=float, nargs="+", default=[0.0, 0.8])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260827, 20260828, 20260829],
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configuration = {
        "replications_per_cell": args.replications,
        "nominal_sizes": args.nominal_sizes,
        "effect_sizes": args.effect_sizes,
        "seeds": args.seeds,
        "alpha": args.alpha,
    }
    rows = run_matrix(
        replications=args.replications,
        nominal_sizes=args.nominal_sizes,
        effect_sizes=args.effect_sizes,
        seeds=args.seeds,
        alpha=args.alpha,
    )
    write_results(rows, args.output_dir, configuration)
    print(
        f"Wrote {len(rows)} cells and "
        f"{sum(int(row['replications']) for row in rows):,} replications"
    )


if __name__ == "__main__":
    main()
