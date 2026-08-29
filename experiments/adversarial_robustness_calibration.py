"""Reproducible Monte Carlo calibration for the current mean-inference policy.

This is a research harness: it observes the public diagnostics and method
selector exactly as shipped, but does not modify or reimplement their policy.
The long-form summary reports unconditional and policy-conditional operating
characteristics with explicit denominators and Wilson binomial intervals.

Examples
--------
python -m experiments.adversarial_robustness_calibration --profile smoke
python -m experiments.adversarial_robustness_calibration --profile calibration
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import platform
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import scipy
from scipy import stats

from pyMagicStat.assumptions import InferenceValidator, SamplingRobustness
from pyMagicStat.assumptions.models import (
    Assessment,
    AssessmentStatus,
    AssumptionReport,
    Estimand,
    InferenceDesign,
)
from pyMagicStat.inference import MethodSelector


DEFAULT_SEED = 20260828
ALPHA = 0.05
EXPLORATORY_SAMPLE_SIZES = (5, 8, 10, 15, 20, 30, 40, 50, 80, 100, 200, 500, 750, 2000)
BUG03_NORMAL_SAMPLE_SIZES = (30, 100, 750, 2000, 10000)
BUG05_SMALL_SAMPLE_SIZES = (3, 4, 5, 8, 10, 15, 20)
DECISION_SCOPES = ("all", "acceptable", "caution", "insufficient")


@dataclass(frozen=True)
class Scenario:
    """One centered distribution with an exactly known population mean."""

    name: str
    family: str
    parameters: Mapping[str, object]
    generator: Callable[[np.random.Generator, int], np.ndarray]
    population_mean: float = 0.0

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return np.asarray(self.generator(rng, n), dtype=float)

    @property
    def parameters_json(self) -> str:
        return json.dumps(dict(self.parameters), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CalibrationCell:
    scenario: Scenario
    n: int
    replications: int
    evidence_tier: str


def _centered_standardized_lognormal(
    sigma: float,
) -> Callable[[np.random.Generator, int], np.ndarray]:
    mean = math.exp(sigma * sigma / 2.0)
    variance = (math.exp(sigma * sigma) - 1.0) * math.exp(sigma * sigma)
    scale = math.sqrt(variance)
    return lambda rng, n: (rng.lognormal(0.0, sigma, n) - mean) / scale


def _standardized_student_t(df: int) -> Callable[[np.random.Generator, int], np.ndarray]:
    scale = math.sqrt((df - 2.0) / df)
    return lambda rng, n: rng.standard_t(df, n) * scale


def _standardized_gamma(shape: float) -> Callable[[np.random.Generator, int], np.ndarray]:
    return lambda rng, n: (rng.gamma(shape, 1.0, n) - shape) / math.sqrt(shape)


def _normal_mixture(
    probability: float,
    base_mean: float,
    base_sd: float,
    component_mean: float,
    component_sd: float,
    *,
    standardize: bool = True,
) -> Callable[[np.random.Generator, int], np.ndarray]:
    mean = (1.0 - probability) * base_mean + probability * component_mean
    second_moment = (
        (1.0 - probability) * (base_sd**2 + base_mean**2)
        + probability * (component_sd**2 + component_mean**2)
    )
    scale = math.sqrt(second_moment - mean**2) if standardize else 1.0

    def draw(rng: np.random.Generator, n: int) -> np.ndarray:
        component = rng.random(n) < probability
        values = rng.normal(base_mean, base_sd, n)
        count = int(component.sum())
        values[component] = rng.normal(component_mean, component_sd, count)
        return (values - mean) / scale

    return draw


def _symmetric_contamination(
    probability: float,
) -> Callable[[np.random.Generator, int], np.ndarray]:
    """N(0, 1) contaminated by a centered N(0, 10) component."""

    return _normal_mixture(probability, 0.0, 1.0, 0.0, 10.0)


def _asymmetric_contamination(
    probability: float,
) -> Callable[[np.random.Generator, int], np.ndarray]:
    """N(0, 1) contaminated by N(10, 1), then exactly centered/scaled."""

    return _normal_mixture(probability, 0.0, 1.0, 10.0, 1.0)


def scenario_catalog() -> tuple[Scenario, ...]:
    """Return the prespecified family catalog for BUG-03 through BUG-06."""

    result: list[Scenario] = [
        Scenario("normal", "normal", {"mean": 0.0, "sd": 1.0}, lambda rng, n: rng.normal(size=n))
    ]
    for df in (30, 10, 5, 3):
        result.append(
            Scenario(
                f"student_t_df_{df}",
                "student_t",
                {"df": df, "standardized_variance": True},
                _standardized_student_t(df),
            )
        )
    for sigma, severity in ((0.25, "mild"), (0.50, "moderate"), (1.00, "severe")):
        result.append(
            Scenario(
                f"lognormal_sigma_{sigma:.2f}",
                "lognormal",
                {"mu": 0.0, "sigma": sigma, "severity": severity, "centered_and_standardized": True},
                _centered_standardized_lognormal(sigma),
            )
        )
    for epsilon in (0.001, 0.005, 0.01, 0.025, 0.05, 0.10):
        label = f"{epsilon:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        result.extend(
            (
                Scenario(
                    f"contamination_symmetric_eps_{label}",
                    "normal_contamination_symmetric",
                    {"epsilon": epsilon, "base": "N(0,1)", "contaminant": "N(0,10)", "centered_and_standardized": True},
                    _symmetric_contamination(epsilon),
                ),
                Scenario(
                    f"contamination_asymmetric_eps_{label}",
                    "normal_contamination_asymmetric",
                    {"epsilon": epsilon, "base": "N(0,1)", "contaminant": "N(10,1)", "centered_and_standardized": True},
                    _asymmetric_contamination(epsilon),
                ),
            )
        )
    result.extend(
        (
            Scenario(
                "bimodal_symmetric",
                "bimodal",
                {"weights": [0.5, 0.5], "means": [-2.0, 2.0], "sds": [1.0, 1.0], "centered_and_standardized": True},
                _normal_mixture(0.5, -2.0, 1.0, 2.0, 1.0),
            ),
            Scenario(
                "bimodal_asymmetric",
                "bimodal",
                {"weights": [0.8, 0.2], "means": [-1.0, 4.0], "sds": [1.0, 1.0], "centered_and_standardized": True},
                _normal_mixture(0.2, -1.0, 1.0, 4.0, 1.0),
            ),
            Scenario(
                "mixture_distinct_means",
                "normal_mixture",
                {"weights": [0.7, 0.3], "means": [-1.0, 3.0], "sds": [1.0, 1.0], "centered_and_standardized": True},
                _normal_mixture(0.3, -1.0, 1.0, 3.0, 1.0),
            ),
            Scenario(
                "mixture_distinct_variances",
                "normal_mixture",
                {"weights": [0.9, 0.1], "means": [0.0, 0.0], "sds": [1.0, 5.0], "centered_and_standardized": True},
                _normal_mixture(0.1, 0.0, 1.0, 0.0, 5.0),
            ),
            Scenario(
                "gamma_shape_2",
                "gamma",
                {"shape": 2.0, "scale": 1.0, "centered_and_standardized": True},
                _standardized_gamma(2.0),
            ),
            Scenario(
                "gamma_shape_4",
                "gamma",
                {"shape": 4.0, "scale": 1.0, "centered_and_standardized": True},
                _standardized_gamma(4.0),
            ),
        )
    )
    return tuple(result)


def _upsert_cell(
    cells: dict[tuple[str, int], CalibrationCell],
    scenario: Scenario,
    n: int,
    replications: int,
    evidence_tier: str,
) -> None:
    key = (scenario.name, int(n))
    current = cells.get(key)
    if current is None or replications > current.replications:
        cells[key] = CalibrationCell(scenario, int(n), int(replications), evidence_tier)


def calibration_plan(
    *,
    exploratory_replications: int = 200,
    confirmatory_replications: int = 10_000,
    minimum_confirmatory_replications: int = 5_000,
) -> tuple[CalibrationCell, ...]:
    """Create the adaptive exploration/confirmation matrix.

    All family/sample-size combinations receive an exploratory pass. The five
    prior adversarial findings receive 10,000 replications. BUG-03 normal cells,
    BUG-05 normal cells, and contamination sensitivity at n=100 receive at
    least 5,000 replications.
    """

    if min(exploratory_replications, confirmatory_replications, minimum_confirmatory_replications) < 1:
        raise ValueError("replication counts must be positive")

    catalog = {scenario.name: scenario for scenario in scenario_catalog()}
    cells: dict[tuple[str, int], CalibrationCell] = {}
    for scenario in catalog.values():
        for n in EXPLORATORY_SAMPLE_SIZES:
            _upsert_cell(cells, scenario, n, exploratory_replications, "exploratory")

    for name, n in (
        ("lognormal_sigma_0.25", 20),
        ("lognormal_sigma_0.50", 50),
        ("lognormal_sigma_1.00", 30),
        ("student_t_df_5", 20),
        ("bimodal_symmetric", 300),
    ):
        _upsert_cell(cells, catalog[name], n, confirmatory_replications, "confirmatory_10000")

    for n in BUG03_NORMAL_SAMPLE_SIZES:
        _upsert_cell(cells, catalog["normal"], n, minimum_confirmatory_replications, "confirmatory_5000")
    for n in BUG05_SMALL_SAMPLE_SIZES:
        _upsert_cell(cells, catalog["normal"], n, minimum_confirmatory_replications, "confirmatory_5000")
    for scenario in catalog.values():
        if scenario.family.startswith("normal_contamination"):
            _upsert_cell(cells, scenario, 100, minimum_confirmatory_replications, "confirmatory_5000")

    return tuple(sorted(cells.values(), key=lambda cell: (cell.scenario.name, cell.n)))


def smoke_plan() -> tuple[CalibrationCell, ...]:
    catalog = {scenario.name: scenario for scenario in scenario_catalog()}
    return (
        CalibrationCell(catalog["normal"], 5, 8, "smoke"),
        CalibrationCell(catalog["lognormal_sigma_0.50"], 20, 8, "smoke"),
    )


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval, returning NaNs when the denominator is zero."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if trials == 0:
        return math.nan, math.nan
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)) / denominator
    return center - half_width, center + half_width


def _mean_inference(sample: np.ndarray, population_mean: float, alpha: float) -> dict[str, float | int]:
    n = int(sample.size)
    sample_mean = float(np.mean(sample))
    sample_std = float(np.std(sample, ddof=1))
    standard_error = sample_std / math.sqrt(n)
    t_statistic = (sample_mean - population_mean) / standard_error
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), n - 1))
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    ci_lower = sample_mean - critical * standard_error
    ci_upper = sample_mean + critical * standard_error
    return {
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "reject_h0": int(p_value < alpha),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_contains_true_mean": int(ci_lower <= population_mean <= ci_upper),
    }


def _influence_metrics(
    sample: np.ndarray,
    extreme_indices: np.ndarray,
    inference: Mapping[str, float | int],
    population_mean: float,
    alpha: float,
) -> dict[str, float | int]:
    if extreme_indices.size == 0:
        return {
            "remaining_after_extreme_removal": int(sample.size),
            "mean_without_extremes": inference["sample_mean"],
            "standard_error_without_extremes": inference["standard_error"],
            "t_statistic_without_extremes": inference["t_statistic"],
            "ci_contains_true_mean_without_extremes": inference["ci_contains_true_mean"],
            "delta_mean_remove_extremes": 0.0,
            "influence_ratio": 0.0,
            "delta_standard_error_remove_extremes": 0.0,
            "delta_t_statistic_remove_extremes": 0.0,
        }

    retained = np.delete(sample, extreme_indices)
    if retained.size < 2 or np.unique(retained).size < 2:
        return {
            "remaining_after_extreme_removal": int(retained.size),
            "mean_without_extremes": math.nan,
            "standard_error_without_extremes": math.nan,
            "t_statistic_without_extremes": math.nan,
            "ci_contains_true_mean_without_extremes": "",
            "delta_mean_remove_extremes": math.nan,
            "influence_ratio": math.nan,
            "delta_standard_error_remove_extremes": math.nan,
            "delta_t_statistic_remove_extremes": math.nan,
        }

    reduced = _mean_inference(retained, population_mean, alpha)
    delta_mean = abs(float(inference["sample_mean"]) - float(reduced["sample_mean"]))
    standard_error = float(inference["standard_error"])
    return {
        "remaining_after_extreme_removal": int(retained.size),
        "mean_without_extremes": reduced["sample_mean"],
        "standard_error_without_extremes": reduced["standard_error"],
        "t_statistic_without_extremes": reduced["t_statistic"],
        "ci_contains_true_mean_without_extremes": reduced["ci_contains_true_mean"],
        "delta_mean_remove_extremes": delta_mean,
        "influence_ratio": delta_mean / standard_error if standard_error > 0.0 else math.nan,
        "delta_standard_error_remove_extremes": abs(standard_error - float(reduced["standard_error"])),
        "delta_t_statistic_remove_extremes": abs(float(inference["t_statistic"]) - float(reduced["t_statistic"])),
    }


def simulate_cell(
    cell: CalibrationCell,
    *,
    cell_seed: np.random.SeedSequence,
    alpha: float = ALPHA,
) -> list[dict[str, object]]:
    """Run one cell using a distinct, recorded uint64 seed per replicate."""

    validator = InferenceValidator(alpha=alpha)
    selector = MethodSelector()
    replicate_seeds = cell_seed.generate_state(cell.replications, dtype=np.uint64)
    records: list[dict[str, object]] = []
    for replication, replicate_seed in enumerate(replicate_seeds):
        rng = np.random.default_rng(replicate_seed)
        sample = cell.scenario.draw(rng, cell.n)
        validation = validator.validate_one_sample(sample, independence="assumed")
        report = validation.report
        shape = report.assessments["shape"]
        outliers = report.assessments["outliers"]
        selection = selector.select(report)
        inference = _mean_inference(sample, cell.scenario.population_mean, alpha)
        extreme_indices = np.asarray(outliers.metrics["indices"], dtype=int)
        influence = _influence_metrics(
            sample,
            extreme_indices,
            inference,
            cell.scenario.population_mean,
            alpha,
        )
        records.append(
            {
                "scenario": cell.scenario.name,
                "distribution_family": cell.scenario.family,
                "distribution_parameters": cell.scenario.parameters_json,
                "population_mean": cell.scenario.population_mean,
                "n": cell.n,
                "evidence_tier": cell.evidence_tier,
                "replication": replication,
                "seed": int(replicate_seed),
                "sample_mean": inference["sample_mean"],
                "sample_std": inference["sample_std"],
                "standard_error": inference["standard_error"],
                "skewness": shape.metrics["skewness"],
                "excess_kurtosis": shape.metrics["excess_kurtosis"],
                "shape_status": shape.status.value,
                "departure_magnitude": shape.metrics["departure_magnitude"],
                "exact_normality_rejected": shape.metrics["exact_normality_rejected"],
                "outlier_status": outliers.status.value,
                "extreme_count": outliers.metrics["count"],
                "extreme_fraction": outliers.metrics["fraction"],
                "sampling_robustness_level": selection.robustness.level.value,
                "selected_method": selection.selected_method or "",
                "t_statistic": inference["t_statistic"],
                "p_value": inference["p_value"],
                "reject_h0": inference["reject_h0"],
                "ci_lower": inference["ci_lower"],
                "ci_upper": inference["ci_upper"],
                "ci_contains_true_mean": inference["ci_contains_true_mean"],
                **influence,
            }
        )
    return records


def _safe_mean(values: Iterable[object]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else math.nan


def _safe_quantile(values: Iterable[object], probability: float) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.quantile(finite, probability)) if finite.size else math.nan


def summarize_cell(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Summarize one cell for all observations and each policy decision."""

    if not records:
        raise ValueError("records must not be empty")
    total = len(records)
    first = records[0]
    output: list[dict[str, object]] = []
    for scope in DECISION_SCOPES:
        rows = list(records) if scope == "all" else [
            row for row in records if row["sampling_robustness_level"] == scope
        ]
        denominator = len(rows)
        classification_count = total if scope == "all" else denominator
        type_i_count = sum(int(row["reject_h0"]) for row in rows)
        coverage_count = sum(int(row["ci_contains_true_mean"]) for row in rows)
        class_low, class_high = wilson_interval(classification_count, total)
        type_low, type_high = wilson_interval(type_i_count, denominator)
        coverage_low, coverage_high = wilson_interval(coverage_count, denominator)
        sample_means = np.asarray([row["sample_mean"] for row in rows], dtype=float)
        population_mean = float(first["population_mean"])
        empirical_sd = float(np.std(sample_means, ddof=1)) if denominator > 1 else math.nan
        mean_se = _safe_mean(row["standard_error"] for row in rows)
        output.append(
            {
                "scenario": first["scenario"],
                "distribution_family": first["distribution_family"],
                "distribution_parameters": first["distribution_parameters"],
                "n": first["n"],
                "evidence_tier": first["evidence_tier"],
                "decision_scope": scope,
                "total_replications": total,
                "conditional_denominator": denominator,
                "classification_count": classification_count,
                "classification_rate": classification_count / total,
                "classification_ci95_lower": class_low,
                "classification_ci95_upper": class_high,
                "type_i_rejections": type_i_count,
                "type_i_error": type_i_count / denominator if denominator else math.nan,
                "type_i_ci95_lower": type_low,
                "type_i_ci95_upper": type_high,
                "coverage_successes": coverage_count,
                "ci_coverage": coverage_count / denominator if denominator else math.nan,
                "coverage_ci95_lower": coverage_low,
                "coverage_ci95_upper": coverage_high,
                "mean_bias": _safe_mean(value - population_mean for value in sample_means),
                "rmse": math.sqrt(_safe_mean((value - population_mean) ** 2 for value in sample_means)),
                "mean_standard_error": mean_se,
                "empirical_sd_sample_mean": empirical_sd,
                "mean_se_to_empirical_sd_ratio": mean_se / empirical_sd if empirical_sd > 0.0 else math.nan,
                "probability_extreme_count_positive": _safe_mean(int(row["extreme_count"] > 0) for row in rows),
                "outlier_warn_rate": _safe_mean(int(row["outlier_status"] == "warn") for row in rows),
                "mean_extreme_count": _safe_mean(row["extreme_count"] for row in rows),
                "mean_extreme_fraction": _safe_mean(row["extreme_fraction"] for row in rows),
                "mean_delta_mean_remove_extremes": _safe_mean(row["delta_mean_remove_extremes"] for row in rows),
                "median_influence_ratio": _safe_quantile((row["influence_ratio"] for row in rows), 0.5),
                "p90_influence_ratio": _safe_quantile((row["influence_ratio"] for row in rows), 0.9),
                "mean_delta_standard_error_remove_extremes": _safe_mean(row["delta_standard_error_remove_extremes"] for row in rows),
                "mean_delta_t_statistic_remove_extremes": _safe_mean(row["delta_t_statistic_remove_extremes"] for row in rows),
                "exact_normality_rejection_rate": _safe_mean(
                    int(row["exact_normality_rejected"] is True) for row in rows
                ),
                "shape_pass_rate": _safe_mean(int(row["shape_status"] == "pass") for row in rows),
            }
        )
    return output


def _synthetic_report(n: int, skewness: float, kurtosis: float, outlier_fraction: float, count: int) -> AssumptionReport:
    abs_skew = abs(skewness)
    abs_kurtosis = abs(kurtosis)
    if abs_skew > 2.0 or abs_kurtosis > 7.0:
        magnitude, shape_status = "severe", AssessmentStatus.FAIL
    elif abs_skew > 1.0 or abs_kurtosis > 3.0:
        magnitude, shape_status = "moderate", AssessmentStatus.WARN
    else:
        magnitude, shape_status = "mild", AssessmentStatus.PASS
    assessments = {
        "data_quality": Assessment("data_quality", AssessmentStatus.PASS, {"n": n}),
        "shape": Assessment(
            "shape_sample",
            shape_status,
            {
                "n": n,
                "skewness": skewness,
                "excess_kurtosis": kurtosis,
                "departure_magnitude": magnitude,
                "exact_normality_rejected": False,
            },
        ),
        "outliers": Assessment(
            "outliers_sample",
            AssessmentStatus.WARN if count else AssessmentStatus.PASS,
            {"count": count, "fraction": outlier_fraction},
        ),
        "independence": Assessment("independence", AssessmentStatus.PASS, {"independence": "assumed"}),
    }
    return AssumptionReport(InferenceDesign.ONE_SAMPLE, Estimand.MEAN, assessments)


def _cliff_series() -> list[tuple[str, str, Sequence[float], Callable[[float], tuple[int, float, float, float, int]]]]:
    around = lambda threshold: np.round(np.arange(threshold - 0.05, threshold + 0.0501, 0.005), 6)
    around_fraction = lambda threshold: np.round(
        np.arange(threshold - 0.005, threshold + 0.00501, 0.0005), 6
    )
    return [
        ("skew_1_n40", "skewness", around(1.0), lambda value: (40, value, 0.0, 0.0, 0)),
        ("skew_2_n80", "skewness", around(2.0), lambda value: (80, value, 0.0, 0.01, 1)),
        ("kurtosis_3_n40", "excess_kurtosis", around(3.0), lambda value: (40, 0.5, value, 0.02, 1)),
        ("kurtosis_7_n80", "excess_kurtosis", around(7.0), lambda value: (80, 1.5, value, 0.05, 1)),
        ("kurtosis_25_n200", "excess_kurtosis", around(25.0), lambda value: (200, 1.5, value, 0.05, 1)),
        ("outlier_fraction_0p025_n40", "outlier_fraction", around_fraction(0.025), lambda value: (40, 0.5, 2.0, value, 1)),
        ("outlier_fraction_0p10_n80", "outlier_fraction", around_fraction(0.10), lambda value: (80, 1.5, 5.0, value, 1)),
        ("n_40", "n", tuple(range(35, 46)), lambda value: (int(value), 0.8, 2.0, 0.02, 1)),
        ("n_80", "n", tuple(range(75, 86)), lambda value: (int(value), 1.5, 5.0, 0.05, 1)),
        ("n_200", "n", tuple(range(195, 206)), lambda value: (int(value), 1.5, 15.0, 0.05, 1)),
    ]


def map_threshold_cliffs() -> list[dict[str, object]]:
    """Evaluate dense, one-dimensional grids against the unchanged policy."""

    policy = SamplingRobustness()
    rows: list[dict[str, object]] = []
    for series, varied_parameter, values, parameterize in _cliff_series():
        previous: str | None = None
        for value in values:
            n, skewness, kurtosis, fraction, count = parameterize(float(value))
            decision = policy.evaluate(_synthetic_report(n, skewness, kurtosis, fraction, count))
            current = decision.level.value
            rows.append(
                {
                    "series": series,
                    "varied_parameter": varied_parameter,
                    "grid_value": value,
                    "n": n,
                    "skewness": skewness,
                    "excess_kurtosis": kurtosis,
                    "outlier_fraction": fraction,
                    "extreme_count": count,
                    "robustness_level": current,
                    "transition": int(previous is not None and previous != current),
                    "transition_from": previous if previous is not None and previous != current else "",
                    "transition_to": current if previous is not None and previous != current else "",
                }
            )
            previous = current
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty result: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_head() -> str:
    executable = shutil.which("git")
    if executable is None and platform.system() == "Windows":
        candidate = Path(r"C:\Program Files\Git\cmd\git.exe")
        executable = str(candidate) if candidate.is_file() else None
    if executable is None:
        return "unavailable"
    try:
        return subprocess.run(
            [executable, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def run_calibration(
    cells: Sequence[CalibrationCell],
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    alpha: float = ALPHA,
    profile: str = "calibration",
) -> dict[str, object]:
    """Run cells sequentially, retaining only one cell in memory at a time."""

    output_dir.mkdir(parents=True, exist_ok=True)
    replicate_path = output_dir / "robustness_calibration_replicates.csv.gz"
    summary_rows: list[dict[str, object]] = []
    root_seed = np.random.SeedSequence(seed)
    cell_seeds = root_seed.spawn(len(cells))
    replicate_writer: csv.DictWriter | None = None
    with gzip.open(replicate_path, "wt", newline="", encoding="utf-8") as replicate_handle:
        for index, (cell, cell_seed) in enumerate(zip(cells, cell_seeds), start=1):
            print(
                f"[{index}/{len(cells)}] {cell.scenario.name} n={cell.n} "
                f"replications={cell.replications}",
                flush=True,
            )
            records = simulate_cell(cell, cell_seed=cell_seed, alpha=alpha)
            if replicate_writer is None:
                replicate_writer = csv.DictWriter(replicate_handle, fieldnames=list(records[0]))
                replicate_writer.writeheader()
            replicate_writer.writerows(records)
            summary_rows.extend(summarize_cell(records))

    summary_path = output_dir / "robustness_calibration_summary.csv"
    cliff_path = output_dir / "robustness_threshold_cliffs.csv"
    _write_csv(summary_path, summary_rows)
    cliff_rows = map_threshold_cliffs()
    _write_csv(cliff_path, cliff_rows)

    metadata = {
        "alpha": alpha,
        "commit_sha_at_start": _git_head(),
        "numpy_version": np.__version__,
        "profile": profile,
        "python_version": platform.python_version(),
        "replications_by_cell": [
            {
                "scenario": cell.scenario.name,
                "n": cell.n,
                "replications": cell.replications,
                "evidence_tier": cell.evidence_tier,
                "seed_spawn_key": list(cell_seed.spawn_key),
            }
            for cell, cell_seed in zip(cells, cell_seeds)
        ],
        "sampling_robustness_policy": SamplingRobustness.POLICY_VERSION,
        "scenario_count": len({cell.scenario.name for cell in cells}),
        "cell_count": len(cells),
        "total_replications": sum(cell.replications for cell in cells),
        "scipy_version": scipy.__version__,
        "seed": seed,
        "seed_strategy": (
            "numpy.random.SeedSequence(global seed), one spawned stream per sorted cell; "
            "each stream generates one recorded uint64 seed per replicate"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "replicates": replicate_path.name,
            "summary": summary_path.name,
            "threshold_cliffs": cliff_path.name,
        },
    }
    (output_dir / "robustness_calibration_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "calibration"), default="calibration")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--exploratory-replications", type=int, default=200)
    parser.add_argument("--confirmatory-replications", type=int, default=10_000)
    parser.add_argument("--minimum-confirmatory-replications", type=int, default=5_000)
    args = parser.parse_args()

    cells = smoke_plan() if args.profile == "smoke" else calibration_plan(
        exploratory_replications=args.exploratory_replications,
        confirmatory_replications=args.confirmatory_replications,
        minimum_confirmatory_replications=args.minimum_confirmatory_replications,
    )
    metadata = run_calibration(
        cells,
        args.output_dir,
        seed=args.seed,
        alpha=args.alpha,
        profile=args.profile,
    )
    print(json.dumps({"cell_count": metadata["cell_count"], "total_replications": metadata["total_replications"]}))


if __name__ == "__main__":
    main()
