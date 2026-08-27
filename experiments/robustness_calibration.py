"""Calibrate ``SamplingRobustness`` against repeated-sampling behaviour.

The experiment measures the coverage of the usual Student-t confidence
interval for a population mean and the type-I error of its equivalent
two-sided one-sample test.  It intentionally calls the public assumption
engine for every replicate, so the recorded diagnostics and decision are the
ones emitted by pyMagicStat rather than approximations maintained by this
script.

Example
-------
python -m experiments.robustness_calibration --replications 1000 \
    --output-dir experiments/results
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import scipy
from scipy import stats

from pyMagicStat.assumptions import InferenceValidator, SamplingRobustness


DEFAULT_SAMPLE_SIZES = (10, 20, 30, 40, 50, 80, 100, 200)
DEFAULT_SEED = 20260826


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    generator: Callable[[np.random.Generator, int], np.ndarray]
    population_mean: float = 0.0
    nominal_contamination: float = 0.0

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return np.asarray(self.generator(rng, n), dtype=float)


def _centered_lognormal(sigma: float) -> Callable[[np.random.Generator, int], np.ndarray]:
    mean = float(np.exp(sigma * sigma / 2.0))
    return lambda rng, n: rng.lognormal(0.0, sigma, n) - mean


def _centered_gamma(shape: float) -> Callable[[np.random.Generator, int], np.ndarray]:
    return lambda rng, n: rng.gamma(shape, 1.0, n) - shape


def _mixture(
    probability: float,
    base_mean: float,
    base_sd: float,
    component_mean: float,
    component_sd: float,
) -> Callable[[np.random.Generator, int], np.ndarray]:
    population_mean = (1.0 - probability) * base_mean + probability * component_mean

    def draw(rng: np.random.Generator, n: int) -> np.ndarray:
        component = rng.random(n) < probability
        values = rng.normal(base_mean, base_sd, n)
        values[component] = rng.normal(component_mean, component_sd, int(component.sum()))
        return values - population_mean

    return draw


def scenarios() -> tuple[Scenario, ...]:
    """Return the fixed distribution matrix used by the calibration."""

    return (
        Scenario("normal", "normal", lambda rng, n: rng.normal(size=n)),
        Scenario("exponential", "exponential", lambda rng, n: rng.exponential(size=n) - 1.0),
        Scenario("gamma_shape_2", "gamma", _centered_gamma(2.0)),
        Scenario("gamma_shape_4", "gamma", _centered_gamma(4.0)),
        Scenario("gamma_shape_9", "gamma", _centered_gamma(9.0)),
        Scenario("lognormal_sigma_0.25", "lognormal", _centered_lognormal(0.25)),
        Scenario("lognormal_sigma_0.5", "lognormal", _centered_lognormal(0.5)),
        Scenario("lognormal_sigma_1.0", "lognormal", _centered_lognormal(1.0)),
        Scenario("lognormal_sigma_1.25", "lognormal", _centered_lognormal(1.25)),
        Scenario("student_t_df_3", "student_t", lambda rng, n: rng.standard_t(3, n)),
        Scenario("student_t_df_5", "student_t", lambda rng, n: rng.standard_t(5, n)),
        Scenario("student_t_df_10", "student_t", lambda rng, n: rng.standard_t(10, n)),
        Scenario("student_t_df_30", "student_t", lambda rng, n: rng.standard_t(30, n)),
        Scenario("laplace", "laplace", lambda rng, n: rng.laplace(size=n)),
        Scenario(
            "mixture_symmetric_5pct_wide",
            "mixture",
            _mixture(0.05, 0.0, 1.0, 0.0, 8.0),
            nominal_contamination=0.05,
        ),
        Scenario(
            "mixture_skewed_10pct",
            "mixture",
            _mixture(0.10, -0.5, 1.0, 4.5, 1.0),
            nominal_contamination=0.10,
        ),
        Scenario(
            "outliers_positive_1pct",
            "contamination",
            _mixture(0.01, 0.0, 1.0, 10.0, 1.0),
            nominal_contamination=0.01,
        ),
        Scenario(
            "outliers_positive_5pct",
            "contamination",
            _mixture(0.05, 0.0, 1.0, 10.0, 1.0),
            nominal_contamination=0.05,
        ),
        Scenario(
            "outliers_positive_10pct",
            "contamination",
            _mixture(0.10, 0.0, 1.0, 10.0, 1.0),
            nominal_contamination=0.10,
        ),
    )


def _mean_inference(sample: np.ndarray, population_mean: float, alpha: float) -> tuple[bool, bool]:
    n = int(sample.size)
    estimate = float(np.mean(sample))
    standard_error = float(np.std(sample, ddof=1) / np.sqrt(n))
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    lower = estimate - critical * standard_error
    upper = estimate + critical * standard_error
    statistic = (estimate - population_mean) / standard_error
    p_value = float(2.0 * stats.t.sf(abs(statistic), n - 1))
    return lower <= population_mean <= upper, p_value < alpha


def simulate(
    *,
    replications: int,
    sample_sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    selected_scenarios: Iterable[Scenario] | None = None,
) -> list[dict[str, object]]:
    """Run the full matrix and return one diagnostic record per replicate."""

    if replications < 1:
        raise ValueError("replications must be positive")
    if any(int(n) < 3 for n in sample_sizes):
        raise ValueError("all sample sizes must be at least 3")

    matrix = tuple(selected_scenarios or scenarios())
    validator = InferenceValidator(alpha=alpha)
    policy = SamplingRobustness()
    seed_sequence = np.random.SeedSequence(seed)
    streams = iter(seed_sequence.spawn(len(matrix) * len(sample_sizes)))
    records: list[dict[str, object]] = []

    for scenario in matrix:
        for n in sample_sizes:
            rng = np.random.default_rng(next(streams))
            for replication in range(replications):
                sample = scenario.draw(rng, int(n))
                validation = validator.validate_one_sample(
                    sample,
                    independence="assumed",
                )
                report = validation.report
                shape = report.assessments["shape"]
                outliers = report.assessments["outliers"]
                decision = policy.evaluate(report)
                covered, rejected = _mean_inference(sample, scenario.population_mean, alpha)
                records.append(
                    {
                        "scenario": scenario.name,
                        "family": scenario.family,
                        "n": int(n),
                        "replication": replication,
                        "seed": seed,
                        "nominal_contamination": scenario.nominal_contamination,
                        "skewness": shape.metrics["skewness"],
                        "excess_kurtosis": shape.metrics["excess_kurtosis"],
                        "outlier_fraction": outliers.metrics["fraction"],
                        "shape_status": shape.status.value,
                        "robustness_decision": decision.level.value,
                        "ci_covered": int(covered),
                        "type_i_rejection": int(rejected),
                    }
                )
    return records


def summarize(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate replicate records by scenario and sample size."""

    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["scenario"]), str(record["family"]), int(record["n"]))].append(record)

    summary: list[dict[str, object]] = []
    for (scenario, family, n), rows in grouped.items():
        skew = np.asarray([row["skewness"] for row in rows], dtype=float)
        kurtosis = np.asarray([row["excess_kurtosis"] for row in rows], dtype=float)
        outliers = np.asarray([row["outlier_fraction"] for row in rows], dtype=float)
        decisions = Counter(str(row["robustness_decision"]) for row in rows)
        shapes = Counter(str(row["shape_status"]) for row in rows)
        summary.append(
            {
                "scenario": scenario,
                "family": family,
                "n": n,
                "replications": len(rows),
                "nominal_contamination": rows[0]["nominal_contamination"],
                "ci_coverage": float(np.mean([row["ci_covered"] for row in rows])),
                "type_i_error": float(np.mean([row["type_i_rejection"] for row in rows])),
                "median_abs_skewness": float(np.median(np.abs(skew))),
                "p90_abs_skewness": float(np.quantile(np.abs(skew), 0.90)),
                "median_abs_excess_kurtosis": float(np.median(np.abs(kurtosis))),
                "p90_abs_excess_kurtosis": float(np.quantile(np.abs(kurtosis), 0.90)),
                "mean_outlier_fraction": float(np.mean(outliers)),
                "p90_outlier_fraction": float(np.quantile(outliers, 0.90)),
                "shape_pass_rate": shapes["pass"] / len(rows),
                "shape_warn_rate": shapes["warn"] / len(rows),
                "shape_fail_rate": shapes["fail"] / len(rows),
                "decision_acceptable_rate": decisions["acceptable"] / len(rows),
                "decision_caution_rate": decisions["caution"] / len(rows),
                "decision_insufficient_rate": decisions["insufficient"] / len(rows),
            }
        )
    return sorted(summary, key=lambda row: (str(row["scenario"]), int(row["n"])))


def write_results(
    records: Sequence[dict[str, object]],
    output_dir: Path,
    *,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)

    with gzip.open(output_dir / "sampling_robustness_replicates.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    with (output_dir / "sampling_robustness_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    (output_dir / "sampling_robustness_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()

    records = simulate(
        replications=args.replications,
        sample_sizes=args.sample_sizes,
        seed=args.seed,
        alpha=args.alpha,
    )
    write_results(
        records,
        args.output_dir,
        metadata={
            "alpha": args.alpha,
            "replications": args.replications,
            "sample_sizes": args.sample_sizes,
            "scenario_count": len(scenarios()),
            "seed": args.seed,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "sampling_robustness_policy": SamplingRobustness.POLICY_VERSION,
        },
    )


if __name__ == "__main__":
    main()
