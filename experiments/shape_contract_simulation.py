"""Small reproducible audit of exact-normality evidence versus shape magnitude.

The runner deliberately stays small. It is a contract smoke test, not a
recalibration of ``SamplingRobustness`` thresholds.

Example
-------
python -m experiments.shape_contract_simulation --replications 50
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from pyMagicStat.assumptions import InferenceValidator, SamplingRobustness


DEFAULT_SAMPLE_SIZES = (30, 100, 750)
DEFAULT_SEED = 20260828


@dataclass(frozen=True)
class ShapeScenario:
    name: str
    generator: Callable[[np.random.Generator, int], np.ndarray]

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return np.asarray(self.generator(rng, n), dtype=float)


def scenarios() -> tuple[ShapeScenario, ...]:
    return (
        ShapeScenario("normal", lambda rng, n: rng.normal(size=n)),
        ShapeScenario("student_t_df10", lambda rng, n: rng.standard_t(10, size=n)),
        ShapeScenario("student_t_df5", lambda rng, n: rng.standard_t(5, size=n)),
        ShapeScenario(
            "lognormal_moderate",
            lambda rng, n: rng.lognormal(mean=0.0, sigma=0.5, size=n),
        ),
        ShapeScenario(
            "lognormal_severe",
            lambda rng, n: rng.lognormal(mean=0.0, sigma=1.25, size=n),
        ),
    )


def simulate(
    *,
    replications: int = 50,
    sample_sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, object]]:
    if replications < 1:
        raise ValueError("replications must be positive")
    if any(int(n) < 8 for n in sample_sizes):
        raise ValueError("sample sizes must be at least 8 for both formal tests")

    matrix = scenarios()
    streams = iter(np.random.SeedSequence(seed).spawn(len(matrix) * len(sample_sizes)))
    validator = InferenceValidator()
    policy = SamplingRobustness()
    records: list[dict[str, object]] = []

    for scenario in matrix:
        for n in sample_sizes:
            rng = np.random.default_rng(next(streams))
            for replication in range(replications):
                sample = scenario.draw(rng, int(n))
                report = validator.validate_one_sample(
                    sample,
                    independence="assumed",
                ).report
                shape = report.assessments["shape"]
                robustness = policy.evaluate(report)
                records.append(
                    {
                        "scenario": scenario.name,
                        "n": int(n),
                        "replication": replication,
                        "skewness": shape.metrics["skewness"],
                        "excess_kurtosis": shape.metrics["excess_kurtosis"],
                        "shapiro_rejects_exact_normality": shape.metrics[
                            "shapiro_rejects_exact_normality"
                        ],
                        "dagostino_rejects_exact_normality": shape.metrics[
                            "dagostino_rejects_exact_normality"
                        ],
                        "exact_normality_rejected": shape.metrics[
                            "exact_normality_rejected"
                        ],
                        "departure_magnitude": shape.metrics["departure_magnitude"],
                        "shape_status": shape.status.value,
                        "robustness": robustness.level.value,
                    }
                )
    return records


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
    }


def summarize(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["scenario"]), int(record["n"]))].append(record)

    summary: list[dict[str, object]] = []
    for (scenario, n), rows in grouped.items():
        count = len(rows)
        shapes = Counter(str(row["shape_status"]) for row in rows)
        robustness = Counter(str(row["robustness"]) for row in rows)
        magnitudes = Counter(str(row["departure_magnitude"]) for row in rows)
        skewness = np.asarray([row["skewness"] for row in rows], dtype=float)
        kurtosis = np.asarray([row["excess_kurtosis"] for row in rows], dtype=float)
        summary.append(
            {
                "scenario": scenario,
                "n": n,
                "replications": count,
                "shapiro_rejection_rate": float(
                    np.mean(
                        [row["shapiro_rejects_exact_normality"] for row in rows]
                    )
                ),
                "dagostino_rejection_rate": float(
                    np.mean(
                        [row["dagostino_rejects_exact_normality"] for row in rows]
                    )
                ),
                "skewness": _quantiles(skewness),
                "excess_kurtosis": _quantiles(kurtosis),
                "departure_magnitude_rates": {
                    key: value / count for key, value in sorted(magnitudes.items())
                },
                "shape_status_rates": {
                    status: shapes[status] / count
                    for status in ("pass", "warn", "fail")
                },
                "robustness_rates": {
                    level: robustness[level] / count
                    for level in ("acceptable", "caution", "insufficient")
                },
            }
        )
    return sorted(summary, key=lambda row: (str(row["scenario"]), int(row["n"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES)
    args = parser.parse_args()

    records = simulate(
        replications=args.replications,
        sample_sizes=args.sample_sizes,
        seed=args.seed,
    )
    print(json.dumps(summarize(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
