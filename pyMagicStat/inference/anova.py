"""Explicit one-way Classical and Welch ANOVA engines.

The methods in this module deliberately compute requested statistics without
routing through :class:`MethodSelector`. Automatic ONE_WAY selection remains
not calibrated; this module is the explicit execution surface frozen by
CP-ANOVA-02/03.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple
import warnings

import numpy as np
from scipy import stats

from pyMagicStat.assumptions import AssessmentStatus, AssumptionReport, InferenceValidator


CLASSICAL_ANOVA_METHOD_VERSION = "classical-one-way-anova-v1"
WELCH_ANOVA_METHOD_VERSION = "welch-one-way-anova-v1"


@dataclass(frozen=True)
class _GroupSummary:
    """Sufficient per-group statistics used by both ANOVA kernels."""

    n: int
    mean: float
    variance: float
    ss_within: float


@dataclass(frozen=True)
class _ClassicalComputation:
    statistic: float
    p_value: float
    numerator_df: float
    denominator_df: float
    components: Mapping[str, Any]


@dataclass(frozen=True)
class _WelchComputation:
    statistic: float
    p_value: float
    numerator_df: float
    denominator_df: float
    components: Mapping[str, Any]


@dataclass(frozen=True)
class ANOVAResult:
    """Immutable structured result for explicit one-way ANOVA execution."""

    method: str
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    numerator_df: float
    denominator_df: float
    k: int
    n_total: int
    group_sizes: Tuple[int, ...]
    group_means: Tuple[float, ...]
    group_variances: Tuple[float, ...]
    assumptions: AssumptionReport
    diagnostics: Mapping[str, Any]
    components: Mapping[str, Any]
    method_version: str

    def __post_init__(self) -> None:
        # A frozen dataclass alone would still expose mutable dictionaries.
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation without exposing mutable internals."""

        return {
            "method": self.method,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "alpha": float(self.alpha),
            "reject_null": bool(self.reject_null),
            "numerator_df": float(self.numerator_df),
            "denominator_df": float(self.denominator_df),
            "k": int(self.k),
            "n_total": int(self.n_total),
            "group_sizes": [int(value) for value in self.group_sizes],
            "group_means": [float(value) for value in self.group_means],
            "group_variances": [float(value) for value in self.group_variances],
            "assumptions": self.assumptions.to_dict(),
            "diagnostics": _json_ready(self.diagnostics),
            "components": _json_ready(self.components),
            "method_version": self.method_version,
        }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _summarize_groups(groups: Sequence[np.ndarray]) -> Tuple[_GroupSummary, ...]:
    """Compute each group's summaries once in O(N)."""

    summaries = []
    for group in groups:
        n = int(group.size)
        mean = float(np.mean(group))
        variance = float(np.var(group, ddof=1))
        ss_within = float((n - 1) * variance)
        if not np.isfinite(mean) or not np.isfinite(variance) or not np.isfinite(ss_within):
            raise ValueError("ANOVA group summaries must be finite")
        if variance <= 0.0 or ss_within <= 0.0:
            raise ValueError("ANOVA groups must have positive, non-degenerate variance")
        summaries.append(_GroupSummary(n, mean, variance, ss_within))
    return tuple(summaries)


def _classical_kernel(summaries: Sequence[_GroupSummary]) -> _ClassicalComputation:
    """Compute Classical one-way ANOVA from group summaries in O(k)."""

    k = len(summaries)
    if k < 2:
        raise ValueError("Classical one-way ANOVA requires at least two groups")

    n_total = sum(summary.n for summary in summaries)
    numerator_df = float(k - 1)
    denominator_df = float(n_total - k)
    if denominator_df <= 0.0:
        raise ValueError("Classical one-way ANOVA requires positive within-group degrees of freedom")

    grand_mean = float(
        sum(summary.n * summary.mean for summary in summaries) / n_total
    )
    ss_between = float(
        sum(summary.n * (summary.mean - grand_mean) ** 2 for summary in summaries)
    )
    ss_within = float(sum(summary.ss_within for summary in summaries))
    ss_total = float(ss_between + ss_within)
    mean_square_between = float(ss_between / numerator_df)
    mean_square_within = float(ss_within / denominator_df)
    if mean_square_within <= 0.0 or not np.isfinite(mean_square_within):
        raise ValueError("Classical ANOVA within-group mean square must be finite and positive")

    statistic = float(mean_square_between / mean_square_within)
    p_value = float(stats.f.sf(statistic, numerator_df, denominator_df))
    eta_squared = float(ss_between / ss_total) if ss_total > 0.0 else 0.0
    _validate_kernel_output(statistic, p_value, numerator_df, denominator_df)

    components = MappingProxyType(
        {
            "grand_mean": grand_mean,
            "ss_between": ss_between,
            "ss_within": ss_within,
            "ss_total": ss_total,
            "mean_square_between": mean_square_between,
            "mean_square_within": mean_square_within,
            "eta_squared": eta_squared,
        }
    )
    return _ClassicalComputation(
        statistic,
        p_value,
        numerator_df,
        denominator_df,
        components,
    )


def _welch_kernel(summaries: Sequence[_GroupSummary]) -> _WelchComputation:
    """Compute Welch one-way ANOVA from group summaries in O(k)."""

    k = len(summaries)
    if k < 2:
        raise ValueError("Welch one-way ANOVA requires at least two groups")

    weights = np.asarray(
        [summary.n / summary.variance for summary in summaries],
        dtype=float,
    )
    means = np.asarray([summary.mean for summary in summaries], dtype=float)
    n_values = np.asarray([summary.n for summary in summaries], dtype=float)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum) or not np.all(np.isfinite(weights)):
        raise ValueError("Welch ANOVA weights must be finite and positive")

    weighted_mean = float(np.dot(weights, means) / weight_sum)
    numerator_df = float(k - 1)
    numerator = float(np.sum(weights * (means - weighted_mean) ** 2) / numerator_df)
    welch_b = float(np.sum((1.0 - weights / weight_sum) ** 2 / (n_values - 1.0)))
    if welch_b <= 0.0 or not np.isfinite(welch_b):
        raise ValueError("Welch ANOVA correction term must be finite and positive")

    correction = float(1.0 + (2.0 * (k - 2) / (k**2 - 1.0)) * welch_b)
    statistic = float(numerator / correction)
    denominator_df = float((k**2 - 1.0) / (3.0 * welch_b))
    p_value = float(stats.f.sf(statistic, numerator_df, denominator_df))
    _validate_kernel_output(statistic, p_value, numerator_df, denominator_df)

    components = MappingProxyType(
        {
            "weights": tuple(float(value) for value in weights),
            "weighted_mean": weighted_mean,
            "welch_B": welch_b,
            "welch_correction": correction,
        }
    )
    return _WelchComputation(
        statistic,
        p_value,
        numerator_df,
        denominator_df,
        components,
    )


def _validate_kernel_output(
    statistic: float,
    p_value: float,
    numerator_df: float,
    denominator_df: float,
) -> None:
    if not np.isfinite(statistic) or statistic < 0.0:
        raise ValueError("ANOVA statistic must be finite and non-negative")
    if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
        raise ValueError("ANOVA p-value must be finite and within [0, 1]")
    if numerator_df <= 0.0 or denominator_df <= 0.0:
        raise ValueError("ANOVA degrees of freedom must be positive")


class _BaseOneWayANOVA:
    _METHOD = ""
    _METHOD_VERSION = ""

    def __init__(
        self,
        *groups: Any,
        alpha: float = 0.05,
        independence: str = "unknown",
        strict: bool = True,
    ) -> None:
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if len(groups) < 2:
            raise ValueError("One-way ANOVA requires at least two groups")

        self.alpha = float(alpha)
        self.strict = bool(strict)
        validation = InferenceValidator(alpha=self.alpha).validate_one_way(
            *groups,
            independence=independence,
        )
        # Copies prevent later mutation of caller-owned arrays from changing an
        # already-constructed ANOVA object.
        self._groups = tuple(np.array(group, dtype=float, copy=True) for group in validation.samples)
        self.assumption_report = validation.report
        self._summaries = _summarize_groups(self._groups)

        structural_reasons = _structural_failure_reasons(self.assumption_report)
        if structural_reasons:
            message = "; ".join(structural_reasons)
            if self.strict:
                raise ValueError(f"ANOVA structural assumptions failed: {message}")
            warnings.warn(
                f"ANOVA computed despite unresolved structural assumptions: {message}",
                UserWarning,
                stacklevel=2,
            )

    @property
    def summaries(self) -> Tuple[_GroupSummary, ...]:
        """Read-only internal summaries useful for audit/testing, not public export."""

        return self._summaries

    def _diagnostics(self) -> Mapping[str, Any]:
        assessments = self.assumption_report.assessments
        independence = assessments.get("independence")
        independence_value = (
            independence.metrics.get("independence") if independence is not None else None
        )
        diagnostic_flags = tuple(
            name
            for name, assessment in assessments.items()
            if assessment.status in {AssessmentStatus.WARN, AssessmentStatus.FAIL}
            and not name.startswith("data_quality")
        )
        unresolved = tuple(
            name
            for name, assessment in assessments.items()
            if assessment.status is AssessmentStatus.NOT_ASSESSED
        )
        return MappingProxyType(
            {
                "strict": self.strict,
                "independence": independence_value,
                "diagnostic_flags": diagnostic_flags,
                "unresolved_assumptions": unresolved,
                "automatic_selection_calibrated": False,
            }
        )

    def _assemble(self, computation: Any) -> ANOVAResult:
        return ANOVAResult(
            method=self._METHOD,
            statistic=float(computation.statistic),
            p_value=float(computation.p_value),
            alpha=self.alpha,
            reject_null=bool(computation.p_value < self.alpha),
            numerator_df=float(computation.numerator_df),
            denominator_df=float(computation.denominator_df),
            k=len(self._summaries),
            n_total=sum(summary.n for summary in self._summaries),
            group_sizes=tuple(summary.n for summary in self._summaries),
            group_means=tuple(summary.mean for summary in self._summaries),
            group_variances=tuple(summary.variance for summary in self._summaries),
            assumptions=self.assumption_report,
            diagnostics=self._diagnostics(),
            components=computation.components,
            method_version=self._METHOD_VERSION,
        )


def _structural_failure_reasons(report: AssumptionReport) -> Tuple[str, ...]:
    """Return hard design failures without treating diagnostic shape as a veto."""

    reasons = []
    for name, assessment in report.assessments.items():
        if assessment.status is not AssessmentStatus.FAIL:
            continue
        # Data quality is already rejected by InferenceValidator and remains a
        # hard failure. Shape/outlier/variance FAILs are diagnostic evidence,
        # not automatic method switches or structural execution failures.
        if name.startswith("shape") or name.startswith("outliers") or name == "variance":
            continue
        reasons.extend(assessment.reasons)
    return tuple(reasons)


class OneWayANOVA(_BaseOneWayANOVA):
    """Explicit Classical one-way ANOVA for independent groups."""

    _METHOD = "classical_one_way_anova"
    _METHOD_VERSION = CLASSICAL_ANOVA_METHOD_VERSION

    def run(self) -> ANOVAResult:
        return self._assemble(_classical_kernel(self._summaries))


class WelchANOVA(_BaseOneWayANOVA):
    """Explicit Welch one-way ANOVA for independent heteroscedastic groups."""

    _METHOD = "welch_one_way_anova"
    _METHOD_VERSION = WELCH_ANOVA_METHOD_VERSION

    def run(self) -> ANOVAResult:
        return self._assemble(_welch_kernel(self._summaries))
