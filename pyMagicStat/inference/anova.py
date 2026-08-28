"""One-way ANOVA executors built on the shared inference engine."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple
import warnings

import numpy as np
from scipy import stats

from pyMagicStat.assumptions import InferenceValidator
from pyMagicStat.inference.selector import MethodSelector


def _classical_anova_statistics(groups: Tuple[np.ndarray, ...]) -> Dict[str, float]:
    sizes = np.asarray([group.size for group in groups], dtype=float)
    means = np.asarray([np.mean(group) for group in groups], dtype=float)
    n_total = int(np.sum(sizes))
    k = len(groups)
    grand_mean = float(np.dot(sizes, means) / n_total)
    ss_between = float(np.sum(sizes * (means - grand_mean) ** 2))
    ss_within = float(
        sum(
            np.sum((group - mean) ** 2)
            for group, mean in zip(groups, means)
        )
    )
    numerator_df = k - 1
    denominator_df = n_total - k
    mean_square_between = ss_between / numerator_df
    mean_square_within = ss_within / denominator_df
    statistic = mean_square_between / mean_square_within
    return {
        "statistic": float(statistic),
        "p_value": float(stats.f.sf(statistic, numerator_df, denominator_df)),
        "numerator_df": float(numerator_df),
        "denominator_df": float(denominator_df),
        "ss_between": ss_between,
        "ss_within": ss_within,
        "mean_square_between": float(mean_square_between),
        "mean_square_within": float(mean_square_within),
        "eta_squared": float(ss_between / (ss_between + ss_within)),
    }


def _welch_anova_statistics(groups: Tuple[np.ndarray, ...]) -> Dict[str, Any]:
    sizes = np.asarray([group.size for group in groups], dtype=float)
    means = np.asarray([np.mean(group) for group in groups], dtype=float)
    variances = np.asarray([np.var(group, ddof=1) for group in groups], dtype=float)
    k = len(groups)
    weights = sizes / variances
    weight_total = float(np.sum(weights))
    weighted_mean = float(np.dot(weights, means) / weight_total)
    numerator_df = k - 1
    numerator = float(
        np.sum(weights * (means - weighted_mean) ** 2) / numerator_df
    )
    correction_term = float(
        np.sum(((1.0 - weights / weight_total) ** 2) / (sizes - 1.0))
    )
    correction = 1.0 + (
        2.0 * (k - 2.0) / (k**2 - 1.0)
    ) * correction_term
    statistic = numerator / correction
    denominator_df = (k**2 - 1.0) / (3.0 * correction_term)
    return {
        "statistic": float(statistic),
        "p_value": float(stats.f.sf(statistic, numerator_df, denominator_df)),
        "numerator_df": float(numerator_df),
        "denominator_df": float(denominator_df),
        "weights": tuple(float(value) for value in weights),
        "weighted_mean": weighted_mean,
        "welch_correction": float(correction),
    }


@dataclass(frozen=True)
class ANOVAResult:
    """Serializable result shared by classical and Welch one-way ANOVA."""

    method: str
    statistic: float
    p_value: float
    numerator_df: float
    denominator_df: float
    alpha: float
    reject_null: bool
    k: int
    n_total: int
    group_sizes: Tuple[int, ...]
    group_means: Tuple[float, ...]
    group_variances: Tuple[float, ...]
    equal_var_requested: bool
    variance_selection_policy: str
    assumptions: Dict[str, Any]
    inference_decision: Dict[str, Any]
    txt: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["Result"] = self.reject_null
        result["df"] = (self.numerator_df, self.denominator_df)
        return result


class _BaseOneWayANOVA:
    _expected_method: str
    _equal_var: bool
    _display_name: str

    def __init__(
        self,
        *groups: Any,
        alpha: float = 0.05,
        independence: str = "unknown",
        strict: bool = True,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        validation = InferenceValidator(alpha=alpha).validate_one_way(
            *groups,
            independence=independence,
        )
        self.groups = validation.samples
        self.assumption_report = validation.report
        self.inference_decision = MethodSelector().select(
            validation.report,
            equal_var=self._equal_var,
        )
        self.alpha = float(alpha)
        selected = self.inference_decision.selected_method
        if selected != self._expected_method:
            message = "; ".join(self.inference_decision.reasons)
            if strict:
                raise ValueError(
                    f"{self._display_name} is not recommended: {message}"
                )
            warnings.warn(message, UserWarning, stacklevel=2)

        self.k = len(self.groups)
        self.group_sizes = tuple(int(group.size) for group in self.groups)
        self.group_means = tuple(float(np.mean(group)) for group in self.groups)
        self.group_variances = tuple(
            float(np.var(group, ddof=1)) for group in self.groups
        )
        self.n_total = int(sum(self.group_sizes))

    def _result(
        self,
        *,
        statistic: float,
        p_value: float,
        numerator_df: float,
        denominator_df: float,
    ) -> Dict[str, Any]:
        reject_null = bool(p_value < self.alpha)
        txt = (
            f"Se rechaza H0 ({self._display_name}, p={p_value:.4e} < "
            f"alpha={self.alpha}): al menos una media poblacional difiere."
            if reject_null
            else f"No se rechaza H0 ({self._display_name}, p={p_value:.4e} >= "
            f"alpha={self.alpha}): no hay evidencia global suficiente de "
            "diferencias entre medias poblacionales."
        )
        result = ANOVAResult(
            method=self._display_name,
            statistic=float(statistic),
            p_value=float(p_value),
            numerator_df=float(numerator_df),
            denominator_df=float(denominator_df),
            alpha=self.alpha,
            reject_null=reject_null,
            k=self.k,
            n_total=self.n_total,
            group_sizes=self.group_sizes,
            group_means=self.group_means,
            group_variances=self.group_variances,
            equal_var_requested=self._equal_var,
            variance_selection_policy=(
                "explicit_classical" if self._equal_var else "explicit_welch"
            ),
            assumptions=self.assumption_report.to_dict(),
            inference_decision=self.inference_decision.to_dict(),
            txt=txt,
        )
        return result.to_dict()


class OneWayANOVA(_BaseOneWayANOVA):
    """Classical one-way ANOVA for a common within-group variance model."""

    _expected_method = "classical_anova"
    _equal_var = True
    _display_name = "Classical one-way ANOVA"

    def run_test(self) -> Dict[str, Any]:
        statistics = _classical_anova_statistics(self.groups)
        result = self._result(
            statistic=statistics["statistic"],
            p_value=statistics["p_value"],
            numerator_df=statistics["numerator_df"],
            denominator_df=statistics["denominator_df"],
        )
        result.update({key: value for key, value in statistics.items() if key not in result})
        return result


class WelchANOVA(_BaseOneWayANOVA):
    """Welch's heteroscedastic one-way ANOVA with Welch correction."""

    _expected_method = "welch_anova"
    _equal_var = False
    _display_name = "Welch one-way ANOVA"

    def run_test(self) -> Dict[str, Any]:
        statistics = _welch_anova_statistics(self.groups)
        result = self._result(
            statistic=statistics["statistic"],
            p_value=statistics["p_value"],
            numerator_df=statistics["numerator_df"],
            denominator_df=statistics["denominator_df"],
        )
        result.update({key: value for key, value in statistics.items() if key not in result})
        return result


__all__ = ["ANOVAResult", "OneWayANOVA", "WelchANOVA"]
