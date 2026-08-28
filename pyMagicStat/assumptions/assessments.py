from typing import Any, Iterable, Sequence, Tuple

import numpy as np
import scipy.stats as stats

from pyMagicStat._descriptive import sample_shape_statistics
from pyMagicStat.assumptions.models import Assessment, AssessmentStatus


class DataQualityAssessment:
    """Validate shape, numeric type, finiteness, size and degeneracy."""

    def __init__(self, min_size: int = 2) -> None:
        self.min_size = int(min_size)

    def normalize(self, data: Any, label: str = "sample") -> Tuple[np.ndarray, Assessment]:
        try:
            array = np.asarray(data, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must contain numeric data") from exc

        reasons = []
        if array.ndim != 1:
            reasons.append(f"{label} must be one-dimensional")
        if array.size < self.min_size:
            reasons.append(f"{label} must contain at least {self.min_size} observations")
        if array.size and not np.all(np.isfinite(array)):
            reasons.append(f"{label} contains NaN or infinite values")

        finite = bool(array.size and np.all(np.isfinite(array)))
        distinct = int(np.unique(array).size) if array.ndim == 1 and array.size else 0
        variance = (
            float(np.var(array, ddof=1))
            if array.ndim == 1 and array.size > 1 and finite
            else np.nan
        )
        scale = (
            float(np.max(np.abs(array)))
            if array.ndim == 1 and array.size and finite
            else np.nan
        )
        variance_tolerance = (
            float((np.finfo(float).eps * scale) ** 2)
            if np.isfinite(scale)
            else np.nan
        )
        if (
            array.ndim == 1
            and array.size >= self.min_size
            and (
                distinct < 2
                or not np.isfinite(variance)
                or variance <= variance_tolerance
            )
        ):
            reasons.append(f"{label} has zero variance or numerically negligible variance")

        assessment = Assessment(
            name=f"data_quality_{label}",
            status=AssessmentStatus.FAIL if reasons else AssessmentStatus.PASS,
            metrics={
                "n": int(array.size),
                "distinct": distinct,
                "variance": variance,
                "variance_tolerance": variance_tolerance,
                "missing": int(np.count_nonzero(~np.isfinite(array))) if array.size else 0,
            },
            reasons=tuple(reasons) if reasons else ("Data are finite, one-dimensional and non-degenerate.",),
        )
        return array, assessment


class ShapeAssessment:
    """Separate exact-normality evidence from observed departure magnitude."""

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)

    def assess(self, data: Any, label: str = "sample") -> Assessment:
        source = getattr(data, "data", data)
        array = np.asarray(source)

        if all(
            hasattr(data, attribute)
            for attribute in ("n", "skewness", "excess_kurtosis")
        ):
            n = int(data.n)
            skewness = float(data.skewness)
            excess_kurtosis = float(data.excess_kurtosis)
        else:
            n, skewness, excess_kurtosis = sample_shape_statistics(array)

        shapiro_p = np.nan
        if 3 <= n <= 5000:
            shapiro_p = float(stats.shapiro(array).pvalue)

        dagostino_p = np.nan
        if n >= 8:
            dagostino_p = float(stats.normaltest(array).pvalue)

        shapiro_rejects = bool(shapiro_p < self.alpha) if np.isfinite(shapiro_p) else None
        dagostino_rejects = (
            bool(dagostino_p < self.alpha) if np.isfinite(dagostino_p) else None
        )
        available_rejections = [
            rejects
            for rejects in (shapiro_rejects, dagostino_rejects)
            if rejects is not None
        ]
        exact_normality_rejected = (
            bool(any(available_rejections)) if available_rejections else None
        )

        finite_shape = np.isfinite(skewness) or np.isfinite(excess_kurtosis)
        abs_skew = abs(skewness) if np.isfinite(skewness) else 0.0
        abs_kurtosis = abs(excess_kurtosis) if np.isfinite(excess_kurtosis) else 0.0

        if not finite_shape:
            departure_magnitude = "not_assessed"
            status = AssessmentStatus.NOT_ASSESSED
            shape_reason = "Observed shape magnitude could not be assessed at this sample size."
        elif abs_skew > 2.0 or abs_kurtosis > 7.0:
            departure_magnitude = "severe"
            status = AssessmentStatus.FAIL
            shape_reason = "Observed skewness or tail weight indicates a severe shape departure."
        elif abs_skew > 1.0 or abs_kurtosis > 3.0:
            departure_magnitude = "moderate"
            status = AssessmentStatus.WARN
            shape_reason = "Observed skewness or tail weight indicates a moderate shape departure."
        else:
            departure_magnitude = "mild"
            status = AssessmentStatus.PASS
            shape_reason = "Observed skewness and tail weight indicate only a mild shape departure."

        if exact_normality_rejected is True:
            exact_reason = (
                "At least one formal test rejects exact Gaussianity; this is evidence, "
                "not an independent veto on mean-based inference."
            )
        elif exact_normality_rejected is False:
            exact_reason = "The available formal tests do not reject exact Gaussianity."
        else:
            exact_reason = "Formal exact-normality tests were not available at this sample size."

        return Assessment(
            name=f"shape_{label}",
            status=status,
            metrics={
                "n": n,
                "skewness": skewness,
                "excess_kurtosis": excess_kurtosis,
                "shapiro_p_value": shapiro_p,
                "dagostino_p_value": dagostino_p,
                "shapiro_rejects_exact_normality": shapiro_rejects,
                "dagostino_rejects_exact_normality": dagostino_rejects,
                "exact_normality_rejected": exact_normality_rejected,
                "departure_magnitude": departure_magnitude,
                "alpha": self.alpha,
            },
            reasons=(shape_reason, exact_reason),
        )


class OutlierAssessment:
    """Detect extreme observations using MAD with an IQR fallback."""

    def __init__(self, modified_z_threshold: float = 3.5) -> None:
        self.modified_z_threshold = float(modified_z_threshold)

    def assess(self, data: np.ndarray, label: str = "sample") -> Assessment:
        median = float(np.median(data))
        absolute_deviation = np.abs(data - median)
        mad = float(np.median(absolute_deviation))

        if mad > 0.0:
            scores = 0.6744897501960817 * absolute_deviation / mad
            indices = np.flatnonzero(scores > self.modified_z_threshold)
            method = "modified_z_score"
        else:
            q1, q3 = np.percentile(data, [25, 75])
            iqr = float(q3 - q1)
            if iqr > 0.0:
                lower = q1 - 3.0 * iqr
                upper = q3 + 3.0 * iqr
                indices = np.flatnonzero((data < lower) | (data > upper))
            else:
                indices = np.array([], dtype=int)
            method = "extreme_iqr"

        count = int(indices.size)
        status = AssessmentStatus.WARN if count else AssessmentStatus.PASS
        reasons = (
            (f"{count} extreme observation(s) may materially influence mean-based inference.")
            if count
            else "No extreme observations were detected by the robust rule."
        ,)
        return Assessment(
            name=f"outliers_{label}",
            status=status,
            metrics={
                "count": count,
                "fraction": float(count / data.size),
                "indices": indices,
                "method": method,
                "threshold": self.modified_z_threshold,
            },
            reasons=reasons,
        )


class VarianceAssessment:
    """Report heteroscedasticity without using it as a binary test selector."""

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)

    def assess(self, groups: Sequence[np.ndarray]) -> Assessment:
        if len(groups) < 2:
            return Assessment(
                name="variance",
                status=AssessmentStatus.NOT_ASSESSED,
                reasons=("Variance comparison requires at least two groups.",),
            )

        variances = np.asarray([np.var(group, ddof=1) for group in groups], dtype=float)
        min_variance = float(np.min(variances))
        ratio = float(np.max(variances) / min_variance) if min_variance > 0.0 else np.inf
        statistic, p_value = stats.levene(*groups, center="median")
        p_value = float(p_value)
        heterogeneous = bool(p_value < self.alpha)
        unbalanced = max(len(group) for group in groups) / min(len(group) for group in groups)
        status = AssessmentStatus.WARN if heterogeneous or ratio > 4.0 else AssessmentStatus.PASS
        reasons = (
            "Variance differences were detected; variance-robust inference is preferred."
            if status is AssessmentStatus.WARN
            else "No material variance difference was detected."
        ,)
        return Assessment(
            name="variance",
            status=status,
            metrics={
                "variances": variances,
                "variance_ratio": ratio,
                "levene_statistic": float(statistic),
                "levene_p_value": p_value,
                "alpha": self.alpha,
                "heterogeneous": heterogeneous,
                "size_ratio": float(unbalanced),
            },
            reasons=reasons,
        )


class IndependenceAssessment:
    """Represent design knowledge that cannot be inferred from observed values."""

    _VALID = {"unknown", "assumed", "verified"}

    def assess(self, independence: str = "unknown") -> Assessment:
        value = str(independence).lower()
        if value not in self._VALID:
            raise ValueError("independence must be 'unknown', 'assumed', or 'verified'")
        if value == "unknown":
            return Assessment(
                name="independence",
                status=AssessmentStatus.NOT_ASSESSED,
                metrics={"independence": value},
                reasons=("Independence cannot be inferred from the observed values.",),
            )
        return Assessment(
            name="independence",
            status=AssessmentStatus.PASS,
            metrics={"independence": value},
            reasons=(f"Independence is {value} from the study design.",),
        )
