from typing import Any, Iterable, Sequence, Tuple

import numpy as np
import scipy.stats as stats

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
    """Describe departure from a Gaussian shape without deciding a method."""

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)

    def assess(self, data: np.ndarray, label: str = "sample") -> Assessment:
        n = int(data.size)
        skewness = float(stats.skew(data, bias=False)) if n >= 3 else np.nan
        excess_kurtosis = float(stats.kurtosis(data, fisher=True, bias=False)) if n >= 4 else np.nan

        shapiro_p = np.nan
        if 3 <= n <= 5000:
            shapiro_p = float(stats.shapiro(data).pvalue)

        dagostino_p = np.nan
        if n >= 8:
            dagostino_p = float(stats.normaltest(data).pvalue)

        abs_skew = abs(skewness) if np.isfinite(skewness) else np.inf
        abs_kurtosis = abs(excess_kurtosis) if np.isfinite(excess_kurtosis) else np.inf
        rejects = [p < self.alpha for p in (shapiro_p, dagostino_p) if np.isfinite(p)]

        if abs_skew > 2.0 or abs_kurtosis > 7.0:
            status = AssessmentStatus.FAIL
            reasons = ("Severe skewness or tail weight was detected.",)
        elif abs_skew > 1.0 or abs_kurtosis > 3.0 or any(rejects):
            status = AssessmentStatus.WARN
            reasons = ("The data show a material departure from a Gaussian shape.",)
        else:
            status = AssessmentStatus.PASS
            reasons = ("No material Gaussian-shape departure was detected.",)

        return Assessment(
            name=f"shape_{label}",
            status=status,
            metrics={
                "n": n,
                "skewness": skewness,
                "excess_kurtosis": excess_kurtosis,
                "shapiro_p_value": shapiro_p,
                "dagostino_p_value": dagostino_p,
                "alpha": self.alpha,
            },
            reasons=reasons,
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
            max_robust_score = float(np.max(scores))
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
            max_robust_score = (
                float(np.max(absolute_deviation / iqr)) if iqr > 0.0 else 0.0
            )

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
                "max_robust_score": max_robust_score,
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

        sizes = np.asarray([len(group) for group in groups], dtype=int)
        variances = np.asarray([np.var(group, ddof=1) for group in groups], dtype=float)
        min_variance = float(np.min(variances))
        ratio = float(np.max(variances) / min_variance) if min_variance > 0.0 else np.inf
        levene_statistic, levene_p_value = stats.levene(*groups, center="median")
        fligner_statistic, fligner_p_value = stats.fligner(*groups, center="median")
        bartlett_statistic, bartlett_p_value = stats.bartlett(*groups)
        levene_p_value = float(levene_p_value)
        fligner_p_value = float(fligner_p_value)
        bartlett_p_value = float(bartlett_p_value)
        heterogeneous = bool(
            levene_p_value < self.alpha or fligner_p_value < self.alpha
        )
        unbalanced = float(np.max(sizes) / np.min(sizes))
        if len(groups) >= 3 and np.ptp(sizes) > 0 and np.ptp(variances) > 0.0:
            size_variance_correlation = float(
                stats.spearmanr(sizes, variances).statistic
            )
        else:
            size_variance_correlation = np.nan
        small_group_large_variance = bool(
            np.isfinite(size_variance_correlation)
            and size_variance_correlation <= -0.5
            and ratio >= 2.0
            and unbalanced >= 1.5
        )
        status = (
            AssessmentStatus.WARN
            if heterogeneous or ratio > 4.0 or small_group_large_variance
            else AssessmentStatus.PASS
        )
        reasons = (
            "Variance differences were detected; variance-robust inference is preferred."
            if status is AssessmentStatus.WARN
            else "No material variance difference was detected."
        ,)
        return Assessment(
            name="variance",
            status=status,
            metrics={
                "sizes": sizes,
                "variances": variances,
                "variance_ratio": ratio,
                "levene_statistic": float(levene_statistic),
                "levene_p_value": levene_p_value,
                "brown_forsythe_statistic": float(levene_statistic),
                "brown_forsythe_p_value": levene_p_value,
                "fligner_statistic": float(fligner_statistic),
                "fligner_p_value": fligner_p_value,
                "bartlett_statistic": float(bartlett_statistic),
                "bartlett_p_value": bartlett_p_value,
                "alpha": self.alpha,
                "heterogeneous": heterogeneous,
                "size_ratio": unbalanced,
                "size_variance_spearman": size_variance_correlation,
                "small_group_large_variance": small_group_large_variance,
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
