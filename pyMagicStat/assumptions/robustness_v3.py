"""Candidate v3 policy for one-sample IID inference about a population mean.

The legacy :mod:`pyMagicStat.assumptions.robustness` policy remains unchanged
and is still the default.  This module separates model provenance, empirical
sample evidence, counterfactual influence, process knowledge, and the action
level so the candidate can be audited before any holdout validation.

The continuous transition anchors were derived only from the checked-in
calibration experiment.  ``experiments/sampling_robustness_v3_calibration.py``
reproduces their derivation and compares this candidate with v2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Tuple

import numpy as np

from pyMagicStat.assumptions.models import (
    AssessmentStatus,
    AssumptionReport,
    Estimand,
    InferenceDesign,
)
from pyMagicStat.assumptions.robustness import RobustnessLevel


class AssumptionProvenance(str, Enum):
    """Origin of support for the model or sampling-process assumptions."""

    EXTERNAL = "external"
    EMPIRICAL = "empirical"
    UNKNOWN = "unknown"


class EmpiricalSupport(str, Enum):
    """What the current sample can say within the calibrated domain."""

    COMPATIBLE = "compatible"
    LIMITED = "limited"
    ADVERSE = "adverse"
    NOT_CALIBRATED = "not_calibrated"


class InfluenceRisk(str, Enum):
    """Counterfactual influence evidence, separate from extremeness."""

    LOW = "low"
    TRANSITION = "transition"
    ELEVATED = "elevated"
    UNKNOWN = "unknown"


class ProcessUncertainty(str, Enum):
    """External knowledge about contamination or process instability."""

    LOW = "low"
    UNKNOWN = "unknown"
    ELEVATED = "elevated"


@dataclass(frozen=True)
class RobustnessContext:
    """Optional information that cannot be inferred from sample values."""

    model_provenance: AssumptionProvenance = AssumptionProvenance.UNKNOWN
    process_uncertainty: ProcessUncertainty = ProcessUncertainty.UNKNOWN


@dataclass(frozen=True)
class CalibrationAnchors:
    """Auditable transition-band anchors learned from the calibration set.

    Compatible anchors are pooled 90th percentiles from confirmatory reference
    cells with target-conforming Type-I error and coverage.  Adverse anchors are
    pooled 25th percentiles from confirmatory cells with clearly deficient
    performance.  Values between each pair form a continuous caution band.
    """

    skewness_compatible: float = 0.664220
    skewness_adverse: float = 1.624111
    positive_kurtosis_compatible: float = 1.054041
    positive_kurtosis_adverse: float = 2.371427
    influence_compatible: float = 0.167155
    influence_elevated: float = 0.689094
    uncertainty_z: float = 1.959964


DEFAULT_CALIBRATION_ANCHORS = CalibrationAnchors()


@dataclass(frozen=True)
class RobustnessResultV3:
    """Multidimensional candidate result with legacy-compatible essentials."""

    level: RobustnessLevel
    reasons: Tuple[str, ...]
    model_support: AssumptionProvenance
    empirical_support: EmpiricalSupport
    influence: InfluenceRisk
    process_uncertainty: ProcessUncertainty
    policy_version: str
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "reasons": list(self.reasons),
            "model_support": self.model_support.value,
            "empirical_support": self.empirical_support.value,
            "influence": self.influence.value,
            "process_uncertainty": self.process_uncertainty.value,
            "policy_version": self.policy_version,
            "diagnostics": _json_ready(dict(self.diagnostics)),
        }


class SamplingRobustnessV3:
    """Candidate policy calibrated only for one-sample IID mean inference.

    No exact-normality p-value enters the risk calculation.  Sample size enters
    only through continuous uncertainty of sample skewness/kurtosis; there is no
    minimum-n or large-n approval rule.  An extreme count never changes the
    action by itself.  The counterfactual influence diagnostic can move an
    otherwise clear case into CAUTION, but never independently forces
    INSUFFICIENT.
    """

    POLICY_VERSION = "mean-v3-candidate-2026-08"
    CALIBRATION_TYPE_I_TARGET = 0.065
    CALIBRATION_COVERAGE_TARGET = 0.935

    def __init__(
        self,
        *,
        context: RobustnessContext | None = None,
        model_provenance: AssumptionProvenance | str = AssumptionProvenance.UNKNOWN,
        process_uncertainty: ProcessUncertainty | str = ProcessUncertainty.UNKNOWN,
        anchors: CalibrationAnchors = DEFAULT_CALIBRATION_ANCHORS,
    ) -> None:
        if context is not None and (
            AssumptionProvenance(model_provenance) is not AssumptionProvenance.UNKNOWN
            or ProcessUncertainty(process_uncertainty) is not ProcessUncertainty.UNKNOWN
        ):
            raise ValueError("Pass either context or explicit provenance/process values, not both")
        if context is not None:
            self.context = RobustnessContext(
                model_provenance=AssumptionProvenance(context.model_provenance),
                process_uncertainty=ProcessUncertainty(context.process_uncertainty),
            )
        else:
            self.context = RobustnessContext(
                model_provenance=AssumptionProvenance(model_provenance),
                process_uncertainty=ProcessUncertainty(process_uncertainty),
            )
        self.anchors = anchors
        self._validate_anchors()

    def evaluate(self, report: AssumptionReport) -> RobustnessResultV3:
        reasons: list[str] = []
        common = {
            "model_support": self.context.model_provenance,
            "process_uncertainty": self.context.process_uncertainty,
            "policy_version": self.POLICY_VERSION,
        }

        structural_failures = [
            item
            for name, item in report.assessments.items()
            if name.startswith("data_quality") and item.status is AssessmentStatus.FAIL
        ]
        if structural_failures:
            reasons.append("Structural data requirements failed.")
            return RobustnessResultV3(
                level=RobustnessLevel.INSUFFICIENT,
                reasons=tuple(reasons),
                empirical_support=EmpiricalSupport.ADVERSE,
                influence=InfluenceRisk.UNKNOWN,
                diagnostics={
                    "calibrated_domain": False,
                    "structural_failure_count": len(structural_failures),
                },
                **common,
            )

        calibrated_domain = (
            report.design is InferenceDesign.ONE_SAMPLE
            and report.estimand is Estimand.MEAN
        )
        if not calibrated_domain:
            reasons.append(
                "SamplingRobustnessV3 is calibrated only for one-sample IID inference about a mean."
            )
            return RobustnessResultV3(
                level=RobustnessLevel.CAUTION,
                reasons=tuple(reasons),
                empirical_support=EmpiricalSupport.NOT_CALIBRATED,
                influence=InfluenceRisk.UNKNOWN,
                diagnostics={"calibrated_domain": False},
                **common,
            )

        shapes = [
            item for name, item in report.assessments.items() if name.startswith("shape")
        ]
        outliers = [
            item for name, item in report.assessments.items() if name.startswith("outliers")
        ]
        independence_supported = any(
            name.startswith("independence")
            and item.status is AssessmentStatus.PASS
            and item.metrics.get("independence") in {"assumed", "verified"}
            for name, item in report.assessments.items()
        )
        independence_unknown = not independence_supported
        if not shapes:
            reasons.append("No shape assessment is available in the calibrated domain.")
            return RobustnessResultV3(
                level=RobustnessLevel.CAUTION,
                reasons=tuple(reasons),
                empirical_support=EmpiricalSupport.LIMITED,
                influence=InfluenceRisk.UNKNOWN,
                diagnostics={"calibrated_domain": True, "shape_available": False},
                **common,
            )

        n = min(int(item.metrics.get("n", 0)) for item in shapes)
        abs_skewness = max(
            (_finite_abs(item.metrics.get("skewness")) for item in shapes),
            default=np.nan,
        )
        positive_kurtosis = max(
            (_finite_positive(item.metrics.get("excess_kurtosis")) for item in shapes),
            default=np.nan,
        )
        shape = self._shape_evidence(abs_skewness, positive_kurtosis, n)
        influence_ratio, influence_available = self._influence_ratio(outliers)
        influence_score = (
            _transition_score(
                influence_ratio,
                self.anchors.influence_compatible,
                self.anchors.influence_elevated,
            )
            if influence_available
            else np.nan
        )
        if not influence_available:
            influence = InfluenceRisk.UNKNOWN
        elif influence_score <= 0.0:
            influence = InfluenceRisk.LOW
        elif influence_score >= 1.0:
            influence = InfluenceRisk.ELEVATED
        else:
            influence = InfluenceRisk.TRANSITION

        if shape["adverse_score"] >= 1.0:
            empirical_support = EmpiricalSupport.ADVERSE
        elif shape["upper_score"] <= 0.0:
            empirical_support = EmpiricalSupport.COMPATIBLE
        else:
            empirical_support = EmpiricalSupport.LIMITED

        extreme_count = sum(int(item.metrics.get("count", 0)) for item in outliers)
        extreme_fraction = max(
            (float(item.metrics.get("fraction", 0.0)) for item in outliers),
            default=0.0,
        )
        exact_rejection = any(
            item.metrics.get("exact_normality_rejected") is True for item in shapes
        )
        diagnostics = {
            "calibrated_domain": True,
            "n": n,
            "abs_skewness": abs_skewness,
            "positive_excess_kurtosis": positive_kurtosis,
            "shape_risk_score": shape["central_score"],
            "shape_risk_lower": shape["lower_score"],
            "shape_risk_upper": shape["upper_score"],
            "shape_adverse_joint_score": shape["adverse_score"],
            "skewness_uncertainty_95": shape["skewness_uncertainty"],
            "kurtosis_uncertainty_95": shape["kurtosis_uncertainty"],
            "influence_ratio": influence_ratio,
            "influence_score": influence_score,
            "extreme_count": extreme_count,
            "extreme_fraction": extreme_fraction,
            "exact_normality_rejected_descriptive_only": exact_rejection,
            "independence_unknown": independence_unknown,
            "calibration_anchors": asdict(self.anchors),
            "calibration_targets": {
                "type_i_max": self.CALIBRATION_TYPE_I_TARGET,
                "coverage_min": self.CALIBRATION_COVERAGE_TARGET,
            },
        }

        external_model_override = (
            self.context.model_provenance is AssumptionProvenance.EXTERNAL
            and self.context.process_uncertainty is ProcessUncertainty.LOW
        )
        if empirical_support is EmpiricalSupport.ADVERSE and not external_model_override:
            reasons.append("Observed shape is beyond the calibrated adverse envelope after uncertainty.")
            level = RobustnessLevel.INSUFFICIENT
        else:
            externally_supported_clear = (
                self.context.model_provenance is AssumptionProvenance.EXTERNAL
                and shape["central_score"] <= 0.0
            )
            empirically_clear = (
                self.context.model_provenance is AssumptionProvenance.EMPIRICAL
                and empirical_support is EmpiricalSupport.COMPATIBLE
            )
            clear_for_acceptance = (
                self.context.process_uncertainty is ProcessUncertainty.LOW
                and not independence_unknown
                and influence is InfluenceRisk.LOW
                and (externally_supported_clear or empirically_clear)
            )
            if clear_for_acceptance:
                reasons.append("Separated model, process, sample-shape and influence evidence supports the calibrated action.")
                level = RobustnessLevel.ACCEPTABLE
            else:
                level = RobustnessLevel.CAUTION
                self._append_caution_reasons(
                    reasons,
                    empirical_support=empirical_support,
                    influence=influence,
                    independence_unknown=independence_unknown,
                )

        return RobustnessResultV3(
            level=level,
            reasons=tuple(reasons),
            empirical_support=empirical_support,
            influence=influence,
            diagnostics=diagnostics,
            **common,
        )

    def _shape_evidence(self, skewness: float, kurtosis: float, n: int) -> dict[str, float]:
        if n > 0:
            skew_uncertainty = self.anchors.uncertainty_z * np.sqrt(6.0 / n)
            kurt_uncertainty = self.anchors.uncertainty_z * np.sqrt(24.0 / n)
        else:
            skew_uncertainty = np.inf
            kurt_uncertainty = np.inf

        central_skew = skewness if np.isfinite(skewness) else 0.0
        central_kurtosis = kurtosis if np.isfinite(kurtosis) else 0.0
        lower_skew = max(0.0, central_skew - skew_uncertainty)
        lower_kurtosis = max(0.0, central_kurtosis - kurt_uncertainty)
        upper_skew = central_skew + skew_uncertainty if np.isfinite(skewness) else np.inf
        upper_kurtosis = (
            central_kurtosis + kurt_uncertainty if np.isfinite(kurtosis) else np.inf
        )

        def score(skew_value: float, kurtosis_value: float) -> float:
            return max(
                _transition_score(
                    skew_value,
                    self.anchors.skewness_compatible,
                    self.anchors.skewness_adverse,
                ),
                _transition_score(
                    kurtosis_value,
                    self.anchors.positive_kurtosis_compatible,
                    self.anchors.positive_kurtosis_adverse,
                ),
            )

        return {
            "central_score": score(central_skew, central_kurtosis),
            "lower_score": score(lower_skew, lower_kurtosis),
            "upper_score": score(upper_skew, upper_kurtosis),
            # Material adverse evidence requires both persistent asymmetry and
            # positive tail weight. Calibration showed that heavy symmetric
            # tails alone can preserve t-test operating characteristics.
            "adverse_score": min(
                _transition_score(
                    lower_skew,
                    self.anchors.skewness_compatible,
                    self.anchors.skewness_adverse,
                ),
                _transition_score(
                    lower_kurtosis,
                    self.anchors.positive_kurtosis_compatible,
                    self.anchors.positive_kurtosis_adverse,
                ),
            ),
            "skewness_uncertainty": float(skew_uncertainty),
            "kurtosis_uncertainty": float(kurt_uncertainty),
        }

    @staticmethod
    def _influence_ratio(outliers: list[Any]) -> tuple[float, bool]:
        ratios = []
        missing_with_extremes = False
        for item in outliers:
            value = item.metrics.get("influence_ratio")
            if value is not None and np.isfinite(value):
                ratios.append(float(value))
            elif int(item.metrics.get("count", 0)) > 0:
                missing_with_extremes = True
            else:
                ratios.append(0.0)
        if missing_with_extremes or not ratios:
            return np.nan, False
        return max(ratios), True

    def _append_caution_reasons(
        self,
        reasons: list[str],
        *,
        empirical_support: EmpiricalSupport,
        influence: InfluenceRisk,
        independence_unknown: bool,
    ) -> None:
        if self.context.model_provenance is AssumptionProvenance.UNKNOWN:
            reasons.append("Model-support provenance is unknown.")
        elif self.context.model_provenance is AssumptionProvenance.EMPIRICAL:
            reasons.append("Model support comes only from the current sample.")
        if self.context.process_uncertainty is ProcessUncertainty.UNKNOWN:
            reasons.append("External contamination/process risk is unknown and cannot be inferred from an apparently clean sample.")
        elif self.context.process_uncertainty is ProcessUncertainty.ELEVATED:
            reasons.append("External process knowledge indicates elevated contamination or instability.")
        if independence_unknown:
            reasons.append("Independence was not assessed from study-design metadata.")
        if empirical_support is EmpiricalSupport.LIMITED:
            reasons.append("Sampling uncertainty overlaps the calibrated shape transition band.")
        if influence is InfluenceRisk.TRANSITION:
            reasons.append("Counterfactual influence lies in the calibrated transition band.")
        elif influence is InfluenceRisk.ELEVATED:
            reasons.append("Counterfactual influence is elevated; this diagnostic does not authorize removing observations.")
        elif influence is InfluenceRisk.UNKNOWN:
            reasons.append("Influence could not be assessed separately from extremeness.")
        if not reasons:
            reasons.append("The evidence is transitional and does not support an ACCEPTABLE classification.")

    def _validate_anchors(self) -> None:
        pairs = (
            (self.anchors.skewness_compatible, self.anchors.skewness_adverse),
            (
                self.anchors.positive_kurtosis_compatible,
                self.anchors.positive_kurtosis_adverse,
            ),
            (self.anchors.influence_compatible, self.anchors.influence_elevated),
        )
        if any(not (np.isfinite(low) and np.isfinite(high) and 0.0 <= low < high) for low, high in pairs):
            raise ValueError("Each calibration transition requires finite 0 <= compatible < adverse anchors")
        if not np.isfinite(self.anchors.uncertainty_z) or self.anchors.uncertainty_z <= 0.0:
            raise ValueError("uncertainty_z must be positive and finite")


def _transition_score(value: float, compatible: float, adverse: float) -> float:
    """Continuous monotone evidence score with a nonzero caution band."""

    if not np.isfinite(value):
        return 1.0
    return float(np.clip((value - compatible) / (adverse - compatible), 0.0, 1.0))


def _finite_abs(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return abs(number) if np.isfinite(number) else np.nan


def _finite_positive(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return max(0.0, number) if np.isfinite(number) else np.nan


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
