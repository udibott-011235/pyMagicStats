import warnings

import numpy as np
import pytest

from pyMagicStat.assumptions import (
    Assessment,
    AssessmentStatus,
    AssumptionReport,
    InferenceValidator,
    OneWayRobustness,
    RobustnessLevel,
)
from pyMagicStat.inference import MethodSelector, WelchANOVA


def test_single_dominating_outlier_cannot_bypass_one_way_hard_constraints():
    ordinary = np.linspace(0.9, 1.4, 12)
    contaminated = ordinary.copy()
    contaminated[-1] = 100.0
    report = InferenceValidator().validate_one_way(
        contaminated,
        ordinary + 0.1,
        ordinary + 0.2,
        independence="assumed",
    ).report

    assert report.assessments["outliers_group_1"].metrics["count"] == 1
    assert report.assessments["outliers_group_1"].metrics["max_robust_score"] >= 8.0

    # Reproduce the historical class of shortcut bugs: even if shape were
    # controlled to PASS, the calibrated influence guardrail must run first.
    assessments = dict(report.assessments)
    for name, item in list(assessments.items()):
        if name.startswith("shape"):
            assessments[name] = Assessment(
                name=item.name,
                status=AssessmentStatus.PASS,
                metrics={**item.metrics, "skewness": 0.2, "excess_kurtosis": 0.2},
                reasons=("Controlled compatible-shape diagnostic.",),
            )
    controlled = AssumptionReport(
        design=report.design,
        estimand=report.estimand,
        assessments=assessments,
    )

    robustness = OneWayRobustness().evaluate(controlled)
    decision = MethodSelector().select(controlled)

    assert robustness.level is RobustnessLevel.INSUFFICIENT
    assert decision.selected_method is None


def test_large_group_mean_offsets_are_not_misdiagnosed_as_pooled_non_normality():
    base = np.linspace(-2.0, 2.0, 31)
    report = InferenceValidator().validate_one_way(
        base,
        base + 1e6,
        base + 1e12,
        independence="verified",
    ).report

    residual_shape = report.assessments["shape_standardized_residuals"]
    assert abs(residual_shape.metrics["skewness"]) < 0.01
    assert abs(residual_shape.metrics["excess_kurtosis"]) < 2.0
    decision = MethodSelector().select(report)
    assert decision.selected_method == "welch_anova"


def test_classical_is_not_selected_from_non_rejection_of_variance_test_alone():
    rng = np.random.default_rng(440)
    report = InferenceValidator().validate_one_way(
        rng.normal(scale=5.0, size=8),
        rng.normal(scale=1.0, size=8),
        rng.normal(scale=1.0, size=8),
        independence="assumed",
    ).report
    variance = report.assessments["variance"]

    assert variance.metrics["variance_ratio"] > 4.0
    # Whether Brown-Forsythe rejects on this small sample is not the contract;
    # magnitude alone prevents an unsupported common-variance selection.
    decision = MethodSelector().select(report, equal_var=True)
    assert decision.selected_method is None


def test_unsupported_execution_requires_explicit_non_strict_override():
    base = np.linspace(0.9, 1.4, 12)
    contaminated = base.copy()
    contaminated[-1] = 100.0
    groups = (contaminated, base + 0.1, base + 0.2)

    with pytest.raises(ValueError, match="not recommended"):
        WelchANOVA(*groups, independence="assumed")

    with pytest.warns(UserWarning, match="guardrail"):
        result = WelchANOVA(
            *groups,
            independence="assumed",
            strict=False,
        ).run_test()
    assert result["inference_decision"]["status"] == "insufficient"


def test_one_way_decisions_and_statistics_are_deterministic_under_repetition():
    rng = np.random.default_rng(700)
    groups = tuple(rng.normal(index, 1.0, size=35) for index in range(3))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        first = WelchANOVA(*groups, independence="assumed", strict=False).run_test()
        second = WelchANOVA(*groups, independence="assumed", strict=False).run_test()

    assert first == second
