import warnings

import numpy as np

from pyMagicStat.assumptions import InferenceValidator, RobustnessLevel
from pyMagicStat.inference import MethodSelector
from pyMagicStat.inference.parametric import PopulationMeanCI, TwoSampleTTest


def test_no_automatic_robustness_discontinuity_at_n_30():
    base = np.random.default_rng(4).lognormal(mean=0.0, sigma=1.8, size=31)
    validator = InferenceValidator()
    selector = MethodSelector()

    at_29 = selector.select(validator.validate_one_sample(base[:29]).report)
    at_30 = selector.select(validator.validate_one_sample(base[:30]).report)

    assert at_29.robustness.level is RobustnessLevel.INSUFFICIENT
    assert at_30.robustness.level is RobustnessLevel.INSUFFICIENT
    assert at_29.selected_method is None
    assert at_30.selected_method is None


def test_student_mean_interval_has_reasonable_seeded_coverage():
    rng = np.random.default_rng(8128)
    covered = 0
    replications = 300
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(replications):
            sample = rng.normal(loc=3.0, scale=2.0, size=20)
            interval = PopulationMeanCI(sample, strict=False).calculate_interval()
            covered += interval["lb"] <= 3.0 <= interval["ub"]

    coverage = covered / replications
    assert 0.91 <= coverage <= 0.98


def test_welch_default_controls_seeded_type_one_error_under_heteroscedasticity():
    rng = np.random.default_rng(2027)
    rejected = 0
    replications = 300
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(replications):
            small_low_variance = rng.normal(loc=0.0, scale=1.0, size=20)
            large_high_variance = rng.normal(loc=0.0, scale=3.0, size=50)
            result = TwoSampleTTest(
                small_low_variance,
                large_high_variance,
                strict=False,
            ).run_test()
            rejected += result["reject_null"]
            assert result["method"] == "Welch's t-test"

    type_one_error = rejected / replications
    assert 0.02 <= type_one_error <= 0.08
