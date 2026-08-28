# pyMagicStats

Applied statistics for Python with explicit diagnostics, method selection and
machine-readable explanations.

## Inference engine

pyMagicStat does not use `n >= 30` as a normality switch and does not bootstrap
sample means to simulate the central limit theorem. The inference workflow is:

```text
observations + study design + estimand
                ↓
structured assumption assessments
                ↓
versioned sampling-robustness policy
                ↓
method recommendation with reasons and alternatives
```

Diagnostics cover data quality, relevant distribution shape, extreme outliers,
variance differences, group balance and independence metadata. Paired inference
assesses differences; independent-group inference assesses within-group shape;
one-way validation assesses group-centered residuals.

See [the inference-engine guide](Docs/inference-engine.md) for the contract,
migration notes and examples.

One-way inference for `k >= 3` independent groups is available through
`OneWayANOVA` and `WelchANOVA`. Welch is the calibrated default selected by
`MethodSelector`; Classical requires an explicit `equal_var=True` request and
compatible joint variance diagnostics. See the
[ANOVA calibration](Docs/anova-calibration.md). Post-hoc comparisons are not
part of this release.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

Dependencies and test configuration are declared in `pyproject.toml`.

## Example: inspect before running a test

```python
import numpy as np

from pyMagicStat.assumptions import InferenceValidator
from pyMagicStat.inference import MethodSelector
from pyMagicStat.inference.parametric import TwoSampleTTest

rng = np.random.default_rng(42)
group_a = rng.normal(10, 2, 80)
group_b = rng.normal(12, 5, 70)

validation = InferenceValidator().validate_two_sample(
    group_a,
    group_b,
    independence="assumed",
)
decision = MethodSelector().select(validation.report)
print(decision.to_dict())

# Welch is the documented default. Student requires equal_var=True.
result = TwoSampleTTest(group_a, group_b).run_test()
print(result["method"], result["p_value"])
```

## Bootstrap and estimands

Bootstrap intervals can target the mean, median, variance, proportion or an
independent-group mean difference. The RNG is explicit and reproducible:

```python
from pyMagicStat.inference import BootstrapCI

interval = BootstrapCI(
    group_a,
    stat="mean",
    interval_method="bca",
    random_state=42,
).compute()
```

Rank procedures are not presented as automatic replacements for tests of means.
Their alternative estimands are included in `InferenceDecision.alternatives`.

For `stat="variance"`, bootstrap uses `ddof=1` by default. This targets the
usual population variance with the same sample-variance estimator in the
observed data and every resample. Pass `ddof=0` explicitly to bootstrap the
empirical/MLE second central moment instead.

Exact chi-square variance intervals have a separate population-model contract:
`strict=True` requires `population_normality="assumed"`. During the compatibility
transition, the historical `PopulationVarianceCI(data)` call still computes an
explicitly unvalidated interval and emits `FutureWarning`; a sample normality
test cannot establish the assumption, and sample size does not replace it.

`TwoSampleTTest` keeps Welch as the variance-robust default when
`equal_var=None`; Levene is diagnostic and does not switch the method. Legacy
`is_normal` and `tlc_applied` group keys remain temporarily available.
`tlc_applied` is deprecated, always `False`, and never means that bootstrap or a
CLT operation was executed.

With an explicit `random_state`, repeated `BootstrapCI.compute()` calls are
idempotent for both SciPy and Numba backends and do not advance a caller-owned
generator.

The mean-inference thresholds and their 152,000-replicate calibration are
documented in [the calibration report](Docs/sampling-robustness-calibration.md).

## Current scope

- distribution and shape assessment;
- confidence intervals for means, proportions and variances;
- one-sample, paired, Student and Welch t-tests;
- classical one-way and Welch ANOVA with a calibrated selector;
- bootstrap confidence intervals;
- Kruskal-Wallis and Mann-Whitney utilities;
- linear-regression diagnostics;
- one-way residual, influence, balance and variance diagnostics.

## License

Apache License 2.0. See [LICENSE](LICENSE).
