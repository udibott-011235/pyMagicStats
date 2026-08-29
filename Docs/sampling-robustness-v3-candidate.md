# SamplingRobustness v3 — candidate policy

## Status

`SamplingRobustnessV3` is a **candidate**, not the production default. It was
calibrated on the existing one-sample IID mean experiment and has not been
evaluated on a holdout. The legacy `SamplingRobustness` v2 policy remains
unchanged and `MethodSelector()` still constructs v2 unless a caller explicitly
injects v3.

```python
from pyMagicStat.assumptions import (
    AssumptionProvenance,
    ProcessUncertainty,
    SamplingRobustnessV3,
)
from pyMagicStat.inference import MethodSelector

candidate = SamplingRobustnessV3(
    model_provenance=AssumptionProvenance.EXTERNAL,
    process_uncertainty=ProcessUncertainty.LOW,
)
selector = MethodSelector(robustness_policy=candidate)
```

The calibrated domain is only one-sample IID inference about an arithmetic
mean. Paired, two-sample, one-way ANOVA, regression, and dependent observations
are not validated by this evidence. A direct v3 evaluation outside that domain
returns `CAUTION` with `empirical_support=NOT_CALIBRATED`. The existing
`MethodSelector` continues to mark one-way designs as `NOT_CALIBRATED` before
calling either mean policy. With an injected v3 policy, paired and two-sample
designs also return `selected_method=None` and `status=NOT_CALIBRATED`.

## Separated evidence contract

`RobustnessResultV3` retains the two legacy essentials, `level` and `reasons`,
and adds:

- `model_support`: provenance is `EXTERNAL`, `EMPIRICAL`, or `UNKNOWN`;
- `empirical_support`: `COMPATIBLE`, `LIMITED`, `ADVERSE`, or
  `NOT_CALIBRATED`;
- `influence`: `LOW`, `TRANSITION`, `ELEVATED`, or `UNKNOWN`;
- `process_uncertainty`: `LOW`, `UNKNOWN`, or `ELEVATED`;
- `policy_version`: `mean-v3-candidate-2026-08`;
- `diagnostics`: the continuous evidence and calibration anchors.

Provenance is caller context. It is never inferred from Shapiro, D'Agostino,
skewness, or the apparent cleanliness of the sample. `UNKNOWN` is the default
and remains a first-class state. Likewise, contamination/process risk is not
set to `LOW` merely because no outlier was observed.

Independence continues to come from `IndependenceAssessment`. V3 recognizes
support only when an explicit assessment is `PASS` and its study-metadata
indicator is `assumed` or `verified`. A missing assessment, `NOT_ASSESSED`,
or any other metric is unknown; v3 reports that uncertainty and cannot emit
`ACCEPTABLE`. Independence is never inferred from sample values.

## Continuous shape evidence

V3 does not fit an arbitrary weighted score. It maintains separate monotone
transition scores for absolute skewness and positive excess kurtosis:

```text
dimension_score(x; compatible, adverse)
    = clip((x - compatible) / (adverse - compatible), 0, 1)
```

The central shape-risk diagnostic is the maximum of the two dimension scores.
This is a conservative logical union rather than a fitted weighted average.
Material adverse shape evidence requires **both** persistent asymmetry and
positive tail weight. This conjunction is important: calibration showed that
heavy symmetric tails can preserve t-test Type-I error and coverage.

The anchors are reproduced by
`experiments/sampling_robustness_v3_calibration.py`:

| Dimension | Compatible endpoint | Adverse endpoint |
|---|---:|---:|
| absolute skewness | .664220 | 1.624111 |
| positive excess kurtosis | 1.054041 | 2.371427 |
| influence / SE | .167155 | .689094 |

Compatible endpoints are pooled 90th percentiles from 45,000 observations in
confirmatory cells that met the provisional Type-I/coverage targets. Adverse
endpoints are pooled 25th percentiles from 20,000 observations in clearly
deficient confirmatory cells. The interval between endpoints is intentionally a
`CAUTION` band, not an optimized binary separator.

The policy exposes approximate 95% sampling-uncertainty envelopes using
`1.959964 * sqrt(6/n)` for skewness and `1.959964 * sqrt(24/n)` for kurtosis.
These are diagnostics, not Gaussianity tests. Adverse action requires the lower
envelope to reach the joint adverse region. Sample size therefore changes the
precision of evidence continuously; it never grants or denies inference via
`MIN_N`, `n >= 30`, or an equivalent rule.

Categorical boundaries still exist because the caller requires one of three
actions, but `ACCEPTABLE` and `INSUFFICIENT` are separated by a nonzero
continuous `CAUTION` region. Dense tests over 1,401 perturbations verify that no
infinitesimal change can jump directly from `ACCEPTABLE` to `INSUFFICIENT`.
The old boundaries 1/2 skewness and 3/7 kurtosis have no special status in v3.

Formal exact-Gaussian rejection is retained in diagnostics as
`exact_normality_rejected_descriptive_only`; its p-values never enter the score
or action.

## Small samples and model support

Small n is not automatically invalid. With externally supported Gaussian
sampling, a stable n=3 sample may be `ACCEPTABLE` even though its
`empirical_support` is `LIMITED`. These statements are intentionally separate:

- the external model can justify the exact t distribution;
- three observations cannot demonstrate robustness against plausible
  non-Gaussian processes.

With `UNKNOWN` provenance, the same sample is `CAUTION`, not `INSUFFICIENT`.
This reports limited evidence without inventing a minimum sample size.

## Extremeness versus influence

`OutlierAssessment` retains its modified-z/IQR detector, threshold, indices,
count, and fraction. It now also reports a non-mutating sensitivity diagnostic:

```text
delta_mean_remove_extremes
    = abs(mean_full - mean_without_detected_extremes)

influence_ratio
    = delta_mean_remove_extremes / SE_full
```

The original array is never modified. No observation is removed, winsorized,
or used to change the estimand. “Without extremes” is a counterfactual
sensitivity calculation, not a recommendation.

V3 ignores count/fraction as action triggers. Low influence can coexist with
many extremes in a large normal sample. Influence in its transition/elevated
band moves an otherwise clear case to `CAUTION`, but influence alone never
forces `INSUFFICIENT`.

## Action semantics

`ACCEPTABLE` requires all of the following:

- calibrated one-sample mean domain and no structural failure;
- assessed independence;
- process uncertainty explicitly `LOW`;
- low counterfactual influence;
- either externally supported model evidence with a compatible central shape,
  or empirical provenance whose full uncertainty envelope is compatible.

`CAUTION` represents unknown provenance/process knowledge, limited empirical
precision, a continuous transition region, elevated influence, or an
externally supported model in tension with the sample. A direct v3 evaluation
also uses this level with `empirical_support=NOT_CALIBRATED` outside its
domain; the selector maps that result to the distinct `NOT_CALIBRATED` status.

`INSUFFICIENT` is reserved for structural failures or material joint adverse
shape evidence not resolved by an externally supported low-risk process.
Process risk `ELEVATED` alone produces `CAUTION`; it is not a substitute for the
degree of risk that this three-state context cannot express.

The production-default v2 selector continues to select a t method for both
`ACCEPTABLE` and `CAUTION`, and withholds it for `INSUFFICIENT`. With an
explicitly injected v3 policy, only `ACCEPTABLE` authorizes automatic
selection. V3 `CAUTION` returns `selected_method=None` with
`REVIEW_REQUIRED`; `INSUFFICIENT` and `NOT_CALIBRATED` also return no
method with their respective statuses. Bootstrap, rank, and permutation
procedures remain labeled alternatives and are never selected automatically.

## Calibration comparison: v2 versus v3

The comparison reused all 237,800 existing calibration replications. No
external holdout or Antigravity artifact was read. Provisional targets were
Type I ≤.065 and coverage ≥.935. A region below reports a reliable conditional
result only when its cell had at least 5,000 total replications and its action
denominator was at least 200.

The reporting separates the contexts that the calibration can actually
support:

- `v3_default` is the sample/default profile, with provenance and process
  uncertainty both `UNKNOWN`;
- no realistic externally supplied context is available in this artifact,
  because it contains no study metadata independent of generator truth;
- `v3_oracle_simulation_truth` is an oracle sensitivity profile. It uses the
  true simulation family/scenario to set external provenance and process risk
  and is excluded from headline real-world safety claims.

Headline false-safe reporting includes the total `ACCEPTABLE` exposure:

| Profile | Confirmatory false-safe regions | ACCEPTABLE denominator | Overall ACCEPTABLE rate |
|---|---:|---:|---:|
| v2 legacy | 5 | 112,694 / 237,800 | 47.3902% |
| v3 default/sample | **NOT EVALUABLE / VACUOUS** | 0 / 237,800 | 0% |

The default profile's zero false-safe count is not safety evidence: no
replication entered `ACCEPTABLE`, so a false-safe rate cannot be evaluated.
The oracle sensitivity accepted 82,814 / 237,800 replications (34.8251%) and
had zero confirmatory false-safe regions, but that result is not a headline
real-world claim because it relies on unavailable generator truth.

For continuity with the earlier audit, confirmatory false-`INSUFFICIENT`
counts are 23 for v2, 12 for v3 default, and 7 for the oracle sensitivity
profile.

Selected results:

| Cell | Policy/profile | Classification result |
|---|---|---|
| normal n=3 | v3 default | 100% caution, 0% insufficient |
| normal n=3 | v3 oracle sensitivity | 24.40% acceptable, 75.60% caution, 0% insufficient |
| normal n=10,000 | v2 | .88% acceptable, 99.12% caution |
| normal n=10,000 | v3 oracle sensitivity | 95.78% acceptable, 4.22% caution |
| Student-t(df=5), n=20 | v3 oracle sensitivity | 49.82% acceptable, 50.18% caution, 0% insufficient |
| symmetric bimodal n=300 | v3 oracle sensitivity | 100% acceptable; Type I .0494, coverage .9506 |
| lognormal σ=.50, n=50 | v2 acceptable | 25.41%; conditional Type I .1015, coverage .8985 |
| lognormal σ=.50, n=50 | v3 oracle sensitivity | 0% acceptable, 91.69% caution, 8.31% insufficient |
| lognormal σ=1, n=30 | v2 acceptable | 4.60%; conditional Type I .3826, coverage .6174 |
| lognormal σ=1, n=30 | v3 oracle sensitivity | 0% acceptable, 64.65% caution, 35.35% insufficient |
| asymmetric contamination ε=.01, n=100 | v2 acceptable | 33.52%; conditional Type I .1718, coverage .8282 |
| asymmetric contamination ε=.01, n=100 | v3 oracle sensitivity | 0% acceptable, 37.44% caution, 62.56% insufficient |
| symmetric contamination ε=.10, n=100 | v2 | 98.96% insufficient despite total Type I .0480 |
| symmetric contamination ε=.10, n=100 | v3 oracle sensitivity | 99.90% caution, 0% insufficient; five unreliable acceptable rows |

The default profile cannot support a claim that the candidate removes v2's
five false-safe regions because its `ACCEPTABLE` denominator is zero. The
oracle sensitivity suggests how explicit low-risk external context can
preserve calibrated non-Gaussian cases and prevent normal large-n extremeness
from acting as an automatic warning, but it is not evidence about realistic
sample-only deployment.

V3 does **not** solve unidentifiable rare contamination. With default unknown
context, apparently benign but contaminated samples remain `CAUTION`; the
sample cannot prove the process clean. In strong lognormal calibration, the v3
`CAUTION` subset still has poor operating characteristics. This is deliberately
visible rather than mislabeled `ACCEPTABLE`; the injected-v3 selector now
stops automatic inference and returns `REVIEW_REQUIRED` for that state.

## Reproducible artifacts

- `experiments/sampling_robustness_v3_calibration.py`
- `experiments/results/sampling_robustness_v3_comparison.csv`
- `experiments/results/sampling_robustness_v3_flagged_regions.csv`
- `experiments/results/sampling_robustness_v3_false_safe_metrics.csv`
- `experiments/results/sampling_robustness_v3_special_cells.csv`
- `experiments/results/sampling_robustness_v3_metadata.json`

The script rejects any input path other than the existing in-repository
calibration replicate artifact. Metadata records `holdout_used: false`.

## Holdout gate

No default change should be considered until an independent, still-blind
holdout evaluates the frozen candidate. This document and the calibration
results are development evidence, not a universal statistical theorem and not
validation for paired, two-sample, ANOVA, regression, or dependent-data use.
