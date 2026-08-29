# Inference capability routing

## Status and scope

The candidate routing architecture is versioned independently as
`inference-capability-routing-v1-candidate-2026-08`. It does not replace the
production-default `SamplingRobustness` v2 policy, change
`SamplingRobustnessV3.POLICY_VERSION`, or add a statistical engine.

The routing question is:

> Which inferential guarantee is available for this estimand and design?

It is not:

> Did the observed sample pass a shape or normality test?

## Decision order

Routing follows this order:

1. preserve the caller's estimand;
2. identify the inference design;
3. validate structural and study-design assumptions;
4. resolve registered inference capabilities and their guarantees;
5. allow automatic selection only for a calibrated capability whose required
   assumptions are explicitly supported;
6. otherwise stop for review or report that the domain is not calibrated.

```text
requested estimand
        |
        v
inference design
        |
        v
structural + independence evidence
        |
        v
capability registry -----> candidate, not-calibrated capabilities
        |
        v
available guarantee
        |
        +---- calibrated + automatic allowed ----> selected engine
        |
        +---- otherwise --------------------------> review required
```

The legacy v2 branch preserves its existing behavior exactly. The capability
path is used to make explicitly injected v3 routing auditable without changing
the production default.

## Guarantee versus engine

`InferenceGuarantee` describes why inference is justified. An
`InferenceCapability` separately records how it is computed:

- method identifier;
- estimand;
- design;
- guarantee;
- required assumptions;
- calibration status;
- automatic-selection permission;
- explanatory notes.

The registry currently contains these one-sample arithmetic-mean capabilities:

| Method | Guarantee | Calibrated | Automatic |
|---|---|---:|---:|
| `one_sample_t` | `EXACT_PARAMETRIC` | yes | yes, when requirements match |
| `empirical_likelihood` | `ASYMPTOTIC_MOMENT_BASED` | no | no |
| `bartlett_empirical_likelihood` | `HIGHER_ORDER_CORRECTED` | no | no |
| `bootstrap_t` | `RESAMPLING_BASED` | no | no |

Registration is metadata, not implementation. This change contains no
empirical-likelihood solver, Bartlett factor, generalized empirical likelihood,
or bootstrap-t routing fallback.

## Exact one-sample t capability

For an explicitly injected v3 policy, `one_sample_t` can expose an
`EXACT_PARAMETRIC` guarantee only when all registered requirements are
available:

- structural data support is explicitly `PASS`;
- independence is explicitly `PASS` with study metadata marked `assumed` or
  `verified`;
- the caller supplies external Gaussian-model support.

The current v3 context represents the last item through explicit
`AssumptionProvenance.EXTERNAL`; empirical or unknown provenance is never
promoted to an exact Gaussian guarantee. V3 `CAUTION` still stops automatic
selection with `REVIEW_REQUIRED`, and an out-of-domain result remains
`NOT_CALIBRATED`.

## Evidence is not method identity

`ShapeAssessment`, `OutlierAssessment`, process uncertainty, provenance,
independence, and influence diagnostics remain intact. They feed
`SamplingRobustnessV3` as an evidence layer. The selector does not inspect
skewness, kurtosis, normality p-values, histogram appearance, outlier counts,
or sample-size cutoffs to name a method.

The current v3 evidence policy can still suspend automatic routing through its
`CAUTION` or `INSUFFICIENT` action. That does not make shape evidence a
method registry: method identity and its guarantee now come from the capability
layer. This separation permits later review or replacement of the current
shape machinery without hard-coding a new inference engine into the selector.

Formal normality-test p-values remain descriptive. Failure to reject normality
does not establish a Gaussian model, and rejection does not by itself choose a
different engine. No `MIN_N`, `n >= 30`, or equivalent routing rule exists.

## Estimand preservation and alternatives

Mean inference remains mean inference. Rank procedures such as Wilcoxon or
Mann–Whitney target different location or ordering estimands in general, so
they are never automatic replacements for a requested arithmetic mean.

Empirical likelihood, Bartlett-corrected empirical likelihood, and bootstrap-t
are registered because they may eventually preserve the mean estimand under
different guarantees. They remain candidates because their numerical engines
and routing calibration are outside this change. There is no automatic chain
from t to empirical likelihood, Bartlett correction, bootstrap, or a rank
procedure.

Calling code can inspect `InferenceDecision.capabilities`, `guarantee`,
`assumptions_used`, `estimand`, `design`, `policy_version`, and
`routing_version` while retaining the existing selected method, status,
reasons, alternatives, robustness result, and assumption report.
