# ANOVA UAT1 boundary and Welch research track

- Fecha: `2026-09-06`
- Stage: `STAGE-ANOVA-001`
- Decision: `UAT1_ANOVA_EXPLICIT_METHODS_CLOSED / SELECTOR_RESEARCH_SEPARATE`

## UAT1 boundary

For UAT1, the ANOVA blocker is considered statistically closed at the level of explicit methods:

```text
OneWayANOVA   explicit API supported within documented assumptions
WelchANOVA    explicit API supported with documented finite-sample limitations
ONE_WAY automatic selector   OUT OF UAT1 / NOT_CALIBRATED
```

UAT1 must not wait for an automatic Classical-vs-Welch routing policy. No production selector capability is authorized by CP-ANOVA-07.

This preserves the original UAT1 boundary: validate explicit statistical methods and their contracts independently of the future decision engine.

## Separate theoretical/statistical research track

The question `when is Welch ANOVA the statistically appropriate procedure?` is opened as a separate research problem. It must not be solved by fitting a post-hoc routing rule to the completed development + holdout calibration evidence.

The research track must begin from statistical mechanics/theory and only then define a prospective empirical validation program.

Required theoretical scope includes at least:

1. Exact estimand and null hypothesis equivalence between Classical and Welch one-way ANOVA.
2. Derivation and approximation underlying Welch's F statistic and denominator degrees of freedom.
3. Finite-sample error mechanisms as functions of `k`, per-group sample sizes and imbalance.
4. Interaction between sample-size ordering and population variance ordering.
5. Behavior under equal variances, where Welch can carry an unnecessary finite-sample approximation cost.
6. Behavior under skewness, heavy tails and contamination; distinguish heteroscedasticity from distributional-shape violations.
7. Competing procedures when Classical/Welch assumptions are not adequate: Brown-Forsythe-type procedures, Alexander-Govern, permutation/randomization tests, bootstrap approaches and robust/trimmed-mean ANOVA where the estimand changes.
8. Why a variance pretest followed by Classical/Welch is not generally a valid two-stage inferential rule.
9. Which observable diagnostics can legitimately support routing without treating noisy sample estimates as population truth.
10. Definition of an abstain/not-authorized region when neither explicit method has sufficient evidence.

## Research methodology

The future selector study must follow this order:

```text
A. theoretical review / method taxonomy
B. frozen decision variables and estimands
C. preregistered routing hypotheses
D. new development calibration data
E. frozen selector candidate
F. new sealed holdout
```

The completed CP-ANOVA-07 holdout may be used as evidence motivating hypotheses, but must not be reused as the validation set for a selector tuned after seeing it.

## Current program status

```text
CP-ANOVA-07                 COMPLETE_WITH_LIMITATIONS
ANOVA explicit-method blocker for UAT1   CLOSED
ANOVA automatic selector                 NOT_CALIBRATED / OUT OF UAT1
Welch theoretical research               OPEN / separate track
```

The next UAT1 work may proceed independently of the Welch research track.
