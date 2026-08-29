# Empirical likelihood for a one-sample mean

## Status

This module implements Owen-style empirical likelihood for:

- estimand: arithmetic population mean;
- design: one sample;
- sampling contract: iid observations with the moments and regularity needed
  by first-order empirical-likelihood theory.

Its inferential guarantee is
`InferenceGuarantee.ASYMPTOTIC_MOMENT_BASED`. The p-value and confidence
interval use the first-order Wilks chi-square approximation with one degree of
freedom. They are not finite-sample exact.

**EMPIRICAL LIKELIHOOD IS NOT YET CALIBRATED AS AN AUTOMATIC PYMAGICSTATS
ROUTING METHOD.**

The capability registry therefore remains:

- `calibrated=False`;
- `automatic_selection_allowed=False`.

No `MethodSelector` fallback to empirical likelihood is implemented.

## Empirical probabilities and moment constraint

For observations `x[1], ..., x[n]` and a candidate mean `mu`, empirical
likelihood maximizes the product of probabilities subject to:

```text
p[i] >= 0
sum(p[i]) = 1
sum(p[i] * (x[i] - mu)) = 0
```

For an interior feasible mean, the probabilities have the form:

```text
p[i] = 1 / (n * (1 + lambda * (x[i] - mu)))
```

where `lambda` solves:

```text
sum((x[i] - mu) / (1 + lambda * (x[i] - mu))) = 0
```

and every denominator is strictly positive.

The profile statistic is:

```text
-2 log R(mu) = 2 * sum(log1p(lambda * (x[i] - mu)))
```

At the sample mean the implementation returns `lambda=0`,
`-2 log R=0`, and `p_value=1` up to floating-point tolerance.

## Convex-hull geometry

The candidate mean must lie in the closed interval
`[min(data), max(data)]`.

- Outside that interval, the result is explicitly `feasible=False`, with no
  ordinary chi-square p-value.
- At a nonconstant sample boundary, moment weights collapse, finite
  `lambda` diverges, and the result is marked feasible but nonregular with an
  infinite profile statistic.
- A constant sample supports only its constant value. It has zero variance, so
  the regular chi-square approximation and profile interval are not reported.
- A one-observation sample is handled as a nonregular degenerate support.

The confidence interval is always profiled inside the convex hull. The module
does not extrapolate beyond empirical support.

## Numerical algorithm

The implementation:

1. rejects empty, multidimensional, NaN, or infinite input without flattening;
2. copies the sample so evaluation cannot mutate caller data;
3. scales candidate-centered observations before solving;
4. constructs the open admissible `lambda` interval from positive and
   negative centered observations;
5. uses the bracketed monotone solver `scipy.optimize.brentq`;
6. evaluates the statistic with `numpy.log1p`;
7. clips a negative statistic only when its magnitude is at most
   `1e-10`;
8. records a dimensionless lambda-equation residual with declared tolerance
   `1e-10`.

The confidence interval uses the `chi2(df=1)` critical value and independently
profiles roots below and above the sample mean. Each endpoint must reproduce
the critical statistic within `1e-8`; otherwise interval construction returns
an explicit failure result.

Public functions:

```python
from pyMagicStat.inference import (
    empirical_likelihood_mean_ci,
    empirical_likelihood_mean_test,
)
```

## Deterministic example

The following values were reproduced directly by the checked-in
implementation:

```python
import numpy as np

from pyMagicStat.inference import (
    empirical_likelihood_mean_ci,
    empirical_likelihood_mean_test,
)

data = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
result = empirical_likelihood_mean_test(data, mu=3.0)
interval = empirical_likelihood_mean_ci(data, confidence_level=0.95)
```

| Quantity | Value |
|---|---:|
| sample mean | 3.8 |
| candidate mean | 3.0 |
| lambda | 0.16802115052603 |
| lambda residual | 2.220446049250313e-16 |
| `-2 log R(3.0)` | 0.6120993671480546 |
| asymptotic p-value | 0.43399845454391495 |
| 95% profile CI | [2.0647311844578904, 6.122212094661059] |

Both interval endpoints reproduce the chi-square critical value
`3.841458820694124` within floating-point tolerance.

## Regularity and robustness limitations

Empirical likelihood avoids specifying a parametric distribution for the
observations, but this does not make arithmetic-mean inference contamination
robust. A large observation can materially alter the mean, the moment
constraint, and the interval. Finite required moments, iid sampling, interior
convex-hull geometry, and empirical-likelihood regularity remain substantive
assumptions.

The implementation does not claim:

- Gaussian robustness;
- universal heavy-tail validity;
- validity without finite required moments;
- superiority to Student t or bootstrap-t;
- generalized empirical likelihood;
- finite-sample exactness.

Bartlett correction is future work. There is no theoretical, estimated,
Edgeworth, or bootstrap-derived Bartlett factor in this module.
