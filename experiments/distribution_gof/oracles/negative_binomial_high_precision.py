"""Independent Decimal recurrence oracle for NB infinite-support statistics.

This path does not call the float64 statistic, SciPy probability methods, or
the production dtype with higher precision.  It evaluates the NB PMF via its
own arbitrary-precision recurrence and sums the defining support formula.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Iterable


ORACLE_PRECISION = 100
ORACLE_RELATIVE_REMAINDER = Decimal("1e-30")
ORACLE_MAX_TERMS = 2_000_000


def _decimal_parameter(value) -> Decimal:
    if isinstance(value, float):
        return Decimal.from_float(value)
    return Decimal(str(value))


def high_precision_nb_statistic(
    sample: Iterable[int],
    *,
    r,
    p,
    statistic: str,
    max_terms: int = ORACLE_MAX_TERMS,
) -> tuple[float, float]:
    """Return an independent high-precision support sum and remainder bound."""

    observations = tuple(sorted(int(value) for value in sample))
    if not observations or observations[0] < 0:
        raise ValueError("sample must contain non-negative integers")
    if any(original != canonical for original, canonical in zip(sorted(sample), observations)):
        raise ValueError("sample must contain exact integers")
    if statistic not in {"AD", "CVM"}:
        raise ValueError("oracle supports only AD and CVM")
    rd, pd = _decimal_parameter(r), _decimal_parameter(p)
    if rd <= 0 or not 0 < pd < 1:
        raise ValueError("oracle requires r > 0 and 0 < p < 1")

    with localcontext() as context:
        context.prec = ORACLE_PRECISION
        one = Decimal(1)
        n = Decimal(len(observations))
        pmf = (rd * pd.ln()).exp()
        cdf = pmf
        partial = Decimal(0)
        position = 0
        maximum = observations[-1]
        for j in range(max_terms):
            while position < len(observations) and observations[position] <= j:
                position += 1
            q = one - cdf
            z = Decimal(position) - n * cdf
            if statistic == "CVM":
                partial += z * z * pmf / n
                remainder = n * q * q * q
            else:
                if cdf <= 0 or q <= 0:
                    if j < maximum:
                        raise ArithmeticError("oracle AD denominator vanished before sample maximum")
                    remainder = Decimal(0)
                else:
                    partial += z * z * pmf / (n * cdf * q)
                    remainder = n * q * q / cdf

            threshold = ORACLE_RELATIVE_REMAINDER * max(one, abs(partial))
            if j >= maximum and remainder <= threshold:
                return float(partial), float(remainder)

            next_j = Decimal(j + 1)
            pmf = pmf * (Decimal(j) + rd) / next_j * (one - pd)
            cdf += pmf
        raise ArithmeticError("high-precision oracle term budget exhausted")
