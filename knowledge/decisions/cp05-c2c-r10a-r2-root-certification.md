# DEC-018 — CP05-C2C R10-A-R2 Negative Binomial root certification

- Estado: `proposed`
- Fecha: 2026-09-22
- Owner: `decision-owner`
- Revisores: `statistical-software-architecture`, `implementation-engineering`, `adversarial-statistical-qa`
- Supersedes: none

## Contexto y estimando

CP05-C2C R10-A introduced a bracketed float64 solver for the experimental
CUDA/RAPIDS Negative Binomial fit. Architecture review identified
`DEFECT-R10A-01` (`MAJOR`, `EVIDENCE / SOFTWARE`) in the historical test
`test_old_false_convergence_is_rejected_by_root_and_objective_contract`: it
required `0.999999999 < old_p < 1.0` and a finite `old_ll`, although the R9
mechanism can legitimately saturate in float64 to `old_p=1.0` and `old_ll=nan`.
This finding questioned the reproducibility of the declared evidence, not the
mathematics of the new solver.

The separate `DEFECT-R10A-02` (`BLOCKER`, `NUMERICAL`) was the R10-A residual
certificate based on fixed `eta +/- 1e-7` probes and
`abs(residual) <= max(abs(probe_left), abs(probe_right)) * 1e-4`. It produced a
false `converged=False` for the frozen `negative_binomial`, `r=5`, `p=0.9`,
`n=250`, `AD`, `raw_outer=2`, namespace `CP05-C2C` cell even though its fit
satisfied DEC-016 with ample margin. Those checks are not derived from the final
bracket and can reject or accept a result for reasons unrelated to the solver's
actual float64 termination state.

This record is scoped to work item `CP05-C2C-R10-A-R2` based on
`c68f0c715c0dbf7fb3c17fce6eb6f0bdc513c130`. It depends on the frozen
equivalence contract in DEC-016. **DEC-016 IS NOT MODIFIED OR SUPERSEDED.**

## Alternativas consideradas

1. Retain the fixed probes and tune their distance or residual factor. Rejected:
   this would replace one empirical tolerance with another.
2. Certify only the objective and fitted parameters. Rejected: that does not
   certify the root solved by the bracketed algorithm.
3. Certify the final bracket, its representable-width termination, and a residual
   envelope derived from its endpoint scores. Selected because every gate follows
   directly from the solver state and float64 ordering.

## Decisión y razón

For each eligible Negative Binomial fit, the final bracket is certified with
`G_low = G(low)` and `G_high = G(high)`. Both scores must be finite and satisfy
`G_low >= 0` and `G_high <= 0`. The bracket must have collapsed exactly or to
adjacent float64 values:

`(low == high) | (nextafter(low, high) == high)`.

The returned `eta` remains the midpoint `(low + high) / 2`. Its residual must be
finite and satisfy:

`abs(G(eta)) <= max(abs(G_low), abs(G_high))`.

`converged=True` requires these three gates in addition to all previously
required eligibility, bracket, finite-fit, objective, and parameter-validity
gates. Any non-finite or failed certification condition remains fail-closed.
The Negative Binomial score, bracket construction and expansion, bisection,
objective, eligibility overflow guard, and returned fit parameters are unchanged.

The historical R9 regression treats `old_p == 1.0` as the expected degenerate
failure mode, and treats a non-finite old objective as independent evidence of
failure rather than attempting an invalid finite improvement comparison.

## Evidencia vinculada

- `knowledge/decisions/cp05-c2b-cuda-equivalence-preregistration.md` (DEC-016)
- `experiments/distribution_gof/cuda_calibration/cuda_candidate.py`
- `tests/research/test_cp05_cuda_nb_r10a.py`
- Exact frozen blocker regression: `negative_binomial`, `r=5`, `p=0.9`,
  `n=250`, `AD`, `raw_outer=2`, namespace `CP05-C2C`
- Complete CPU-only census of observed Negative Binomial samples in the frozen
  primary fixture matrix
- Focused frozen-bootstrap regression for the exact blocker cell with `B=15`

## Límites y consecuencias

This decision changes only root certification for the isolated experimental
Negative Binomial candidate and its CPU-only regression coverage. It does not
change the frozen runner, seed derivation, bootstrap retry, Monte Carlo logic,
artifacts, checkpointing, adversarial fixtures, `R/B`, or the primary matrix.
It makes no CUDA equivalence, statistical calibration, performance, production,
GPU execution, targeted replay, holdout, or full-campaign claim.

## Condición que obliga a revisar

A new architecture decision is required if the score equation, bracket update
invariant, float64 dtype, eligibility definition, DEC-016 tolerances, frozen
fixtures, or solver termination rule changes, or if GPU evidence shows that
`nextafter` or endpoint score evaluation does not preserve the certified
invariant.

## Impacto en API, código, tests y documentación

There is no public API change. The candidate's internal diagnostic mapping gains
`root_precision_check`; `root_sign_check` and `root_residual_check` now describe
the final-bracket certificate. Tests cover representative fits, the exact
blocker, the complete observed-sample census, and its frozen bootstraps. This
proposed record and the knowledge registry document the scope for architecture
review.
