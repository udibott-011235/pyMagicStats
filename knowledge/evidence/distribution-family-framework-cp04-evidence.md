# EV-012 — CP04 Wave 1 fitting implementation, adversarial and integration evidence

- **Status:** accepted.
- **Materialization date:** 2026-09-10.
- **Authority:** Owner-authorized post-merge governance; accepted frozen
  [DEC-013](../decisions/distribution-family-framework-cp04-contract.md).
- **Historical baseline:** [EV-011](distribution-family-framework-cp04-baseline.md)
  remains immutable. Its pre-implementation observations are not post-implementation evidence.
- **Stage:** STAGE-DIST-FAMILIES-001, IN_PROGRESS; CP05–CP08 NOT_STARTED.

## Certified identity and handoff

| Artifact | Exact identity |
|---|---|
| Architecture and implementation parent | 9d7a9ea5dcf3f0d962e72a4a0bd2211357e3af52 |
| Certified implementation | 6e92ef20aca375878964321596ba525539433f79 |
| Implementation tree | a3c4e308107cc3042ebcb006973b7a6f38fa1b36 |
| Certified implementation bundle SHA-256 | 28ac5594eb4dd523f17b37ba285a8d49e78776a6a055eac505cc04412b1f0542 |
| Certified implementation bundle byte size | 5990747 |

The candidate contains exactly one implementation commit after the accepted
architecture. Its complete-history bundle was independently verified, including
identity, sole parent, ancestry and strict Git object integrity. This digest
identifies the implementation handoff, not the later governance bundle.

## Review findings and remediation

CP04-A, CP04-B and CP04-C passed final architectural review. Earlier
IMPL-CP04-001 was remediated in d96ca712c28ce3cdb9c205733c606c0c04c6957a:
FitResult rejects false or incomplete provenance, wrong fixed/estimated names,
generic or incorrect wrappers and inconsistent finite AIC/BIC values; canonical
Gamma, Exponential and Negative Binomial results and copy/deepcopy/pickle remain accepted.

The independent CP04-D audit then required two remediations:

- **CP04-D-001 (MAJOR): canonical NB representability.** For the finite sample
  [10**12 - 1 - 10**6, 10**12 - 1 + 10**6], conversion of an internal root
  to public r/p could produce an unsafe canonical result whose likelihood
  was materially worse than a representable competitor. The final implementation
  validates the canonical result using neighboring representable parameters
  and the implied-r competitor, with a likelihood-based floating-point budget.
  It raises FitNumericalError for the unsafe case; it does not return or
  substitute that competitor, cap r/p, or redefine the frozen MLE.
  Safe large-root cases remain accepted. No unsafe NB canonical result is returned.
- **CP04-D-002 (MINOR): warning normalization.** Paths containing spaces and
  UNC paths could escape provenance sanitization. Normalization now covers
  those path forms without exposing filesystem paths in returned warnings.
  Dedicated regression cases cover the corrected warning surface.

The certified remediation is 6e92ef20aca375878964321596ba525539433f79.
Final independent result: **ADVERSARIAL_PASS; CP04_D=ACCEPTED**.
The final implementation added 20 regression cases (2 NB and 18 warning cases)
relative to d96ca712…; this explains 293 → 313 CP04, 868 → 888 distribution
and 1099 → 1119 broader-suite counts. Governance adds no test functions.

## Rehearsal and authoritative integration

| Artifact | Exact identity |
|---|---|
| Synthetic rehearsal SHA (not the real merge) | 40bc595e51709a7f46d5e3755559a7056d2f10fc |
| Rehearsal tree | a3c4e308107cc3042ebcb006973b7a6f38fa1b36 |
| PR | [#12](https://github.com/udibott-011235/pyMagicStats/pull/12) |
| Real merge / post-merge main | 2b6e1263b8489592030b0838cd3851f193fbfd7f |
| First parent | b3f35d4d7b221c457e2e730bfba2b104e1d07144 |
| Second parent | 6e92ef20aca375878964321596ba525539433f79 |
| Merge tree | a3c4e308107cc3042ebcb006973b7a6f38fa1b36 |

The authorized normal merge commit has exactly two parents in the order above.
Independent remote verification confirmed PR #12 closed and merged, main at
the returned merge SHA, certified head ancestry, and exact tree equivalence
to the rehearsal. A fresh checkout passed git fsck --full --strict and
git diff --check. No manual replacement or extra implementation commit was created.

The source branch feature/distribution-family-framework-cp04-wave1-fitting
remains at 6e92ef20aca375878964321596ba525539433f79; it was not deleted.
Ruleset **3811593** remained present and unchanged. Publication had reported
bypassed protected-ref and pull-request requirements; the merge API reported
success without an explicit bypass warning. No protection rule was modified.

## Validation accounting and limits

Recorded final implementation and independent audit evidence:

| Surface | Pre-governance result |
|---|---|
| Registry | PASS |
| CP04 fitting tests | 313 passed |
| Complete distribution surface | 888 passed; 2 inherited warnings |
| Broader non-Knowledge suite | 1119 passed; 3 skipped; 2 inherited warnings |
| Knowledge tests | 7 passed; 2 inherited failures |
| Complete pre-governance accounting | 1126 passed; 2 inherited failures; 3 skipped; 2 warnings |

The two inherited failures were stale governed-branch inventory and canonical-main
SHA expectations in tests/test_knowledge_base.py. Post-merge governance updates
only that test module to BR-001–BR-024 and the exact post-PR-12 main SHA, preserving
all nine tests and unrelated lifecycle states, including BR-018.

The three skipped tests in tests/experiments/test_el_vs_t_harness.py are:

- test_all_canonical_scenarios_gpu_smoke_when_available
- test_gpu_generation_is_reproducible_and_shard_invariant_when_available
- test_gpu_canonical_family_population_moments_when_available

These require an available real CuPy/CUDA backend. The backend was unavailable;
CPU evidence and skipped hardware tests do not validate GPU hardware.
**No GPU hardware validation is claimed in Antigravity's Windows environment.**

The two inherited SciPy RuntimeWarnings, "invalid value encountered in subtract",
come from scipy/stats/_continuous_distns.py:3633 in the frozen Gamma boundary tests:

- tests/test_continuous_distribution_families.py::test_extreme_gamma_backend_nan_raises_explicit_numerical_failure[5e-324-pdf-0.0]
- tests/test_continuous_distribution_families.py::test_extreme_gamma_backend_nan_raises_explicit_numerical_failure[1.7976931348623157e+308-pdf-1.7976931348623157e+308]

Those cases deliberately exercise backend Gamma PDF NaN boundaries and confirm
explicit numerical failure. They are expected inherited warnings, not new fitting
failures or evidence that unsafe numerical output is accepted.

## Governance closure

CP04_IMPLEMENTATION=COMPLETE; CP04_INTEGRATION=COMPLETE;
CP04_GOVERNANCE=CLOSED; CP04_OVERALL=COMPLETE.
BR-001 records canonical main at the real merge. BR-023 is archived /
fully_contained / merged through PR #12 with its remote source preserved.
BR-024 opens docs/distribution-family-framework-cp04-post-merge from that exact
main as under_review / same_head / pending; its registry SHA is the opening
snapshot, not a reference to the future documentation commit.

The governance candidate changes only the six authorized knowledge files and
tests/test_knowledge_base.py. Production and all non-Knowledge tests remain
identical to the post-merge baseline. The stage stays IN_PROGRESS.
CP05–CP08 remain NOT_STARTED.
