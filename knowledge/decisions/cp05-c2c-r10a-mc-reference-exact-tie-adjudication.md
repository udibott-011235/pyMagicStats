# DEC-024 — Canonical CPU reference exact-tie adjudication for CPU/CUDA Monte Carlo equivalence

- Status: `accepted` by explicit Project Owner decision after architecture review and independent QA.
- Date recorded: 2026-09-27.
- Owner: Project Owner — Ehud Bottaro.
- Architecture: ChatGPT; documentary implementation: Cortex.
- Reviewers: ChatGPT architecture and Antigravity independent QA.
- Supersedes: none. This is a prospective, narrowly scoped MC
  equivalence-adjudication decision, not an edit or retroactive replacement of
  accepted DEC-016 or historical DEC-023/R3.
- Repository: `udibott-011235/pyMagicStats`.
- Branch: `docs/cp05-c2c-r10a-mc-reference-exact-tie-adjudication`.

## Acceptance record

The Project Owner supplied the completed architecture/Antigravity review and
explicitly accepted DEC-024. This closure records that authority; Cortex does
not issue its own independent audit. The reviewed candidate was:

```text
REVIEWED_SHA=843d932a8f63282d4dfc1e7f70b4a0a7b7b2b6b2
REVIEWED_TREE=bc885233840ece806c35b7b1738967c3f7dedf96
REMOTE_IDENTITY=EXACT_MATCH
DIFF_SCOPE=EXACT_FOUR_FILES
BLOCKER=NONE
MAJOR=NONE
MINOR=NONE
NOTE=NONE
VERDICT=PASS
READY_FOR_DEC024_ACCEPTANCE=YES
ARCHITECTURE_REVIEW=PASS
ADVERSARIAL_QA=PASS
OWNER_ACCEPTANCE=YES
DEC024_OWNER_DECISION=ACCEPT
```

Acceptance preserves the complete rule below without a tolerance or comparator
change. EV-020 is `validated_with_limits`, not `accepted`: its R3 archive remains
Owner/Quantum-reported. R3 remains FAILED and consumed. This documentary closure
does not authorize R4 implementation or execution, and the recorded review is
bound to the reviewed SHA, not an independent audit of this new closure commit.

## Context, scope and frozen identities

[DEC-016](cp05-c2b-cuda-equivalence-preregistration.md) simultaneously freezes:

```text
STATISTIC_EQUIVALENCE:
abs(T_cuda-T_cpu) <= 2e-11*max(1,abs(T_cpu))

MC:
b_cpu == b_cuda exactly
ties use T* >= T_obs
```

The Owner/Quantum R3 report in
[EV-020](../evidence/cp05-c2c-r10a-r3-mc-comparison-cliff.md) exhibits the comparison
discontinuity: individually equivalent statistics can give different raw
`>=` indicators when the canonical CPU reference has an exact tie.
The question here is software equivalence on the same fixed observed/bootstrap
samples, not Type-I calibration, power, performance or production suitability.
This decision changes only prospective equivalence adjudication; it does not
change the statistical test or either engine's raw results.

```text
BASE_SHA=9282d5b082c4977ee12a7c6e0a54d908ee7283ff
BASE_TREE=f2f55f1e90df24d24a3d113bda8da7be4e24f6c9
BASE_PARENT=bff34ba8062433713177871b5c9e8927a0359ec9
SCIENTIFIC_SHA=d2abd57e65bb7433eff81872a4f6510144d7c267
SCIENTIFIC_TREE=70b20d1f9cbc20d6cd77de4c25b3017a6413932c
HARNESS_SHA=9282d5b082c4977ee12a7c6e0a54d908ee7283ff
```

## Preserved statistical rule

The scientific rule remains exactly:

```text
MC_EXCEEDANCE(T_boot,T_obs) := T_boot >= T_obs
p_MC=(b+1)/(B+1)
B_EQ=15
alpha=0.05
reject := p_MC <= 0.05
```

No fuzzy comparison, rounding, `isclose`, ULP threshold, new MC tolerance or
modification of `STATISTIC_RTOL` is permitted. No reconstruction, seeds,
classification, NB fitting, value grid, certified support, quantities, statistic
calculation, numerical tolerances or workload selection change is proposed.
CPU and CUDA continue to calculate their own raw indicators, counts, p-values
and reject decisions. Adjudication never substitutes a CPU value into a raw
CUDA statistic, count or p-value.

## Exact definition and individual certification

For each frozen bootstrap identity `j`, define:

```text
CPU_REFERENCE_EXACT_TIE(j) :=
    finite(T_cpu_boot[j])
    AND finite(T_cpu_obs)
    AND T_cpu_boot[j] == T_cpu_obs

CPU_EXCEEDANCE(j) :=
    T_cpu_boot[j] >= T_cpu_obs

CUDA_EXCEEDANCE(j) :=
    T_cuda_boot[j] >= T_cuda_obs
```

`CERTIFIED_REFERENCE_EXACT_TIE_CROSSING(j)` exists if and only if **all**
of the following hold; missing or unverified prerequisites do not certify:

1. `CPU_REFERENCE_EXACT_TIE(j)=true`.
2. `CPU_EXCEEDANCE(j) != CUDA_EXCEEDANCE(j)`.
3. Observed CPU/CUDA classification gate PASS.
4. Observed fit gate PASS.
5. Observed distribution-value gate PASS.
6. Observed statistic gate PASS.
7. Bootstrap CPU/CUDA classification gate PASS.
8. Bootstrap fit gate PASS.
9. Bootstrap distribution-value gate PASS.
10. Bootstrap statistic gate PASS.
11. Both CPU/CUDA observed and bootstrap statistics are finite.
12. The frozen bootstrap identity is exactly the expected identity.
13. Sample digest and seed identity match the frozen/reconstructed expected values.
14. No structural failure reason exists.
15. No CUDA non-convergence or failure exists.

The CPU reference determines canonical tie semantics. Its finite exact tie
counts as an exceedance because the preserved comparator is `>=`.
Consequently a certified indicator mismatch has direction CPU `true`,
CUDA `false`. Any mismatch whose direction cannot be explained by such an
exact CPU-reference tie fails; this does not generalize to near-ties.
Each crossing retains identity, sample/seed provenance, four statistics,
raw indicators and all prerequisite gate/failure evidence.

## Non-certifiable near-ties

A crossing cannot be certified when `T_cpu_boot != T_cpu_obs`, even if the
difference is one ULP, a `nextafter` neighbor or below `STATISTIC_RTOL`.
The frozen [adversarial fixtures](../../experiments/distribution_gof/cuda_calibration/cp05_c2b_adversarial_fixtures.json)
and their [evaluator](../../experiments/distribution_gof/cuda_calibration/a2_artifacts.py)
remain unchanged. In particular:

```text
mc_near_comparison_cliff:
below < T_obs
equal == T_obs
above > T_obs
```

Only `equal` is an exact tie. `mc_exact_tie` retains exact `>=` semantics.
This decision cannot waive a gate failure or absorb a non-tie mismatch merely
because individual statistic errors are small.

## Prospective MC equivalence adjudication

Always preserve, without overwriting:

```text
raw_b_cpu
raw_b_cuda
raw_p_cpu
raw_p_cuda
raw_reject_cpu
raw_reject_cuda
```

Additionally calculate from the individual frozen identity comparisons:

```text
indicator_mismatch_count
certified_reference_exact_tie_crossing_count
unexplained_indicator_mismatch_count
```

A raw count mismatch is `EXPLAINED_FOR_EQUIVALENCE` only when all conditions
below hold and every individual mismatch satisfies the full certification
definition above:

```text
raw_b_cpu != raw_b_cuda

indicator_mismatch_count
==
certified_reference_exact_tie_crossing_count

unexplained_indicator_mismatch_count == 0

abs(raw_b_cpu - raw_b_cuda)
==
certified_reference_exact_tie_crossing_count

raw_reject_cpu == raw_reject_cuda
```

A rule such as `abs(b_cpu-b_cuda) <= N` without individual provenance is
forbidden. Opposite-direction mismatches, cancellation hidden by equal raw
counts, missing evidence, unexplained mismatches and unequal reject decisions
fail closed. Count equality alone does not explain an indicator mismatch.
All other frozen scientific gates remain required.

The future gate/evidence surface must distinguish:

```text
MC_RAW_EXCEEDANCE_COUNT_MATCH
MC_REFERENCE_EXACT_TIE_CROSSING_COUNT
MC_UNEXPLAINED_INDICATOR_MISMATCH
MC_REJECT_DECISION_MATCH
MC_EQUIVALENCE_ADJUDICATED
```

Raw agreement and adjudicated agreement are different facts. A raw mismatch
remains visible even if prospectively explained; historical counters and
artifacts are not erased, renamed into agreement or rewritten.

## R3 history and claim boundary

[DEC-023](cp05-c2c-r10a-targeted-gpu-replay-r3-preregistration.md) and the
[R3 harness](../../experiments/distribution_gof/cuda_calibration/targeted_replay_r3/targeted_gpu_replay.py)
remain unchanged. The sole Owner-reported R3 execution remains:

```text
R3_EXECUTION_COUNT=1
R3_EXECUTION_STATUS=FAILED
R3_AUTHORIZATION_CONSUMED=YES
AUTO_RERUN=NO
AUTO_RESUME=NO
```

EV-020 motivates and supplies the reported design case; it is not a Cortex
re-execution, independent archive verification or completed certification under
this future rule. DEC-024 is prospective and must never convert R3 to PASS.

A future targeted PASS under this accepted/audited DEC-024 may claim only:

> CPU/CUDA targeted equivalence passed on the frozen workload,
> with any raw MC count differences limited to certified canonical
> CPU-reference exact-tie crossings and with identical reject decisions.

It cannot claim:

```text
RAW_MC_PVALUE_BIT_IDENTITY
GPU_NATIVE_MC_PVALUE_IDENTITY
FULL_C2C_EQUIVALENCE
TYPE_I_CALIBRATION
POWER
PERFORMANCE
PRODUCTION_READINESS
CP05_D_READINESS
```

Raw CPU/GPU p-value identity remains explicit unresolved debt if standalone GPU
production requires it; this adjudication does not discharge that requirement.

## Alternatives, future work and review trigger

Fuzzy/rounded/ULP comparisons and a widened MC tolerance are rejected because
they would change or blur the statistical comparator. A count-only allowance
is rejected because it hides individual unexplained mismatches. Retrospective
R3 acceptance is rejected because the R3 contract and consumed execution are
immutable. This decision instead exposes raw disagreement and requires
individual canonical-reference exact-tie provenance.

No implementation or execution is authorized by this documentary task.
Acceptance and audit are recorded above; any separately authorized next execution is:

```text
NEXT_EXECUTION_VERSION=R4
READY_FOR_R4_IMPLEMENTATION=NO
```

R4 must be fresh; no resume or reuse of R3 outputs/authorization is permitted.
The frozen scientific candidate may remain
`d2abd57e65bb7433eff81872a4f6510144d7c267` only if future changes are limited
to adjudication/harness and leave engine mathematics unchanged. Future
implementation, candidate review, preregistration and execution authorization
remain separate gates.

Future tests must distinguish exact/equal from below/above neighbors, reject
each failed certification prerequisite, preserve raw evidence and reject
opposite-direction/canceling/unexplained mismatches or reject disagreement.
These are review criteria, not implemented tests or execution evidence here.
Any change to comparator, tolerances, individual certification or claim scope
requires architecture review rather than an implementation assumption.

Current materialization changes only DEC-024, EV-020, the decision index and
registry. Validate strict registry/JSON, links, free IDs before insertion,
the frozen R9 topology, unchanged adversarial fixtures/DEC-016/023/R3 harness,
diff whitespace and the exact four-file allowlist. No scientific tests, GPU,
replay, Quantum, full campaign, CP05-D or holdout access are part of this task.
Next role: ChatGPT architecture on the exact documentary commit; no push,
PR, merge or main modification.
