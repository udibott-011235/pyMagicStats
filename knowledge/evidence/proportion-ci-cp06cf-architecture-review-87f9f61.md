# CP-06 C–F harness architecture review — `87f9f61`

**Stage:** `STAGE-PROP-CI-001`  
**Work item:** `CP06-C-F-HARNESS-CORRECTION-1`  
**Reviewed harness SHA:** `87f9f61e444af634f66f5b8b46496b0d8f8d8d06`  
**Previous harness SHA:** `72e869a1239b3a7b83dd5d16eb20e4454cfa3aa4`  
**Frozen production candidate:** `2df5b90a5395163e723f9c52aafbb91fdce96d43`  
**Review status:** `HOLD — targeted correction required before Quantum heavy execution`

## Scope integrity

GitHub compare confirms `87f9f61...` is exactly one commit ahead of `72e869a...`. The delta remains restricted to the five approved harness/test files and does not modify `pyMagicStat/`.

## Previous HOLD findings now resolved

### Wald nonmonotone coverage — RESOLVED

The harness now checks endpoint monotonicity before using the contiguous `searchsorted` fast path. Nonmonotone endpoints are routed through an explicit endpoint sweep that reconstructs `A(p)` as inclusive integer runs and evaluates probability mass over those runs without forcing `[first,last]`.

Focused tests compare a real nonmonotone Wald case with brute-force enumeration and also demonstrate a synthetic noncontiguous acceptance set where filling the gap would be wrong.

### Dedicated E artifact — RESOLVED in schema and routing

Checkpoint E now writes one adversarial-minimum row per `(method, alpha, n)` with acceptance kind, inclusive run representation, origin, search method and optimizer status.

### Cache provenance — RESOLVED

Shard cache is now namespaced by candidate and harness schema. Payloads include checkpoint spec hash, shard semantic hash, result hash and interval hash. Legacy/incompatible payloads are rejected. Cross-checkpoint B→C reuse requires semantic equality.

### Wald pathology probabilities — RESOLVED

The harness now computes probability-weighted `P_outside` and `P_degenerate`, including worst `p` and reconstructible outcome runs. Endpoint failure counts are no longer used as a substitute for the preregistered probability metrics.

### Endpoint-cache reuse — RESOLVED as an execution architecture

A persistent endpoint cache with candidate/schema/n/alpha/method provenance and SHA-256 validation now prevents C/D/E from rebuilding already validated endpoint grids unnecessarily. The first materialization still evaluates the frozen production API, preserving semantic authority.

## Blocking finding 1 — high-precision endpoint trigger is incomplete

CP-04 requires every Wilson/Jeffreys coverage minimum within `1e-10` of an induced endpoint to enter high precision, independent of the undercoverage tier.

The current trigger logic computes `near_endpoint` only as:

`abs(worst_p - interval.lower[first_worst]) < 1e-10`

This checks only one lower endpoint. It can miss a minimum adjacent to the corresponding upper endpoint, another endpoint responsible for the partition change, or an endpoint represented by a nextafter candidate when the deficit is not severe/critical.

Required correction: determine endpoint proximity against the complete relevant induced endpoint set (or an equivalent exact partition provenance) and test both sides. Severe/critical Wilson must continue to trigger regardless of endpoint proximity.

## Blocking finding 2 — CP06-F does not yet produce an interpretation-governing HP verdict

The approved C–F handoff requires the HP audit to preserve both float64 and arbitrary-precision values and explicitly classify which representation governs interpretation.

Current coverage audit output records float64 coverage, HP coverage, absolute error and HP undercoverage, but sets `status = "resolved"` unconditionally. It does not emit the required interpretive fields such as high-precision probability/endpoint relation, float64 vs HP deficit comparison, classification/verdict, or notes explaining whether a float64 finding is confirmed or resolved as a numerical artifact.

This is especially material for Wilson boundary cases such as the CP06-B apparent `coverage=0` at `p=0`: the purpose of F is not merely to recompute the number at 80 digits but to state whether the apparent shortfall survives the high-precision endpoint semantics.

Required correction: the coverage audit must retain/derive at least:

- `p_float64`
- high-precision endpoint/probability relation sufficient to decide inclusion at a triggered boundary
- `coverage_float64`
- `coverage_hp`
- `deficit_float64`
- `deficit_hp`
- `endpoint_relation`
- `classification` (e.g. confirmed_statistical_shortfall / float64_boundary_artifact / confirmed_exact_coverage / oracle_numerical_difference / unresolved)
- `resolved` boolean
- `notes`

For endpoint/oracle structural rows, provide the analogous classification and do not mark a row resolved solely because an 80-digit value was computed.

High precision must govern interpretation where CP-04 says it does.

## Execution note — D remains computationally expensive but not a semantic blocker

The endpoint cache resolves gratuitous recomputation across checkpoints, but first-time production endpoint materialization remains expensive by design because it evaluates the frozen API for every `x`. Expected-width evaluation at very large stress `n` can also dominate runtime. Before D, benchmark one bounded stress shard on Quantum and record wall/RSS; do not change the statistical grid based on performance without an architecture decision.

## Decision

`87f9f61e444af634f66f5b8b46496b0d8f8d8d06` is **not frozen for full CP06-B→F heavy execution**.

Return to Codex for a targeted harness-only correction of the two HP/trigger findings. Do not modify production. Do not run B/C/D/E/F completely until the next SHA is reviewed.
