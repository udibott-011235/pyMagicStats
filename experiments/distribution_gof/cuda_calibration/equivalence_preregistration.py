"""CP05-C2B fixed-data CPU/CUDA equivalence preregistration harness."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isclose, log, sqrt
from typing import Iterable, Mapping

REFERENCE_SHA = "e759137b5f1f10f97746fb241dc2fe6c02ef7061"
REFERENCE_ENGINE = "pyMagicStats CP04 + DEC-014 float64 implementation"
EXPERIMENTAL_ENGINE = "CP05 CUDA/RAPIDS float64 experimental engine"
FLOAT_PRECISION = "float64"
R_EQ, B_EQ, GENERATOR_SANITY_N, PRIMARY_CELL_COUNT = 8, 15, 1_000_000, 144
FIT_ATOL, FIT_RTOL = 5e-12, 5e-10
NB_TRANSFORM_ATOL, NB_OBJECTIVE_RTOL = 1e-8, 1e-9
VALUE_ATOL, VALUE_RTOL, STATISTIC_RTOL = 5e-13, 5e-11, 2e-11
CLASSIFICATIONS = frozenset({"ELIGIBLE", "ALL_ZERO_NON_IDENTIFYING", "VARIANCE_NOT_GREATER_THAN_MEAN", "NOT_ASSESSED", "FAILED", "RETRY_CAP_EXHAUSTED"})
REQUIRED_ARTIFACTS = ("equivalence_manifest.json", "fixture_manifest.json", "fit_comparison.parquet", "statistic_comparison.parquet", "classification_comparison.parquet", "batch_invariance.json", "rng_identity.json", "generator_sanity.json", "environment.json", "summary.json", "digests.json")

@dataclass(frozen=True)
class FixtureCell:
    family: str
    parameters: tuple[tuple[str, float], ...]
    n: int
    statistic: str
    @property
    def canonical_id(self) -> str:
        return f"{self.family}|{','.join(f'{k}={v:g}' for k, v in self.parameters)}|n={self.n}|{self.statistic}|composite"

def primary_fixture_matrix() -> tuple[FixtureCell, ...]:
    cells = []
    for n in (20, 50, 100, 250):
        for statistic in ("AD", "CVM"):
            for shape in (0.25, 0.5, 1.0, 2.0, 10.0):
                cells.append(FixtureCell("gamma", (("shape", shape), ("scale", 1.0)), n, statistic))
            cells.append(FixtureCell("exponential", (("scale", 1.0),), n, statistic))
            for r in (0.25, 1.0, 5.0, 20.0):
                for p in (0.1, 0.5, 0.9):
                    cells.append(FixtureCell("negative_binomial", (("r", r), ("p", p)), n, statistic))
    assert len(cells) == PRIMARY_CELL_COUNT
    return tuple(cells)

ADVERSARIAL_FIXTURES = ("nb_all_zero", "nb_variance_equal_mean", "nb_variance_just_below_mean", "nb_variance_just_above_mean", "nb_very_sparse", "nb_heavy_tail", "gamma_shape_0p25", "very_small_observations", "large_observations", "ad_extreme_tails", "cdf_near_zero", "cdf_near_one", "mc_exact_tie", "mc_near_comparison_cliff")
GENERATOR_SANITY_CASES = (("gamma", {"shape": 0.25, "scale": 1.0}), ("gamma", {"shape": 2.0, "scale": 1.0}), ("gamma", {"shape": 10.0, "scale": 1.0}), ("exponential", {"scale": 1.0}), ("negative_binomial", {"r": 0.25, "p": 0.1}), ("negative_binomial", {"r": 1.0, "p": 0.5}), ("negative_binomial", {"r": 20.0, "p": 0.9}))

def _close(cpu: float, cuda: float, *, atol: float, rtol: float) -> bool:
    return isclose(cpu, cuda, abs_tol=atol, rel_tol=rtol)

def logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("NB p must lie strictly between zero and one")
    return log(p / (1.0 - p))

def categorical_agreement(cpu: str, cuda: str) -> bool:
    if cpu not in CLASSIFICATIONS or cuda not in CLASSIFICATIONS:
        raise ValueError("unknown preregistered classification")
    return cpu == cuda

def nb_objective_tolerance(ll_cpu: float) -> float:
    return NB_OBJECTIVE_RTOL * max(1.0, abs(ll_cpu))

def nb_objective_agreement(ll_cpu: float, ll_cuda: float) -> bool:
    return abs(ll_cuda - ll_cpu) <= nb_objective_tolerance(ll_cpu)

def _nb_eta(parameters: Mapping[str, float]) -> tuple[float, float]:
    return log(parameters["r"]), logit(parameters["p"])

def _from_nb_eta(eta: tuple[float, float]) -> dict[str, float]:
    return {"r": exp(eta[0]), "p": 1.0 / (1.0 + exp(-eta[1]))}

def nb_flat_objective_diagnostic(cpu: Mapping[str, float], cuda: Mapping[str, float], *, reference_log_likelihood) -> dict[str, object]:
    """Conservative nine-point CP04-reference likelihood path diagnostic."""
    eta_cpu, eta_cuda = _nb_eta(cpu), _nb_eta(cuda)
    t_grid = tuple(index / 8.0 for index in range(9))
    ll_path = tuple(
        float(reference_log_likelihood(_from_nb_eta((
            eta_cpu[0] + t * (eta_cuda[0] - eta_cpu[0]),
            eta_cpu[1] + t * (eta_cuda[1] - eta_cpu[1]),
        ))))
        for t in t_grid
    )
    ll_cpu, ll_cuda = ll_path[0], ll_path[-1]
    tau = nb_objective_tolerance(ll_cpu)
    finite = all(value == value and value not in (float("inf"), float("-inf")) for value in ll_path)
    span = max(ll_path) - min(ll_path) if finite else float("inf")
    return {"eta_cpu": eta_cpu, "eta_cuda": eta_cuda, "t_grid": t_grid, "LL_PATH": ll_path, "LL_span": span, "TAU_LL": tau, "FLAT_OBJECTIVE": finite and span <= tau and nb_objective_agreement(ll_cpu, ll_cuda)}

def fit_agreement(family: str, cpu: Mapping[str, float], cuda: Mapping[str, float], *, cpu_log_likelihood: float | None = None, cuda_log_likelihood: float | None = None, reference_log_likelihood=None, same_eligibility: bool = False, downstream_passed: bool = False) -> tuple[bool, str | None, dict[str, object] | None]:
    if family in {"gamma", "exponential"}:
        keys = ("shape", "scale") if family == "gamma" else ("scale",)
        return all(_close(cpu[k], cuda[k], atol=FIT_ATOL, rtol=FIT_RTOL) for k in keys), None, None
    if family != "negative_binomial" or cpu_log_likelihood is None or cuda_log_likelihood is None or reference_log_likelihood is None:
        raise ValueError("NB fit comparison requires r, p, objective values and CP04 reference likelihood")
    objective_ok = nb_objective_agreement(cpu_log_likelihood, cuda_log_likelihood)
    transformed_ok = abs(log(cpu["r"]) - log(cuda["r"])) <= NB_TRANSFORM_ATOL and abs(logit(cpu["p"]) - logit(cuda["p"])) <= NB_TRANSFORM_ATOL
    if transformed_ok and objective_ok:
        return True, None, None
    diagnostic = nb_flat_objective_diagnostic(cpu, cuda, reference_log_likelihood=reference_log_likelihood)
    if same_eligibility and downstream_passed and objective_ok and diagnostic["FLAT_OBJECTIVE"]:
        return True, "PARAMETERIZATION_DIFFERENCE_ON_FLAT_OBJECTIVE", diagnostic
    return False, None, diagnostic

def distribution_value_agreement(cpu: float, cuda: float) -> bool:
    return _close(cpu, cuda, atol=VALUE_ATOL, rtol=VALUE_RTOL)

def statistic_agreement(cpu: float, cuda: float) -> bool:
    return abs(cuda - cpu) <= STATISTIC_RTOL * max(1.0, abs(cpu))

def mc_agreement(cpu_exceedances: int, cuda_exceedances: int, cpu_reject: bool, cuda_reject: bool) -> bool:
    return cpu_exceedances == cuda_exceedances and cpu_reject == cuda_reject

def seed_identity_agreement(cpu_identities: Iterable[int], cuda_identities: Iterable[int]) -> bool:
    return tuple(cpu_identities) == tuple(cuda_identities)

def batch_agreement(cpu: Mapping[str, object], cuda: Mapping[str, object]) -> bool:
    keys = ("outer_indices", "inner_indices", "classifications", "replicate_counts", "retry_accounting")
    return all(cpu.get(key) == cuda.get(key) for key in keys)

def global_equivalence_pass(gate_results: Iterable[bool]) -> bool:
    """The C2B global gate is all-or-nothing; there is no pass percentage."""
    results = tuple(gate_results)
    return bool(results) and all(results)

def initial_summary() -> dict[str, object]:
    return {"equivalence_gate_passed": False, "calibration_claim": False, "reference_sha": REFERENCE_SHA, "float_precision": FLOAT_PRECISION, "R_EQ": R_EQ, "B_EQ": B_EQ, "execution_started": False}

def theoretical_moments(family: str, parameters: Mapping[str, float]) -> tuple[float, float, float]:
    if family == "gamma":
        alpha, theta = parameters["shape"], parameters["scale"]
        variance = alpha * theta ** 2
        return alpha * theta, variance, 3 * alpha * (alpha + 2) * theta ** 4
    if family == "exponential":
        theta = parameters["scale"]
        return theta, theta ** 2, 9 * theta ** 4
    if family == "negative_binomial":
        r, p, q = parameters["r"], parameters["p"], 1.0 - parameters["p"]
        variance = r * q / p ** 2
        excess = (1 + 4 * q + q ** 2) / (r * q)
        return r * q / p, variance, (3 + excess) * variance ** 2
    raise ValueError("unknown generator family")

def generator_sanity_check(sample_mean: float, sample_variance: float, family: str, parameters: Mapping[str, float], *, n: int = GENERATOR_SANITY_N) -> dict[str, float | bool]:
    if n != GENERATOR_SANITY_N:
        raise ValueError("generator sanity must use the frozen N_GENERATOR_SANITY")
    mean, variance, mu4 = theoretical_moments(family, parameters)
    z_mean = (sample_mean - mean) / sqrt(variance / n)
    variance_s2 = (mu4 - ((n - 3) / (n - 1)) * variance ** 2) / n
    z_variance = (sample_variance - variance) / sqrt(variance_s2)
    return {"z_mean": z_mean, "z_variance": z_variance, "passed": abs(z_mean) <= 5 and abs(z_variance) <= 5}
