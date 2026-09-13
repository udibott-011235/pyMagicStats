"""Deterministic simple/composite-null CP05-B execution machinery."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Callable, Iterable

from pyMagicStat.distributions.families import (
    FitIdentifiabilityError,
    FitNumericalError,
    NoFiniteMLEError,
)

from .accounting import AssessmentStatus, InnerAttempt, OuterResult, ReasonCode
from .bootstrap import monte_carlo_p_value
from .generators import GenerationError, bind_family, family_for, generate_sample
from .manifest import ExperimentManifest, OUTPUT_SCHEMA_VERSION
from .seed_derivation import derive_seed, numpy_rng
from .statistics import (
    ComparatorNotAssessable,
    OracleMismatchError,
    StatisticError,
    TailCertificationError,
    evaluate_statistic,
)


MAX_ATTEMPT_MULTIPLIER = 100


@dataclass(frozen=True, slots=True)
class RunnerHooks:
    generate: Callable[..., Any] = generate_sample
    fit: Callable[..., Any] | None = None
    statistic: Callable[..., Any] = evaluate_statistic


@dataclass(frozen=True, slots=True)
class RunOutput:
    outer_results: tuple[OuterResult, ...]
    inner_accounting: tuple[InnerAttempt, ...]

    def logical_results(self) -> dict[int, dict[str, Any]]:
        return {
            result.raw_outer_index: result.to_dict()
            for result in sorted(self.outer_results, key=lambda item: item.raw_outer_index)
        }


def _fit_provenance(fit_result) -> dict[str, Any]:
    parameters = asdict(fit_result.fitted_distribution.parameters)
    return {
        "backend": fit_result.backend,
        "backend_version": fit_result.backend_version,
        "estimation_method": fit_result.estimation_method,
        "estimator_id": fit_result.metadata["estimator_id"],
        "solver_id": fit_result.metadata["solver_id"],
        "warnings": list(fit_result.warnings),
        "parameters": parameters,
    }


def _failure_reason(exc: BaseException) -> ReasonCode:
    if isinstance(exc, TailCertificationError):
        return ReasonCode.TAIL_CERTIFICATION_FAILURE
    if isinstance(exc, OracleMismatchError):
        return ReasonCode.ORACLE_MISMATCH
    if isinstance(exc, ComparatorNotAssessable):
        return ReasonCode.COMPARATOR_NOT_ASSESSABLE
    if isinstance(exc, FitNumericalError):
        return (
            ReasonCode.FIT_BACKEND_FAILURE
            if exc.__cause__ is not None
            else ReasonCode.FIT_NUMERICAL_FAILURE
        )
    if isinstance(exc, (FitIdentifiabilityError, NoFiniteMLEError)):
        return ReasonCode.MATHEMATICAL_INELIGIBILITY
    if isinstance(exc, GenerationError):
        return ReasonCode.GENERATION_FAILURE
    return ReasonCode.STATISTIC_FAILURE


def _result_shell(manifest: ExperimentManifest, raw_outer_index: int, seed_hex: str) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "canonical_cell_id": manifest.canonical_cell_id,
        "raw_outer_index": raw_outer_index,
        "eligible_outer_index": None,
        "seed_identity": seed_hex,
        "status": AssessmentStatus.FAILED.value,
        "reason_code": ReasonCode.STATISTIC_FAILURE.value,
        "observed_fit_status": "NOT_RUN",
        "statistic_value": None,
        "T_obs": None,
        "exceedance_count": None,
        "p_mc": None,
        "reject": None,
        "inner_attempts": 0,
        "inner_eligible": 0,
        "inner_ineligible": 0,
        "failure_class": None,
        "observed_mle_eligible": None,
        "outer_ineligibility_reason": None,
        "inner_ineligibility_reason_counts": {},
        "tail_remainder_bound": None,
        "oracle_value": None,
        "oracle_abs_error": None,
        "oracle_relative_scaled_error": None,
        "observed_fit_provenance": None,
        "observed_fit_calls": 0,
        "replicate_fit_calls": 0,
        "raw_inner_indices": [],
    }


def _terminal_failure(shell: dict[str, Any], exc: BaseException) -> OuterResult:
    reason = _failure_reason(exc)
    shell.update(
        status=(
            AssessmentStatus.NOT_ASSESSED.value
            if reason is ReasonCode.COMPARATOR_NOT_ASSESSABLE
            else AssessmentStatus.FAILED.value
        ),
        reason_code=reason.value,
        failure_class=type(exc).__name__,
    )
    return OuterResult(**shell)


def _call_fit(family, sample, hooks: RunnerHooks):
    return hooks.fit(family, sample) if hooks.fit is not None else family.fit(sample)


def run_outer_unit(
    manifest: ExperimentManifest,
    raw_outer_index: int,
    *,
    hooks: RunnerHooks | None = None,
) -> tuple[OuterResult, tuple[InnerAttempt, ...]]:
    """Run one raw outer unit without assigning topology-dependent labels."""

    hooks = hooks or RunnerHooks()
    outer_seed = derive_seed(
        manifest.canonical_cell_id, raw_outer_index, "outer_observed"
    )
    shell = _result_shell(manifest, raw_outer_index, outer_seed.digest_hex)
    family = family_for(manifest.family)
    generating_bound = bind_family(manifest.family, manifest.canonical_parameters)
    try:
        observed = hooks.generate(generating_bound, manifest.n, numpy_rng(outer_seed))
    except (GenerationError, ValueError, OverflowError, FloatingPointError) as exc:
        wrapped = exc if isinstance(exc, GenerationError) else GenerationError(str(exc))
        return _terminal_failure(shell, wrapped), ()

    if manifest.null_type == "simple":
        observed_bound = generating_bound
        shell.update(
            observed_fit_status="NOT_REQUESTED_SIMPLE_NULL",
            observed_mle_eligible=True,
        )
        estimated_count = 0
    else:
        shell["observed_fit_calls"] = 1
        try:
            fit_result = _call_fit(family, observed, hooks)
        except (FitIdentifiabilityError, NoFiniteMLEError) as exc:
            if manifest.family == "negative_binomial":
                shell.update(
                    status=AssessmentStatus.NOT_ASSESSED.value,
                    reason_code=ReasonCode.MATHEMATICAL_INELIGIBILITY.value,
                    observed_fit_status=type(exc).__name__,
                    observed_mle_eligible=False,
                    outer_ineligibility_reason=type(exc).__name__,
                )
                return OuterResult(**shell), ()
            return _terminal_failure(shell, exc), ()
        except FitNumericalError as exc:
            shell.update(observed_fit_status=type(exc).__name__, observed_mle_eligible=True)
            return _terminal_failure(shell, exc), ()
        observed_bound = fit_result.fitted_distribution
        shell.update(
            observed_fit_status="SUCCESS",
            observed_mle_eligible=True,
            observed_fit_provenance=_fit_provenance(fit_result),
        )
        estimated_count = len(fit_result.estimated_parameters)

    try:
        observed_evaluation = hooks.statistic(
            observed,
            observed_bound,
            manifest.family,
            manifest.statistic,
            parameter_count_estimated=estimated_count,
        )
    except (StatisticError, ValueError, OverflowError, FloatingPointError) as exc:
        return _terminal_failure(shell, exc), ()
    shell.update(
        T_obs=observed_evaluation.value,
        statistic_value=observed_evaluation.value,
        tail_remainder_bound=observed_evaluation.remainder_bound,
        oracle_value=observed_evaluation.oracle_value,
        oracle_abs_error=observed_evaluation.absolute_error,
        oracle_relative_scaled_error=observed_evaluation.relative_scaled_error,
    )

    inner_rows: list[InnerAttempt] = []
    replicate_statistics: list[float] = []
    ineligibility = Counter()
    raw_inner_index = 0
    max_attempts = manifest.B * (
        MAX_ATTEMPT_MULTIPLIER
        if manifest.null_type == "composite" and manifest.family == "negative_binomial"
        else 1
    )
    while len(replicate_statistics) < manifest.B and raw_inner_index < max_attempts:
        inner_seed = derive_seed(
            manifest.canonical_cell_id,
            raw_outer_index,
            "inner_bootstrap",
            raw_inner_index,
        )
        shell["raw_inner_indices"].append(raw_inner_index)
        try:
            replicate = hooks.generate(
                observed_bound, manifest.n, numpy_rng(inner_seed)
            )
        except (GenerationError, ValueError, OverflowError, FloatingPointError) as exc:
            wrapped = exc if isinstance(exc, GenerationError) else GenerationError(str(exc))
            shell.update(inner_attempts=raw_inner_index + 1)
            inner_rows.append(
                InnerAttempt(
                    manifest.canonical_cell_id,
                    raw_outer_index,
                    raw_inner_index,
                    inner_seed.digest_hex,
                    False,
                    ReasonCode.GENERATION_FAILURE.value,
                    0,
                    None,
                )
            )
            return _terminal_failure(shell, wrapped), tuple(inner_rows)

        replicate_bound = observed_bound
        replicate_estimated_count = 0
        fit_calls = 0
        if manifest.null_type == "composite":
            fit_calls = 1
            shell["replicate_fit_calls"] += 1
            try:
                replicate_fit = _call_fit(family, replicate, hooks)
            except (FitIdentifiabilityError, NoFiniteMLEError) as exc:
                if manifest.family != "negative_binomial":
                    shell.update(inner_attempts=raw_inner_index + 1)
                    inner_rows.append(
                        InnerAttempt(
                            manifest.canonical_cell_id,
                            raw_outer_index,
                            raw_inner_index,
                            inner_seed.digest_hex,
                            False,
                            ReasonCode.MATHEMATICAL_INELIGIBILITY.value,
                            fit_calls,
                            None,
                        )
                    )
                    return _terminal_failure(shell, exc), tuple(inner_rows)
                reason_name = type(exc).__name__
                ineligibility[reason_name] += 1
                inner_rows.append(
                    InnerAttempt(
                        manifest.canonical_cell_id,
                        raw_outer_index,
                        raw_inner_index,
                        inner_seed.digest_hex,
                        False,
                        ReasonCode.MATHEMATICAL_INELIGIBILITY.value,
                        fit_calls,
                        None,
                    )
                )
                raw_inner_index += 1
                continue
            except FitNumericalError as exc:
                shell.update(inner_attempts=raw_inner_index + 1)
                inner_rows.append(
                    InnerAttempt(
                        manifest.canonical_cell_id,
                        raw_outer_index,
                        raw_inner_index,
                        inner_seed.digest_hex,
                        False,
                        _failure_reason(exc).value,
                        fit_calls,
                        None,
                    )
                )
                return _terminal_failure(shell, exc), tuple(inner_rows)
            replicate_bound = replicate_fit.fitted_distribution
            replicate_estimated_count = len(replicate_fit.estimated_parameters)
        try:
            evaluation = hooks.statistic(
                replicate,
                replicate_bound,
                manifest.family,
                manifest.statistic,
                parameter_count_estimated=replicate_estimated_count,
            )
        except (StatisticError, ValueError, OverflowError, FloatingPointError) as exc:
            shell.update(inner_attempts=raw_inner_index + 1)
            inner_rows.append(
                InnerAttempt(
                    manifest.canonical_cell_id,
                    raw_outer_index,
                    raw_inner_index,
                    inner_seed.digest_hex,
                    False,
                    _failure_reason(exc).value,
                    fit_calls,
                    None,
                )
            )
            return _terminal_failure(shell, exc), tuple(inner_rows)
        replicate_statistics.append(evaluation.value)
        inner_rows.append(
            InnerAttempt(
                manifest.canonical_cell_id,
                raw_outer_index,
                raw_inner_index,
                inner_seed.digest_hex,
                True,
                ReasonCode.ASSESSMENT_COMPLETE.value,
                fit_calls,
                evaluation.value,
            )
        )
        raw_inner_index += 1

    shell.update(
        inner_attempts=raw_inner_index,
        inner_eligible=len(replicate_statistics),
        inner_ineligible=sum(ineligibility.values()),
        inner_ineligibility_reason_counts=dict(sorted(ineligibility.items())),
    )
    if len(replicate_statistics) != manifest.B:
        shell.update(
            status=AssessmentStatus.NOT_ASSESSED.value,
            reason_code=ReasonCode.RETRY_CAP_EXHAUSTED.value,
            failure_class=None,
            p_mc=None,
            reject=None,
        )
        return OuterResult(**shell), tuple(inner_rows)

    exceedances, p_value = monte_carlo_p_value(
        observed_evaluation.value, replicate_statistics, B=manifest.B
    )
    shell.update(
        status=AssessmentStatus.ASSESSED.value,
        reason_code=ReasonCode.ASSESSMENT_COMPLETE.value,
        exceedance_count=exceedances,
        p_mc=p_value,
        reject=p_value <= manifest.alpha,
        failure_class=None,
    )
    return OuterResult(**shell), tuple(inner_rows)


def owned_raw_indices(manifest: ExperimentManifest) -> tuple[int, ...]:
    start, stop = manifest.raw_outer_range
    shard_id = manifest.shard_spec["shard_id"]
    shard_count = manifest.shard_spec["shard_count"]
    return tuple(index for index in range(start, stop) if index % shard_count == shard_id)


def _assign_eligible_indices(results: Iterable[OuterResult]) -> tuple[OuterResult, ...]:
    ordered = sorted(results, key=lambda item: item.raw_outer_index)
    eligible_index = 0
    assigned = []
    for result in ordered:
        if result.observed_mle_eligible is True:
            assigned.append(replace(result, eligible_outer_index=eligible_index))
            eligible_index += 1
        else:
            assigned.append(replace(result, eligible_outer_index=None))
    return tuple(assigned)


def run_manifest(
    manifest: ExperimentManifest,
    *,
    execution_order: Iterable[int] | None = None,
    hooks: RunnerHooks | None = None,
) -> RunOutput:
    owned = owned_raw_indices(manifest)
    order = tuple(owned if execution_order is None else execution_order)
    if len(order) != len(set(order)) or set(order) != set(owned):
        raise ValueError("execution_order must contain every owned raw index exactly once")
    outer: list[OuterResult] = []
    inner: list[InnerAttempt] = []
    batch_size = manifest.batch_spec["batch_size"]
    workers = manifest.worker_spec["workers"]

    def execute(raw_outer_index: int):
        return run_outer_unit(manifest, raw_outer_index, hooks=hooks)

    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for offset in range(0, len(order), batch_size):
            batch = order[offset : offset + batch_size]
            completed = map(execute, batch) if executor is None else executor.map(execute, batch)
            for result, attempts in completed:
                outer.append(result)
                inner.extend(attempts)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return RunOutput(
        outer_results=_assign_eligible_indices(outer),
        inner_accounting=tuple(
            sorted(inner, key=lambda item: (item.raw_outer_index, item.raw_inner_index))
        ),
    )


def merge_outputs(outputs: Iterable[RunOutput]) -> RunOutput:
    """Merge independently written shards by logical identity, not file order."""

    outer: list[OuterResult] = []
    inner: list[InnerAttempt] = []
    for output in outputs:
        outer.extend(output.outer_results)
        inner.extend(output.inner_accounting)
    keys = [(item.canonical_cell_id, item.raw_outer_index) for item in outer]
    if len(keys) != len(set(keys)):
        raise ValueError("shard outputs contain duplicate raw outer units")
    return RunOutput(
        outer_results=_assign_eligible_indices(outer),
        inner_accounting=tuple(
            sorted(inner, key=lambda item: (item.raw_outer_index, item.raw_inner_index))
        ),
    )
