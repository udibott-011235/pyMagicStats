from __future__ import annotations

import numpy as np
import threading

from pyMagicStat.distributions.families import FitNumericalError

from experiments.distribution_gof.accounting import AssessmentStatus, ReasonCode
from experiments.distribution_gof.generators import GenerationError, generate_sample
from experiments.distribution_gof.runner import (
    RunnerHooks,
    merge_outputs,
    run_manifest,
    run_outer_unit,
)
from experiments.distribution_gof.statistics import (
    OracleMismatchError,
    StatisticEvaluation,
    StatisticError,
    TailCertificationError,
)

from ._helpers import manifest


def _cheap_statistic(sample, bound, family, statistic, *, parameter_count_estimated):
    return StatisticEvaluation(float(np.mean(sample)))


def test_simple_null_fit_count_remains_zero_even_with_fit_bomb():
    def forbidden_fit(*args):
        raise AssertionError("simple-null attempted a fit")

    output = run_manifest(
        manifest(raw_outer_range=(0, 1)),
        hooks=RunnerHooks(fit=forbidden_fit),
    )
    result = output.outer_results[0]
    assert result.status == AssessmentStatus.ASSESSED.value
    assert result.observed_fit_calls == 0
    assert result.replicate_fit_calls == 0
    assert all(item.fit_calls == 0 for item in output.inner_accounting)


def test_composite_observed_once_and_each_eligible_replicate_once_with_cp04():
    output = run_manifest(
        manifest(
            null_type="composite",
            raw_outer_range=(0, 1),
            statistic="KS",
        )
    )
    result = output.outer_results[0]
    assert result.status == AssessmentStatus.ASSESSED.value
    assert result.observed_fit_calls == 1
    assert result.replicate_fit_calls == 199
    assert result.inner_attempts == result.inner_eligible == 199
    assert all(item.fit_calls == 1 for item in output.inner_accounting)
    assert result.observed_fit_provenance["estimation_method"] == "maximum_likelihood"
    assert result.observed_fit_provenance["estimator_id"].endswith("-v1")


def test_order_batch_and_worker_invariance_compare_logical_units():
    baseline_manifest = manifest(raw_outer_range=(0, 5), batch_spec={"batch_size": 1})
    baseline = run_manifest(baseline_manifest)
    reordered = run_manifest(baseline_manifest, execution_order=(4, 2, 0, 3, 1))
    batched = run_manifest(
        manifest(raw_outer_range=(0, 5), batch_spec={"batch_size": 2})
    )
    barrier = threading.Barrier(3, timeout=10)
    local = threading.local()
    thread_ids = set()
    thread_ids_lock = threading.Lock()

    def concurrent_generation(bound, n, rng):
        if not getattr(local, "entered", False):
            local.entered = True
            with thread_ids_lock:
                thread_ids.add(threading.get_ident())
            barrier.wait()
        return generate_sample(bound, n, rng)

    workers = run_manifest(
        manifest(
            raw_outer_range=(0, 5),
            batch_spec={"batch_size": 3},
            worker_spec={"workers": 3},
        ),
        hooks=RunnerHooks(generate=concurrent_generation),
    )
    assert len(thread_ids) == 3
    for candidate in (reordered, batched, workers):
        assert candidate.logical_results() == baseline.logical_results()
        assert [item.to_dict() for item in candidate.inner_accounting] == [
            item.to_dict() for item in baseline.inner_accounting
        ]


def test_sharding_invariance_after_identity_based_merge():
    baseline = run_manifest(manifest(raw_outer_range=(0, 4)))
    shard_zero = run_manifest(
        manifest(
            raw_outer_range=(0, 4),
            shard_spec={"shard_id": 0, "shard_count": 2},
        )
    )
    shard_one = run_manifest(
        manifest(
            raw_outer_range=(0, 4),
            shard_spec={"shard_id": 1, "shard_count": 2},
        )
    )
    merged = merge_outputs((shard_one, shard_zero))
    assert merged.logical_results() == baseline.logical_results()
    assert [item.to_dict() for item in merged.inner_accounting] == [
        item.to_dict() for item in baseline.inner_accounting
    ]


def test_nb_retry_cap_is_not_assessed_and_has_no_boolean_decision():
    eligible = np.array([0, 0, 1, 1, 10], dtype=np.int64)
    ineligible = np.array([0, 1, 1, 1, 1], dtype=np.int64)
    calls = 0

    def never_eligible_inner(bound, n, rng):
        nonlocal calls
        calls += 1
        return eligible.copy() if calls == 1 else ineligible.copy()

    result, attempts = run_outer_unit(
        manifest(
            null_type="composite",
            family="negative_binomial",
            statistic="KS",
            n=5,
            canonical_parameters={"r": "2", "p": "0.5"},
            raw_outer_range=(0, 1),
        ),
        0,
        hooks=RunnerHooks(generate=never_eligible_inner, statistic=_cheap_statistic),
    )
    assert result.status == AssessmentStatus.NOT_ASSESSED.value
    assert result.reason_code == ReasonCode.RETRY_CAP_EXHAUSTED.value
    assert result.p_mc is result.reject is None
    assert result.inner_attempts == 100 * 199
    assert result.inner_ineligible == 100 * 199
    assert len(attempts) == 100 * 199


def test_mathematical_ineligibility_and_failures_are_separate_states():
    all_zero = np.zeros(5, dtype=np.int64)
    ineligible, _ = run_outer_unit(
        manifest(
            null_type="composite",
            family="negative_binomial",
            statistic="KS",
            n=5,
            canonical_parameters={"r": "2", "p": "0.5"},
            raw_outer_range=(0, 1),
        ),
        0,
        hooks=RunnerHooks(generate=lambda *args: all_zero.copy()),
    )
    assert ineligible.status == AssessmentStatus.NOT_ASSESSED.value
    assert ineligible.reason_code == ReasonCode.MATHEMATICAL_INELIGIBILITY.value
    assert ineligible.reject is None

    generation_failure, _ = run_outer_unit(
        manifest(raw_outer_range=(0, 1)),
        0,
        hooks=RunnerHooks(generate=lambda *args: (_ for _ in ()).throw(GenerationError("x"))),
    )
    assert generation_failure.status == AssessmentStatus.FAILED.value
    assert generation_failure.reason_code == ReasonCode.GENERATION_FAILURE.value

    numerical_failure, _ = run_outer_unit(
        manifest(null_type="composite", raw_outer_range=(0, 1)),
        0,
        hooks=RunnerHooks(fit=lambda *args: (_ for _ in ()).throw(FitNumericalError("x"))),
    )
    assert numerical_failure.status == AssessmentStatus.FAILED.value
    assert numerical_failure.reason_code == ReasonCode.FIT_NUMERICAL_FAILURE.value

    statistic_failure, _ = run_outer_unit(
        manifest(raw_outer_range=(0, 1)),
        0,
        hooks=RunnerHooks(
            statistic=lambda *args, **kwargs: (_ for _ in ()).throw(StatisticError("x"))
        ),
    )
    assert statistic_failure.status == AssessmentStatus.FAILED.value
    assert statistic_failure.reason_code == ReasonCode.STATISTIC_FAILURE.value


def test_tail_and_oracle_failures_keep_distinct_reason_codes():
    for exception, reason in (
        (TailCertificationError("x"), ReasonCode.TAIL_CERTIFICATION_FAILURE),
        (OracleMismatchError("x"), ReasonCode.ORACLE_MISMATCH),
    ):
        result, _ = run_outer_unit(
            manifest(raw_outer_range=(0, 1)),
            0,
            hooks=RunnerHooks(
                statistic=lambda *args, _exception=exception, **kwargs: (
                    (_ for _ in ()).throw(_exception)
                )
            ),
        )
        assert result.status == AssessmentStatus.FAILED.value
        assert result.reason_code == reason.value
