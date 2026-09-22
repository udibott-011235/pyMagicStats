"""CPU-only structural/oracle tests for the R10-A CUDA NB mathematics.

These tests exercise the backend-generic numerical kernel with NumPy/SciPy.
They are not evidence of real CUDA execution or C2C equivalence.
"""
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest
from scipy import special

from experiments.distribution_gof.cuda_calibration import cuda_candidate
from experiments.distribution_gof.cuda_calibration import cp05_c2c_equivalence_runner as runner
from experiments.distribution_gof.cuda_calibration.cp05_cuda_engine import _cp04_fit, _parameters, nb_eligibility


ESCAPE_TO_P1 = np.asarray(
    [0, 0, 0, 2, 2, 0, 0, 2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    dtype=np.int64,
)
FALSE_CONVERGENCE = np.asarray(
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    dtype=np.int64,
)
R025_LIKE = np.asarray(
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    dtype=np.int64,
)


def _solve(sample, **kwargs):
    return cuda_candidate._fit_negative_binomial_impl(
        np.asarray(sample, dtype=np.int64), np, special, **kwargs
    )


def _result_at(result, index):
    item={}
    for key,value in result.items():
        array=np.asarray(value)
        item[key]=value if array.ndim==0 else value[index]
    return item


def _assert_root_certification(result, context=""):
    assert bool(result["bracket_found"]), context
    assert bool(result["root_inside_bracket"]), context
    assert bool(result["root_sign_check"]), context
    assert bool(result["root_precision_check"]), context
    assert bool(result["root_residual_check"]), context
    assert bool(result["objective_valid"]), context
    assert bool(result["converged"]), context


def _assert_dec016_fit(sample, result, context=""):
    reference=_cp04_fit("negative_binomial",sample)
    parameters=_parameters(reference.fitted_distribution)
    candidate_r=float(result["r"]); candidate_p=float(result["p"])
    candidate_ll=float(result["log_likelihood"])
    assert math.isfinite(candidate_r) and candidate_r>0, context
    assert math.isfinite(candidate_p) and 0<candidate_p<1, context
    assert math.isfinite(candidate_ll), context
    assert abs(math.log(candidate_r)-math.log(parameters["r"]))<=1e-8, context
    logit=lambda p: math.log(p/(1-p))
    assert abs(logit(candidate_p)-logit(parameters["p"]))<=1e-8, context
    assert abs(candidate_ll-reference.log_likelihood)<=1e-9*max(
        1.0,abs(reference.log_likelihood)
    ), context


def _primary_nb_cell(*, r, p, n, statistic):
    return next(
        cell for cell in runner.primary_fixture_matrix()
        if cell.family=="negative_binomial"
        and dict(cell.parameters)=={"r":float(r),"p":float(p)}
        and cell.n==n and cell.statistic==statistic
    )


def _old_unbracketed_newton(sample):
    """Exact R9 mechanism, retained only to prove the regression fixtures."""
    values=np.asarray(sample,dtype=np.float64)
    mean=np.mean(values); variance=np.var(values,ddof=0)
    r=max(mean*mean/max(variance-mean,1e-300),1e-10)
    with np.errstate(all="ignore"):
        for _ in range(128):
            p=r/(r+mean)
            score=np.sum(special.digamma(values+r)-special.digamma(r))+values.size*np.log(p)
            derivative=(np.sum(special.polygamma(1,values+r)-special.polygamma(1,r))
                        +values.size*(1/r-1/(r+mean)))
            proposal=r-score/derivative
            r=proposal if proposal>0 and np.isfinite(proposal) else r/2
        p=r/(r+mean)
        ll=np.sum(special.gammaln(values+r)-special.gammaln(r)-special.gammaln(values+1)
                  +r*np.log(p)+values*np.log1p(-p))
    return float(r),float(p),float(ll)


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ([0, 0, 0, 0], "ALL_ZERO_NON_IDENTIFYING"),
        ([0, 2], "VARIANCE_NOT_GREATER_THAN_MEAN"),  # D == 0
        ([0, 1], "VARIANCE_NOT_GREATER_THAN_MEAN"),  # D == -1
        ([0, 3], "ELIGIBLE"),                         # smallest point above this boundary
    ],
)
def test_canonical_integer_eligibility_boundaries(sample, expected):
    code=cuda_candidate._nb_classification_codes(np.asarray(sample,dtype=np.int64),np)
    assert cuda_candidate._nb_classification_metadata(code)==expected
    eligible,reason=nb_eligibility(np.asarray(sample,dtype=np.int64))
    assert ("ELIGIBLE" if eligible else reason)==expected


def test_r9_exact_categorical_boundary_regression_uses_canonical_predicate():
    cell=next(
        item for item in runner.primary_fixture_matrix()
        if item.family=="negative_binomial" and dict(item.parameters)=={"r":1.0,"p":0.9}
        and item.n==50 and item.statistic=="CVM"
    )
    sample,_=runner.fixed_observed(cell,6,"CP05-C2C")
    code=cuda_candidate._nb_classification_codes(sample,np)
    assert runner._cpu_nb_classification(sample)=="VARIANCE_NOT_GREATER_THAN_MEAN"
    assert cuda_candidate._nb_classification_metadata(code)=="VARIANCE_NOT_GREATER_THAN_MEAN"


def test_canonical_classification_is_batched_and_matches_cpu():
    samples=np.asarray([[0,0,0,0],[0,2,0,2],[0,3,0,3]],dtype=np.int64)
    labels=cuda_candidate._nb_classification_metadata(
        cuda_candidate._nb_classification_codes(samples,np)
    )
    expected=["ELIGIBLE" if nb_eligibility(row)[0] else nb_eligibility(row)[1] for row in samples]
    assert labels==expected


def test_canonical_classification_fails_closed_before_int64_wraparound():
    # n=2 permits at most floor(sqrt(INT64_MAX))/2 under the documented
    # sufficient bound. At that limit the saturated RHS comparison stays exact.
    safe=math.isqrt((1 << 63)-1)//2
    code=cuda_candidate._nb_classification_codes(np.asarray([safe,safe],dtype=np.int64),np)
    assert cuda_candidate._nb_classification_metadata(code)=="VARIANCE_NOT_GREATER_THAN_MEAN"
    # This remains a valid int64 input but is outside the certified envelope.
    with pytest.raises(cuda_candidate.CudaCandidateError,match="overflow risk"):
        cuda_candidate._nb_classification_codes(
            np.asarray([0,2_000_000_000],dtype=np.int64),np
        )


def test_small_z_cancellation_helper_matches_decimal_oracle():
    for text in ("1e-4","1e-8","1e-12"):
        with localcontext() as context:
            context.prec=80
            z=Decimal(text)
            expected=float(z-(Decimal(1)+z).ln())
        observed=float(cuda_candidate._nb_z_minus_log1p(np.asarray(float(z)),np))
        assert observed==pytest.approx(expected,rel=2e-15,abs=0.0)


@pytest.mark.parametrize(
    "sample",
    [
        np.asarray([0,3],dtype=np.int64),
        ESCAPE_TO_P1,
        FALSE_CONVERGENCE,
        R025_LIKE,
        np.random.default_rng(8128).negative_binomial(2.0,0.5,size=50),
        np.random.default_rng(2718).negative_binomial(50.0,0.5,size=250),
    ],
)
def test_bracketed_solver_matches_cp04_parameters_and_objective(sample):
    assert nb_eligibility(sample)[0]
    result=_solve(sample)
    _assert_root_certification(result)
    _assert_dec016_fit(sample,result)


def test_final_bracket_certifies_root_at_float64_precision():
    result=_solve(FALSE_CONVERGENCE)
    r=float(result["r"]); mean=float(np.mean(FALSE_CONVERGENCE))
    _assert_root_certification(result)
    assert math.isclose(float(result["p"]),r/(r+mean),rel_tol=0.0,abs_tol=2e-16)


def test_old_escape_and_hard_nonconvergence_cannot_silently_pass():
    old_r,old_p,old_ll=_old_unbracketed_newton(ESCAPE_TO_P1)
    assert old_r>1e12 and old_p==1.0 and not math.isfinite(old_ll)
    result=_solve(ESCAPE_TO_P1)
    assert bool(result["converged"])
    assert float(result["r"])<10 and float(result["p"])<0.9
    assert math.isfinite(float(result["log_likelihood"]))


def test_old_false_convergence_is_rejected_by_root_and_objective_contract():
    old_r,old_p,old_ll=_old_unbracketed_newton(FALSE_CONVERGENCE)
    result=_solve(FALSE_CONVERGENCE)
    assert old_r>1e9 and old_p>0.999999999
    _assert_root_certification(result)
    _assert_dec016_fit(FALSE_CONVERGENCE,result)
    if math.isfinite(old_ll):
        assert float(result["log_likelihood"])>old_ll+1.0
    else:
        assert not math.isfinite(old_ll)
    assert float(result["r"])<1.0 and float(result["p"])<0.9


def test_exact_r10a_blocker_cell_is_certified_and_matches_dec016():
    cell=_primary_nb_cell(r=5,p=0.9,n=250,statistic="AD")
    sample,_=runner.fixed_observed(cell,2,"CP05-C2C")
    assert runner._cpu_nb_classification(sample)=="ELIGIBLE"
    result=_solve(sample)
    assert result["classification"]=="ELIGIBLE"
    _assert_root_certification(result,cell.canonical_id)
    _assert_dec016_fit(sample,result,cell.canonical_id)


def test_all_primary_nb_observed_samples_are_certified_or_classified_exactly():
    samples_by_n={}
    for cell in runner.primary_fixture_matrix():
        if cell.family!="negative_binomial":
            continue
        for raw_outer_index in range(runner.R_EQ):
            context=f"{cell.canonical_id}|raw_outer={raw_outer_index}"
            sample,_=runner.fixed_observed(cell,raw_outer_index,"CP05-C2C")
            samples_by_n.setdefault(cell.n,[]).append((context,sample))

    for entries in samples_by_n.values():
        batched=_solve(np.stack([sample for _,sample in entries]))
        for index,(context,sample) in enumerate(entries):
            cpu_classification=runner._cpu_nb_classification(sample)
            result=_result_at(batched,index)
            assert result["classification"]==cpu_classification, context
            if cpu_classification=="ELIGIBLE":
                _assert_root_certification(result,context)
                _assert_dec016_fit(sample,result,context)
            else:
                assert not bool(result["converged"]), context


def test_exact_r10a_blocker_bootstraps_are_certified_and_match_dec016():
    cell=_primary_nb_cell(r=5,p=0.9,n=250,statistic="AD")
    observed,_=runner.fixed_observed(cell,2,"CP05-C2C")
    _,attempts,eligible=runner.fixed_bootstraps(cell,observed,2,"CP05-C2C")
    assert len(eligible)==runner.B_EQ==15
    assert len(attempts)>=len(eligible)
    batched=_solve(np.stack([record["sample"] for record in eligible]))
    for index,record in enumerate(eligible):
        context=f"{cell.canonical_id}|raw_outer=2|raw_inner={record['raw_inner_index']}"
        assert record["canonical_status"]=="ELIGIBLE", context
        result=_result_at(batched,index)
        assert result["classification"]=="ELIGIBLE", context
        _assert_root_certification(result,context)
        _assert_dec016_fit(record["sample"],result,context)


def test_solver_is_batch_compatible():
    samples=np.stack((ESCAPE_TO_P1,FALSE_CONVERGENCE,R025_LIKE))
    batched=_solve(samples)
    assert batched["classification"]==["ELIGIBLE"]*3
    assert np.all(batched["converged"])
    for index,sample in enumerate(samples):
        scalar=_solve(sample)
        assert float(batched["r"][index])==pytest.approx(float(scalar["r"]),rel=2e-15)
        assert float(batched["p"][index])==pytest.approx(float(scalar["p"]),rel=2e-15)


def test_solver_fails_closed_when_no_safe_bracket_can_be_certified():
    result=_solve(FALSE_CONVERGENCE,bracket_steps=0)
    assert not bool(result["bracket_found"])
    assert not bool(result["converged"])


def test_mom_is_not_returned_as_the_estimate():
    sample=FALSE_CONVERGENCE.astype(np.float64)
    mean=float(np.mean(sample)); variance=float(np.var(sample,ddof=0))
    mom=mean*mean/(variance-mean)
    result=_solve(sample)
    assert not math.isclose(float(result["r"]),mom,rel_tol=1e-6,abs_tol=0.0)
