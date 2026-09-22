"""CUDA float64 candidate operations. CPU fallback is structurally forbidden."""
from __future__ import annotations
import math
try:
    import cupy as cp
    from cupyx.scipy import special as csp
except ImportError: cp = csp = None

class CudaCandidateError(RuntimeError): pass
_NB_CLASSIFICATION_LABELS={0:"ALL_ZERO_NON_IDENTIFYING",1:"ELIGIBLE",2:"VARIANCE_NOT_GREATER_THAN_MEAN"}
_INT64_MAX=(1 << 63)-1
_NB_BRACKET_STEPS=1024
_NB_SOLVER_ITERATIONS=128
_NB_SCORE_CHUNK=4096
_NB_SCORE_TERM_BUDGET=1_000_000
_NB_SMALL_Z=1e-3

def _nb_classification_metadata(device_codes):
    """Map already-computed CUDA integer codes to JSON-safe labels outside CUDA math."""
    def labels(value):
        if isinstance(value,list): return [labels(item) for item in value]
        return _NB_CLASSIFICATION_LABELS[int(value)]
    return labels(device_codes.tolist() if hasattr(device_codes,"tolist") else device_codes)
def require_cuda():
    if cp is None: raise CudaCandidateError("CUDA_CANDIDATE_UNIMPLEMENTED: CuPy unavailable")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1: raise CudaCandidateError("CUDA_CANDIDATE_UNIMPLEMENTED: GPU unavailable")
    except CudaCandidateError: raise
    except Exception as exc: raise CudaCandidateError("CUDA_CANDIDATE_UNIMPLEMENTED: CUDA runtime") from exc
    return cp
def _x(a, positive=False):
    xp=require_cuda(); a=xp.asarray(a,dtype=xp.float64)
    if a.ndim<1 or a.shape[-1]==0 or not bool(xp.all(xp.isfinite(a))): raise CudaCandidateError("nonfinite sample")
    if positive and bool(xp.any(a<=0)): raise CudaCandidateError("nonpositive sample")
    return a
def fit_exponential(a):
    a=_x(a,True); scale=cp.mean(a,axis=-1)
    return {"scale":scale,"converged":cp.isfinite(scale)&(scale>0),"iterations":0}
def fit_gamma(a,iterations=96):
    a=_x(a,True); mean=cp.mean(a,axis=-1); s=cp.log(mean)-cp.mean(cp.log(a),axis=-1); shape=cp.maximum((3-s+cp.sqrt((s-3)**2+24*s))/(12*s),1e-12)
    # Same device-native ψ₁ order representation as the R5 NB compatibility fix.
    trigamma_order=cp.asarray(1,dtype=cp.int32)
    for i in range(iterations):
        f=cp.log(shape)-csp.digamma(shape)-s; d=1/shape-csp.polygamma(trigamma_order,shape); proposal=shape-f/d; shape=cp.where((proposal>0)&cp.isfinite(proposal),proposal,shape/2)
    ok=cp.abs(cp.log(shape)-csp.digamma(shape)-s)<=1e-12*cp.maximum(1,cp.abs(s))
    return {"shape":shape,"scale":mean/shape,"converged":ok,"iterations":iterations}


def _nb_classification_codes(a, xp):
    """Canonical integer NB classification without overflow or float fallback.

    The sufficient input bound makes each of ``n*sum(x*x)``, ``sum(x)**2``
    and ``n*sum(x)`` representable in signed int64.  The final comparison uses
    a saturated right-hand side, so ``sum(x)**2 + n*sum(x)`` cannot wrap.
    Inputs outside this certifiable experiment envelope fail closed.
    """
    values=xp.asarray(a,dtype=xp.float64)
    if values.ndim<1 or values.shape[-1]==0 or not bool(xp.all(xp.isfinite(values))):
        raise CudaCandidateError("nonfinite sample")
    if bool(xp.any(values<0)) or bool(xp.any(values!=xp.floor(values))):
        raise CudaCandidateError("NB integral sample required")
    n=int(values.shape[-1])
    safe_max=math.isqrt(_INT64_MAX)//n
    if bool(xp.any(values>safe_max)):
        raise CudaCandidateError("NB canonical classification overflow risk")
    integers=values.astype(xp.int64)
    total=xp.sum(integers,axis=-1,dtype=xp.int64)
    sum_squares=xp.sum(integers*integers,axis=-1,dtype=xp.int64)
    left=n*sum_squares
    total_square=total*total
    linear=n*total
    rhs_overflow=total_square>(_INT64_MAX-linear)
    safe_square=xp.where(rhs_overflow,0,total_square)
    right=safe_square+linear
    allzero=xp.all(integers==0,axis=-1)
    eligible=(~allzero)&(~rhs_overflow)&(left>right)
    return xp.where(allzero,0,xp.where(eligible,1,2)).astype(xp.int8)


def _nb_survival_counts(values, xp):
    """Return count(x > k) on bounded chunks for the integer recurrences."""
    maximum=int(xp.max(values).item())
    if maximum>_NB_SCORE_TERM_BUDGET:
        raise CudaCandidateError("NB score recurrence term budget exhausted")
    pieces=[]
    for start in range(0,maximum,_NB_SCORE_CHUNK):
        stop=min(maximum,start+_NB_SCORE_CHUNK)
        k=xp.arange(start,stop,dtype=xp.float64)
        pieces.append(xp.sum(values[...,None]>k,axis=-2,dtype=xp.float64))
    if not pieces:
        return xp.zeros((values.shape[0],0),dtype=xp.float64)
    return xp.concatenate(pieces,axis=-1)


def _nb_small_z_polynomial(z,xp):
    """Return ``(z-log1p(z))/z**2`` by its twelve-term series."""
    polynomial=xp.zeros_like(z)+1/13
    for denominator in range(12,1,-1):
        polynomial=1/denominator-z*polynomial
    return polynomial


def _nb_z_minus_log1p(z, xp):
    """Stable ``z-log1p(z)``; the series remainder is <1e-39 at 1e-3."""
    # z-log(1+z) = z^2 * (1/2-z/3+z^2/4-...).  Twelve terms keep
    # truncation far below float64 representation error at the transition.
    polynomial=_nb_small_z_polynomial(z,xp)
    series=z*z*polynomial
    small=z<_NB_SMALL_Z
    direct_z=xp.where(small,1.0,z)
    direct=direct_z-xp.log1p(direct_z)
    return xp.where(small,series,direct)


def _nb_scaled_score(r, mean, survival, xp):
    """Cancellation-resistant ``r**2``-scaled CP04 profile score."""
    k=xp.arange(survival.shape[-1],dtype=xp.float64)
    # ``survival`` is already aggregated over observations; n is supplied by
    # normalizing it before this helper is called.
    recurrence=xp.sum(survival*(k/(1+k/r[...,None])),axis=-1)
    z=mean/r
    small=z<_NB_SMALL_Z
    # Avoid eagerly evaluating r**2 or a small-z cancellation in either branch.
    direct_z=xp.where(small,1.0,z)
    direct_ratio=(direct_z-xp.log1p(direct_z))/(direct_z*direct_z)
    ratio=xp.where(small,_nb_small_z_polynomial(z,xp),direct_ratio)
    profile_term=mean*mean*ratio
    return profile_term-recurrence


def _nb_profile_log_likelihood(r, mean, survival, log_factorial_mean, n, xp):
    """Stable integer-recurrence form of the same profiled likelihood."""
    k=xp.arange(survival.shape[-1],dtype=xp.float64)
    recurrence=xp.sum(survival*xp.log1p(k/r[...,None]),axis=-1)
    z=mean/r
    normalized_log=xp.log1p(z)/z
    average=recurrence-log_factorial_mean+mean*xp.log(mean)-mean*(normalized_log+xp.log1p(z))
    return n*average


def _fit_negative_binomial_impl(a,xp,special,*,bracket_steps=_NB_BRACKET_STEPS,solver_iterations=_NB_SOLVER_ITERATIONS):
    """Backend-generic R10-A solver; production calls it only with CuPy."""
    values=xp.asarray(a,dtype=xp.float64)
    codes=_nb_classification_codes(values,xp)
    n=int(values.shape[-1]); batch_shape=values.shape[:-1]
    flat=values.reshape((-1,n)); flat_codes=codes.reshape((-1,))
    integers=flat.astype(xp.int64)
    survival=_nb_survival_counts(integers,xp)/n
    mean=xp.mean(flat,axis=-1)
    safe_mean=xp.where(flat_codes==1,mean,1.0)
    variance=xp.mean((flat-safe_mean[...,None])**2,axis=-1)
    excess=xp.maximum(variance-safe_mean,xp.finfo(xp.float64).tiny)
    r0=xp.where(flat_codes==1,safe_mean*safe_mean/excess,1.0)
    r0=xp.where(xp.isfinite(r0)&(r0>0),r0,1.0)
    eta0=xp.log(r0); low=eta0.copy(); high=eta0.copy()
    score=lambda eta: _nb_scaled_score(xp.exp(eta),safe_mean,survival,xp)
    left=score(low); right=left.copy()
    valid=xp.isfinite(left); found=(left>0)&(right<0)&valid
    log_two=math.log(2.0)
    lower_limit=math.log(xp.finfo(xp.float64).tiny)+log_two
    upper_limit=math.log(xp.finfo(xp.float64).max)-log_two
    bracket_iterations=0
    for index in range(bracket_steps):
        active=(flat_codes==1)&valid&(~found)
        if not bool(xp.any(active)): break
        move_low=active&(left<=0); move_high=active&(right>=0)
        next_low=xp.where(move_low,low-log_two,low)
        next_high=xp.where(move_high,high+log_two,high)
        within=(next_low>=lower_limit)&(next_high<=upper_limit)
        valid=valid&xp.where(active,within,True)
        low=xp.where(valid,next_low,low); high=xp.where(valid,next_high,high)
        left=score(low); right=score(high)
        valid=valid&xp.isfinite(left)&xp.isfinite(right)
        found=(left>0)&(right<0)&valid
        bracket_iterations=index+1
    original_low=low.copy(); original_high=high.copy()
    root_valid=found.copy()
    for _ in range(solver_iterations):
        middle=(low+high)/2
        middle_score=score(middle)
        finite=xp.isfinite(middle_score)
        root_valid=root_valid&finite
        low=xp.where(root_valid&(middle_score>=0),middle,low)
        high=xp.where(root_valid&(middle_score<0),middle,high)
    eta=(low+high)/2; r=xp.exp(eta)
    p=r/(r+safe_mean)
    residual=score(eta)
    probe=1e-7
    probe_left=score(eta-probe); probe_right=score(eta+probe)
    probe_inside=(eta-probe>original_low)&(eta+probe<original_high)
    sign_ok=probe_inside&(probe_left>0)&(probe_right<0)
    residual_ok=xp.abs(residual)<=xp.maximum(xp.abs(probe_left),xp.abs(probe_right))*1e-4
    inside=(eta>original_low)&(eta<original_high)
    log_factorial_mean=xp.mean(special.gammaln(flat+1),axis=-1)
    ll=_nb_profile_log_likelihood(r,safe_mean,survival,log_factorial_mean,n,xp)
    objective_probe=1e-5
    ll_left=_nb_profile_log_likelihood(xp.exp(eta-objective_probe),safe_mean,survival,log_factorial_mean,n,xp)
    ll_right=_nb_profile_log_likelihood(xp.exp(eta+objective_probe),safe_mean,survival,log_factorial_mean,n,xp)
    ll_start=_nb_profile_log_likelihood(r0,safe_mean,survival,log_factorial_mean,n,xp)
    competitor=xp.maximum(ll_start,xp.maximum(ll_left,ll_right))
    objective_budget=64*xp.finfo(xp.float64).eps*xp.maximum(1,xp.maximum(xp.abs(ll),xp.abs(competitor)))
    objective_ok=xp.isfinite(ll)&xp.isfinite(competitor)&(ll+objective_budget>=competitor)
    finite_pair=xp.isfinite(r)&(r>0)&xp.isfinite(p)&(p>0)&(p<1)
    converged=(flat_codes==1)&root_valid&found&inside&sign_ok&residual_ok&finite_pair&objective_ok
    shape=batch_shape
    result={"r":r.reshape(shape),"p":p.reshape(shape),"log_likelihood":ll.reshape(shape),
            "converged":converged.reshape(shape),"iterations":bracket_iterations+solver_iterations,
            "classification":_nb_classification_metadata(codes),
            "bracket_found":found.reshape(shape),"root_inside_bracket":inside.reshape(shape),
            "root_sign_check":sign_ok.reshape(shape),"root_residual_check":residual_ok.reshape(shape),
            "objective_valid":objective_ok.reshape(shape)}
    return result


def fit_negative_binomial(a,iterations=_NB_SOLVER_ITERATIONS):
    """Batched bracketed real-r profile MLE; MoM supplies only bracket scale."""
    if iterations!=_NB_SOLVER_ITERATIONS:
        raise CudaCandidateError("NB solver iteration contract is fixed")
    a=_x(a)
    return _fit_negative_binomial_impl(a,cp,csp,solver_iterations=iterations)
def fit(family, sample):
    if family=="exponential": return fit_exponential(sample)
    if family=="gamma": return fit_gamma(sample)
    if family=="negative_binomial": return fit_negative_binomial(sample)
    raise CudaCandidateError("unsupported CUDA family")
def continuous_statistic(logcdf,logsf,statistic):
    logcdf=_x(logcdf); logsf=_x(logsf); n=logcdf.shape[-1]
    if statistic=="AD": return -n-cp.sum((2*cp.arange(1,n+1)-1)*(logcdf+cp.flip(logsf,axis=-1)),axis=-1)/n
    if statistic=="CVM": return 1/(12*n)+cp.sum((cp.exp(logcdf)-(2*cp.arange(1,n+1)-1)/(2*n))**2,axis=-1)
    raise CudaCandidateError("unsupported statistic")
def nb_statistic(pmf,cdf,sample,statistic,*,remainder_bound):
    pmf=_x(pmf); cdf=_x(cdf); sample=_x(sample)
    if remainder_bound is None or remainder_bound<0: raise CudaCandidateError("uncertified NB tail")
    n=sample.shape[-1]; z=cp.sum(sample[..., :,None]<=cp.arange(pmf.shape[-1]),axis=-2)-n*cdf
    if statistic=="CVM": return cp.sum(z*z*pmf,axis=-1)/n
    if statistic=="AD":
        if bool(cp.any(cdf<=0)) or bool(cp.any(cdf>=1)): raise CudaCandidateError("uncertified NB AD endpoint")
        return cp.sum(z*z*pmf/(cdf*(1-cdf)),axis=-1)/n
    raise CudaCandidateError("unsupported NB statistic")

def generate(family, parameters, n, seed):
    """CUDA generator reserved for the seven frozen generator-sanity cases."""
    require_cuda(); rng=cp.random.RandomState(seed & 0xffffffff)
    if family=="gamma": return rng.gamma(parameters["shape"],parameters["scale"],size=n).astype(cp.float64)
    if family=="exponential": return rng.exponential(parameters["scale"],size=n).astype(cp.float64)
    if family=="negative_binomial": return rng.negative_binomial(parameters["r"],parameters["p"],size=n).astype(cp.float64)
    raise CudaCandidateError("unsupported CUDA generator")

def distribution_values(family, x, parameters):
    """Candidate-only float64 distribution quantities."""
    require_cuda(); x=cp.asarray(x,dtype=cp.float64)
    if family=="exponential":
        z=-x/parameters["scale"]; sf=cp.where(x<0,1.,cp.exp(z)); cdf=cp.where(x<0,0.,-cp.expm1(z))
        return {"cdf":cdf,"sf":sf,"logCDF":cp.where(x<0,-cp.inf,cp.log1p(-cp.exp(z))),"logSF":cp.where(x<0,0.,z)}
    if family=="gamma":
        if not hasattr(csp,"gammaincc"): raise CudaCandidateError("CUDA_CANDIDATE_UNIMPLEMENTED: gamma tail")
        z=x/parameters["scale"]; cdf=csp.gammainc(parameters["shape"],z); sf=csp.gammaincc(parameters["shape"],z)
        return {"cdf":cdf,"sf":sf,"logCDF":cp.log(cdf),"logSF":cp.log(sf)}
    if family=="negative_binomial":
        if not hasattr(csp,"betainc"): raise CudaCandidateError("CUDA_CANDIDATE_UNIMPLEMENTED: NB CDF")
        r,p=parameters["r"],parameters["p"]; lp=csp.gammaln(x+r)-csp.gammaln(r)-csp.gammaln(x+1)+r*cp.log(p)+x*cp.log1p(-p); cdf=csp.betainc(r,x+1,p); sf=csp.betainc(x+1,r,1-p)
        return {"pmf":cp.exp(lp),"logPMF":lp,"cdf":cdf,"sf":sf,"logCDF":cp.log(cdf),"logSF":cp.log(sf)}
    raise CudaCandidateError("unsupported distribution")

def candidate_statistic(family,sample,parameters,statistic,*,support=None,remainder_bound=None):
    x=_x(sample, family!="negative_binomial")
    if family!="negative_binomial":
        x=cp.sort(x); q=distribution_values(family,x,parameters); result=continuous_statistic(q["logCDF"],q["logSF"],statistic)
    else:
        if support is None: raise CudaCandidateError("uncertified NB tail")
        q=distribution_values(family,support,parameters); result=nb_statistic(q["pmf"],q["cdf"],x,statistic,remainder_bound=remainder_bound)
    if not bool(cp.all(cp.isfinite(result))): raise CudaCandidateError("nonfinite candidate statistic")
    return result
