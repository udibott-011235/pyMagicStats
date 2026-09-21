"""CUDA float64 candidate operations. CPU fallback is structurally forbidden."""
from __future__ import annotations
try:
    import cupy as cp
    from cupyx.scipy import special as csp
except ImportError: cp = csp = None

class CudaCandidateError(RuntimeError): pass
_NB_CLASSIFICATION_LABELS={0:"ALL_ZERO_NON_IDENTIFYING",1:"ELIGIBLE",2:"VARIANCE_NOT_GREATER_THAN_MEAN"}

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
def fit_negative_binomial(a,iterations=128):
    """Batched real-r profiled likelihood solver; MoM only starts the bracket."""
    a=_x(a)
    if bool(cp.any(a<0)) or bool(cp.any(a!=cp.floor(a))): raise CudaCandidateError("NB integral sample required")
    mean=cp.mean(a,axis=-1); var=cp.var(a,axis=-1,ddof=1); allzero=cp.all(a==0,axis=-1); eligible=(~allzero)&(var>mean); r=cp.maximum(mean*mean/cp.maximum(var-mean,1e-300),1e-10)
    # cupyx.polygamma requires a device-side order to avoid Python-bool dispatch.
    # This remains exactly trigamma ψ₁(a+r) - ψ₁(r), evaluated on CUDA.
    trigamma_order=cp.asarray(1,dtype=cp.int32)
    for i in range(iterations):
        p=r/(r+mean); score=cp.sum(csp.digamma(a+r[...,None])-csp.digamma(r[...,None]),axis=-1)+a.shape[-1]*cp.log(p); deriv=cp.sum(csp.polygamma(trigamma_order,a+r[...,None])-csp.polygamma(trigamma_order,r[...,None]),axis=-1)+a.shape[-1]*(1/r-1/(r+mean)); proposal=r-score/deriv; r=cp.where((proposal>0)&cp.isfinite(proposal),proposal,r/2)
    p=r/(r+mean); ll=cp.sum(csp.gammaln(a+r[...,None])-csp.gammaln(r[...,None])-csp.gammaln(a+1)+r[...,None]*cp.log(p[...,None])+a*cp.log1p(-p[...,None]),axis=-1); ok=eligible&cp.isfinite(r)&cp.isfinite(p)&cp.isfinite(ll)
    # CuPy expressions remain numeric; labels are exported only after predicates resolve.
    classification_code=cp.where(allzero,0,cp.where(eligible,1,2))
    return {"r":r,"p":p,"log_likelihood":ll,"converged":ok,"iterations":iterations,"classification":_nb_classification_metadata(classification_code)}
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
