"""CPU_REFERENCE DEC-014 canonical NB support certification for C2C only."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from experiments.distribution_gof.statistics import TAIL_ABSOLUTE_TOLERANCE, TAIL_RELATIVE_TOLERANCE, MAX_SUPPORT_TERMS, _positive_float_bound, stable_ad_upper_tail_term

@dataclass(frozen=True)
class CanonicalNBSupport:
    statistic:str; n:int; sample_max:int; support_stop:int; remainder_bound:float; required_bound:float; reference_partial_statistic:float
    support_start:int=0; certified:bool=True; support_source:str="CPU_REFERENCE_DEC014_TAIL_CERTIFICATION"
    tail_absolute_tolerance:float=TAIL_ABSOLUTE_TOLERANCE; tail_relative_tolerance:float=TAIL_RELATIVE_TOLERANCE; max_support_terms:int=MAX_SUPPORT_TERMS
    @property
    def support_size(self): return self.support_stop+1
    @property
    def indices(self): return tuple(range(self.support_stop+1))

class SupportCertificationError(RuntimeError): pass

def certify_nb_support(sample,bound,statistic):
    values=np.sort(np.asarray(sample,dtype=np.int64)); n=int(values.size); maximum=int(values[-1]); terms=[]
    for j in range(MAX_SUPPORT_TERMS):
        lp,lc,ls=float(bound.logpmf(j)),float(bound.logcdf(j)),float(bound.logsf(j))
        if not all(math.isfinite(x) for x in (lp,lc,ls)): raise SupportCertificationError("TAIL_CERTIFICATION_FAILED")
        if j<maximum:
            z=int(np.searchsorted(values,j,side="right"))-n*math.exp(lc)
            term=0. if z==0 else math.exp(2*math.log(abs(z))-math.log(n)+lp-(lc+ls if statistic=="AD" else 0)); remainder=math.inf
        elif statistic=="AD":
            term=stable_ad_upper_tail_term(bound,n,j); remainder=_positive_float_bound(math.log(n)+2*ls-lc)
        else:
            term=math.exp(math.log(n)+2*ls+lp); remainder=_positive_float_bound(math.log(n)+3*ls)
        terms.append(term); partial=math.fsum(terms); required=max(TAIL_ABSOLUTE_TOLERANCE,TAIL_RELATIVE_TOLERANCE*abs(partial))
        if j>=maximum and remainder<=required: return CanonicalNBSupport(statistic,n,maximum,j,remainder,required,partial)
    raise SupportCertificationError("TAIL_CERTIFICATION_TERM_BUDGET_EXHAUSTED")
