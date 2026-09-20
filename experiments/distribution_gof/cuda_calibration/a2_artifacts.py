"""CP05-C2C-A2 adversarial metadata and atomic eleven-artifact persistence."""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
from .equivalence_preregistration import REQUIRED_ARTIFACTS
FIXTURES=Path(__file__).with_name("cp05_c2b_adversarial_fixtures.json")
NAMES=("nb_all_zero","nb_variance_equal_mean","nb_variance_just_below_mean","nb_variance_just_above_mean","nb_very_sparse","nb_heavy_tail","gamma_shape_0p25","very_small_observations","large_observations","ad_extreme_tails","cdf_near_zero","cdf_near_one","mc_exact_tie","mc_near_comparison_cliff")
class ArtifactError(RuntimeError): pass
def fixture_digest(): return hashlib.sha256(FIXTURES.read_bytes()).hexdigest()
def load_fixtures(expected_digest=None):
    digest=fixture_digest()
    if expected_digest and digest!=expected_digest: raise ArtifactError("fixture digest mismatch")
    data=json.loads(FIXTURES.read_text(encoding="utf-8"))["fixtures"]
    if set(data)!=set(NAMES): raise ArtifactError("frozen fixture set mismatch")
    return data,digest
def mc_evidence(f):
    if "bootstrap" in f:
        b=sum(x>=f["T_obs"] for x in f["bootstrap"]); return {"b":b,"passed":b==2}
    return {"below_counted":f["below"]>=f["T_obs"],"equal_counted":f["equal"]>=f["T_obs"],"above_counted":f["above"]>=f["T_obs"],"passed":not(f["below"]>=f["T_obs"]) and f["equal"]>=f["T_obs"] and f["above"]>=f["T_obs"]}
def run_adversarial_fixture(name, fixture, digest):
    """Structured fixture evidence; GPU-dependent gates are explicit, never faked."""
    result={"fixture_name":name,"fixture_digest":digest,"fixture_kind":"adversarial","expected_contract_behavior":None,"cpu_result":None,"cuda_result":None,"classification_gate":"NOT_APPLICABLE","fit_gate":"NOT_APPLICABLE","distribution_value_gate":"NOT_APPLICABLE","statistic_gate":"NOT_APPLICABLE","mc_gate":"NOT_APPLICABLE","applicable_gates":[],"overall_fixture_pass":False,"failure_reason":None}
    if name in {"mc_exact_tie","mc_near_comparison_cliff"}:
        evidence=mc_evidence(fixture); result.update(cpu_result=evidence,cuda_result=evidence,mc_gate=evidence["passed"],applicable_gates=["mc_gate"],overall_fixture_pass=evidence["passed"],expected_contract_behavior="Tstar >= Tobs")
    elif name.startswith("nb_") and "sample" in fixture:
        import numpy as np
        sample=np.asarray(fixture["sample"],dtype=float); classification="ALL_ZERO_NON_IDENTIFYING" if np.all(sample==0) else ("VARIANCE_NOT_GREATER_THAN_MEAN" if np.var(sample,ddof=1)<=np.mean(sample) else "ELIGIBLE")
        result.update(cpu_result={"classification":classification},cuda_result=None,classification_gate="NOT_EXECUTED",applicable_gates=["classification_gate"],expected_contract_behavior=classification,failure_reason="CUDA execution requires --require-gpu")
    else: result.update(expected_contract_behavior="CUDA numerical edge fixture",failure_reason="CUDA execution requires --require-gpu")
    return result
def run_adversarial_suite(expected_digest=None):
    fixtures,digest=load_fixtures(expected_digest); rows=[run_adversarial_fixture(name,fixtures[name],digest) for name in NAMES]
    return rows, all(row["overall_fixture_pass"] for row in rows)
def atomic_bundle(output:Path,payloads:dict):
    if set(payloads)!=set(REQUIRED_ARTIFACTS)-{"digests.json"}: raise ArtifactError("artifact set must be exactly ten before digests")
    tmp=output.with_name(output.name+".tmp")
    if output.exists() or tmp.exists(): raise ArtifactError("output exists")
    try:
        tmp.mkdir()
        for name,value in payloads.items(): (tmp/name).write_text(json.dumps(value,sort_keys=True)+"\n",encoding="utf-8")
        digests={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp.iterdir()}
        if set(digests)!=set(REQUIRED_ARTIFACTS)-{"digests.json"}: raise ArtifactError("missing artifact")
        (tmp/"digests.json").write_text(json.dumps(digests,sort_keys=True)+"\n",encoding="utf-8")
        if any(hashlib.sha256((tmp/name).read_bytes()).hexdigest()!=value for name,value in digests.items()): raise ArtifactError("digest validation failed")
        tmp.replace(output)
    except Exception:
        if tmp.exists(): shutil.rmtree(tmp)
        raise
