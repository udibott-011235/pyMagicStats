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
