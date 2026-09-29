"""Deterministic CP05-C2C artifact schemas and writers (no execution layer)."""
from __future__ import annotations
import hashlib,json,platform,sys
from pathlib import Path
from .equivalence_preregistration import REQUIRED_ARTIFACTS,R_EQ,B_EQ,PRIMARY_CELL_COUNT
REQUIRED_ARTIFACT_COUNT=11
FIT_COLUMNS=("identity","cell_id","family","n","statistic","raw_outer_index","raw_inner_index","record_type","cpu_classification","cuda_classification","classification_match","cpu_parameters_json","cuda_parameters_json","cpu_log_likelihood","cuda_log_likelihood","fit_gate_pass","flat_objective_used","flat_objective_diagnostic_json","cuda_failure_reason","nb_support_stop","nb_support_size","nb_remainder_bound","nb_required_bound")
STAT_COLUMNS=("identity","cell_id","family","statistic","raw_outer_index","raw_inner_index","record_type","cpu_statistic","cuda_statistic","abs_error","allowed_tolerance","gate_pass")
CLASS_COLUMNS=("identity","cell_id","raw_outer_index","raw_inner_index","record_type","cpu_classification","cuda_classification","exact_match")
def _json(path,payload): path.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def write_equivalence_manifest(path,**runtime):
    base={"work_item":"CP05-C2C","reference_engine":"CPU_REFERENCE","reference_sha":runtime.get("baseline_sha"),"cuda_candidate_id":"CUDA_CANDIDATE","dec016_identifier":"DEC-016","float_precision":"float64","R_EQ":R_EQ,"B_EQ":B_EQ,"primary_cells":PRIMARY_CELL_COUNT,"primary_outer_target":1152,"bootstrap_fixture_source":"CPU_REFERENCE_FITTED_PARAMETERS","batch_partitions":[[1,1],[2,3],[4,5]],"artifact_schema_version":"cp05-c2c-v1"}; base.update(runtime); _json(path,base)
def write_fixture_manifest(path,observed,bootstrap,adversarial): _json(path,{"observed":observed,"bootstrap_attempts":bootstrap,"adversarial":adversarial})
def _parquet(path,rows,columns):
    import pandas as pd
    frame=pd.DataFrame(rows,columns=columns); frame.to_parquet(path,index=False); loaded=pd.read_parquet(path)
    if tuple(loaded.columns)!=columns: raise ValueError("parquet schema mismatch")
def write_fit_comparison(path,rows): _parquet(path,rows,FIT_COLUMNS)
def write_statistic_comparison(path,rows): _parquet(path,rows,STAT_COLUMNS)
def write_classification_comparison(path,rows): _parquet(path,rows,CLASS_COLUMNS)
def write_batch_invariance(path,subset,results,passed=False): _json(path,{"partitions":[[1,1],[2,3],[4,5]],"subset":subset,"results":results,"passed":bool(passed)})
def write_rng_identity(path,rows): _json(path,{"backend_independent":True,"batch_independent":True,"execution_order_independent":True,"resume_boundary_independent":True,"identities":rows})
def write_generator_sanity(path, *, passed=False, results=None): _json(path,{"executed":False,"passed":bool(passed),"case_count":7,"N":1000000,"results":list(results or [])})
def write_environment(path,git_sha): _json(path,{"git_sha":git_sha,"python_version":sys.version,"platform":platform.platform(),"execution_environment":"NOT_EXECUTED","cuda_runtime":None,"nvidia_driver":None,"device_name":None,"compute_capability":None,"total_vram":None,"cupy_version":None,"numpy_version":None,"scipy_version":None,"cudf_version_or_null":None,"CUDA_VISIBLE_DEVICES":None,"float_precision":"float64"})
def write_summary(path,**values):
    base={"execution_mode":"NOT_EXECUTED","primary_outer_expected":1152,"primary_outer_observed":0,"adversarial_fixture_expected":14,"adversarial_fixture_observed":0,"equivalence_gate_passed":False,"generator_sanity_passed":False,"overall_pass":False,"batch_invariance_passed":False,"artifact_validation_passed":False,"calibration_claim":False,"failure_reasons":[]}; base.update(values)
    if base["calibration_claim"] is not False: raise ValueError("calibration_claim must remain false")
    _json(path,base)
def write_digests(path,digests):
    if set(digests)!=set(REQUIRED_ARTIFACTS)-{"digests.json"}: raise ValueError("digests require exactly ten artifacts")
    _json(path,digests)
