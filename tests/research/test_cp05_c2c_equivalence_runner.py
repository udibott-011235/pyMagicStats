"""C2C non-GPU orchestration tests; not scientific equivalence evidence."""
import pytest
from experiments.distribution_gof.cuda_calibration.cp05_c2c_equivalence_runner import *

def test_frozen_matrix_counts_and_fixture_digest():
    assert len(primary_fixture_matrix()) == 144
    assert PRIMARY_OUTER_TARGET == 1152 and (R_EQ, B_EQ) == (8, 15)
    assert frozen_contract(BASELINE_SHA) == fixture_digest()

def test_fails_closed_for_wrong_contract_and_duplicate_identity():
    with pytest.raises(C2CError): frozen_contract("0" * 40)
    with pytest.raises(C2CError): frozen_contract(BASELINE_SHA, R=9)
    with pytest.raises(C2CError): assert_identities([{"identity": "x"}, {"identity": "x"}])

def test_artifact_contract_and_gpu_cli_are_non_claiming():
    bundle = artifact_skeleton(BASELINE_SHA)
    assert set(bundle) == set(REQUIRED_ARTIFACTS)
    assert bundle["summary.json"] == {"equivalence_gate_passed": False, "calibration_claim": False}
    assert bundle["equivalence_manifest.json"]["BOOTSTRAP_FIXTURE_SOURCE"] == BOOTSTRAP_FIXTURE_SOURCE
    with pytest.raises(SystemExit): main(["--output", "x"])

def test_fixed_data_and_bootstraps_are_shared_cpu_fixture_objects():
    cell = primary_fixture_matrix()[0]
    observed, identity = fixed_observed(cell, 0, "fixed")
    cpu, attempts, eligible = fixed_bootstraps(cell, observed, 0, "fixed")
    assert identity["cell_id"] == cell.canonical_id and cpu["engine"] == "CPU_REFERENCE"
    assert len(eligible) == B_EQ and len(attempts) == B_EQ
    assert all(item["canonical_status"] == "ELIGIBLE" for item in eligible)

def test_atomic_bundle_rejects_missing_parquet_artifacts(tmp_path):
    with pytest.raises(C2CError, match="parquet"):
        write_atomic_bundle(tmp_path / "x", artifact_skeleton(BASELINE_SHA))

def test_dual_engine_gate_uses_shared_input_and_requires_every_gate():
    cell=primary_fixture_matrix()[0]; sample, _=fixed_observed(cell,0,"n"); seen=[]
    def adapter(label):
        def run(_, value):
            seen.append((label,id(value)))
            return {"classification":"ELIGIBLE","parameters":{"scale":1.0},"log_likelihood":None,"statistic":1.0,"evaluation_points":[1.0],"distribution_values":{"cdf":[.5],"sf":[.5],"logCDF":[-.7],"logSF":[-.7]}}
        return run
    row=evaluate_fixed_record(identity="i",record_type="observed",cell=cell,raw_outer_index=0,raw_inner_index=None,sample=sample,reference_adapter=adapter("cpu"),cuda_adapter=adapter("cuda"))
    assert seen[0][1] == seen[1][1] == id(sample)
    assert all(row[key] for key in ("classification_gate_pass","fit_gate_pass","distribution_value_gate_pass","statistic_gate_pass"))
    assert provisional_global([]) is False

def test_mc_mismatch_fails_outer():
    record={"classification_gate_pass":True,"fit_gate_pass":True,"distribution_value_gate_pass":True,"statistic_gate_pass":True,"cpu_statistic":1.,"cuda_statistic":1.}
    boot=[dict(record) for _ in range(B_EQ)]; boot[0]["cuda_statistic"]=2.
    assert aggregate_outer(record,boot)["outer_gate_pass"] is False

def test_cuda_adapter_fails_closed_without_gpu_and_never_returns_reference_values():
    cell=primary_fixture_matrix()[0]; sample,_=fixed_observed(cell,0,"n")
    result=evaluate_cuda_record(cell,sample)
    assert result["classification"] == "FAILED"
    assert result["parameters"] == {} and result["solver_converged"] is False
