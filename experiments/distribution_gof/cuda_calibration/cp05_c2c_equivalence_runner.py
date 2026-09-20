"""Fail-closed CP05-C2C CUDA equivalence candidate.

This runner intentionally prepares, but never starts, a GPU campaign without
an authorized CUDA host and the frozen DEC-016 dimensions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import math
from pathlib import Path

from .equivalence_preregistration import B_EQ, GENERATOR_SANITY_CASES, GENERATOR_SANITY_N, PRIMARY_CELL_COUNT, R_EQ, REQUIRED_ARTIFACTS, primary_fixture_matrix, initial_summary
from .cp05_cuda_engine import _cp04_fit, _cp04_statistic, _generate, _parameters, derive_seed, mc_pvalue
from . import cuda_candidate
from .nb_support import certify_nb_support
from .equivalence_preregistration import categorical_agreement, distribution_value_agreement, fit_agreement, global_equivalence_pass, statistic_agreement

BASELINE_SHA = "fc92de4fc9b51303924d21b877eeece18caaf846"
BOOTSTRAP_FIXTURE_SOURCE = "CPU_REFERENCE_FITTED_PARAMETERS"
PRIMARY_OUTER_TARGET = 1152
FIXTURE_PATH = Path(__file__).with_name("cp05_c2b_adversarial_fixtures.json")


class C2CError(RuntimeError): pass


def fixture_digest() -> str:
    return hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()


def frozen_contract(git_sha: str, *, R: int = R_EQ, B: int = B_EQ, expected_fixture_digest: str | None = None) -> str:
    if git_sha != BASELINE_SHA: raise C2CError("wrong git SHA")
    if (R, B) != (R_EQ, B_EQ): raise C2CError("C2C rejects calibration-sized R/B")
    if len(primary_fixture_matrix()) != PRIMARY_CELL_COUNT or PRIMARY_OUTER_TARGET != PRIMARY_CELL_COUNT * R_EQ: raise C2CError("missing expected cell")
    digest = fixture_digest()
    if expected_fixture_digest is not None and digest != expected_fixture_digest: raise C2CError("wrong fixture digest")
    return digest


def assert_identities(rows: list[dict]) -> None:
    keys = [row["identity"] for row in rows]
    if len(keys) != len(set(keys)): raise C2CError("duplicate identity")
    if len(rows) != PRIMARY_OUTER_TARGET: raise C2CError("missing expected outer identity")


def generator_sanity_plan() -> dict:
    return {"executed": False, "cases": list(GENERATOR_SANITY_CASES), "N": GENERATOR_SANITY_N,
            "generator": "CUDA float64 required"}


def artifact_skeleton(git_sha: str) -> dict:
    """All artifact names and non-claiming initial semantics, before execution."""
    return {"equivalence_manifest.json": {"git_sha": git_sha, "reference_sha": BASELINE_SHA, "float_precision": "float64", "R_EQ": R_EQ, "B_EQ": B_EQ, "fixture_file_sha256": fixture_digest(), "BOOTSTRAP_FIXTURE_SOURCE": BOOTSTRAP_FIXTURE_SOURCE},
            "fixture_manifest.json": {"sha256": fixture_digest()},
            "fit_comparison.parquet": None, "statistic_comparison.parquet": None,
            "classification_comparison.parquet": None,
            "batch_invariance.json": {"partitions": [[1, 1], [2, 3], "hardware-selected"], "passed": False},
            "rng_identity.json": {"algorithm": "SHA256", "passed": False},
            "generator_sanity.json": generator_sanity_plan(),
            "environment.json": {"platform": platform.platform(), "float_precision": "float64"},
            "summary.json": {"equivalence_gate_passed": False, "generator_sanity_passed": False, "calibration_claim": False},
            "digests.json": None}


def reference_fit(family, sample):
    """Explicit CPU_REFERENCE adapter; never used by CUDA_CANDIDATE mathematics."""
    fit = _cp04_fit(family, sample)
    return {"engine": "CPU_REFERENCE", "bound": fit.fitted_distribution,
            "parameters": _parameters(fit.fitted_distribution)}


def evaluate_reference_record(cell, sample):
    result=reference_fit(cell.family,sample); bound=result["bound"]
    statistic=_cp04_statistic(sample,bound,cell.family,cell.statistic)
    # Values are deliberately CPU_REFERENCE evidence only.
    points=sorted(set(float(x) for x in sample))
    if cell.family=="negative_binomial":
        points=list(range(max(int(max(sample)),0)+1))
        values={"pmf":[float(bound.pmf(x)) for x in points],"logPMF":[float(bound.logpmf(x)) for x in points],"cdf":[float(bound.cdf(x)) for x in points],"sf":[float(bound.sf(x)) for x in points],"logCDF":[float(bound.logcdf(x)) for x in points],"logSF":[float(bound.logsf(x)) for x in points]}
    else: values={"cdf":[float(bound.cdf(x)) for x in points],"sf":[float(bound.sf(x)) for x in points],"logCDF":[float(bound.logcdf(x)) for x in points],"logSF":[float(bound.logsf(x)) for x in points]}
    record={"classification":"ELIGIBLE","parameters":result["parameters"],"log_likelihood":None,"statistic":statistic,"evaluation_points":points,"distribution_values":values}
    if cell.family=="negative_binomial":
        values_np=sample
        def likelihood(parameters):
            r,p=float(parameters["r"]),float(parameters["p"])
            if not (r>0 and 0<p<1): return float("-inf")
            return math.fsum(math.lgamma(float(x)+r)-math.lgamma(r)-math.lgamma(float(x)+1)+r*math.log(p)+float(x)*math.log1p(-p) for x in values_np)
        record["reference_log_likelihood"]=likelihood
        record["log_likelihood"]=likelihood(record["parameters"])
    return record


def evaluate_cuda_record(cell, sample, *, certified_support=None, remainder_bound=None):
    """CUDA_CANDIDATE-only adapter. Every exception is an explicit FAILED result."""
    try:
        fitted=cuda_candidate.fit(cell.family,sample)
        if not bool(cuda_candidate.cp.all(fitted["converged"])):
            raise C2CError("CUDA solver non-convergence")
        # Candidate primitives own fitting/statistics; adapter never calls CP04 here.
        params={key:float(cuda_candidate.cp.asnumpy(value)) for key,value in fitted.items() if key in {"shape","scale","r","p"}}
        points=sorted(set(float(x) for x in sample)) if cell.family!="negative_binomial" else list(certified_support or [])
        if cell.family=="negative_binomial" and not points: raise C2CError("uncertified NB tail")
        q=cuda_candidate.distribution_values(cell.family, points, params)
        statistic=float(cuda_candidate.cp.asnumpy(cuda_candidate.candidate_statistic(cell.family,sample,params,cell.statistic,support=points if cell.family=="negative_binomial" else None,remainder_bound=remainder_bound)))
        values={name:[float(x) for x in cuda_candidate.cp.asnumpy(value)] for name,value in q.items()}
        return {"classification":"ELIGIBLE","parameters":params,"log_likelihood":float(cuda_candidate.cp.asnumpy(fitted.get("log_likelihood",cuda_candidate.cp.asarray(float("nan"))))),"statistic":statistic,"evaluation_points":points,"distribution_values":values,"solver_converged":True,"iterations":fitted["iterations"],"dtype":"float64","failure_reason":None}
    except Exception as exc:
        return {"classification":"FAILED","parameters":{},"log_likelihood":None,"statistic":float("nan"),"evaluation_points":[],"distribution_values":{},"solver_converged":False,"iterations":0,"dtype":"float64","failure_reason":str(exc)}


def traverse_primary(namespace, *, reference_adapter=evaluate_reference_record, cuda_adapter=evaluate_cuda_record):
    """Complete fixed-data topology; execution remains caller-authorized."""
    records=[]; outers=[]
    for cell in primary_fixture_matrix():
        for outer in range(R_EQ):
            observed,meta=fixed_observed(cell,outer,namespace)
            support=certify_nb_support(observed,reference_fit(cell.family,observed)["bound"],cell.statistic) if cell.family=="negative_binomial" else None
            cuda=lambda c,s: cuda_adapter(c,s,certified_support=support.indices if support else None,remainder_bound=support.remainder_bound if support else None)
            obs=evaluate_fixed_record(identity=f"{cell.canonical_id}|raw_outer={outer}",record_type="observed",cell=cell,raw_outer_index=outer,raw_inner_index=None,sample=observed,reference_adapter=reference_adapter,cuda_adapter=cuda)
            cpu,attempts,eligible=fixed_bootstraps(cell,observed,outer,namespace); boots=[]
            for item in eligible:
                bs=certify_nb_support(item["sample"],reference_fit(cell.family,item["sample"])["bound"],cell.statistic) if cell.family=="negative_binomial" else None
                cuda_bs=lambda c,s: cuda_adapter(c,s,certified_support=bs.indices if bs else None,remainder_bound=bs.remainder_bound if bs else None)
                row=evaluate_fixed_record(identity=f"{cell.canonical_id}|raw_outer={outer}|raw_inner={item['raw_inner_index']}",record_type="bootstrap",cell=cell,raw_outer_index=outer,raw_inner_index=item["raw_inner_index"],sample=item["sample"],reference_adapter=reference_adapter,cuda_adapter=cuda_bs); boots.append(row)
            records.extend([obs,*boots]); outers.append({"identity":obs["identity"],**aggregate_outer(obs,boots),"raw_attempts":len(attempts)})
    assert_identities([{ "identity": item["identity"]} for item in outers])
    return records,outers


def fixed_observed(cell, raw_outer_index, namespace):
    seed = derive_seed(namespace, cell.canonical_id, raw_outer_index, "outer_observed")
    sample = _generate(cell.family, dict(cell.parameters), cell.n, seed)
    return sample, {"cell_id": cell.canonical_id, "raw_outer_index": raw_outer_index,
                    "seed_identity": seed, "sample_digest_sha256": hashlib.sha256(sample.tobytes()).hexdigest()}


def fixed_bootstraps(cell, observed, raw_outer_index, namespace):
    """Construct once from CPU fitted parameters; both engines receive these objects."""
    cpu = reference_fit(cell.family, observed); attempts=[]; eligible=[]; cap=100*B_EQ if cell.family=="negative_binomial" else B_EQ
    for raw_inner_index in range(cap):
        seed=derive_seed(namespace,cell.canonical_id,raw_outer_index,"inner_bootstrap",raw_inner_index)
        sample=_generate(cell.family,cpu["parameters"],cell.n,seed); record={"raw_inner_index":raw_inner_index,"seed_identity":seed,"sample":sample,"sample_digest":hashlib.sha256(sample.tobytes()).hexdigest()}
        try: reference_fit(cell.family,sample); record["canonical_status"]="ELIGIBLE"; eligible.append(record)
        except Exception: record["canonical_status"]="INELIGIBLE"
        attempts.append(record)
        if len(eligible)==B_EQ: return cpu, attempts, eligible
    raise C2CError("NB retry cap exhaustion")


def _tol(cpu): return max(5e-13, 5e-11*abs(cpu))


def evaluate_fixed_record(*, identity, record_type, cell, raw_outer_index, raw_inner_index, sample,
                          reference_adapter, cuda_adapter):
    """Evaluate the two engines independently on the *same* canonical array."""
    cpu=reference_adapter(cell, sample); cuda=cuda_adapter(cell, sample)
    classification=categorical_agreement(cpu["classification"],cuda["classification"])
    evidence=[]
    # Adapters supply the same fitted-point values; no result is shared between engines.
    for quantity, cpu_values in cpu.get("distribution_values",{}).items():
        cuda_values=cuda.get("distribution_values",{}).get(quantity,())
        if len(cpu_values)!=len(cuda_values):
            evidence.append({"quantity":quantity,"evaluation_point":None,"cpu_value":None,"cuda_value":None,"abs_error":float("inf"),"allowed_tolerance":None,"passed":False})
            continue
        for point,left,right in zip(cpu.get("evaluation_points",()),cpu_values,cuda_values):
            error=abs(right-left); evidence.append({"quantity":quantity,"evaluation_point":point,"cpu_value":left,"cuda_value":right,"abs_error":error,"allowed_tolerance":_tol(left),"passed":distribution_value_agreement(left,right)})
    distribution_pass=bool(evidence) and all(item["passed"] for item in evidence)
    statistic_error=abs(cuda["statistic"]-cpu["statistic"]); statistic_tol=2e-11*max(1,abs(cpu["statistic"])); statistic_pass=math.isfinite(cuda["statistic"]) and statistic_agreement(cpu["statistic"],cuda["statistic"])
    # NB's preregistered flat exception is only considered after downstream gates.
    downstream=classification and distribution_pass and statistic_pass
    kwargs={"cpu_log_likelihood":cpu.get("log_likelihood"),"cuda_log_likelihood":cuda.get("log_likelihood"),"same_eligibility":classification,"downstream_passed":downstream}
    if cell.family=="negative_binomial": kwargs["reference_log_likelihood"]=cpu["reference_log_likelihood"]
    try: fit_pass, flat_used, diagnostic=fit_agreement(cell.family,cpu["parameters"],cuda["parameters"],**kwargs)
    except Exception: fit_pass, flat_used, diagnostic=False, None, None
    return {"identity":identity,"record_type":record_type,"cell_id":cell.canonical_id,"family":cell.family,"n":cell.n,"statistic":cell.statistic,"raw_outer_index":raw_outer_index,"raw_inner_index":raw_inner_index,"sample_digest":hashlib.sha256(sample.tobytes()).hexdigest(),"cpu_classification":cpu["classification"],"cuda_classification":cuda["classification"],"cuda_failure_reason":cuda.get("failure_reason"),"cpu_parameters":cpu["parameters"],"cuda_parameters":cuda["parameters"],"cpu_log_likelihood":cpu.get("log_likelihood"),"cuda_log_likelihood":cuda.get("log_likelihood"),"cpu_statistic":cpu["statistic"],"cuda_statistic":cuda["statistic"],"fit_gate_pass":fit_pass,"classification_gate_pass":classification,"distribution_value_gate_pass":distribution_pass,"statistic_gate_pass":statistic_pass,"distribution_evidence":evidence,"statistic_abs_error":statistic_error,"statistic_allowed_tolerance":statistic_tol,"flat_objective_used":flat_used,"flat_objective_diagnostic":diagnostic}


def aggregate_outer(observed, bootstraps):
    if len(bootstraps)!=B_EQ: raise C2CError("B_EQ eligible bootstrap results required")
    cpu_b,cpu_p=mc_pvalue(observed["cpu_statistic"],[x["cpu_statistic"] for x in bootstraps]); cuda_b,cuda_p=mc_pvalue(observed["cuda_statistic"],[x["cuda_statistic"] for x in bootstraps])
    mc=cpu_b==cuda_b and (cpu_p<=.05)==(cuda_p<=.05)
    required=("classification_gate_pass","fit_gate_pass","distribution_value_gate_pass","statistic_gate_pass")
    passed=all(observed[key] for key in required) and all(all(row[key] for key in required) for row in bootstraps) and mc
    return {"b_cpu":cpu_b,"b_cuda":cuda_b,"p_cpu":cpu_p,"p_cuda":cuda_p,"reject_cpu":cpu_p<=.05,"reject_cuda":cuda_p<=.05,"mc_gate_pass":mc,"outer_gate_pass":passed}


def provisional_global(outer_results):
    """A1 can never claim final PASS: adversarial execution is deliberately absent."""
    if len(outer_results)!=PRIMARY_OUTER_TARGET: return False
    return False


def write_atomic_bundle(output: Path, payloads: dict) -> None:
    """No final directory represents success until all eleven artifacts verify."""
    tmp=output.with_name(output.name+".tmp")
    if output.exists() or tmp.exists(): raise C2CError("output already exists")
    try:
        tmp.mkdir()
        for name,payload in payloads.items():
            if name.endswith(".parquet"): raise C2CError("parquet write requires execution backend")
            if name!="digests.json": (tmp/name).write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        missing=set(REQUIRED_ARTIFACTS)-{p.name for p in tmp.iterdir()}-{"digests.json"}
        if missing: raise C2CError("missing artifact")
        digests={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp.iterdir()}
        if set(digests)!=set(REQUIRED_ARTIFACTS)-{"digests.json"}: raise C2CError("digest schema mismatch")
        (tmp/"digests.json").write_text(json.dumps(digests,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        tmp.replace(output)
    except Exception:
        if tmp.exists(): shutil.rmtree(tmp)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CP05-C2C CUDA equivalence only")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("equivalence", "generator-sanity", "all"), default="equivalence")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--R", type=int, default=R_EQ); parser.add_argument("--B", type=int, default=B_EQ)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.require_gpu: raise SystemExit("--require-gpu is mandatory; no CPU fallback")
    # Dispatch is complete; A2-2 owns final persistence, so no partial PASS exists.
    if args.mode == "equivalence":
        from .a2_artifacts import run_adversarial_suite
        run_adversarial_suite()
        raise SystemExit("artifact-layer-not-complete")
    if args.mode == "generator-sanity":
        raise SystemExit("generator-sanity requires separately authorized Quantum execution")
    if args.mode == "all":
        from .a2_artifacts import run_adversarial_suite
        run_adversarial_suite()
        raise SystemExit("artifact-layer-not-complete")
    raise SystemExit("invalid mode")


if __name__ == "__main__": main()
