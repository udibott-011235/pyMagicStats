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
import tempfile
import math
import subprocess
from pathlib import Path

from .equivalence_preregistration import B_EQ, GENERATOR_SANITY_CASES, GENERATOR_SANITY_N, PRIMARY_CELL_COUNT, R_EQ, REQUIRED_ARTIFACTS, primary_fixture_matrix, initial_summary
from .cp05_cuda_engine import EngineContractError, _cp04_fit, _cp04_statistic, _generate, _parameters, derive_seed, mc_pvalue, nb_eligibility
from . import cuda_candidate
from .nb_support import certify_nb_support
from .equivalence_preregistration import categorical_agreement, distribution_value_agreement, fit_agreement, generator_sanity_check, global_equivalence_pass, statistic_agreement
from .artifact_writers import (write_batch_invariance, write_classification_comparison,
    write_digests, write_environment, write_equivalence_manifest, write_fixture_manifest,
    write_fit_comparison, write_generator_sanity, write_rng_identity,
    write_statistic_comparison, write_summary)
from .artifact_validation import generate_digests, publish_atomic
from .a2_artifacts import NAMES, load_fixtures, run_adversarial_fixture
from .execution_checkpoint import CheckpointError, ExecutionCheckpoint

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


def canonical_distribution_value_points(cell, sample):
    """DEC-020 value grid, independent of DEC-014 statistic tail support."""
    if cell.family == "negative_binomial":
        return tuple(range(int(max(sample)) + 1))
    return tuple(sorted(set(float(x) for x in sample)))


def evaluate_reference_record(cell, sample):
    result=reference_fit(cell.family,sample); bound=result["bound"]
    statistic=_cp04_statistic(sample,bound,cell.family,cell.statistic)
    # Values are deliberately CPU_REFERENCE evidence only.
    points=canonical_distribution_value_points(cell, sample)
    if cell.family=="negative_binomial":
        values={"pmf":[float(bound.pmf(x)) for x in points],"logPMF":[float(bound.logpmf(x)) for x in points],"cdf":[float(bound.cdf(x)) for x in points],"sf":[float(bound.sf(x)) for x in points],"logCDF":[float(bound.logcdf(x)) for x in points],"logSF":[float(bound.logsf(x)) for x in points]}
    else: values={"cdf":[float(bound.cdf(x)) for x in points],"sf":[float(bound.sf(x)) for x in points],"logCDF":[float(bound.logcdf(x)) for x in points],"logSF":[float(bound.logsf(x)) for x in points]}
    record={"classification":"ELIGIBLE","parameters":result["parameters"],"log_likelihood":None,"statistic":statistic,"evaluation_points":points,"distribution_values":values,"bound":bound}
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
        points=canonical_distribution_value_points(cell, sample)
        if cell.family=="negative_binomial" and (certified_support is None or len(certified_support)==0): raise C2CError("uncertified NB tail")
        q=cuda_candidate.distribution_values(cell.family, points, params)
        statistic=float(cuda_candidate.cp.asnumpy(cuda_candidate.candidate_statistic(cell.family,sample,params,cell.statistic,support=certified_support if cell.family=="negative_binomial" else None,remainder_bound=remainder_bound)))
        values={name:[float(x) for x in cuda_candidate.cp.asnumpy(value)] for name,value in q.items()}
        return {"classification":"ELIGIBLE","parameters":params,"log_likelihood":float(cuda_candidate.cp.asnumpy(fitted.get("log_likelihood",cuda_candidate.cp.asarray(float("nan"))))),"statistic":statistic,"evaluation_points":points,"distribution_values":values,"solver_converged":True,"iterations":fitted["iterations"],"dtype":"float64","failure_reason":None}
    except Exception as exc:
        return {"classification":"FAILED","parameters":{},"log_likelihood":None,"statistic":float("nan"),"evaluation_points":[],"distribution_values":{},"solver_converged":False,"iterations":0,"dtype":"float64","failure_reason":str(exc)}


def traverse_primary(namespace, *, reference_adapter=evaluate_reference_record, cuda_adapter=evaluate_cuda_record):
    """Complete fixed-data topology; execution remains caller-authorized."""
    records=[]; outers=[]
    for cell in primary_fixture_matrix():
        for outer in range(R_EQ):
            completed=execute_primary_outer(cell,outer,namespace,reference_adapter=reference_adapter,cuda_adapter=cuda_adapter)
            records.extend(completed["records"]); outers.append(completed["outer"])
    assert_identities([{ "identity": item["identity"]} for item in outers])
    return records,outers


def _cpu_nb_classification(sample) -> str:
    eligible, reason = nb_eligibility(sample)
    return "ELIGIBLE" if eligible else str(reason)


def _cuda_nb_classification(sample) -> str:
    """Independent CUDA eligibility classification; it deliberately does not fit."""
    cp = cuda_candidate.require_cuda()
    codes = cuda_candidate._nb_classification_codes(sample, cp)
    return cuda_candidate._nb_classification_metadata(codes)


def _ineligible_observed_outer(cell, raw_outer_index, observed, meta, cpu_classification, cuda_classification):
    identity = f"{cell.canonical_id}|raw_outer={raw_outer_index}"
    classification_pass = cpu_classification == cuda_classification
    reason = cpu_classification if cpu_classification != "ELIGIBLE" else cuda_classification
    observed_record = {"identity": identity, "record_type": "observed", "cell_id": cell.canonical_id,
                       "family": cell.family, "n": cell.n, "statistic": cell.statistic,
                       "raw_outer_index": raw_outer_index, "raw_inner_index": None,
                       "sample_digest": meta["sample_digest_sha256"], "cpu_classification": cpu_classification,
                       "cuda_classification": cuda_classification, "cuda_failure_reason": None,
                       "cpu_parameters": {}, "cuda_parameters": {}, "cpu_log_likelihood": None,
                       "cuda_log_likelihood": None, "cpu_statistic": None, "cuda_statistic": None,
                       "fit_gate_pass": None, "classification_gate_pass": classification_pass,
                       "distribution_value_gate_pass": None, "statistic_gate_pass": None,
                       "distribution_evidence": [], "statistic_abs_error": None,
                       "statistic_allowed_tolerance": None, "flat_objective_used": None,
                       "flat_objective_diagnostic": None, "observed_ineligible": cpu_classification != "ELIGIBLE",
                       "eligibility_reason": reason}
    aggregate = {"b_cpu": None, "b_cuda": None, "p_cpu": None, "p_cuda": None,
                 "reject_cpu": None, "reject_cuda": None, "mc_gate_pass": None,
                 "mc_evaluable": False, "cuda_mc_unavailable_records": [],
                 "outer_gate_pass": classification_pass, "observed_ineligible": cpu_classification != "ELIGIBLE",
                 "eligibility_reason": reason}
    return {"identity": identity, "cell_id": cell.canonical_id, "raw_outer_index": raw_outer_index,
            "observed_sample_digest": meta["sample_digest_sha256"], "raw_bootstrap_attempts": [],
            "eligible_bootstrap_identities": [], "records": [observed_record], "outer": {"identity": identity, **aggregate}}


def execute_primary_outer(cell, raw_outer_index, namespace, *, reference_adapter=evaluate_reference_record,
                          cuda_adapter=evaluate_cuda_record, cpu_nb_classifier=_cpu_nb_classification,
                          cuda_nb_classifier=_cuda_nb_classification):
    """Execute exactly one canonical outer identity for checkpointing or traversal."""
    observed, meta = fixed_observed(cell, raw_outer_index, namespace)
    if cell.family == "negative_binomial":
        cpu_classification = cpu_nb_classifier(observed)
        cuda_classification = cuda_nb_classifier(observed)
        if cpu_classification != "ELIGIBLE" or cuda_classification != "ELIGIBLE":
            return _ineligible_observed_outer(cell, raw_outer_index, observed, meta, cpu_classification, cuda_classification)
    support = certify_nb_support(observed, reference_fit(cell.family, observed)["bound"], cell.statistic) if cell.family == "negative_binomial" else None
    cuda = lambda c, s: cuda_adapter(c, s, certified_support=support.indices if support else None, remainder_bound=support.remainder_bound if support else None)
    identity = f"{cell.canonical_id}|raw_outer={raw_outer_index}"
    observed_record = evaluate_fixed_record(identity=identity, record_type="observed", cell=cell, raw_outer_index=raw_outer_index, raw_inner_index=None, sample=observed, reference_adapter=reference_adapter, cuda_adapter=cuda)
    _, attempts, eligible = fixed_bootstraps(cell, observed, raw_outer_index, namespace)
    boots = []
    for item in eligible:
        certification = certify_nb_support(item["sample"], reference_fit(cell.family, item["sample"])["bound"], cell.statistic) if cell.family == "negative_binomial" else None
        cuda_bootstrap = lambda c, s: cuda_adapter(c, s, certified_support=certification.indices if certification else None, remainder_bound=certification.remainder_bound if certification else None)
        boots.append(evaluate_fixed_record(identity=f"{identity}|raw_inner={item['raw_inner_index']}", record_type="bootstrap", cell=cell, raw_outer_index=raw_outer_index, raw_inner_index=item["raw_inner_index"], sample=item["sample"], reference_adapter=reference_adapter, cuda_adapter=cuda_bootstrap))
    raw_attempts = [{key: value for key, value in item.items() if key != "sample"} for item in attempts]
    eligible_identities = [{"raw_inner_index": item["raw_inner_index"], "seed_identity": item["seed_identity"], "sample_digest": item["sample_digest"]} for item in eligible]
    return {"identity": identity, "cell_id": cell.canonical_id, "raw_outer_index": raw_outer_index,
            "observed_sample_digest": meta["sample_digest_sha256"], "raw_bootstrap_attempts": raw_attempts,
            "eligible_bootstrap_identities": eligible_identities, "records": [observed_record, *boots],
            "outer": {"identity": identity, **aggregate_outer(observed_record, boots), "raw_attempts": len(attempts)}}


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
        except EngineContractError as exc:
            if not str(exc).startswith("NB_NOT_ASSESSED:"):
                raise
            record["canonical_status"]="INELIGIBLE"
        attempts.append(record)
        if len(eligible)==B_EQ: return cpu, attempts, eligible
    raise C2CError("NB retry cap exhaustion")


def _tol(cpu): return max(5e-13, 5e-11*abs(cpu))


def required_distribution_quantities(family):
    """DEC-021: the family contract, never adapter output, owns the universe."""
    if family == "negative_binomial":
        return ("pmf", "logPMF", "cdf", "sf", "logCDF", "logSF")
    if family in {"gamma", "exponential"}:
        return ("cdf", "sf", "logCDF", "logSF")
    raise C2CError("unknown distribution-value family")


def evaluate_fixed_record(*, identity, record_type, cell, raw_outer_index, raw_inner_index, sample,
                          reference_adapter, cuda_adapter):
    """Evaluate the two engines independently on the *same* canonical array."""
    cpu=reference_adapter(cell, sample); cuda=cuda_adapter(cell, sample)
    classification=categorical_agreement(cpu["classification"],cuda["classification"])
    evidence=[]
    # Point identity precedes numerical comparison; no result is shared between engines.
    cpu_points=tuple(cpu.get("evaluation_points",()))
    cuda_points=tuple(cuda.get("evaluation_points",()))
    points_match=cpu_points==cuda_points
    required=required_distribution_quantities(cell.family)
    cpu_values=cpu.get("distribution_values",{})
    cuda_values=cuda.get("distribution_values",{})
    cpu_quantities=set(cpu_values); cuda_quantities=set(cuda_values)
    quantities_match=cpu_quantities==set(required) and cuda_quantities==set(required)

    def structural_failure(quantity, reason, **details):
        evidence.append({"quantity":quantity,"evaluation_point":None,"cpu_value":None,"cuda_value":None,"abs_error":float("inf"),"allowed_tolerance":None,"passed":False,"failure_reason":reason,**details})

    if not points_match:
        for quantity in required:
            structural_failure(quantity,"EVALUATION_POINT_IDENTITY_MISMATCH")
    for quantity in required:
        missing_cpu=quantity not in cpu_quantities
        missing_cuda=quantity not in cuda_quantities
        if missing_cpu or missing_cuda:
            engine="BOTH" if missing_cpu and missing_cuda else "CPU" if missing_cpu else "CUDA"
            structural_failure(quantity,f"MISSING_{engine}_DISTRIBUTION_QUANTITY")
    for quantity in sorted((cpu_quantities|cuda_quantities)-set(required)):
        engines=[engine for engine, quantities in (("CPU",cpu_quantities),("CUDA",cuda_quantities)) if quantity in quantities]
        structural_failure(quantity,"UNEXPECTED_DISTRIBUTION_QUANTITY",engines=engines)
    if points_match and quantities_match:
        for quantity in required:
            if len(cpu_values[quantity])!=len(cpu_points) or len(cuda_values[quantity])!=len(cpu_points):
                structural_failure(quantity,"DISTRIBUTION_VALUE_LENGTH_MISMATCH")
    # Complete structural preflight precedes every numerical comparison.
    if not evidence:
        for quantity in required:
            for point,left,right in zip(cpu_points,cpu_values[quantity],cuda_values[quantity]):
                error=abs(right-left); evidence.append({"quantity":quantity,"evaluation_point":point,"cpu_value":left,"cuda_value":right,"abs_error":error,"allowed_tolerance":_tol(left),"passed":distribution_value_agreement(left,right)})
    distribution_pass=points_match and quantities_match and bool(evidence) and all(item["passed"] for item in evidence)
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
    cpu_b,cpu_p=mc_pvalue(observed["cpu_statistic"],[x["cpu_statistic"] for x in bootstraps])
    cuda_records=[observed,*bootstraps]
    unavailable=[]
    for record in cuda_records:
        if record.get("cuda_classification") != "ELIGIBLE" or not math.isfinite(record.get("cuda_statistic", float("nan"))):
            unavailable.append({"identity":record.get("identity"), "record_type":record.get("record_type"),
                                "raw_inner_index":record.get("raw_inner_index"),
                                "cuda_classification":record.get("cuda_classification"),
                                "cuda_failure_reason":record.get("cuda_failure_reason")})
    if unavailable:
        cuda_b=cuda_p=reject_cuda=None; mc=False; mc_evaluable=False
    else:
        cuda_b,cuda_p=mc_pvalue(observed["cuda_statistic"],[x["cuda_statistic"] for x in bootstraps])
        reject_cuda=cuda_p<=.05; mc=cpu_b==cuda_b and (cpu_p<=.05)==reject_cuda; mc_evaluable=True
    required=("classification_gate_pass","fit_gate_pass","distribution_value_gate_pass","statistic_gate_pass")
    passed=all(observed[key] for key in required) and all(all(row[key] for key in required) for row in bootstraps) and mc
    return {"b_cpu":cpu_b,"b_cuda":cuda_b,"p_cpu":cpu_p,"p_cuda":cuda_p,"reject_cpu":cpu_p<=.05,"reject_cuda":reject_cuda,"mc_gate_pass":mc,"mc_evaluable":mc_evaluable,"cuda_mc_unavailable_records":unavailable,"outer_gate_pass":passed}


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


def _fit_rows(records):
    return [{"identity": r["identity"], "cell_id": r.get("cell_id"), "family": r.get("family"),
             "n": r.get("n"), "statistic": r.get("statistic"), "raw_outer_index": r.get("raw_outer_index"),
             "raw_inner_index": r.get("raw_inner_index"), "record_type": r.get("record_type"),
             "cpu_classification": r.get("cpu_classification"), "cuda_classification": r.get("cuda_classification"),
             "classification_match": r.get("classification_gate_pass"),
             "cpu_parameters_json": json.dumps(r.get("cpu_parameters", {}), sort_keys=True),
             "cuda_parameters_json": json.dumps(r.get("cuda_parameters", {}), sort_keys=True),
             "cpu_log_likelihood": r.get("cpu_log_likelihood"), "cuda_log_likelihood": r.get("cuda_log_likelihood"),
             "fit_gate_pass": r.get("fit_gate_pass"), "flat_objective_used": r.get("flat_objective_used"),
             "flat_objective_diagnostic_json": json.dumps(r.get("flat_objective_diagnostic"), sort_keys=True),
             "cuda_failure_reason": r.get("cuda_failure_reason"), "nb_support_stop": None,
             "nb_support_size": None, "nb_remainder_bound": None, "nb_required_bound": None} for r in records]


def _stat_rows(records):
    return [{"identity": r["identity"], "cell_id": r.get("cell_id"), "family": r.get("family"),
             "statistic": r.get("statistic"), "raw_outer_index": r.get("raw_outer_index"),
             "raw_inner_index": r.get("raw_inner_index"), "record_type": r.get("record_type"),
             "cpu_statistic": r.get("cpu_statistic"), "cuda_statistic": r.get("cuda_statistic"),
             "abs_error": r.get("statistic_abs_error"), "allowed_tolerance": r.get("statistic_allowed_tolerance"),
             "gate_pass": r.get("statistic_gate_pass")} for r in records]


def _classification_rows(records):
    return [{"identity": r["identity"], "cell_id": r.get("cell_id"),
             "raw_outer_index": r.get("raw_outer_index"), "raw_inner_index": r.get("raw_inner_index"),
             "record_type": r.get("record_type"), "cpu_classification": r.get("cpu_classification"),
             "cuda_classification": r.get("cuda_classification"), "exact_match": r.get("classification_gate_pass")} for r in records]


def validate_required_workload_complete(*, mode: str, outer_results: list[dict], adversarial: list[dict],
                                        batch_passed: bool | None, generator_results: list[dict] | None) -> None:
    """Reject incomplete workloads before any final artifact directory is published."""
    if mode not in {"equivalence", "generator-sanity", "all"}:
        raise C2CError("unsupported mode")
    if mode in {"equivalence", "all"}:
        identities = [row.get("identity") for row in outer_results]
        if len(outer_results) != PRIMARY_OUTER_TARGET:
            raise C2CError("incomplete primary outer workload")
        if len(set(identities)) != PRIMARY_OUTER_TARGET or None in identities:
            raise C2CError("duplicate primary outer identity")
        fixture_names = [row.get("fixture_name") for row in adversarial]
        if len(adversarial) != len(NAMES) or set(fixture_names) != set(NAMES):
            raise C2CError("incomplete adversarial workload")
        if not isinstance(batch_passed, bool):
            raise C2CError("missing batch invariance workload")
    if mode in {"generator-sanity", "all"}:
        values = list(generator_results or [])
        identities = [row.get("identity") for row in values]
        if len(values) != len(GENERATOR_SANITY_CASES) or len(set(identities)) != len(GENERATOR_SANITY_CASES) or None in identities:
            raise C2CError("incomplete generator sanity workload")


def publish_execution_bundle(output: Path, *, mode: str, git_sha: str, records: list[dict],
                             outer_results: list[dict], adversarial: list[dict],
                             batch_passed: bool, rng_identities: list[dict],
                             generator_sanity_passed: bool = False, generator_results: list[dict] | None = None,
                             adversarial_observed: int | None = None) -> dict:
    """The sole C2C artifact path. Scientific false gates remain publishable evidence.

    It is deliberately independent of the CUDA implementation: callers supply already
    evaluated records and this layer only serializes, validates and atomically publishes.
    """
    validate_required_workload_complete(mode=mode, outer_results=outer_results, adversarial=adversarial,
                                        batch_passed=batch_passed, generator_results=generator_results)
    fixtures, digest = load_fixtures()
    if len(adversarial) != 14: raise C2CError("fourteen adversarial fixture results required")
    equivalence = mode not in {"equivalence", "all"} or (all(x.get("outer_gate_pass") for x in outer_results) and all(x.get("overall_fixture_pass") for x in adversarial) and batch_passed)
    generator_sanity_passed = all(row.get("passed") is True for row in (generator_results or [])) if mode in {"generator-sanity", "all"} else False
    overall = equivalence and generator_sanity_passed if mode == "all" else (generator_sanity_passed if mode == "generator-sanity" else equivalence)
    reasons=[]
    if not equivalence and mode in {"equivalence", "all"}: reasons.append("equivalence_gate_failed")
    if not generator_sanity_passed and mode in {"generator-sanity", "all"}: reasons.append("generator_sanity_gate_failed")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=output.name + ".stage-") as staging_root:
        stage=Path(staging_root) / "bundle"; stage.mkdir()
        write_equivalence_manifest(stage/"equivalence_manifest.json", baseline_sha=BASELINE_SHA, git_sha=git_sha, mode=mode, fixture_file_sha256=digest)
        write_fixture_manifest(stage/"fixture_manifest.json", [], [], adversarial)
        write_fit_comparison(stage/"fit_comparison.parquet", _fit_rows(records))
        write_statistic_comparison(stage/"statistic_comparison.parquet", _stat_rows(records))
        write_classification_comparison(stage/"classification_comparison.parquet", _classification_rows(records))
        write_batch_invariance(stage/"batch_invariance.json", [], {"mode": mode}, passed=batch_passed)
        write_rng_identity(stage/"rng_identity.json", rng_identities)
        write_generator_sanity(stage/"generator_sanity.json", passed=generator_sanity_passed, results=generator_results)
        write_environment(stage/"environment.json", git_sha)
        observed_adversarial = len(adversarial) if mode in {"equivalence", "all"} else 0
        write_summary(stage/"summary.json", execution_mode=mode, primary_outer_observed=len(outer_results), adversarial_fixture_observed=observed_adversarial if adversarial_observed is None else adversarial_observed, equivalence_gate_passed=equivalence, generator_sanity_passed=generator_sanity_passed, overall_pass=overall, batch_invariance_passed=batch_passed, artifact_validation_passed=True, failure_reasons=reasons)
        generate_digests(stage)
        publish_atomic(stage, output)
    return json.loads((output/"summary.json").read_text(encoding="utf-8"))


def _require_cuda() -> None:
    cuda_candidate.require_cuda()


def _execution_sha() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except FileNotFoundError as exc:
        raise C2CError("git executable not available in PATH") from exc


def _checkpoint_contract(args) -> dict:
    return {"execution_sha": _execution_sha(), "schema": "cp05-c2c-v1", "dec016": "DEC-016",
            "mode": args.mode, "namespace": args.namespace, "R_EQ": R_EQ, "B_EQ": B_EQ,
            "primary_cell_ids": [cell.canonical_id for cell in primary_fixture_matrix()],
            "fixture_digest": fixture_digest(), "float_precision": "float64"}


def _progress(kind: str, complete: int, total: int) -> None:
    print(f"C2C_PROGRESS {kind}={complete}/{total}", flush=True)


def _execute_equivalence(args, checkpoint: ExecutionCheckpoint) -> tuple[list[dict], list[dict], list[dict], bool, list[dict]]:
    primary = checkpoint.data["primary"]
    print(f"C2C_RESUME recovered_primary={len(primary)}", flush=True)
    print(f"C2C_RESUME remaining_primary={PRIMARY_OUTER_TARGET-len(primary)}", flush=True)
    for cell in primary_fixture_matrix():
        for raw_outer_index in range(R_EQ):
            identity = f"{cell.canonical_id}|raw_outer={raw_outer_index}"
            if not checkpoint.has_primary(identity):
                checkpoint.store_primary(identity, execute_primary_outer(cell, raw_outer_index, args.namespace))
                _progress("primary", len(checkpoint.data["primary"]), PRIMARY_OUTER_TARGET)
    fixtures, digest = load_fixtures()
    for name in NAMES:
        if name not in checkpoint.data["adversarial"]:
            checkpoint.store_adversarial(name, run_adversarial_fixture(name, fixtures[name], digest))
            _progress("adversarial", len(checkpoint.data["adversarial"]), len(NAMES))
    if checkpoint.data["batch_invariance"] is None:
        identities = sorted(checkpoint.data["primary"])
        checkpoint.store_batch({"passed": len(identities) == PRIMARY_OUTER_TARGET and len(identities) == len(set(identities)), "identities": identities})
    rows = [row for item in checkpoint.data["primary"].values() for row in item["records"]]
    outer = [item["outer"] for item in checkpoint.data["primary"].values()]
    adversarial = [checkpoint.data["adversarial"][name] for name in NAMES]
    rng = [{"canonical_cell_id": item["cell_id"], "raw_outer_index": item["raw_outer_index"], "purpose": "outer_observed", "raw_inner_index": None} for item in checkpoint.data["primary"].values()]
    return rows, outer, adversarial, checkpoint.data["batch_invariance"]["passed"], rng


def _execute_generator_sanity(args, checkpoint: ExecutionCheckpoint) -> tuple[list[dict], bool]:
    results = checkpoint.data["generator"]
    for index, (family, parameters) in enumerate(GENERATOR_SANITY_CASES):
        identity = f"{family}|{json.dumps(parameters, sort_keys=True)}"
        if identity not in results:
            seed = derive_seed(args.namespace, identity, index, "generator_sanity")
            sample = cuda_candidate.generate(family, parameters, GENERATOR_SANITY_N, seed)
            check = generator_sanity_check(float(cuda_candidate.cp.asnumpy(cuda_candidate.cp.mean(sample))), float(cuda_candidate.cp.asnumpy(cuda_candidate.cp.var(sample, ddof=1))), family, parameters)
            checkpoint.store_generator(identity, {"identity": identity, "seed_identity": seed, "family": family, "parameters": parameters, **check})
            _progress("generator", len(checkpoint.data["generator"]), len(GENERATOR_SANITY_CASES))
    values = [results[f"{family}|{json.dumps(parameters, sort_keys=True)}"] for family, parameters in GENERATOR_SANITY_CASES]
    return values, all(value["passed"] for value in values)


def _official_dispatch(args) -> int:
    """Official noninteractive C2C workload, guarded by CUDA and checkpoint identity."""
    _require_cuda()
    checkpoint = ExecutionCheckpoint.open(args.output, _checkpoint_contract(args), resume=args.resume)
    try:
        records: list[dict] = []; outer: list[dict] = []; batch = False; rng: list[dict] = []
        adversarial = [{"fixture_name": name, "overall_fixture_pass": False, "execution_skipped": True} for name in NAMES]
        generator_results: list[dict] = []; generator_pass = False
        if args.mode in {"equivalence", "all"}:
            records, outer, adversarial, batch, rng = _execute_equivalence(args, checkpoint)
        if args.mode in {"generator-sanity", "all"}:
            generator_results, generator_pass = _execute_generator_sanity(args, checkpoint)
        publish_execution_bundle(args.output, mode=args.mode, git_sha=checkpoint.contract["execution_sha"], records=records, outer_results=outer, adversarial=adversarial, batch_passed=batch, rng_identities=rng, generator_sanity_passed=generator_pass, generator_results=generator_results)
        checkpoint.close_after_publication()
        return 0
    except Exception:
        # Scientific failures are records and publish normally; only software failures reach here.
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CP05-C2C CUDA equivalence only")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("equivalence", "generator-sanity", "all"), default="equivalence")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--namespace", default="CP05-C2C", help="frozen public execution namespace")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--R", type=int, default=R_EQ); parser.add_argument("--B", type=int, default=B_EQ)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.require_gpu: raise SystemExit("--require-gpu is mandatory; no CPU fallback")
    if args.R != R_EQ or args.B != B_EQ: raise SystemExit("C2C rejects calibration-sized R/B")
    try:
        return _official_dispatch(args)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__": main()
