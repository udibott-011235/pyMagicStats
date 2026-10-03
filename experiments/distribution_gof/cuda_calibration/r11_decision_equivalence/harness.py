"""One-shot frozen-payload orchestration; CLI static mode never starts science."""
from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

from ..r11_reference_workload.codec import SamplePayload, require, strict_json
from .adjudication import adjudicate_outer, record_checks, summarize
from .artifacts import EvidenceBundle, encode, snapshot
from .boundary_fixtures import evaluate_fixtures
from .contract import (ALPHA, B_R11, BUILDER_SHA, BUILDER_TREE, GATES,
                       REFERENCE_WORKLOAD_SHA256, SOURCE_OUTERS, TOTAL_RECORDS)
from .preflight import ordered_records, prepare, repository_identity
from .runtime import CanonicalRuntime


class ExecutionEvidenceError(RuntimeError):
    def __init__(self, message, consumed, output):
        super().__init__(message)
        self.consumed = consumed
        self.output = output


@dataclass
class _State:
    records: list[bytes] = field(default_factory=list)
    outers: list[bytes] = field(default_factory=list)
    consumed: bool = False
    failure: dict | None = None


def _error(exc, stage):
    return {"stage": stage, "error_type": type(exc).__name__, "message": str(exc)}


def _failure_record(item, capture, exc):
    cpu, cuda = capture.get("raw_cpu_result") or {}, capture.get("raw_cuda_result") or {}
    return {
        **item, **{k: v for k, v in capture.items() if k != "phase"},
        "raw_cpu_result": capture.get("raw_cpu_result"),
        "raw_cuda_result": capture.get("raw_cuda_result"),
        **{f"{engine}_{field}": result.get(source) for engine, result in (("cpu", cpu), ("cuda", cuda))
           for field, source in (("classification", "classification"), ("parameters", "parameters"),
                                 ("log_likelihood", "log_likelihood"), ("statistic", "statistic"))},
        **{gate: False for gate in GATES}, "distribution_evidence": [],
        "evaluation_points": [], "flat_objective_used": None, "flat_objective_diagnostic": None,
        "cuda_solver_converged": cuda.get("solver_converged"),
        "cuda_failure_reason": (cuda.get("failure_reason") or str(exc)
                                if capture.get("phase") == "cuda" else cuda.get("failure_reason")),
        "structural_failure": _error(exc, capture.get("phase", "record")),
    }


def _check_topology(workload):
    require(len(workload["outers"]) == SOURCE_OUTERS, "exactly 12 frozen outers required")
    require(len({o["outer_identity"] for o in workload["outers"]}) == SOURCE_OUTERS,
            "duplicate frozen outer")
    require(all(len(o["accepted"]) == B_R11 for o in workload["outers"]),
            "exactly 199 frozen accepted bootstraps required")


def _verify_checkout(receipt):
    identity = repository_identity(receipt.repository)
    frozen = receipt.identity()
    require(all(identity[k] == frozen[k] for k in ("R11_HARNESS_SHA", "R11_HARNESS_TREE")),
            "executing harness identity changed")
    return identity


def _walk(workload, runtime, bundle, state, before_cuda):
    current = []
    expected = []
    for outer, item, serialized in ordered_records(workload):
        capture = {"phase": "deserialize", "family": item["cell_id"].split("|", 1)[0]}
        try:
            payload = SamplePayload.from_serialized(serialized)
            sample = payload.deserialize()
            record = runtime.evaluate(item, sample, payload, capture, before_cuda)
        except BaseException as exc:
            data = snapshot(_failure_record(item, capture, exc))
            state.records.append(data)
            bundle.append("records.jsonl", data)
            raise
        data = snapshot(record)
        state.records.append(data)
        bundle.append("records.jsonl", data)
        immutable_record = strict_json(data, canonical=True)
        require(record_checks(immutable_record, item)["valid"], "record gate/provenance failure")
        current.append(immutable_record)
        expected.append(item)
        if len(current) == B_R11 + 1:
            result = adjudicate_outer(current, expected, outer["outer_identity"])
            state.outers.append(snapshot(result))
            for row in result["indicator_adjudication"]:
                bundle.append("indicator_adjudication.jsonl",
                              snapshot({"outer_identity": outer["outer_identity"], **row}))
            require(result["R11_OUTER_PASS"], "outer decision-equivalence failure: "
                    + ", ".join(result["failures"]))
            current, expected = [], []
    require(not current, "incomplete outer traversal")


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "UNAVAILABLE"


def execute(*, workload, r4_archive, r4_crossings, output, require_gpu):
    """Future scientific entry point. Implementation tests replace all runtime dependencies."""
    require(require_gpu is True, "--require-gpu mandatory; CPU fallback prohibited")
    receipt = prepare(workload, r4_archive, r4_crossings)
    frozen = receipt.workload()
    _check_topology(frozen)
    fixtures = evaluate_fixtures()
    require(len(fixtures) == 12 and all(f["fixture_pass"] for f in fixtures),
            "boundary fixture preflight failure")
    runtime = CanonicalRuntime(receipt.repository)
    runtime.require_gpu()
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "numpy": _version("numpy"), "scipy": _version("scipy"),
                   **runtime.environment()}
    _verify_checkout(receipt)
    bundle = EvidenceBundle(output, receipt.repository)
    state = _State()

    def consume():
        if not state.consumed:
            _verify_checkout(receipt)
            bundle.publish("authorization_consumed.json", {
                "EXECUTION_AUTHORIZATION_CONSUMED": True,
                "R11_HARNESS_SHA": receipt.identity()["R11_HARNESS_SHA"],
                "trigger": "first CUDA scientific evaluation"})
            state.consumed = True

    try:
        _walk(frozen, runtime, bundle, state, consume)
        runtime.verify_sources()
        _verify_checkout(receipt)
    except BaseException as exc:
        state.failure = _error(exc, "scientific_traversal")
    try:
        records = [strict_json(data, canonical=True) for data in state.records]
        outers = [strict_json(data, canonical=True) for data in state.outers]
        expected = [item for _, item, _ in ordered_records(frozen)]
        summary = summarize(outers, records, expected, fixtures, accepted=True,
                            consumed=state.consumed, failure=state.failure)
        manifest = {
            **receipt.identity(), "alpha": ALPHA, "B_R11": B_R11, "source_outer_count": SOURCE_OUTERS,
            "required_records": TOTAL_RECORDS, "EXECUTION_AUTHORIZATION_CONSUMED": state.consumed,
            "SAMPLE_REGENERATION": False, "CUDA_CPU_FALLBACK": False,
            "AUTO_RERUN": False, "AUTO_RESUME": False, "CHECKPOINT_RESUME": False,
            "failure": state.failure,
        }
        bundle.publish("execution_manifest.json", manifest)
        bundle.publish("reference_workload.json", receipt.workload_bytes, raw=True)
        bundle.publish("outer_results.json", outers)
        bundle.publish("boundary_fixtures.json", fixtures)
        bundle.publish("environment.json", environment)
        bundle.publish("summary.json", summary)
        bundle.finish_digests()
        return summary
    except BaseException as exc:
        raise ExecutionEvidenceError(str(exc), state.consumed, str(bundle.output)) from exc
    finally:
        bundle.close()


def validate_static():
    require((B_R11, ALPHA, SOURCE_OUTERS, TOTAL_RECORDS) == (199, 0.05, 12, 2400),
            "frozen R11 dimensions changed")
    fixtures = evaluate_fixtures()
    require(len(fixtures) == 12 and all(f["fixture_pass"] for f in fixtures),
            "logical boundary fixtures failed")
    return {
        "STATIC_VALIDATION_PASS": True, "B_R11": B_R11, "ALPHA": ALPHA,
        "R11_BUILDER_SHA": BUILDER_SHA, "R11_BUILDER_TREE": BUILDER_TREE,
        "R11_REFERENCE_WORKLOAD_SHA256": REFERENCE_WORKLOAD_SHA256,
        "boundary_fixture_count": 12, "boundary_fixture_expected_dispositions": True,
        "REAL_R11_WORKLOAD_LOADED": False, "REAL_R11_SAMPLES_EVALUATED": False,
        "GPU_CUDA_SCIENTIFIC_EXECUTED": False,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-static", action="store_true")
    modes.add_argument("--execute", action="store_true")
    for flag in ("workload", "r4-archive", "r4-crossings", "output"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = (args.workload, args.r4_archive, args.r4_crossings, args.output)
    if args.validate_static:
        if any(p is not None for p in paths) or args.require_gpu:
            parser.error("--validate-static accepts no execution arguments")
        result = validate_static()
    else:
        if any(p is None for p in paths) or not args.require_gpu:
            parser.error("--execute requires all four paths and --require-gpu")
        try:
            result = execute(workload=args.workload, r4_archive=args.r4_archive,
                             r4_crossings=args.r4_crossings, output=args.output,
                             require_gpu=args.require_gpu)
        except Exception as exc:
            result = {"R11_GLOBAL_PASS": False,
                      "EXECUTION_AUTHORIZATION_CONSUMED": getattr(exc, "consumed", False),
                      "failure": _error(exc, "evidence_publication" if isinstance(
                          exc, ExecutionEvidenceError) else "preflight")}
            if isinstance(exc, ExecutionEvidenceError):
                result["evidence_directory"] = exc.output
    print(encode(result).decode("utf-8"))
    return 0 if result.get("STATIC_VALIDATION_PASS") or result.get("R11_GLOBAL_PASS") else 2
