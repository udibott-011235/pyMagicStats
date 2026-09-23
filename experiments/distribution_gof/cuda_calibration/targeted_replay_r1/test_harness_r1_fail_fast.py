"""External stdlib-only orchestration tests; scientific imports are blocked."""
import copy
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
BLOCKED_ROOTS = {"cupy", "cupyx", "numpy", "scipy", "experiments", "pyMagicStats", "pymagicstats"}


class NoScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED_ROOTS:
            raise AssertionError(f"Scientific import forbidden in mock tests: {fullname}")
        return None


sys.meta_path.insert(0, NoScientificImports())
spec = importlib.util.spec_from_file_location(
    "external_harness_r1", Path(__file__).with_name("targeted_gpu_replay.py")
)
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)


FAILURES = (
    ("classification_gate_pass", False, "CLASSIFICATION_MISMATCH"),
    ("cuda_failure_reason", "CUDA solver non-convergence", "CUDA_NONCONVERGENCE"),
    ("fit_gate_pass", False, "FIT_GATE_FAILURE"),
    ("distribution_value_gate_pass", False, "DISTRIBUTION_VALUE_GATE_FAILURE"),
    ("statistic_gate_pass", False, "STATISTIC_GATE_FAILURE"),
    ("fit_gate_pass", None, "UNEXPLAINED_DISCREPANCY"),
)


def passing_record(identity, record_type):
    return {
        "identity": identity, "record_type": record_type,
        "classification_gate_pass": True, "fit_gate_pass": True,
        "distribution_value_gate_pass": True, "statistic_gate_pass": True,
        "cuda_failure_reason": None,
    }


class HarnessR1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="harness-r1-mocks-")
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.manifest, _ = h.load_manifest()
        self.state = h.initial_state()

    def evidence_at_exception(self, filename, check):
        """Inspect real persisted JSONL during exception construction, before raise."""
        seen = []
        original_error = h.HarnessContractError
        test = self

        class EvidenceCheckedError(original_error):
            def __init__(self, message):
                path = test.output / filename
                test.assertTrue(path.is_file(), "evidence must exist BEFORE STOP")
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                check(rows)
                seen.append(str(message))
                super().__init__(message)

        return patch.object(h, "HarnessContractError", EvidenceCheckedError), seen

    def prepare_b(self, failure_position=None, failure=None, mc="pass", outer_count=2):
        manifest = {"mc_failed_outers": self.manifest["mc_failed_outers"][:outer_count]}
        cells = {row["cell_id"]: SimpleNamespace(canonical_id=row["cell_id"]) for row in manifest["mc_failed_outers"]}
        runner = SimpleNamespace(
            fixed_observed=Mock(return_value=("mock-observed", {})),
            _cpu_nb_classification=Mock(return_value="ELIGIBLE"),
            _cuda_nb_classification=Mock(return_value="ELIGIBLE"),
            _ineligible_observed_outer=Mock(side_effect=AssertionError("unexpected ineligible path")),
        )

        def bootstrap_sequence(cell, observed, raw_outer, namespace):
            outer = next(row for row in manifest["mc_failed_outers"]
                         if row["cell_id"] == cell.canonical_id and row["raw_outer_index"] == raw_outer)
            eligible = [{**item, "seed_identity": "mock-seed", "sample_digest": "mock-digest",
                         "sample": "mock-bootstrap"} for item in outer["bootstrap_identities"]]
            return {}, eligible, eligible

        runner.fixed_bootstraps = Mock(side_effect=bootstrap_sequence)
        aggregate = {
            "b_cpu": 4, "b_cuda": 5 if mc in {"exceedance", "both"} else 4,
            "p_cpu": 0.3, "p_cuda": 0.4,
            "reject_cpu": False, "reject_cuda": mc in {"reject", "both"},
            "mc_evaluable": True, "cuda_mc_unavailable_records": [], "outer_gate_pass": mc == "pass",
        }
        runner.aggregate_outer = Mock(return_value=aggregate)
        calls = []

        def evaluate(runtime, **kwargs):
            position = len(calls)
            calls.append(kwargs["identity"])
            record = passing_record(kwargs["identity"], kwargs["record_type"])
            if position == failure_position:
                field, value, _ = failure
                record[field] = value
            return record

        return SimpleNamespace(runner=runner), manifest, cells, calls, evaluate

    def test_a_first_failure_stops_and_persists_each_named_counter(self):
        for field, value, counter in FAILURES:
            with self.subTest(counter=counter):
                case_output = self.output / counter
                case_output.mkdir()
                original_output = self.output
                self.output = case_output
                state = h.initial_state()
                records = self.manifest["persisted_failure_record_identities"]
                failing = passing_record(records[0]["identity"], records[0]["record_type"])
                failing[field] = value
                evaluator = Mock(return_value=failing)

                def check(rows):
                    self.assertEqual(rows, [failing])
                    self.assertEqual(state[counter], 1)
                    self.assertEqual(state["WORKLOAD_A_RECORDS_EXECUTED"], 1)

                error_patch, seen = self.evidence_at_exception("workload_a_records.jsonl", check)
                with error_patch, patch.object(h, "workload_a_record", evaluator):
                    with self.assertRaisesRegex(h.HarnessContractError, "Workload A record discrepancy"):
                        h.run_workload_a(None, self.manifest, {r["cell_id"]: object() for r in records}, self.output, state)
                self.assertEqual(evaluator.call_count, 1)
                self.assertEqual(len(seen), 1)
                self.output = original_output

    def test_a_existing_global_failure_does_not_trigger_current_record_delta(self):
        self.state["FIT_GATE_FAILURE"] = 7
        records = self.manifest["persisted_failure_record_identities"]
        with patch.object(h, "workload_a_record", side_effect=lambda runtime, cell, item:
                          passing_record(item["identity"], item["record_type"])) as evaluator:
            h.run_workload_a(None, self.manifest, {r["cell_id"]: object() for r in records}, self.output, self.state)
        self.assertEqual(evaluator.call_count, 134)
        self.assertEqual(self.state["FIT_GATE_FAILURE"], 7)

    def test_b_observed_failure_stops_before_bootstrap_evaluation_for_all_counters(self):
        for failure in FAILURES:
            with self.subTest(counter=failure[2]):
                self.check_b_record_failure(0, failure)

    def test_b_each_bootstrap_position_stops_before_next_for_all_counters(self):
        for position in range(1, 16):
            for failure in FAILURES:
                with self.subTest(bootstrap_position=position, counter=failure[2]):
                    self.check_b_record_failure(position, failure)

    def check_b_record_failure(self, position, failure):
        # Unique directories keep each partial-evidence assertion independent.
        previous_output = self.output
        self.output = previous_output / f"record-{position}-{failure[2]}"
        self.output.mkdir()
        runtime, manifest, cells, calls, evaluate = self.prepare_b(position, failure)
        state = h.initial_state()
        outer = manifest["mc_failed_outers"][0]
        expected = [outer["outer_identity"], *[r["identity"] for r in outer["bootstrap_identities"]]][:position + 1]

        def check(rows):
            self.assertEqual(len(rows), 1)
            self.assertEqual([r["identity"] for r in rows[0]["record_level_results"]], expected)
            self.assertEqual(rows[0]["record_level_results"][-1][failure[0]], failure[1])
            self.assertEqual(state[failure[2]], 1)
            self.assertEqual(state["WORKLOAD_B_RECORDS_EXECUTED"], position + 1)
            self.assertNotIn("b_cpu", rows[0])
            runtime.runner.aggregate_outer.assert_not_called()

        error_patch, seen = self.evidence_at_exception("workload_b_outers.jsonl", check)
        with error_patch, patch.object(h, "evaluate_eligible_record", side_effect=evaluate):
            with self.assertRaisesRegex(h.HarnessContractError, "Workload B .* discrepancy"):
                h.run_workload_b(runtime, manifest, cells, self.output, state)
        self.assertEqual(calls, expected)
        self.assertEqual(runtime.runner.fixed_observed.call_count, 1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(state["MC_OUTERS_RECONSTRUCTED"], 0)
        self.output = previous_output

    def test_mc_exceedance_mismatch_stops_before_next_outer(self):
        self.check_mc_failure("exceedance")

    def test_mc_reject_mismatch_stops_before_next_outer(self):
        self.check_mc_failure("reject")

    def test_mc_simultaneous_mismatches_preserve_both_counters(self):
        self.check_mc_failure("both")

    def check_mc_failure(self, kind):
        runtime, manifest, cells, calls, evaluate = self.prepare_b(mc=kind)

        def check(rows):
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(len(row["record_level_results"]), 16)
            self.assertEqual(row["mc_exceedance_match"], kind not in {"exceedance", "both"})
            self.assertEqual(row["mc_reject_match"], kind not in {"reject", "both"})
            self.assertEqual(self.state["MC_EXCEEDANCE_COUNT_MISMATCH"], int(kind in {"exceedance", "both"}))
            self.assertEqual(self.state["MC_REJECT_DECISION_MISMATCH"], int(kind in {"reject", "both"}))

        error_patch, seen = self.evidence_at_exception("workload_b_outers.jsonl", check)
        with error_patch, patch.object(h, "evaluate_eligible_record", side_effect=evaluate):
            with self.assertRaisesRegex(h.HarnessContractError, "Workload B MC discrepancy"):
                h.run_workload_b(runtime, manifest, cells, self.output, self.state)
        self.assertEqual(len(calls), 16)
        self.assertEqual(len(seen), 1)
        self.assertEqual(runtime.runner.fixed_observed.call_count, 1)
        runtime.runner.aggregate_outer.assert_called_once()

    def test_b_identity_mismatch_stops_before_record_evaluation(self):
        runtime, manifest, cells, calls, evaluate = self.prepare_b()
        manifest = copy.deepcopy(manifest)
        manifest["mc_failed_outers"][0]["bootstrap_identities"][0]["raw_inner_index"] += 999
        error_patch, seen = self.evidence_at_exception(
            "workload_b_outers.jsonl", lambda rows: self.assertEqual(rows[0]["record_level_results"], [])
        )
        with error_patch, patch.object(h, "evaluate_eligible_record", side_effect=evaluate):
            with self.assertRaisesRegex(h.HarnessContractError, "MC_BOOTSTRAP_IDENTITY_MISMATCH"):
                h.run_workload_b(runtime, manifest, cells, self.output, self.state)
        self.assertEqual(calls, [])
        self.assertEqual(self.state["MC_BOOTSTRAP_IDENTITY_MISMATCH"], 1)
        self.assertEqual(len(seen), 1)
        runtime.runner._cuda_nb_classification.assert_not_called()
        runtime.runner.aggregate_outer.assert_not_called()

    def test_b_success_does_not_double_count_and_aggregates_all_records(self):
        runtime, manifest, cells, calls, evaluate = self.prepare_b(outer_count=12)
        self.state["FIT_GATE_FAILURE"] = 7  # Past values must not masquerade as a new record delta.
        with patch.object(h, "evaluate_eligible_record", side_effect=evaluate):
            h.run_workload_b(runtime, manifest, cells, self.output, self.state)
        self.assertEqual(len(calls), 192)
        self.assertEqual(self.state["WORKLOAD_B_RECORDS_EXECUTED"], 192)
        self.assertEqual(self.state["MC_OUTERS_RECONSTRUCTED"], 12)
        self.assertEqual(self.state["FIT_GATE_FAILURE"], 7)
        self.assertEqual(runtime.runner.aggregate_outer.call_count, 12)
        for call in runtime.runner.aggregate_outer.call_args_list:
            self.assertEqual(len(call.args[1]), 15)

    def test_preserve_failure_writes_policy_fail_summary_and_verified_digests(self):
        try:
            raise h.HarnessContractError("mock scientific discrepancy")
        except h.HarnessContractError as exc:
            h.preserve_failure(self.output, self.state, exc)
        failure = json.loads((self.output / "failure.json").read_text())
        summary = json.loads((self.output / "summary.json").read_text())
        digests = json.loads((self.output / "digests.json").read_text())
        self.assertIs(failure["preservation_policy"]["NO_FIXTURE_CHANGE"], True)
        self.assertEqual(summary["TARGETED_GPU_REPLAY"], "FAIL")
        self.assertEqual(summary["execution_state"], "INCOMPLETE_UNEXPECTED_FAILURE")
        self.assertEqual(set(digests), {"failure.json", "summary.json"})
        for name, digest in digests.items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)

    def test_static_audit_and_imports_remain_science_free(self):
        self.assertEqual(h.validate_static()["validation"], "PASS")
        self.assertFalse(any(name.split(".")[0] in BLOCKED_ROOTS for name in sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
