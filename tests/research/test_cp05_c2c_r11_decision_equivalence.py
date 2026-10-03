"""Synthetic/mocked software tests only; never read an accepted R11 artifact."""
from __future__ import annotations

import ast
import base64
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.distribution_gof.cuda_calibration.r11_reference_workload.codec import (
    ContractError, SamplePayload, canonical_json, sample_digest, sha256, strict_json,
)
from experiments.distribution_gof.cuda_calibration.r11_reference_workload.oracle import OracleError
from experiments.distribution_gof.cuda_calibration.r11_decision_equivalence import (
    adjudication, artifacts, boundary_fixtures, contract, harness, preflight, runtime,
)

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "experiments.distribution_gof.cuda_calibration.r11_decision_equivalence"
IDENTITY = {"R11_HARNESS_SHA": "a" * 40, "R11_HARNESS_TREE": "b" * 40}


def toy_workload():
    """Literal toy data/provenance only: no RNG, model fits or sample generators."""
    outers = []
    for index in (9, 2, 11, 1, 10, 0, 8, 3, 7, 4, 6, 5):
        cell = f"exponential|SYNTHETIC_SOFTWARE_TEST={index}"
        identity = f"{cell}|raw_outer={index}"
        observed = {
            "identity": identity, "seed_identity": 1000 + index,
            "payload": SamplePayload.from_sample(np.array([1., 2., 3.])).serialize(),
        }
        accepted = []
        for ordinal in range(199):
            raw = ordinal * 2 + 3  # skipped raw indices must not be regenerated/pruned
            accepted.append({
                "identity": f"{identity}|raw_inner={raw}", "accepted_ordinal": ordinal,
                "raw_inner_index": raw, "seed_identity": 2000 + raw,
                "payload": SamplePayload.from_sample(
                    np.array([2. if ordinal < 8 else 0., 2., 3.])).serialize(),
            })
        outers.append({"outer_identity": identity, "cell_id": cell, "raw_outer_index": index,
                       "observed": observed, "accepted": accepted})
    return {"source_kind": "SYNTHETIC_TEST", "outers": outers}


class ToyRuntime:
    """Never loads the canonical science, SciPy, CuPy or a CUDA device."""
    def __init__(self, *, mode=None):
        self.mode = mode
        self.calls = []
        self.cpu_calls = self.cuda_calls = 0

    def require_gpu(self):
        if self.mode == "missing_gpu":
            raise ContractError("fake GPU unavailable; no fallback")

    def environment(self):
        return {"software_test_mock": True}

    def verify_sources(self):
        pass

    def evaluate(self, item, sample, payload, capture, before_cuda):
        runtime.verify_sample(sample, payload)
        self.calls.append(item["identity"])
        self.cpu_calls += 1
        statistic = float(sample[0])
        cpu = {"classification": "ELIGIBLE", "statistic": statistic,
               "parameters": {"scale": 1.0}, "log_likelihood": None}
        capture.update(phase="cpu", raw_cpu_result=copy.deepcopy(cpu),
                       cpu_input_identity_pass=True, cpu_sample_digest=sample_digest(sample))
        if self.mode == "after_cpu":
            raise RuntimeError("injected failure after raw CPU result")
        runtime.verify_sample(sample, payload)
        capture.update(phase="cuda", cuda_input_identity_pass=True,
                       cuda_sample_digest=sample_digest(sample))
        before_cuda()
        self.cuda_calls += 1
        if self.mode == "cuda_exception":
            raise RuntimeError("injected CUDA exception")
        cuda = {**cpu, "parameters": {"scale": 1.0}, "solver_converged": True,
                "failure_reason": None}
        capture["raw_cuda_result"] = copy.deepcopy(cuda)
        result = boundary_fixtures.logical_record(item, statistic, statistic)
        result.update(raw_cpu_result=copy.deepcopy(cpu), raw_cuda_result=copy.deepcopy(cuda))
        if self.mode == "nonconvergence":
            result["cuda_solver_converged"] = False
            result["raw_cuda_result"]["solver_converged"] = False
        if self.mode == "gate_failure":
            result["fit_gate_pass"] = False
        if self.mode == "identity_drift":
            result["identity"] += "|DRIFT"
        return result


class AdjudicationTests(unittest.TestCase):
    def fixture(self, name="equal_b9"):
        return boundary_fixtures.materialize(name)

    def adjudicate(self, name):
        outer, records, expected, _ = self.fixture(name)
        return adjudication.adjudicate_outer(records, expected, outer)

    def test_all_twelve_expected_dispositions(self):
        rows = boundary_fixtures.evaluate_fixtures()
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["fixture_pass"] for row in rows))
        self.assertTrue(all(row["adjudication"]["bootstrap_count"] == 199 for row in rows))
        self.assertEqual([row["adjudication"]["R11_OUTER_PASS"] for row in rows],
                         [True, True, True, True, True, False, False, False,
                          False, False, False, False])

    def test_equal_counts_at_each_boundary(self):
        for b in (8, 9, 10, 11):
            result = self.adjudicate(f"equal_b{b}")
            self.assertEqual((result["raw_b_cpu"], result["raw_b_cuda"]), (b, b))
            self.assertEqual(result["raw_p_cpu"], (b + 1) / 200)
            self.assertIs(result["raw_reject_cpu"], b <= 9)

    def test_certified_count_difference_with_same_reject_passes(self):
        result = self.adjudicate("certified_b9_b8")
        self.assertEqual((result["raw_b_cpu"], result["raw_b_cuda"]), (9, 8))
        self.assertEqual(result["CERTIFIED_EXACT_TIE_CROSSING_COUNT"], 1)
        self.assertEqual(result["MC_UNEXPLAINED_INDICATOR_MISMATCH"], 0)
        self.assertTrue(result["DEC024_SIGNED_ACCOUNTING"])
        self.assertTrue(result["R11_OUTER_PASS"])

    def test_certified_b10_b9_reject_mismatch_is_mandatory_failure(self):
        result = self.adjudicate("certified_b10_b9")
        self.assertEqual((result["raw_p_cpu"], result["raw_p_cuda"]), (.055, .05))
        self.assertEqual((result["raw_reject_cpu"], result["raw_reject_cuda"]), (False, True))
        self.assertEqual(result["CERTIFIED_EXACT_TIE_CROSSING_COUNT"], 1)
        self.assertFalse(result["R11_OUTER_PASS"])
        self.assertIn("REJECT_DECISION_MISMATCH", result["failures"])

    def test_opposite_crossing_never_certifies(self):
        result = self.adjudicate("opposite_crossing")
        self.assertEqual(result["CERTIFIED_EXACT_TIE_CROSSING_COUNT"], 0)
        self.assertFalse(result["DEC024_SIGNED_ACCOUNTING"])
        self.assertFalse(result["R11_OUTER_PASS"])

    def test_nextafter_is_not_exact_tie(self):
        result = self.adjudicate("nextafter_neighbor")
        row = result["indicator_adjudication"][8]
        self.assertFalse(row["cpu_reference_exact_tie"])
        self.assertFalse(row["certified_exact_tie_crossing"])

    def test_one_ulp_unequal_is_not_exact_tie(self):
        result = self.adjudicate("one_ulp_unequal")
        row = result["indicator_adjudication"][8]
        self.assertEqual(row["T_cpu_boot"] - row["T_cpu_obs"], math.ulp(1.0))
        self.assertFalse(row["cpu_reference_exact_tie"])

    def test_isclose_only_is_not_exact_tie(self):
        result = self.adjudicate("isclose_only")
        row = result["indicator_adjudication"][8]
        self.assertTrue(math.isclose(row["T_cpu_boot"], row["T_cpu_obs"]))
        self.assertFalse(row["cpu_reference_exact_tie"])

    def test_cancelling_mismatches_fail_even_when_raw_counts_equal(self):
        result = self.adjudicate("cancelling_mismatches")
        self.assertEqual(result["raw_b_cpu"], result["raw_b_cuda"])
        self.assertEqual(result["INDICATOR_MISMATCH_COUNT"], 2)
        self.assertEqual(result["CERTIFIED_EXACT_TIE_CROSSING_COUNT"], 1)
        self.assertEqual(result["MC_UNEXPLAINED_INDICATOR_MISMATCH"], 1)
        self.assertFalse(result["DEC024_SIGNED_ACCOUNTING"])
        self.assertFalse(result["R11_OUTER_PASS"])

    def test_unexplained_mismatch_fails_with_unchanged_reject(self):
        result = self.adjudicate("unexplained_same_reject")
        self.assertTrue(result["RAW_REJECT_DECISION_MATCH"])
        self.assertEqual(result["MC_UNEXPLAINED_INDICATOR_MISMATCH"], 1)
        self.assertFalse(result["R11_OUTER_PASS"])

    def test_exactly_199_bootstraps_required(self):
        outer, records, expected, _ = self.fixture()
        for rs, es in ((records[:-1], expected[:-1]), (records + [records[-1]], expected)):
            with self.assertRaisesRegex(ContractError, "199"):
                adjudication.adjudicate_outer(rs, es, outer)

    def test_aggregate_fields_are_ignored_and_never_overwritten(self):
        outer, records, expected, _ = self.fixture()
        for record in records:
            record.update(raw_b_cpu=123, raw_b_cuda=125, raw_p_cpu=.999, raw_reject_cpu=False)
        before = copy.deepcopy(records)
        result = adjudication.adjudicate_outer(records, expected, outer)
        self.assertEqual((result["raw_b_cpu"], result["raw_b_cuda"]), (9, 9))
        self.assertEqual(result["raw_p_cpu"], .05)
        self.assertTrue(result["raw_reject_cpu"])
        self.assertEqual(records, before)

    def test_exact_greater_equal_indicator(self):
        outer, records, expected, _ = self.fixture()
        records[1]["cpu_statistic"] = records[1]["cuda_statistic"] = 1.0
        result = adjudication.adjudicate_outer(records, expected, outer)
        row = result["indicator_adjudication"][0]
        self.assertIs(row["cpu_indicator"], True)
        self.assertIs(row["cuda_indicator"], True)
        self.assertTrue(row["cpu_reference_exact_tie"])
        records[1]["cpu_statistic"] = math.nextafter(1.0, 0.0)
        result = adjudication.adjudicate_outer(records, expected, outer)
        self.assertIs(result["indicator_adjudication"][0]["cpu_indicator"], False)

    def test_nonfinite_statistics_cannot_certify_or_pass(self):
        for value in (math.nan, math.inf, -math.inf, True, None):
            outer, records, expected, _ = self.fixture("certified_b9_b8")
            records[9]["cpu_statistic"] = value
            result = adjudication.adjudicate_outer(records, expected, outer)
            self.assertFalse(result["R11_OUTER_PASS"])
            self.assertFalse(result["indicator_adjudication"][8]["certified_exact_tie_crossing"])
            self.assertIsNone(result["raw_b_cpu"])

    def test_each_provenance_and_numerical_gate_required_for_tie(self):
        changes = [
            ("identity", "wrong"), ("raw_outer_index", True), ("raw_inner_index", True),
            ("seed_identity", -1), ("accepted_ordinal", True), ("sample_digest", "0" * 64),
            ("cpu_sample_digest", "0" * 64), ("cuda_sample_digest", "0" * 64),
            ("payload_dtype", ">f8"), ("payload_shape", [3]),
            ("cpu_input_identity_pass", False), ("cuda_input_identity_pass", False),
            ("cpu_classification", "FAILED"), ("cuda_classification", "FAILED"),
            ("cuda_failure_reason", "failure"), ("cuda_solver_converged", False),
            ("distribution_evidence", []), ("cuda_evaluation_points", [2.0]),
            ("structural_failure", {"reason": "injected"}),
            *[(g, False) for g in contract.GATES],
        ]
        for position in (0, 9):
            for key, value in changes:
                with self.subTest(position=position, key=key):
                    outer, records, expected, _ = self.fixture("certified_b9_b8")
                    records[position][key] = value
                    result = adjudication.adjudicate_outer(records, expected, outer)
                    self.assertEqual(result["CERTIFIED_EXACT_TIE_CROSSING_COUNT"], 0)
                    self.assertFalse(result["R11_OUTER_PASS"])

    def test_unknown_structural_reason_fails_closed(self):
        outer, records, expected, _ = self.fixture("certified_b9_b8")
        records[9]["distribution_evidence"][0]["failure_reason"] = "UNKNOWN"
        result = adjudication.adjudicate_outer(records, expected, outer)
        self.assertFalse(result["R11_OUTER_PASS"])
        self.assertEqual(result["CERTIFIED_EXACT_TIE_CROSSING_COUNT"], 0)

    def test_missing_evidence_field_and_truthy_gate_fail(self):
        for mutation in ("missing", "truthy"):
            outer, records, expected, _ = self.fixture()
            if mutation == "missing":
                del records[0]["raw_cpu_result"]
            else:
                records[0]["fit_gate_pass"] = 1
            self.assertFalse(adjudication.adjudicate_outer(records, expected, outer)["R11_OUTER_PASS"])

    def test_frozen_order_and_duplicate_identity_rejected(self):
        outer, records, expected, _ = self.fixture()
        altered = copy.deepcopy(expected)
        altered[2] = altered[1]
        with self.assertRaisesRegex(ContractError, "duplicate"):
            adjudication.adjudicate_outer(records, altered, outer)
        with self.assertRaisesRegex(ContractError, "order"):
            adjudication.adjudicate_outer(records, [expected[0], expected[2], expected[1],
                                                    *expected[3:]], outer)

    def test_boundary_metric_zero_is_allowed(self):
        results, records, expected = [], [], []
        for index in range(12):
            outer, rs, es, _ = self.fixture("equal_b8")
            # Give every logical outer distinct identities while keeping its complete surface.
            for r, e in zip(rs, es):
                suffix = r["identity"].split("|raw_outer=", 1)[1]
                identity = f"exponential|LOGICAL_OUTER={index}|raw_outer={suffix}"
                r["identity"] = e["identity"] = identity
                r["cell_id"] = e["cell_id"] = f"exponential|LOGICAL_OUTER={index}"
            for record in rs[1:]:
                record["cpu_statistic"] = record["cuda_statistic"] = 0.0
            results.append(adjudication.adjudicate_outer(rs, es, es[0]["identity"]))
            records.extend(rs)
            expected.extend(es)
        summary = adjudication.summarize(results, records, expected,
            boundary_fixtures.evaluate_fixtures(), accepted=True, consumed=True)
        self.assertTrue(summary["R11_GLOBAL_PASS"])
        self.assertEqual(summary["SCIENTIFIC_BOUNDARY_NEIGHBORHOOD_OBSERVED"], 0)
        for kwargs in ({"accepted": False, "consumed": True}, {"accepted": True, "consumed": False},
                       {"accepted": True, "consumed": True, "failure": {"reason": "incomplete"}}):
            self.assertFalse(adjudication.summarize(results, records, expected,
                boundary_fixtures.evaluate_fixtures(), **kwargs)["R11_GLOBAL_PASS"])
        self.assertFalse(adjudication.summarize(results[:-1], records[:-200], expected,
            boundary_fixtures.evaluate_fixtures(), accepted=True, consumed=True)["R11_GLOBAL_PASS"])


class PreflightTests(unittest.TestCase):
    def test_expected_workload_sha_mismatch_before_oracle(self):
        with patch.object(preflight.oracle, "load_for_use") as load:
            with self.assertRaisesRegex(ContractError, "workload SHA256"):
                preflight.verify_workload(b"SYNTHETIC", b"manifest", b"archive", b"crossings")
            load.assert_not_called()

    def test_oracle_failure_fails_closed(self):
        data = b"SYNTHETIC_ONLY"
        with patch.object(preflight, "REFERENCE_WORKLOAD_SHA256", sha256(data)), patch.object(
                preflight.oracle, "load_for_use", side_effect=OracleError("R4 prefix FAIL")):
            with self.assertRaisesRegex(OracleError, "R4 prefix"):
                preflight.verify_workload(data, b"m", b"a", b"c")

    def test_loader_receives_all_historical_inputs_and_exact_binding_required(self):
        data = b"SYNTHETIC_ONLY"
        for binding in ({"sha": preflight.BUILDER_SHA, "tree": preflight.BUILDER_TREE},
                        {"sha": "c" * 40, "tree": preflight.BUILDER_TREE}):
            with patch.object(preflight, "REFERENCE_WORKLOAD_SHA256", sha256(data)), patch.object(
                    preflight.oracle, "load_for_use", return_value={"builder_binding": binding}) as load:
                if binding["sha"] == preflight.BUILDER_SHA:
                    preflight.verify_workload(data, b"m", b"a", b"c")
                else:
                    with self.assertRaisesRegex(ContractError, "builder SHA/tree"):
                        preflight.verify_workload(data, b"m", b"a", b"c")
                load.assert_called_once_with(data, b"m", b"a", crossings_bytes=b"c")

    def test_crossings_mandatory(self):
        data = b"SYNTHETIC_ONLY"
        with patch.object(preflight, "REFERENCE_WORKLOAD_SHA256", sha256(data)), patch.object(
                preflight.oracle, "load_for_use") as load:
            with self.assertRaisesRegex(ContractError, "crossings required"):
                preflight.verify_workload(data, b"m", b"a", None)
            load.assert_not_called()

    def test_real_oracle_rejects_synthetic_archive(self):
        with self.assertRaises(OracleError):
            preflight.oracle.runtime_records(b"NOT_AN_R4_ARCHIVE", crossings_bytes=b"NOT_R4")

    def test_failed_preflight_does_not_construct_scientific_runtime(self):
        with patch.object(harness, "prepare", side_effect=OracleError("R4 prefix FAIL")), patch.object(
                harness, "CanonicalRuntime") as factory:
            with self.assertRaises(OracleError):
                harness.execute(workload="toy", r4_archive="toy", r4_crossings="toy",
                                output="unused-software-output", require_gpu=True)
            factory.assert_not_called()


class GitIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="r11-software-git-")
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.git("init", "-q")
        self.git("config", "core.autocrlf", "false")
        self.write("experiments/scientific_fixture.py", "SOFTWARE_MARKER = 1\n")
        self.scientific = self.commit("synthetic scientific marker")
        self.scientific_tree = self.git("rev-parse", "HEAD^{tree}")
        self.write(contract.BUILDER_PATH + "/codec.py", "SOFTWARE_BUILDER_MARKER = 1\n")
        self.base = self.commit("synthetic builder marker")
        self.base_tree = self.git("rev-parse", "HEAD^{tree}")
        self.write(contract.PACKAGE_PATH + "/harness.py", "SOFTWARE_HARNESS_MARKER = 1\n")
        self.head = self.commit("synthetic harness marker")
        self.tree = self.git("rev-parse", "HEAD^{tree}")
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("SCIENTIFIC_SHA", self.scientific),
                            ("SCIENTIFIC_TREE", self.scientific_tree),
                            ("BUILDER_SHA", self.base), ("BUILDER_TREE", self.base_tree)):
            self.stack.enter_context(patch.object(preflight, name, value))

    def git(self, *args):
        result = subprocess.run([shutil.which("git"), "-c", "core.autocrlf=false",
                                 "-c", "commit.gpgSign=false", "-c", "user.name=SoftwareFixture",
                                 "-c", "user.email=fixture@example.invalid", *args],
                                cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def write(self, relative, value):
        file = self.repo / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(value, encoding="utf-8", newline="\n")

    def commit(self, message):
        self.git("add", ".")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD")

    def test_actual_harness_sha_and_tree_captured(self):
        result = preflight.repository_identity(self.repo)
        self.assertEqual(result["R11_HARNESS_SHA"], self.head)
        self.assertEqual(result["R11_HARNESS_TREE"], self.tree)
        self.assertTrue(result["builder_surface_unchanged"])

    def test_dirty_worktree_rejected(self):
        self.write(contract.PACKAGE_PATH + "/harness.py", "DIRTY\n")
        with self.assertRaisesRegex(ContractError, "clean"):
            preflight.repository_identity(self.repo)

    def test_untracked_file_rejected(self):
        self.write("untracked.txt", "untracked")
        with self.assertRaisesRegex(ContractError, "clean"):
            preflight.repository_identity(self.repo)

    def test_scientific_mutation_rejected_after_commit(self):
        self.write("experiments/scientific_fixture.py", "CHANGED\n")
        self.commit("synthetic mutation")
        with self.assertRaisesRegex(ContractError, "scientific code changed"):
            preflight.repository_identity(self.repo)

    def test_scientific_rename_rejected(self):
        self.git("mv", "experiments/scientific_fixture.py", "experiments/renamed.py")
        self.commit("synthetic rename")
        with self.assertRaisesRegex(ContractError, "scientific code changed"):
            preflight.repository_identity(self.repo)

    def test_builder_mutation_rejected_after_commit(self):
        self.write(contract.BUILDER_PATH + "/codec.py", "CHANGED\n")
        self.commit("synthetic builder mutation")
        with self.assertRaisesRegex(ContractError, "builder surface changed"):
            preflight.repository_identity(self.repo)

    def test_other_baseline_mutation_or_addition_rejected(self):
        self.write("unrelated.txt", "unrelated")
        self.commit("synthetic out of scope addition")
        with self.assertRaisesRegex(ContractError, "authorized isolated scope"):
            preflight.repository_identity(self.repo)

    def test_wrong_scientific_and_builder_tree_rejected(self):
        for field, message in (("SCIENTIFIC_TREE", "scientific tree"), ("BUILDER_TREE", "builder base tree")):
            with patch.object(preflight, field, "0" * 40):
                with self.assertRaisesRegex(ContractError, message):
                    preflight.repository_identity(self.repo)

    def test_module_origin_must_be_tracked_and_in_checkout(self):
        with self.assertRaisesRegex(ContractError, "outside"):
            runtime._verify_file(self.repo, ROOT / "experiments/__init__.py")
        self.write(contract.PACKAGE_PATH + "/untracked.py", "UNTRACKED\n")
        with self.assertRaises(ContractError):
            runtime._verify_file(self.repo, self.repo / contract.PACKAGE_PATH / "untracked.py")
        path = self.repo / contract.PACKAGE_PATH / "harness.py"
        runtime._verify_file(self.repo, path)
        path.write_text("MUTATED\n")
        with self.assertRaisesRegex(ContractError, "differs"):
            runtime._verify_file(self.repo, path)

    def test_windows_line_ending_checkout_filters_preserve_git_identity(self):
        path = self.repo / contract.PACKAGE_PATH / "harness.py"
        self.git("config", "core.autocrlf", "true")
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        runtime._verify_file(self.repo, path)


class CanonicalDelegationTests(unittest.TestCase):
    def setup_runtime(self, *, family="exponential", drift=False):
        outer = toy_workload()["outers"][0]
        item = preflight.descriptor(outer, outer["observed"])
        cell = SimpleNamespace(canonical_id=item["cell_id"], family=family, statistic="CVM", n=3)
        payload = SamplePayload.from_serialized(outer["observed"]["payload"])
        sample = payload.deserialize()
        cpu_result = {"classification": "ELIGIBLE", "statistic": 1., "parameters": {"scale": 1.},
                      "log_likelihood": None, "evaluation_points": (0.,),
                      "bound": object(), "reference_log_likelihood": lambda params: 0.}
        cuda_result = {**cpu_result, "solver_converged": True, "failure_reason": None}
        runner = SimpleNamespace(
            evaluate_reference_record=Mock(return_value=cpu_result),
            evaluate_cuda_record=Mock(return_value=cuda_result),
            canonical_distribution_value_points=Mock(return_value=(0.,)))
        def fixed(**kwargs):
            left = kwargs["reference_adapter"](cell, kwargs["sample"])
            right = kwargs["cuda_adapter"](cell, kwargs["sample"])
            result = boundary_fixtures.logical_record(item, left["statistic"], right["statistic"])
            if drift:
                result["identity"] = "DRIFT"
            return result
        runner.evaluate_fixed_record = Mock(side_effect=fixed)
        support = SimpleNamespace(certify_nb_support=Mock(return_value=SimpleNamespace(
            indices=(0, 1), remainder_bound=1e-14)))
        cuda = SimpleNamespace(require_cuda=Mock(return_value=object()))
        modules = SimpleNamespace(runner=runner, support=support, cuda=cuda,
            preregistration=SimpleNamespace(primary_fixture_matrix=lambda: [cell]),
            loaded_module_paths={})
        with patch.object(runtime, "_load_runtime", return_value=modules):
            adapter = runtime.CanonicalRuntime(ROOT)
        adapter.require_gpu()
        return adapter, runner, support, item, sample, payload

    def test_same_frozen_sample_reaches_independent_delegates(self):
        adapter, runner, support, item, sample, payload = self.setup_runtime()
        capture, before = {}, Mock()
        record = adapter.evaluate(item, sample, payload, capture, before)
        self.assertIs(runner.evaluate_reference_record.call_args.args[1], sample)
        self.assertIs(runner.evaluate_cuda_record.call_args.args[1], sample)
        self.assertFalse(sample.flags.writeable)
        self.assertTrue(record["cpu_input_identity_pass"])
        self.assertTrue(record["cuda_input_identity_pass"])
        support.certify_nb_support.assert_not_called()
        before.assert_called_once()

    def test_nb_support_delegated_from_same_cpu_bound(self):
        adapter, runner, support, item, sample, payload = self.setup_runtime(family="negative_binomial")
        capture = {}
        adapter.evaluate(item, sample, payload, capture, Mock())
        bound = runner.evaluate_reference_record.return_value["bound"]
        support.certify_nb_support.assert_called_once_with(sample, bound, "CVM")
        self.assertEqual(runner.evaluate_cuda_record.call_args.kwargs,
                         {"certified_support": (0, 1), "remainder_bound": 1e-14})

    def test_digest_or_shape_tamper_blocks_both_engines(self):
        adapter, runner, _, item, sample, payload = self.setup_runtime()
        tampered = np.array([1., 2., 4.])
        for array in (tampered, sample.reshape(1, 3)):
            with self.assertRaisesRegex(ContractError, "identity mismatch"):
                adapter.evaluate(item, array, payload, {}, Mock())
        runner.evaluate_reference_record.assert_not_called()
        runner.evaluate_cuda_record.assert_not_called()

    def test_cpu_mutated_interpretation_blocks_cuda(self):
        adapter, runner, _, item, sample, payload = self.setup_runtime()
        result = runner.evaluate_reference_record.return_value
        def cpu(cell, array):
            array.shape = (1, 3)
            return result
        runner.evaluate_reference_record.side_effect = cpu
        capture, before = {}, Mock()
        with self.assertRaisesRegex(ContractError, "identity mismatch"):
            adapter.evaluate(item, sample, payload, capture, before)
        self.assertEqual(capture["raw_cpu_result"]["statistic"], 1.)
        runner.evaluate_cuda_record.assert_not_called()
        before.assert_not_called()

    def test_cuda_failure_never_falls_back_to_cpu(self):
        adapter, runner, _, item, sample, payload = self.setup_runtime()
        runner.evaluate_cuda_record.side_effect = RuntimeError("fake CUDA failure")
        capture, before = {}, Mock()
        with self.assertRaisesRegex(RuntimeError, "fake CUDA"):
            adapter.evaluate(item, sample, payload, capture, before)
        runner.evaluate_reference_record.assert_called_once()
        runner.evaluate_cuda_record.assert_called_once()
        self.assertIn("raw_cpu_result", capture)
        before.assert_called_once()

    def test_missing_cuda_cannot_enable_numpy_fallback(self):
        adapter, runner, _, item, sample, payload = self.setup_runtime()
        adapter.modules.cuda.require_cuda.side_effect = ContractError("CuPy unavailable")
        adapter.cuda_interface = None
        with self.assertRaises(ContractError):
            adapter.require_gpu()
        with self.assertRaisesRegex(ContractError, "GPU preflight"):
            adapter.evaluate(item, sample, payload, {}, Mock())
        runner.evaluate_reference_record.assert_not_called()

    def test_delegated_identity_drift_not_silently_repaired(self):
        adapter, _, _, item, sample, payload = self.setup_runtime(drift=True)
        capture = {}
        with self.assertRaisesRegex(ContractError, "identity drift"):
            adapter.evaluate(item, sample, payload, capture, Mock())
        self.assertEqual(capture["delegated_record"]["identity"], "DRIFT")

    def test_support_failure_preserves_cpu_without_consuming_cuda(self):
        adapter, runner, support, item, sample, payload = self.setup_runtime(family="negative_binomial")
        support.certify_nb_support.side_effect = RuntimeError("fake support failure")
        capture, before = {}, Mock()
        with self.assertRaisesRegex(RuntimeError, "support failure"):
            adapter.evaluate(item, sample, payload, capture, before)
        self.assertIn("raw_cpu_result", capture)
        runner.evaluate_cuda_record.assert_not_called()
        before.assert_not_called()


class ExecutionSoftwareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="r11-software-evidence-")
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "bundle"
        self.toy = toy_workload()

    def receipt(self):
        return preflight.PreparedWorkload(ROOT, b"SYNTHETIC_INPUT_BYTES_ONLY",
            canonical_json(self.toy), canonical_json(IDENTITY))

    def mocked_execution(self, fake, *, output=None):
        with patch.object(harness, "prepare", return_value=self.receipt()), patch.object(
                harness, "CanonicalRuntime", return_value=fake), patch.object(
                harness, "repository_identity", return_value=IDENTITY):
            return harness.execute(workload="SYNTHETIC_PATH", r4_archive="MOCK_ARCHIVE",
                r4_crossings="MOCK_CROSSINGS", output=output or self.output, require_gpu=True)

    def read(self, name):
        return strict_json((self.output / name).read_bytes())

    def test_complete_mock_traversal_order_and_evidence(self):
        fake = ToyRuntime()
        summary = self.mocked_execution(fake)
        expected = [item["identity"] for _, item, _ in preflight.ordered_records(self.toy)]
        self.assertEqual(fake.calls, expected)
        self.assertEqual((fake.cpu_calls, fake.cuda_calls), (2400, 2400))
        self.assertTrue(summary["R11_GLOBAL_PASS"])
        self.assertEqual(summary["outer_count"], 12)
        self.assertEqual(summary["records_evaluated"], 2400)
        self.assertEqual(summary["reject_agreement_count"], 12)
        self.assertEqual(summary["SCIENTIFIC_BOUNDARY_NEIGHBORHOOD_OBSERVED"], 12)
        self.assertEqual((self.output / "reference_workload.json").read_bytes(),
                         b"SYNTHETIC_INPUT_BYTES_ONLY")
        manifest = self.read("execution_manifest.json")
        self.assertEqual(manifest["R11_HARNESS_SHA"], IDENTITY["R11_HARNESS_SHA"])
        self.assertTrue(manifest["EXECUTION_AUTHORIZATION_CONSUMED"])
        self.assertTrue((self.output / "authorization_consumed.json").exists())
        self.assertEqual(len((self.output / "records.jsonl").read_bytes().splitlines()), 2400)
        self.assertEqual(len((self.output / "indicator_adjudication.jsonl").read_bytes().splitlines()), 2388)
        digests = self.read("digests.json")["artifacts"]
        self.assertEqual(set(digests), {p.name for p in self.output.iterdir()} - {"digests.json"})
        for name, details in digests.items():
            data = (self.output / name).read_bytes()
            self.assertEqual(details, {"sha256": sha256(data), "bytes": len(data)})
        calls = fake.cpu_calls
        with self.assertRaisesRegex(ContractError, "exists"):
            self.mocked_execution(fake)
        self.assertEqual(fake.cpu_calls, calls)

    def test_payload_tamper_blocks_cpu_and_cuda_before_evaluation(self):
        payload = self.toy["outers"][0]["observed"]["payload"]
        payload["payload_base64"] = base64.b64encode(bytes(24)).decode()
        fake = ToyRuntime()
        result = self.mocked_execution(fake)
        self.assertEqual((fake.cpu_calls, fake.cuda_calls), (0, 0))
        self.assertFalse(result["R11_GLOBAL_PASS"])
        self.assertFalse(result["EXECUTION_AUTHORIZATION_CONSUMED"])
        self.assertFalse((self.output / "authorization_consumed.json").exists())

    def test_partial_cpu_failure_preserves_raw_and_unconsumed_status(self):
        fake = ToyRuntime(mode="after_cpu")
        result = self.mocked_execution(fake)
        record = strict_json((self.output / "records.jsonl").read_bytes().splitlines()[0])
        self.assertEqual(record["raw_cpu_result"]["statistic"], 1.)
        self.assertIsNone(record["raw_cuda_result"])
        self.assertEqual((fake.cpu_calls, fake.cuda_calls), (1, 0))
        self.assertFalse(result["EXECUTION_AUTHORIZATION_CONSUMED"])
        self.assertFalse(result["R11_GLOBAL_PASS"])
        self.assertEqual(result["outer_count"], 0)
        self.assertTrue((self.output / "digests.json").exists())

    def test_cuda_exception_preserves_raw_cpu_and_stops_without_retry(self):
        fake = ToyRuntime(mode="cuda_exception")
        result = self.mocked_execution(fake)
        record = strict_json((self.output / "records.jsonl").read_bytes().splitlines()[0])
        self.assertEqual(record["raw_cpu_result"]["parameters"], {"scale": 1.})
        self.assertIsNone(record["raw_cuda_result"])
        self.assertIn("injected CUDA", record["cuda_failure_reason"])
        self.assertEqual((fake.cpu_calls, fake.cuda_calls), (1, 1))
        self.assertTrue(result["EXECUTION_AUTHORIZATION_CONSUMED"])
        self.assertFalse(result["R11_GLOBAL_PASS"])

    def test_nonconvergence_gate_failure_and_identity_drift_stop(self):
        for mode in ("nonconvergence", "gate_failure", "identity_drift"):
            with self.subTest(mode=mode):
                output = self.output.parent / mode
                fake = ToyRuntime(mode=mode)
                result = self.mocked_execution(fake, output=output)
                self.assertEqual((fake.cpu_calls, fake.cuda_calls), (1, 1))
                self.assertFalse(result["R11_GLOBAL_PASS"])
                self.assertTrue(result["EXECUTION_AUTHORIZATION_CONSUMED"])

    def test_topology_failure_cannot_start_evaluators(self):
        for edit in ("outer", "bootstrap"):
            original = copy.deepcopy(self.toy)
            if edit == "outer":
                self.toy["outers"].pop()
            else:
                self.toy["outers"][0]["accepted"].pop()
            fake = ToyRuntime()
            with self.assertRaisesRegex(ContractError, "exactly"):
                self.mocked_execution(fake)
            self.assertEqual(fake.cpu_calls, 0)
            self.assertFalse(self.output.exists())
            self.toy = original

    def test_gpu_preflight_failure_never_evaluates_cpu(self):
        fake = ToyRuntime(mode="missing_gpu")
        with self.assertRaisesRegex(ContractError, "fake GPU"):
            self.mocked_execution(fake)
        self.assertEqual(fake.cpu_calls, 0)
        self.assertFalse(self.output.exists())

    def test_require_gpu_false_rejected_before_loading_workload(self):
        with patch.object(harness, "prepare") as prepare:
            with self.assertRaisesRegex(ContractError, "mandatory"):
                harness.execute(workload="toy", r4_archive="toy", r4_crossings="toy",
                                output=self.output, require_gpu=False)
            prepare.assert_not_called()

    def test_harness_identity_drift_rejected_before_science(self):
        fake = ToyRuntime()
        with patch.object(harness, "prepare", return_value=self.receipt()), patch.object(
                harness, "CanonicalRuntime", return_value=fake), patch.object(
                harness, "repository_identity", return_value={**IDENTITY, "R11_HARNESS_SHA": "c" * 40}):
            with self.assertRaisesRegex(ContractError, "identity changed"):
                harness.execute(workload="toy", r4_archive="toy", r4_crossings="toy",
                                output=self.output, require_gpu=True)
        self.assertEqual(fake.cpu_calls, 0)

    def test_evidence_error_retains_consumption_status(self):
        fake = ToyRuntime(mode="cuda_exception")
        publish = artifacts.EvidenceBundle.publish
        def fail_summary(bundle, name, value, **kwargs):
            if name == "summary.json":
                raise OSError("injected publication failure")
            return publish(bundle, name, value, **kwargs)
        with patch.object(artifacts.EvidenceBundle, "publish", new=fail_summary):
            with self.assertRaises(harness.ExecutionEvidenceError) as caught:
                self.mocked_execution(fake)
        self.assertTrue(caught.exception.consumed)
        self.assertTrue((self.output / "records.jsonl").exists())
        self.assertTrue((self.output / "authorization_consumed.json").exists())


class StaticAndArtifactTests(unittest.TestCase):
    def test_b_alpha_and_frozen_workload_sha(self):
        self.assertEqual(contract.B_R11, 199)
        self.assertEqual(contract.ALPHA, .05)
        self.assertEqual(contract.REFERENCE_WORKLOAD_SHA256,
                         "77f53d616deeafcba51aa185dae8a8d2ca60ee4172f4a6353a3aa97b75324b16")
        self.assertEqual(contract.BUILDER_SHA, "1ace65bf9e01ab05df84a1cbca5031fac93bfa88")

    def test_no_supported_regeneration_or_old_aggregate_calls(self):
        forbidden = {"_generate", "derive_seed", "fixed_observed", "fixed_bootstraps",
                     "traverse_primary", "execute_primary_outer", "aggregate_outer"}
        directory = ROOT / contract.PACKAGE_PATH
        for file in directory.glob("*.py"):
            tree = ast.parse(file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = (node.func.id if isinstance(node.func, ast.Name) else
                            node.func.attr if isinstance(node.func, ast.Attribute) else "")
                    self.assertNotIn(name, forbidden, str(file))
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(forbidden.intersection(alias.name for alias in node.names))

    def test_cpu_only_import_and_static_validation_without_science(self):
        script = """
import importlib.abc, sys
class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in ('cupy', 'cupyx', 'scipy')
            or fullname.endswith(('cp05_cuda_engine', 'cp05_c2c_equivalence_runner', 'nb_support'))):
            raise AssertionError('scientific import attempted: ' + fullname)
sys.meta_path.insert(0, NoScience())
from experiments.distribution_gof.cuda_calibration.r11_decision_equivalence import harness
assert harness.validate_static()['STATIC_VALIDATION_PASS']
assert not any(n.startswith(('cupy', 'cupyx', 'scipy')) for n in sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_static_mode_never_prepares_or_constructs_runtime(self):
        with patch.object(harness, "prepare", side_effect=AssertionError("load")), patch.object(
                harness, "CanonicalRuntime", side_effect=AssertionError("science")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(harness.main(["--validate-static"]), 0)

    def test_cli_rejects_resume_and_all_mutable_contract_flags(self):
        for flag in ("--resume", "--B", "--alpha", "--namespace", "--tolerance",
                     "--tie-rule", "--retry-cap", "--workload-expected-sha", "--source-hash"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                harness.build_parser().parse_args(["--validate-static", flag, "1"])

    def test_execute_requires_all_paths_and_gpu(self):
        for arguments in (["--execute"], ["--execute", "--workload", "toy",
                "--r4-archive", "toy", "--r4-crossings", "toy", "--output", "toy"],
                ["--validate-static", "--workload", "toy"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                harness.main(arguments)

    def test_late_error_cli_does_not_claim_unconsumed_authorization(self):
        error = harness.ExecutionEvidenceError("late error", True, "toy-output")
        stream = io.StringIO()
        with patch.object(harness, "execute", side_effect=error), contextlib.redirect_stdout(stream):
            code = harness.main(["--execute", "--workload", "toy", "--r4-archive", "toy",
                                 "--r4-crossings", "toy", "--output", "toy", "--require-gpu"])
        self.assertEqual(code, 2)
        self.assertTrue(json.loads(stream.getvalue())["EXECUTION_AUTHORIZATION_CONSUMED"])

    def test_raw_snapshots_detached_and_nonfinite_preserved(self):
        result = {"parameters": {"scale": 1.}, "statistic": math.nan, "p": .05,
                  "b": 9, "reject": True}
        data = artifacts.snapshot(result)
        result["parameters"]["scale"] = 99.
        restored = strict_json(data)
        self.assertEqual(restored["parameters"]["scale"], 1.)
        self.assertEqual(restored["statistic"], {"nonfinite_float": "nan"})
        self.assertEqual((restored["b"], restored["p"], restored["reject"]), (9, .05, True))
        self.assertIsInstance(data, bytes)

    def test_write_once_no_overwrite_or_inside_repository(self):
        with tempfile.TemporaryDirectory(prefix="r11-software-write-once-") as directory:
            output = Path(directory) / "bundle"
            with contextlib.closing(artifacts.EvidenceBundle(output, ROOT)) as bundle:
                bundle.publish("one.json", {"raw": 1})
                with self.assertRaisesRegex(ContractError, "duplicate"):
                    bundle.publish("one.json", {"raw": 2})
                self.assertEqual(strict_json((output / "one.json").read_bytes()), {"raw": 1})
                with self.assertRaisesRegex(ContractError, "exists"):
                    artifacts.EvidenceBundle(output, ROOT)
                with self.assertRaisesRegex(ContractError, "outside"):
                    artifacts.EvidenceBundle(ROOT / "forbidden-evidence", ROOT)

    def test_stream_creation_failure_closes_already_open_stream(self):
        opened = []
        original = Path.open
        def fail_second(path, *args, **kwargs):
            if path.name == "indicator_adjudication.jsonl":
                raise OSError("injected stream creation failure")
            stream = original(path, *args, **kwargs)
            opened.append(stream)
            return stream
        with tempfile.TemporaryDirectory(prefix="r11-software-open-failure-") as directory:
            with patch.object(Path, "open", new=fail_second), self.assertRaises(OSError):
                artifacts.EvidenceBundle(Path(directory) / "bundle", ROOT)
            self.assertTrue(opened)
            self.assertTrue(all(stream.closed for stream in opened))


if __name__ == "__main__":
    unittest.main()
