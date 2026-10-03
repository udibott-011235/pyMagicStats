"""Software fixtures only: never bind canonical scientific generation adapters."""
from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.distribution_gof.cuda_calibration.r11_reference_workload import oracle
from experiments.distribution_gof.cuda_calibration.r11_reference_workload.binding import bind_artifact, binding_from_git
from experiments.distribution_gof.cuda_calibration.r11_reference_workload.builder import (
    BuildFailure, CPUAdapters, ReferenceWorkloadBuilder, artifact_document,
)
from experiments.distribution_gof.cuda_calibration.r11_reference_workload.codec import (
    ContractError, SamplePayload, canonical_json, canonicalize_sample,
    sample_bytes, sample_digest, sha256, strict_json, write_once,
)
from experiments.distribution_gof.cuda_calibration.r11_reference_workload.contract import (
    ALPHA, ARCHIVE_NAME, ARTIFACT_HASHES, B_R11, CROSSINGS_NAME, NB_RETRY_CAP,
    PROJECTION_BYTES, PROJECTION_HASH, SOURCE_HASH, SOURCE_PATH, WORKLOAD_NAME,
)
from experiments.distribution_gof.cuda_calibration.r11_reference_workload.schema import load_artifact
from experiments.distribution_gof.cuda_calibration.r11_reference_workload.source import SourceSurface, projection

ROOT = Path(__file__).resolve().parents[2]
BINDING = {"sha": "a" * 40, "tree": "b" * 40}  # synthetic identities only


class FakeCanonicalError(ValueError):
    pass


def toy_row(label="9", family="negative_binomial", *, prefix_start=0):
    cell = f"{family}|toy={label}|n=3|CVM|composite"
    identity = f"{cell}|raw_outer=0"
    return {
        "outer_identity": identity, "cell_id": cell, "raw_outer_index": 0,
        "observed_identity": identity, "r9_b_cpu": 1, "r9_b_cuda": 2,
        "r9_p_cpu": 0.1, "r9_p_cuda": 0.2,
        "r9_reject_cpu": False, "r9_reject_cuda": False,
        "bootstrap_identities": [{"identity": f"{identity}|raw_inner={raw}",
                                  "raw_inner_index": raw}
                                 for raw in range(prefix_start, prefix_start + 15)],
    }


class ToyAdapters:
    """No RNG, distributions, fitting, SciPy, canonical cells or real samples."""
    def __init__(self, *, skip=(), all_ineligible=False, fail_at=None,
                 failure_type=FakeCanonicalError, failure_message="BROKEN",
                 engine="CPU_REFERENCE"):
        self.skip = set(skip)
        self.all_ineligible = all_ineligible
        self.fail_at = fail_at
        self.failure_type = failure_type
        self.failure_message = failure_message
        self.engine = engine
        self.seed_calls = []
        self.generate_calls = []
        self.observed_calls = []
        self.fit_calls = 0

    def observed(self, row, namespace):
        self.observed_calls.append((row["outer_identity"], namespace))
        return np.array([1, 2, 3], dtype="<i2"), 42

    def derive(self, *args):
        self.seed_calls.append(args)
        return 100 + args[-1]

    def generate(self, row, parameters, seed):
        self.generate_calls.append((row["outer_identity"], parameters, seed))
        return np.array([seed, 2, 3], dtype="<i2")

    def fit(self, family, sample):
        self.fit_calls += 1
        raw = int(sample[0]) - 100
        if raw >= 0:
            if raw == self.fail_at:
                raise self.failure_type(self.failure_message)
            if self.all_ineligible or raw in self.skip:
                raise FakeCanonicalError("NB_NOT_ASSESSED: fixed toy classification")
        return {"engine": self.engine, "parameters": {"toy_parameter": 7}}

    def adapters(self):
        return CPUAdapters(self.observed, self.fit, self.derive, self.generate,
                           FakeCanonicalError)


def toy_build(rows=None, fake=None, *, binding=BINDING):
    rows = rows if rows is not None else [toy_row()]
    fake = fake if fake is not None else ToyAdapters()
    source = SourceSurface.synthetic_for_testing(rows)
    data = ReferenceWorkloadBuilder().build(source, fake.adapters(), builder_binding=binding)
    return data, load_artifact(data, allow_synthetic=True, require_binding=binding is not None), fake


def historical(workload):
    result = []
    for outer in workload["outers"]:
        for index, item in enumerate([outer["observed"]] + outer["accepted"][:15]):
            result.append({
                "identity": item["identity"], "cell_id": outer["cell_id"],
                "raw_outer_index": outer["raw_outer_index"],
                "record_type": "observed" if index == 0 else "bootstrap",
                "raw_inner_index": None if index == 0 else item["raw_inner_index"],
                "seed_identity": item["seed_identity"],
                "sample_digest": item["payload"]["sample_digest"],
            })
    return result


class PayloadTests(unittest.TestCase):
    def test_cpu_only_import_without_cupy(self):
        code = """
import importlib.abc, sys
class NoCUDA(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cupy', 'cupyx'):
            raise AssertionError('CUDA import attempted')
sys.meta_path.insert(0, NoCUDA())
from experiments.distribution_gof.cuda_calibration.r11_reference_workload import builder, schema, oracle, binding
assert not any(name.startswith(('cupy', 'cupyx')) for name in sys.modules)
assert 'experiments.distribution_gof.cuda_calibration.cp05_cuda_engine' not in sys.modules
assert 'experiments.distribution_gof.cuda_calibration.cp05_c2c_equivalence_runner' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_c_contiguous_bytes(self):
        array = np.arange(12, dtype="<i2").reshape(3, 4).T[:, ::-1]
        canonical = canonicalize_sample(array)
        self.assertTrue(canonical.flags.c_contiguous)
        self.assertEqual(sample_bytes(array), np.ascontiguousarray(array).tobytes(order="C"))

    def test_dtype_shape_endianness_and_roundtrip(self):
        for dtype in ("<i2", ">i2", "<f8"):
            with self.subTest(dtype=dtype):
                array = np.array([[1, 2], [3, 4]], dtype=dtype).T
                payload = SamplePayload.from_sample(array)
                loaded = SamplePayload.from_serialized(payload.serialize()).deserialize()
                self.assertEqual(loaded.dtype.str, array.dtype.str)
                self.assertEqual(loaded.shape, array.shape)
                self.assertEqual(loaded.tobytes(), sample_bytes(array))
                self.assertFalse(loaded.flags.writeable)

    def test_known_answer_digest(self):
        array = np.array([1, 256, -2], dtype="<i2")
        self.assertEqual(sample_bytes(array), bytes.fromhex("01000001feff"))
        self.assertEqual(sample_digest(array),
                         "41c7d51c9d64bbec8a8f0068d4e180eded8a06bcc2b1bce770a69646e1817c02")

    def test_payload_tamper_blocks_deserialization(self):
        value = SamplePayload.from_sample(np.array([1, 2], dtype="<i2")).serialize()
        value["payload_base64"] = base64.b64encode(b"\0" * 4).decode()
        with self.assertRaisesRegex(ContractError, "sample digest mismatch"):
            SamplePayload.from_serialized(value).deserialize()

    def test_shape_dtype_and_digest_tamper(self):
        original = SamplePayload.from_sample(np.array([1, 2], dtype="<i2")).serialize()
        for key, value in (("shape", [3]), ("dtype_str", "<i8"), ("sample_digest", "0" * 64),
                           ("shape", [True]), ("dtype_str", "int16"),
                           ("payload_base64", "!")):
            with self.subTest(key=key, value=value):
                item = {**original, key: value}
                with self.assertRaises(ContractError):
                    SamplePayload.from_serialized(item)

    def test_object_structured_and_device_protocol_rejected(self):
        class DeviceLike:
            def __array__(self):
                raise AssertionError("device transfer forbidden")
        for item in (np.array([object()], dtype=object),
                     np.array([(1,)], dtype=[("x", "i4")]), DeviceLike()):
            with self.assertRaises(ContractError):
                SamplePayload.from_sample(item)

    def test_empty_and_scalar_payloads_follow_canonical_array(self):
        for array in (np.array([], dtype="<i4"), np.array(2, dtype="<i4")):
            payload = SamplePayload.from_sample(array)
            loaded = payload.deserialize()
            self.assertEqual(loaded.shape, array.shape)
            self.assertEqual(sample_bytes(loaded), sample_bytes(array))

    def test_input_mutation_cannot_change_frozen_payload(self):
        array = np.array([1, 2], dtype="<i2")
        payload = SamplePayload.from_sample(array)
        array[:] = 99
        self.assertEqual(payload.deserialize().tolist(), [1, 2])

    def test_strict_json_nan_duplicates_noncanonical(self):
        for data in (b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}',
                     b'{"x":1,"x":2}', b'{ "x":1}', b'{"x":1}\n'):
            with self.subTest(data=data), self.assertRaises(ContractError):
                strict_json(data, canonical=True)
        with self.assertRaises(ValueError):
            canonical_json({"x": float("nan")})

    def test_write_once_no_overwrite_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            path = Path(directory) / "toy.json"
            self.assertEqual(write_once(path, b"toy"), sha256(b"toy"))
            with self.assertRaises(FileExistsError):
                write_once(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"toy")


class SourceTests(unittest.TestCase):
    def test_documentary_source_projection_known_answer_only(self):
        # Read existing documentary identities; no adapter is invoked.
        raw = (ROOT / SOURCE_PATH).read_bytes()
        surface = SourceSurface.frozen(raw)
        projected = canonical_json(projection(surface.rows()))
        self.assertEqual(sha256(raw), SOURCE_HASH)
        self.assertEqual(len(projected), PROJECTION_BYTES)
        self.assertEqual(sha256(projected), PROJECTION_HASH)

    def test_source_order_preserved_without_sorting(self):
        rows = [toy_row("9"), toy_row("1")]
        surface = SourceSurface.synthetic_for_testing(rows)
        self.assertEqual([r["cell_id"] for r in surface.rows()], [r["cell_id"] for r in rows])
        self.assertEqual(projection(surface.rows()), projection(rows))

    def test_source_tamper_and_reordering_fail_frozen_gate(self):
        raw = (ROOT / SOURCE_PATH).read_bytes()
        value = strict_json(raw)
        value["mc_failed_outers"].reverse()
        with self.assertRaises(ContractError):
            SourceSurface.frozen(canonical_json(value))
        with self.assertRaises(ContractError):
            SourceSurface.frozen(raw + b" ")

    def test_malformed_duplicate_source_never_skipped(self):
        for rows in ([toy_row(), toy_row()], [{**toy_row(), "observed_identity": "wrong"}],
                     [{**toy_row(), "raw_outer_index": True}]):
            with self.assertRaises(ContractError):
                SourceSurface.synthetic_for_testing(rows)


class BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.workload, cls.fake = toy_build([toy_row("9"), toy_row("1")])

    def test_fixed_configuration_and_exact_count(self):
        self.assertEqual((ALPHA, B_R11, NB_RETRY_CAP), (0.05, 199, 19900))
        for outer in self.workload["outers"]:
            self.assertEqual((outer["accepted_count"], outer["attempt_count"], outer["retry_cap"]),
                             (199, 199, 19900))

    def test_prospective_canonical_seed_call(self):
        for index, args in enumerate(self.fake.seed_calls):
            outer = self.workload["outers"][index // 199]
            self.assertEqual(args, ("CP05-C2C", outer["cell_id"], 0, "inner_bootstrap", index % 199))

    def test_accepted_ordinals_and_no_posthoc_sorting(self):
        self.assertEqual([row["cell_id"] for row in self.workload["outers"]],
                         [toy_row("9")["cell_id"], toy_row("1")["cell_id"]])
        for outer in self.workload["outers"]:
            self.assertEqual([r["accepted_ordinal"] for r in outer["accepted"]], list(range(199)))
            self.assertEqual([r["raw_inner_index"] for r in outer["accepted"]], list(range(199)))

    def test_ineligible_nb_accounting_and_gap_order(self):
        _, work, fake = toy_build([toy_row(prefix_start=1)], ToyAdapters(skip={0, 2, 4}))
        outer = work["outers"][0]
        self.assertEqual((outer["attempt_count"], outer["accepted_count"]), (202, 199))
        self.assertEqual([r["raw_inner_index"] for r in outer["accepted"][:4]], [1, 3, 5, 6])
        self.assertEqual(sum(r["canonical_eligibility_status"] == "INELIGIBLE"
                             for r in outer["attempts"]), 3)
        self.assertEqual(len(fake.generate_calls), 202)

    def test_retry_cap_exhaustion_preserves_fixed_toy_evidence(self):
        builder = ReferenceWorkloadBuilder()
        fake = ToyAdapters(all_ineligible=True)
        with self.assertRaises(BuildFailure) as caught:
            builder.build(SourceSurface.synthetic_for_testing([toy_row()]), fake.adapters())
        work = load_artifact(caught.exception.artifact_bytes, allow_synthetic=True,
                             require_complete=False, require_binding=False)
        outer = work["outers"][0]
        self.assertEqual(len(fake.generate_calls), 19900)
        self.assertEqual((outer["attempt_count"], outer["accepted_count"]), (19900, 0))
        self.assertTrue(outer["retry_cap_reached"])
        self.assertEqual(work["failure"]["stage"], "retry_cap")
        with self.assertRaisesRegex(ContractError, "incomplete workload"):
            load_artifact(caught.exception.artifact_bytes, allow_synthetic=True, require_binding=False)
        with self.assertRaisesRegex(ContractError, "consumed"):
            builder.build(SourceSurface.synthetic_for_testing([toy_row()]), fake.adapters())

    def test_non_nb_canonical_failure_and_arbitrary_exception_stop(self):
        for exception, message in ((FakeCanonicalError, "OTHER_FAILURE"),
                                   (RuntimeError, "NB_NOT_ASSESSED: spoof")):
            with self.subTest(exception=exception):
                builder = ReferenceWorkloadBuilder()
                fake = ToyAdapters(fail_at=2, failure_type=exception, failure_message=message)
                with self.assertRaises(BuildFailure) as caught:
                    builder.build(SourceSurface.synthetic_for_testing([toy_row()]), fake.adapters())
                work = load_artifact(caught.exception.artifact_bytes, allow_synthetic=True,
                                     require_complete=False, require_binding=False)
                outer = work["outers"][0]
                self.assertEqual((outer["attempt_count"], outer["accepted_count"]), (3, 2))
                self.assertEqual(len(fake.generate_calls), 3)
                self.assertEqual(outer["attempts"][-1]["canonical_eligibility_status"], "FAILED")
                self.assertIsNotNone(outer["attempts"][-1]["sample_digest"])
                with self.assertRaisesRegex(ContractError, "consumed"):
                    builder.build(SourceSurface.synthetic_for_testing([toy_row()]), fake.adapters())

    def test_nb_status_for_non_nb_family_fails(self):
        fake = ToyAdapters(skip={0})
        with self.assertRaises(BuildFailure):
            toy_build([toy_row(family="gamma")], fake)
        self.assertEqual(len(fake.generate_calls), 1)

    def test_gpu_eligibility_marker_rejected(self):
        with self.assertRaises(BuildFailure):
            toy_build(fake=ToyAdapters(engine="CUDA_CANDIDATE"))
        adapters = ToyAdapters().adapters()
        object.__setattr__(adapters, "engine", "CUDA_CANDIDATE")
        with self.assertRaisesRegex(ContractError, "CPU reference"):
            ReferenceWorkloadBuilder().build(SourceSurface.synthetic_for_testing([toy_row()]), adapters)

    def test_observed_fit_failure_stops_before_generation(self):
        fake = ToyAdapters()
        def broken(*args):
            raise FakeCanonicalError("NB_NOT_ASSESSED: observed fixture")
        adapters = CPUAdapters(fake.observed, broken, fake.derive, fake.generate, FakeCanonicalError)
        with self.assertRaises(BuildFailure) as caught:
            ReferenceWorkloadBuilder().build(SourceSurface.synthetic_for_testing([toy_row()]), adapters)
        work = strict_json(caught.exception.artifact_bytes)["workload"]
        self.assertIsNotNone(work["outers"][0]["observed"])
        self.assertEqual(fake.generate_calls, [])
        self.assertEqual(work["failure"]["stage"], "observed_reference_fit")

    def test_complete_builder_cannot_rerun(self):
        builder = ReferenceWorkloadBuilder()
        source = SourceSurface.synthetic_for_testing([toy_row()])
        fake = ToyAdapters()
        builder.build(source, fake.adapters())
        with self.assertRaisesRegex(ContractError, "consumed"):
            builder.build(source, fake.adapters())

    def test_seed_generation_and_canonicalization_failure_preserve_attempt(self):
        for phase in ("derive_seed", "generate", "canonicalize"):
            fake = ToyAdapters()
            def broken(*args):
                raise RuntimeError("fixed toy failure")
            adapters = CPUAdapters(
                fake.observed, fake.fit,
                broken if phase == "derive_seed" else fake.derive,
                broken if phase == "generate" else (
                    (lambda *args: [1, 2, 3]) if phase == "canonicalize" else fake.generate),
                FakeCanonicalError,
            )
            with self.subTest(phase=phase), self.assertRaises(BuildFailure) as caught:
                ReferenceWorkloadBuilder().build(SourceSurface.synthetic_for_testing([toy_row()]), adapters)
            work = load_artifact(caught.exception.artifact_bytes, allow_synthetic=True,
                                 require_complete=False, require_binding=False)
            attempt = work["outers"][0]["attempts"][0]
            self.assertEqual(work["failure"]["stage"], phase)
            self.assertEqual(attempt["seed_identity"], None if phase == "derive_seed" else 100)
            self.assertIsNone(attempt["sample_digest"])

    def test_bootstrap_cuda_marker_cannot_define_eligibility(self):
        fake = ToyAdapters()
        def fit(family, sample):
            return {"engine": "CPU_REFERENCE" if int(sample[0]) == 1 else "CUDA_CANDIDATE",
                    "parameters": {"toy_parameter": 7}}
        adapters = CPUAdapters(fake.observed, fit, fake.derive, fake.generate, FakeCanonicalError)
        with self.assertRaises(BuildFailure) as caught:
            ReferenceWorkloadBuilder().build(SourceSurface.synthetic_for_testing([toy_row()]), adapters)
        work = load_artifact(caught.exception.artifact_bytes, allow_synthetic=True,
                             require_complete=False, require_binding=False)
        self.assertEqual(work["outers"][0]["accepted_count"], 0)
        self.assertEqual(work["failure"]["stage"], "bootstrap_reference_fit")

    def test_reference_cannot_mutate_frozen_sample(self):
        fake = ToyAdapters()
        def mutating_fit(family, sample):
            sample[0] = 999
        adapters = CPUAdapters(fake.observed, mutating_fit, fake.derive, fake.generate, FakeCanonicalError)
        with self.assertRaises(BuildFailure) as caught:
            ReferenceWorkloadBuilder().build(SourceSurface.synthetic_for_testing([toy_row()]), adapters)
        work = load_artifact(caught.exception.artifact_bytes, allow_synthetic=True,
                             require_complete=False, require_binding=False)
        self.assertEqual(SamplePayload.from_serialized(work["outers"][0]["observed"]["payload"])
                         .deserialize().tolist(), [1, 2, 3])

    def test_malformed_reference_fit_never_silently_accepted(self):
        for malformed in (None, {"engine": "CPU_REFERENCE"},
                          {"engine": "CPU_REFERENCE", "parameters": []}):
            fake = ToyAdapters()
            def fit(family, sample):
                return fake.fit(family, sample) if int(sample[0]) == 1 else malformed
            adapters = CPUAdapters(fake.observed, fit, fake.derive, fake.generate, FakeCanonicalError)
            with self.subTest(malformed=malformed), self.assertRaises(BuildFailure) as caught:
                ReferenceWorkloadBuilder().build(SourceSurface.synthetic_for_testing([toy_row()]), adapters)
            work = load_artifact(caught.exception.artifact_bytes, allow_synthetic=True,
                                 require_complete=False, require_binding=False)
            self.assertEqual(work["outers"][0]["accepted_count"], 0)
            self.assertEqual(len(fake.generate_calls), 1)

    def test_preflight_rejection_consumes_instance_without_scientific_calls(self):
        fake = ToyAdapters()
        adapters = CPUAdapters(fake.observed, fake.fit, fake.derive, fake.generate, ValueError)
        builder = ReferenceWorkloadBuilder()
        source = SourceSurface.synthetic_for_testing([toy_row()])
        with self.assertRaisesRegex(ContractError, "generic built-in"):
            builder.build(source, adapters)
        with self.assertRaisesRegex(ContractError, "consumed"):
            builder.build(source, fake.adapters())
        self.assertEqual(fake.observed_calls, [])

    def test_last_permitted_toy_attempt_can_complete_without_extra_draw(self):
        _, work, fake = toy_build(fake=ToyAdapters(skip=range(19701)))
        outer = work["outers"][0]
        self.assertEqual((outer["attempt_count"], outer["accepted_count"]), (19900, 199))
        self.assertEqual(outer["accepted"][-1]["raw_inner_index"], 19899)
        self.assertTrue(outer["retry_cap_reached"])
        self.assertEqual(len(fake.generate_calls), 19900)

    def test_generator_uses_only_observed_cpu_parameters(self):
        self.assertTrue(all(params == {"toy_parameter": 7}
                            for _, params, _ in self.fake.generate_calls))

    def test_schema_roundtrip_exact_bytes_and_payload_order_digest(self):
        self.assertEqual(canonical_json(strict_json(self.data)), self.data)
        for outer in self.workload["outers"]:
            for item in [outer["observed"]] + outer["accepted"]:
                payload = SamplePayload.from_serialized(item["payload"])
                self.assertEqual(sample_digest(payload.deserialize()), payload.sample_digest)
        reverse = copy.deepcopy(self.workload)
        reverse["outers"].reverse()
        self.assertNotEqual(strict_json(artifact_document(reverse))["workload"]["ordered_workload_payload_digest"],
                            self.workload["ordered_workload_payload_digest"])

    def test_duplicate_identity_ordinal_attempt_and_sort_tamper(self):
        mutations = [
            lambda w: w["outers"][0]["accepted"][1].update(identity=w["outers"][0]["accepted"][0]["identity"]),
            lambda w: w["outers"][0]["accepted"][1].update(accepted_ordinal=0),
            lambda w: w["outers"][0]["attempts"].reverse(),
            lambda w: w["outers"][0]["accepted"].reverse(),
            lambda w: w["outers"].reverse(),
            lambda w: w.update(B_R11=15),
            lambda w: w.update(nb_retry_cap=20000),
            lambda w: w["outers"][0].update(accepted_count=198),
            lambda w: w["outers"][0]["attempts"][0].update(seed_identity=True),
            lambda w: w["outers"][0].update(retry_cap=199),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = copy.deepcopy(self.workload)
                mutate(value)
                with self.assertRaises(ContractError):
                    load_artifact(artifact_document(value), allow_synthetic=True)

    def test_incomplete_outer_rejected_even_with_recomputed_manifest_digest(self):
        value = copy.deepcopy(self.workload)
        value["outers"].pop()
        with self.assertRaisesRegex(ContractError, "missing source outer"):
            load_artifact(artifact_document(value), allow_synthetic=True)

    def test_digest_mismatch_blocks_downstream_bytes(self):
        value = strict_json(self.data)
        value["workload"]["outers"][0]["observed"]["payload"]["payload_base64"] = "AAAAAA=="
        with self.assertRaises(ContractError):
            load_artifact(canonical_json(value), allow_synthetic=True)

    def test_unknown_schema_keys_rejected(self):
        value = copy.deepcopy(self.workload)
        value["resume"] = True
        with self.assertRaisesRegex(ContractError, "schema keys"):
            load_artifact(artifact_document(value), allow_synthetic=True)

    def test_synthetic_artifact_cannot_enter_scientific_loader(self):
        with self.assertRaisesRegex(ContractError, "synthetic workload"):
            load_artifact(self.data)
        with self.assertRaises(oracle.OracleError):
            oracle.load_for_use(self.data, b"fake manifest", b"fake archive")

    def test_binding_placeholder_and_explicit_immutable_binding(self):
        data, _, _ = toy_build(binding=None)
        with self.assertRaisesRegex(ContractError, "binding required"):
            load_artifact(data, allow_synthetic=True)
        bound = bind_artifact(data, BINDING, allow_synthetic=True)
        self.assertEqual(load_artifact(bound, allow_synthetic=True)["builder_binding"], BINDING)
        with self.assertRaisesRegex(ContractError, "immutable"):
            bind_artifact(bound, BINDING, allow_synthetic=True)

    def test_git_binding_reads_tree_and_rejects_preexisting_mutation(self):
        from experiments.distribution_gof.cuda_calibration.r11_reference_workload.contract import SCIENTIFIC_TREE
        def fake_git(args, **kwargs):
            command = args[1:]
            if command[0] == "status":
                output = ""
            elif command[0] == "ls-tree":
                output = "experiments/frozen.py\n"
            elif command[0] == "diff":
                output = "experiments/new_builder.py\n"
            else:
                output = SCIENTIFIC_TREE if command[-1].startswith("d741") else (
                    BINDING["tree"] if command[-1].endswith("^{tree}") else BINDING["sha"])
            return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")
        with patch("experiments.distribution_gof.cuda_calibration.r11_reference_workload.binding.subprocess.run",
                   side_effect=fake_git):
            self.assertEqual(binding_from_git(ROOT), BINDING)
        def mutated(args, **kwargs):
            result = fake_git(args, **kwargs)
            if args[1] == "diff":
                result.stdout = "experiments/frozen.py\n"
            return result
        with patch("experiments.distribution_gof.cuda_calibration.r11_reference_workload.binding.subprocess.run",
                   side_effect=mutated), self.assertRaisesRegex(ContractError, "scientific code changed"):
            binding_from_git(ROOT)


class PrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [toy_row("9"), toy_row("1")]
        _, cls.workload, _ = toy_build(cls.rows)
        cls.history = historical(cls.workload)

    def test_mock_prefix_match_pass(self):
        self.assertTrue(oracle.compare_prefix(self.workload, self.rows, self.history))

    def test_mock_prefix_with_raw_gaps_matches_eligible_order(self):
        row = toy_row()
        raw_indices = [raw for raw in range(18) if raw not in (0, 2, 4)][:15]
        row["bootstrap_identities"] = [
            {"identity": f"{row['outer_identity']}|raw_inner={raw}", "raw_inner_index": raw}
            for raw in raw_indices]
        _, work, _ = toy_build([row], ToyAdapters(skip={0, 2, 4}))
        self.assertTrue(oracle.compare_prefix(work, [row], historical(work)))

    def test_changed_source_prefix_never_replaced_by_runtime_sequence(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["bootstrap_identities"][0]["raw_inner_index"] = 99
        with self.assertRaises(oracle.OracleError):
            oracle.compare_prefix(self.workload, rows, self.history)

    def test_mock_prefix_identity_raw_order_seed_and_digest_mismatch(self):
        for position, field, value in (
            (0, "identity", "wrong"), (0, "seed_identity", 43),
            (0, "sample_digest", "0" * 64), (1, "raw_inner_index", 1),
            (1, "identity", "wrong"), (1, "seed_identity", 999),
            (1, "sample_digest", "0" * 64), (1, "record_type", "observed"),
            (1, "cell_id", self.rows[1]["cell_id"]),
            (1, "raw_outer_index", True),
        ):
            history = copy.deepcopy(self.history)
            history[position][field] = value
            with self.subTest(field=field), self.assertRaises(oracle.OracleError):
                oracle.compare_prefix(self.workload, self.rows, history)

    def test_history_reordering_missing_and_duplicates_fail(self):
        for history in (list(reversed(self.history)), self.history[:-1],
                        [self.history[0]] + self.history[:-1]):
            with self.assertRaises(oracle.OracleError):
                oracle.compare_prefix(self.workload, self.rows, history)

    def test_artifact_hash_associations_never_substituted(self):
        for name, correct in ARTIFACT_HASHES.items():
            oracle.verify_artifact_digest(name, correct)
            for other, wrong in ARTIFACT_HASHES.items():
                if other != name:
                    with self.assertRaises(ContractError):
                        oracle.verify_artifact_digest(name, wrong)
        with self.assertRaises(ContractError):
            oracle.verify_artifact_digest("unknown", ARTIFACT_HASHES[ARCHIVE_NAME])

    def test_absent_or_wrong_oracle_fails_closed(self):
        for archive in (None, b"wrong archive"):
            with self.assertRaises(oracle.OracleError) as caught:
                oracle.runtime_records(archive)
            self.assertEqual(caught.exception.R4_RUNTIME_ORACLE_VERIFIED, "NO")
            self.assertEqual(caught.exception.R11_REFERENCE_WORKLOAD_ACCEPTED, "NO")
            self.assertEqual(caught.exception.GPU_EXECUTION, "PROHIBITED")

    def test_archive_parse_and_member_hash_gate_with_synthetic_archive(self):
        data = b"\n".join(canonical_json(row) for row in self.history) + b"\n"
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("toy/" + WORKLOAD_NAME)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        calls = []
        def fixed_verifier(name, actual):
            calls.append(name)
            self.assertEqual(actual, buffer.getvalue() if name == ARCHIVE_NAME else data)
        # This replaces only the byte-identity verifier for a synthetic archive.
        # Real mandatory hash associations are tested separately above.
        with patch.object(oracle, "verify_artifact", side_effect=fixed_verifier):
            self.assertEqual(oracle.runtime_records(buffer.getvalue()), self.history)
        self.assertEqual(calls, [ARCHIVE_NAME, WORKLOAD_NAME])

    def test_crossings_report_uses_own_byte_gate(self):
        with patch.object(oracle, "verify_artifact") as verify:
            verify.side_effect = lambda name, data: (
                None if name == ARCHIVE_NAME else oracle.verify_artifact_digest(name, sha256(data)))
            with self.assertRaises(oracle.OracleError):
                oracle.runtime_records(b"synthetic archive", crossings_bytes=b"wrong crossings")
            self.assertEqual(verify.call_args_list[1].args[0], CROSSINGS_NAME)

    def test_ambiguous_archive_members_and_malformed_jsonl_fail(self):
        for members, data in ((2, b"{}\n"), (1, b'{"x":NaN}\n'), (1, b"{}\n\n")):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                for index in range(members):
                    info = tarfile.TarInfo(f"toy{index}/" + WORKLOAD_NAME)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            with self.subTest(members=members, data=data):
                with patch.object(oracle, "verify_artifact"), self.assertRaises(oracle.OracleError):
                    oracle.runtime_records(buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
