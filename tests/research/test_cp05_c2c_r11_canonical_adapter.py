"""Canonical wiring software tests; no real FROZEN_R11 construction is invoked."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.distribution_gof.cuda_calibration.r11_reference_workload import (
    binding, canonical_adapter,
)
from experiments.distribution_gof.cuda_calibration.r11_reference_workload.builder import (
    ReferenceWorkloadBuilder,
)
from experiments.distribution_gof.cuda_calibration.r11_reference_workload.codec import (
    ContractError,
)
from experiments.distribution_gof.cuda_calibration.r11_reference_workload.source import (
    SourceSurface,
)
from tests.research.test_cp05_c2c_r11_reference_workload import (
    BINDING, ToyAdapters, toy_build, toy_row,
)

ROOT = Path(__file__).resolve().parents[2]
ENGINE_MODULE = "experiments.distribution_gof.cuda_calibration.cp05_cuda_engine"


@dataclass(frozen=True)
class TinyParameters:
    scale: float


def definition_only_row():
    # Metadata from an existing non-R11 exponential fixture; no sample is
    # reconstructed from this cell. Scientific calls below use patched delegates.
    from experiments.distribution_gof.cuda_calibration.equivalence_preregistration import (
        primary_fixture_matrix,
    )
    cell = next(cell for cell in primary_fixture_matrix()
                if cell.family == "exponential" and cell.statistic == "CVM" and cell.n == 20)
    return cell, {"cell_id": cell.canonical_id, "raw_outer_index": 7,
                  "outer_identity": f"{cell.canonical_id}|raw_outer=7",
                  "observed_identity": f"{cell.canonical_id}|raw_outer=7"}


class ConstructionSeparationTests(unittest.TestCase):
    def test_frozen_rejects_every_external_scientific_binding_before_construction(self):
        # Invalid synthetic bytes deliberately stand in for a frozen manifest;
        # rejection must precede source loading and all scientific calls.
        source = SourceSurface("FROZEN_R11", b"fixed non-scientific placeholder")
        external = [ToyAdapters().adapters(), canonical_adapter.CanonicalCPUAdapters(), object()]
        for supplied in external:
            with self.subTest(supplied=type(supplied).__name__):
                with patch.object(SourceSurface, "rows", side_effect=AssertionError("source used")), (
                    patch.object(canonical_adapter, "CanonicalCPUAdapters",
                                 side_effect=AssertionError("factory used"))
                ):
                    with self.assertRaisesRegex(ContractError, "prohibits external adapter injection"):
                        ReferenceWorkloadBuilder().build(source, supplied)

    def test_synthetic_still_requires_and_permits_explicit_fake_adapters(self):
        _, workload, fake = toy_build()
        self.assertEqual(workload["source_kind"], "SYNTHETIC_TEST")
        self.assertEqual(len(fake.generate_calls), 199)
        with self.assertRaisesRegex(ContractError, "requires explicit fake adapters"):
            ReferenceWorkloadBuilder().build(SourceSurface.synthetic_for_testing([toy_row()]))

    def test_frozen_selects_internal_concrete_factory_without_scientific_execution(self):
        source = SourceSurface("FROZEN_R11", b"fixed non-scientific placeholder")
        # Stop immediately at factory selection; do not traverse source outers.
        with patch.object(canonical_adapter, "CanonicalCPUAdapters",
                          side_effect=RuntimeError("factory-selection fixture")) as factory:
            with self.assertRaisesRegex(RuntimeError, "factory-selection fixture"):
                ReferenceWorkloadBuilder().build(source)
            factory.assert_called_once_with()

    def test_unsupported_source_and_overriding_source_subclass_fail_closed(self):
        class OverriddenSource(SourceSurface):
            def rows(self):
                raise AssertionError("override used")
        for source in (SourceSurface("UNKNOWN", b"toy"), OverriddenSource("SYNTHETIC_TEST", b"toy")):
            with self.assertRaises(ContractError):
                ReferenceWorkloadBuilder().build(source, ToyAdapters().adapters())

    def test_frozen_binding_must_match_internal_version_before_any_adapter_call(self):
        # This is a preflight-only software stub: no real frozen bytes/rows,
        # scientific samples, or numerical functions are present.
        source = SourceSurface("FROZEN_R11", b"fixed non-scientific placeholder")
        fake_runtime = {"python": "toy", "numpy": "toy", "scipy": "toy"}
        with patch.object(SourceSurface, "rows", return_value=[]), (
            patch("experiments.distribution_gof.cuda_calibration.r11_reference_workload.builder.environment",
                  return_value=fake_runtime)
        ), patch.object(binding, "binding_from_git", return_value=BINDING), (
            patch.object(canonical_adapter.CanonicalCPUAdapters, "verify",
                         side_effect=AssertionError("adapters used before identity gate"))
        ):
            with self.assertRaisesRegex(ContractError, "binding does not match"):
                ReferenceWorkloadBuilder().build(
                    source, builder_binding={"sha": "c" * 40, "tree": "d" * 40})

    def test_canonical_instance_accepts_no_scientific_override_arguments(self):
        with self.assertRaises(TypeError):
            canonical_adapter.CanonicalCPUAdapters(observed=lambda *args: None)
        adapter = canonical_adapter.CanonicalCPUAdapters()
        for field in ("observed", "reference_fit", "derive_seed", "generate", "canonical_error_type"):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(adapter, field, lambda *args: None)

    def test_frozen_identity_gate_precedes_canonical_verification_without_construction(self):
        source = SourceSurface("FROZEN_R11", b"fixed non-scientific placeholder")
        fake_runtime = {"python": "toy", "numpy": "toy", "scipy": "toy"}
        events = []
        def identify(repository):
            events.append("binding")
            self.assertEqual(repository, ROOT)
            return BINDING
        def stop_at_verify():
            events.append("verify")
            raise RuntimeError("preflight-only fixture stop")
        with patch.object(SourceSurface, "rows", return_value=[]), (
            patch("experiments.distribution_gof.cuda_calibration.r11_reference_workload.builder.environment",
                  return_value=fake_runtime)
        ), patch.object(binding, "binding_from_git", side_effect=identify), (
            patch.object(canonical_adapter.CanonicalCPUAdapters, "verify", side_effect=stop_at_verify)
        ):
            with self.assertRaisesRegex(RuntimeError, "preflight-only fixture stop"):
                ReferenceWorkloadBuilder().build(source)
        self.assertEqual(events, ["binding", "verify"])

    def test_import_is_lazy_and_canonical_wiring_works_without_cupy(self):
        code = """
import importlib.abc, sys
class WithoutCUDA(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cupy', 'cupyx'):
            raise ModuleNotFoundError('CUDA deliberately unavailable')
sys.meta_path.insert(0, WithoutCUDA())
from experiments.distribution_gof.cuda_calibration.r11_reference_workload import canonical_adapter
assert 'experiments.distribution_gof.cuda_calibration.cp05_cuda_engine' not in sys.modules
adapter = canonical_adapter.CanonicalCPUAdapters()
adapter.verify()
from experiments.distribution_gof.cuda_calibration import cp05_cuda_engine
assert adapter.canonical_error_type is cp05_cuda_engine.EngineContractError
assert cp05_cuda_engine.cp is None
assert not any(name.startswith(('cupy', 'cupyx')) for name in sys.modules)
assert 'experiments.distribution_gof.cuda_calibration.cp05_c2c_equivalence_runner' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class CanonicalDelegationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Canonical engine import only; guarantee optional CuPy is unavailable.
        # Its mathematical fit/generate functions are never invoked here.
        with patch.dict(sys.modules, {"cupy": None, "cupyx": None}):
            cls.canonical = importlib.import_module(ENGINE_MODULE)
        cls.cell, cls.row = definition_only_row()

    def setUp(self):
        self.adapter = canonical_adapter.CanonicalCPUAdapters()
        self.sample = np.array([1.0, 2.0, 3.0], dtype="<f8")

    def test_exact_canonical_error_class_exposed(self):
        self.assertIs(self.adapter.canonical_error_type, self.canonical.EngineContractError)
        self.adapter.verify()

    def test_observed_uses_canonical_seed_and_generator_without_transform(self):
        with patch.object(self.canonical, "derive_seed", return_value=123) as derive, (
            patch.object(self.canonical, "_generate", return_value=self.sample)
        ) as generate:
            sample, seed = self.adapter.observed(self.row, "CP05-C2C")
        derive.assert_called_once_with("CP05-C2C", self.cell.canonical_id, 7, "outer_observed")
        generate.assert_called_once_with(self.cell.family, dict(self.cell.parameters), self.cell.n, 123)
        self.assertIs(sample, self.sample)
        self.assertEqual(seed, 123)

    def test_observed_namespace_and_cell_mismatch_never_call_generator(self):
        for row, namespace in (
            (self.row, "WRONG_NAMESPACE"),
            ({**self.row, "cell_id": "unknown"}, "CP05-C2C"),
            ({**self.row, "outer_identity": "wrong"}, "CP05-C2C"),
            ({**self.row, "observed_identity": "wrong"}, "CP05-C2C"),
            ({**self.row, "family": "gamma"}, "CP05-C2C"),
            ({**self.row, "n": 3}, "CP05-C2C"),
            ({**self.row, "n": True}, "CP05-C2C"),
        ):
            with patch.object(self.canonical, "_generate") as generate:
                with self.subTest(row=row, namespace=namespace), self.assertRaises(ContractError):
                    self.adapter.observed(row, namespace)
                generate.assert_not_called()

    def test_reference_fit_and_parameter_extraction_delegate_exactly(self):
        bound = SimpleNamespace(parameters=TinyParameters(scale=1.25))
        parameters = {"scale": 1.25}
        fit = SimpleNamespace(fitted_distribution=bound)
        with patch.object(self.canonical, "_cp04_fit", return_value=fit) as cp04, (
            patch.object(self.canonical, "_parameters", return_value=parameters)
        ) as extract:
            result = self.adapter.reference_fit("exponential", self.sample)
        cp04.assert_called_once()
        self.assertEqual(cp04.call_args.args[0], "exponential")
        self.assertIs(cp04.call_args.args[1], self.sample)
        extract.assert_called_once_with(bound)
        self.assertEqual(set(result), {"engine", "bound", "parameters"})
        self.assertEqual(result["engine"], "CPU_REFERENCE")
        self.assertIs(result["bound"], bound)
        self.assertIs(result["parameters"], parameters)

    def test_existing_parameter_extraction_on_tiny_fixed_dataclass(self):
        # Existing canonical parameter utility, with no scientific fit.
        bound = SimpleNamespace(parameters=TinyParameters(scale=1.25))
        with patch.object(self.canonical, "_cp04_fit",
                          return_value=SimpleNamespace(fitted_distribution=bound)):
            self.assertEqual(self.adapter.reference_fit("exponential", self.sample)["parameters"],
                             {"scale": 1.25})

    def test_reference_exceptions_propagate_without_translation(self):
        for error in (self.canonical.EngineContractError("NB_NOT_ASSESSED:toy"),
                      RuntimeError("NB_NOT_ASSESSED:generic toy failure")):
            with patch.object(self.canonical, "_cp04_fit", side_effect=error), (
                patch.object(self.canonical, "_parameters")
            ) as extract:
                with self.subTest(error=type(error).__name__), self.assertRaises(type(error)) as caught:
                    self.adapter.reference_fit("negative_binomial", self.sample)
                self.assertIs(caught.exception, error)
                extract.assert_not_called()

    def test_bootstrap_generate_passes_parameter_object_seed_family_and_n_unchanged(self):
        parameters = {"scale": 1.25}
        with patch.object(self.canonical, "_generate", return_value=self.sample) as generate:
            result = self.adapter.generate(self.row, parameters, 456)
        generate.assert_called_once_with(self.cell.family, parameters, self.cell.n, 456)
        self.assertIs(generate.call_args.args[1], parameters)
        self.assertIs(result, self.sample)

    def test_five_bootstrap_seed_arguments_delegate_exactly(self):
        args = ("CP05-C2C", self.cell.canonical_id, 7, "inner_bootstrap", 2)
        with patch.object(self.canonical, "derive_seed", return_value=789) as derive:
            self.assertEqual(self.adapter.derive_seed(*args), 789)
        derive.assert_called_once_with(*args)

    def test_actual_seed_utility_only_matches_direct_call_on_definition_metadata(self):
        # Hash identity only: no sample, RNG, fitting, or real R11 cell is used.
        args = ("CP05-C2C", self.cell.canonical_id, 7, "inner_bootstrap", 2)
        self.assertEqual(self.adapter.derive_seed(*args), self.canonical.derive_seed(*args))

    def test_alternative_bootstrap_namespace_purpose_or_unknown_cell_rejected(self):
        for args in (
            ("WRONG", self.cell.canonical_id, 7, "inner_bootstrap", 2),
            ("CP05-C2C", self.cell.canonical_id, 7, "outer_observed", 2),
            ("CP05-C2C", "unknown", 7, "inner_bootstrap", 2),
            ("CP05-C2C", self.cell.canonical_id, True, "inner_bootstrap", 2),
        ):
            with patch.object(self.canonical, "derive_seed") as derive:
                with self.subTest(args=args), self.assertRaises(ContractError):
                    self.adapter.derive_seed(*args)
                derive.assert_not_called()

    def test_unknown_bootstrap_cell_fails_before_generation(self):
        with patch.object(self.canonical, "_generate") as generate:
            with self.assertRaises(ContractError):
                self.adapter.generate({**self.row, "cell_id": "unknown"}, {"scale": 1.25}, 456)
            generate.assert_not_called()

    def test_duplicate_or_mismatching_fixture_definition_rejected(self):
        from experiments.distribution_gof.cuda_calibration import equivalence_preregistration
        with patch.object(equivalence_preregistration, "primary_fixture_matrix",
                          return_value=(self.cell, self.cell)):
            with self.assertRaisesRegex(ContractError, "duplicate canonical cell"):
                self.adapter.generate(self.row, {"scale": 1.25}, 456)

    def test_external_engine_or_fixture_module_location_rejected(self):
        from experiments.distribution_gof.cuda_calibration import equivalence_preregistration
        for module in (self.canonical, equivalence_preregistration):
            with patch.object(module, "__file__", str(ROOT / "external" / "fake.py")):
                with self.subTest(module=module.__name__), self.assertRaises(ContractError):
                    self.adapter.observed(self.row, "CP05-C2C")


if __name__ == "__main__":
    unittest.main()
