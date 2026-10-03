"""Lazy delegation to the accepted fixed-sample CPU/CUDA implementations."""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
from pathlib import Path
from types import SimpleNamespace
import sys

from ..r11_reference_workload.codec import require, sample_digest, strict_json
from .artifacts import snapshot
from .contract import PROJECT_ROOTS
from .preflight import git

PREFIX = "experiments.distribution_gof.cuda_calibration."
MODULES = {"runner": "cp05_c2c_equivalence_runner", "support": "nb_support",
           "cuda": "cuda_candidate", "preregistration": "equivalence_preregistration"}


def _project(name):
    return name.split(".")[0] in PROJECT_ROOTS


def _verify_file(repository, path):
    path = Path(path).resolve()
    require(path.is_relative_to(repository), "project import outside executing checkout")
    relative = path.relative_to(repository).as_posix()
    require(git(repository, "rev-parse", "HEAD:" + relative)
            == git(repository, "hash-object", "--path=" + relative, str(path)),
            "loaded project source differs from executing Git tree")


def verify_loaded_modules(repository):
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if not _project(name):
            continue
        file = getattr(module, "__file__", None)
        namespaces = list(getattr(module, "__path__", ()))
        require(file is not None or namespaces, "project module without physical origin")
        if file:
            _verify_file(repository, file)
        for directory in namespaces:
            require(Path(directory).resolve().is_relative_to(repository),
                    "project namespace outside executing checkout")
        paths[name] = str(Path(file).resolve()) if file else namespaces
    return paths


class _CheckoutGuard(importlib.abc.MetaPathFinder):
    def __init__(self, repository):
        self.repository = repository

    def find_spec(self, fullname, path=None, target=None):
        if not _project(fullname):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        require(spec is not None, "missing project module: " + fullname)
        if spec.origin:
            _verify_file(self.repository, spec.origin)
        for directory in spec.submodule_search_locations or ():
            require(Path(directory).resolve().is_relative_to(self.repository),
                    "foreign project import namespace")
        return spec


def _load_runtime(repository):
    verify_loaded_modules(repository)
    guard = _CheckoutGuard(repository)
    sys.meta_path.insert(0, guard)
    try:
        modules = {key: importlib.import_module(PREFIX + name) for key, name in MODULES.items()}
        for key, name in MODULES.items():
            require(Path(modules[key].__file__).resolve() == (
                repository / "experiments/distribution_gof/cuda_calibration" / (name + ".py")).resolve(),
                "wrong fixed-sample implementation origin")
        paths = verify_loaded_modules(repository)
    finally:
        sys.meta_path.remove(guard)
    return SimpleNamespace(**modules, loaded_module_paths=paths)


def verify_sample(sample, payload):
    require(sample.dtype.str == payload.dtype_str and sample.shape == payload.shape
            and sample_digest(sample) == payload.sample_digest,
            "sample dtype/shape/raw-byte identity mismatch")


class CanonicalRuntime:
    """No adapter parameters, generation calls or NumPy CUDA fallback."""
    __slots__ = ("repository", "modules", "cells", "cuda_interface")

    def __init__(self, repository):
        self.repository = Path(repository).resolve()
        self.modules = _load_runtime(self.repository)
        cells = self.modules.preregistration.primary_fixture_matrix()
        self.cells = {cell.canonical_id: cell for cell in cells}
        require(len(self.cells) == len(cells), "duplicate canonical cell")
        self.cuda_interface = None

    def require_gpu(self):
        self.cuda_interface = self.modules.cuda.require_cuda()
        require(self.cuda_interface is not None, "CUDA unavailable; CPU fallback prohibited")

    def environment(self):
        cp = self.cuda_interface
        require(cp is not None, "GPU preflight required")
        return {"cupy": cp.__version__, "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
                "device_count": cp.cuda.runtime.getDeviceCount(),
                "loaded_project_module_paths": self.modules.loaded_module_paths,
                "CUDA_CPU_FALLBACK": False}

    def verify_sources(self):
        verify_loaded_modules(self.repository)

    def evaluate(self, item, sample, payload, capture, before_cuda):
        require(self.cuda_interface is not None, "GPU preflight required; no CPU fallback")
        require(item["cell_id"] in self.cells, "unknown canonical cell")
        cell = self.cells[item["cell_id"]]
        runner = self.modules.runner
        capture["family"] = cell.family
        verify_sample(sample, payload)

        def reference_adapter(c, x):
            require(c is cell and x is sample, "CPU sample object identity mismatch")
            verify_sample(x, payload)
            capture.update(phase="cpu", cpu_input_identity_pass=True,
                           cpu_sample_digest=sample_digest(x))
            result = runner.evaluate_reference_record(c, x)
            capture["raw_cpu_result"] = strict_json(snapshot(result), canonical=True)
            capture["cpu_evaluation_points"] = list(result.get("evaluation_points", ()))
            verify_sample(x, payload)
            return result

        def cuda_adapter(c, x):
            require(c is cell and x is sample, "CUDA sample object identity mismatch")
            verify_sample(x, payload)
            support = None
            if cell.family == "negative_binomial":
                # Retrieve the live bound through the CPU closure, never from serialized evidence.
                support = self.modules.support.certify_nb_support(
                    x, cpu_result["bound"], cell.statistic)
                capture["statistic_support_evidence"] = strict_json(snapshot(support), canonical=True)
            verify_sample(x, payload)
            capture.update(phase="cuda", cuda_input_identity_pass=True,
                           cuda_sample_digest=sample_digest(x))
            before_cuda()  # consumed immediately before the first actual CUDA evaluation
            result = runner.evaluate_cuda_record(
                c, x, certified_support=support.indices if support else None,
                remainder_bound=support.remainder_bound if support else None)
            capture["raw_cuda_result"] = strict_json(snapshot(result), canonical=True)
            capture["cuda_evaluation_points"] = list(result.get("evaluation_points", ()))
            capture["cuda_solver_converged"] = result.get("solver_converged")
            verify_sample(x, payload)
            return result

        cpu_result = None

        def cpu(c, x):
            nonlocal cpu_result
            cpu_result = reference_adapter(c, x)
            return cpu_result

        result = runner.evaluate_fixed_record(
            identity=item["identity"], record_type=item["record_type"], cell=cell,
            raw_outer_index=item["raw_outer_index"], raw_inner_index=item["raw_inner_index"],
            sample=sample, reference_adapter=cpu, cuda_adapter=cuda_adapter)
        capture["delegated_record"] = strict_json(snapshot(result), canonical=True)
        for key in ("identity", "record_type", "cell_id", "raw_outer_index",
                    "raw_inner_index", "sample_digest"):
            require(key in result and type(result[key]) is type(item[key])
                    and result[key] == item[key], "delegated record identity drift")
        # Delegated gates and numerical results are retained unchanged.
        return {**result, **{k: v for k, v in item.items() if k not in result},
                **{k: v for k, v in capture.items() if k not in ("phase", "delegated_record")},
                "evaluation_points": list(runner.canonical_distribution_value_points(cell, sample))}
