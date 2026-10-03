"""Versioned canonical CPU wiring; delegate to the frozen scientific functions.

Imports of the optional-CuPy engine are lazy. None of these wrappers starts a
workload or implements RNG, fitting, parameter, or eligibility mathematics.
"""
from __future__ import annotations

from pathlib import Path

from .codec import integer, require, text_identity
from .contract import NAMESPACE


def _canonical_engine():
    from .. import cp05_cuda_engine
    require(Path(cp05_cuda_engine.__file__).resolve()
            == Path(__file__).resolve().parents[1] / "cp05_cuda_engine.py",
            "canonical engine loaded outside the builder repository")
    return cp05_cuda_engine


def _canonical_cells():
    from .. import equivalence_preregistration
    require(Path(equivalence_preregistration.__file__).resolve()
            == Path(__file__).resolve().parents[1] / "equivalence_preregistration.py",
            "canonical fixtures loaded outside the builder repository")
    cells = equivalence_preregistration.primary_fixture_matrix()
    index = {cell.canonical_id: cell for cell in cells}
    require(len(index) == len(cells), "duplicate canonical cell identity")
    return index


def _resolve_cell(cell_id):
    text_identity(cell_id)
    cell = _canonical_cells().get(cell_id)
    require(cell is not None and cell.canonical_id == cell_id,
            "unknown or mismatching canonical cell_id")
    return cell


def _resolve_outer(row):
    require(type(row) is dict, "invalid canonical outer")
    cell = _resolve_cell(row["cell_id"])
    raw_outer = integer(row["raw_outer_index"])
    identity = f"{cell.canonical_id}|raw_outer={raw_outer}"
    require(row["outer_identity"] == identity and row["observed_identity"] == identity,
            "mismatching canonical outer identity")
    # The frozen projection has neither field; if supplied, metadata cannot
    # override the accepted cell definition.
    if "family" in row:
        require(row["family"] == cell.family, "mismatching canonical family")
    if "n" in row:
        require(type(row["n"]) is int and row["n"] == cell.n,
                "mismatching canonical sample size")
    return cell


class CanonicalCPUAdapters:
    """No constructor arguments or per-instance scientific overrides."""
    __slots__ = ()
    engine = "CPU_REFERENCE"

    @property
    def canonical_error_type(self):
        return _canonical_engine().EngineContractError

    def verify(self):
        # Import/reference identity only; no generator, fit, or CUDA runtime call.
        require(self.canonical_error_type is _canonical_engine().EngineContractError,
                "non-canonical error type")

    def observed(self, row, namespace):
        require(namespace == NAMESPACE, "non-canonical observed namespace")
        cell = _resolve_outer(row)
        canonical = _canonical_engine()
        seed = canonical.derive_seed(namespace, cell.canonical_id,
                                     row["raw_outer_index"], "outer_observed")
        sample = canonical._generate(cell.family, dict(cell.parameters), cell.n, seed)
        return sample, seed

    def reference_fit(self, family, sample):
        canonical = _canonical_engine()
        fit = canonical._cp04_fit(family, sample)
        bound = fit.fitted_distribution
        return {"engine": "CPU_REFERENCE", "bound": bound,
                "parameters": canonical._parameters(bound)}

    def derive_seed(self, namespace, cell_id, raw_outer_index, purpose, raw_inner_index):
        require(namespace == NAMESPACE and purpose == "inner_bootstrap",
                "non-canonical bootstrap seed purpose/namespace")
        _resolve_cell(cell_id)
        integer(raw_outer_index)
        integer(raw_inner_index)
        return _canonical_engine().derive_seed(
            namespace, cell_id, raw_outer_index, purpose, raw_inner_index)

    def generate(self, row, fitted_parameters, seed):
        cell = _resolve_outer(row)
        return _canonical_engine()._generate(
            cell.family, fitted_parameters, cell.n, seed)
