"""DEC-014 development null matrix; independent of the frozen CP05-B manifest."""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
from itertools import product
import re
from types import MappingProxyType

from ..manifest import canonical_json, _canonical_decimal, SOURCE_REPOSITORY

BASE_SHA = "da407a629bdc8d16be775a7eb378f409972332dd"
BASE_TREE = "3afdac1758090b7b2463c90152bd5943e3712998"
PHASE = "CP05-C"
NAMESPACE = "pyMagicStats/STAGE-DIST-FAMILIES-001/CP05-C/v1"
SCHEMA = "cp05-c-development-null-v1"
OUTPUT_SCHEMA = "cp05-c-null-output-v1"
R_C = 2000
OUTER_ATTEMPT_MULTIPLIER_NB = 100
SHADOW_OUTER_RAW_CAP = SHADOW_INNER_RAW_CAP = 64
SHADOW_PERIOD = 500
ALPHA = 0.05
WILSON_UPPER_LIMIT = 0.065
SAMPLE_SIZES = (20, 50, 100, 250)
B_VALUES = (199, 999)
NULL_TYPES = ("simple", "composite")
PARAMETERS = {
    "gamma": tuple({"shape": s, "scale": "1"} for s in ("0.25", "0.5", "1", "2", "10")),
    "exponential": ({"scale": "1"},),
    "negative_binomial": tuple({"r": r, "p": p} for r, p in product(
        ("0.25", "1", "5", "20"), ("0.1", "0.5", "0.9"))),
}
STATISTICS = {
    "gamma": ("AD", "CVM", "KS"),
    "exponential": ("AD", "CVM", "KS"),
    "negative_binomial": ("AD", "CVM", "KS", "PEARSON"),
}


@dataclass(frozen=True, slots=True)
class DevelopmentManifest:
    family: str
    statistic: str
    n: int
    canonical_parameters: dict
    null_type: str
    B: int
    source_sha: str = BASE_SHA
    phase: str = PHASE
    namespace_id: str = NAMESPACE
    alpha: float = ALPHA
    R_C: int = R_C
    workers: int = 1
    batch_size: int = 1
    schema_version: str = SCHEMA
    output_schema_version: str = OUTPUT_SCHEMA
    source_repository: str = SOURCE_REPOSITORY

    def __post_init__(self):
        if (self.phase != PHASE or self.namespace_id != NAMESPACE
                or self.schema_version != SCHEMA or self.output_schema_version != OUTPUT_SCHEMA
                or self.source_repository != SOURCE_REPOSITORY):
            raise ValueError("only the CP05-C public development null contract is allowed")
        if type(self.source_sha) is not str or not re.fullmatch("[0-9a-f]{40}", self.source_sha):
            raise ValueError("source_sha must be a lowercase commit SHA")
        if type(self.R_C) is not int or self.R_C != R_C:
            raise ValueError("R_C is frozen at 2000; use explicit software fixtures for tests")
        if type(self.alpha) is not float or self.alpha != ALPHA:
            raise ValueError("alpha is frozen at 0.05")
        if (type(self.n) is not int or self.n not in SAMPLE_SIZES
                or type(self.B) is not int or self.B not in B_VALUES
                or self.null_type not in NULL_TYPES or self.family not in PARAMETERS
                or self.statistic not in STATISTICS[self.family]):
            raise ValueError("cell is outside the frozen null matrix")
        for value in (self.workers, self.batch_size):
            if type(value) is not int or value < 1:
                raise ValueError("workers and batch_size must be positive Python ints")
        params = {k: _canonical_decimal(v) for k, v in self.canonical_parameters.items()}
        if params not in PARAMETERS[self.family]:
            raise ValueError("parameters are outside the frozen null matrix")
        object.__setattr__(self, "canonical_parameters", MappingProxyType(params))

    @property
    def nb_composite(self):
        return self.family == "negative_binomial" and self.null_type == "composite"

    @property
    def primary(self):
        return self.statistic in ("AD", "CVM")

    @property
    def cell_identity(self):
        return {name: (dict(self.canonical_parameters) if name == "canonical_parameters"
                       else getattr(self, name)) for name in
                ("phase", "null_type", "family", "statistic", "n", "canonical_parameters", "B")}

    @property
    def canonical_cell_id(self):
        return canonical_json(self.cell_identity)

    @property
    def directory_name(self):
        return hashlib.sha256(self.canonical_cell_id.encode()).hexdigest()

    def to_dict(self):
        return {f.name: (dict(self.canonical_parameters) if f.name == "canonical_parameters"
                         else getattr(self, f.name)) for f in fields(self)}

    def to_json(self):
        return canonical_json(self.to_dict())

    @property
    def digest(self):
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @property
    def resume_identity(self):
        # Topology is operational metadata, never scientific or seed identity.
        return {k: v for k, v in self.to_dict().items() if k not in ("workers", "batch_size")}

    @classmethod
    def from_dict(cls, data):
        if set(data) != {f.name for f in fields(cls)}:
            raise ValueError("development manifest schema mismatch")
        return cls(**data)


def null_matrix(*, source_sha=BASE_SHA, workers=1, batch_size=1):
    return tuple(DevelopmentManifest(family, statistic, n, params, null, B,
                                    source_sha=source_sha, workers=workers, batch_size=batch_size)
                 for family in PARAMETERS
                 for params, n, statistic, null, B in product(
                     PARAMETERS[family], SAMPLE_SIZES, STATISTICS[family], NULL_TYPES, B_VALUES))


NULL_CONFIGURATION_COUNT = len(null_matrix())
PRIMARY_CONFIGURATION_COUNT = sum(c.primary for c in null_matrix())
COMPARATOR_CONFIGURATION_COUNT = NULL_CONFIGURATION_COUNT - PRIMARY_CONFIGURATION_COUNT
ELIGIBLE_OUTER_TARGET = NULL_CONFIGURATION_COUNT * R_C

BLOCKED_STAGES = {
    "POWER_STAGE_IMPLEMENTED": "NO", "POWER_PREREG_SPEC_COMPLETE": "NO",
    "POWER_EXECUTION_AUTHORIZED": "NO", "METHOD_SELECTION_PERFORMED": "NO",
    "CP05_C_COMPLETE": "NO", "CP05_D_ACCESSED": "NO", "CP05_D_EXECUTED": "NO",
}
