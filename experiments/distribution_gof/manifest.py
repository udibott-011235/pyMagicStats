"""Immutable, fail-closed manifests for the CP05 research harness."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Mapping


MANIFEST_SCHEMA_VERSION = "cp05-b-manifest-v1"
OUTPUT_SCHEMA_VERSION = "cp05-b-output-v1"
SOURCE_REPOSITORY = "udibott-011235/pyMagicStats"
DEVELOPMENT_NAMESPACE = "pyMagicStats/STAGE-DIST-FAMILIES-001/CP05-C/v1"
ALPHA = 0.05
SUPPORTED_B = frozenset({199, 999})
NON_CALIBRATION_PHASE = "CP05-B_NON_CALIBRATION"
_FAMILIES = frozenset({"gamma", "exponential", "negative_binomial"})
_NULL_TYPES = frozenset({"simple", "composite"})
_CONTINUOUS_STATISTICS = frozenset({"AD", "CVM", "KS"})
_DISCRETE_STATISTICS = frozenset({"AD", "CVM", "KS", "PEARSON"})
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _canonical_decimal(value: Any) -> str:
    if isinstance(value, bool):
        raise TypeError("canonical parameters must be decimal scalars, not bool")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("canonical parameters must be finite decimals") from exc
    if not decimal.is_finite():
        raise ValueError("canonical parameters must be finite decimals")
    if decimal == 0:
        return "0"
    rendered = format(decimal.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def canonical_json(value: Any) -> str:
    """Serialize identity material without binary-float representation drift."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    schema_version: str
    phase: str
    source_repository: str
    source_sha: str
    namespace_id: str
    null_type: str
    family: str
    statistic: str
    n: int
    canonical_parameters: Mapping[str, str]
    B: int
    alpha: float
    raw_outer_range: tuple[int, int]
    shard_spec: Mapping[str, int]
    batch_spec: Mapping[str, int]
    worker_spec: Mapping[str, int]
    environment_requirements: Mapping[str, str]
    output_schema_version: str
    claim_status: str = "NON_CLAIMING"

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema_version")
        if self.phase != NON_CALIBRATION_PHASE:
            raise ValueError("CP05-B runner accepts only NON_CALIBRATION phase")
        if self.source_repository != SOURCE_REPOSITORY:
            raise ValueError("source_repository is not the authorized repository")
        if not isinstance(self.source_sha, str) or not _SHA_RE.fullmatch(self.source_sha):
            raise ValueError("source_sha must be a lowercase 40-hex commit id")
        if self.namespace_id != DEVELOPMENT_NAMESPACE:
            raise ValueError("only the public development namespace is allowed")
        if self.null_type not in _NULL_TYPES:
            raise ValueError("unknown null_type")
        if self.family not in _FAMILIES:
            raise ValueError("unknown family")
        allowed_statistics = (
            _DISCRETE_STATISTICS if self.family == "negative_binomial"
            else _CONTINUOUS_STATISTICS
        )
        if self.statistic not in allowed_statistics:
            raise ValueError("statistic is not authorized for this family")
        if type(self.n) is not int or self.n <= 0:
            raise ValueError("n must be a positive Python int")
        if type(self.B) is not int or self.B not in SUPPORTED_B:
            raise ValueError("B must be exactly 199 or 999")
        if type(self.alpha) is not float or self.alpha != ALPHA:
            raise ValueError("alpha must be exactly 0.05")
        if self.claim_status != "NON_CLAIMING":
            raise ValueError("CP05-B artifacts must be NON_CLAIMING")

        expected_names = {
            "gamma": {"shape", "scale"},
            "exponential": {"scale"},
            "negative_binomial": {"r", "p"},
        }[self.family]
        if set(self.canonical_parameters) != expected_names:
            raise ValueError("canonical parameter names do not match the family")
        parameters = {
            str(name): _canonical_decimal(value)
            for name, value in self.canonical_parameters.items()
        }
        numeric = {name: Decimal(value) for name, value in parameters.items()}
        if any(value <= 0 for value in numeric.values()):
            raise ValueError("canonical parameters must be positive")
        if self.family == "negative_binomial" and numeric["p"] > 1:
            raise ValueError("negative_binomial p must not exceed one")

        if (
            type(self.raw_outer_range) is not tuple
            or len(self.raw_outer_range) != 2
            or any(type(item) is not int for item in self.raw_outer_range)
            or self.raw_outer_range[0] < 0
            or self.raw_outer_range[1] <= self.raw_outer_range[0]
        ):
            raise ValueError("raw_outer_range must be a nonempty half-open range")
        shard = _validate_int_mapping(self.shard_spec, {"shard_id", "shard_count"})
        if shard["shard_count"] < 1 or not 0 <= shard["shard_id"] < shard["shard_count"]:
            raise ValueError("invalid shard_spec")
        batch = _validate_int_mapping(self.batch_spec, {"batch_size"})
        workers = _validate_int_mapping(self.worker_spec, {"workers"})
        if batch["batch_size"] < 1 or workers["workers"] < 1:
            raise ValueError("batch_size and workers must be positive")
        environment = _validate_str_mapping(self.environment_requirements)
        if self.output_schema_version != OUTPUT_SCHEMA_VERSION:
            raise ValueError("unsupported output_schema_version")

        object.__setattr__(self, "canonical_parameters", MappingProxyType(parameters))
        object.__setattr__(self, "shard_spec", MappingProxyType(shard))
        object.__setattr__(self, "batch_spec", MappingProxyType(batch))
        object.__setattr__(self, "worker_spec", MappingProxyType(workers))
        object.__setattr__(self, "environment_requirements", MappingProxyType(environment))

    @property
    def cell_identity(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "B": self.B,
                "canonical_parameters": dict(self.canonical_parameters),
                "family": self.family,
                "n": self.n,
                "null_type": self.null_type,
                "phase": self.phase,
                "statistic": self.statistic,
            }
        )

    @property
    def canonical_cell_id(self) -> str:
        return canonical_json(dict(self.cell_identity))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "source_repository": self.source_repository,
            "source_sha": self.source_sha,
            "namespace_id": self.namespace_id,
            "null_type": self.null_type,
            "family": self.family,
            "statistic": self.statistic,
            "n": self.n,
            "canonical_parameters": dict(self.canonical_parameters),
            "B": self.B,
            "alpha": self.alpha,
            "raw_outer_range": list(self.raw_outer_range),
            "shard_spec": dict(self.shard_spec),
            "batch_spec": dict(self.batch_spec),
            "worker_spec": dict(self.worker_spec),
            "environment_requirements": dict(self.environment_requirements),
            "output_schema_version": self.output_schema_version,
            "claim_status": self.claim_status,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentManifest":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(payload) != expected:
            raise ValueError("manifest fields do not match the frozen schema")
        converted = dict(payload)
        raw_range = converted.get("raw_outer_range")
        if isinstance(raw_range, list):
            converted["raw_outer_range"] = tuple(raw_range)
        return cls(**converted)


def _validate_int_mapping(value: Mapping[str, Any], names: set[str]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != names:
        raise ValueError(f"mapping must contain exactly {sorted(names)}")
    result = dict(value)
    if any(type(item) is not int for item in result.values()):
        raise TypeError("execution topology values must be Python ints")
    return result


def _validate_str_mapping(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("environment_requirements must be a nonempty mapping")
    result = dict(value)
    if any(type(key) is not str or type(item) is not str for key, item in result.items()):
        raise TypeError("environment requirements must map strings to strings")
    return result
