"""Read-only frozen source binding; array order is normative."""
from __future__ import annotations

from dataclasses import dataclass

from .codec import (canonical_json, integer, require, sha256, strict_json,
                    text_identity)
from .contract import (PROJECTION_BYTES, PROJECTION_FIELDS, PROJECTION_HASH,
                       SOURCE_HASH, SOURCE_OUTERS, PREFIX_COUNT)


def validate_outer(row):
    for field in PROJECTION_FIELDS:
        require(field in row, "missing source projection field")
    cell = text_identity(row["cell_id"])
    index = integer(row["raw_outer_index"])
    identity = f"{cell}|raw_outer={index}"
    require(row["outer_identity"] == identity and row["observed_identity"] == identity,
            "source identity mismatch")
    prefixes = row["bootstrap_identities"]
    require(type(prefixes) is list and len(prefixes) == PREFIX_COUNT, "invalid R4 prefix")
    previous = -1
    for item in prefixes:
        raw = integer(item["raw_inner_index"])
        require(raw > previous, "R4 prefix order mismatch")
        require(item["identity"] == f"{identity}|raw_inner={raw}", "R4 prefix identity mismatch")
        previous = raw


def projection(rows):
    """Sort object keys only; never sort the source array."""
    return [{field: row[field] for field in PROJECTION_FIELDS} for row in rows]


@dataclass(frozen=True)
class SourceSurface:
    kind: str
    manifest_bytes: bytes

    def rows(self):
        manifest = strict_json(self.manifest_bytes)
        require(type(manifest) is dict, "invalid manifest")
        rows = manifest["mc_failed_outers"]
        require(type(rows) is list and bool(rows), "empty source")
        require(len({row["outer_identity"] for row in rows}) == len(rows),
                "duplicate source outer")
        for row in rows:
            validate_outer(row)
        projected = canonical_json(projection(rows))
        if self.kind == "FROZEN_R11":
            require(sha256(self.manifest_bytes) == SOURCE_HASH, "source manifest hash mismatch")
            require(len(rows) == SOURCE_OUTERS, "source outer count mismatch")
            require(len(projected) == PROJECTION_BYTES and sha256(projected) == PROJECTION_HASH,
                    "source projection hash mismatch")
        else:
            require(self.kind == "SYNTHETIC_TEST", "unknown source kind")
        return rows

    @classmethod
    def frozen(cls, manifest_bytes):
        result = cls("FROZEN_R11", manifest_bytes)
        result.rows()
        return result

    @classmethod
    def synthetic_for_testing(cls, rows):
        """Explicitly non-scientific; downstream scientific loader rejects it."""
        result = cls("SYNTHETIC_TEST", canonical_json({"mc_failed_outers": rows}))
        result.rows()
        return result
