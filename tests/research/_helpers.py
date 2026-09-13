from __future__ import annotations

from experiments.distribution_gof.manifest import (
    ALPHA,
    DEVELOPMENT_NAMESPACE,
    MANIFEST_SCHEMA_VERSION,
    NON_CALIBRATION_PHASE,
    OUTPUT_SCHEMA_VERSION,
    SOURCE_REPOSITORY,
    ExperimentManifest,
)


BASE_SHA = "5eb179be578594aa900a29bf5ae2f5540e05ffa2"


def manifest(**overrides):
    values = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase": NON_CALIBRATION_PHASE,
        "source_repository": SOURCE_REPOSITORY,
        "source_sha": BASE_SHA,
        "namespace_id": DEVELOPMENT_NAMESPACE,
        "null_type": "simple",
        "family": "exponential",
        "statistic": "CVM",
        "n": 8,
        "canonical_parameters": {"scale": "1.0"},
        "B": 199,
        "alpha": ALPHA,
        "raw_outer_range": (0, 2),
        "shard_spec": {"shard_id": 0, "shard_count": 1},
        "batch_spec": {"batch_size": 1},
        "worker_spec": {"workers": 1},
        "environment_requirements": {
            "python": ">=3.10",
            "numpy": ">=1.24",
            "scipy": ">=1.11",
        },
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "claim_status": "NON_CLAIMING",
    }
    values.update(overrides)
    return ExperimentManifest(**values)
