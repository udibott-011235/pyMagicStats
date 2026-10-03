"""Frozen R11 architecture. No CLI overrides or development execution waiver."""
from ..r11_reference_workload.contract import (
    ALPHA, B_R11, SCIENTIFIC_SHA, SCIENTIFIC_TREE, SOURCE_PATH, SOURCE_HASH,
    PROJECTION_HASH, ARCHIVE_NAME, CROSSINGS_NAME, WORKLOAD_NAME, ARTIFACT_HASHES,
)
SOURCE_OUTERS = 12
RECORDS_PER_OUTER = B_R11 + 1
TOTAL_RECORDS = SOURCE_OUTERS * RECORDS_PER_OUTER
BUILDER_SHA = "1ace65bf9e01ab05df84a1cbca5031fac93bfa88"
BUILDER_TREE = "0ba6d9804a7c46ab636cc47012091f7b9e80dcc6"
READINESS_IMPLEMENTATION_BASE_SHA = "7891f17a3af77ebdc818896b31cfd12c4179b759"
READINESS_IMPLEMENTATION_BASE_TREE = "31f93907b9f57956e09fb181749b340b108f5e6f"
REFERENCE_WORKLOAD_SHA256 = "77f53d616deeafcba51aa185dae8a8d2ca60ee4172f4a6353a3aa97b75324b16"
SCHEMA_VERSION = "cp05-c2c-r11-decision-equivalence-v1"
PACKAGE_PATH = "experiments/distribution_gof/cuda_calibration/r11_decision_equivalence"
BUILDER_PATH = "experiments/distribution_gof/cuda_calibration/r11_reference_workload"
PROJECT_ROOTS = ("experiments", "pyMagicStat", "pyMagicStats", "pymagicstats")
GATES = ("classification_gate_pass", "fit_gate_pass",
         "distribution_value_gate_pass", "statistic_gate_pass")
IDENTITY_FIELDS = ("identity", "record_type", "cell_id", "raw_outer_index",
                   "raw_inner_index", "seed_identity", "accepted_ordinal")
PAYLOAD_FIELDS = ("sample_digest", "payload_dtype", "payload_shape")
RECORD_FIELDS = frozenset(IDENTITY_FIELDS + PAYLOAD_FIELDS + GATES + (
    "cpu_classification", "cuda_classification", "cpu_parameters", "cuda_parameters",
    "cpu_log_likelihood", "cuda_log_likelihood", "cpu_statistic", "cuda_statistic",
    "distribution_evidence", "flat_objective_used", "flat_objective_diagnostic",
    "cuda_failure_reason", "cuda_solver_converged", "cpu_input_identity_pass",
    "cuda_input_identity_pass", "cpu_sample_digest", "cuda_sample_digest",
    "evaluation_points", "cpu_evaluation_points", "cuda_evaluation_points",
    "family", "raw_cpu_result", "raw_cuda_result",
))
QUANTITIES = {
    "negative_binomial": ("pmf", "logPMF", "cdf", "sf", "logCDF", "logSF"),
    "gamma": ("cdf", "sf", "logCDF", "logSF"),
    "exponential": ("cdf", "sf", "logCDF", "logSF"),
}
EVIDENCE_NAMES = (
    "execution_manifest.json", "reference_workload.json", "records.jsonl",
    "indicator_adjudication.jsonl", "outer_results.json", "boundary_fixtures.json",
    "environment.json", "summary.json",
)
