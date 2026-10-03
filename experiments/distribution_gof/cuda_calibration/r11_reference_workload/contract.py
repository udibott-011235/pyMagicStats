"""Frozen documentary constants from accepted DEC-027 / DEC-028."""
ALPHA = 0.05
B_R11 = 199
NB_RETRY_CAP = 19900
SOURCE_OUTERS = 12
RECORDS_PER_OUTER = 200
TOTAL_RECORDS = 2400
NAMESPACE = "CP05-C2C"
SCHEMA_VERSION = "cp05-c2c-r11-reference-workload-v1"
SCIENTIFIC_SHA = "d7412911dcf8402f01a502fe0d756813ec10bbee"
SCIENTIFIC_TREE = "421c07a3466a689e8c49da5c1cfe25fb946a0a5b"
SOURCE_PATH = "experiments/distribution_gof/cuda_calibration/targeted_replay_r4/frozen_identity_manifest.json"
SOURCE_HASH = "3f300ff5a678e3eac4956fe10514bf7634ec0ea824f0bfef56f4688a86adb286"
PROJECTION_BYTES = 4250
PROJECTION_HASH = "91d090100179f96e5d562a6d2019becd249af3b6b20405ac5cc041622fd7c8bd"
PROJECTION_FIELDS = (
    "outer_identity", "cell_id", "raw_outer_index", "r9_b_cpu", "r9_b_cuda",
    "r9_p_cpu", "r9_p_cuda", "r9_reject_cpu", "r9_reject_cuda", "observed_identity",
)
ARCHIVE_NAME = "cp05_c2c_r10a_targeted_r4_501d752_run2_PASS_EVIDENCE.tar.gz"
CROSSINGS_NAME = "cp05_c2c_r10a_targeted_r4_501d752_run2_crossings.json"
WORKLOAD_NAME = "workload_b_records.jsonl"
ARTIFACT_HASHES = {
    ARCHIVE_NAME: "dd8de17dfa17ac54855f9823d053e6820a2a285aeb8467eec390aea51dcba6ad",
    CROSSINGS_NAME: "f7c35e51e0273eac73f9743010b0191cc3c33fac9dde88cdbc123200d86e10f7",
    WORKLOAD_NAME: "beea7dffa4a059de241703079e254a696a8ad7764cd9b6288079a84188ee5f37",
}
PREFIX_COUNT = 15
