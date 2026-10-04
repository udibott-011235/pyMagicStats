"""PERF-01 identities and measurement order, without scientific overrides."""
from ..r11_decision_equivalence.contract import REFERENCE_WORKLOAD_SHA256
from ..r11_reference_workload.contract import B_R11, SOURCE_OUTERS, TOTAL_RECORDS

BASE_SHA = "2a8041579eaf4a4855c84525766f22770a6ba605"
BASE_TREE = "7b3bd60c29a3aec4617a8efebaef5db64995a480"
PACKAGE_PATH = "experiments/distribution_gof/cuda_calibration/nb_fitting_perf01"
TEST_PATH = "tests/research/test_cp05_c2d_perf01_nb_benchmark.py"
FAST_CPU_ENGINE = "EXPERIMENTAL_FAST_CPU_FLOAT64"
PRODUCTION_BACKEND = "NO"
ORDER = (("CANONICAL_CPU", 1), (FAST_CPU_ENGINE, 1),
         ("CUDA", 1), ("CUDA", 32), ("CUDA", 128), ("CUDA", 512),
         ("CUDA", "full-compatible-batch"))
BUNDLE_NAMES = ("manifest.json", "scientific_eligibility.json", "timings.json",
                "records_summary.json", "environment.json", "summary.json")
