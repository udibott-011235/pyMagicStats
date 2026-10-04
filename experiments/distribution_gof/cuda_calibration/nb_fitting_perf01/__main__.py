"""Explicit workload validation or one fitting campaign; never invoked on import."""
import argparse
from pathlib import Path

from .benchmark import execute, load_records, repository_identity
from .contract import REFERENCE_WORKLOAD_SHA256


def main(argv=None):
    parser = argparse.ArgumentParser(description="CP05-C2D-PERF-01 research-only NB FIT benchmark")
    parser.add_argument("--workload", type=Path, required=True, help="existing accepted R11 payload artifact")
    parser.add_argument("--r4-archive", type=Path, required=True)
    parser.add_argument("--r4-crossings", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new bundle directory outside checkout")
    parser.add_argument("--execute", action="store_true", help="explicitly request one real performance campaign")
    parser.add_argument("--cuda", action="store_true", help="explicitly request actual CUDA execution, without fallback")
    args = parser.parse_args(argv)
    if args.cuda and not args.execute:
        parser.error("--cuda requires --execute")
    if args.execute and args.output is None:
        parser.error("--execute requires --output")
    repository = Path(__file__).resolve().parents[4]
    repository_identity(repository)
    if not args.execute:
        records = load_records(repository, args.workload, args.r4_archive, args.r4_crossings)
        print(f"WORKLOAD_SHA256={REFERENCE_WORKLOAD_SHA256}\nRECORD_COUNT={len(records)}\nREAL_PERFORMANCE_EXECUTION=NO")
        return 0
    result = execute(repository, args.workload, args.r4_archive, args.r4_crossings,
                     args.output, cuda_requested=args.cuda)
    print(f"BENCHMARK_COMPLETE={result['BENCHMARK_COMPLETE']}\nBUNDLE={args.output.resolve()}")
    return 0 if result["BENCHMARK_COMPLETE"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
