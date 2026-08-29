"""Command-line entry point for one calibration shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .runner import RunConfig, run_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic shard of the paired one-sample Student-t versus "
            "uncorrected empirical-likelihood calibration. No network is used."
        )
    )
    parser.add_argument("--replicates-per-cell", type=int, required=True)
    parser.add_argument("--master-seed", type=int, default=20260829)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--shard", dest="shard_id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--format", dest="storage_format", choices=("auto", "parquet", "csv.gz"), default="auto")
    parser.add_argument("--scenario", dest="scenario_ids", action="append")
    parser.add_argument("--sample-size", dest="sample_sizes", action="append", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute completed blocks; never implied by a normal resume",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    outcome = run_shard(
        RunConfig(
            output=arguments.output,
            replicates_per_cell=arguments.replicates_per_cell,
            master_seed=arguments.master_seed,
            backend=arguments.backend,
            workers=arguments.workers,
            batch_size=arguments.batch_size,
            shard_id=arguments.shard_id,
            num_shards=arguments.num_shards,
            alpha=arguments.alpha,
            confidence_level=arguments.confidence_level,
            storage_format=arguments.storage_format,
            force=arguments.force,
            scenario_ids=tuple(arguments.scenario_ids) if arguments.scenario_ids else None,
            sample_sizes=tuple(arguments.sample_sizes) if arguments.sample_sizes else None,
        )
    )
    print(json.dumps(outcome.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
