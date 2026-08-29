"""Command-line entry point for simulation-free shard aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .aggregate import aggregate_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate completed EL-versus-t shards; never runs simulation."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    metadata = aggregate_calibration(arguments.input, arguments.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
