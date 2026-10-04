"""Development null CLI. Merely constructing the matrix never generates samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

from ..artifacts import write_json_atomic
from .manifest import null_matrix, BLOCKED_STAGES
from .aggregate import aggregate
from .runner import run_cells
from .source_identity import current_source_sha


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="CP05-C development null calibration (research only)")
    parser.add_argument("action", choices=("manifest", "run", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha", help="defaults to the current committed checkout SHA")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cell-id", action="append", help="exact canonical cell JSON; default is full null matrix")
    parser.add_argument("--fixture-target", type=int, help="explicit software fixture; excluded from scientific aggregation")
    parser.add_argument("--max-new-units", type=int, help="checkpoint boundary per cell")
    args = parser.parse_args(arguments)
    args.source_sha = args.source_sha or current_source_sha()
    matrix = null_matrix(source_sha=args.source_sha, workers=args.workers, batch_size=args.batch_size)
    if args.action == "manifest":
        write_json_atomic(args.root / "null_matrix.json", {
            "cells": [m.to_dict() for m in matrix], "NULL_CONFIGURATION_COUNT": len(matrix),
            "PRIMARY_CONFIGURATION_COUNT": sum(m.primary for m in matrix),
            "COMPARATOR_CONFIGURATION_COUNT": sum(not m.primary for m in matrix), **BLOCKED_STAGES})
    elif args.action == "run":
        selected = matrix
        if args.cell_id:
            identities = set(args.cell_id)
            selected = tuple(m for m in matrix if m.canonical_cell_id in identities)
            if len(selected) != len(args.cell_id):
                parser.error("cell-id list has missing, duplicate or unauthorized cells")
        results = run_cells(selected, args.root, workers=args.workers, batch_size=args.batch_size,
                            fixture_target=args.fixture_target, max_new_units=args.max_new_units,
                            command=shlex.join([sys.executable, "-m", "experiments.distribution_gof.cp05_c", *arguments]))
        print(json.dumps(results, sort_keys=True))
    else:
        directories = [args.root / m.directory_name for m in matrix]
        write_json_atomic(args.root / "null_matrix_summary.json", aggregate(directories, source_sha=args.source_sha))


if __name__ == "__main__":
    main()
