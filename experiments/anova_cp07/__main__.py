"""python -m experiments.anova_cp07: explicit validation, parity, run, combine."""
import argparse
import json
from .manifest import COUNTS, load_manifest
from .simulation import parity_gate
from .runner import run_phase, recompose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('validate')
    parity = commands.add_parser('parity')
    parity.add_argument('--phase', choices=list(COUNTS), required=True)
    run = commands.add_parser('run')
    run.add_argument('--phase', choices=list(COUNTS), required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--workers', type=int, default=1)
    run.add_argument('--batch-size', type=int, default=200)
    run.add_argument('--shard-id', type=int, default=0)
    run.add_argument('--shard-count', type=int, default=1)
    run.add_argument('--resume', action='store_true')
    run.add_argument('--holdout-authorization', help='Future explicit Product Owner opening declaration')
    combine = commands.add_parser('recompose')
    combine.add_argument('--output', required=True)
    combine.add_argument('inputs', nargs='+')
    combine.add_argument('--holdout-authorization')
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'validate':
        manifest = load_manifest()
        result = {p: len(c) for p,c in manifest['phase_cells'].items()}
    elif command == 'parity':
        result = parity_gate(load_manifest(), args['phase'])
    elif command == 'run':
        result = run_phase(**args)
    else:
        result = recompose(**args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
