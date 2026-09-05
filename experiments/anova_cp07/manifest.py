"""Materialize and validate the frozen CP06 v1.1 identity layer."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'knowledge/experiments/anova-cp06-calibration-manifest.json'
VERSION = 'anova-calibration-prereg-v1.1'
SOURCE_CANONICAL_SHA256 = 'ad9cec5fd15a45531e65c6a41b2ce8f0c073f11b198dd2a9b8a3bb8506f72bf7'
COUNTS = dict(zip(('E0', 'D-core-h0', 'D-robustness-h0', 'D-stress-h0',
                   'D-power-h1', 'H-core-normal', 'H-robustness', 'H-power'),
                  (12, 42, 54, 10, 36, 10, 21, 12)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _expand(source):
    result = deepcopy(source)
    phases = result['phase_cells']
    def add(phase, cell_id, family, sizes, sd, delta=None):
        cell = dict(cell_id=cell_id, family=family, sizes=sizes, sd=sd,
                    hypothesis='H0' if delta is None else 'H1')
        if delta is not None:
            cell['delta_range'] = delta
        phases[phase].append(cell)

    designs = [([5]*3, [1]*3), ([30]*3, [1]*3), ([5,10,20], [1]*3),
               ([5,10,20], [4,2,1]), ([5,10,20], [1,2,4]), ([10]*5, [1]*5)]
    for f, family in enumerate(source['development_families'][1:], 1):
        for r, (sizes, sd) in enumerate(designs, 1):
            add('D-robustness-h0', f'DRH0-F{f:02d}-R{r:02d}', family, sizes, sd)
    power = [('normal', [10]*3, [1]*3), ('normal', [30]*3, [1]*3),
             ('normal', [10]*5, [1]*5), ('normal', [5,10,20], [1]*3),
             ('normal', [5,10,20], [4,2,1]), ('normal', [5,10,20], [1,2,4]),
             ('gamma_shape_1', [10]*3, [1]*3), ('lognormal_sigma_1p2', [10]*3, [1]*3),
             ('student_t_df_3', [10]*3, [1]*3),
             ('contamination_asymmetric_5pct_loc10', [10]*3, [1]*3),
             ('laplace', [10]*5, [1]*5), ('mixture_symmetric_5pct_scale6', [30]*3, [1]*3)]
    for p, (family, sizes, sd) in enumerate(power, 1):
        for d, delta in enumerate(source['power_delta_range'], 1):
            add('D-power-h1', f'DPH1-P{p:02d}-D{d:02d}', family, sizes, sd, delta)
    holdout_designs = [([7]*3, [1]*3), ([6,15,40], [1]*3), ([6,15,40], [3.5,2,1])]
    for f, family in enumerate(source['holdout_only_families'], 1):
        for r, (sizes, sd) in enumerate(holdout_designs, 1):
            add('H-robustness', f'HRH0-F{f:02d}-HRD{r:02d}', family, sizes, sd)
    for f, family in enumerate(source['holdout_only_families'][:4], 1):
        for d, delta in enumerate(source['power_delta_range'], 1):
            add('H-power', f'HPH1-F{f:02d}-D{d:02d}', family, [10]*3, [1]*3, delta)
    return result


def _source(path=SOURCE):
    source = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_keys)
    if digest(source) != SOURCE_CANONICAL_SHA256:
        raise ValueError('compact manifest differs from frozen v1.1 (including IDs/configuration)')
    return source


def validate_manifest(manifest):
    """Reject missing/extra/duplicate identities and any frozen-field drift.

    The compact source is pinned by a canonical digest, independent of checkout
    CRLF. Full-record equality additionally checks every cross-product mapping.
    """
    expected = _expand(_source())
    phases = manifest.get('phase_cells', {})
    if set(phases) != set(COUNTS):
        raise ValueError('missing or unexpected phase')
    seen = set()
    for phase, count in COUNTS.items():
        cells = phases[phase]
        if len(cells) != count:
            raise ValueError(f'{phase}: missing or unexpected cell')
        expected_cells = {c['cell_id']: c for c in expected['phase_cells'][phase]}
        for cell in cells:
            cell_id = cell.get('cell_id')
            if cell_id in seen:
                raise ValueError(f'duplicate cell_id: {cell_id}')
            seen.add(cell_id)
            if cell_id not in expected_cells:
                raise ValueError(f'unexpected cell_id: {cell_id}')
            if cell != expected_cells[cell_id]:
                raise ValueError(f'inconsistent frozen cell: {cell_id}')
    if len(seen) != 197:
        raise ValueError('expected 197 unique cell_id')
    # Ordering is also frozen; no altered seeds, counts, bands or extra fields.
    if canonical(manifest) != canonical(expected):
        raise ValueError('manifest configuration/order differs from frozen v1.1')
    return manifest


def load_manifest(path=SOURCE):
    return validate_manifest(_expand(_source(path)))
