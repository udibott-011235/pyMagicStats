"""Deterministic paired production-kernel path and public-API parity gate."""
import hashlib
import math
import os
import warnings
from dataclasses import dataclass

THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
# Set before importing numeric libraries, also in spawned workers.
for _name in THREAD_VARS:
    os.environ[_name] = '1'

import numpy as np
from threadpoolctl import threadpool_limits
from pyMagicStat.inference import anova as engine

FIELDS = ('statistic', 'p_value', 'numerator_df', 'denominator_df')
METHODS = ('classical', 'welch')


@dataclass(frozen=True)
class HoldoutPermit:
    """Explicit PO opening declaration, validated against the frozen candidate."""
    harness_sha: str
    manifest_sha256: str
    authorization_sha256: str


def check_phase(phase, permit=None):
    if phase.startswith('H-'):
        if phase not in ('H-core-normal', 'H-robustness', 'H-power'):
            raise ValueError(f'unknown phase: {phase}')
        if not isinstance(permit, HoldoutPermit):
            raise PermissionError('Phase H is sealed; Product Owner authorization required')
        return
    if phase not in ('E0', 'D-core-h0', 'D-robustness-h0', 'D-stress-h0', 'D-power-h1'):
        raise ValueError(f'unknown phase: {phase}')


def replica_rng(phase, cell_id, replicate_index, master_seed):
    if type(replicate_index) is not int or replicate_index < 0:
        raise ValueError('replicate_index must be a nonnegative integer')
    raw = hashlib.sha256(f'{phase}|{cell_id}|{replicate_index}'.encode('utf-8')).digest()
    # Explicit little endian avoids host-native uint32 interpretation.
    words = np.frombuffer(raw[:16], dtype='<u4').tolist()
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([master_seed, *words])))


def standardized(rng, family, n):
    if family == 'normal':
        return rng.normal(size=n)
    if family.startswith('gamma_shape_'):
        shape = float(family.removeprefix('gamma_shape_'))
        return (rng.gamma(shape, size=n) - shape) / math.sqrt(shape)
    if family.startswith('lognormal_sigma_'):
        sigma = float(family.removeprefix('lognormal_sigma_').replace('p', '.'))
        mean = math.exp(sigma*sigma/2)
        sd = math.sqrt(math.expm1(sigma*sigma) * math.exp(sigma*sigma))
        return (rng.lognormal(0, sigma, n) - mean) / sd
    if family.startswith('student_t_df_'):
        df = float(family.removeprefix('student_t_df_').replace('p', '.'))
        return rng.standard_t(df, n) * math.sqrt((df-2)/df)
    if family == 'laplace':
        return rng.laplace(0, 1/math.sqrt(2), n)
    if family == 'mixture_symmetric_5pct_scale6':
        mask = rng.random(n) < .05
        return rng.normal(size=n) * np.where(mask, 6., 1.) / math.sqrt(.95+.05*36)
    if family.startswith('contamination_asymmetric_'):
        fraction = float(family.split('_')[2].removesuffix('pct'))/100
        mask = rng.random(n) < fraction
        return (rng.normal(size=n)+10*mask-10*fraction)/math.sqrt(1+fraction*(1-fraction)*100)
    if family == 'weibull_shape_1p5':
        mean = math.gamma(1+1/1.5)
        return (rng.weibull(1.5, n)-mean)/math.sqrt(math.gamma(1+2/1.5)-mean*mean)
    if family == 'pareto_alpha_3p5':
        a = 3.5
        return (rng.pareto(a, n)+1-a/(a-1))/math.sqrt(a/((a-1)**2*(a-2)))
    if family == 'beta_2_5':
        return (rng.beta(2, 5, n)-2/7)/math.sqrt(10/(49*8))
    raise ValueError(f'unsupported family: {family}')


def generate_groups(phase, cell, index, master_seed, permit=None):
    check_phase(phase, permit)
    rng = replica_rng(phase, cell['cell_id'], index, master_seed)
    means = np.linspace(-.5, .5, len(cell['sizes']))
    means = (means-means.mean())*cell.get('delta_range', 0.)
    return tuple(np.asarray(standardized(rng, cell['family'], n)*sd+mu, dtype='<f8')
                 for n, sd, mu in zip(cell['sizes'], cell['sd'], means))


def sample_digest(groups):
    h = hashlib.sha256()
    for group in groups:
        h.update(int(group.size).to_bytes(8, 'little'))
        h.update(np.asarray(group, dtype='<f8').tobytes())
    return h.hexdigest()


def paired_replica(phase, cell, index, master_seed, permit=None):
    check_phase(phase, permit)
    row = dict(phase=phase, cell_id=cell['cell_id'], replicate_index=index,
               sample_sha256='', generation_error='', warning_count=0)
    for method in METHODS:
        row[method+'_status'] = 'generation_error'
        row[method+'_error'] = ''
        for field in FIELDS:
            row[method+'_'+field] = None
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:
            groups = (generate_groups(phase, cell, index, master_seed) if permit is None else
                      generate_groups(phase, cell, index, master_seed, permit))
            if not all(np.all(np.isfinite(g)) for g in groups):
                raise ValueError('generated nonfinite observations')
            row['sample_sha256'] = sample_digest(groups)
        except Exception as exc:
            row['generation_error'] = f'{type(exc).__name__}: {exc}'
        else:
            try:
                summaries = engine._summarize_groups(groups)
            except Exception as exc:
                for method in METHODS:
                    row[method+'_status'] = 'kernel_error'
                    row[method+'_error'] = f'summaries: {type(exc).__name__}: {exc}'
            else:
                for method, kernel in zip(METHODS, (engine._classical_kernel, engine._welch_kernel)):
                    try:
                        result = kernel(summaries)
                        values = [float(getattr(result, f)) for f in FIELDS]
                        if not all(math.isfinite(v) for v in values):
                            row[method+'_status'] = 'nonfinite'
                        else:
                            row[method+'_status'] = 'ok'
                        for field, value in zip(FIELDS, values):
                            row[method+'_'+field] = value
                    except Exception as exc:
                        row[method+'_status'] = 'kernel_error'
                        row[method+'_error'] = f'{type(exc).__name__}: {exc}'
        row['warning_count'] = len(captured)
    return row


def compute_batch(task):
    phase, cell, indices, seed, *extra = task
    permit = extra[0] if extra else None
    with threadpool_limits(limits=1):
        return [paired_replica(phase, cell, i, seed, permit) for i in indices]


def parity_gate(manifest, phase, permit=None):
    check_phase(phase, permit)
    seed = manifest['holdout_master_seed' if phase.startswith('H-') else 'development_master_seed']
    count = warning_count = 0
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for cell in manifest['phase_cells'][phase]:
            for index in range(32):
                groups = generate_groups(phase, cell, index, seed, permit)
                summaries = engine._summarize_groups(groups)
                for kernel, api in ((engine._classical_kernel, engine.OneWayANOVA),
                                    (engine._welch_kernel, engine.WelchANOVA)):
                    fast = kernel(summaries)
                    public = api(*groups, independence='assumed').run()
                    actual = np.array([getattr(fast, f) for f in FIELDS])
                    expected = np.array([getattr(public, f) for f in FIELDS])
                    if not (np.isfinite(actual).all() and np.isfinite(expected).all()
                            and np.allclose(actual, expected, rtol=1e-12, atol=1e-14)):
                        raise RuntimeError(f'parity mismatch: {phase}/{cell["cell_id"]}/{index}/{api.__name__}')
                    count += 1
        warning_count = len(captured)
    return dict(status='PASS', cells=len(manifest['phase_cells'][phase]),
                replicas_per_cell=32, comparisons=count, warning_count=warning_count,
                rtol=1e-12, atol=1e-14)
