"""CPU runner with immutable transactional shard directories and recomposition."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import heapq
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from .manifest import ROOT, SOURCE, canonical, digest, load_manifest, validate_manifest
from .simulation import FIELDS, METHODS, THREAD_VARS, HoldoutPermit, check_phase, compute_batch, parity_gate, engine
from .accounting import Accounting
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

ENGINE_BLOB = '2d00ae2a2812b8c390125fefe244dcb4830176c5'
ENGINE_COMMIT = '376677ca32dfd1e3f5b5b64bec48e3160c35d5a9'
PREFIX = 'anova_calibration_'


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def provenance(manifest):
    if Path(engine.__file__).resolve() != (ROOT/'pyMagicStat/inference/anova.py').resolve():
        raise RuntimeError('imported engine is not from this candidate checkout')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('evidence requires a clean committed harness checkout')
    if git('branch', '--show-current') in ('main', 'feature/anova-engine', ''):
        raise RuntimeError('evidence requires the technical harness branch')
    if git('hash-object', 'pyMagicStat/inference/anova.py') != ENGINE_BLOB:
        raise RuntimeError('production ANOVA differs from frozen engine blob')
    return dict(harness_sha=git('rev-parse', 'HEAD'), engine_sha=ENGINE_COMMIT,
                engine_blob=ENGINE_BLOB, preregistration_version=manifest['preregistration_version'],
                manifest_sha256=digest(manifest), source_manifest_sha256=file_hash(SOURCE),
                rng='PCG64; SHA256 first 16 bytes as four little-endian uint32; master first',
                versions={p: importlib.metadata.version(p) for p in
                          ('numpy', 'scipy', 'statsmodels', 'pandas', 'pyarrow', 'threadpoolctl')},
                python=platform.python_version(), os=platform.platform(), cpu=platform.processor())


def holdout_permit(phase, authorization, identity):
    if not phase.startswith('H-'):
        if authorization is not None:
            raise ValueError('holdout authorization is not applicable to development/E0')
        return None
    if authorization is None:
        raise PermissionError('Phase H is sealed; Product Owner authorization required')
    declaration = json.loads(Path(authorization).read_text(encoding='utf-8'))
    required = dict(action='open-holdout', authorized_by='Product Owner',
                    harness_sha=identity['harness_sha'], manifest_sha256=identity['manifest_sha256'],
                    phase_d_complete=True, remediations_closed=True, candidate_frozen=True)
    if any(type(declaration.get(k)) is not type(v) or declaration[k] != v for k,v in required.items()):
        raise PermissionError('holdout opening declaration does not authorize this frozen candidate')
    return HoldoutPermit(identity['harness_sha'], identity['manifest_sha256'], file_hash(authorization))


def _schema(identity):
    fields = [('phase', pa.string()), ('cell_id', pa.string()), ('replicate_index', pa.int64()),
              ('sample_sha256', pa.string()), ('generation_error', pa.string()), ('warning_count', pa.int64())]
    for method in METHODS:
        fields += [(method+'_status', pa.string()), (method+'_error', pa.string())]
        fields += [(method+'_'+field, pa.float64()) for field in FIELDS]
    return pa.schema(fields, metadata={b'provenance': canonical(identity)})


def _rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def _expected(manifest, phase, shard_id, shard_count, reps):
    for cell in sorted(manifest['phase_cells'][phase], key=lambda c: c['cell_id']):
        for index in range(shard_id, reps, shard_count):
            yield cell['cell_id'], index


def _validate_rows(rows, expected, accounting):
    expected = iter(expected)
    for row in rows:
        key = (row['cell_id'], row['replicate_index'])
        if row['phase'] != accounting.phase or key != next(expected, None):
            raise ValueError(f'missing, duplicate, unexpected or out-of-order replica: {key}')
        accounting.consume(row)
        yield row
    if next(expected, None) is not None:
        raise ValueError('missing replicas')


def _json(path, value):
    Path(path).write_bytes(canonical(value)+b'\n')


def verify_directory(directory, expected_identity=None, permit=None):
    directory = Path(directory)
    meta = json.loads((directory/(PREFIX+'metadata.json')).read_text())
    check_phase(meta['phase'], permit)
    if expected_identity is not None and meta['identity'] != expected_identity:
        raise ValueError('cannot mix harness/engine/manifest/environment identities')
    sid, count, reps = meta['shard_id'], meta['shard_count'], meta['replications_per_cell']
    if any(type(v) is not int for v in (sid, count, reps)) or not 0 <= sid < count <= reps:
        raise ValueError('invalid stored shard configuration')
    replica_name = PREFIX+f'replicates-{sid:05d}-of-{count:05d}.parquet'
    required = {PREFIX+name for name in ('manifest.json', 'summary.parquet', 'summary.csv',
                                        'disagreement.csv', 'report.md')} | {replica_name}
    files = meta['checksums']
    if set(files) != required or meta['replicate_file'] != replica_name:
        raise ValueError('incorrect artifact names')
    if set(p.name for p in directory.iterdir()) != set(files) | {PREFIX+'metadata.json'}:
        raise ValueError('unexpected or missing artifact')
    for name, checksum in files.items():
        if Path(name).name != name or file_hash(directory/name) != checksum:
            raise ValueError(f'checksum mismatch: {name}')
    manifest = json.loads((directory/(PREFIX+'manifest.json')).read_text())
    validate_manifest(manifest)
    if digest(manifest) != meta['identity']['manifest_sha256']:
        raise ValueError('manifest identity mismatch')
    parquet = directory/meta['replicate_file']
    embedded = json.loads(pq.read_schema(parquet).metadata[b'provenance'])
    if embedded != meta['identity']:
        raise ValueError('parquet provenance mismatch')
    if meta['phase'] not in manifest['phase_cells']:
        raise ValueError('unknown artifact phase')
    if meta['replications_per_cell'] != manifest['replications'][meta['phase']]:
        raise ValueError('non-frozen replication count')
    seed_key = 'holdout_master_seed' if meta['phase'].startswith('H-') else 'development_master_seed'
    if meta['master_seed'] != manifest[seed_key] or meta['alpha_grid'] != manifest['alpha_grid']:
        raise ValueError('stored seed/alpha mismatch')
    if meta['parity'].get('status') != 'PASS' or meta['parity'].get('replicas_per_cell') != 32:
        raise ValueError('missing parity gate evidence')
    accounting = Accounting(manifest, meta['phase'])
    for _ in _validate_rows(_rows(parquet), _expected(manifest, meta['phase'],
                            meta['shard_id'], meta['shard_count'], meta['replications_per_cell']), accounting):
        pass
    summary, pairs = accounting.finish(meta['shard_count'] == 1)
    if summary != meta['counts'] or pairs != meta['paired_counts']:
        raise ValueError('metadata accounting mismatch')
    invalid = any(r['confirmatory_gate'] == 'INVALID_EXECUTION' for r in summary)
    if meta['execution_status'] != ('INVALID_EXECUTION' if invalid else 'ACCOUNTED'):
        raise ValueError('metadata execution status mismatch')
    return meta


def _publish(output, manifest, identity, phase, shard_id, shard_count, reps, batches, settings, gate, permit=None):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name+'.pending-', dir=output.parent))
    started = utc()
    replica_name = PREFIX+f'replicates-{shard_id:05d}-of-{shard_count:05d}.parquet'
    accounting = Accounting(manifest, phase)
    schema = _schema(identity)
    expected = iter(_expected(manifest, phase, shard_id, shard_count, reps))
    try:
        with pq.ParquetWriter(staging/replica_name, schema, compression='zstd') as writer:
            for batch in batches:
                for row in batch:
                    if row['phase'] != phase or (row['cell_id'], row['replicate_index']) != next(expected, None):
                        raise ValueError('batch has unexpected/duplicate/out-of-order replica')
                    accounting.consume(row)
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        if next(expected, None) is not None:
            raise ValueError('batch stream is incomplete')
        summary, disagreement = accounting.finish(shard_count == 1)
        _json(staging/(PREFIX+'manifest.json'), manifest)
        for name, rows in (('summary', summary), ('disagreement', disagreement)):
            enriched = [dict(row, harness_sha=identity['harness_sha'], engine_sha=identity['engine_sha'],
                             engine_blob=identity['engine_blob'], preregistration_version=identity['preregistration_version'],
                             manifest_sha256=identity['manifest_sha256']) for row in rows]
            frame = pd.DataFrame(enriched)
            frame.to_csv(staging/(PREFIX+name+'.csv'), index=False, float_format='%.17g', lineterminator='\n')
            if name == 'summary':
                table = pa.Table.from_pandas(frame, preserve_index=False)
                table = table.replace_schema_metadata({**(table.schema.metadata or {}), b'provenance': canonical(identity)})
                pq.write_table(table, staging/(PREFIX+'summary.parquet'), compression='zstd')
        invalid = any(r['confirmatory_gate'] == 'INVALID_EXECUTION' for r in summary)
        report = ('# ANOVA calibration report\n\n'
                  f'Phase: {phase}. Complete phase: {shard_count == 1}.\n\n'
                  f'Execution: {"INVALID_EXECUTION" if invalid else "ACCOUNTED"}. Parity: {gate["status"]}.\n\n'
                  'E0 is engineering-only. Robustness bands and power are descriptive. '
                  'This report does not authorize a selector or close CP-ANOVA-07.\n\n'
                  'Method completed denominators count finite successful method results. '
                  'Paired completed denominators count replicas where both methods succeeded. '
                  'Errors are explicit and never counted as non-rejections.\n\n'
                  'Per-cell/method/alpha rates, Wilson 99% CI, MC SE, gates and power monotonicity '
                  'are in summary.csv/parquet; paired counts are in disagreement.csv.\n\n'
                  '```json\n'+json.dumps(identity, indent=2)+'\n```\n')
        (staging/(PREFIX+'report.md')).write_text(report, encoding='utf-8')
        meta = dict(identity=identity, phase=phase, alpha_grid=manifest['alpha_grid'],
                    master_seed=manifest['holdout_master_seed' if phase.startswith('H-') else 'development_master_seed'], shard_id=shard_id, shard_count=shard_count,
                    replications_per_cell=reps, settings=settings, parity=gate,
                    replicate_file=replica_name, start_utc=started, end_utc=utc(),
                    execution_status='INVALID_EXECUTION' if invalid else 'ACCOUNTED',
                    counts=summary, paired_counts=disagreement,
                    checksums={p.name: file_hash(p) for p in staging.iterdir()})
        _json(staging/(PREFIX+'metadata.json'), meta)
        verify_directory(staging, identity, permit)
        # A directory becomes visible only when all artifacts and checksums exist.
        # Existing nonempty completed outputs cannot be replaced by rename.
        if output.exists():
            raise FileExistsError(output)
        os.rename(staging, output)
        return meta
    finally:
        if staging.exists():
            if staging.resolve().parent != output.parent or not staging.name.startswith(output.name+'.pending-'):
                raise RuntimeError('refusing cleanup outside this transaction directory')
            shutil.rmtree(staging)


def _batches(phase, cell, reps, shard_id, shard_count, batch_size, seed, permit=None):
    indices = list(range(shard_id, reps, shard_count))
    for offset in range(0, len(indices), batch_size):
        task = (phase, cell, indices[offset:offset+batch_size], seed)
        yield task if permit is None else (*task, permit)


def run_phase(phase, output, *, workers=1, batch_size=200, shard_id=0, shard_count=1, resume=False, holdout_authorization=None):
    """Only full frozen replication counts; there is no pilot-count override."""
    if holdout_authorization is None:
        check_phase(phase)
    manifest = load_manifest()
    reps = manifest['replications'][phase]
    if (any(type(v) is not int or v < 1 for v in (workers, batch_size, shard_count))
            or type(shard_id) is not int or not 0 <= shard_id < shard_count <= reps):
        raise ValueError('invalid worker/batch/shard configuration')
    identity = provenance(manifest)
    permit = holdout_permit(phase, holdout_authorization, identity)
    check_phase(phase, permit)
    gate = parity_gate(manifest, phase) if permit is None else parity_gate(manifest, phase, permit)
    if Path(output).exists():
        if not resume:
            raise FileExistsError(output)
        meta = verify_directory(output, identity, permit)
        if (meta['phase'], meta['shard_id'], meta['shard_count'], meta['replications_per_cell']) != (phase, shard_id, shard_count, reps):
            raise ValueError('resume shard configuration mismatch')
        if meta['execution_status'] == 'INVALID_EXECUTION':
            raise RuntimeError('confirmatory execution invalid; stored errors cannot be resumed as success')
        return meta
    def batches(pool):
        for cell in sorted(manifest['phase_cells'][phase], key=lambda c: c['cell_id']):
            seed = manifest['holdout_master_seed' if phase.startswith('H-') else 'development_master_seed']
            tasks = _batches(phase, cell, reps, shard_id, shard_count, batch_size, seed, permit)
            yield from (map(compute_batch, tasks) if pool is None else pool.map(compute_batch, tasks))
    context = (nullcontext(None) if workers == 1 else
               ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')))
    with context as pool:
        meta = _publish(output, manifest, identity, phase, shard_id, shard_count, reps, batches(pool),
                        dict(workers=workers, batch_size=batch_size, backend='cpu',
                             holdout_authorization_sha256=permit.authorization_sha256 if permit else None,
                             thread_environment={k: os.environ[k] for k in THREAD_VARS}), gate, permit)
    if meta['execution_status'] == 'INVALID_EXECUTION':
        raise RuntimeError('confirmatory execution invalid; accounting artifacts preserved')
    return meta


def recompose(inputs, output, holdout_authorization=None):
    if not inputs:
        raise ValueError('no shards')
    manifest = load_manifest()
    identity = provenance(manifest)
    first = json.loads((Path(inputs[0])/(PREFIX+'metadata.json')).read_text())
    phase, count, reps = first['phase'], first['shard_count'], first['replications_per_cell']
    permit = holdout_permit(phase, holdout_authorization, identity)
    check_phase(phase, permit)
    metas = [verify_directory(p, identity, permit) for p in inputs]
    if reps != manifest['replications'][phase]:
        raise ValueError('non-frozen replication count')
    if len(metas) != count or {m['shard_id'] for m in metas} != set(range(count)):
        raise ValueError('missing or duplicate shard')
    if any((m['phase'], m['shard_count'], m['replications_per_cell']) != (phase, count, reps) for m in metas):
        raise ValueError('incompatible shards')
    gate = parity_gate(manifest, phase) if permit is None else parity_gate(manifest, phase, permit)
    streams = [_rows(Path(p)/m['replicate_file']) for p, m in zip(inputs, metas)]
    merged = heapq.merge(*streams, key=lambda r: (r['cell_id'], r['replicate_index']))
    def batches():
        batch = []
        for row in merged:
            batch.append(row)
            if len(batch) == 1024:
                yield batch
                batch = []
        if batch:
            yield batch
    meta = _publish(output, manifest, identity, phase, 0, 1, reps, batches(),
                    dict(holdout_authorization_sha256=permit.authorization_sha256 if permit else None,
                         recomposed_shards=[dict(path=str(p), checksums=m['checksums']) for p,m in zip(inputs, metas)]), gate, permit)
    if meta['execution_status'] == 'INVALID_EXECUTION':
        raise RuntimeError('confirmatory execution invalid; accounting artifacts preserved')
    return meta
