"""Engineering tests only: parity samples and synthetic storage fixtures, no phase run."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.anova_cp07 import manifest as config
from experiments.anova_cp07 import simulation as sim
from experiments.anova_cp07 import runner
from experiments.anova_cp07.accounting import Accounting, interval


@pytest.fixture
def manifest():
    return config.load_manifest()


def test_exact_manifest(manifest):
    assert config.digest(manifest) == 'affa3a1ae3c02b8081d0bdc761e6ce3725bb736899b0d2771d5d185530c0262a'
    assert {p:len(c) for p,c in manifest['phase_cells'].items()} == config.COUNTS
    assert len({c['cell_id'] for cells in manifest['phase_cells'].values() for c in cells}) == 197
    assert [c['cell_id'] for c in manifest['phase_cells']['E0']] == [f'E0-{i:02}' for i in range(1,13)]
    assert manifest['replications']['E0'] == 200
    cells = manifest['phase_cells']
    assert cells['D-robustness-h0'][0] == dict(cell_id='DRH0-F01-R01', family='gamma_shape_4', sizes=[5]*3, sd=[1]*3, hypothesis='H0')
    assert cells['D-robustness-h0'][-1]['cell_id'] == 'DRH0-F09-R06'
    assert cells['D-power-h1'][-1] == dict(cell_id='DPH1-P12-D03', family='mixture_symmetric_5pct_scale6', sizes=[30]*3, sd=[1]*3, hypothesis='H1', delta_range=1.)
    assert cells['H-robustness'][-1]['cell_id'] == 'HRH0-F07-HRD03'
    assert cells['H-power'][-1]['cell_id'] == 'HPH1-F04-D03'


@pytest.mark.parametrize('mutation', ['duplicate','missing','unexpected','sd','size','family','h0delta',
    'h1delta','holdout_leak','holdout_substitution','seed','alpha','count','order','phase'])
def test_validator_rejects_drift(manifest, mutation):
    cells = manifest['phase_cells']['E0']
    if mutation == 'duplicate': cells[1]['cell_id'] = cells[0]['cell_id']
    elif mutation == 'missing': cells.pop()
    elif mutation == 'unexpected': cells[0]['cell_id'] = 'E0-13'
    elif mutation == 'sd': cells[0]['sd'].pop()
    elif mutation == 'size': cells[0]['sizes'][0] = 6
    elif mutation == 'family': cells[0]['family'] = 'unknown'
    elif mutation == 'h0delta': cells[0]['delta_range'] = .5
    elif mutation == 'h1delta': del cells[-1]['delta_range']
    elif mutation == 'holdout_leak': cells[0]['family'] = 'beta_2_5'
    elif mutation == 'holdout_substitution': manifest['phase_cells']['H-power'][0]['family'] = 'normal'
    elif mutation == 'seed': manifest['development_master_seed'] += 1
    elif mutation == 'alpha': manifest['alpha_grid'][0] = .02
    elif mutation == 'count': manifest['replications']['E0'] = 201
    elif mutation == 'order': cells.reverse()
    elif mutation == 'phase': del manifest['phase_cells']['H-power']
    with pytest.raises(ValueError): config.validate_manifest(manifest)


def test_compact_tamper_and_duplicate_json_keys(tmp_path):
    path = tmp_path/'manifest.json'
    data = json.loads(config.SOURCE.read_text())
    data['phase_cells']['E0'][0]['sizes'][0] = 6
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): config.load_manifest(path)
    path.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match='duplicate JSON key'): config.load_manifest(path)


def test_rng_exact_contract_and_byte_replay(manifest):
    phase, cell, index, seed = 'E0', manifest['phase_cells']['E0'][4], 17, 2026090501
    raw = hashlib.sha256(f'{phase}|{cell["cell_id"]}|{index}'.encode()).digest()
    words = [int.from_bytes(raw[j:j+4], 'little') for j in range(0,16,4)]
    expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed]+words)))
    assert sim.replica_rng(phase,cell['cell_id'],index,seed).bytes(256) == expected.bytes(256)
    groups = sim.generate_groups(phase,cell,index,seed)
    again = sim.generate_groups(phase,cell,index,seed)
    assert [g.tobytes() for g in groups] == [g.tobytes() for g in again]
    assert sim.paired_replica(phase,cell,index,seed) == sim.paired_replica(phase,cell,index,seed)
    assert sim.sample_digest(groups) != sim.sample_digest(sim.generate_groups(phase,cell,index+1,seed))


def test_same_sample_one_summary_and_both_before_next(monkeypatch, manifest):
    calls = []
    original_gen, original_sum = sim.generate_groups, sim.engine._summarize_groups
    original_c, original_w = sim.engine._classical_kernel, sim.engine._welch_kernel
    def gen(*args):
        groups = original_gen(*args); calls.append(('generate', groups)); return groups
    def summarize(groups):
        assert groups is calls[-1][1]
        summaries = original_sum(groups); calls.append(('summarize', summaries)); return summaries
    def classical(summaries):
        assert summaries is calls[-1][1]
        calls.append(('classical', summaries)); return original_c(summaries)
    def welch(summaries):
        assert summaries is calls[-1][1]
        calls.append(('welch', summaries)); return original_w(summaries)
    monkeypatch.setattr(sim, 'generate_groups', gen)
    monkeypatch.setattr(sim.engine, '_summarize_groups', summarize)
    monkeypatch.setattr(sim.engine, '_classical_kernel', classical)
    monkeypatch.setattr(sim.engine, '_welch_kernel', welch)
    sim.compute_batch(('E0',manifest['phase_cells']['E0'][0],[35,36],2026090501))
    assert [c[0] for c in calls] == ['generate','summarize','classical','welch']*2


def test_workers_batches_shards_and_execution_order(manifest):
    cell = manifest['phase_cells']['E0'][6]
    def tasks(batch_size, indices):
        return [('E0',cell,indices[i:i+batch_size],2026090501) for i in range(0,len(indices),batch_size)]
    def flat(batches):
        return sorted([r for b in batches for r in b], key=lambda r:r['replicate_index'])
    indices = list(range(32,44))
    serial = flat(map(sim.compute_batch,tasks(1,indices)))
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        parallel = flat(pool.map(sim.compute_batch,tasks(5,list(reversed(indices)))))
    assert parallel == serial
    shards = [sim.compute_batch(('E0',cell,indices[s::3],2026090501)) for s in range(3)]
    assert flat(shards) == serial
    assert flat(map(sim.compute_batch,tasks(7,indices))) == serial


@pytest.mark.parametrize('phase', ['E0','D-core-h0','D-robustness-h0','D-stress-h0','D-power-h1'])
def test_production_parity_all_unsealed_cells(manifest, phase):
    gate = sim.parity_gate(manifest,phase)
    assert gate['comparisons'] == config.COUNTS[phase]*32*2
    assert gate['status'] == 'PASS'


def test_parity_mismatch_aborts(monkeypatch,manifest):
    monkeypatch.setattr(sim.engine.OneWayANOVA,'run',lambda self:SimpleNamespace(statistic=999.,p_value=.5,numerator_df=2.,denominator_df=12.))
    with pytest.raises(RuntimeError,match='parity mismatch'): sim.parity_gate(manifest,'E0')


@pytest.mark.parametrize('phase',['H-core-normal','H-robustness','H-power'])
def test_holdout_stops_before_generation(monkeypatch,manifest,tmp_path,phase):
    def forbidden(*a,**kw): raise AssertionError('holdout sampled')
    monkeypatch.setattr(sim,'replica_rng',forbidden)
    with pytest.raises(PermissionError): sim.generate_groups(phase,manifest['phase_cells'][phase][0],0,2026090599)
    with pytest.raises(PermissionError): sim.parity_gate(manifest,phase)
    with pytest.raises(PermissionError): runner.run_phase(phase,tmp_path/'sealed')
    assert not (tmp_path/'sealed').exists()


def test_error_accounting_and_no_error_as_nonrejection(monkeypatch,manifest):
    cell = manifest['phase_cells']['E0'][0]
    def fail(*args): raise ValueError('injected')
    original_gen = sim.generate_groups
    monkeypatch.setattr(sim,'generate_groups',fail)
    gen_error = sim.paired_replica('E0',cell,45,2026090501)
    monkeypatch.setattr(sim,'generate_groups',original_gen)
    original_c = sim.engine._classical_kernel
    monkeypatch.setattr(sim.engine,'_classical_kernel',fail)
    kernel_error = sim.paired_replica('E0',cell,45,2026090501)
    monkeypatch.setattr(sim.engine,'_classical_kernel',lambda s:SimpleNamespace(statistic=float('nan'),p_value=.2,numerator_df=2.,denominator_df=12.))
    nonfinite = sim.paired_replica('E0',cell,46,2026090501)
    monkeypatch.setattr(sim.engine,'_classical_kernel',original_c)
    ok = sim.paired_replica('E0',cell,47,2026090501)
    accounting = Accounting(manifest,'E0')
    for row in (gen_error,kernel_error,nonfinite,ok): accounting.consume(row)
    summary,pairs = accounting.finish(False)
    c = next(r for r in summary if r['cell_id']==cell['cell_id'] and r['method']=='classical')
    assert [c[k] for k in ('replications_requested','replications_completed','generation_error_count','kernel_error_count','nonfinite_count')] == [4,1,1,1,1]
    p = next(r for r in pairs if r['cell_id']==cell['cell_id'])
    assert p['replications_completed'] == 1
    assert sum(p[k] for k in ('both_reject_count','classical_only_reject_count','welch_only_reject_count','neither_reject_count')) == 1


def test_wilson_known_values():
    from statsmodels.stats.proportion import proportion_confint
    rate,low,high,se = interval(2500,50000)
    assert rate == .05
    expected = proportion_confint(2500,50000,alpha=.01,method='wilson')
    np.testing.assert_allclose([low,high],expected,rtol=1e-14,atol=1e-16)
    assert se == pytest.approx(np.sqrt(.05*.95/50000))
    assert interval(0,0) == (None,)*4
    assert interval(0,10)[1] == pytest.approx(0,abs=1e-16)


def synthetic_batch(task):
    """No RNG or engine calls; only handcrafted rows for persistence tests."""
    phase,cell,indices,seed = task
    rows=[]
    for i in indices:
        row=dict(phase=phase,cell_id=cell['cell_id'],replicate_index=i,sample_sha256='synthetic',generation_error='',warning_count=0)
        for method,p in [('classical',[.005,.03,.08,.5][i%4]),('welch',[.5,.08,.03,.005][i%4])]:
            row.update({method+'_status':'ok',method+'_error':'',method+'_statistic':1.,method+'_p_value':p,
                        method+'_numerator_df':2.,method+'_denominator_df':12.})
        rows.append(row)
    return rows


@pytest.fixture
def synthetic_runner(monkeypatch,manifest):
    identity=dict(harness_sha='synthetic-test-sha',engine_sha=runner.ENGINE_COMMIT,engine_blob=runner.ENGINE_BLOB,
                  preregistration_version=config.VERSION,manifest_sha256=config.digest(manifest))
    monkeypatch.setattr(runner,'provenance',lambda m:identity)
    monkeypatch.setattr(runner,'parity_gate',lambda m,p:dict(status='PASS',replicas_per_cell=32,fixture='synthetic'))
    monkeypatch.setattr(runner,'compute_batch',synthetic_batch)
    return identity


def test_transactional_shards_recompose_resume(tmp_path,synthetic_runner,monkeypatch):
    whole = tmp_path/'whole'
    runner.run_phase('E0',whole,batch_size=31)
    shards = [tmp_path/f'shard{i}' for i in range(3)]
    for i,path in enumerate(shards): runner.run_phase('E0',path,shard_id=i,shard_count=3,batch_size=17)
    hashes = {p.name:runner.file_hash(p) for p in shards[1].iterdir()}
    def forbidden(*args): raise AssertionError('valid resume recomputed a replica')
    monkeypatch.setattr(runner,'compute_batch',forbidden)
    runner.run_phase('E0',shards[1],shard_id=1,shard_count=3,batch_size=99,resume=True)
    assert hashes == {p.name:runner.file_hash(p) for p in shards[1].iterdir()}
    combined=tmp_path/'combined'
    runner.recompose(list(reversed(shards)),combined)
    for name in ('summary.csv','disagreement.csv'):
        assert (whole/(runner.PREFIX+name)).read_bytes() == (combined/(runner.PREFIX+name)).read_bytes()
    original=runner.verify_directory(whole)
    merged=runner.verify_directory(combined)
    assert list(runner._rows(whole/original['replicate_file'])) == list(runner._rows(combined/merged['replicate_file']))
    assert len(list(whole.iterdir())) == 7
    with pytest.raises(FileExistsError): runner.run_phase('E0',whole)
    with pytest.raises(ValueError,match='shard'): runner.recompose([shards[0],shards[0],shards[2]],tmp_path/'bad')
    with pytest.raises(ValueError,match='shard'): runner.recompose(shards[:2],tmp_path/'bad')
    with pytest.raises(ValueError,match='configuration'): runner.run_phase('E0',shards[0],resume=True)


def test_corrupt_shard_and_mixed_identity_rejected(tmp_path,synthetic_runner):
    output=tmp_path/'out'
    meta=runner.run_phase('E0',output)
    changed=dict(synthetic_runner,harness_sha='different')
    with pytest.raises(ValueError,match='mix'): runner.verify_directory(output,changed)
    with (output/meta['replicate_file']).open('ab') as f: f.write(b'corruption')
    with pytest.raises(ValueError,match='checksum'): runner.run_phase('E0',output,resume=True)


def test_interrupted_publish_never_exposes_partial_output(tmp_path,synthetic_runner,monkeypatch):
    calls=0
    def interrupted(task):
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError('simulated interruption')
        return synthetic_batch(task)
    monkeypatch.setattr(runner,'compute_batch',interrupted)
    output=tmp_path/'out'
    with pytest.raises(RuntimeError,match='interruption'): runner.run_phase('E0',output,batch_size=7)
    assert not output.exists()
    monkeypatch.setattr(runner,'compute_batch',synthetic_batch)
    runner.run_phase('E0',output,batch_size=13,resume=True)
    runner.verify_directory(output)


def test_gate_failure_prevents_even_synthetic_mc(tmp_path,synthetic_runner,monkeypatch):
    def fail(*args): raise RuntimeError('parity mismatch')
    def forbidden(*args): raise AssertionError('Monte Carlo started')
    monkeypatch.setattr(runner,'parity_gate',fail)
    monkeypatch.setattr(runner,'compute_batch',forbidden)
    with pytest.raises(RuntimeError,match='parity mismatch'): runner.run_phase('E0',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_no_selector_execution(monkeypatch,manifest):
    from pyMagicStat.inference.selector import MethodSelector
    def forbidden(*args,**kwargs): raise AssertionError('selector executed')
    monkeypatch.setattr(MethodSelector,'select',forbidden)
    row=sim.paired_replica('E0',manifest['phase_cells']['E0'][0],48,2026090501)
    assert row['classical_status']==row['welch_status']=='ok'


def test_holdout_declaration_validation_without_opening(manifest,tmp_path):
    identity=dict(harness_sha='candidate',manifest_sha256=config.digest(manifest))
    with pytest.raises(PermissionError): runner.holdout_permit('H-power',None,identity)
    path=tmp_path/'invalid-authorization.json'
    path.write_text(json.dumps(dict(action='open-holdout',authorized_by='Product Owner',
        harness_sha='wrong-candidate',manifest_sha256=config.digest(manifest),phase_d_complete=True,
        remediations_closed=True,candidate_frozen=True)))
    with pytest.raises(PermissionError): runner.holdout_permit('H-power',path,identity)


def test_exact_generator_transforms_without_random_samples():
    """Deterministic RNG stub verifies formulas, including sealed-family support.

    No holdout stream or holdout data are generated.
    """
    class Stub:
        def normal(self,size): return np.ones(size)
        def gamma(self,shape,size): return np.full(size,shape+2)
        def lognormal(self,mean,sigma,n): return np.full(n,3.)
        def standard_t(self,df,n): return np.full(n,2.)
        def laplace(self,mean,scale,n):
            assert scale == 1/np.sqrt(2)
            return np.full(n,scale)
        def random(self,n): return np.zeros(n)
        def weibull(self,a,n): return np.full(n,2.)
        def pareto(self,a,n): return np.full(n,1.)
        def beta(self,a,b,n): return np.full(n,.5)
    import math
    stub=Stub()
    for shape in (1,2,4):
        np.testing.assert_array_equal(sim.standardized(stub,f'gamma_shape_{shape}',2),np.full(2,2/math.sqrt(shape)))
    for label,sigma in [('0p5',.5),('1p2',1.2),('1p5',1.5),('0p8',.8)]:
        value=(3-math.exp(sigma*sigma/2))/math.sqrt(math.expm1(sigma*sigma)*math.exp(sigma*sigma))
        np.testing.assert_array_equal(sim.standardized(stub,'lognormal_sigma_'+label,2),np.full(2,value))
    for label,df in [('3',3),('5',5),('2p5',2.5),('7',7)]:
        np.testing.assert_array_equal(sim.standardized(stub,'student_t_df_'+label,2),np.full(2,2*math.sqrt((df-2)/df)))
    np.testing.assert_array_equal(sim.standardized(stub,'mixture_symmetric_5pct_scale6',2),np.full(2,6/math.sqrt(.95+.05*36)))
    for pct in (2,5,10):
        q=pct/100
        value=(11-10*q)/math.sqrt(1+q*(1-q)*100)
        np.testing.assert_array_equal(sim.standardized(stub,f'contamination_asymmetric_{pct}pct_loc10',2),np.full(2,value))
    mean=math.gamma(1+1/1.5)
    np.testing.assert_array_equal(sim.standardized(stub,'weibull_shape_1p5',2),np.full(2,(2-mean)/math.sqrt(math.gamma(1+2/1.5)-mean*mean)))
    np.testing.assert_array_equal(sim.standardized(stub,'pareto_alpha_3p5',2),np.full(2,(2-3.5/2.5)/math.sqrt(3.5/(2.5**2*1.5))))
    np.testing.assert_array_equal(sim.standardized(stub,'beta_2_5',2),np.full(2,(.5-2/7)/math.sqrt(10/(49*8))))


def test_exact_offsets_and_sd_without_rescaling(monkeypatch,manifest):
    monkeypatch.setattr(sim,'standardized',lambda rng,f,n:np.ones(n))
    cell=manifest['phase_cells']['D-power-h1'][14]  # P05, delta 1: sd [4,2,1]
    groups=sim.generate_groups('D-power-h1',cell,50,2026090501)
    for actual,n,value in zip(groups,[5,10,20],[3.5,2.,1.5]):
        np.testing.assert_array_equal(actual,np.full(n,value))


@pytest.mark.parametrize('kind',['duplicate','missing','unexpected'])
def test_replica_coverage_rejects_bad_streams(manifest,kind):
    cell=manifest['phase_cells']['E0'][0]
    rows=synthetic_batch(('E0',cell,[0,1,2],2026090501))
    if kind=='duplicate': rows[1]=deepcopy(rows[0])
    elif kind=='missing': rows.pop()
    else: rows[-1]['cell_id']='unexpected'
    with pytest.raises(ValueError):
        list(runner._validate_rows(rows,[(cell['cell_id'],i) for i in range(3)],Accounting(manifest,'E0')))


def test_paired_categories_and_independent_denominator(manifest):
    cell=manifest['phase_cells']['E0'][0]
    rows=synthetic_batch(('E0',cell,[0,1,2,3],2026090501))
    accounting=Accounting(manifest,'E0')
    for row,(c,w) in zip(rows,[(.001,.002),(.003,.2),(.3,.004),(.4,.5)]):
        row['classical_p_value'],row['welch_p_value']=c,w
        accounting.consume(row)
    _,pairs=accounting.finish(True)
    pair=next(p for p in pairs if p['cell_id']==cell['cell_id'] and p['alpha']==.05)
    assert [pair[k] for k in ('both_reject_count','classical_only_reject_count','welch_only_reject_count','neither_reject_count')] == [1]*4
    assert pair['replications_completed']==4
    assert pair['classical_minus_welch_rejection_rate']==0
    accounting.pairs[cell['cell_id']][1][0]+=1
    with pytest.raises(ValueError,match='!= completed'): accounting.finish(True)


def test_confirmatory_gates_and_error_invalidation(manifest):
    accounting=Accounting(manifest,'D-core-h0')
    for c in accounting.counts.values():
        c.update(requested=50000,ok=50000,reject=[500,2500,5000])
    summary,_=accounting.finish(True)
    eligible=[r for r in summary if r['alpha']==.05 and (r['method']=='welch' or r['cell_id'].startswith('DCEV'))]
    assert len(eligible)==66 and all(r['confirmatory_gate']=='PASS' for r in eligible)
    partial,_=accounting.finish(False)
    assert not any(r['confirmatory_gate']=='PASS' for r in partial)
    c=accounting.counts['DCEV-k2-01','classical']
    c.update(ok=49999,kernel_error=1)
    summary,_=accounting.finish(True)
    assert next(r for r in summary if r['cell_id']=='DCEV-k2-01' and r['method']=='classical')['confirmatory_gate']=='INVALID_EXECUTION'


def test_provenance_rejects_dirty_and_changed_engine(monkeypatch,manifest):
    monkeypatch.setattr(runner,'git',lambda *args:' M changed' if args[0]=='status' else '')
    with pytest.raises(RuntimeError,match='clean'): runner.provenance(manifest)
    def changed(*args):
        if args[0]=='status': return ''
        if args[0]=='branch': return 'engineering/cp-anova-07a-harness'
        return 'changed-blob'
    monkeypatch.setattr(runner,'git',changed)
    with pytest.raises(RuntimeError,match='frozen engine'): runner.provenance(manifest)
