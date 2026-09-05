"""Integer accounting; rates never treat an error as a non-rejection."""
import math
from .simulation import METHODS


def interval(rejections, completed):
    if not completed:
        return (None, None, None, None)
    p = rejections/completed
    z = 2.5758293035489004
    denominator = 1+z*z/completed
    center = (p+z*z/(2*completed))/denominator
    half = z*math.sqrt(p*(1-p)/completed+z*z/(4*completed**2))/denominator
    return p, max(0., center-half), min(1., center+half), math.sqrt(p*(1-p)/completed)


class Accounting:
    def __init__(self, manifest, phase):
        self.manifest, self.phase = manifest, phase
        self.counts = {}
        self.pairs = {}
        self.paired_completed = {}
        for cell in manifest['phase_cells'][phase]:
            key = cell['cell_id']
            for method in METHODS:
                self.counts[key, method] = dict(requested=0, ok=0, generation_error=0,
                                                kernel_error=0, nonfinite=0, warnings=0,
                                                reject=[0]*len(manifest['alpha_grid']))
            self.pairs[key] = [[0]*4 for _ in manifest['alpha_grid']]
            self.paired_completed[key] = 0

    def consume(self, row):
        key = row['cell_id']
        for method in METHODS:
            counts = self.counts[key, method]
            status = row[method+'_status']
            if status not in ('ok', 'generation_error', 'kernel_error', 'nonfinite'):
                raise ValueError('unknown replica status')
            counts['requested'] += 1
            counts[status] += 1
            counts['warnings'] += row['warning_count']
            if status == 'ok':
                p = row[method+'_p_value']
                if not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError('invalid completed p-value')
                for j, alpha in enumerate(self.manifest['alpha_grid']):
                    counts['reject'][j] += int(p < alpha)
        if all(row[m+'_status'] == 'ok' for m in METHODS):
            self.paired_completed[key] += 1
            for j, alpha in enumerate(self.manifest['alpha_grid']):
                c = row['classical_p_value'] < alpha
                w = row['welch_p_value'] < alpha
                slot = 0 if c and w else 1 if c else 2 if w else 3
                self.pairs[key][j][slot] += 1

    def finish(self, complete_phase):
        summary, disagreement = [], []
        for cell in sorted(self.manifest['phase_cells'][self.phase], key=lambda c: c['cell_id']):
            key = cell['cell_id']
            for method in METHODS:
                c = self.counts[key, method]
                if sum(c[s] for s in ('ok', 'generation_error', 'kernel_error', 'nonfinite')) != c['requested']:
                    raise ValueError('method accounting mismatch')
                for j, alpha in enumerate(self.manifest['alpha_grid']):
                    rate, low, high, se = interval(c['reject'][j], c['ok'])
                    gate = 'NOT_APPLICABLE'
                    core = self.phase in ('D-core-h0', 'H-core-normal')
                    eligible = core and alpha == .05 and min(cell['sizes']) >= 5 and (
                        method == 'welch' or len(set(cell['sd'])) == 1)
                    if core and (c['ok'] != c['requested']):
                        gate = 'INVALID_EXECUTION'
                    elif eligible:
                        gate = ('PASS' if low is not None and low >= .04 and high <= .06 else 'FAIL') if complete_phase else 'INCOMPLETE'
                    band = None
                    if self.phase in ('D-robustness-h0', 'H-robustness') and alpha == .05 and rate is not None:
                        deviation = abs(rate-.05)
                        band = 'green' if deviation <= .01 else 'amber' if deviation <= .025 else 'red'
                    summary.append(dict(phase=self.phase, cell_id=key, method=method, alpha=alpha,
                        family=cell['family'], hypothesis=cell['hypothesis'], delta_range=cell.get('delta_range'),
                        replications_requested=c['requested'], replications_completed=c['ok'],
                        generation_error_count=c['generation_error'], kernel_error_count=c['kernel_error'],
                        nonfinite_count=c['nonfinite'], warning_count=c['warnings'],
                        rejection_count=c['reject'][j], rejection_rate=rate, ci99_low=low, ci99_high=high,
                        mc_standard_error=se, confirmatory_gate=gate, robustness_band=band,
                        power_monotonicity_flag=None))
            for j, alpha in enumerate(self.manifest['alpha_grid']):
                both, c_only, w_only, neither = self.pairs[key][j]
                completed = sum(self.pairs[key][j])
                if completed != self.paired_completed[key]:
                    raise ValueError('both + classical_only + welch_only + neither != completed')
                if j and completed != sum(self.pairs[key][0]):
                    raise ValueError('paired accounting mismatch across alpha')
                disagreement.append(dict(phase=self.phase, cell_id=key, alpha=alpha,
                    replications_requested=self.counts[key, 'classical']['requested'],
                    replications_completed=completed, both_reject_count=both,
                    classical_only_reject_count=c_only, welch_only_reject_count=w_only,
                    neither_reject_count=neither,
                    classical_minus_welch_rejection_rate=(c_only-w_only)/completed if completed else None))
        # Descriptive monotonicity only, within identical family/design/method/alpha.
        cells = {c['cell_id']: c for c in self.manifest['phase_cells'][self.phase]}
        groups = {}
        for row in summary:
            cell = cells[row['cell_id']]
            if cell['hypothesis'] == 'H1':
                key = (cell['family'], tuple(cell['sizes']), tuple(cell['sd']), row['method'], row['alpha'])
                groups.setdefault(key, []).append(row)
        for rows in groups.values():
            rows.sort(key=lambda r: r['delta_range'])
            if len(rows) > 1 and all(r['rejection_rate'] is not None for r in rows):
                flag = any(a['rejection_rate'] > b['rejection_rate'] for a, b in zip(rows, rows[1:]))
                for row in rows:
                    row['power_monotonicity_flag'] = flag
        return summary, disagreement
