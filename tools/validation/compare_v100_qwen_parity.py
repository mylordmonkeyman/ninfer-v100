"""Compare unchanged public Engine probes; report and reject numerical/token drift."""
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
report = {}
for mode in ('score', 'eager', 'graph', 'mtp') + tuple(sys.argv[2:]):
    def read(label):
        rows = [line.split() for line in (root / f'{label}-{mode}.txt').read_text().splitlines()]
        return {(r[0], r[1]): float(r[2]) for r in rows}
    baseline, candidate = read('baseline'), read('candidate')
    if baseline.keys() != candidate.keys():
        raise SystemExit(f'{mode}: row keys differ')
    errors = [abs(baseline[k] - candidate[k]) for k in baseline if k[0] == 'score']
    mismatches = [k for k in baseline if k[0] != 'score' and baseline[k] != candidate[k]]
    if not all(math.isfinite(x) for x in (*baseline.values(), *candidate.values())):
        raise SystemExit('nonfinite output')
    report[mode] = {'rows': len(baseline), 'max_logprob_error': max(errors, default=0),
                    'mean_logprob_error': sum(errors)/len(errors) if errors else 0,
                    'token_mismatches': len(mismatches)}
    # Same artifact/kernels/profile: tight numerical parity, exact input and greedy tokens.
    report[mode]['pass'] = not mismatches and max(errors, default=0) <= 1e-5
# MTP must also preserve ordinary graph greedy generation in each implementation.
for label in ('baseline', 'candidate'):
    def generation(mode):
        return [x for x in (root/f'{label}-{mode}.txt').read_text().splitlines() if x.startswith('generation ')]
    report[f'{label}_mtp_vs_graph'] = {'pass': generation('mtp') == generation('graph')}
(root/'parity.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
if not all(row['pass'] for row in report.values()):
    raise SystemExit('FAIL: accuracy parity')
print('PASS: original V100 fork accuracy parity')
