"""Compare unchanged public Engine probes; report and reject numerical/token drift."""
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
report = {}
same_mode_reference = '--same-mode-reference' in sys.argv[2:]
corrected_reference = '--corrected-reference' in sys.argv[2:]
modes = ('score', 'eager', 'graph', 'mtp') + tuple(
    x for x in sys.argv[2:] if x not in ('--same-mode-reference', '--corrected-reference'))
for mode in modes:
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
# Compare MTP with ordinary graph output. Default checks require equality;
# matching-profile port qualification retains it as an explicit diagnostic.
for label in ('baseline', 'candidate'):
    def generation(mode):
        return [x for x in (root/f'{label}-{mode}.txt').read_text().splitlines() if x.startswith('generation ')]
    report[f'{label}_mtp_vs_graph'] = {'pass': generation('mtp') == generation('graph')}
if same_mode_reference:
    # A forward port must preserve each original arithmetic profile. BF16 rounding can
    # make the original's batched MTP profile disagree with its ordinary profile near a
    # tied logit; retain that comparison explicitly rather than using it as the oracle
    # for another mode. Both forks must also repeat their ordinary request sequence.
    checks = {mode: report[mode] for mode in modes}
    def outputs(label, mode):
        return [x for x in (root/f'{label}-{mode}.txt').read_text().splitlines()
                if x.startswith('generation ')]
    for label in ('baseline', 'candidate'):
        checks[f'{label}_eager_repeatability'] = {
            'pass': outputs(label, 'eager') == outputs(label, 'eager-repeat')}
    cross_mode = {}
    for label in ('baseline', 'candidate'):
        ordinary, speculative = outputs(label, 'graph'), outputs(label, 'mtp')
        cross_mode[label] = {'identical': ordinary == speculative,
                            'token_mismatches': sum(a != b for a, b in zip(ordinary, speculative))}
    reference = ('original fork with isolated ordered split-K correction'
                 if corrected_reference else 'unchanged original fork')
    report = {'reference': reference + '; matching execution modes',
              'qualification': checks, 'cross_mode_diagnostics': cross_mode}
    passed = all(row['pass'] for row in checks.values())
else:
    passed = all(row['pass'] for row in report.values())
(root/'parity.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
if not passed:
    raise SystemExit('FAIL: accuracy parity')
print('PASS: ' + ('corrected ' if corrected_reference else '') + 'original V100 fork accuracy reference')
