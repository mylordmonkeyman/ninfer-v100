"""Compare unchanged public Engine probes; report and reject numerical/token drift."""
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
report = {}
eager_reference = '--eager-reference' in sys.argv[2:]
modes = ('score', 'eager', 'graph', 'mtp') + tuple(x for x in sys.argv[2:] if x != '--eager-reference')
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
# MTP must also preserve ordinary graph greedy generation in each implementation.
for label in ('baseline', 'candidate'):
    def generation(mode):
        return [x for x in (root/f'{label}-{mode}.txt').read_text().splitlines() if x.startswith('generation ')]
    report[f'{label}_mtp_vs_graph'] = {'pass': generation('mtp') == generation('graph')}
if eager_reference:
    # The old fork's cold graph output can disagree with its own eager output. Keep
    # same-mode differences visible, and check every candidate mode against the
    # unchanged original Engine's deterministic eager greedy result instead.
    same_mode = report
    checks = {}
    def outputs(label, mode):
        return [x for x in (root/f'{label}-{mode}.txt').read_text().splitlines()
                if x.startswith('generation ')]
    expected = outputs('baseline', 'eager')
    checks['baseline_eager_repeatability'] = {'pass': expected == outputs('baseline', 'eager-repeat')}
    checks['score'] = same_mode['score']
    diagnostics = {}
    for mode in modes:
        if mode == 'score':
            continue
        actual = outputs('candidate', mode)
        mismatches = sum(a != b for a, b in zip(expected, actual))
        checks[mode] = {'token_mismatches': mismatches, 'pass': len(expected) == len(actual) and mismatches == 0}
        # All explicit inputs and tokenizer results must still match across versions.
        def inputs(label):
            return [x for x in (root/f'{label}-{mode}.txt').read_text().splitlines()
                    if not x.startswith('generation ')]
        checks[mode]['pass'] &= inputs('baseline') == inputs('candidate')
        old = outputs('baseline', mode)
        diagnostics[mode] = {'token_mismatches': sum(a != b for a, b in zip(expected, old)),
                             'matches_eager': expected == old}
    report = {'reference': 'unchanged original fork eager greedy generation; exact scoring',
              'qualification': checks, 'same_mode_comparison': same_mode,
              'original_modes_vs_original_eager': diagnostics}
    passed = all(row['pass'] for row in checks.values())
else:
    passed = all(row['pass'] for row in report.values())
(root/'parity.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
if not passed:
    raise SystemExit('FAIL: accuracy parity')
print('PASS: original V100 fork accuracy reference')
