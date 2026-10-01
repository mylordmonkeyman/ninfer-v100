#!/usr/bin/env python3
"""Summarize warmed paired model-execution timings; not a server throughput claim."""
import json
import re
import statistics
import sys
from pathlib import Path


def summarize(text):
    rows = []
    for match in re.finditer(r'phase11.moe_output_benchmark mode=(decode|prefill) pair=(\d+) fp32=([01]) positions=(\d+) elapsed_s=([\d.eE+-]+) tokens_per_s=([\d.eE+-]+)', text):
        mode, pair, fp32, positions, elapsed, tps = match.groups()
        rows.append(dict(mode=mode, pair=int(pair), fp32=int(fp32),
                         positions=int(positions), elapsed_s=float(elapsed), tokens_per_s=float(tps)))
    keys = {(r['mode'], r['pair'], r['fp32']) for r in rows}
    expected = {(m, p, f) for m in ('decode','prefill') for p in range(4) for f in (0,1)}
    if len(rows) != 16 or keys != expected or any(r['elapsed_s'] <= 0 or r['positions'] != 64 for r in rows):
        raise ValueError('Incomplete paired timings')
    result = {'scope':'64-position fresh-prefix host-backed Phase 11 execution, not optimized server throughput',
              'warmup_pairs_excluded':[0], 'samples':rows, 'modes':{}}
    for mode in ('decode','prefill'):
        samples = [r for r in rows if r['mode'] == mode and r['pair'] > 0]
        medians = {f:statistics.median(r['tokens_per_s'] for r in samples if r['fp32'] == f) for f in (0,1)}
        ratio = medians[1] / medians[0]
        result['modes'][mode] = {'baseline_median_tokens_per_s':medians[0],
            'modified_median_tokens_per_s':medians[1], 'throughput_ratio':ratio,
            'within_5_percent_loss':ratio >= .95}
    result['both_within_5_percent_loss'] = all(r['within_5_percent_loss'] for r in result['modes'].values())
    return result


if __name__ == '__main__':
    result = summarize(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
