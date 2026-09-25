#!/usr/bin/env python3
"""Evaluate FP32 GDN convolution history over complete teacher-forced prefixes."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from run_oracle import build_oracle
from run_precision_reference import read_ids, read_oracle, run_decode, compare_logits
from precision_profile import PROFILES


def summarize(oracle, computed):
    rows = [compare_logits(item[0], logits) for item, logits in zip(oracle, computed)]
    kl = np.array([row['kl'] for row in rows])
    return {'mean_kl': float(kl.mean()), 'p99_kl': float(np.percentile(kl, 99)),
            'top1_agreement': sum(row['top1_agree'] for row in rows),
            'positions': len(rows), 'per_position': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--ple-dir', type=Path, required=True)
    parser.add_argument('--ids-file', type=Path, required=True)
    parser.add_argument('--fp32-oracle', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    ids = read_ids(args.ids_file, 14)
    oracle = read_oracle(args.fp32_oracle, ids)
    model, head = build_oracle(str(args.model_dir), str(args.ple_dir))
    model.eval()
    for layer in model.layers:
        layer.mlp.experts.round_activations_to_bf16 = True
    baseline = PROFILES['v100-phase11-storage']
    upgraded = replace(baseline, name='fp32-conv-history', conv_state_bf16=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    normal = run_decode(model, head, ids, baseline, args.out_dir/'baseline')
    changed = run_decode(model, head, ids, upgraded, args.out_dir/'fp32-conv-history')
    report = {'baseline': summarize(oracle, normal),
              'fp32_conv_history': summarize(oracle, changed)}
    (args.out_dir/'summary.json').write_text(json.dumps(report, indent=2))
    for name, data in report.items():
        print(name, 'mean_kl', data['mean_kl'], 'p99_kl', data['p99_kl'],
              'top1', data['top1_agreement'], '/', data['positions'], flush=True)
    if abs(report['baseline']['mean_kl'] - 0.03190201725371168) > 1e-5:
        raise RuntimeError('Baseline CPU precision profile did not reproduce frozen run')


if __name__ == '__main__':
    main()
