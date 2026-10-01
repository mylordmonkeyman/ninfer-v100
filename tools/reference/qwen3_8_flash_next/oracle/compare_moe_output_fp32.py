#!/usr/bin/env python3
"""Natural sequential CPU candidate with only final MoE output rounding removed."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from precision_metrics import compare_logits
from precision_profile import PROFILES, round_to_bf16
from run_oracle import build_oracle
from run_precision_reference import read_ids, run_decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'ple-dir', 'oracle-root', 'out-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    ids = read_ids(args.oracle_root / 'token_ids.json', 4096)
    entries = json.loads((args.oracle_root / 'manifest.json').read_text())['positions']
    if len(ids) != 4096 or len(entries) < 4096:
        raise ValueError('Need 4096 complete sequential positions')
    paths = []
    for pos, token in enumerate(ids):
        entry = entries[pos]
        assert entry['position'] == pos and entry['token_id'] == token
        tensors = [x for x in entry['tensors'] if x['name'] == 'logits']
        assert len(tensors) == 1
        paths.append(args.oracle_root / tensors[0]['file'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'cpu-logits').mkdir(exist_ok=True)
    log = args.out_dir / 'cpu-vs-oracle.jsonl'
    log.write_text('')
    model, head = build_oracle(str(args.model_dir), str(args.ple_dir))
    model.eval()
    state = {'router_calls': 0, 'fp32_output_calls': 0, 'unrounded_outputs': 0}
    handles = []
    for block in model.layers:
        block.mlp.experts.round_activations_to_bf16 = True
        def verify_input(_module, inputs):
            assert len(inputs) == 1 and torch.equal(inputs[0], round_to_bf16(inputs[0]))
            state['router_calls'] += 1
        def verify_output(_module, _args, output):
            assert output.dtype == torch.float32 and torch.isfinite(output).all()
            state['fp32_output_calls'] += 1
            state['unrounded_outputs'] += int(not torch.equal(output, round_to_bf16(output)))
        handles.append(block.mlp.gate.register_forward_pre_hook(verify_input))
        handles.append(block.mlp.register_forward_hook(verify_output))
    rows = []
    def sink(pos, logits):
        assert state['router_calls'] == (pos + 1) * 48
        assert state['fp32_output_calls'] == (pos + 1) * 48
        bits = logits.view('<u4')
        assert np.isfinite(logits).all() and not np.any(bits & 0xffff)
        (bits >> 16).astype('<u2').tofile(args.out_dir / 'cpu-logits' / f'pos{pos:04d}.bf16')
        metrics = compare_logits(np.fromfile(paths[pos], dtype='<f4'), logits)
        row = {**metrics, 'position': pos}
        rows.append(row)
        with log.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if (pos + 1) % 128 == 0:
            print(f'FP32 MoE output CPU complete_prefixes={pos+1}/4096', flush=True)
    try:
        run_decode(model, head, ids, PROFILES['v100-phase11-storage'], None,
                   fused_hyper_updates=True, moe_output_fp32=True,
                   logits_sink=sink, retain_logits=False)
    finally:
        for handle in handles:
            handle.remove()
    assert len(rows) == 4096 and state['unrounded_outputs'] > 0
    report = {**state, 'positions': len(rows), 'routing': 'natural',
              'profile': 'v100-phase11-storage; fused hyper; FP32 combined MoE output',
              'mean_kl': float(np.mean([r['kl'] for r in rows])),
              'top1': sum(int(r['top1_agree']) for r in rows)}
    (args.out_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
