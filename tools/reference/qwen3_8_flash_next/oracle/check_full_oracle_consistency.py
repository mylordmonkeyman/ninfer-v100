#!/usr/bin/env python3
"""Full-forward oracle consistency and identical-input router arithmetic diagnostic.

No candidate precision change, incremental oracle, or qualification threshold edit.
Run in separate processes with different pinned CPU thread counts.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from precision_metrics import compare_logits
from run_oracle import build_oracle
from run_precision_reference import read_ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'ple-dir', 'oracle-root', 'out-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prior-dir', type=Path)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    count = 4096
    ids = read_ids(a.oracle_root / 'token_ids.json', count)
    entries = json.loads((a.oracle_root / 'manifest.json').read_text())['positions']
    paths = []
    for pos, token in enumerate(ids):
        e = entries[pos]
        assert e['position'] == pos and e['token_id'] == token
        ts = [t for t in e['tensors'] if t['name'] == 'logits']
        assert len(ts) == 1
        paths.append(a.oracle_root / ts[0]['file'])
    # Fixed stratified sample includes the previously observed incremental mismatch.
    sample = sorted(set(range(0, count, 64)) | {12, 127, 189, 190, 191, 511, 512, 767, 768, 1023, 2047, 3071, 4095})
    model, head = build_oracle(str(a.model_dir), str(a.ple_dir))
    handles, routing_rows = [], []
    for layer, block in enumerate(model.layers):
        def capture(gate, args, output, layer=layer):
            x = args[0].detach().reshape(count, -1)[sample].clone()
            scores = output[0].detach().reshape(count, 512)[sample].clone()
            chosen = output[2].detach().reshape(count, 10)
            np.save(a.out_dir / f'L{layer:02d}_ids.npy', chosen.numpy())
            np.save(a.out_dir / f'L{layer:02d}_inputs.npy', x.numpy())
            np.save(a.out_dir / f'L{layer:02d}_scores.npy', scores.numpy())
            weight = gate.weight.detach()
            for i, pos in enumerate(sample):
                row = x[i:i+1]
                precise = F.linear(row.double(), weight.double()).float()[0]
                single = F.linear(row, weight)[0]
                rounded = F.linear(row.to(torch.bfloat16).float(), weight)[0]
                expected = set(chosen[pos].tolist())
                item = {'position': pos, 'layer': layer}
                for name, values in [('fp64_same_input', precise), ('fp32_single_same_input', single), ('bf16_input_single', rounded)]:
                    selected = torch.topk(F.softmax(values, dtype=torch.float, dim=-1), 10).indices
                    item[name + '_membership_changed'] = set(selected.tolist()) != expected
                    item[name + '_score_max_abs'] = float((values - scores[i]).abs().max())
                routing_rows.append(item)
            print(f'full_forward_layer={layer}', flush=True)
        handles.append(block.mlp.gate.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            forward = model(input_ids=torch.tensor([ids]), use_cache=False)
            rows = []
            for start in range(0, count, 8):
                logits = F.linear(forward.last_hidden_state[:, start:start+8], head)[0]
                for offset, values in enumerate(logits):
                    pos = start + offset
                    reference = np.fromfile(paths[pos], dtype='<f4')
                    assert np.isfinite(reference).all()
                    rows.append({'position': pos, **compare_logits(reference, values.numpy())})
    finally:
        for h in handles:
            h.remove()
    assert len(rows) == count and len(routing_rows) == 48 * len(sample)
    for name, data in [('per_position.csv', rows), ('identical_input_router.csv', routing_rows)]:
        with (a.out_dir / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    kls = sorted(r['kl'] for r in rows)
    report = {'positions': count, 'torch_threads': torch.get_num_threads(),
              'torch_version': torch.__version__, 'sample_positions': sample,
              'fresh_full_vs_frozen': {'mean_kl': sum(kls)/count, 'max_kl': max(kls),
                  'p99_nearest_rank': kls[math.ceil(.99*count)-1],
                  'top1_matches': sum(r['top1_agree'] for r in rows)},
              'worst_positions': sorted(rows, key=lambda r: r['kl'], reverse=True)[:20],
              'same_input_router_calls': len(routing_rows),
              'same_input_membership_changes': {name: sum(r[name + '_membership_changed'] for r in routing_rows)
                  for name in ('fp64_same_input', 'fp32_single_same_input', 'bf16_input_single')},
              'meaning': 'Oracle consistency and sampled CPU score arithmetic only; no actual V100 same-input claim or natural qualification'}
    if a.prior_dir:
        compared, changed, sample_diffs = 0, 0, []
        for layer in range(48):
            before = np.load(a.prior_dir / f'L{layer:02d}_ids.npy')
            after = np.load(a.out_dir / f'L{layer:02d}_ids.npy')
            assert before.shape == after.shape == (count, 10)
            changed += int(np.any(np.sort(before, axis=1) != np.sort(after, axis=1), axis=1).sum())
            compared += count
            x0 = np.load(a.prior_dir / f'L{layer:02d}_inputs.npy')
            x1 = np.load(a.out_dir / f'L{layer:02d}_inputs.npy')
            s0 = np.load(a.prior_dir / f'L{layer:02d}_scores.npy')
            s1 = np.load(a.out_dir / f'L{layer:02d}_scores.npy')
            sample_diffs.append({'layer': layer, 'input_max_abs': float(np.max(np.abs(x1-x0))),
                                 'score_max_abs': float(np.max(np.abs(s1-s0)))})
        report['cross_thread_full_forward'] = {'membership_calls': compared, 'membership_changes': changed,
                                              'sample_input_and_score_differences': sample_diffs}
    (a.out_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
