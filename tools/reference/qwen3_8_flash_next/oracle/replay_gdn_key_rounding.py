#!/usr/bin/env python3
"""Test one observed BF16 convolution-history tie at layer 0, token 10.

The intervention is diagnostic: it uses the measured V100 storage value and
does not propose a deployable routing or precision policy.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_oracle import build_oracle
from run_precision_reference import read_ids, read_oracle
from precision_profile import (PROFILES, install_projection_boundaries,
                               materialize_persistent_states, round_to_bf16)
from precision_metrics import compare_logits


def nrmse(reference, candidate):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    assert a.shape == b.shape and a.size
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def run(model, head, ids, patch):
    profile = PROFILES['v100-phase11-storage']
    captured = []
    def recurrent(_module, args):
        captured.append(args[0].detach().float().reshape(-1).cpu().numpy().copy())

    hook = model.layers[0].linear_attn.norm.register_forward_pre_hook(recurrent)
    handles = install_projection_boundaries(model, profile)
    cache = None
    snapshots = {}
    logits = []
    try:
        with torch.inference_mode():
            for pos, token in enumerate(ids):
                output = model(input_ids=torch.tensor([[token]], dtype=torch.long),
                               past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                materialize_persistent_states(cache, model, profile)
                snapshots[pos] = captured[-1]
                hidden = output.last_hidden_state[0, -1].to(torch.float32)
                logit = round_to_bf16(F.linear(hidden, head))
                logits.append(logit.detach().cpu().numpy().astype('<f4'))
                if patch and pos == 10:
                    matches = []
                    for key, state in cache.layers[0].conv_states.items():
                        if state is None or not state.is_floating_point():
                            continue
                        for axis, length in enumerate(state.shape):
                            if length != 10240:
                                continue
                            channel = state.select(axis, 3097)
                            for location in (channel == -29.125).nonzero(as_tuple=False):
                                matches.append((key, state, axis, tuple(location.tolist())))
                    if len(matches) != 1:
                        shape_map = {key: tuple(value.shape) for key, value in
                                     cache.layers[0].conv_states.items() if value is not None}
                        raise RuntimeError(f'Expected one key history match, got {len(matches)} '
                                           f'in {shape_map}')
                    key, state, axis, location = matches[0]
                    channel = state.select(axis, 3097)
                    channel[location] = -29.0
                    print(f'patched token=10 state={key} shape={tuple(state.shape)} '
                          f'channel=3097 location={location} before=-29.125 after=-29', flush=True)
    finally:
        hook.remove()
        for handle in handles:
            handle.remove()
    return snapshots, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--ple-dir', type=Path, required=True)
    parser.add_argument('--ids-file', type=Path, required=True)
    parser.add_argument('--cpu-trace', type=Path, required=True)
    parser.add_argument('--v100-trace', type=Path, required=True)
    parser.add_argument('--fp32-oracle', type=Path, required=True)
    args = parser.parse_args()
    ids = read_ids(args.ids_file, 14)
    oracle = read_oracle(args.fp32_oracle, ids)
    model, head = build_oracle(str(args.model_dir), str(args.ple_dir))
    model.eval()
    for layer in model.layers:
        layer.mlp.experts.round_activations_to_bf16 = True
    baseline, baseline_logits = run(model, head, ids, False)
    patched, patched_logits = run(model, head, ids, True)
    for pos in (10, 11, 12, 13):
        relative = f'pos{pos:04d}/L00_gdn_recurrent_output.bin'
        cpu = np.fromfile(args.cpu_trace / relative, dtype='<f4')
        v100 = np.fromfile(args.v100_trace / relative, dtype='<f4')
        print(f'replay position={pos} baseline-frozen-cpu={nrmse(cpu,baseline[pos]):.9g} '
              f'baseline-v100={nrmse(v100,baseline[pos]):.9g} '
              f'patched-v100={nrmse(v100,patched[pos]):.9g}', flush=True)
        if nrmse(cpu, baseline[pos]) > 1e-6:
            raise RuntimeError('Baseline CPU replay differs from frozen profile')
        if pos >= 11:
            for head in (24, 25, 26):
                sl = slice(head*128, (head+1)*128)
                print(f'replay position={pos} head={head} '
                      f'baseline-v100={nrmse(v100[sl],baseline[pos][sl]):.9g} '
                      f'patched-v100={nrmse(v100[sl],patched[pos][sl]):.9g}', flush=True)
    v100_logits = [np.fromfile(args.v100_trace/f'pos{pos:04d}/logits.bin', dtype='<f4')
                   for pos in range(len(ids))]
    for label, logits in (('baseline',baseline_logits),('patched',patched_logits),
                          ('v100',v100_logits)):
        rows = [compare_logits(ref[0], actual)
                for ref, actual in zip(oracle,logits)]
        print(f'logits variant={label} oracle_mean_kl={np.mean([r["kl"] for r in rows]):.9g} '
              f'oracle_top1={sum(r["top1_agree"] for r in rows)}/14 '
              f'position10_kl={rows[10]["kl"]:.9g} '
              f'position13_kl={rows[13]["kl"]:.9g}', flush=True)
        if label != 'v100':
            nearby = [compare_logits(reference,actual)
                      for reference,actual in zip(v100_logits,logits)]
            print(f'logits variant={label} v100_mean_kl={np.mean([r["kl"] for r in nearby]):.9g}',
                  flush=True)


if __name__ == '__main__':
    main()
