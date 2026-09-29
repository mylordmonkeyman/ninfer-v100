#!/usr/bin/env python3
"""Replay FP32 expert sets in separate CPU layer groups over 768 full prefixes.

Each run retains the candidate CPU storage profile and its own router scores.
Only selected membership at one layer group is replaced; other gates route
naturally from the evolving state of that same complete-prefix run.
"""
import argparse
import csv
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from compare_router_precision_4096 import router_sets, summarize
from precision_metrics import compare_logits
from precision_profile import PROFILES
from run_oracle import build_oracle
from run_precision_reference import read_ids, run_decode


COUNT = 768
GROUPS = ((0, 16), (16, 32), (32, 48))


def replay_group(model, selected, group, state):
    originals = []
    for layer in range(*group):
        gate = model.layers[layer].mlp.gate
        original = gate.forward
        originals.append((gate, original))

        def replay(self, hidden, layer=layer, original=original):
            scores, _alpha, _natural = original(hidden)
            position = state["position"]
            if position >= COUNT or scores.shape[0] != 1:
                raise ValueError(f"unexpected replay position/shape {position}, {tuple(scores.shape)}")
            chosen = selected[layer][position].to(device=scores.device).reshape(1, 10)
            if len(set(chosen[0].tolist())) != 10:
                raise ValueError(f"duplicate selected experts at {position}, layer {layer}")
            alpha = torch.softmax(scores.float(), dim=-1).gather(-1, chosen)
            if self.norm_topk_prob:
                alpha = alpha / alpha.sum(dim=-1, keepdim=True)
            state["replayed_calls"] += 1
            return scores, alpha.to(scores.dtype), chosen

        gate.forward = MethodType(replay, gate)
    return originals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ple-dir", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--three-way-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    ids = read_ids(args.oracle_root / "token_ids.json", COUNT)
    manifest = json.loads((args.oracle_root / "manifest.json").read_text())
    if len(manifest["positions"]) < COUNT:
        raise ValueError("frozen oracle has fewer than 768 positions")
    frozen_paths = []
    for position, token in enumerate(ids):
        entry = manifest["positions"][position]
        if entry["position"] != position or entry["token_id"] != token:
            raise ValueError(f"frozen FP32 token/position mismatch at {position}")
        tensors = [t for t in entry["tensors"] if t["name"] == "logits"]
        if len(tensors) != 1:
            raise ValueError(f"missing FP32 logits at {position}")
        frozen_paths.append(args.oracle_root / tensors[0]["file"])
    with args.three_way_csv.open(newline="") as stream:
        natural = list(csv.DictReader(stream))
    if len(natural) != 4096 or any(int(row["position"]) != i
                                   for i, row in enumerate(natural)):
        raise ValueError("natural three-way report lacks aligned 4096 positions")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, head = build_oracle(str(args.model_dir), str(args.ple_dir))
    model.eval()
    selected, parity = router_sets(model, head, ids, frozen_paths, COUNT)
    if len(selected) != 48 or any(tuple(s.shape) != (COUNT, 10)
                                  for s in selected.values()):
        raise ValueError("incomplete FP32 membership capture")
    for block in model.layers:
        block.mlp.experts.round_activations_to_bf16 = True
    report = {
        "meaning": "diagnostic FP32 membership in one layer group at a time; natural remaining routing, CPU scores and weights, complete prefixes; no V100 arithmetic change",
        "provenance": "frozen 4096-position FP32 oracle and natural CPU/V100 report; fresh 768-position full FP32 router sets",
        "fresh_768_full_vs_frozen_fp32": parity,
        "groups": {},
    }
    for group in GROUPS:
        name = f"layers_{group[0]:02d}_{group[1]-1:02d}"
        state = {"position": 0, "layer": 0, "replayed_calls": 0,
                 "observed_calls": 0, "natural_calls": 0}
        rows = []
        originals = replay_group(model, selected, group, state)
        handles = []
        for layer, block in enumerate(model.layers):
            def capture(_module, _args, output, layer=layer):
                if state["position"] >= COUNT or state["layer"] != layer:
                    raise ValueError(f"unexpected gate at {state['position']}, layer {layer}")
                chosen = output[2].detach().cpu().reshape(-1)
                if chosen.numel() != 10 or len(set(chosen.tolist())) != 10:
                    raise ValueError(f"invalid gate selection at {state['position']}, layer {layer}")
                if group[0] <= layer < group[1]:
                    expected = selected[layer][state["position"]]
                    if not torch.equal(chosen, expected):
                        raise ValueError(f"replay selection mismatch at {state['position']}, layer {layer}")
                else:
                    state["natural_calls"] += 1
                state["observed_calls"] += 1
                state["layer"] += 1
                if state["layer"] == len(model.layers):
                    state["layer"] = 0
                    state["position"] += 1
            handles.append(block.mlp.gate.register_forward_hook(capture))
        try:
            def sink(position, logits):
                if state["position"] != position + 1 or state["layer"] != 0:
                    raise ValueError(f"incomplete router calls at {position}")
                metrics = compare_logits(np.fromfile(frozen_paths[position], dtype="<f4"),
                                         logits)
                prior = natural[position]
                rows.append({
                    "position": position,
                    "natural_cpu_kl": float(prior["oracle_cpu_kl"]),
                    "natural_cpu_top1": int(prior["oracle_cpu_top1"]),
                    "natural_v100_kl": float(prior["oracle_v100_kl"]),
                    "natural_v100_top1": int(prior["oracle_v100_top1"]),
                    "variant_kl": metrics["kl"],
                    "variant_top1": int(metrics["top1_agree"]),
                })
                if (position + 1) % 128 == 0:
                    print(f"{name}: complete_prefixes={position + 1}/{COUNT}", flush=True)

            run_decode(model, head, ids, PROFILES["v100-phase11-storage"], None,
                       fused_hyper_updates=True, logits_sink=sink,
                       retain_logits=False)
        finally:
            for handle in handles:
                handle.remove()
            for gate, original in originals:
                gate.forward = original
        if (len(rows) != COUNT or state["position"] != COUNT or
                state["observed_calls"] != COUNT * 48 or
                state["replayed_calls"] != COUNT * (group[1] - group[0]) or
                state["natural_calls"] != COUNT * (48 - group[1] + group[0])):
            raise ValueError(f"incomplete group comparison {name}: {state}")
        with (args.out_dir / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["groups"][name] = {
            "router_calls": state,
            "ranges": {f"{lo}:{hi}": summarize(rows, lo, hi)
                       for lo, hi in ((0, 512), (512, 768), (0, 768))},
        }
        (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({name: report["groups"][name]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
