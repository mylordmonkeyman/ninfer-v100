#!/usr/bin/env python3
"""Test unrounded router input with otherwise unchanged candidate CPU storage.

This is a hypothetical reference profile over complete prefixes. The MLP and
experts still consume the candidate's BF16 block input; only the router gate
receives the unrounded FP32 value from the same hyper connection.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from compare_router_precision_4096 import router_sets, summarize
from precision_metrics import compare_logits
from precision_profile import PROFILES, round_to_bf16
from run_oracle import build_oracle
from run_precision_reference import read_ids, run_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ple-dir", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--three-way-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--positions", type=int, choices=(768, 4096), default=768)
    args = parser.parse_args()
    count, membership_length = args.positions, 768
    ids = read_ids(args.oracle_root / "token_ids.json", count)
    manifest = json.loads((args.oracle_root / "manifest.json").read_text())
    if len(manifest["positions"]) < count:
        raise ValueError("frozen oracle does not cover requested positions")
    frozen_paths = []
    for position, token in enumerate(ids):
        entry = manifest["positions"][position]
        if entry["position"] != position or entry["token_id"] != token:
            raise ValueError(f"frozen FP32 token/position mismatch at {position}")
        matches = [tensor for tensor in entry["tensors"] if tensor["name"] == "logits"]
        if len(matches) != 1:
            raise ValueError(f"missing FP32 logits at {position}")
        frozen_paths.append(args.oracle_root / matches[0]["file"])
    with args.three_way_csv.open(newline="") as stream:
        natural = list(csv.DictReader(stream))
    if len(natural) != 4096 or any(int(row["position"]) != index
                                   for index, row in enumerate(natural)):
        raise ValueError("natural three-way report lacks aligned 4096 positions")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, head = build_oracle(str(args.model_dir), str(args.ple_dir))
    model.eval()
    reference_sets, fp32_parity = router_sets(
        model, head, ids, frozen_paths, membership_length)
    for block in model.layers:
        block.mlp.experts.round_activations_to_bf16 = True

    rows = []
    state = {"position": 0, "layer": 0, "matching_sets": 0,
             "compared_sets": 0, "rounded_input_checks": 0,
             "unrounded_input_calls": 0}
    raw_inputs = {}
    handles = []
    for layer, block in enumerate(model.layers):
        def save_raw(_module, _args, output, layer=layer):
            if layer in raw_inputs:
                raise ValueError(f"unconsumed router input at layer {layer}")
            raw_inputs[layer] = output[0]

        def router_input(_module, inputs, layer=layer):
            if layer not in raw_inputs or len(inputs) != 1:
                raise ValueError(f"missing raw router input at layer {layer}")
            raw = raw_inputs.pop(layer)
            rounded = inputs[0]
            if raw.numel() != rounded.numel() or raw.dtype != torch.float32:
                raise ValueError(f"unexpected router input geometry at layer {layer}")
            raw = raw.reshape(rounded.shape)
            if not torch.equal(rounded, round_to_bf16(raw)):
                raise ValueError(f"experts do not retain BF16 block input at layer {layer}")
            state["rounded_input_checks"] += 1
            state["unrounded_input_calls"] += int(not torch.equal(raw, rounded))
            return (raw,)

        def capture(_module, _args, output, layer=layer):
            position = state["position"]
            if layer != state["layer"] or position >= count:
                raise ValueError(f"unexpected router call at {position}, layer {layer}")
            if position < membership_length:
                actual = output[2].detach().cpu().reshape(-1)
                expected = reference_sets[layer][position]
                if actual.numel() != 10 or len(set(actual.tolist())) != 10:
                    raise ValueError(f"invalid selected experts at {position}, layer {layer}")
                state["matching_sets"] += set(actual.tolist()) == set(expected.tolist())
                state["compared_sets"] += 1
            state["layer"] += 1
            if state["layer"] == len(model.layers):
                state["layer"] = 0
                state["position"] += 1

        # Registered before run_decode installs its BF16 output hook. Thus
        # save_raw sees the original FP32 value, while gate's pre-hook sees
        # the already rounded input passed by the unchanged MLP forward.
        handles.append(block.mlp_hyper_connection.register_forward_hook(save_raw))
        handles.append(block.mlp.gate.register_forward_pre_hook(router_input))
        handles.append(block.mlp.gate.register_forward_hook(capture))

    try:
        def sink(position, logits):
            if state["position"] != position + 1 or state["layer"] != 0 or raw_inputs:
                raise ValueError(f"incomplete router calls at position {position}")
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
                print(f"router_input_only: complete_prefixes={position + 1}/{count}",
                      flush=True)

        run_decode(model, head, ids, PROFILES["v100-phase11-storage"], None,
                   fused_hyper_updates=True, logits_sink=sink,
                   retain_logits=False)
    finally:
        for handle in handles:
            handle.remove()
    expected_calls = count * len(model.layers)
    if (len(rows) != count or state["position"] != count or
            state["rounded_input_checks"] != expected_calls or raw_inputs or
            state["compared_sets"] != membership_length * len(model.layers)):
        raise ValueError("incomplete router-input-only comparison")
    with (args.out_dir / "per_position.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ranges = ((0, 768), (512, 768)) if count == 768 else (
        (0, 768), (512, 768), (0, 1024), (0, 4096))
    report = {
        "provenance": "frozen independent FP32 oracle and natural CPU/V100 4096-position report",
        "meaning": "hypothetical CPU router input precision only; natural scores and routing over complete prefixes; unchanged V100 and section 7 gate",
        "positions": count,
        "fresh_768_full_vs_frozen_fp32": fp32_parity,
        "router_calls": expected_calls,
        "rounded_expert_input_verified_calls": state["rounded_input_checks"],
        "unrounded_router_input_effective_calls": state["unrounded_input_calls"],
        "fp32_router_set_agreement_0_768": {
            "matching": state["matching_sets"], "total": state["compared_sets"]},
        "ranges": {f"{start}:{stop}": summarize(rows, start, stop)
                   for start, stop in ranges},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
