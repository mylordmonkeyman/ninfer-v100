#!/usr/bin/env python3
"""Compare isolated CPU precision upgrades with the frozen 4096-prefix three-way run.

These are hypothetical reference profiles. Neither candidate engine nor its
qualification gate is changed. All decode paths carry their own complete prefix.
"""
import argparse
import csv
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

from compare_full_prefix import nearest_p99
from precision_metrics import compare_logits
from precision_profile import PROFILES
from run_oracle import build_oracle
from run_precision_reference import read_ids, run_decode


def router_sets(model, head, ids, frozen_paths, length):
    selected = {}
    handles = []
    for layer, block in enumerate(model.layers):
        def capture(_module, _args, output, layer=layer):
            values = output[2].detach().cpu()
            if values.numel() != length * 10 or values.shape[-1] != 10:
                raise ValueError(f"invalid full-forward router shape at layer {layer}")
            selected[layer] = values.reshape(length, 10).clone()
        handles.append(block.mlp.gate.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            forward = model(input_ids=torch.tensor([ids[:length]], dtype=torch.long),
                            use_cache=False)
            hidden = forward.last_hidden_state
            if tuple(hidden.shape[:2]) != (1, length) or len(selected) != len(model.layers):
                raise ValueError("incomplete full-forward routing capture")
            differences = []
            for start in range(0, length, 8):
                logits = F.linear(hidden[:, start:start + 8], head)[0]
                for offset, values in enumerate(logits):
                    frozen = np.fromfile(frozen_paths[start + offset], dtype="<f4")
                    differences.append(compare_logits(frozen, values.cpu().numpy())["kl"])
    finally:
        for handle in handles:
            handle.remove()
    if max(differences) > 1e-4:
        raise ValueError(f"fresh FP32 forward diverges from frozen oracle: {max(differences)}")
    return selected, {"max_kl": max(differences),
                      "mean_kl": sum(differences) / length}


def use_fp64_router_scores(model):
    originals = []
    for block in model.layers:
        gate = block.mlp.gate
        if tuple(gate.weight.shape) != (512, 2560) or gate.top_k != 10:
            raise ValueError("unexpected gate geometry")
        originals.append((gate, gate.forward))

        def precise(self, hidden):
            hidden = hidden.reshape(-1, self.hidden_dim)
            # Only the matrix-product accumulation changes. Keep the original
            # materialized input, weight values, FP32 scores, softmax, and top-k.
            scores = F.linear(hidden.double(), self.weight.double()).float()
            probabilities = F.softmax(scores, dtype=torch.float, dim=-1)
            alpha, chosen = torch.topk(probabilities, self.top_k, dim=-1)
            if self.norm_topk_prob:
                alpha = alpha / alpha.sum(dim=-1, keepdim=True)
            return scores, alpha.to(scores.dtype), chosen

        gate.forward = MethodType(precise, gate)
    return originals


def summarize(rows, start, stop):
    part = rows[start:stop]
    result = {"positions": len(part)}
    for label in ("natural_cpu", "natural_v100", "variant"):
        result[label] = {
            "mean_kl": sum(r[f"{label}_kl"] for r in part) / len(part),
            "p99_kl_nearest_rank": nearest_p99([r[f"{label}_kl"] for r in part]),
            "top1_agreement": sum(r[f"{label}_top1"] for r in part),
        }
    result["variant_lower_kl_than_cpu_positions"] = sum(
        r["variant_kl"] < r["natural_cpu_kl"] for r in part)
    result["variant_lower_kl_than_v100_positions"] = sum(
        r["variant_kl"] < r["natural_v100_kl"] for r in part)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ple-dir", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--three-way-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    count, membership_length = 4096, 768
    ids = read_ids(args.oracle_root / "token_ids.json", count)
    manifest = json.loads((args.oracle_root / "manifest.json").read_text())
    if len(manifest["positions"]) < count:
        raise ValueError("frozen oracle does not cover 4096 positions")
    frozen_paths = []
    for pos, token in enumerate(ids):
        entry = manifest["positions"][pos]
        if entry["position"] != pos or entry["token_id"] != token:
            raise ValueError(f"frozen FP32 token/position mismatch at {pos}")
        matches = [t for t in entry["tensors"] if t["name"] == "logits"]
        if len(matches) != 1:
            raise ValueError(f"missing FP32 logits at {pos}")
        frozen_paths.append(args.oracle_root / matches[0]["file"])
    with args.three_way_csv.open(newline="") as stream:
        natural = list(csv.DictReader(stream))
    if len(natural) != count or any(int(row["position"]) != i
                                    for i, row in enumerate(natural)):
        raise ValueError("natural three-way comparison lacks aligned 4096 positions")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, head = build_oracle(str(args.model_dir), str(args.ple_dir))
    model.eval()
    reference_sets, fp32_parity = router_sets(
        model, head, ids, frozen_paths, membership_length)
    for block in model.layers:
        block.mlp.experts.round_activations_to_bf16 = True
    report = {
        "provenance": "frozen independent full-forward FP32 oracle and natural CPU/V100 4096-position report",
        "meaning": "hypothetical CPU reference precision changes; natural routing and complete prefixes; unchanged V100 and section 7 gate",
        "fresh_768_full_vs_frozen_fp32": fp32_parity,
        "variants": {},
    }
    for name in ("fp64_router_scores", "fp32_mlp_block_input_control"):
        rows, current = [], {"position": 0, "layer": 0, "matching_sets": 0,
                             "compared_sets": 0}
        handles = []
        for layer, block in enumerate(model.layers):
            def capture(_module, _args, output, layer=layer):
                position = current["position"]
                if layer != current["layer"] or position >= count:
                    raise ValueError(f"unexpected router call at {position}, layer {layer}")
                if position < membership_length:
                    actual = output[2].detach().cpu().reshape(-1)
                    expected = reference_sets[layer][position]
                    if actual.numel() != 10 or len(set(actual.tolist())) != 10:
                        raise ValueError(f"invalid selected experts at {position}, layer {layer}")
                    current["matching_sets"] += set(actual.tolist()) == set(expected.tolist())
                    current["compared_sets"] += 1
                current["layer"] += 1
                if current["layer"] == len(model.layers):
                    current["layer"] = 0
                    current["position"] += 1
            handles.append(block.mlp.gate.register_forward_hook(capture))
        originals = []
        try:
            if name == "fp64_router_scores":
                originals = use_fp64_router_scores(model)

            def sink(position, logits):
                if current["position"] != position + 1 or current["layer"] != 0:
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
                if (position + 1) % 256 == 0:
                    print(f"{name}: complete_prefixes={position + 1}/{count}", flush=True)

            run_decode(model, head, ids, PROFILES["v100-phase11-storage"], None,
                       mlp_block_input_fp32=name == "fp32_mlp_block_input_control",
                       fused_hyper_updates=True, logits_sink=sink,
                       retain_logits=False)
        finally:
            for handle in handles:
                handle.remove()
            for gate, original in originals:
                gate.forward = original
        if len(rows) != count or current["position"] != count:
            raise ValueError(f"{name} completed only {len(rows)} positions")
        if current["compared_sets"] != membership_length * len(model.layers):
            raise ValueError(f"{name} missing router membership observations")
        with (args.out_dir / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["variants"][name] = {
            "fp32_router_set_agreement_0_768": {
                "matching": current["matching_sets"],
                "total": current["compared_sets"],
            },
            "ranges": {f"{start}:{stop}": summarize(rows, start, stop)
                       for start, stop in ((0, 768), (512, 768), (0, 1024), (0, 4096))},
        }
        (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({name: report["variants"][name]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
