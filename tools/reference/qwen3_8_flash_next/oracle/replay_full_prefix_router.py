#!/usr/bin/env python3
"""Replay full-forward FP32 router sets in the CPU precision profile over complete prefixes.

All paths use the same complete token prefix. The existing 4096-position report
supplies natural CPU and V100 results; this run adds a controlled CPU replay.
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
from precision_profile import PROFILES
from run_oracle import build_oracle
from run_precision_reference import read_ids, run_decode
from replay_v100_full_prefix_router import parse_top1


def summary(rows, start, stop):
    part = rows[start:stop]
    return {
        "positions": len(part),
        "oracle_cpu_natural_mean_kl": sum(r["natural_cpu_kl"] for r in part) / len(part),
        "oracle_cpu_replay_mean_kl": sum(r["replayed_cpu_kl"] for r in part) / len(part),
        "oracle_v100_natural_mean_kl": sum(r["natural_v100_kl"] for r in part) / len(part),
        "cpu_natural_top1": sum(r["natural_cpu_top1"] for r in part),
        "cpu_replay_top1": sum(r["replayed_cpu_top1"] for r in part),
        "v100_natural_top1": sum(r["natural_v100_top1"] for r in part),
        "replay_lower_kl_positions": sum(
            r["replayed_cpu_kl"] < r["natural_cpu_kl"] for r in part),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ple-dir", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--three-way-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--positions", type=int, choices=(768, 4096), default=768)
    parser.add_argument("--export-v100-stage", action="store_true")
    parser.add_argument("--export-logits", action="store_true")
    args = parser.parse_args()
    count = args.positions
    ids = read_ids(args.oracle_root / "token_ids.json", count)
    manifest = json.loads((args.oracle_root / "manifest.json").read_text())
    entries = manifest["positions"]
    if len(entries) < count:
        raise ValueError("frozen FP32 oracle has fewer than 4096 positions")
    frozen_paths = []
    for pos, token in enumerate(ids):
        entry = entries[pos]
        if entry["position"] != pos or entry["token_id"] != token:
            raise ValueError(f"frozen oracle token/position mismatch at {pos}")
        matches = [t for t in entry["tensors"] if t["name"] == "logits"]
        if len(matches) != 1:
            raise ValueError(f"frozen oracle lacks one logits tensor at {pos}")
        frozen_paths.append(args.oracle_root / matches[0]["file"])
    with args.three_way_csv.open(newline="") as stream:
        original = list(csv.DictReader(stream))
    if len(original) < count or any(int(original[i]["position"]) != i
                                    for i in range(count)):
        raise ValueError("three-way report lacks 4096 aligned positions")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, head = build_oracle(str(args.model_dir), str(args.ple_dir))
    model.eval()
    routed = {}
    handles = []
    for layer, block in enumerate(model.layers):
        def capture(_module, _args, output, layer=layer):
            selected = output[2].detach().cpu()
            if selected.numel() != count * 10 or selected.shape[-1] != 10:
                raise ValueError(
                    f"unexpected full-forward routing shape {tuple(selected.shape)} "
                    f"at layer {layer}")
            routed[layer] = selected.reshape(count, 10).clone()
            print(f"fp32_router_capture_layer={layer}", flush=True)
        handles.append(block.mlp.gate.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            forward = model(input_ids=torch.tensor([ids], dtype=torch.long),
                            use_cache=False)
        hidden = forward.last_hidden_state
        if hidden.shape[:2] != (1, count) or len(routed) != len(model.layers):
            raise ValueError("incomplete full-forward router capture")
        fresh_frozen = []
        with torch.inference_mode():
            for start in range(0, count, 8):
                logits = F.linear(hidden[:, start:start + 8], head)[0]
                for offset, values in enumerate(logits):
                    pos = start + offset
                    frozen = np.fromfile(frozen_paths[pos], dtype="<f4")
                    fresh_frozen.append(compare_logits(
                        frozen, values.numpy().astype("<f4", copy=False))["kl"])
    finally:
        for handle in handles:
            handle.remove()
    del forward, hidden
    forced = {(pos, layer): routed[layer][pos] for layer in routed
              for pos in range(count)}
    if args.export_v100_stage:
        # Export exactly the same full-forward sets for the actual V100 replay.
        stage_root = args.out_dir / "router-stage"
        stage_root.mkdir()
        for pos in range(count):
            folder = stage_root / f"pos{pos:04d}"
            folder.mkdir()
            for layer in range(48):
                chosen = forced[pos, layer].numpy().astype(np.int64)
                if len(set(chosen.tolist())) != 10 or chosen.min() < 0 or chosen.max() >= 512:
                    raise ValueError(f"invalid membership at {pos}, layer {layer}")
                chosen.astype("<f4").tofile(folder / f"L{layer:02d}_moe_router_ids.bin")
                np.zeros(10, dtype="<f4").tofile(folder / f"L{layer:02d}_moe_router_alpha.bin")
    parity = {"positions": count, "max_kl": max(fresh_frozen),
              "mean_kl": sum(fresh_frozen) / count}
    (args.out_dir / "oracle-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    print("fresh_full_vs_frozen=" + json.dumps(parity), flush=True)
    del routed
    logits_root = args.out_dir / "cpu-logits"
    if args.export_logits:
        logits_root.mkdir()
    cpu_metrics = (args.out_dir / "cpu-vs-oracle.jsonl").open("w")
    for layer in model.layers:
        layer.mlp.experts.round_activations_to_bf16 = True

    rows = []
    def capture_replay(pos, values):
        frozen = np.fromfile(frozen_paths[pos], dtype="<f4")
        if not np.isfinite(frozen).all():
            raise ValueError(f"nonfinite frozen oracle at {pos}")
        result = compare_logits(frozen, values)
        # The candidate profile returns BF16-representable FP32 logits.
        words = (values.view("<u4") >> 16).astype("<u2")
        restored = (words.astype("<u4") << 16).view("<f4")
        if not np.array_equal(values, restored):
            raise ValueError(f"CPU logits are not BF16 representable at {pos}")
        if args.export_logits:
            words.tofile(logits_root / f"pos{pos:04d}.bf16")
        cpu_metrics.write(json.dumps({"position": pos, **result}) + "\n")
        cpu_metrics.flush()
        prior = original[pos]
        rows.append({
            "position": pos,
            "natural_cpu_kl": float(prior["oracle_cpu_kl"]),
            "replayed_cpu_kl": result["kl"],
            "natural_v100_kl": float(prior["oracle_v100_kl"]),
            "natural_cpu_top1": parse_top1(prior["oracle_cpu_top1"]),
            "replayed_cpu_top1": result["top1_agree"],
            "natural_v100_top1": parse_top1(prior["oracle_v100_top1"]),
        })
        if (pos + 1) % 64 == 0:
            print(f"controlled_complete_prefixes={pos + 1}/{count}", flush=True)

    run_decode(model, head, ids, PROFILES["v100-phase11-storage"], None,
               forced_router_ids=forced, fused_hyper_updates=True,
               logits_sink=capture_replay, retain_logits=False)
    cpu_metrics.close()
    if len(rows) != count:
        raise RuntimeError(f"CPU replay completed {len(rows)} of {count} positions")
    with (args.out_dir / "per_position.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "provenance": {
            "fp32": f"frozen 4096-position full forward; fresh {count}-position full forward supplies routing",
            "cpu_natural_and_v100_natural": "completed 4096-position three-way report",
            "cpu_replay": "same quantized-weight BF16-storage CPU profile, full-forward FP32 expert sets, recomputed CPU router weights",
            "prefix": "all paths carry complete prefixes through each reported position",
        },
        "fresh_full_vs_frozen_fp32_max_kl": max(fresh_frozen),
        "fresh_full_vs_frozen_fp32_mean_kl": sum(fresh_frozen) / count,
        "ranges": {f"{start}:{stop}": summary(rows, start, stop)
                   for start, stop in (((0, 512), (512, 768), (0, 768)) if count == 768 else
                                      ((0, 768), (512, 768), (768, 1024), (1024, 2048), (2048, 3072), (3072, 4096), (0, 4096)))},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
