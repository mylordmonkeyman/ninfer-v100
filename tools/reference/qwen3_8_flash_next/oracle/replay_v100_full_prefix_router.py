#!/usr/bin/env python3
"""Export full-forward FP32 expert sets and compare a V100 replay over complete prefixes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_top1(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return 1
    if normalized in {"0", "false"}:
        return 0
    raise ValueError(f"invalid top-1 value {value!r}")


def export(args):
    count = args.positions
    import torch
    from run_oracle import build_oracle
    from run_precision_reference import read_ids

    ids = read_ids(args.oracle_root / "token_ids.json", count)
    manifest = json.loads((args.oracle_root / "manifest.json").read_text())
    for pos, token in enumerate(ids):
        entry = manifest["positions"][pos]
        if entry["position"] != pos or entry["token_id"] != token:
            raise ValueError(f"oracle token mismatch at {pos}")
    model, _head = build_oracle(str(args.model_dir), str(args.ple_dir))
    routed = {}
    handles = []
    for layer, block in enumerate(model.layers):
        def capture(_module, _inputs, output, layer=layer):
            selected = output[2].detach().cpu()
            if selected.numel() != count * 10 or selected.shape[-1] != 10:
                raise ValueError(f"router shape {tuple(selected.shape)} at layer {layer}")
            routed[layer] = selected.reshape(count, 10).clone()
        handles.append(block.mlp.gate.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(routed) != 48:
        raise RuntimeError(f"only {len(routed)} router layers captured")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # The V100 trace callback reaches the replay hook only for stages that have
    # a same-shaped oracle fixture.  Alpha is recomputed below from the V100's
    # own scores, so these values are deliberately inert placeholders: they
    # keep the callback on the replay path and are overwritten before MoE use.
    alpha_fixture = np.zeros(10, dtype="<f4")
    for pos in range(count):
        folder = args.out_dir / f"pos{pos:04d}"
        folder.mkdir()
        for layer in range(48):
            ids = routed[layer][pos].numpy().astype(np.int64)
            if (len(set(ids.tolist())) != 10 or ids.min() < 0 or ids.max() >= 512):
                raise ValueError(f"invalid expert set at position {pos}, layer {layer}")
            ids.astype("<f4").tofile(folder / f"L{layer:02d}_moe_router_ids.bin")
            alpha_fixture.tofile(folder / f"L{layer:02d}_moe_router_alpha.bin")
    print(f"exported {count * 48} full-forward FP32 expert sets", flush=True)


def report(args):
    count = args.positions
    from compare_full_prefix import kl, logp, read_bf16

    entries = json.loads((args.oracle_root / "manifest.json").read_text())["positions"]
    with args.three_way_csv.open(newline="") as stream:
        natural = list(csv.DictReader(stream))
    with args.cpu_replay_csv.open(newline="") as stream:
        cpu_replay = list(csv.DictReader(stream))
    if len(natural) < count or len(cpu_replay) != count:
        raise ValueError("missing complete-prefix comparison rows")
    rows = []
    for pos in range(count):
        if int(natural[pos]["position"]) != pos or int(cpu_replay[pos]["position"]) != pos:
            raise ValueError(f"comparison position mismatch at {pos}")
        entry = entries[pos]
        logits = [t for t in entry["tensors"] if t["name"] == "logits"]
        if len(logits) != 1:
            raise ValueError(f"oracle logits missing at {pos}")
        oracle = np.fromfile(args.oracle_root / logits[0]["file"], dtype="<f4")
        v100 = read_bf16(args.v100_root / f"pos{pos:04d}.bf16")
        if not np.isfinite(oracle).all():
            raise ValueError(f"nonfinite oracle at {pos}")
        if oracle.size != v100.size:
            raise ValueError(f"V100 replay logits shape mismatch at {pos}")
        replay_kl = kl(logp(oracle.astype(np.float64)), logp(v100))
        rows.append({
            "position": pos,
            "natural_cpu_kl": float(natural[pos]["oracle_cpu_kl"]),
            "replayed_cpu_kl": float(cpu_replay[pos]["replayed_cpu_kl"]),
            "natural_v100_kl": float(natural[pos]["oracle_v100_kl"]),
            "replayed_v100_kl": replay_kl,
            "natural_cpu_top1": parse_top1(natural[pos]["oracle_cpu_top1"]),
            "replayed_cpu_top1": parse_top1(cpu_replay[pos]["replayed_cpu_top1"]),
            "natural_v100_top1": parse_top1(natural[pos]["oracle_v100_top1"]),
            "replayed_v100_top1": int(np.argmax(oracle) == np.argmax(v100)),
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "per_position.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def summarize(lo, hi):
        part = rows[lo:hi]
        names = ("natural_cpu", "replayed_cpu", "natural_v100", "replayed_v100")
        result = {"positions": len(part)}
        for name in names:
            result[name + "_mean_kl"] = sum(r[name + "_kl"] for r in part) / len(part)
            result[name + "_top1"] = sum(r[name + "_top1"] for r in part)
        result["v100_replay_lower_kl_positions"] = sum(
            r["replayed_v100_kl"] < r["natural_v100_kl"] for r in part)
        return result
    summary = {
        "provenance": f"frozen full FP32 oracle, natural CPU and V100 4096-prefix report, and controlled full-FP32-membership CPU/V100 replays through {count} complete prefixes",
        "diagnostic_only": True,
        "natural_qualification": "Not established by forced membership; evaluate natural-routing runs separately",
        "ranges": {f"{lo}:{hi}": summarize(lo, hi)
                   for lo, hi in (((0, 512), (512, 768), (0, 768)) if count == 768 else
                                 ((0, 768), (512, 768), (768, 1024), (1024, 2048), (2048, 3072), (3072, 4096), (0, 4096)))},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exporter = sub.add_parser("export")
    exporter.add_argument("--model-dir", type=Path, required=True)
    exporter.add_argument("--ple-dir", type=Path, required=True)
    exporter.add_argument("--oracle-root", type=Path, required=True)
    exporter.add_argument("--out-dir", type=Path, required=True)
    reporter = sub.add_parser("report")
    reporter.add_argument("--oracle-root", type=Path, required=True)
    reporter.add_argument("--three-way-csv", type=Path, required=True)
    reporter.add_argument("--cpu-replay-csv", type=Path, required=True)
    reporter.add_argument("--v100-root", type=Path, required=True)
    reporter.add_argument("--out-dir", type=Path, required=True)
    for command in (exporter, reporter):
        command.add_argument("--positions", type=int, choices=(768, 4096), default=768)
    args = parser.parse_args()
    {"export": export, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
