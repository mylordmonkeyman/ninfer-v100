#!/usr/bin/env python3
"""Stream the 4,096-position CPU precision profile against the frozen FP32 oracle.

The frozen full-sequence oracle is the independent reference. The incremental
FP32 diagnostic is not rerun here: it failed its local parity guard at position
190 in the 256-position experiment. This script records every CPU-vs-oracle
difference without using that diagnostic as a gate or changing the Phase 11
acceptance thresholds.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from precision_metrics import compare_logits
from precision_profile import PROFILES
from run_oracle import build_oracle
from run_precision_reference import read_ids, run_decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--ple-dir", type=Path, required=True)
    p.add_argument("--ids-file", type=Path, required=True)
    p.add_argument("--oracle-root", type=Path, required=True)
    p.add_argument("--positions", type=int, default=4096)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.positions != 4096:
        p.error("full-corpus comparison requires exactly 4096 positions")
    ids = read_ids(args.ids_file, args.positions)
    manifest = json.loads((args.oracle_root / "manifest.json").read_text())
    entries = manifest["positions"]
    if len(entries) < args.positions:
        raise ValueError("frozen oracle has fewer than 4096 positions")
    oracle_paths = []
    for pos, token in enumerate(ids):
        row = entries[pos]
        if row["position"] != pos or row["token_id"] != token:
            raise ValueError(f"frozen oracle position/token mismatch at {pos}")
        logits = [t for t in row["tensors"] if t["name"] == "logits"]
        if len(logits) != 1:
            raise ValueError(f"frozen oracle lacks one logits tensor at {pos}")
        oracle_paths.append(args.oracle_root / logits[0]["file"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cpu_dir = args.out_dir / "cpu-logits"
    cpu_dir.mkdir(exist_ok=True)
    metric_path = args.out_dir / "cpu-vs-oracle.jsonl"
    model, head = build_oracle(str(args.model_dir), str(args.ple_dir))
    model.eval()
    for layer in model.layers:
        layer.mlp.experts.round_activations_to_bf16 = True

    with metric_path.open("w") as metric_file:
        def capture(pos, cpu):
            if not np.isfinite(cpu).all():
                raise ValueError(f"nonfinite CPU logits at {pos}")
            bits = cpu.view("<u4")
            if np.any(bits & 0xffff):
                raise ValueError(f"CPU logits lost BF16 storage at {pos}")
            compact = (bits >> 16).astype("<u2")
            compact.tofile(cpu_dir / f"pos{pos:04d}.bf16")
            oracle = np.fromfile(oracle_paths[pos], dtype="<f4")
            if oracle.size != cpu.size or not np.isfinite(oracle).all():
                raise ValueError(f"frozen oracle logits invalid at {pos}")
            row = compare_logits(oracle, cpu)
            row["position"] = pos
            metric_file.write(json.dumps(row) + "\n")
            metric_file.flush()
            if (pos + 1) % 64 == 0:
                print(f"cpu_complete_prefixes={pos+1}/4096", flush=True)
        run_decode(model, head, ids, PROFILES["v100-phase11-storage"],
                   None, fused_hyper_updates=True,
                   logits_sink=capture, retain_logits=False)
    rows = [json.loads(line) for line in metric_path.read_text().splitlines()]
    if len(rows) != 4096:
        raise RuntimeError(f"incomplete CPU profile: {len(rows)} positions")
    print(json.dumps({
        "positions": 4096,
        "oracle_vs_cpu_mean_kl": sum(r["kl"] for r in rows) / 4096,
        "oracle_vs_cpu_top1": sum(r["top1_agree"] for r in rows),
        "provenance": "frozen independent full-sequence FP32 oracle versus separate CPU precision profile",
        "incremental_fp32_parity": "not used as a stop condition; prior 256-position diagnostic failed at position 190",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
