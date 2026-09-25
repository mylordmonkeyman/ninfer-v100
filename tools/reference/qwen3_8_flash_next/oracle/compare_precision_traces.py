#!/usr/bin/env python3
"""Compare FP32 oracle, CPU storage profile, and optional V100 stage traces."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def manifest_entries(root):
    manifest = json.loads((root / "manifest.json").read_text())
    return {(p["position"], item["name"]): root / item["file"]
            for p in manifest["positions"] for item in p["tensors"]}


def candidate_entries(root):
    lines = (root / "stages.jsonl").read_text().splitlines()
    entries = {}
    for line in lines:
        record = json.loads(line)
        key = (record["position"], record["name"])
        if key in entries:
            raise ValueError(f"duplicate V100 candidate stage {key}")
        entries[key] = root / record["file"]
    return entries


def load(path):
    if path.stat().st_size % 4:
        raise ValueError(f"invalid FP32 stage byte count: {path}")
    value = np.fromfile(path, dtype="<f4")
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"empty or nonfinite stage: {path}")
    return value.astype(np.float64)


def nrmse(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(f"stage size mismatch: {reference.size} vs {candidate.size}")
    scale = max(float(np.sqrt(np.mean(reference * reference))), 1e-12)
    return float(np.sqrt(np.mean((reference - candidate) ** 2)) / scale)


def compare(oracle_root, cpu_root, out_dir, v100_root=None, incremental_fp32_root=None):
    oracle = manifest_entries(oracle_root)
    cpu = manifest_entries(cpu_root)
    incremental = (manifest_entries(incremental_fp32_root)
                   if incremental_fp32_root is not None else {})
    v100 = candidate_entries(v100_root) if v100_root is not None else {}
    if v100_root is not None and not v100:
        raise ValueError("V100 candidate trace is empty")
    rows = []
    for (position, name), cpu_path in sorted(cpu.items()):
        key = (position, name)
        reference_source = "independent-fp32"
        if key in oracle:
            reference_path = oracle[key]
        elif name == "L01_mlp_block_input_fp32" and \
                (position, "L01_mlp_block_input") in oracle:
            # Both are the same mathematical mixer output. The diagnostic
            # suffix distinguishes its FP32 value before candidate BF16
            # storage from the separately compared BF16 block input.
            reference_path = oracle[position, "L01_mlp_block_input"]
            reference_source = "independent-fp32-unrounded-mixer"
        elif name.endswith("moe_router_scores") and key in incremental:
            # The frozen stage oracle contains selected router IDs but no
            # 512-score vector. Incremental FP32 has already passed its
            # independent-oracle logits gate; label this auxiliary reference.
            reference_path = incremental[key]
            reference_source = "incremental-fp32-scores"
        elif name in ("L00_hyper_before_attn", "L00_attn_injection") and key in incremental:
            # These optional internal boundaries have no frozen full-sequence
            # oracle tensors. Their incremental FP32 reference has passed the
            # independent oracle logits parity check; label the provenance.
            reference_path = incremental[key]
            reference_source = "incremental-fp32-hyper-boundary"
        else:
            raise ValueError(f"FP32 oracle is missing CPU stage {key}")
        expected = load(reference_path)
        profiled = load(cpu_path)
        row = {"position": position, "stage": name,
               "reference_source": reference_source,
               "cpu_vs_oracle_nrmse": "", "v100_vs_oracle_nrmse": "",
               "v100_vs_cpu_nrmse": "", "cpu_same_expert_set": "",
               "v100_same_expert_set": ""}
        if name.endswith("moe_router_ids"):
            if expected.shape != profiled.shape:
                raise ValueError(f"router shape mismatch at {key}")
            row["cpu_same_expert_set"] = int(set(expected) == set(profiled))
            if key in v100:
                value = load(v100[key])
                row["v100_same_expert_set"] = int(set(expected) == set(value))
        else:
            row["cpu_vs_oracle_nrmse"] = nrmse(expected, profiled)
            if key in v100:
                value = load(v100[key])
                row["v100_vs_oracle_nrmse"] = nrmse(expected, value)
                row["v100_vs_cpu_nrmse"] = nrmse(profiled, value)
        rows.append(row)
    if v100_root is not None:
        missing = set(cpu) - set(v100)
        if missing:
            raise ValueError(f"V100 candidate trace lacks {len(missing)} CPU stages; "
                             f"first: {sorted(missing)[0]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stage_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"positions": len({row["position"] for row in rows}),
              "stage_rows": len(rows), "has_v100": v100_root is not None,
              "incremental_router_score_rows": sum(
                  row["reference_source"] == "incremental-fp32-scores"
                  for row in rows),
              "worst_cpu_stage": max((row for row in rows
                                      if isinstance(row["cpu_vs_oracle_nrmse"], float)),
                                     key=lambda row: row["cpu_vs_oracle_nrmse"]),
              "router_set_flips_cpu": sum(row["cpu_same_expert_set"] == 0
                                          for row in rows if row["cpu_same_expert_set"] != ""),
              "router_set_flips_v100": sum(row["v100_same_expert_set"] == 0
                                           for row in rows if row["v100_same_expert_set"] != "")}
    (out_dir / "stage_comparison.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--v100", type=Path)
    parser.add_argument("--incremental-fp32", type=Path,
                        help="FP32 decode scores when frozen oracle lacks router scores")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.oracle, args.cpu, args.out_dir, args.v100,
                             args.incremental_fp32), indent=2))


if __name__ == "__main__":
    main()
