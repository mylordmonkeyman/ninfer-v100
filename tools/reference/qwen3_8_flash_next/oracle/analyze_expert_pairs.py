#!/usr/bin/env python3
"""Decompose the layer-zero MoE boundary between the CPU storage profile, the
natural V100 candidate trace, and the matched-input V100 trace.

For every traced position the tool compares the stored MoE boundary values
(MLP block input, router IDs/alpha, per-expert pair outputs, routed sum, and
shared-expert activation) between the CPU profile and the V100 candidate.
At position zero it additionally verifies that the V100's own pair outputs
and alpha weights reconstruct its stored routed sum with the selected-path
FP32 FMA accumulation, and quantifies the counterfactuals that separate
expert-compute differences from alpha and upstream-input differences.

The matched-input trace re-runs the V100 decode with the CPU's stored
L00_mlp_block_input injected at every position, so its position-zero pair
outputs are computed on exactly the stored input the CPU used: the
matched pair NRMSE isolates expert computation from upstream drift.
"""

import argparse
import ctypes
import json
import math
from pathlib import Path

import numpy as np

kHidden = 2560
kPairSlabs = 10
kPairCount = kPairSlabs * kHidden
kShared = 640
kExperts = 10


def read(root, position, name, count):
    path = root / f"pos{position:04d}" / f"{name}.bin"
    data = path.read_bytes()
    if len(data) != count * 4:
        raise ValueError(f"{path}: expected {count * 4} bytes, got {len(data)}")
    values = np.frombuffer(data, dtype="<f4").astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: nonfinite stage value")
    return values


def nrmse(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: {reference.shape} vs {candidate.shape}")
    scale = math.sqrt(float(np.dot(reference, reference)))
    if scale == 0.0:
        scale = 1.0
    return float(math.sqrt(float(np.mean((reference - candidate) ** 2)))
                 / scale)


def metrics(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: {reference.shape} vs {candidate.shape}")
    delta = reference - candidate
    return {
        "nrmse": nrmse(reference, candidate),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "different": int(np.count_nonzero(reference != candidate)),
    }


def fmaf_sum(pairs, alpha):
    """Selected-path FP32 FMA accumulation in path order (the V100 host-expert
    routed-sum contract)."""
    fmaf = ctypes.CDLL("libm.so.6").fmaf
    fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fmaf.restype = ctypes.c_float
    acc = [0.0] * kHidden
    for path in range(kPairSlabs):
        base = path * kHidden
        alpha_f = float(alpha[path])
        for row in range(kHidden):
            acc[row] = fmaf(alpha_f, float(pairs[base + row]), acc[row])
    return acc


def position_boundary(cpu_root, v100_root, position):
    row = {"position": position}
    cpu_ids = read(cpu_root, position, "L00_moe_router_ids", kExperts)
    v100_ids = read(v100_root, position, "L00_moe_router_ids", kExperts)
    row["cpu_expert_ids"] = [int(v) for v in cpu_ids]
    row["v100_expert_ids"] = [int(v) for v in v100_ids]
    row["same_expert_set"] = bool(set(cpu_ids.tolist()) == set(v100_ids.tolist()))
    for name, count in (("L00_mlp_block_input", kHidden),
                        ("L00_moe_router_alpha", kExperts),
                        ("L00_moe_shared_activation", kShared),
                        ("L00_moe_routed_sum", kHidden)):
        row[name] = metrics(read(cpu_root, position, name, count),
                            read(v100_root, position, name, count))
    row["L00_moe_pair_outputs"] = metrics(
        read(cpu_root, position, "L00_moe_pair_outputs", kPairCount),
        read(v100_root, position, "L00_moe_pair_outputs", kPairCount))
    return row


def deep_position(cpu_root, v100_root, position, label):
    """Position-zero style decomposition: per-path pair metrics, alpha, and
    the FMA reconstruction invariant plus counterfactuals."""
    row = {"position": position, "label": label}
    ids = read(cpu_root, position, "L00_moe_router_ids", kExperts)
    ids_v100 = read(v100_root, position, "L00_moe_router_ids", kExperts)
    row["expert_ids_aligned"] = bool((ids == ids_v100).all())
    cpu_pairs = read(cpu_root, position, "L00_moe_pair_outputs", kPairCount)
    v100_pairs = read(v100_root, position, "L00_moe_pair_outputs", kPairCount)
    row["per_path"] = []
    for path in range(kPairSlabs):
        base = path * kHidden
        path_row = {
            "path": path,
            "cpu_expert": int(ids[path]),
            "v100_expert": int(ids_v100[path]),
        }
        if ids[path] == ids_v100[path]:
            path_row["cpu_vs_v100"] = metrics(
                cpu_pairs[base:base + kHidden],
                v100_pairs[base:base + kHidden])
        else:
            # Different experts selected: a per-path pair comparison would
            # measure the wrong thing; record the set disagreement instead.
            path_row["cpu_vs_v100"] = {"expert_set_mismatch": True}
        row["per_path"].append(path_row)
    row["input"] = metrics(
        read(cpu_root, position, "L00_mlp_block_input", kHidden),
        read(v100_root, position, "L00_mlp_block_input", kHidden))
    row["alpha"] = metrics(
        read(cpu_root, position, "L00_moe_router_alpha", kExperts),
        read(v100_root, position, "L00_moe_router_alpha", kExperts))

    cpu_sum = read(cpu_root, position, "L00_moe_routed_sum", kHidden)
    v100_sum = read(v100_root, position, "L00_moe_routed_sum", kHidden)
    cpu_alpha = read(cpu_root, position, "L00_moe_router_alpha", kExperts)
    v100_alpha = read(v100_root, position, "L00_moe_router_alpha", kExperts)

    # Invariant: the V100 pair trace must reconstruct its stored routed sum
    # under the selected-path FP32 FMA contract. A mismatch means the pair
    # trace and the routed sum came from different executions.
    natural = np.array(fmaf_sum(v100_pairs, v100_alpha), dtype=np.float64)
    row["v100_reconstructed_vs_stored"] = metrics(v100_sum, natural)
    if not (natural == v100_sum).all():
        raise AssertionError(
            f"{label}: V100 pair trace does not reconstruct its stored "
            f"routed sum (max abs "
            f"{float(np.max(np.abs(natural - v100_sum))):.3e})")
    # Counterfactuals separate the residual into alpha, expert-compute, and
    # upstream-input components.
    row["cpu_pairs_with_v100_alpha_vs_v100_sum"] = metrics(
        v100_sum, np.array(fmaf_sum(cpu_pairs, v100_alpha), dtype=np.float64))
    row["cpu_pairs_with_cpu_alpha_vs_v100_sum"] = metrics(
        v100_sum, np.array(fmaf_sum(cpu_pairs, cpu_alpha), dtype=np.float64))
    row["cpu_stored_sum_vs_v100_sum"] = metrics(v100_sum, cpu_sum)
    cpu_own = np.array(fmaf_sum(cpu_pairs, cpu_alpha), dtype=np.float64)
    row["cpu_stored_sum_vs_cpu_path_fma"] = metrics(cpu_sum, cpu_own)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-root", type=Path, required=True,
                        help="CPU profile stage tree (v100-phase11-storage)")
    parser.add_argument("--v100-root", type=Path, required=True,
                        help="natural V100 candidate stage tree")
    parser.add_argument("--matched-root", type=Path, default=None,
                        help="matched-input V100 stage tree (optional)")
    parser.add_argument("--positions", type=int, default=14)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = {"positions": args.positions,
              "per_position": [],
              "position_zero_natural": None,
              "position_zero_matched": None}
    lines = []

    for position in range(args.positions):
        row = position_boundary(args.cpu_root, args.v100_root, position)
        report["per_position"].append(row)
        lines.append(
            f"pos{position:04d} experts_match={int(row['same_expert_set'])} "
            f"ids_cpu={row['cpu_expert_ids']} ids_v100={row['v100_expert_ids']}")
        for name in ("L00_mlp_block_input", "L00_moe_router_alpha",
                     "L00_moe_shared_activation", "L00_moe_routed_sum",
                     "L00_moe_pair_outputs"):
            metric = row[name]
            lines.append(
                f"  {name} cpu_vs_v100 "
                f"nrmse={metric['nrmse']:.6g} max_abs={metric['max_abs']:.6g} "
                f"different={metric['different']}")

    report["position_zero_natural"] = deep_position(
        args.cpu_root, args.v100_root, 0, "natural")
    natural = report["position_zero_natural"]
    lines.append("position-zero natural decomposition")
    lines.append(f"  input cpu_vs_v100 nrmse={natural['input']['nrmse']:.6g} "
                 f"(upstream drift feeding the MoE)")
    lines.append(f"  alpha cpu_vs_v100 nrmse={natural['alpha']['nrmse']:.6g}")
    lines.append("  per-path pair metrics (cpu vs v100, same expert only):")
    for path in natural["per_path"]:
        metric = path["cpu_vs_v100"]
        if metric.get("expert_set_mismatch"):
            lines.append(
                f"    path {path['path']} cpu_expert={path['cpu_expert']} "
                f"v100_expert={path['v100_expert']} (different experts)")
        else:
            lines.append(
                f"    path {path['path']} expert={path['cpu_expert']} "
                f"nrmse={metric['nrmse']:.6g} max_abs={metric['max_abs']:.6g}")
    lines.append(f"  v100 reconstructed vs stored routed sum: "
                 f"{json.dumps(natural['v100_reconstructed_vs_stored'])}")
    lines.append("  counterfactuals vs the V100 stored routed sum:")
    for key in ("cpu_pairs_with_v100_alpha_vs_v100_sum",
                "cpu_pairs_with_cpu_alpha_vs_v100_sum",
                "cpu_stored_sum_vs_v100_sum",
                "cpu_stored_sum_vs_cpu_path_fma"):
        lines.append(f"    {key}: {json.dumps(natural[key])}")

    if args.matched_root is not None:
        report["position_zero_matched"] = deep_position(
            args.cpu_root, args.matched_root, 0, "matched")
        matched = report["position_zero_matched"]
        lines.append("position-zero matched-input decomposition "
                     "(V100 expert computed on the CPU stored input)")
        lines.append(f"  input cpu_vs_v100 nrmse={matched['input']['nrmse']:.6g} "
                     "(natural drift; the injection happens after this dump)")
        lines.append(f"  alpha cpu_vs_v100 nrmse={matched['alpha']['nrmse']:.6g}")
        lines.append("  per-path pair metrics on matched input:")
        for path in matched["per_path"]:
            metric = path["cpu_vs_v100"]
            if metric.get("expert_set_mismatch"):
                lines.append(
                    f"    path {path['path']} cpu_expert={path['cpu_expert']} "
                    f"v100_expert={path['v100_expert']} (different experts)")
            else:
                lines.append(
                    f"    path {path['path']} expert={path['cpu_expert']} "
                    f"nrmse={metric['nrmse']:.6g} max_abs={metric['max_abs']:.6g}")
        lines.append(f"  v100 reconstructed vs stored routed sum: "
                     f"{json.dumps(matched['v100_reconstructed_vs_stored'])}")

    output = "\n".join(lines) + "\n"
    if args.out is not None:
        args.out.write_text(output)
        args.out.with_suffix(args.out.suffix + ".json").write_text(
            json.dumps(report, indent=2))
    print(output, end="")


if __name__ == "__main__":
    main()
