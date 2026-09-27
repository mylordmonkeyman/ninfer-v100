#!/usr/bin/env python3
"""Compare frozen FP32 oracle, CPU precision profile, and natural V100 logits."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def logp(v):
    m = float(np.max(v))
    return v - m - math.log(float(np.sum(np.exp(v - m), dtype=np.float64)))


def kl(log_ref, log_candidate):
    return float(np.sum(np.exp(log_ref) * (log_ref - log_candidate),
                        dtype=np.float64))


def read_bf16(path):
    words = np.fromfile(path, dtype="<u2")
    if words.size == 0:
        raise ValueError(f"empty logits: {path}")
    result = (words.astype("<u4") << 16).view("<f4")
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite logits: {path}")
    return result.astype(np.float64)


def nearest_p99(values):
    ordered = sorted(values)
    return ordered[math.ceil(.99 * len(ordered)) - 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle-root", type=Path, required=True)
    p.add_argument("--cpu-root", type=Path, required=True)
    p.add_argument("--v100-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    entries = json.loads((args.oracle_root / "manifest.json").read_text())["positions"]
    if len(entries) < 4096:
        raise ValueError("frozen oracle has fewer than 4096 positions")
    cpu_rows = [json.loads(x) for x in
                (args.cpu_root / "cpu-vs-oracle.jsonl").read_text().splitlines()]
    if len(cpu_rows) != 4096:
        raise ValueError(f"CPU reference completed only {len(cpu_rows)} positions")
    if len(list((args.v100_root / "v100-logits").glob("*.bf16"))) != 4096:
        raise ValueError("natural V100 did not export all 4096 logits")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sums = [0., 0., 0.]
    nll_oracle = nll_cpu = nll_v100 = 0.
    for pos in range(4096):
        entry = entries[pos]
        if entry["position"] != pos or cpu_rows[pos]["position"] != pos:
            raise ValueError(f"position mismatch at {pos}")
        logits = [x for x in entry["tensors"] if x["name"] == "logits"]
        if len(logits) != 1:
            raise ValueError(f"frozen oracle lacks logits at {pos}")
        o = np.fromfile(args.oracle_root / logits[0]["file"], dtype="<f4")
        if not o.size or not np.isfinite(o).all():
            raise ValueError(f"invalid frozen oracle logits at {pos}")
        c = read_bf16(args.cpu_root / "cpu-logits" / f"pos{pos:04d}.bf16")
        v = read_bf16(args.v100_root / "v100-logits" / f"pos{pos:04d}.bf16")
        if o.shape != c.shape or o.shape != v.shape:
            raise ValueError(f"logit shape mismatch at {pos}")
        ol, cl, vl = map(logp, (o.astype(np.float64), c, v))
        koc, kov, kcv = kl(ol, cl), kl(ol, vl), kl(cl, vl)
        if not all(math.isfinite(x) for x in (koc, kov, kcv)):
            raise ValueError(f"nonfinite KL at {pos}")
        if abs(koc - cpu_rows[pos]["kl"]) > 1e-5:
            raise ValueError(f"CPU capture/recalculation mismatch at {pos}")
        target = entries[pos + 1]["token_id"] if pos + 1 < 4096 else None
        if target is not None:
            if not 0 <= target < o.size:
                raise ValueError(f"invalid next target at {pos}")
            nll_oracle -= ol[target]
            nll_cpu -= cl[target]
            nll_v100 -= vl[target]
        row = {
            "position": pos, "oracle_cpu_kl": koc,
            "oracle_v100_kl": kov, "cpu_v100_kl": kcv,
            "oracle_cpu_top1": int(np.argmax(o) == np.argmax(c)),
            "oracle_v100_top1": int(np.argmax(o) == np.argmax(v)),
            "v100_lower_kl": int(kov < koc),
        }
        rows.append(row)
        for i, value in enumerate((koc, kov, kcv)):
            sums[i] += value
        if (pos + 1) % 256 == 0:
            print(f"compared_complete_prefixes={pos+1}/4096", flush=True)
    with (args.out_dir / "per_position.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def summarise(chunk):
        return {
            "positions": len(chunk),
            "oracle_cpu_mean_kl": sum(x["oracle_cpu_kl"] for x in chunk) / len(chunk),
            "oracle_v100_mean_kl": sum(x["oracle_v100_kl"] for x in chunk) / len(chunk),
            "cpu_v100_mean_kl": sum(x["cpu_v100_kl"] for x in chunk) / len(chunk),
            "oracle_cpu_top1": sum(x["oracle_cpu_top1"] for x in chunk),
            "oracle_v100_top1": sum(x["oracle_v100_top1"] for x in chunk),
            "v100_lower_kl_positions": sum(x["v100_lower_kl"] for x in chunk),
        }
    report = {
        "provenance": {
            "fp32": "frozen independent full-sequence oracle",
            "cpu": "separate quantized-weight, BF16-storage, fused-hyper precision-profile reference",
            "v100": "natural V100 implementation",
            "prefix": "all paths carry their own complete sequential history",
            "incremental_fp32_guard": "not used to stop this full-corpus diagnostic; prior 256-position mismatch remains recorded",
        },
        "all": summarise(rows),
        "ranges": {f"{lo}:{hi}": summarise(rows[lo:hi]) for lo, hi in
                   ((0,128),(128,512),(512,1024),(1024,2048),
                    (2048,3072),(3072,4096))},
        "cpu_p99_kl_nearest_rank": nearest_p99([x["oracle_cpu_kl"] for x in rows]),
        "v100_p99_kl_nearest_rank": nearest_p99([x["oracle_v100_kl"] for x in rows]),
        "cpu_max_kl": max(x["oracle_cpu_kl"] for x in rows),
        "v100_max_kl": max(x["oracle_v100_kl"] for x in rows),
        "oracle_mean_target_nll": nll_oracle / 4095,
        "cpu_mean_target_nll": nll_cpu / 4095,
        "v100_mean_target_nll": nll_v100 / 4095,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
