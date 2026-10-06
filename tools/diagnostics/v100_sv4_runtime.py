#!/usr/bin/env python3
"""Qualify opt-in grouped CPU experts on a represented full-model V100 prefill."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading

if __package__:
    from .v100_sv0_benchmark import gpu_snapshot
    from .v100_sv3_calibrate import atomic_json, parse_probe, parse_json_records
else:
    from v100_sv0_benchmark import gpu_snapshot
    from v100_sv3_calibrate import atomic_json, parse_probe, parse_json_records


LAYERS = 48
ROUTES_PER_TOKEN = 10
EXPERT_BYTES = 2_764_808
MODES = ("single", "grouped")


def environment(mode: str, positions: int, telemetry: bool, logits: Path) -> dict:
    env = os.environ.copy()
    for key in tuple(env):
        if (key.startswith("NINFER_PHASE") or key.startswith("NINFER_V100_PREFILL_") or
                key.startswith("NINFER_V100_CPU_EXPERT_GROUP") or
                key.startswith("NINFER_FLASH_NEXT_EXPERT_CACHE")):
            env.pop(key)
    env.update({
        "NINFER_PHASE11_PREFILL_PROBE_POSITIONS": str(positions),
        "NINFER_V100_PREFILL_EXPERT_POLICY": "cpu-cache",
        "NINFER_V100_DEVICE_ROUTE_COMBINE": "1",
        "NINFER_V100_CPU_EXPERT_GROUP": "1" if mode == "grouped" else "0",
        "NINFER_V100_TELEMETRY": "1" if telemetry else "0",
        "NINFER_FLASH_NEXT_EXPERT_CACHE": "0",
        "NINFER_FLASH_NEXT_STAGE_LEDGER": "0",
        "NINFER_FLASH_NEXT_FP32_MOE_ROUTED_INPUT": "0",
        "NINFER_FLASH_NEXT_CPU_EXPERT_FP32_INTERMEDIATE": "0",
        "NINFER_V100_SV4_LOGITS": str(logits),
    })
    return env


def validate_layers(stderr: str, positions: int, grouped: bool) -> dict:
    rows = [row for row in parse_json_records(stderr)
            if row.get("kind") == "expert_layer" and row.get("prefill")]
    if len(rows) != LAYERS or sorted(row["layer"] for row in rows) != list(range(LAYERS)):
        raise ValueError("SV4 diagnostic is missing full-model expert layers")
    routes = positions * ROUTES_PER_TOKEN
    if any(row["tokens"] != positions or row["routes"] != routes or
           row["cpu_miss_routes"] != routes or row["gpu_hit_routes"] != 0
           for row in rows):
        raise ValueError("SV4 diagnostic route accounting mismatch")
    if any(row["stream_routes"] or row["cache_result_d2h_bytes"] or
           row["routed_sum_h2d_bytes"] for row in rows):
        raise ValueError("SV4 diagnostic used a non-CPU expert route")
    if grouped:
        if any(row["cpu_groups"] < 1 or row["cpu_grouped_pairs"] < 2 or
               row["cpu_grouped_pairs"] > routes for row in rows):
            raise ValueError("SV4 grouped diagnostic did not group repeated experts")
        if any(not 0 < row["cpu_weight_read_bytes"] < routes * EXPERT_BYTES
               for row in rows):
            raise ValueError("SV4 grouped effective weight reads were not reduced")
    elif any(row["cpu_groups"] or row["cpu_grouped_pairs"] or
             row["cpu_weight_read_bytes"] != routes * EXPERT_BYTES for row in rows):
        raise ValueError("SV4 single-token control has invalid grouping accounting")
    wall_us = sum(row["cpu_branch_us"] for row in rows)
    weight_bytes = sum(row["cpu_weight_read_bytes"] for row in rows)
    return {
        "layers": len(rows),
        "routes": sum(row["routes"] for row in rows),
        "groups": sum(row["cpu_groups"] for row in rows),
        "grouped_pairs": sum(row["cpu_grouped_pairs"] for row in rows),
        "effective_weight_read_bytes": weight_bytes,
        "cpu_branch_wall_us": wall_us,
        "effective_weight_read_gbps": weight_bytes / wall_us / 1.0e3,
        "provenance": [(row["layer"], tuple(row["frequency"])) for row in rows],
    }


def invoke(executable: Path, output: Path, name: str, mode: str,
           positions: int, telemetry: bool) -> dict:
    logits = output / f"{name}-logits"
    logits.mkdir(parents=True, exist_ok=True)
    env = environment(mode, positions, telemetry, logits)
    snapshots, errors, done = [gpu_snapshot()], [], threading.Event()

    def monitor():
        while not done.wait(5):
            try:
                snapshots.append(gpu_snapshot())
            except Exception as error:  # preserve monitoring failure
                errors.append(str(error))

    worker = threading.Thread(target=monitor)
    worker.start()
    try:
        process = subprocess.run([str(executable.resolve())], env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=3600)
    finally:
        done.set()
        worker.join()
    snapshots.append(gpu_snapshot())
    (output / f"{name}.stdout.log").write_text(process.stdout)
    (output / f"{name}.stderr.log").write_text(process.stderr)
    atomic_json(output / f"{name}-hardware.json", {
        "snapshots": snapshots, "errors": errors,
        "environment": {key: value for key, value in env.items()
                        if key.startswith("NINFER_")},
    })
    if process.returncode or errors:
        raise RuntimeError(f"{name}: model process or hardware monitor failed")
    if any(not row["thermal_status_observed"] or row["thermal_throttled"]
           for row in snapshots):
        raise ValueError(f"{name}: thermal status unavailable or throttling observed")
    result = {
        "name": name, "mode": mode, "probe": parse_probe(process.stdout, positions),
        "logits_sha256": hashlib.sha256(
            (logits / "pass0.bf16").read_bytes()).hexdigest(),
    }
    if telemetry:
        result["diagnostic"] = validate_layers(
            process.stderr, positions, mode == "grouped")
    return result


def summarize(observations: list[dict], diagnostics: dict, repeats: int) -> dict:
    cells = {}
    for mode in MODES:
        selected = [row for row in observations if row["mode"] == mode]
        if len(selected) != repeats:
            raise ValueError("missing SV4 timing observations")
        if len({row["logits_sha256"] for row in selected}) != 1:
            raise ValueError(f"{mode}: repeated BF16 output changed")
        numerical = [tuple(row["probe"][key] for key in
                     ("candidate_top1", "oracle_top1", "kl",
                      "relative_nll_delta", "max_logit_error")) for row in selected]
        if len(set(numerical)) != 1:
            raise ValueError(f"{mode}: repeated numerical metrics changed")
        rates = [row["probe"]["tokens_per_s"] for row in selected]
        cells[mode] = {"median": statistics.median(rates),
                       "minimum": min(rates), "maximum": max(rates),
                       "expert_pairs_per_second": statistics.median(
                           row["probe"]["expert_pairs"] / row["probe"]["elapsed_s"]
                           for row in selected),
                       "numerical": numerical[0],
                       "logits_sha256": selected[0]["logits_sha256"]}
    if cells["single"]["logits_sha256"] != cells["grouped"]["logits_sha256"]:
        raise ValueError("grouped CPU experts changed final BF16 logits")
    if diagnostics["single"]["diagnostic"]["provenance"] != \
            diagnostics["grouped"]["diagnostic"]["provenance"]:
        raise ValueError("grouped CPU experts changed route provenance")
    for mode in MODES:
        cells[mode].pop("logits_sha256")
    return {
        "schema": 1, "milestone": "SV4", "qualified": False,
        "scope": "cache_off_represented_full_model_prefill",
        "repeats": repeats, "single": cells["single"], "grouped": cells["grouped"],
        "median_change_percent": 100 * (cells["grouped"]["median"] /
                                          cells["single"]["median"] - 1),
        "diagnostics": {mode: {key: value for key, value in row["diagnostic"].items()
                                if key != "provenance"}
                        for mode, row in diagnostics.items()},
        "default_enabled": False,
        "limitations": [
            "standalone represented prefill probe, not production HTTP serving or MTP",
            "persistent expert cache disabled to exercise all CPU misses",
            "grouping remains opt-in pending cached, decode-batch, MTP, and production gates",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positions", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.positions < 2 or args.positions > 2048 or args.repeats < 3:
        parser.error("positions must be in [2, 2048] and repeats at least 3")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = {mode: invoke(args.executable, output, f"diagnostic-{mode}",
                                mode, args.positions, True) for mode in MODES}
    observations = []
    for repeat in range(args.repeats):
        order = MODES if repeat % 2 == 0 else tuple(reversed(MODES))
        for mode in order:
            observations.append(invoke(args.executable, output,
                                       f"timing-{repeat}-{mode}", mode,
                                       args.positions, False))
    report = summarize(observations, diagnostics, args.repeats)
    report["candidate_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    atomic_json(output / "sv4-runtime.json", report)
    lines = [
        "SV4 grouped CPU expert screen; opt-in only.", "",
        "| Mode | Prefill t/s median (range) | Expert pairs/s | Effective weight reads | CPU branch wall |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        cell, diag = report[mode], report["diagnostics"][mode]
        lines.append(f"| {mode} | {cell['median']:.3f} ({cell['minimum']:.3f}–{cell['maximum']:.3f}) | "
                     f"{cell['expert_pairs_per_second']:.1f} | "
                     f"{diag['effective_weight_read_bytes']} B | {diag['cpu_branch_wall_us']:.1f} us |")
    lines += ["", f"Grouped median change: {report['median_change_percent']:+.2f}%.",
              "Grouped and single final BF16 logits and route provenance are exact.",
              "No default change or production claim."]
    (output / "sv4-runtime.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
