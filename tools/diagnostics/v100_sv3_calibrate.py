#!/usr/bin/env python3
"""Calibrate Flash-Next CPU versus staged-GPU expert prefill on V100."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading

if __package__:
    from .v100_sv0_benchmark import gpu_snapshot
else:
    from v100_sv0_benchmark import gpu_snapshot


MODES = ("cpu-cache", "stream")
DEFAULT_SIZES = (128, 256, 512, 1024, 2048)
EXPERT_SLOT_BYTES = 2_765_056
EXPERT_HIDDEN = 2560
ROUTES_PER_TOKEN = 10
LAYERS = 48


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_probe(stdout: str, positions: int) -> dict:
    prefix = "phase11.prefill_probe."
    lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError(f"expected one prefill probe result, found {len(lines)}")
    fields = {}
    for item in lines[0].split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key.removeprefix(prefix)] = value
    integer_keys = ("positions", "final_position", "candidate_top1", "oracle_top1", "expert_pairs")
    float_keys = ("kl", "relative_nll_delta", "max_logit_error", "elapsed_s")
    if any(key not in fields for key in integer_keys + float_keys):
        raise ValueError("incomplete prefill probe result")
    result = {key: int(fields[key]) for key in integer_keys}
    result.update({key: float(fields[key]) for key in float_keys})
    if (result["positions"] != positions or result["final_position"] != positions - 1 or
            result["expert_pairs"] != positions * ROUTES_PER_TOKEN * LAYERS):
        raise ValueError("prefill probe shape or expert count mismatch")
    if result["elapsed_s"] <= 0 or any(not math.isfinite(result[key]) for key in float_keys):
        raise ValueError("invalid prefill timing or numerical evidence")
    result["tokens_per_s"] = positions / result["elapsed_s"]
    return result


def parse_json_records(stderr: str) -> list[dict]:
    records = []
    for line in stderr.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def validate_diagnostic(stderr: str, positions: int, mode: str) -> dict:
    records = parse_json_records(stderr)
    layers = [row for row in records if row.get("kind") == "expert_layer" and row.get("prefill")]
    memory = [row for row in records if row.get("kind") == "runtime_memory"]
    if len(layers) != LAYERS or len(memory) != 1:
        raise ValueError("diagnostic is missing expert layers or runtime memory")
    if sorted(row["layer"] for row in layers) != list(range(LAYERS)):
        raise ValueError("diagnostic expert layers are incomplete or duplicated")
    routes = positions * ROUTES_PER_TOKEN
    if any(row["tokens"] != positions or row["routes"] != routes for row in layers):
        raise ValueError("diagnostic route shape mismatch")
    if any(row["gpu_hit_routes"] != 0 for row in layers):
        raise ValueError("calibration unexpectedly used the persistent cache")
    if any(row["cache_result_d2h_bytes"] or row["routed_sum_h2d_bytes"] for row in layers):
        raise ValueError("calibration did not retain SV2 transfer elimination")

    if mode == "stream":
        if any(row["stream_routes"] != routes or row["cpu_miss_routes"] != 0 for row in layers):
            raise ValueError("stream mode did not stage every route")
        if any(row["stream_experts"] < 1 or
               row["stream_expert_h2d_bytes"] != row["stream_experts"] * EXPERT_SLOT_BYTES
               for row in layers):
            raise ValueError("stream expert transfer accounting mismatch")
        if any(row["route_input_d2h_bytes"] != routes * 4 or row["cpu_miss_h2d_bytes"] != 0
               for row in layers):
            raise ValueError("stream mode copied route activations or CPU results")
    elif mode == "cpu-cache":
        if any(row["stream_routes"] != 0 or row["stream_experts"] != 0 or
               row["stream_expert_h2d_bytes"] != 0 for row in layers):
            raise ValueError("CPU control unexpectedly used the staging ring")
        if any(row["cpu_miss_routes"] != routes for row in layers):
            raise ValueError("CPU control did not execute every route on the host")
        expected_inputs = positions * EXPERT_HIDDEN * 2 + routes * 4
        expected_outputs = routes * EXPERT_HIDDEN * 4
        if any(row["route_input_d2h_bytes"] != expected_inputs or
               row["cpu_miss_h2d_bytes"] != expected_outputs for row in layers):
            raise ValueError("CPU control transfer accounting mismatch")
    else:
        raise ValueError(f"unknown prefill policy: {mode}")

    return {
        "layers": len(layers),
        "routes": sum(row["routes"] for row in layers),
        "distinct_experts": sum(row["distinct_experts"] for row in layers),
        "stream_expert_h2d_bytes": sum(row["stream_expert_h2d_bytes"] for row in layers),
        "route_input_d2h_bytes": sum(row["route_input_d2h_bytes"] for row in layers),
        "cpu_miss_h2d_bytes": sum(row["cpu_miss_h2d_bytes"] for row in layers),
        "runtime_memory": memory[0],
    }


def equal_capacity(cpu: dict, stream: dict) -> dict:
    # The staging ring is a separately planned optimization cost. The logical KV,
    # state, and shared workspace capacities must remain identical.
    keys = (
        "attention_kv_bytes", "indexer_kv_bytes", "block_tables_bytes",
        "persistent_allocation_bytes",
        "recurrent_state_bytes", "mtp_persistent_state_bytes",
        "round_tensors_bytes", "mtp_round_tensors_bytes",
        "workspace_allocation_bytes", "sampling_workspace_allocation_bytes",
        "graph_allowance_bytes",
    )
    for key in keys:
        if cpu[key] != stream[key]:
            raise ValueError(f"CPU/stream runtime capacity differs: {key}")
    overhead = stream["runtime_reserved_bytes"] - cpu["runtime_reserved_bytes"]
    if overhead <= 0:
        raise ValueError("stream runtime did not report its separately planned ring cost")
    return {"equal_fields": list(keys), "stream_runtime_overhead_bytes": overhead}


def environment(mode: str, positions: int, telemetry: bool) -> dict:
    env = os.environ.copy()
    for key in tuple(env):
        if (key.startswith("NINFER_PHASE") or key.startswith("NINFER_V100_PREFILL_") or
                key.startswith("NINFER_FLASH_NEXT_EXPERT_CACHE")):
            env.pop(key)
    env.update({
        "NINFER_PHASE11_PREFILL_PROBE_POSITIONS": str(positions),
        "NINFER_V100_PREFILL_EXPERT_POLICY": mode,
        "NINFER_V100_DEVICE_ROUTE_COMBINE": "1",
        "NINFER_V100_TELEMETRY": "1" if telemetry else "0",
        "NINFER_FLASH_NEXT_EXPERT_CACHE": "0",
        "NINFER_FLASH_NEXT_STAGE_LEDGER": "0",
        "NINFER_FLASH_NEXT_FP32_MOE_ROUTED_INPUT": "0",
        "NINFER_FLASH_NEXT_CPU_EXPERT_FP32_INTERMEDIATE": "0",
    })
    return env


def invoke(executable: Path, output: Path, name: str, mode: str,
           positions: int, telemetry: bool) -> dict:
    env = environment(mode, positions, telemetry)
    snapshots = [gpu_snapshot()]
    done, errors = threading.Event(), []

    def monitor():
        while not done.wait(5):
            try:
                snapshots.append(gpu_snapshot())
            except Exception as error:  # preserve the diagnostic instead of hiding it
                errors.append(str(error))

    worker = threading.Thread(target=monitor)
    worker.start()
    try:
        process = subprocess.run(
            [str(executable.resolve())], env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
    finally:
        done.set()
        worker.join()
    snapshots.append(gpu_snapshot())
    (output / f"{name}.stdout.log").write_text(process.stdout)
    (output / f"{name}.stderr.log").write_text(process.stderr)
    atomic_json(output / f"{name}-hardware.json", {
        "snapshots": snapshots, "errors": errors,
        "environment": {key: value for key, value in env.items() if key.startswith("NINFER_")},
    })
    if process.returncode or errors:
        raise RuntimeError(f"{name}: prefill process or hardware monitor failed")
    if any(not row["thermal_status_observed"] or row["thermal_throttled"] for row in snapshots):
        raise ValueError(f"{name}: thermal status unavailable or throttling observed")
    result = {
        "name": name, "mode": mode, "positions": positions, "telemetry": telemetry,
        "probe": parse_probe(process.stdout, positions),
    }
    if telemetry:
        result["diagnostic"] = validate_diagnostic(process.stderr, positions, mode)
    print(f"{name}: {result['probe']['tokens_per_s']:.3f} tokens/s", flush=True)
    return result


def summarize(observations: list[dict], diagnostics: dict, repeats: int) -> dict:
    rows = []
    for positions in sorted({row["positions"] for row in observations}):
        cells = {}
        for mode in MODES:
            selected = [row for row in observations
                        if row["positions"] == positions and row["mode"] == mode]
            if len(selected) != repeats:
                raise ValueError("missing independent timing observations")
            signatures = {(row["probe"]["candidate_top1"], row["probe"]["oracle_top1"],
                           row["probe"]["kl"], row["probe"]["relative_nll_delta"],
                           row["probe"]["max_logit_error"]) for row in selected}
            if len(signatures) != 1:
                raise ValueError(f"{mode} {positions}: repeated numerical output changed")
            rates = [row["probe"]["tokens_per_s"] for row in selected]
            cells[mode] = {
                "observations": len(rates), "median": statistics.median(rates),
                "minimum": min(rates), "maximum": max(rates),
                "numerical": selected[0]["probe"],
            }
        if cells["cpu-cache"]["numerical"]["candidate_top1"] != cells["stream"]["numerical"]["candidate_top1"]:
            raise ValueError(f"CPU and stream top-1 differ at {positions} tokens")
        capacity = equal_capacity(
            diagnostics[f"cpu-cache-{positions}"]["diagnostic"]["runtime_memory"],
            diagnostics[f"stream-{positions}"]["diagnostic"]["runtime_memory"])
        rows.append({
            "positions": positions, "cpu-cache": cells["cpu-cache"], "stream": cells["stream"],
            "median_change_percent": 100 * (cells["stream"]["median"] /
                                              cells["cpu-cache"]["median"] - 1),
            "capacity": capacity,
            "transfers": {
                mode: {key: diagnostics[f"{mode}-{positions}"]["diagnostic"][key]
                       for key in ("routes", "distinct_experts", "stream_expert_h2d_bytes",
                                   "route_input_d2h_bytes", "cpu_miss_h2d_bytes")}
                for mode in MODES
            },
        })
    return {"schema": 1, "milestone": "SV3", "qualified": False,
            "scope": "fresh_process_prefill_stream_threshold_calibration",
            "repeats": repeats, "results": rows,
            "automatic_threshold": None,
            "limitations": [
                "screening uses the standalone real-model prefill probe, not HTTP serving",
                "persistent expert cache is disabled in both arms",
                "stream mode stages every selected expert; persistent-hit coexistence remains pending",
                "no automatic threshold or default change is made by this evidence collector",
            ]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three fresh-process observations are required")
    if tuple(sorted(set(args.sizes))) != tuple(args.sizes) or any(size < 1 or size > 2048 for size in args.sizes):
        parser.error("--sizes must be unique, increasing, and within the current 2048-token prefill chunk")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    diagnostics = {}
    for positions in args.sizes:
        for mode in MODES:
            row = invoke(args.executable, output, f"diagnostic-{mode}-{positions}",
                         mode, positions, True)
            diagnostics[f"{mode}-{positions}"] = row
            atomic_json(output / "diagnostics.json", diagnostics)

    observations = []
    for repeat in range(args.repeats):
        order = MODES if repeat % 2 == 0 else tuple(reversed(MODES))
        for positions in args.sizes:
            for mode in order:
                observations.append(invoke(
                    args.executable, output, f"timing-{repeat}-{mode}-{positions}",
                    mode, positions, False))
                atomic_json(output / "timing-observations.json", observations)

    report = summarize(observations, diagnostics, args.repeats)
    report["candidate_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    atomic_json(output / "calibration.json", report)
    lines = [
        "SV3 V100 prefill calibration; screening only, no automatic threshold selected.", "",
        "| Tokens | CPU t/s median (range) | Stream t/s median (range) | Median change | Expert H2D |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        cpu, stream = row["cpu-cache"], row["stream"]
        h2d = row["transfers"]["stream"]["stream_expert_h2d_bytes"]
        lines.append(
            f"| {row['positions']} | {cpu['median']:.3f} ({cpu['minimum']:.3f}–{cpu['maximum']:.3f}) | "
            f"{stream['median']:.3f} ({stream['minimum']:.3f}–{stream['maximum']:.3f}) | "
            f"{row['median_change_percent']:+.2f}% | {h2d} |")
    lines.extend(["", "Both arms use identical KV/state/workspace capacity and exact repeated outputs within each arm.",
                  "The staging-ring VRAM cost is reported separately; defaults remain unchanged."])
    (output / "calibration.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
