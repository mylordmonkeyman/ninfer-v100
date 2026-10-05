#!/usr/bin/env python3
"""Run a bounded real-model SV3 streaming integration check."""
import argparse
import json
import os
import pathlib
import subprocess

EXPERT_SLOT_BYTES = 2_765_056


def validate(stderr: str, positions: int) -> dict:
    records = []
    for line in stderr.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "expert_layer" and record.get("prefill"):
            records.append(record)
    if len(records) != 48:
        raise RuntimeError(f"expected 48 streamed prefill layers, found {len(records)}")
    for record in records:
        experts = record.get("stream_experts", 0)
        if record.get("tokens") != positions or record.get("routes") != positions * 10:
            raise RuntimeError("streamed prefill route shape mismatch")
        if record.get("stream_routes") != positions * 10:
            raise RuntimeError("not every prefill route used the staging ring")
        if record.get("gpu_hit_routes") != 0 or record.get("cpu_miss_routes") != 0:
            raise RuntimeError("streamed prefill polluted cache/CPU route accounting")
        if experts < 1 or record.get("stream_expert_h2d_bytes") != experts * EXPERT_SLOT_BYTES:
            raise RuntimeError("streamed expert transfer accounting mismatch")
        if record.get("route_input_d2h_bytes") != positions * 10 * 4:
            raise RuntimeError("streamed prefill copied route activations to the host")
    return {
        "positions": positions,
        "layers": len(records),
        "routes": sum(r["stream_routes"] for r in records),
        "experts": sum(r["stream_experts"] for r in records),
        "expert_h2d_bytes": sum(r["stream_expert_h2d_bytes"] for r in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--positions", type=int, default=32)
    args = parser.parse_args()
    if args.positions < 1:
        parser.error("--positions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "NINFER_V100_PREFILL_EXPERT_POLICY": "stream",
        "NINFER_V100_DEVICE_ROUTE_COMBINE": "1",
        "NINFER_V100_TELEMETRY": "1",
        "NINFER_PHASE11_PREFILL_PROBE_POSITIONS": str(args.positions),
    })
    result = subprocess.run([str(args.executable)], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (args.output / "runtime.stdout.log").write_text(result.stdout)
    (args.output / "runtime.stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"streamed real-model probe exited {result.returncode}")
    summary = validate(result.stderr, args.positions)
    (args.output / "runtime-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
