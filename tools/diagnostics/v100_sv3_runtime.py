#!/usr/bin/env python3
"""Run a bounded real-model SV3 streaming integration check."""
import argparse
import json
import os
import pathlib
import subprocess

EXPERT_SLOT_BYTES = 2_765_056


def validate(stderr: str, positions: int, cached: bool = False) -> dict:
    records = []
    for line in stderr.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "expert_layer" and record.get("prefill"):
            records.append(record)
    passes = 2 if cached else 1
    if len(records) != 48 * passes:
        raise RuntimeError(f"expected {48 * passes} streamed prefill layers, found {len(records)}")
    for start in range(0, len(records), 48):
        if sorted(record.get("layer", -1) for record in records[start:start + 48]) != list(range(48)):
            raise RuntimeError("streamed prefill layer coverage mismatch")
    for record in records:
        experts = record.get("stream_experts", 0)
        if record.get("tokens") != positions or record.get("routes") != positions * 10:
            raise RuntimeError("streamed prefill route shape mismatch")
        hits = record.get("gpu_hit_routes", 0)
        streamed = record.get("stream_routes", 0)
        if streamed + hits != positions * 10 or (not cached and hits):
            raise RuntimeError("resident/streamed prefill route partition mismatch")
        if record.get("cpu_miss_routes") != 0:
            raise RuntimeError("streamed prefill polluted cache/CPU route accounting")
        if (experts < 0 or (streamed > 0 and experts < 1) or
                (streamed == 0 and experts != 0) or
                record.get("stream_expert_h2d_bytes") != experts * EXPERT_SLOT_BYTES):
            raise RuntimeError("streamed expert transfer accounting mismatch")
        if record.get("route_input_d2h_bytes") != positions * 10 * 4:
            raise RuntimeError("streamed prefill copied route activations to the host")
        if any(record.get(key, 0) for key in (
                "cache_result_d2h_bytes", "routed_sum_h2d_bytes", "cpu_miss_h2d_bytes")):
            raise RuntimeError("streamed prefill introduced expert result transfers")
        if cached:
            cache = record.get("cache", {})
            if (cache.get("admissions_total") != 3072 or
                    cache.get("fills_total") != 3072 or
                    cache.get("evictions_total") != 0 or
                    cache.get("leased_experts") != 0):
                raise RuntimeError("streamed prefill changed the seeded resident set or retained leases")
    if cached and (sum(r["gpu_hit_routes"] for r in records) == 0 or
                   sum(r["stream_routes"] for r in records) == 0):
        raise RuntimeError("cached prefill did not exercise both resident hits and staged misses")
    return {
        "positions": positions,
        "layers": 48,
        "passes": passes,
        "routes": sum(r["stream_routes"] for r in records),
        "resident_routes": sum(r.get("gpu_hit_routes", 0) for r in records),
        "experts": sum(r["stream_experts"] for r in records),
        "expert_h2d_bytes": sum(r["stream_expert_h2d_bytes"] for r in records),
    }


def parse_runtime_probe(stdout: str, positions: int, cached: bool = False) -> dict:
    try:
        from .v100_sv3_calibrate import parse_probe
    except ImportError:
        from v100_sv3_calibrate import parse_probe
    lines = [line for line in stdout.splitlines()
             if line.startswith("phase11.prefill_probe.")]
    passes = 2 if cached else 1
    if len(lines) != passes:
        raise RuntimeError("prefill probe pass count mismatch")
    probes = []
    for index, line in enumerate(lines):
        cumulative = positions * 10 * 48 * (index + 1)
        if f"expert_pairs={cumulative}" not in line.split():
            raise RuntimeError("prefill probe cumulative expert count mismatch")
        normalized = line.replace(f"expert_pairs={cumulative}",
                                  f"expert_pairs={positions * 10 * 48}")
        probes.append(parse_probe(normalized, positions))
    keys = ("candidate_top1", "oracle_top1", "kl", "relative_nll_delta", "max_logit_error")
    if any(probe[key] != probes[0][key] for probe in probes for key in keys):
        raise RuntimeError("prefill probe repeated numerical metrics differ")
    return probes[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--profile", type=pathlib.Path)
    args = parser.parse_args()
    if args.positions < 1:
        parser.error("--positions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("NINFER_V100_EXPERT_") or key.startswith("NINFER_FLASH_NEXT_EXPERT_CACHE"):
            env.pop(key)
    env.update({
        "NINFER_V100_PREFILL_EXPERT_POLICY": "stream",
        "NINFER_V100_DEVICE_ROUTE_COMBINE": "1",
        "NINFER_V100_TELEMETRY": "1",
        "NINFER_PHASE11_PREFILL_PROBE_POSITIONS": str(args.positions),
        "NINFER_FLASH_NEXT_EXPERT_CACHE": "0",
    })
    if args.profile:
        env.update({
            "NINFER_FLASH_NEXT_EXPERT_CACHE": "1",
            "NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS": "64",
            "NINFER_V100_EXPERT_POLICY": "lru",
            "NINFER_V100_EXPERT_PROFILE": str(args.profile.resolve()),
        })
    result = subprocess.run([str(args.executable)], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (args.output / "runtime.stdout.log").write_text(result.stdout)
    (args.output / "runtime.stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"streamed real-model probe exited {result.returncode}")
    parse_runtime_probe(result.stdout, args.positions, cached=bool(args.profile))
    summary = validate(result.stderr, args.positions, cached=bool(args.profile))
    (args.output / "runtime-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
