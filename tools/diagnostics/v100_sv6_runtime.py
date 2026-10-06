#!/usr/bin/env python3
"""Bounded SV6 mmap/direct PLE integration and process-wall screen."""
import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import time


NUMERIC_KEYS = ("candidate_top1", "oracle_top1", "kl", "relative_nll_delta", "max_logit_error")


def parse_probe(stdout: str, positions: int) -> dict:
    prefix = f"phase11.prefill_probe.positions={positions} "
    lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(f"expected one {positions}-position prefill probe, found {len(lines)}")
    values = {}
    for field in lines[0].split():
        if "=" in field:
            key, value = field.split("=", 1)
            values[key] = value
    return {
        "candidate_top1": int(values["candidate_top1"]),
        "oracle_top1": int(values["oracle_top1"]),
        "kl": float(values["kl"]),
        "relative_nll_delta": float(values["relative_nll_delta"]),
        "max_logit_error": float(values["max_logit_error"]),
        "elapsed_s": float(values["elapsed_s"]),
    }


def parse_ple(stderr: str, positions: int, mode: str) -> dict:
    records = []
    for line in stderr.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "ple_gather" and record.get("compressed"):
            records.append(record)
    if len(records) != 1:
        raise RuntimeError(f"expected one compressed PLE record, found {len(records)}")
    record = records[0]
    if record.get("tokens") != positions or record.get("payload_bytes") != positions * 1600:
        raise RuntimeError("PLE record shape/accounting mismatch")
    if record.get("storage_backend") != mode or record.get("storage_fallback"):
        raise RuntimeError(f"PLE {mode} diagnostic used {record.get('storage_backend')} fallback")
    if mode == "direct":
        if record.get("coalesced_pages", 0) <= 0 or record.get("page_read_us") is None:
            raise RuntimeError("direct PLE diagnostic did not report coalesced page reads")
    elif record.get("coalesced_pages") != 0 or record.get("page_read_us") is not None:
        raise RuntimeError("mmap PLE diagnostic unexpectedly reported direct page reads")
    return record


def run(executable: pathlib.Path, output: pathlib.Path, mode: str, positions: int,
        repeat: int, telemetry: bool) -> dict:
    name = f"{mode}-{'diagnostic' if telemetry else f'timing-{repeat}'}"
    run_dir = output / name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("NINFER_V100_PLE_") or key == "NINFER_V100_TELEMETRY":
            env.pop(key)
    env.update({
        "NINFER_V100_PLE_STORAGE": mode,
        "NINFER_V100_PLE_QUEUE_DEPTH": "64",
        "NINFER_V100_TELEMETRY": "1" if telemetry else "0",
        "NINFER_PHASE11_PREFILL_PROBE_POSITIONS": str(positions),
        "NINFER_FLASH_NEXT_EXPERT_CACHE": "0",
        "NINFER_V100_SV4_LOGITS": str(run_dir),
    })
    started = time.monotonic()
    result = subprocess.run([str(executable)], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wall = time.monotonic() - started
    (run_dir / "stdout.log").write_text(result.stdout)
    (run_dir / "stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}")
    logits = run_dir / "pass0.bf16"
    if not logits.is_file():
        raise RuntimeError(f"{name} did not preserve represented logits")
    row = {
        "mode": mode,
        "repeat": repeat,
        "telemetry": telemetry,
        "process_wall_s": wall,
        "probe": parse_probe(result.stdout, positions),
        "logits_sha256": hashlib.sha256(logits.read_bytes()).hexdigest(),
    }
    if telemetry:
        row["ple"] = parse_ple(result.stderr, positions, mode)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--positions", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.positions < 1 or args.repeats < 1:
        parser.error("positions and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = [run(args.executable, args.output, mode, args.positions, -1, True)
            for mode in ("mmap", "direct")]
    for repeat in range(args.repeats):
        modes = ("mmap", "direct") if repeat % 2 == 0 else ("direct", "mmap")
        rows.extend(run(args.executable, args.output, mode, args.positions, repeat, False)
                    for mode in modes)

    reference = rows[0]
    for row in rows[1:]:
        if row["logits_sha256"] != reference["logits_sha256"] or any(
                row["probe"][key] != reference["probe"][key] for key in NUMERIC_KEYS):
            raise RuntimeError("mmap/direct PLE outputs or selected numerical metrics differ")
    summary = {}
    for mode in ("mmap", "direct"):
        timing = [row for row in rows if row["mode"] == mode and not row["telemetry"]]
        summary[mode] = {
            "prefill_elapsed_s_median": statistics.median(row["probe"]["elapsed_s"] for row in timing),
            "prefill_elapsed_s_range": [min(row["probe"]["elapsed_s"] for row in timing),
                                         max(row["probe"]["elapsed_s"] for row in timing)],
            "process_wall_s_median": statistics.median(row["process_wall_s"] for row in timing),
            "process_wall_s_range": [min(row["process_wall_s"] for row in timing),
                                      max(row["process_wall_s"] for row in timing)],
        }
    report = {
        "schema": 1,
        "milestone": "SV6",
        "qualified": False,
        "positions": args.positions,
        "exact_logits_and_selected_metrics": True,
        "results": summary,
        "diagnostics": {row["mode"]: row["ple"] for row in rows if row["telemetry"]},
        "limitations": [
            "single represented prefill probe; not served TTFT",
            "direct backend bypasses page cache; mmap arm retains established full-table startup warm",
            "no concurrent requests or filesystem portability qualification",
        ],
    }
    (args.output / "observations.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["SV6 PLE mmap/direct integration screen; qualified=false.", "",
             "| Mode | Prefill seconds median (range) | Process wall seconds median (range) |",
             "|---|---:|---:|"]
    for mode in ("mmap", "direct"):
        item = summary[mode]
        lines.append(f"| {mode} | {item['prefill_elapsed_s_median']:.3f} "
                     f"({item['prefill_elapsed_s_range'][0]:.3f}–{item['prefill_elapsed_s_range'][1]:.3f}) | "
                     f"{item['process_wall_s_median']:.3f} "
                     f"({item['process_wall_s_range'][0]:.3f}–{item['process_wall_s_range'][1]:.3f}) |")
    (args.output / "sv6-runtime.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
