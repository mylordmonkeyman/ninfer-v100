#!/usr/bin/env python3
"""Bounded HTTP TTFT screen for warm mmap, file-cold mmap, and strict direct PLE I/O."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request

from v100_expert_profile import atomic_json
from v100_sv0_benchmark import gpu_snapshot


MODES = ("mmap-warm", "mmap-cold", "direct")

CURRENT_SERVER = {"name": "none", "log": None}


def request(base, path, payload=None):
    encoded = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, encoded, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def response_signature(response):
    choice = response["choices"][0]
    return {"message": choice["message"], "finish_reason": choice["finish_reason"],
            "completion_tokens": response["usage"]["completion_tokens"]}


def parse_startup_log(text, mode):
    warm = re.findall(r"flash_next host_ple_warm mode=(mmap|skipped) bytes=(\d+) duration_ms=([0-9.]+)", text)
    if len(warm) != 1:
        raise ValueError("missing unique PLE warm record")
    cache = re.findall(r"flash_next host_ple_cache mode=(warm|cold) total_pages=(\d+) resident_pages=(\d+)", text)
    if mode == "direct":
        if warm[0][0] != "skipped" or cache:
            raise ValueError("strict direct server unexpectedly warmed or inspected mmap PLE pages")
        return {"warm_mode": "skipped", "warm_bytes": int(warm[0][1]),
                "warm_ms": float(warm[0][2])}
    expected = "warm" if mode == "mmap-warm" else "cold"
    if warm[0][0] != "mmap" or len(cache) != 1 or cache[0][0] != expected:
        raise ValueError("mmap server did not establish the requested PLE cache condition")
    total, resident = int(cache[0][1]), int(cache[0][2])
    if total <= 0 or resident != (total if expected == "warm" else 0):
        raise ValueError("PLE cache residency verification failed")
    return {"warm_mode": "mmap", "warm_bytes": int(warm[0][1]),
            "warm_ms": float(warm[0][2]), "cache_mode": expected,
            "total_pages": total, "resident_pages": resident}


def parse_ple_records(text, mode, telemetry):
    records = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "ple_gather" and row.get("compressed"):
            records.append(row)
    if not telemetry:
        if records:
            raise ValueError("telemetry-off timing server emitted PLE diagnostics")
        return None
    if len(records) != 1:
        raise ValueError(f"expected one compressed PLE record, found {len(records)}")
    row = records[0]
    backend = "direct" if mode == "direct" else "mmap"
    if row.get("storage_backend") != backend or row.get("storage_fallback"):
        raise ValueError("PLE request used the wrong storage backend or fallback")
    if row.get("tokens", 0) <= 0 or row.get("payload_bytes") != row["tokens"] * 1600:
        raise ValueError("PLE request compressed-byte accounting differs")
    if backend == "direct" and (row.get("coalesced_pages", 0) <= 0 or
                                row.get("page_read_us") is None):
        raise ValueError("strict direct request lacks page-read evidence")
    return row


def run_server(executable, artifact, output, mode, repeat, telemetry):
    name = f"{mode}-{'diagnostic' if telemetry else f'timing-{repeat}'}"
    log_path = output / f"{name}.log"
    CURRENT_SERVER["name"] = name
    CURRENT_SERVER["log"] = log_path
    request_path = output / f"{name}-requests.jsonl"
    env = os.environ.copy()
    for key in tuple(env):
        if (key.startswith("NINFER_V100_PLE_") or key.startswith("NINFER_PHASE") or
                key.startswith("NINFER_V100_EXPERT_") or
                key.startswith("NINFER_V100_PREFILL_") or
                key.startswith("NINFER_V100_ROUTE_HANDOFF") or
                key.startswith("NINFER_V100_CPU_EXPERT_GROUP")):
            env.pop(key)
    if mode == "direct":
        env.update(NINFER_V100_PLE_IO="direct", NINFER_V100_PLE_STRICT_DIRECT="1")
    else:
        env.update(NINFER_V100_PLE_IO="mmap",
                   NINFER_V100_PLE_DIAGNOSTIC_CACHE="warm" if mode == "mmap-warm" else "cold")
    env.update(NINFER_V100_PLE_QUEUE_DEPTH="64",
               NINFER_V100_TELEMETRY="1" if telemetry else "0",
               NINFER_FLASH_NEXT_EXPERT_CACHE="0", NINFER_V100_DEVICE_ROUTE_COMBINE="0",
               NINFER_V100_CPU_EXPERT_GROUP="0", NINFER_V100_ROUTE_HANDOFF="0")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]
    command = [str(executable.resolve()), str(artifact.absolute()), "--host", "127.0.0.1",
               "--port", str(port), "--max-context", "4096", "--kv-capacity", "4096",
               "--max-concurrency", "1", "--prefill-chunk", "2048", "--kv-dtype", "bf16",
               "--device-state-slots", "2", "--host-state-slots", "0", "--host-kv-mib", "0",
               "--max-private-continuations", "1", "--max-shared-prefixes", "0",
               "--max-long-anchors-per-continuation", "0", "--no-cuda-graph",
               "--no-qsa-prefill-mma", "--no-thinking", "--request-log-jsonl", str(request_path)]
    snapshots, monitor_errors = [gpu_snapshot()], []
    stopped = threading.Event()
    def monitor():
        while not stopped.wait(5):
            try: snapshots.append(gpu_snapshot())
            except Exception as error: monitor_errors.append(str(error))
    worker = threading.Thread(target=monitor); worker.start()
    response = None
    started = time.monotonic()
    ready_seconds = None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                base = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 900
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{name}: server exited before readiness; see {log_path}")
                    try:
                        models = request(base, "/v1/models"); break
                    except (urllib.error.URLError, TimeoutError, ConnectionResetError):
                        if time.monotonic() > deadline: raise TimeoutError("server readiness timeout")
                        time.sleep(0.5)
                ready_seconds = time.monotonic() - started
                model = models["data"][0]["id"]
                paragraphs = " ".join(
                    f"Record {i}: the solar station stores energy during daylight and supplies the village after sunset."
                    for i in range(64))
                payload = {"model": model,
                           "messages": [{"role": "user", "content": paragraphs +
                                         " Explain how its battery storage works in detail."}],
                           "max_tokens": 16, "temperature": 0, "seed": 42,
                           "enable_thinking": False, "prompt_cache_read_only": True}
                response = request(base, "/v1/chat/completions", payload)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=10)
    finally:
        stopped.set(); worker.join(); snapshots.append(gpu_snapshot())
        atomic_json(output / f"{name}-hardware.json",
                    {"snapshots": snapshots, "errors": monitor_errors, "command": command,
                     "environment": {k: v for k, v in env.items() if k.startswith("NINFER_")}})
        if response is not None: atomic_json(output / f"{name}-response.json", response)
    if monitor_errors or any(not s["thermal_status_observed"] or s["thermal_throttled"]
                             for s in snapshots):
        raise ValueError("production thermal evidence missing or throttled")
    events = [json.loads(line) for line in request_path.read_text().splitlines() if line.strip()]
    done = [row for row in events if row.get("event") == "request_done"]
    if len(done) != 1 or any(row.get("event") in ("request_error", "request_rejected") for row in events):
        raise ValueError("missing successful HTTP request evidence")
    event = done[0]
    timings = event["timings_seconds"]
    if event["result"]["prefix_cache_hit_tokens"] != 0:
        raise ValueError("TTFT request unexpectedly reused a prefix")
    if event["result"]["prompt_tokens"] != response["usage"]["prompt_tokens"]:
        raise ValueError("HTTP and engine prompt accounting differ")
    if not 1024 <= event["result"]["prompt_tokens"] <= 2048:
        raise ValueError("TTFT request did not exercise one large prefill chunk")
    if any(not math.isfinite(timings[key]) or timings[key] <= 0
           for key in ("ttft", "prefill", "total")):
        raise ValueError("invalid TTFT timing evidence")
    text = log_path.read_text()
    row = {"name": name, "mode": mode, "repeat": repeat, "telemetry": telemetry,
           "ready_seconds": ready_seconds, "startup": parse_startup_log(text, mode),
           "ple": parse_ple_records(text, mode, telemetry), "request": event,
           "response_signature": response_signature(response)}
    print(f"{name}: ready={ready_seconds:.3f}s ttft={timings['ttft']:.3f}s", flush=True)
    return row
def failure_digest(output):
    """Bounded digest of the in-flight state for the workflow's public annotations."""
    parts = ["SV6 serve screen failed", f"in-flight server: {CURRENT_SERVER['name']}"]
    log_path = CURRENT_SERVER["log"]
    if log_path is not None and log_path.exists():
        text = log_path.read_text(errors="replace")
        for line in text.splitlines():
            if "host_ple_warm" in line or "host_ple_cache" in line:
                parts.append("startup: " + line)
        records = [line for line in text.splitlines() if '"kind":"ple_gather"' in line]
        parts.append(f"ple_gather records: {len(records)}")
        for record in records[:2]:
            parts.append("ple: " + record[:400])
        parts.append("log tail:")
        parts.extend(line for line in text.splitlines()[-12:])
    name = CURRENT_SERVER["name"]
    request_path = output / f"{name}-requests.jsonl"
    if request_path.exists():
        for line in request_path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "request_done":
                result = row.get("result", {})
                timings = row.get("timings_seconds", {})
                parts.append(
                    "request_done: prompt_tokens={} prefix_cache_hit_tokens={} ttft={} "
                    "prefill={} total={}".format(
                        result.get("prompt_tokens"),
                        result.get("prefix_cache_hit_tokens"),
                        timings.get("ttft"), timings.get("prefill"), timings.get("total")))
            elif row.get("event") in ("request_error", "request_rejected"):
                parts.append("event: " + json.dumps(row)[:400])
    response_path = output / f"{name}-response.json"
    if response_path.exists():
        try:
            parts.append(
                "response: " + json.dumps(json.loads(response_path.read_text()))[:400])
        except (json.JSONDecodeError, OSError):
            parts.append("response: unparseable")
    digest = "\n".join(parts)[:2800]
    (output / "failure-digest.txt").write_text(digest + "\n")
    return digest



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.artifact.suffix != ".ninfer" or not args.artifact.is_file():
        parser.error("--artifact requires an explicit readable .ninfer file")
    if args.repeats < 3: parser.error("need at least three fresh-process repetitions")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        return run_screen(args)
    except Exception as error:
        traceback_tail = traceback.format_exc().splitlines()
        digest = failure_digest(args.output)
        print(f"SV6 serve screen failed: {type(error).__name__}: {error}", flush=True)
        print("traceback (tail):", flush=True)
        print("\n".join(traceback_tail[-12:]), flush=True)
        print("failure digest:", flush=True)
        print(digest, flush=True)
        return 1


def run_screen(args):
    rows = [run_server(args.executable, args.artifact, args.output, mode, -1, True)
            for mode in MODES]
    for repeat in range(args.repeats):
        order = MODES[repeat % len(MODES):] + MODES[:repeat % len(MODES)]
        for mode in order:
            rows.append(run_server(args.executable, args.artifact, args.output,
                                   mode, repeat, False))
            atomic_json(args.output / "observations.json", rows)
    signature = rows[0]["response_signature"]
    if any(row["response_signature"] != signature for row in rows):
        raise ValueError("PLE storage/cache condition changed the exact greedy response")
    results = {}
    for mode in MODES:
        timing = [row for row in rows if row["mode"] == mode and not row["telemetry"]]
        ttft = [row["request"]["timings_seconds"]["ttft"] for row in timing]
        ready = [row["ready_seconds"] for row in timing]
        results[mode] = {"ttft_seconds_median": statistics.median(ttft),
                         "ttft_seconds_range": [min(ttft), max(ttft)],
                         "ready_seconds_median": statistics.median(ready),
                         "ready_seconds_range": [min(ready), max(ready)]}
    report = {"schema": 1, "milestone": "SV6", "qualified": False,
              "scope": "production_http_ple_cold_warm_ttft_screen",
              "exact_responses": True, "results": results,
              "limitations": ["single sequential HTTP request per fresh server",
                              "file-specific cold verification may refuse to run when another process pins PLE pages",
                              "no concurrent request or alternate-filesystem qualification"]}
    atomic_json(args.output / "report.json", report)
    lines = ["SV6 HTTP PLE cold/warm TTFT screen; qualified=false.", "",
             "| Mode | TTFT seconds median (range) | Ready seconds median (range) |",
             "|---|---:|---:|"]
    for mode in MODES:
        row = results[mode]
        lines.append(f"| {mode} | {row['ttft_seconds_median']:.3f} "
                     f"({row['ttft_seconds_range'][0]:.3f}–{row['ttft_seconds_range'][1]:.3f}) | "
                     f"{row['ready_seconds_median']:.3f} "
                     f"({row['ready_seconds_range'][0]:.3f}–{row['ready_seconds_range'][1]:.3f}) |")
    (args.output / "sv6-serve.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
