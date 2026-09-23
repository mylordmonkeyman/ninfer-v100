#!/usr/bin/env python3
"""Sequence D20: Code / Technical / Tool-Call Acceptance vs Workload Confound Benchmark.

Resolves whether the 87% (D3) vs 8.8% (D12) acceptance discrepancy is due to Workload
(Code/Technical English vs Conversational Portuguese) or Head Trimming (32k shortlist vs 248k full).

Arms:
  - spec-off: Base model decode without MTP speculation
  - 32k: MTP draft-1 with trimmed 32k shortlist draft head

Workloads:
  1. code_completion: Real C++20 engine codebase continuations (arena.h, layout.h, gated_delta_net.h)
  2. code_explanation: Formal English technical prose about runtime/compilers (D3 control profile)
  3. agentic_tool_json: Strict structured JSON tool calls for agentic coding workflows
  4. portuguese_concierge: Dialogue from D12 (control anchor)

Hard Memory Budget (Desktop Reserve):
  --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --max-private-continuations 16
  Allocates ~74-76 GiB, guaranteeing >= 20 GiB free for desktop display and DWM.

Reports:
  1. Single-stream decode tok/s (B=1)
  2. Per-stream decode tok/s under concurrency (noting C=1 budget constraint)
  3. Peak VRAM / Free Desktop VRAM
  Alongside Position-1 draft acceptance rate (accepted / proposed) with full distribution (Min, Median, IQR, Max).
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

DEFAULT_EXE = r"P:\NInfer\build-win\apps\Release\ninfer-serve.exe"
DEFAULT_MODEL = r"E:\models\Qwen3.8-Flash-Next\qwen3_8_flash_next_nvfp4_mtp.ninfer"
DEFAULT_PORT = 8160
DEFAULT_FIXTURES = r"P:\NInfer\bench\d20\fixtures\d20_workload_fixtures.json"
RANKING_PATH = r"P:\NInfer\tools\freq_corpus\fixtures\ranking\shortlist_32k.i32"
FFMPEG_BIN = r"P:\third_party\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin"
CURL_BIN = r"P:\third_party\curl-inst\bin"

def kill_any_ninfer_serve():
    subprocess.run(
        ["powershell", "-Command", "Stop-Process -Name 'ninfer-serve' -Force -ErrorAction SilentlyContinue"],
        capture_output=True
    )
    time.sleep(1.5)

def get_vram_info():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if res.returncode == 0:
            used, free, total = [int(x.strip()) for x in res.stdout.strip().split(",")]
            return {"used_mib": used, "free_mib": free, "total_mib": total}
    except Exception:
        pass
    return {"used_mib": 0, "free_mib": 0, "total_mib": 0}

def wait_vram_drained(threshold_mib=4000, max_wait=30):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        vram = get_vram_info()
        if vram["used_mib"] <= threshold_mib:
            return True
        time.sleep(1.0)
    return False

def load_fixtures(fixtures_path):
    with open(fixtures_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    workloads = {}
    for w_key, w_val in data.get("workloads", {}).items():
        workloads[w_key] = [
            (p["system"], p["user"])
            for p in w_val.get("prompts", [])
        ]
    return workloads

def start_server(exe_path, model_path, port, arm_name, config_mode, req_log_path, stdout_log, stderr_log):
    kill_any_ninfer_serve()
    wait_vram_drained()

    env = os.environ.copy()
    env["PATH"] = f"{FFMPEG_BIN};{CURL_BIN};" + env.get("PATH", "")

    args = [
        model_path,
        "--port", str(port),
        "--greedy",
        "--no-thinking",
        "--request-log-jsonl", req_log_path,
    ]

    if config_mode == "c1":
        args.extend([
            "--max-context", "32768",
            "--max-concurrency", "1",
            "--max-private-continuations", "16",
            "--prefill-chunk", "2048",
            "--kv-capacity", "32768",
        ])
    elif config_mode == "c8":
        args.extend([
            "--max-context", "262144",
            "--max-concurrency", "8",
            "--max-private-continuations", "48",
            "--prefill-chunk", "8192",
        ])
    else:
        raise ValueError(f"Unknown config_mode {config_mode}")

    if arm_name == "spec-off":
        pass
    elif arm_name == "32k":
        env["NINFER_FLASH_NEXT_DRAFT_HEAD_ROWS"] = "32768"
        env["NINFER_RANKING_PATH"] = RANKING_PATH
        args.extend(["--spec", "mtp", "--draft-tokens", "1", "--lm-head-draft"])
    elif arm_name == "fullhead":
        # Speculation through the full 248,320-row output head: no --lm-head-draft.
        # This is the configuration D3 measured at ~87% acceptance.
        args.extend(["--spec", "mtp", "--draft-tokens", "1"])
    else:
        raise ValueError(f"Unknown arm {arm_name}")

    cmd = [exe_path] + args
    print(f"  [Server] Launching {arm_name} ({config_mode}): {' '.join(cmd)}", flush=True)

    out_f = open(stdout_log, "w", encoding="utf-8")
    err_f = open(stderr_log, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=r"P:\NInfer", env=env, stdout=out_f, stderr=err_f)

    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    ready = False
    last_print = t0
    while time.time() - t0 < 240:
        if proc.poll() is not None:
            print(f"  [Server] Exited early with code {proc.returncode}", flush=True)
            break
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            if time.time() - last_print >= 15:
                print(f"  [Server] Waiting for readiness... ({time.time() - t0:.1f}s elapsed)", flush=True)
                last_print = time.time()
            time.sleep(1.0)

    if not ready:
        print("  [Server] Timed out waiting for readiness.", flush=True)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        kill_any_ninfer_serve()
        out_f.close()
        err_f.close()
        return None, out_f, err_f

    print(f"  [Server] Ready in {time.time() - t0:.2f}s (PID {proc.pid})", flush=True)
    return proc, out_f, err_f

def stop_server(proc, out_f, err_f):
    if proc is not None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    kill_any_ninfer_serve()
    if out_f: out_f.close()
    if err_f: err_f.close()
    wait_vram_drained()

def send_chat_request(port, sys_prompt, user_prompt, max_tokens=350):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    salted_prompt = f"[{time.time():.6f}] {user_prompt}"
    body = {
        "model": "qwen3.8-flash-next",
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": salted_prompt}
        ]
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            d = json.loads(raw)
        wall = time.time() - t0
        gen = d["usage"]["completion_tokens"]
        content = d["choices"][0]["message"].get("content") or ""
        return {
            "success": True,
            "gen_tokens": gen,
            "wall_s": wall,
            "tok_s": gen / max(wall, 1e-9),
            "content_len": len(content),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "wall_s": time.time() - t0
        }

def parse_request_log(jsonl_path):
    records = []
    if not os.path.exists(jsonl_path):
        return records
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records

def compute_stats(values):
    if not values:
        return {"n": 0, "min": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0, "p25": 0.0, "p75": 0.0, "iqr": 0.0, "std": 0.0}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / max(n - 1, 1)
    std = variance ** 0.5
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    p25 = s[int(n * 0.25)]
    p75 = s[int(n * 0.75)]
    iqr = p75 - p25
    return {
        "n": n, "min": s[0], "max": s[-1], "median": median, "mean": mean,
        "p25": p25, "p75": p75, "iqr": iqr, "std": std
    }

def main():
    parser = argparse.ArgumentParser(description="Sequence D20 Workload Acceptance Benchmark")
    parser.add_argument("--exe", default=DEFAULT_EXE, help="Path to ninfer-serve.exe")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model .ninfer artifact")
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURES, help="Path to d20_workload_fixtures.json")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to serve on")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per arm (default: 3)")
    parser.add_argument("--configs", nargs="+", default=["c1"], help="Configs to run (default: ['c1'], C=8 requires scheduled window)")
    parser.add_argument("--out-dir", default=r"P:\NInfer\bench\d20_results", help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "d20_raw_data.json")
    summary_path = os.path.join(args.out_dir, "d20_summary.json")

    workload_prompts = load_fixtures(args.fixtures)
    workload_names = list(workload_prompts.keys())
    arms = ["spec-off", "32k", "fullhead"]

    all_data = []
    if os.path.exists(raw_path):
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            print(f"Loaded {len(all_data)} existing measurements from {raw_path}", flush=True)
        except Exception:
            all_data = []

    print("================================================================================", flush=True)
    print("SEQUENCE D20: Code / Technical / Tool-Call Acceptance Benchmark", flush=True)
    print(f"Configs: {args.configs} | Workloads: {workload_names} | Arms: {arms} | Repeats: {args.repeats}", flush=True)
    print(f"Budget: --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --max-private-continuations 16", flush=True)
    print("================================================================================", flush=True)

    for cfg in args.configs:
        for rep in range(1, args.repeats + 1):
            for arm in arms:
                pending_workloads = [
                    w for w in workload_names
                    if not any(
                        d.get("config_mode") == cfg and d.get("repeat") == rep and d.get("arm") == arm and d.get("workload") == w
                        for d in all_data
                    )
                ]
                if not pending_workloads:
                    print(f"[Skip] Already completed all workloads for {cfg} rep {rep} {arm}", flush=True)
                    continue

                print(f"\n================================================================================", flush=True)
                print(f">>> Booting Server: Config={cfg} | Repeat={rep}/{args.repeats} | Arm={arm} <<<", flush=True)
                print(f"================================================================================", flush=True)

                prefix = f"{cfg}_rep{rep}_{arm}"
                req_log = os.path.join(args.out_dir, f"{prefix}_req.jsonl")
                stdout_log = os.path.join(args.out_dir, f"{prefix}_stdout.log")
                stderr_log = os.path.join(args.out_dir, f"{prefix}_stderr.log")

                if os.path.exists(req_log):
                    try: os.remove(req_log)
                    except Exception: pass

                proc, out_f, err_f = start_server(args.exe, args.model, args.port, arm, cfg, req_log, stdout_log, stderr_log)
                if proc is None:
                    print(f"[ERROR] Failed to start server for {cfg} rep {rep} {arm}", flush=True)
                    continue

                try:
                    print(f"  [{arm}] Warmup request...", flush=True)
                    _ = send_chat_request(args.port, "Warmup system", "Warmup hello", max_tokens=20)
                    time.sleep(1.0)

                    peak_vram = get_vram_info()

                    for w_name in pending_workloads:
                        prompts = workload_prompts[w_name]
                        print(f"  --- Running Workload: {w_name} ({arm}, {cfg}, rep {rep}) ---", flush=True)
                        w_results = []

                        if cfg == "c1":
                            for i, (sys_p, user_p) in enumerate(prompts):
                                res = send_chat_request(args.port, sys_p, user_p, max_tokens=350)
                                w_results.append(res)
                                print(f"    Q{i+1}: {res.get('gen_tokens', 0)} tok in {res.get('wall_s', 0):.2f}s -> {res.get('tok_s', 0):.1f} tok/s", flush=True)
                                curr_v = get_vram_info()
                                if curr_v["used_mib"] > peak_vram["used_mib"]:
                                    peak_vram = curr_v
                        elif cfg == "c8":
                            print(f"    [C=8] Launching 8 concurrent requests for {w_name}...", flush=True)
                            c8_prompts = [prompts[i % len(prompts)] for i in range(8)]
                            t0 = time.time()
                            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                                futs = [
                                    executor.submit(send_chat_request, args.port, s, u, 350)
                                    for (s, u) in c8_prompts
                                ]
                                w_results = [f.result() for f in futs]
                            wall = time.time() - t0
                            tot_tok = sum(r.get("gen_tokens", 0) for r in w_results if r.get("success"))
                            agg_rate = tot_tok / max(wall, 1e-9)
                            per_stream_rate = agg_rate / 8.0
                            print(f"    [C=8] {tot_tok} tokens in {wall:.2f}s | Aggregate: {agg_rate:.1f} tok/s | Per-Stream: {per_stream_rate:.1f} tok/s", flush=True)
                            curr_v = get_vram_info()
                            if curr_v["used_mib"] > peak_vram["used_mib"]:
                                peak_vram = curr_v

                        entry = {
                            "config_mode": cfg,
                            "repeat": rep,
                            "arm": arm,
                            "workload": w_name,
                            "results": w_results,
                            "peak_vram": peak_vram,
                        }
                        all_data.append(entry)

                    time.sleep(0.5)
                    engine_recs = parse_request_log(req_log)
                    done_recs = [r for r in engine_recs if r.get("event") == "request_done" and r.get("request", {}).get("request_id", 0) > 1]
                    reqs_per_w = 3 if cfg == "c1" else 8
                    for idx, w_name in enumerate(pending_workloads):
                        start_r = idx * reqs_per_w
                        end_r = start_r + reqs_per_w
                        sub_recs = done_recs[start_r:end_r]
                        for d in all_data:
                            if d.get("config_mode") == cfg and d.get("repeat") == rep and d.get("arm") == arm and d.get("workload") == w_name:
                                d["engine_records"] = sub_recs

                    with open(raw_path, "w", encoding="utf-8") as f:
                        json.dump(all_data, f, indent=2)

                finally:
                    stop_server(proc, out_f, err_f)

    # Compute Summary & Final Matrix
    summary = {}
    for cfg in args.configs:
        summary[cfg] = {}
        for w_name in workload_names:
            summary[cfg][w_name] = {}
            for arm in arms:
                entries = [
                    d for d in all_data
                    if d["config_mode"] == cfg and d["workload"] == w_name and d["arm"] == arm
                ]

                client_throughputs = []
                per_stream_throughputs = []
                for e in entries:
                    valid = [r for r in e["results"] if r.get("success")]
                    if cfg == "c1" and valid:
                        t = sum(r["tok_s"] for r in valid) / len(valid)
                        client_throughputs.append(t)
                        per_stream_throughputs.append(t)
                    elif cfg == "c8" and valid:
                        tot_tok = sum(r["gen_tokens"] for r in valid)
                        max_w = max(r["wall_s"] for r in valid)
                        agg = tot_tok / max(max_w, 1e-9)
                        client_throughputs.append(agg)
                        per_stream_throughputs.append(agg / 8.0)

                engine_decode_tok_s = []
                for e in entries:
                    for rec in e.get("engine_records", []):
                        dec_s = rec.get("timings_seconds", {}).get("decode", 0.0)
                        comp = rec.get("result", {}).get("completion_tokens", 0)
                        if dec_s > 0 and comp > 0:
                            engine_decode_tok_s.append(comp / dec_s)

                th_stats = compute_stats(client_throughputs)
                ps_stats = compute_stats(per_stream_throughputs)
                eng_stats = compute_stats(engine_decode_tok_s)

                acceptance_pct = None
                total_proposed = 0
                total_accepted = 0
                per_req_acceptance_rates = []
                position_acceptance = [0, 0, 0, 0, 0]

                if arm == "32k":
                    for e in entries:
                        for rec in e.get("engine_records", []):
                            spec = rec.get("speculative", {})
                            if spec:
                                prop = spec.get("drafted_tokens", 0)
                                acc = spec.get("accepted_tokens", 0)
                                pos_arr = spec.get("accepted_per_position", [])
                                if prop > 0:
                                    total_proposed += prop
                                    total_accepted += acc
                                    per_req_acceptance_rates.append(100.0 * acc / prop)
                                    for p_i, p_val in enumerate(pos_arr):
                                        if p_i < len(position_acceptance):
                                            position_acceptance[p_i] += p_val

                    if total_proposed > 0:
                        acceptance_pct = 100.0 * total_accepted / total_proposed

                acc_dist = compute_stats(per_req_acceptance_rates)
                peak_vram_list = [e.get("peak_vram", {}).get("used_mib", 0) for e in entries if e.get("peak_vram", {}).get("used_mib", 0) > 0]
                peak_vram_stats = compute_stats(peak_vram_list)

                summary[cfg][w_name][arm] = {
                    "throughput_aggregate": th_stats,
                    "throughput_per_stream": ps_stats,
                    "engine_decode_tok_s": eng_stats,
                    "acceptance_pct": acceptance_pct,
                    "acceptance_distribution": acc_dist,
                    "total_proposed_draft": total_proposed,
                    "total_accepted_draft": total_accepted,
                    "position_acceptance": position_acceptance,
                    "peak_vram_mib": peak_vram_stats,
                }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 125, flush=True)
    print("FINAL RESULTS MATRIX (TRIAD: SINGLE-STREAM / CONCURRENT PER-STREAM / PEAK VRAM + ACCEPTANCE RATE)", flush=True)
    print("=" * 125, flush=True)
    print(f"{'Config':6s} | {'Workload':22s} | {'Arm':8s} | {'Per-Stream (Client)':20s} | {'Per-Stream (Engine)':20s} | {'Acceptance Rate':16s} | {'Peak VRAM':12s}", flush=True)
    print("-" * 125, flush=True)
    for cfg in args.configs:
        for w_name in workload_names:
            for arm in arms:
                s = summary[cfg][w_name][arm]
                ps_med = s["throughput_per_stream"]["median"]
                eng_med = s["engine_decode_tok_s"]["median"]
                acc_rate = s["acceptance_pct"]
                acc_str = f"{acc_rate:5.1f}% (IQR: {s['acceptance_distribution']['iqr']:4.1f})" if acc_rate is not None else "n/a"
                vram_med = s["peak_vram_mib"]["median"]
                vram_str = f"{vram_med:.0f} MiB" if vram_med > 0 else "n/a"
                print(f"{cfg:6s} | {w_name:22s} | {arm:8s} | {ps_med:6.1f} tok/s (med)    | {eng_med:6.1f} tok/s (med)    | {acc_str:16s} | {vram_str:12s}", flush=True)

    print("\n" + "=" * 125, flush=True)
    print("RELATIVE SPECULATIVE SPEEDUP & ACCEPTANCE SUMMARY (32k vs spec-off)", flush=True)
    print("=" * 125, flush=True)
    for cfg in args.configs:
        print(f"\nConfiguration: {cfg.upper()}")
        print(f"{'Workload':22s} | {'Base tok/s':12s} | {'32k tok/s':12s} | {'Speedup':10s} | {'Acceptance Rate':18s} | {'Draft (Acc/Prop)':18s}")
        print("-" * 105)
        for w_name in workload_names:
            s_base = summary[cfg][w_name]["spec-off"]
            s_32k = summary[cfg][w_name]["32k"]
            base_tok = s_base["engine_decode_tok_s"]["median"]
            mtp_tok = s_32k["engine_decode_tok_s"]["median"]
            speedup = (mtp_tok / max(base_tok, 1e-9)) if base_tok > 0 else 1.0
            acc_rate = s_32k["acceptance_pct"]
            acc_str = f"{acc_rate:5.1f}%" if acc_rate is not None else "n/a"
            acc_cnt = f"{s_32k['total_accepted_draft']}/{s_32k['total_proposed_draft']}"
            print(f"{w_name:22s} | {base_tok:6.1f} tok/s | {mtp_tok:6.1f} tok/s | {speedup:6.2f}x   | {acc_str:18s} | {acc_cnt:18s}")

if __name__ == "__main__":
    main()
