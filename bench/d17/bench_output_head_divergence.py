#!/usr/bin/env python3
"""Sequence D17: Output-Head FP8 vs BF16 Greedy Divergence Benchmark.

Measures the exact divergence position (P_div) of autoregressively generated token
streams between:
  - Arm 1 (Baseline): BF16 unquantized output head (1.18 GiB tensor)
  - Arm 2 (Quantized): FP8 E4M3 row-scaled output head (~607 MiB tensor, ~605 MiB VRAM saving)

Quality Requirement:
  Unlike speculative draft heads (where mispredictions are caught and discarded by
  the target model), the primary output head directly projects final hidden states to
  vocabulary logits. A defect or loss of precision here directly alters emitted tokens.
  This harness evaluates greedy generation (temperature=0.0) on identical prompts and seeds,
  recording:
    1. Exact divergence position (P_div, 1-based token index of first differing token)
    2. Prefix match length and ratio
    3. First differing token pair
    4. Statistical distribution of P_div (Min, P25, Median, P75, Max, Mean)
    5. VRAM reduction and desktop reserve preservation (>= 16 GiB free VRAM)

Usage:
  # Fully automated sequential execution (starts BF16 server, evaluates, stops, starts FP8 server, evaluates):
  python bench/d17/bench_output_head_divergence.py --port 8160

  # Point to already running servers:
  python bench/d17/bench_output_head_divergence.py --bf16-url http://127.0.0.1:8160 --fp8-url http://127.0.0.1:8161
"""

import argparse
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
DEFAULT_FIXTURES = r"P:\NInfer\bench\d17\fixtures\d17_head_divergence_fixtures.json"
DEFAULT_OUT_JSON = r"P:\NInfer\bench\d17\d17_output_head_divergence_summary.json"
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
            parts = [int(x.strip()) for x in res.stdout.strip().split(",")]
            return {"used_mib": parts[0], "free_mib": parts[1], "total_mib": parts[2]}
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


def load_fixtures(fixtures_path, filter_category=None):
    with open(fixtures_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    categories = data.get("categories", {})
    all_prompts = []
    for cat_name, cat_info in categories.items():
        if filter_category and cat_name != filter_category:
            continue
        for p in cat_info.get("prompts", []):
            all_prompts.append({
                "category": cat_name,
                "id": p["id"],
                "system": p["system"],
                "user": p["user"],
            })
    return all_prompts


def start_server(exe_path, model_path, port, is_quantized=False, quant_target="output-head", spec_backend="off", stdout_log=None, stderr_log=None):
    kill_any_ninfer_serve()
    wait_vram_drained()

    env = os.environ.copy()
    env["PATH"] = f"{FFMPEG_BIN};{CURL_BIN};" + env.get("PATH", "")

    if not is_quantized:
        head_dtype = "bf16"
        emb_dtype = "bf16"
    else:
        if quant_target == "output-head":
            head_dtype = "fp8"
            emb_dtype = "bf16"
        elif quant_target == "token-embedding":
            head_dtype = "bf16"
            emb_dtype = "fp8"
        elif quant_target == "both":
            head_dtype = "fp8"
            emb_dtype = "fp8"
        else:
            raise ValueError(f"Unknown quant_target: {quant_target}")

    args = [
        model_path,
        "--port", str(port),
        "--output-head-dtype", head_dtype,
        "--token-embedding-dtype", emb_dtype,
        "--max-context", "32768",
        "--kv-capacity", "32768",
        "--max-concurrency", "1",
        "--max-private-continuations", "16",
        "--prefill-chunk", "2048",
        "--greedy",
        "--no-thinking",
    ]

    if spec_backend == "mtp":
        args.extend(["--spec", "mtp", "--draft-tokens", "1"])
    elif spec_backend == "off":
        pass  # speculation disabled by default when --spec is omitted

    cmd = [exe_path] + args
    print(f"  [Server] Launching (head={head_dtype}, embed={emb_dtype}, port {port}): {' '.join(cmd)}", flush=True)

    out_f = open(stdout_log, "w", encoding="utf-8") if stdout_log else subprocess.DEVNULL
    err_f = open(stderr_log, "w", encoding="utf-8") if stderr_log else subprocess.DEVNULL
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
        if stdout_log: out_f.close()
        if stderr_log: err_f.close()
        return None, out_f, err_f

    print(f"  [Server] Ready in {time.time() - t0:.2f}s (PID {proc.pid})", flush=True)
    return proc, out_f, err_f


def stop_server(proc, out_f=None, err_f=None):
    if proc is not None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    kill_any_ninfer_serve()
    if out_f and out_f != subprocess.DEVNULL: out_f.close()
    if err_f and err_f != subprocess.DEVNULL: err_f.close()
    wait_vram_drained()


def send_chat_completion_streaming(url, sys_prompt, user_prompt, max_tokens=256):
    """Sends chat completion with stream=True and collects individual token delta chunks."""
    body = {
        "model": "qwen3.8-flash-next",
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"}
    )
    tokens = []
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line_bytes in resp:
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    payload = json.loads(data_str)
                    choices = payload.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            tokens.append(content)
                except json.JSONDecodeError:
                    pass
        wall_s = time.time() - t0
        full_text = "".join(tokens)
        return {
            "success": True,
            "tokens": tokens,
            "token_count": len(tokens),
            "text": full_text,
            "wall_s": wall_s,
            "tok_per_sec": len(tokens) / wall_s if wall_s > 0 else 0.0,
        }
    except Exception as exc:
        wall_s = time.time() - t0
        return {
            "success": False,
            "tokens": [],
            "token_count": 0,
            "text": "",
            "wall_s": wall_s,
            "tok_per_sec": 0.0,
            "error": str(exc),
        }


def compare_token_streams(bf16_tokens, fp8_tokens):
    """Calculates the divergence position and prefix match statistics between two token streams."""
    n_bf16 = len(bf16_tokens)
    n_fp8 = len(fp8_tokens)
    min_len = min(n_bf16, n_fp8)

    first_diff_idx = None
    for i in range(min_len):
        if bf16_tokens[i] != fp8_tokens[i]:
            first_diff_idx = i
            break

    if first_diff_idx is not None:
        # Divergence occurred at index first_diff_idx (1-based position = first_diff_idx + 1)
        p_div = first_diff_idx + 1
        prefix_match_length = first_diff_idx
        diff_token_bf16 = bf16_tokens[first_diff_idx]
        diff_token_fp8 = fp8_tokens[first_diff_idx]
    else:
        if n_bf16 == n_fp8:
            # 100% exact match across all generated tokens
            p_div = None
            prefix_match_length = min_len
            diff_token_bf16 = None
            diff_token_fp8 = None
        else:
            # One stream terminated earlier than the other
            p_div = min_len + 1
            prefix_match_length = min_len
            diff_token_bf16 = bf16_tokens[min_len] if n_bf16 > min_len else "<EOS/END>"
            diff_token_fp8 = fp8_tokens[min_len] if n_fp8 > min_len else "<EOS/END>"

    prefix_match_ratio = prefix_match_length / min_len if min_len > 0 else 1.0

    return {
        "p_div": p_div,
        "prefix_match_length": prefix_match_length,
        "prefix_match_ratio": prefix_match_ratio,
        "first_diff_bf16": diff_token_bf16,
        "first_diff_fp8": diff_token_fp8,
        "bf16_total_tokens": n_bf16,
        "fp8_total_tokens": n_fp8,
    }


def compute_quantiles(values):
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0, "mean": 0.0}
    s = sorted(values)
    n = len(s)
    def q(p):
        idx = int(p * (n - 1))
        return s[idx]
    return {
        "min": s[0],
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "max": s[-1],
        "mean": round(sum(s) / n, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Sequence D17 FP8 Quantization Greedy Divergence Benchmark")
    parser.add_argument("--exe", default=DEFAULT_EXE, help="Path to ninfer-serve.exe")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .ninfer model artifact")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for server instance(s)")
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURES, help="Path to test fixtures JSON")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max completion tokens per prompt")
    parser.add_argument("--spec-backend", choices=["off", "mtp"], default="off", help="Speculative decoding mode (default: off to isolate output head)")
    parser.add_argument("--category", default=None, help="Filter fixtures by category")
    parser.add_argument("--quant-target", choices=["output-head", "token-embedding", "both"], default="output-head",
                        help="Target tensor to quantize to FP8 (output-head, token-embedding, or both)")
    parser.add_argument("--output-json", default=None, help="Path to output summary JSON")
    parser.add_argument("--bf16-url", default=None, help="URL for existing BF16 server instance (e.g. http://127.0.0.1:8160)")
    parser.add_argument("--fp8-url", default=None, help="URL for existing FP8 server instance (e.g. http://127.0.0.1:8161)")
    args = parser.parse_args()

    output_json_path = args.output_json if args.output_json else rf"P:\NInfer\bench\d17\d17_divergence_{args.quant_target.replace('-', '_')}_summary.json"

    print("================================================================================")
    print(f"Sequence D17: FP8 ({args.quant_target}) vs BF16 Greedy Divergence Benchmark")
    print("================================================================================")
    print(f"Model:        {args.model}")
    print(f"Quant Target: {args.quant_target}")
    print(f"Fixtures:     {args.fixtures}")
    print(f"Max Tokens:   {args.max_tokens}")
    print(f"Spec Mode:    {args.spec_backend}")
    print("================================================================================\n")

    prompts = load_fixtures(args.fixtures, args.category)
    print(f"Loaded {len(prompts)} test prompt(s) across categories.\n")

    bf16_results = {}
    fp8_results = {}
    vram_stats = {}

    # -------------------------------------------------------------------------
    # ARM 1: BF16 Baseline
    # -------------------------------------------------------------------------
    if args.bf16_url:
        bf16_url = args.bf16_url
        print(f"[ARM 1: BF16] Connecting to pre-existing server at {bf16_url}...")
    else:
        bf16_url = f"http://127.0.0.1:{args.port}"
        print(f"[ARM 1: BF16 Baseline] Launching unquantized server on port {args.port}...")
        log_out = r"P:\NInfer\bench\d17\server_baseline_bf16.log"
        log_err = r"P:\NInfer\bench\d17\server_baseline_bf16_err.log"
        proc, out_f, err_f = start_server(args.exe, args.model, args.port, is_quantized=False,
                                          quant_target=args.quant_target, spec_backend=args.spec_backend,
                                          stdout_log=log_out, stderr_log=log_err)
        if proc is None:
            print("ERROR: Failed to launch BF16 server.")
            sys.exit(1)
        vram_stats["bf16"] = get_vram_info()
        print(f"  [ARM 1: BF16] Resident VRAM: {vram_stats['bf16']['used_mib']} MiB used ({vram_stats['bf16']['free_mib']} MiB free)")

    print(f"  [ARM 1: BF16] Running {len(prompts)} prompts...")
    for idx, p in enumerate(prompts, 1):
        print(f"    ({idx}/{len(prompts)}) [{p['category']}] {p['id']}...", end=" ", flush=True)
        res = send_chat_completion_streaming(bf16_url, p["system"], p["user"], args.max_tokens)
        if res["success"]:
            print(f"OK ({res['token_count']} tok, {res['tok_per_sec']:.1f} tok/s)")
        else:
            print(f"FAILED: {res.get('error')}")
        bf16_results[p["id"]] = res

    if not args.bf16_url:
        print("  [ARM 1: BF16] Stopping server and draining VRAM...")
        stop_server(proc, out_f, err_f)

    # -------------------------------------------------------------------------
    # ARM 2: FP8 Quantized (output-head / token-embedding / both)
    # -------------------------------------------------------------------------
    if args.fp8_url:
        fp8_url = args.fp8_url
        print(f"\n[ARM 2: FP8 ({args.quant_target})] Connecting to pre-existing server at {fp8_url}...")
    else:
        fp8_url = f"http://127.0.0.1:{args.port}"
        target_name = args.quant_target.replace('-', '_')
        print(f"\n[ARM 2: FP8 ({args.quant_target})] Launching quantized server on port {args.port}...")
        log_out = rf"P:\NInfer\bench\d17\server_quant_{target_name}.log"
        log_err = rf"P:\NInfer\bench\d17\server_quant_{target_name}_err.log"
        proc, out_f, err_f = start_server(args.exe, args.model, args.port, is_quantized=True,
                                          quant_target=args.quant_target, spec_backend=args.spec_backend,
                                          stdout_log=log_out, stderr_log=log_err)
        if proc is None:
            print("ERROR: Failed to launch FP8 server.")
            sys.exit(1)
        vram_stats["fp8"] = get_vram_info()
        print(f"  [ARM 2: FP8] Resident VRAM: {vram_stats['fp8']['used_mib']} MiB used ({vram_stats['fp8']['free_mib']} MiB free)")
        expected_savings = {
            "output-head": "~605 MiB",
            "token-embedding": "~605 MiB",
            "both": "~1,210 MiB (~1.21 GiB)",
        }.get(args.quant_target, "~605 MiB")
        if "bf16" in vram_stats:
            vram_saved = vram_stats["bf16"]["used_mib"] - vram_stats["fp8"]["used_mib"]
            print(f"  [ARM 2: FP8] Measured VRAM Savings: {vram_saved} MiB (expected {expected_savings})")

    print(f"  [ARM 2: FP8] Running {len(prompts)} prompts...")
    for idx, p in enumerate(prompts, 1):
        print(f"    ({idx}/{len(prompts)}) [{p['category']}] {p['id']}...", end=" ", flush=True)
        res = send_chat_completion_streaming(fp8_url, p["system"], p["user"], args.max_tokens)
        if res["success"]:
            print(f"OK ({res['token_count']} tok, {res['tok_per_sec']:.1f} tok/s)")
        else:
            print(f"FAILED: {res.get('error')}")
        fp8_results[p["id"]] = res

    if not args.fp8_url:
        print("  [ARM 2: FP8] Stopping server and draining VRAM...")
        stop_server(proc, out_f, err_f)

    # -------------------------------------------------------------------------
    # DIVERGENCE ANALYSIS
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("DIVERGENCE EVALUATION REPORT")
    print("================================================================================")

    detailed_comparisons = []
    p_div_all = []
    p_div_by_category = {}
    ratios_all = []

    for p in prompts:
        pid = p["id"]
        cat = p["category"]
        bf16_res = bf16_results.get(pid, {})
        fp8_res = fp8_results.get(pid, {})

        if not bf16_res.get("success") or not fp8_res.get("success"):
            print(f"Skipping {pid} due to request failure.")
            continue

        cmp = compare_token_streams(bf16_res["tokens"], fp8_res["tokens"])
        cmp["id"] = pid
        cmp["category"] = cat
        cmp["bf16_tok_s"] = bf16_res["tok_per_sec"]
        cmp["fp8_tok_s"] = fp8_res["tok_per_sec"]
        cmp["bf16_sample"] = bf16_res["text"][:120].replace("\n", "\\n")
        cmp["fp8_sample"] = fp8_res["text"][:120].replace("\n", "\\n")

        detailed_comparisons.append(cmp)
        ratios_all.append(cmp["prefix_match_ratio"])

        if cat not in p_div_by_category:
            p_div_by_category[cat] = {"all_ratios": [], "divergent_p_divs": [], "identical_count": 0, "total": 0}
        p_div_by_category[cat]["total"] += 1
        p_div_by_category[cat]["all_ratios"].append(cmp["prefix_match_ratio"])

        if cmp["p_div"] is not None:
            p_div_all.append(cmp["p_div"])
            p_div_by_category[cat]["divergent_p_divs"].append(cmp["p_div"])
        else:
            p_div_by_category[cat]["identical_count"] += 1

    # Print Table
    header = f"{'Category':<22} | {'Prompt ID':<24} | {'Tokens':<9} | {'P_div':<10} | {'Match%':<8} | {f'First Divergent Token (BF16 vs FP8[{args.quant_target}])'}"
    print(header)
    print("-" * len(header))
    for c in detailed_comparisons:
        tok_str = f"{c['bf16_total_tokens']}/{c['fp8_total_tokens']}"
        p_div_str = str(c["p_div"]) if c["p_div"] is not None else "None (100%)"
        ratio_str = f"{c['prefix_match_ratio']*100:.1f}%"
        if c["p_div"] is not None:
            diff_str = f"{repr(c['first_diff_bf16'])} vs {repr(c['first_diff_fp8'])}"
        else:
            diff_str = "Identical output stream"
        print(f"{c['category']:<22} | {c['id']:<24} | {tok_str:<9} | {p_div_str:<10} | {ratio_str:<8} | {diff_str}")

    total_prompts = len(detailed_comparisons)
    divergent_prompts = len(p_div_all)
    identical_prompts = total_prompts - divergent_prompts
    identical_pct = (identical_prompts / total_prompts * 100) if total_prompts > 0 else 0.0
    mean_match_ratio = (sum(ratios_all) / len(ratios_all) * 100) if ratios_all else 100.0

    quantiles = compute_quantiles(p_div_all)

    print("\n--------------------------------------------------------------------------------")
    print("STATISTICAL SUMMARY ACROSS WORKLOAD CORPUS")
    print("--------------------------------------------------------------------------------")
    print(f"Quantization Target:        {args.quant_target}")
    print(f"Total Evaluated Prompts:    {total_prompts}")
    print(f"Zero-Divergence Prompts:    {identical_prompts} ({identical_pct:.1f}%)")
    print(f"Divergent Prompts:          {divergent_prompts} ({100.0 - identical_pct:.1f}%)")
    print(f"Mean Prefix Match Ratio:    {mean_match_ratio:.2f}%")
    print(f"P_div Distribution (Divergent):")
    print(f"  Min:    {quantiles['min']}")
    print(f"  P25:    {quantiles['p25']}")
    print(f"  Median: {quantiles['median']}")
    print(f"  P75:    {quantiles['p75']}")
    print(f"  Max:    {quantiles['max']}")
    print(f"  Mean:   {quantiles['mean']}")

    category_summaries = {}
    for cat, data in p_div_by_category.items():
        cat_q = compute_quantiles(data["divergent_p_divs"])
        cat_match = (sum(data["all_ratios"]) / len(data["all_ratios"]) * 100) if data["all_ratios"] else 100.0
        category_summaries[cat] = {
            "total_prompts": data["total"],
            "identical_prompts": data["identical_count"],
            "divergent_prompts": len(data["divergent_p_divs"]),
            "zero_divergence_pct": round((data["identical_count"] / data["total"]) * 100, 1) if data["total"] > 0 else 0.0,
            "mean_prefix_match_pct": round(cat_match, 2),
            "p_div_stats": cat_q,
        }

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "quant_target": args.quant_target,
        "max_tokens": args.max_tokens,
        "spec_backend": args.spec_backend,
        "vram_stats": vram_stats,
        "overall": {
            "total_prompts": total_prompts,
            "identical_prompts": identical_prompts,
            "divergent_prompts": divergent_prompts,
            "zero_divergence_pct": round(identical_pct, 2),
            "mean_prefix_match_pct": round(mean_match_ratio, 2),
            "p_div_distribution": quantiles,
        },
        "by_category": category_summaries,
        "comparisons": detailed_comparisons,
    }

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary JSON saved to: {output_json_path}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
