"""Validation suite for the 27B under the supervisor, before real traffic returns.

The engine wedged once under production traffic in a way I could not reproduce with
a handful of requests: alive, holding the port and the GPU, answering nothing. So
this is built to catch that specifically -- a health probe runs on its own thread
throughout, independent of the load, and every phase records whether health ever
dropped and whether the supervisor had to restart anything.

Phases, in increasing nastiness:
  1 functional  - does it actually answer correctly across the features in use
  2 concurrency - 1..12 concurrent, i.e. up to and past max_concurrency 8
  3 soak        - sustained mixed load, the phase most likely to reproduce a wedge
  4 abuse       - client disconnects, oversized bodies, bad auth, long context
  5 verdict     - health, restarts, VRAM drift, latency vs the pre-update baseline

Nothing here is a pass just because it returned 200: replies are checked for content,
and a phase that degrades is reported as degraded rather than averaged away.
"""
import json
import pathlib
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import os

# Defaults match the reference workstation deployment; override with the
# environment so this runs against any engine.
BASE = os.environ.get("NINFER_BASE", "http://127.0.0.1:8010")
DASH = os.environ.get("NINFER_DASHBOARD", "http://127.0.0.1:8099")
MODEL = os.environ.get("NINFER_MODEL", "qwen3.8-27b-nvfp4")
_key_file = os.environ.get("NINFER_API_KEY_FILE", "")
KEY = pathlib.Path(_key_file).read_text().strip() if _key_file else os.environ.get("NINFER_API_KEY", "")
AUTH = {"Content-Type": "application/json"}
if KEY:
    AUTH["Authorization"] = f"Bearer {KEY}"

health_samples = []      # (t, status)
stop_monitor = False
results = {}


# ---------------------------------------------------------------- infrastructure
def post(path, payload, timeout=300, headers=None, raw=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=body, headers=headers or AUTH)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.monotonic() - t0


def chat(prompt, max_tokens=200, stream=False, effort=None, extra=None, timeout=300):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": stream}
    if effort:
        payload["reasoning_effort"] = effort
    if extra:
        payload.update(extra)
    if not stream:
        j, secs = post("/v1/chat/completions", payload, timeout=timeout)
        c = j["choices"][0]
        return {"ok": True, "text": (c["message"].get("content") or ""), "secs": secs,
                "usage": j.get("usage", {}), "finish": c.get("finish_reason")}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers=AUTH)
    t0 = time.monotonic()
    ttft, out = None, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                o = json.loads(d)
            except Exception:
                continue
            for ch in o.get("choices", []):
                piece = ch.get("delta", {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.monotonic() - t0
                out.append(piece)
    return {"ok": True, "text": "".join(out), "secs": time.monotonic() - t0, "ttft": ttft}


def health_monitor():
    while not stop_monitor:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=3) as r:
                health_samples.append((time.time(), r.status))
        except Exception:
            health_samples.append((time.time(), 0))
        time.sleep(2)


def engine_pid():
    try:
        with urllib.request.urlopen(
            urllib.request.Request(DASH + "/api/state",
                                   headers={"X-NInfer-Supervisor": "1"}), timeout=5) as r:
            return json.load(r)["engine"].get("pid")
    except Exception:
        return None


def vram_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def banner(text):
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}", flush=True)


# ------------------------------------------------------------------- phase 1
def phase_functional():
    banner("PHASE 1  functional correctness")
    checks = []

    r = chat("What is 17 multiplied by 23? Reply with only the number.", 400)
    checks.append(("arithmetic", "391" in r["text"], r["text"][:60]))

    r = chat("Reply with exactly the word: PING", 300)
    checks.append(("instruction following", "PING" in r["text"].upper(), r["text"][:60]))

    r = chat("List exactly three primary colours, comma separated, nothing else.", 400)
    txt = r["text"].lower()
    checks.append(("factual list", sum(c in txt for c in ("red", "blue", "yellow")) >= 2, r["text"][:60]))

    r = chat("Write a Python function that returns the nth Fibonacci number.", 600)
    checks.append(("code generation", "def " in r["text"], r["text"][:60].replace("\n", " ")))

    r = chat("Count from 1 to 10.", 400, stream=True)
    checks.append(("streaming", "10" in r["text"] and r.get("ttft") is not None,
                   f"ttft={r.get('ttft')}"))

    # Multi-turn, and whether prefix reuse fires at all.
    #
    # The system prompt carries a unique marker deliberately. With a fixed one,
    # this check is decided by catalog luck rather than by the mechanism: after a
    # soak leaves hundreds of conversations sharing a prefix, the lookup can match
    # another conversation's session endpoint, which is captured AFTER generation
    # and is therefore longer than this prompt and unusable. That reads as "reuse
    # is broken" when reuse is fine. A unique prefix asks the question actually
    # intended -- does turn 2 reuse turn 1 of THIS conversation.
    marker = f"session-{time.time_ns():x}"
    msgs = [{"role": "system", "content": f"You are terse. Session {marker}."},
            {"role": "user", "content": "Say ONE."}]
    j, _ = post("/v1/chat/completions",
                {"model": MODEL, "messages": msgs, "max_tokens": 300, "temperature": 0.0})
    a1 = j["choices"][0]["message"].get("content") or ""
    msgs += [{"role": "assistant", "content": a1}, {"role": "user", "content": "Say TWO."}]
    j2, _ = post("/v1/chat/completions",
                 {"model": MODEL, "messages": msgs, "max_tokens": 300, "temperature": 0.0})
    cached = j2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    checks.append(("multi-turn", bool(j2["choices"][0]["message"].get("content") is not None), ""))
    checks.append(("prefix reuse fires", cached > 0, f"cached_tokens={cached}"))

    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<24} {detail}", flush=True)
    results["functional"] = all(ok for _, ok, _ in checks)
    return checks


# ------------------------------------------------------------------- phase 2
def phase_concurrency():
    banner("PHASE 2  concurrency ramp (max_concurrency is 8)")
    table = []
    for n in (1, 2, 4, 8, 12):
        out = [None] * n
        threads = []
        t0 = time.monotonic()

        def one(i):
            try:
                out[i] = chat(f"Count from 1 to 20. Task {i}.", 300, stream=True, timeout=300)
            except Exception as e:
                out[i] = {"ok": False, "err": f"{type(e).__name__}: {e}"}

        for i in range(n):
            th = threading.Thread(target=one, args=(i,))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        wall = time.monotonic() - t0
        good = [o for o in out if o and o.get("ok")]
        ttfts = [o["ttft"] for o in good if o.get("ttft")]
        ok = len(good) == n
        table.append((n, len(good), wall, statistics.median(ttfts) if ttfts else 0, ok))
        print(f"  {n:>2} concurrent: {len(good)}/{n} ok  wall={wall:6.1f}s  "
              f"ttft_med={statistics.median(ttfts) if ttfts else 0:5.2f}s  {'PASS' if ok else 'FAIL'}",
              flush=True)
        for o in out:
            if o and not o.get("ok"):
                print(f"      failure: {o.get('err')}", flush=True)
    results["concurrency"] = all(row[4] for row in table)
    return table


# ------------------------------------------------------------------- phase 3
def phase_soak(minutes=12):
    banner(f"PHASE 3  sustained soak, {minutes} minutes (the wedge hunt)")
    deadline = time.monotonic() + minutes * 60
    sent = ok = fail = 0
    ttfts, errors = [], []
    vram_start = vram_used_mib()
    round_no = 0
    while time.monotonic() < deadline:
        round_no += 1
        n = 4
        out = [None] * n
        threads = []

        def one(i):
            nonlocal_prompt = ("Summarise in two sentences why testing matters. "
                               f"Round {round_no} task {i}. " + "Filler context. " * (40 * (i + 1)))
            try:
                out[i] = chat(nonlocal_prompt, 250, stream=True, timeout=300)
            except Exception as e:
                out[i] = {"ok": False, "err": f"{type(e).__name__}: {e}"}

        for i in range(n):
            th = threading.Thread(target=one, args=(i,))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        for o in out:
            sent += 1
            if o and o.get("ok") and o.get("text"):
                ok += 1
                if o.get("ttft"):
                    ttfts.append(o["ttft"])
            else:
                fail += 1
                errors.append((round_no, (o or {}).get("err", "empty reply")))
        if round_no % 5 == 0:
            recent_bad = sum(1 for _, s in health_samples[-30:] if s != 200)
            print(f"  round {round_no:>3}  sent={sent:>4} ok={ok:>4} fail={fail:>3}  "
                  f"ttft_med={statistics.median(ttfts) if ttfts else 0:5.2f}s  "
                  f"vram={vram_used_mib()}MiB  health_bad_recent={recent_bad}", flush=True)
    vram_end = vram_used_mib()
    print(f"  soak done: {ok}/{sent} ok, {fail} failed", flush=True)
    if errors:
        for r, e in errors[:8]:
            print(f"      round {r}: {e}", flush=True)
    print(f"  VRAM {vram_start} -> {vram_end} MiB (drift {(vram_end or 0)-(vram_start or 0):+d})", flush=True)
    results["soak"] = fail == 0
    results["soak_stats"] = {"sent": sent, "ok": ok, "fail": fail,
                             "ttft_med": statistics.median(ttfts) if ttfts else 0,
                             "vram_drift": (vram_end or 0) - (vram_start or 0)}
    return results["soak_stats"]


# ------------------------------------------------------------------- phase 4
def phase_abuse():
    banner("PHASE 4  abuse and edge cases")
    checks = []

    # Bad auth must be refused, not crash anything.
    try:
        req = urllib.request.Request(BASE + "/v1/models",
                                     headers={"Authorization": "Bearer wrong"})
        urllib.request.urlopen(req, timeout=10)
        checks.append(("bad auth rejected", False, "got 200"))
    except urllib.error.HTTPError as e:
        checks.append(("bad auth rejected", e.code == 401, f"HTTP {e.code}"))

    # Client disconnect mid-stream: the engine must survive it.
    try:
        payload = {"model": MODEL, "messages": [{"role": "user", "content": "Count to 500 slowly."}],
                   "max_tokens": 2000, "temperature": 0.0, "stream": True}
        req = urllib.request.Request(BASE + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers=AUTH)
        r = urllib.request.urlopen(req, timeout=60)
        r.read(200)
        r.close()  # hang up early, on purpose
        time.sleep(3)
        checks.append(("survives client disconnect", True, "closed mid-stream"))
    except Exception as e:
        checks.append(("survives client disconnect", False, str(e)[:60]))

    # Malformed JSON must be a 400, not a wedge.
    try:
        req = urllib.request.Request(BASE + "/v1/chat/completions", data=b"{not json",
                                     headers=AUTH)
        urllib.request.urlopen(req, timeout=20)
        checks.append(("malformed JSON rejected", False, "accepted"))
    except urllib.error.HTTPError as e:
        checks.append(("malformed JSON rejected", 400 <= e.code < 500, f"HTTP {e.code}"))
    except Exception as e:
        checks.append(("malformed JSON rejected", False, str(e)[:60]))

    # Unknown model must 404 rather than serve something else.
    try:
        post("/v1/chat/completions", {"model": "does-not-exist",
                                      "messages": [{"role": "user", "content": "hi"}],
                                      "max_tokens": 10}, timeout=30)
        checks.append(("unknown model rejected", False, "accepted"))
    except urllib.error.HTTPError as e:
        checks.append(("unknown model rejected", 400 <= e.code < 500, f"HTTP {e.code}"))

    # A genuinely long prompt: well within 262144 context but far past a chunk.
    try:
        big = "This is a filler sentence for a long context test. " * 1200  # ~12k tokens
        r = chat(big + "\n\nReply with exactly: LONGCTX", 300, timeout=600)
        checks.append(("long context (~12k tok)", "LONGCTX" in r["text"].upper(),
                       f"prompt_tokens={r['usage'].get('prompt_tokens')}"))
    except Exception as e:
        checks.append(("long context (~12k tok)", False, str(e)[:70]))

    # Still alive and serving after all of that.
    try:
        r = chat("Reply with exactly: STILL ALIVE", 300, timeout=120)
        checks.append(("serving after abuse", "ALIVE" in r["text"].upper(), r["text"][:40]))
    except Exception as e:
        checks.append(("serving after abuse", False, str(e)[:60]))

    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<28} {detail}", flush=True)
    results["abuse"] = all(ok for _, ok, _ in checks)
    return checks


# ------------------------------------------------------------------- verdict
def phase_verdict(pid_before):
    banner("VERDICT")
    bad = [s for _, s in health_samples if s != 200]
    pid_after = engine_pid()
    print(f"  health probes      : {len(health_samples)} taken, {len(bad)} not-200")
    print(f"  engine pid         : {pid_before} -> {pid_after} "
          f"({'UNCHANGED, no restart' if pid_before == pid_after else 'RESTARTED during testing'})")
    for k in ("functional", "concurrency", "soak", "abuse"):
        if k in results:
            print(f"  {k:<18} : {'PASS' if results[k] else 'FAIL'}")
    st = results.get("soak_stats", {})
    if st:
        print(f"  soak               : {st['ok']}/{st['sent']} ok, ttft_med={st['ttft_med']:.2f}s, "
              f"VRAM drift {st['vram_drift']:+d} MiB")
    everything = all(results.get(k, False) for k in ("functional", "concurrency", "soak", "abuse"))
    clean = not bad and pid_before == pid_after
    print(f"\n  OVERALL: {'PASS' if everything and clean else 'NEEDS ATTENTION'}")
    if bad:
        print(f"    health dropped {len(bad)} times during testing -- investigate before traffic")
    if pid_before != pid_after:
        print("    the supervisor restarted the engine during testing -- something wedged or crashed")
    return everything and clean


def main():
    minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    pid_before = engine_pid()
    print(f"engine pid at start: {pid_before}", flush=True)
    th = threading.Thread(target=health_monitor, daemon=True)
    th.start()
    try:
        phase_functional()
        phase_concurrency()
        phase_soak(minutes)
        phase_abuse()
    finally:
        global stop_monitor
        stop_monitor = True
        time.sleep(0.3)
    ok = phase_verdict(pid_before)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
