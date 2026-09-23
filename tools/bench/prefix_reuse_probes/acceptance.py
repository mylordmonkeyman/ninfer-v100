"""Independent client-side acceptance for the checkpoint-pool fix.

Astra proved 24 fresh conversations reuse. This tests the shape that actually broke for
Igor: a conversation that STARTS AFTER the pool is already saturated, plus a return to a
conversation old enough to have been evicted, plus interleaving (his real traffic runs
several clients at once). Anything that wedges the pool shows up as reuse never returning.
"""
import json, os, time, urllib.request

URL = "http://localhost:8010/v1/chat/completions"
KEY = os.environ["NINFER_KEY"]
TOOLS = [{"type": "function", "function": {"name": f"t{i}", "description": f"tool {i}",
          "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}
         for i in range(4)]
FILLER = "The quick brown fox jumps over the lazy dog. " * 40


def send(msgs):
    body = {"model": "qwen3.8-flash-next", "messages": msgs, "stream": False,
            "max_tokens": 24, "temperature": 0.0, "tools": TOOLS, "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY,
                 "User-Agent": "acceptance-probe/1"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    u = out.get("usage", {})
    return (u.get("prompt_tokens", 0),
            (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            out["choices"][0]["message"].get("content") or "",
            time.time() - t0)


def new_convo(seed):
    # A distinct system prompt per conversation so each one is a genuinely separate owner.
    return [{"role": "system", "content": f"You are assistant #{seed}.\n" + FILLER * 8}]


def turn(msgs, text):
    msgs.append({"role": "user", "content": text})
    p, c, reply, dt = send(msgs)
    msgs.append({"role": "assistant", "content": reply})
    return p, c, dt


N_FILL = 20
print(f"PHASE 1  saturate the pool with {N_FILL} distinct conversations, 2 turns each")
convos = []
for i in range(N_FILL):
    m = new_convo(i)
    turn(m, "Say A.")
    p, c, dt = turn(m, "Say B.")
    convos.append(m)
    if i in (0, 1, N_FILL - 2, N_FILL - 1):
        print(f"   convo {i:>2} turn2  prompt={p:>6}  cached={c:>6}  {dt:5.2f}s")
print(f"   ...{N_FILL} conversations created\n")

print("PHASE 2  brand-new conversation started AFTER saturation (the Liquid case)")
m = new_convo(999)
ok2 = True
for n, t in enumerate(["Say ONE.", "Say TWO.", "Say THREE.", "Say FOUR."], start=1):
    p, c, dt = turn(m, t)
    flag = ""
    if n > 1:
        if c == 0:
            flag = "   <-- FAIL, no reuse"
            ok2 = False
        else:
            flag = f"   reuse {100*c/p:.0f}%"
    print(f"   turn{n}  prompt={p:>6}  cached={c:>6}  {dt:5.2f}s{flag}")

print("\nPHASE 3  return to conversation 0, long since evicted")
p, c, dt = turn(convos[0], "Say C.")
print(f"   prompt={p:>6}  cached={c:>6}  {dt:5.2f}s"
      f"   ({'reused' if c else 'evicted, re-prefilled (expected, not a fault)'})")
p, c, dt = turn(convos[0], "Say D.")
ok3 = c > 0
print(f"   follow-up  prompt={p:>6}  cached={c:>6}  {dt:5.2f}s"
      f"   {'republished and reused' if ok3 else '<-- FAIL, cannot republish'}")

print("\nPHASE 4  interleave 3 conversations, as two clients would")
ok4 = True
inter = [new_convo(1000 + i) for i in range(3)]
for m in inter:
    turn(m, "Say X.")
for rnd in range(2):
    for i, m in enumerate(inter):
        p, c, dt = turn(m, f"Say Y{rnd}.")
        if c == 0:
            ok4 = False
        print(f"   round{rnd} convo{i}  prompt={p:>6}  cached={c:>6}  {dt:5.2f}s")

print("\n=== ACCEPTANCE ===")
print(f"  new conversation after saturation reuses : {'PASS' if ok2 else 'FAIL'}")
print(f"  evicted conversation can republish       : {'PASS' if ok3 else 'FAIL'}")
print(f"  interleaved conversations all reuse      : {'PASS' if ok4 else 'FAIL'}")
print(f"  OVERALL: {'PASS' if (ok2 and ok3 and ok4) else 'FAIL'}")
