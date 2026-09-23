"""Controlled A/B: does INTERLEAVING two conversations destroy reuse, on the same engine?

Arm A: two conversations, strictly interleaved (A,B,A,B,...).
Arm B: the same two conversations, run to completion one after the other.
Identical prompts, identical engine, back to back. If A fails and B passes, interleaving
is the variable and my acceptance failure is real, not a test artifact.
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
                 "User-Agent": "interleave-ctl/1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    u = out.get("usage", {})
    return (u.get("prompt_tokens", 0),
            (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            out["choices"][0]["message"].get("content") or "")


def convo(seed):
    return [{"role": "system", "content": f"You are assistant #{seed}.\n" + FILLER * 8}]


def turn(msgs, text):
    msgs.append({"role": "user", "content": text})
    p, c, reply = send(msgs)
    msgs.append({"role": "assistant", "content": reply})
    return p, c


def report(tag, rows):
    follow = [c for n, (p, c) in enumerate(rows) if n > 0]
    hits = sum(1 for c in follow if c > 0)
    print(f"  {tag}: {hits}/{len(follow)} follow-up turns reused")
    return hits, len(follow)


print("ARM A  interleaved A,B,A,B")
a, b = convo(2001), convo(2002)
ra, rb = [], []
for i in range(4):
    ra.append(turn(a, f"Say A{i}."))
    rb.append(turn(b, f"Say B{i}."))
for (p, c), i in zip(ra, range(4)):
    print(f"   convoA turn{i+1}  prompt={p:>6}  cached={c:>6}")
for (p, c), i in zip(rb, range(4)):
    print(f"   convoB turn{i+1}  prompt={p:>6}  cached={c:>6}")
ha, na = report("A interleaved", ra + rb[:0])
hb, nb = report("B interleaved", rb)

print("\nARM B  sequential, same two conversations restarted")
c1, c2 = convo(2003), convo(2004)
r1, r2 = [], []
for i in range(4):
    r1.append(turn(c1, f"Say A{i}."))
for i in range(4):
    r2.append(turn(c2, f"Say B{i}."))
for (p, c), i in zip(r1, range(4)):
    print(f"   convo1 turn{i+1}  prompt={p:>6}  cached={c:>6}")
for (p, c), i in zip(r2, range(4)):
    print(f"   convo2 turn{i+1}  prompt={p:>6}  cached={c:>6}")
h1, n1 = report("1 sequential", r1)
h2, n2 = report("2 sequential", r2)

print("\n=== VERDICT ===")
print(f"  interleaved follow-ups reused: {ha+hb}/{na+nb}")
print(f"  sequential  follow-ups reused: {h1+h2}/{n1+n2}")
