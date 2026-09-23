"""How many intervening requests does it take to lose an ESTABLISHED checkpoint?

A1,A2 first, so A is provably published and reusing. Then one single turn of another
conversation. Then A again. If A3 misses, one intervening request is enough to destroy a
working checkpoint, i.e. the cache effectively retains only the last conversation.
"""
import json, os, urllib.request

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
                 "User-Agent": "depth-probe/1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    u = out.get("usage", {})
    return (u.get("prompt_tokens", 0),
            (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            out["choices"][0]["message"].get("content") or "")


def convo(seed):
    return [{"role": "system", "content": f"You are assistant #{seed}.\n" + FILLER * 8}]


def turn(msgs, text, tag):
    msgs.append({"role": "user", "content": text})
    p, c, reply = send(msgs)
    msgs.append({"role": "assistant", "content": reply})
    print(f"   {tag:<28} prompt={p:>6}  cached={c:>6}  {'REUSED' if c else 'MISS'}")
    return c


A = convo(3001)
B = convo(3002)
print("establish A")
turn(A, "Say A1.", "A turn1 (cold, expect MISS)")
c2 = turn(A, "Say A2.", "A turn2 (expect REUSED)")
print("one single turn of another conversation")
turn(B, "Say B1.", "B turn1")
print("back to A")
c3 = turn(A, "Say A3.", "A turn3 (the question)")
print("\nand again with no interruption")
c4 = turn(A, "Say A4.", "A turn4 (expect REUSED)")

print("\n=== RESULT ===")
print(f"  A reused before interruption      : {'yes' if c2 else 'no'}")
print(f"  A reused after ONE intervening req: {'yes' if c3 else 'NO  <-- retention depth is 1'}")
print(f"  A reused again once uninterrupted  : {'yes' if c4 else 'no'}")
