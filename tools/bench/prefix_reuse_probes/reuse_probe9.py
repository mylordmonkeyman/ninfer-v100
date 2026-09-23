import json, urllib.request, time

import os
KEY = os.environ["NINFER_KEY"]
URL = "http://localhost:8010/v1/chat/completions"
TOOLS = [{"type":"function","function":{"name":f"t{i}","description":f"tool {i}",
          "parameters":{"type":"object","properties":{"a":{"type":"string"}}}}} for i in range(9)]

filler = ("The quick brown fox jumps over the lazy dog. " * 40)
msgs = [{"role":"system","content":"You are a helpful assistant.\n" + filler*8}]

def turn(user_text, tag):
    msgs.append({"role":"user","content":user_text})
    body = {"model":"qwen3.8-flash-next","messages":msgs,"stream":False,
            "max_tokens":48,"temperature":0.0,"tools":TOOLS,"tool_choice":"auto",
            "chat_template_kwargs":{"enable_thinking":False,"preserve_thinking":True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","User-Agent":"reuse-probe-9tools/1","Authorization":"Bearer "+KEY})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    u = out.get("usage",{})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    txt = out["choices"][0]["message"].get("content") or ""
    print(f"{tag:<10} prompt={u.get('prompt_tokens'):>7}  cached={cached:>7}  "
          f"completion={u.get('completion_tokens'):>5}  {time.time()-t0:5.2f}s")
    msgs.append({"role":"assistant","content":txt})

turn("Say READY and nothing else.", "turn1")
turn("Say ONE and nothing else.",   "turn2")
turn("Say TWO and nothing else.",   "turn3")
turn("Say THREE and nothing else.", "turn4")
