"""Does flipping enable_thinking between turns of ONE conversation lose the prefix?
Turns 1-2 thinking off, turns 3-4 thinking on, then turn 5 off again. Everything else fixed."""
import json, os, urllib.request
URL="http://localhost:8010/v1/chat/completions"; KEY=os.environ["NINFER_KEY"]
TOOLS=[{"type":"function","function":{"name":f"t{i}","description":f"tool {i}","parameters":{"type":"object","properties":{"a":{"type":"string"}}}}} for i in range(9)]
FILL="The quick brown fox jumps over the lazy dog. "*40
msgs=[{"role":"system","content":"You are a helpful assistant.\n"+FILL*8}]
def turn(text, think):
    msgs.append({"role":"user","content":text})
    body={"model":"qwen3.8-flash-next","messages":msgs,"stream":False,"max_tokens":48,"temperature":0.0,
          "tools":TOOLS,"tool_choice":"auto","chat_template_kwargs":{"enable_thinking":think,"preserve_thinking":True}}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY,"User-Agent":"thinking-toggle-probe/1"})
    with urllib.request.urlopen(req,timeout=300) as r: out=json.loads(r.read())
    u=out.get("usage",{}); m=out["choices"][0]["message"]
    print(f"  thinking={str(think):<5} prompt={u.get('prompt_tokens'):>5} cached={(u.get('prompt_tokens_details') or {}).get('cached_tokens',0):>5}")
    msgs.append({"role":"assistant","content":m.get("content") or ""})
turn("Say ONE.",False); turn("Say TWO.",False); turn("Say THREE.",True); turn("Say FOUR.",True); turn("Say FIVE.",False)
