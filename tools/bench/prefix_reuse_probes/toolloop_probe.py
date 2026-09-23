"""Reproduce Liquid's shape: a tool loop. Turn 1 forces a tool call, turn 2 supplies the tool
result (assistant tool_call + tool message appended), turn 3 again. Does the follow-up reuse?"""
import json, os, urllib.request
URL="http://localhost:8010/v1/chat/completions"; KEY=os.environ["NINFER_KEY"]
TOOLS=[{"type":"function","function":{"name":f"t{i}","description":f"tool {i}","parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}} for i in range(9)]
FILL="The quick brown fox jumps over the lazy dog. "*40
msgs=[{"role":"system","content":"You are a tool-using assistant.\n"+FILL*8}]
def send(tool_choice):
    body={"model":"qwen3.8-flash-next","messages":msgs,"stream":False,"max_tokens":64,"temperature":0.0,
          "tools":TOOLS,"tool_choice":tool_choice,"chat_template_kwargs":{"enable_thinking":True,"preserve_thinking":True}}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY,"User-Agent":"toolloop-probe/1"})
    with urllib.request.urlopen(req,timeout=300) as r: out=json.loads(r.read())
    u=out.get("usage",{}); m=out["choices"][0]["message"]
    return u.get("prompt_tokens",0),(u.get("prompt_tokens_details") or {}).get("cached_tokens",0),out["choices"][0].get("finish_reason"),m
msgs.append({"role":"user","content":"Call tool t3 with a='x'. Then answer DONE."})
for turn in range(1,5):
    p,c,fr,m=send("auto")
    tcs=m.get("tool_calls") or []
    print(f"turn{turn} prompt={p:>5} cached={c:>5} finish={fr} tool_calls={len(tcs)}")
    if tcs:
        msgs.append({"role":"assistant","content":m.get("content") or "","tool_calls":tcs})
        for tc in tcs: msgs.append({"role":"tool","tool_call_id":tc["id"],"content":"ok"})
    else:
        msgs.append({"role":"assistant","content":m.get("content") or ""}); msgs.append({"role":"user","content":"Call tool t3 again with a='y'."})
