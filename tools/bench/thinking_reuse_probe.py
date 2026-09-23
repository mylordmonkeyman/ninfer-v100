"""Assert multi-turn prefix reuse with reasoning on and off on a live engine."""
import argparse
import json
from pathlib import Path
import urllib.request
import uuid

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--api-key-file", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--echo-reasoning", action="store_true")
p.add_argument("--thinking", choices=("both", "on", "off"), default="both")
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
key = args.api_key_file.read_text().strip()
failures = []
for thinking in ([False, True] if args.thinking == "both" else [args.thinking == "on"]):
    messages = [{"role": "system", "content": f"Probe {uuid.uuid4()}. Reply with only OK. " + "Reference: this is a harmless cache reuse test. " * 300}]
    previous_prompt = 0
    for turn in range(3):
        messages.append({"role": "user", "content": f"Turn {turn+1}. Reply with only OK."})
        body = {"model": "qwen3.8-flash-next", "messages": messages, "temperature": 0,
                "max_tokens": 1024, "enable_thinking": thinking, "preserve_thinking": True,
                "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0}
        req = urllib.request.Request("http://127.0.0.1:8010/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=90) as response:
            raw = json.load(response)
        (args.out / f"{'on' if thinking else 'off'}-{turn+1}.json").write_text(json.dumps({"request": body, "response": raw}, ensure_ascii=False, indent=2), encoding="utf-8")
        usage = raw["usage"]
        prompt = usage["prompt_tokens"]
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        message = raw["choices"][0]["message"]
        print(f"thinking={thinking} turn={turn+1} prompt={prompt} cached={cached} finish={raw['choices'][0]['finish_reason']} reasoning_chars={len(message.get('reasoning_content') or '')}", flush=True)
        if turn and cached < previous_prompt * 0.9:
            failures.append(f"thinking={thinking} turn={turn+1}: cached={cached}, previous_prompt={previous_prompt}")
        echo = {"role": "assistant", "content": message.get("content", "")}
        if args.echo_reasoning:
            echo["reasoning_content"] = message.get("reasoning_content", "")
        messages.append(echo)
        previous_prompt = prompt
if failures:
    raise SystemExit("FAIL: " + "; ".join(failures))
print("PASS: every follow-up reused at least 90% of its preceding prompt.")
