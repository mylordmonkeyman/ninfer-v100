"""Live structured-output qualification through all three public HTTP adapters.

Requires jsonschema for an independent Draft 2020-12 oracle. Uses synthetic content only.
Run against the selected resident artifact; the runner never switches or restarts models.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from jsonschema import Draft202012Validator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key = args.api_key_file.read_text(encoding="utf-8").strip()
    args.out.mkdir(parents=True, exist_ok=False)
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    records = []

    def request(path, body):
        req = urllib.request.Request(args.base_url + path, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=240) as response:
            return response.read().decode("utf-8")

    def run(name, schema, *, stream=False, temperature=0, thinking=False, protocol="chat", limit=256):
        started = time.monotonic()
        record = {"name": name, "schema": schema, "stream": stream, "temperature": temperature, "thinking": thinking, "protocol": protocol}
        prompt = "Return the requested structured result. Use Portuguese text ação when a free string is needed. Follow the response schema."
        body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": limit, "temperature": temperature}
        fmt = {"type": "json_object"} if schema is None else {"type": "json_schema", "name": "result", "strict": True, "schema": schema}
        if protocol == "chat":
            path = "/v1/chat/completions"
            body.update(enable_thinking=thinking, stream=stream)
            body["response_format"] = fmt if schema is None else {"type": "json_schema", "json_schema": {k: v for k, v in fmt.items() if k != "type"}}
        elif protocol == "responses":
            path = "/v1/responses"
            body.pop("messages")
            body.pop("max_tokens")
            body.update(input=prompt, max_output_tokens=limit, text={"format": fmt}, reasoning={"effort": "medium" if thinking else "none"}, store=False)
        else:
            path = "/v1/messages"
            body.update(thinking={"type": "disabled"}, output_config={"format": fmt})
        try:
            raw = request(path, body)
            record["raw"] = raw
            if stream:
                chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
                text = "".join(c["choices"][0]["delta"].get("content", "") or "" for c in chunks if c.get("choices"))
                finish = next(c["choices"][0]["finish_reason"] for c in reversed(chunks) if c.get("choices") and c["choices"][0].get("finish_reason"))
            else:
                result = json.loads(raw)
                if protocol == "chat":
                    text = result["choices"][0]["message"].get("content") or ""
                    finish = result["choices"][0]["finish_reason"]
                elif protocol == "responses":
                    assert result["text"]["format"] == fmt, "response format descriptor lost"
                    text = "".join(c.get("text", "") for item in result["output"] for c in item.get("content", []) if c.get("type") == "output_text")
                    finish = result["status"]
                else:
                    text = "".join(c.get("text", "") for c in result["content"] if c["type"] == "text")
                    finish = result["stop_reason"]
            record.update(text=text, finish=finish)
            if limit == 1:
                assert finish in ("length", "incomplete", "max_tokens"), finish
            else:
                assert finish in ("stop", "completed", "end_turn"), finish
                value = json.loads(text)
                Draft202012Validator(schema or {"type": "object"}).validate(value)
            record["passed"] = True
        except Exception as error:
            record["passed"] = False
            record["error"] = str(error)
            if isinstance(error, urllib.error.HTTPError):
                record["error_body"] = error.read().decode()
        record["elapsed_s"] = time.monotonic() - started
        return record

    schema = {"type": "object", "properties": {"status": {"type": "string", "enum": ["confirmada", "nao_verificada"]}, "sources": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 3}, "minItems": 1, "maxItems": 2}, "note": {"type": ["string", "null"]}}, "required": ["status", "sources", "note"], "additionalProperties": False}
    for protocol in ("chat", "responses", "anthropic"):
        records.append(run(protocol + "-schema", schema, protocol=protocol))
    records.append(run("json-object", None))
    records.append(run("schema-stream", schema, stream=True, temperature=0.8))
    records.append(run("thinking-boundary", {"type": "object", "properties": {"ok": {"const": True}}, "required": ["ok"], "additionalProperties": False}, thinking=True, limit=2048))
    records.append(run("length-prefix", schema, limit=1))
    # Identical prompt, distinct output grammars: exercise concurrent isolation and prefix reuse.
    def isolated(i):
        target = {"const": {"lane": i, "value": "ação"}}
        return run("isolated-" + str(i), target, temperature=0.8 if i % 2 else 0)

    def plain_control():
        record = {"name": "mixed-unconstrained-request"}
        try:
            raw = request("/v1/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": "Write a detailed 400 word explanation of how a sailboat works."}], "max_tokens": 320, "temperature": 0, "enable_thinking": False})
            result = json.loads(raw)
            choice = result["choices"][0]
            record.update(raw=raw, passed=bool(choice["message"].get("content")) and choice["finish_reason"] in ("stop", "length"))
        except Exception as error:
            record.update(passed=False, error=str(error))
            if isinstance(error, urllib.error.HTTPError):
                record["error_body"] = error.read().decode()
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        # The longer text request outlives multiple short constrained lanes in the same backend.
        plain = pool.submit(plain_control)
        records.extend(pool.map(isolated, range(8)))
        records.append(plain.result())
    for i in (8, 9):
        records.append(isolated(i))
    for invalid in ({"type": "string", "pattern": ".*"}, {"type": "object", "required": ["missing"]}, {"default": {"type": "string", "pattern": ".*"}, "$ref": "#/default"}):
        record = {"name": "unsupported-schema", "schema": invalid}
        try:
            request("/v1/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": "test"}], "response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": invalid}}})
            record["passed"] = False
        except urllib.error.HTTPError as error:
            record["passed"] = error.code == 400
            record["error_body"] = error.read().decode()
        records.append(record)
    (args.out / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = [r["name"] for r in records if not r["passed"]]
    print(json.dumps({"total": len(records), "passed": len(records) - len(failed), "failed": failed}))
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
