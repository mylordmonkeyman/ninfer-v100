"""Collect matched Tribuno outputs for manual review; never grades answer text."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--api-key-file", type=Path, required=True)
    p.add_argument("--reasoning", choices=("none", "low", "medium", "xhigh"), default="none")
    p.add_argument("--case-id", action="append", help="Repeat to select named corpus cases")
    p.add_argument("--target", action="append", choices=("flash-next", "vllm-27b", "ollama-27b"))
    p.add_argument("--max-output-tokens", type=int, help="Override the total reasoning plus answer allowance")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    trials = [json.loads(s) for s in (args.corpus_run / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = [{"id": t["caseId"], "messages": t["calls"][0]["messages"], "max_tokens": t["calls"][0]["options"]["maxTokens"]}
             for t in trials if t["repeat"] == 1 and t["concurrency"] == 1 and t["calls"]]
    assert len(cases) == 32 and len({c["id"] for c in cases}) == 32
    if args.case_id:
        unknown = set(args.case_id) - {c["id"] for c in cases}
        if unknown:
            p.error(f"Unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in args.case_id]
    if args.max_output_tokens is not None:
        if args.max_output_tokens < 1:
            p.error("--max-output-tokens must be positive")
        cases = [{**c, "max_tokens": args.max_output_tokens} for c in cases]
    key = args.api_key_file.read_text(encoding="utf-8").strip()
    targets = [
        {"id": "flash-next", "base": "http://127.0.0.1:8010", "model": "qwen3.8-flash-next", "api": "ninfer"},
        {"id": "vllm-27b", "base": "http://z590-vision-d:18020", "model": "qwen3.8-27b", "api": "vllm"},
        {"id": "ollama-27b", "base": "http://z690-ex-glacial-win:11434", "model": "qwen3.8:27b-mtp-q4_K_M", "api": "ollama"},
    ]
    if args.target:
        targets = [t for t in targets if t["id"] in args.target]
    def http(target, path, body=None):
        headers = {"Content-Type": "application/json"}
        if target["api"] == "ninfer":
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(target["base"] + path, None if body is None else json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=900) as response:
            return json.load(response)

    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "targets": targets,
                "corpus_run": str(args.corpus_run), "case_count": len(cases),
                "prompts_sha256": hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                "reasoning": args.reasoning,
                "max_output_tokens_override": args.max_output_tokens,
                "comparison": f"One fresh response per selected model-facing case per endpoint; concurrency 1 per endpoint. Captured Tribuno prompts, recorded output budgets, greedy temperature=0, reasoning={args.reasoning}, JSON object mode, neutral presence/frequency/repetition penalties. No retries, no fallback, no automatic semantic grades.",
                "limitations": "Deployment comparison includes model, quantization, tokenizer/template, and engine effects. One sample per case does not establish a population error rate. Two no-call application guards excluded."}
    write(args.out / "manifest.json", manifest)
    write(args.out / "prompts.json", cases)
    write(args.out / "sources.json", json.loads((args.corpus_run / "cases.json").read_text(encoding="utf-8")))

    def collect(target):
        path = args.out / target["id"]
        path.mkdir()
        inventory_path = "/api/tags" if target["api"] == "ollama" else "/v1/models"
        write(path / "inventory-before.json", http(target, inventory_path))
        if target["api"] == "ollama":
            write(path / "model-show.json", http(target, "/api/show", {"model": target["model"]}))
        for case in cases:
            if target["api"] == "ollama":
                body = {"model": target["model"], "messages": case["messages"], "stream": False, "think": {"none": False, "low": "low", "medium": "medium", "xhigh": "max"}[args.reasoning], "format": "json",
                        "options": {"temperature": 0, "num_predict": case["max_tokens"], "num_ctx": 32768, "repeat_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0}}
                route = "/api/chat"
            else:
                body = {"model": target["model"], "messages": case["messages"], "stream": False, "temperature": 0,
                        "max_tokens": case["max_tokens"], "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1,
                        "response_format": {"type": "json_object"}}
                if args.reasoning != "none":
                    body.update({"reasoning_effort": args.reasoning} if target["api"] == "ninfer" else {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": args.reasoning}})
                else:
                    body.update({"enable_thinking": False} if target["api"] == "ninfer" else {"chat_template_kwargs": {"enable_thinking": False}})
                route = "/v1/chat/completions"
            record = {"case_id": case["id"], "target": target["id"], "request": body}
            started = time.monotonic()
            try:
                raw = http(target, route, body)
                record["response"] = raw
                if target["api"] == "ollama":
                    record.update(text=raw["message"].get("content", ""), reasoning=raw["message"].get("thinking", ""), finish_reason=raw.get("done_reason"), input_tokens=raw.get("prompt_eval_count"), output_tokens=raw.get("eval_count"))
                else:
                    choice = raw["choices"][0]
                    record.update(text=choice["message"].get("content") or "", reasoning=choice["message"].get("reasoning_content") or choice["message"].get("reasoning") or "", finish_reason=choice.get("finish_reason"), input_tokens=raw.get("usage", {}).get("prompt_tokens"), output_tokens=raw.get("usage", {}).get("completion_tokens"))
            except Exception as error:
                record["error"] = str(error)
                if isinstance(error, urllib.error.HTTPError):
                    record["error_body"] = error.read().decode()
            record["elapsed_s"] = time.monotonic() - started
            write(path / (case["id"] + ".json"), record)
            print(target["id"], case["id"], record.get("finish_reason", record.get("error")), round(record["elapsed_s"], 2), flush=True)
        write(path / "inventory-after.json", http(target, inventory_path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(collect, target) for target in targets]:
            future.result()
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
