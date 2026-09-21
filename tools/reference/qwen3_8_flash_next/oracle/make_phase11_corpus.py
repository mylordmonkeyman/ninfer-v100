#!/usr/bin/env python3
"""Build the deterministic >=4096-position Phase 11 teacher-forced corpus."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, List

from transformers import AutoTokenizer


SECTION_BUDGETS = (
    ("instruction_chat", 768),
    ("code", 768),
    ("ordinary_prose", 768),
    ("reasoning_math", 768),
    ("long_context", 1024),
)


def chat_text(i: int) -> str:
    return (
        "<|im_start|>system\n"
        "You are a careful engineering assistant. Preserve stated constraints and "
        "show calculations when they matter.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Turn {i}: compare two implementation choices for a deterministic inference "
        "service. State assumptions, identify one failure mode, and give a concise "
        "verification step.<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"For turn {i}, I would first keep the storage contract fixed, then isolate "
        "the execution backend. The main failure mode is silently changing numerical "
        "semantics while optimizing. Verification should use a fixed teacher-forced "
        "prefix and compare logits before performance tuning.<|im_end|>\n"
    )


def code_text(i: int) -> str:
    return f"""
// deterministic kernel-planning example {i}
struct TilePlan{i} {{
    int rows;
    int cols;
    int stages;
}};

static float reduce_{i}(const float* x, int n) {{
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {{
        const float v = x[j];
        sum = std::fma(v, 0.5f, sum);
    }}
    return sum;
}}

def reference_{i}(values):
    total = 0.0
    for index, value in enumerate(values):
        total += (index + 1) * value
    return total / max(1, len(values))

# Preserve tie order: higher score first, then lower integer ID.
pairs_{i} = sorted(pairs_{i}, key=lambda item: (-item[0], item[1]))
"""


def prose_text(i: int) -> str:
    return (
        f"Paragraph {i}. A production inference system is constrained by more than "
        "arithmetic throughput. Memory residency, transfer bandwidth, launch latency, "
        "NUMA placement, and the shape of the active workload can each become the "
        "dominant resource. A useful experiment changes one architectural assumption "
        "at a time and records both average behavior and tail behavior. Measurements "
        "should distinguish persistent model storage from temporary workspace, and "
        "critical-path transfers from asynchronous background work. Reproducibility "
        "also matters: a result is much easier to trust when the exact model bytes, "
        "token sequence, build revision, and numerical comparison procedure are fixed. "
    )


def math_text(i: int) -> str:
    a = 17 + (i % 19)
    b = 31 + (i % 23)
    c = 5 + (i % 7)
    return (
        f"Problem {i}. Let a={a}, b={b}, and c={c}. Compute the affine recurrence "
        "x_(n+1) = (a*x_n + b) / c starting from x_0 = 1, and explain how rounding "
        "error could accumulate if each step were evaluated in reduced precision. "
        f"For a bandwidth example, suppose {a} blocks each contain {b * 1024} bytes "
        f"and are consumed {c} times per second. The minimum byte rate is the product "
        "of block count, bytes per block, and frequency. Keep units explicit, convert "
        "bytes/s to GiB/s using 2^30, and compare the result with a hypothetical "
        "10.8 GiB/s link before drawing a conclusion. "
    )


def long_context_text(i: int) -> str:
    anchor = i % 97
    previous = (i * 37 + 11) % 97
    return (
        f"Long-context record {i}. ANCHOR_{anchor:02d} states that subsystem "
        f"{anchor:02d} uses generation tag {1000 + anchor}, checksum family "
        f"C{anchor:02d}, and owner shard {(anchor * 7) % 16}. This statement is an "
        "artificial fact for context-retention testing. Later checks must not replace "
        "it with a nearby record. Cross-reference request: recall ANCHOR_{previous:02d} "
        "and keep its identifier distinct from ANCHOR_"
        f"{anchor:02d}. Transition marker LC_{i:04d} closes this record. "
    )


GENERATORS: dict[str, Callable[[int], str]] = {
    "instruction_chat": chat_text,
    "code": code_text,
    "ordinary_prose": prose_text,
    "reasoning_math": math_text,
    "long_context": long_context_text,
}


def encode_exact_section(tokenizer, name: str, budget: int) -> List[int]:
    out: List[int] = []
    i = 0
    generator = GENERATORS[name]
    while len(out) < budget:
        piece = generator(i)
        ids = tokenizer.encode(piece, add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"tokenizer produced no IDs for section {name}")
        need = budget - len(out)
        out.extend(int(token) for token in ids[:need])
        i += 1
    return out


def token_hash(token_ids: List[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the deterministic Qwen3.8 Flash-Next Phase 11 token corpus"
    )
    parser.add_argument("--model-dir", required=True, help="Local mixed source checkpoint")
    parser.add_argument("--output", required=True, help="Output token_ids.json path")
    parser.add_argument(
        "--positions",
        type=int,
        default=sum(size for _, size in SECTION_BUDGETS),
        help="Required position count; default is the 4096-position qualification corpus",
    )
    args = parser.parse_args()

    default_positions = sum(size for _, size in SECTION_BUDGETS)
    if args.positions != default_positions:
        raise SystemExit(
            f"Phase 11 frozen corpus currently requires exactly {default_positions} positions"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )

    token_ids: List[int] = []
    sections = []
    for name, budget in SECTION_BUDGETS:
        start = len(token_ids)
        section_ids = encode_exact_section(tokenizer, name, budget)
        token_ids.extend(section_ids)
        sections.append(
            {
                "name": name,
                "start_position": start,
                "end_position_exclusive": len(token_ids),
                "positions": budget,
            }
        )

    if len(token_ids) != args.positions:
        raise RuntimeError(
            f"internal corpus length mismatch: {len(token_ids)} != {args.positions}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ninfer.phase11.token_corpus.v1",
        "model_family": "Qwen3.8-Flash-Next",
        "positions": len(token_ids),
        "tokenizer_class": tokenizer.__class__.__name__,
        "token_ids_sha256": token_hash(token_ids),
        "sections": sections,
        "token_ids": token_ids,
    }

    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, output)

    print(f"Wrote {len(token_ids)} Phase 11 token IDs to {output}")
    print(f"token_ids_sha256={payload['token_ids_sha256']}")
    for section in sections:
        print(
            f"{section['name']}: "
            f"[{section['start_position']}, {section['end_position_exclusive']})"
        )


if __name__ == "__main__":
    main()
