# Sequence D20: Code / Technical / Tool-Call Acceptance vs Workload Confound

## Objective & Hypothesis

Disentangle whether the dramatic speculative draft acceptance discrepancy observed between **Sequence D3 (~87% acceptance)** and **Sequence D12 (8.8% acceptance)** was caused by:
1. **Head Trimming**: Shortlist vocabulary truncation (32k shortlist vs 248k full vocabulary), or
2. **Workload Confound**: Technical English / code continuations (high syntactic predictability) vs conversational Portuguese (dense inflected morphology and vocabulary divergence).

## Workload Fixtures

Located at `bench/d20/fixtures/d20_workload_fixtures.json`:

1. **`code_completion`**: Real C++20 engine codebase continuations from NInfer (`MemoryArena` in `arena.h`, `TensorLayout` in `layout.h`, `launch_recurrent_step` in `gated_delta_net.h`). Evaluates draft acceptance on structured code syntax, templates, namespaces, and standard systems idioms.
2. **`code_explanation`**: Formal English technical prose about runtime/compilers (CUDA Graph capture/replay, Gated DeltaNet associative vs recurrent math, paged KV cache allocation). Serves as the direct positive control matching Sequence D3.
3. **`agentic_tool_json`**: Strict structured JSON tool calls for agentic coding workflows (`read_file`, `replace_file_content`, `run_command`). Evaluates draft acceptance on punctuation, braces, schema keys, and shell strings.
4. **`portuguese_concierge`**: Multi-turn Portuguese restaurant concierge dialogue from Sequence D12 (`Terra e Mar`). Serves as the control anchor reproducing the low-acceptance environment.

## Hard Memory Budget Protocol (Desktop Reserve)

Following the workstation VRAM exhaustion incidents where the desktop compositor (DWM) stalled when VRAM dipped below ~1.5 GiB, all runs adhere to the strict **16 GiB Desktop Reserve Floor**:

```bash
--max-context 32768 --kv-capacity 32768 --max-concurrency 1 --max-private-continuations 16
```

- **Physical Footprint**: Allocates ~74–76 GiB on `sm_120` (96 GB RTX PRO 6000 Blackwell).
- **Free VRAM**: Leaves **$\ge$ 20 GiB free** at all times for the Windows desktop display and OS processes.
- **Concurrency**: $C=1$ is the primary target for pair programming and single-user local inference. Multi-sequence batches ($B > 1$) bypass speculative decoding in Flash-Next by design (`program.cpp:2279`). Concurrency scaling sweeps ($C=8$) require scheduled windows when the operator is away.

## Execution Command

Run the standalone benchmark harness:

```bash
python bench/d20/bench_d20_acceptance.py --port 8160 --repeats 3
```

Optional arguments:
- `--exe`: Path to `ninfer-serve.exe` (default: `P:\NInfer\build-win\apps\Release\ninfer-serve.exe`)
- `--model`: Path to model artifact (default: `E:\models\Qwen3.8-Flash-Next\qwen3_8_flash_next_nvfp4_mtp.ninfer`)
- `--fixtures`: Path to fixtures JSON (default: `P:\NInfer\bench\d20\fixtures\d20_workload_fixtures.json`)
- `--configs`: Concurrency configurations to evaluate (default: `c1`)
- `--out-dir`: Directory for results (default: `P:\NInfer\bench\d20_results`)

## Telemetry Wiring (C++ Fix)

This sequence resolves the telemetry bug where `rec["speculative"]` returned empty values in Flash-Next `request_done` logs.
- `src/targets/qwen3_8_flash_next/impl/program_impl.h`: Added `SpeculativeStats speculative_stats{};` to `struct LaneState`.
- `src/targets/qwen3_8_flash_next/impl/program.cpp`:
  - `start_resource_transaction`: Initialized `st.speculative_stats` with draft window and backend.
  - `decode`: Accurately accumulated `rounds`, `drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position[p]`.
  - `finish` / `abort`: Moved `st.speculative_stats` into `out.speculative` for both catalogued and released dispositions.

## Metric Reporting Triad

Every execution reports the mandatory triad:
1. **Single-Stream Decode Throughput** ($B=1$, tokens/s)
2. **Per-Stream Decode Throughput Under Concurrency** (or explicitly noting $C=1$ budget constraint)
3. **Peak VRAM / Free Desktop VRAM** (MiB)

Alongside the **Position-1 Draft Acceptance Rate** ($100 \times \text{accepted} / \text{proposed}$) and full distribution statistics ($\text{Min}$, $\text{P25}$, $\text{Median}$, $\text{P75}$, $\text{IQR}$, $\text{Max}$, $\text{Std}$).
