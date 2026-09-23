# Sequence D17: Output-Head & Token-Embedding FP8 Quantizers & Greedy Divergence Benchmark

## 1. Overview & Rationale

During the Flash-Next architecture audit, two high-footprint BF16 projection tensors were identified:
1. **Output Head** (`src/targets/qwen3_8_flash_next/impl/load/materialized.cpp:271`):
   - Projects final hidden state to vocabulary logits ($248,320 \times 2,560$).
   - In BF16, occupies **1,271,398,400 bytes (~1.18 GiB)**.
   - In FP8 E4M3 with row-wise float32 scaling, occupies **636,692,480 bytes (~607 MiB)**, saving **~605 MiB** device VRAM.
2. **Token Embedding** (`text.token_embedding`, shape $248,320 \times 2,560$):
   - Maps input token IDs to hidden states across the trunk and MTP stem.
   - Shares the exact same dimension ($248,320 \times 2,560$, **~1.18 GiB** in BF16).
   - In FP8 E4M3 with row-wise float32 scaling, occupies **636,692,480 bytes (~607 MiB)**, saving an additional **~605 MiB** device VRAM.
   - **Combined Net Savings**: **~1,273 MiB (~1.21 GiB)** GPU VRAM reduction when both output head and token embedding are quantized.

### Critical Quality Seams:
- **Output Head**: Projects final hidden states to vocabulary logits. A defect or loss of precision here directly alters emitted token probabilities.
- **Token Embedding**: Feeds the input representations into every single layer and recurrent block. While output head errors affect the final decision boundary, embedding errors propagate through the entire forward pass.
- **Acceptance Contract**: Rather than crude perplexity proxies or uncalibrated loss metrics, acceptance is evaluated through **greedy divergence position ($P_{\text{div}}$)** against the unquantized BF16 baseline across identical prompts and seeds under temperature 0.0.

---

## 2. CLI & Server Interface

The following options have been wired into `ninfer` (CLI) and `ninfer-serve` (REST service), defaulting to `false`/`bf16` (preserving unquantized baseline behavior):

| Flag | Values | Default | Description |
|---|---|---|---|
| `--output-head-fp8` | boolean flag | `false` | Enable FP8 E4M3 row-scaled output head |
| `--no-output-head-fp8` | boolean flag | `false` | Explicitly disable FP8 output head |
| `--output-head-dtype` | `bf16` \| `fp8` | `bf16` | Output head datatype selector |
| `--token-embedding-fp8` | boolean flag | `false` | Enable FP8 E4M3 row-scaled token embedding |
| `--no-token-embedding-fp8` | boolean flag | `false` | Explicitly disable FP8 token embedding |
| `--token-embedding-dtype` | `bf16` \| `fp8` | `bf16` | Token embedding datatype selector |

---

## 3. Greedy Divergence Harness (`bench_output_head_divergence.py`)

The benchmark harness executes paired A/B evaluation across a diverse, curated 14-prompt corpus spanning five distinct workloads:
1. `code_generation`: C++ and Python systems tasks (CUDA reductions, SPSC ring buffer, prefix trie)
2. `technical_reasoning`: Low-level hardware, memory hierarchies, and floating-point representations (FP8 formats, NVLink vs PCIe, cache coherency)
3. `structured_json`: Schema-constrained tool invocation and observability telemetry
4. `mathematics_logic`: Multi-step analytical math, linear algebra, and combinatorics
5. `dialogue_multilingual`: Conversational English and localized Portuguese concierge dialogue

### Metrics Reported:
- **Divergence Position ($P_{\text{div}}$)**: The 1-based index of the first differing token ($t_i^{\text{BF16}} \ne t_i^{\text{FP8}}$). If output streams match completely up to maximum generated length, $P_{\text{div}} = \text{None}$ (100% agreement).
- **Prefix Match Ratio**: Length of identical prefix divided by total generated tokens.
- **First Divergent Token Pair**: Exact token representation from both arms at the point of divergence.
- **$P_{\text{div}}$ Statistical Distribution**: Min, P25, Median, P75, Max, and Mean across all divergent prompts.
- **Empirical VRAM Reduction**: Verified via `nvidia-smi` delta before and after quantization pass.

---

## 4. Execution Commands for Coordinator (Sole GPU Owner)

Under the active GPU stand-down protocol, only the coordinator (`windows:claude:ninfer`) runs GPU processes.

### A. Fully Automated Sequential Runs by Target

The harness supports evaluating any of the three quantization configurations via `--quant-target`:

1. **Evaluate Output Head FP8** (expected ~605 MiB savings):
```powershell
python P:\NInfer\bench\d17\bench_output_head_divergence.py --quant-target output-head --port 8160 --max-tokens 256
```

2. **Evaluate Token Embedding FP8** (expected ~605 MiB savings):
```powershell
python P:\NInfer\bench\d17\bench_output_head_divergence.py --quant-target token-embedding --port 8160 --max-tokens 256
```

3. **Evaluate Both Combined (Output Head + Token Embedding)** (expected ~1.21 GiB savings):
```powershell
python P:\NInfer\bench\d17\bench_output_head_divergence.py --quant-target both --port 8160 --max-tokens 256
```

### B. Evaluating Against Pre-Existing Servers

If the coordinator manages server lifecycles manually:
```powershell
# Server 1 (Baseline BF16):
.\build-win\apps\Release\ninfer-serve.exe E:\models\Qwen3.8-Flash-Next\qwen3_8_flash_next_nvfp4_mtp.ninfer --port 8160 --output-head-dtype bf16 --token-embedding-dtype bf16 --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --max-private-continuations 16 --greedy --no-thinking

# Server 2 (Combined FP8 Head + FP8 Embedding):
.\build-win\apps\Release\ninfer-serve.exe E:\models\Qwen3.8-Flash-Next\qwen3_8_flash_next_nvfp4_mtp.ninfer --port 8161 --output-head-dtype fp8 --token-embedding-dtype fp8 --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --max-private-continuations 16 --greedy --no-thinking

# Run divergence harness pointing to both ports:
python P:\NInfer\bench\d17\bench_output_head_divergence.py --quant-target both --bf16-url http://127.0.0.1:8160 --fp8-url http://127.0.0.1:8161 --max-tokens 256
```

---

## 5. Artifact Output

The benchmark automatically generates:
- `bench/d17/d17_divergence_output_head_summary.json`
- `bench/d17/d17_divergence_token_embedding_summary.json`
- `bench/d17/d17_divergence_both_summary.json`

Each summary contains:
- Complete execution configuration, model path, quant target, and timestamp
- VRAM usage and savings measurements
- Overall statistics ($P_{\text{div}}$ distribution, zero-divergence rate, mean prefix match)
- Category breakdown
- Full per-prompt token streams, timing, and divergence analysis

