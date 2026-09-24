# Qwen3.8-Flash-Next Oracle & State Diagnostic Harness

This directory provides the authoritative CPU FP32 reference forward pass and state divergence analysis harness for Qwen3.8-Flash-Next.

## Phase 11 sequential storage-profile experiment

`run_precision_reference.py` is a CPU-only supplementary diagnostic. It first
decodes the frozen tokens with FP32 state and requires agreement with the
independent full-sequence oracle. It then repeats sequential decoding with
explicit BF16 materialization for selected projections, attention and MLP
inputs, expert intermediate values, convolution history, attention KV, and
logits, while retaining FP32 hyper and GDN SSM state. It
writes position-indexed FP32 stage dumps and KL/top-1 summaries.

The experiment is **not** an exact V100 precision oracle or a proven lower
bound. It currently computes CPU FP32 arithmetic between materialization
boundaries, and does not reproduce the reduction order of FP8 GEMV or fused
GDN readout. Only stage agreement with V100 can justify attributing final
errors to the precision profile. The authoritative mathematical oracle remains
`run_oracle.py`.

With the existing oracle environment and read-only source assets, the small
14-position experiment is invoked as follows:

```bash
python run_precision_reference.py \
  --model-dir /srv/ninfer/source/mixed \
  --ple-dir /srv/ninfer/source/ple/ples_int4 \
  --ids-file /srv/ninfer/oracle/phase11/token_ids.json \
  --fp32-oracle /srv/ninfer/oracle/phase11-stage-trace14 \
  --positions 14 \
  --out-dir /srv/ninfer/precision-reference/phase11-14
```

The V100 test can optionally write pre-injection selected tensors using
`NINFER_PHASE11_CANDIDATE_TRACE_ROOT`, with the existing all-position stage
trace enabled. Compare the three sets with:

```bash
python compare_precision_traces.py \
  --oracle /srv/ninfer/oracle/phase11-stage-trace14 \
  --cpu /srv/ninfer/precision-reference/phase11-14/v100-phase11-storage \
  --v100 /path/to/candidate-trace \
  --out-dir /path/to/comparison
```

Until a real-model run establishes incremental FP32 parity and V100 stage
agreement, these scripts produce diagnostic hypotheses rather than Phase 11
qualification results.

## 1. Environment Setup

Run the setup script using Python 3.14 to create the isolated virtual environment:

```powershell
.\setup_env.ps1
```

This creates the isolated venv at `E:\NInfer\venv-qwen4exp` with PyTorch (CPU), safetensors, numpy, and transformers.

## 2. Generating State Dumps

### Deliverable A (NInfer C++ Engine Dump)
Generate raw stage tensor dumps for single token or prompt execution using the reference tool:

- **Single Token Execution (e.g. `<|im_start|>` token 248045):**
  ```powershell
  $env:PATH = "P:\third_party\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;" + $env:PATH
  .\build-win\tools\reference\qwen3_8_flash_next\ninfer_qwen3_8_flash_next_reference.exe `
    -m E:\NInfer\qwen3_8_flash_next.ninfer `
    --execute-token --token-id 248045 `
    --dump-states P:\dumps\ninfer_tok0
  ```

- **Chat Diagnostic Prompt Prefill:**
  ```powershell
  .\build-win\tools\reference\qwen3_8_flash_next\ninfer_qwen3_8_flash_next_reference.exe `
    -m E:\NInfer\qwen3_8_flash_next.ninfer `
    --chat-diagnostic --prompt "Hello" `
    --dump-states P:\dumps\ninfer_chat
  ```

### Deliverable B (Python CPU Oracle Dump)
Generate reference tensors for the same token(s):

```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe run_oracle.py `
  --token-id 248045 `
  --dump-states P:\dumps\oracle_tok0
```

For multi-token sequence:
```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe run_oracle.py `
  --ids "248045,846,198,20206" `
  --dump-states P:\dumps\oracle_seq
```

### Phase 11 logits-only acceptance oracle

For the frozen >=4096-position Phase 11 teacher-forced oracle, use the dedicated generator rather than `--dump-states`. The generator creates a deterministic 4096-position corpus with explicit instruction/chat, code, ordinary prose, reasoning/math, and long-context ranges; runs the independent CPU FP32 oracle without intermediate-state hooks; validates every logits file; and only then publishes the finished artifact.

On the physical V100 host, from a checkout containing these oracle tools:

```bash
tools/reference/qwen3_8_flash_next/oracle/generate_phase11_oracle.sh
```

Defaults are the pinned host assets:

```text
Python: /home/$USER/ninfer-v100/venv-oracle/bin/python
mixed:  /srv/ninfer/source/mixed
PLE:    /srv/ninfer/source/ple/ples_int4
output: /srv/ninfer/oracle/phase11
```

The final directory contains `token_ids.json`, `manifest.json`, one FP32 logits file per teacher-forced position, and `provenance.json`. The validator requires contiguous positions, exact token-ID agreement with the frozen corpus, one FP32 logits tensor per position, and exact file sizes.

If a previously published `/srv/ninfer/oracle/phase11` must intentionally be replaced, rerun with `--replace`; the old artifact is renamed to a timestamped backup before the new one is published.

For lower-level use, the equivalent manual oracle invocation is:

```bash
~/ninfer-v100/venv-oracle/bin/python \
  tools/reference/qwen3_8_flash_next/oracle/run_oracle.py \
  --model-dir /srv/ninfer/source/mixed \
  --ple-dir /srv/ninfer/source/ple/ples_int4 \
  --ids-file /srv/ninfer/oracle/phase11/token_ids.json \
  --dump-logits /srv/ninfer/oracle/phase11 \
  --logits-chunk-size 8
```

Keep the existing `--dump-states` mode for small detailed divergence investigations; it is intentionally not the full-oracle format.

### MTP reference qualification

`mtp_reference.py` implements the MTP stem and wraps one Transformers decoder layer,
following the [vLLM MTP implementation](https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/nvidia/mtp.py).
The hidden normalization covers all 10,240 features, its shared projection preserves
four distinct streams, and the sole mixer follows the decoder layer. The first draft
pairs the last target hidden with the target's next token; later drafts carry MTP hidden.

Run the focused comparison from the repository root with the environment above:

```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe tests/targets/qwen3_8_flash_next/test_mtp_oracle_parity.py `
  --ninfer-exe build-win/tests/Release/ninfer_qwen3_8_flash_next_mtp_test.exe `
  --dump-dir profiles/bench/flash-next-mtp/oracle-parity
```

This compares the five stem stages against the FP32 reference under shared synthetic
inputs and weights, and checks the remaining C++ stages for finite, nonzero values.
It does not compare decoder-layer or head values: their synthetic weights differ.
Zero indexer counts are legitimate diagnostics. `--stem-only` runs the pure tensor
semantics check; full mode fails explicitly if Qwen4Exp Transformers is unavailable.

The September 6 C++ comparison passed with maximum stem relative L2 error 0.002428.
The synthetic Transformers MTP forward also ran successfully. `run_oracle.py --mtp-real`
supports real-weight teacher seeding and chained drafts, but full real-checkpoint CPU
MTP parity has not been run and is not implied by those focused checks.

## 3. Comparing States and Finding First Divergence

Run `compare_states.py` to compare stage tensors across positions:

```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe compare_states.py `
  P:\dumps\ninfer_tok0 `
  P:\dumps\oracle_tok0 `
  --threshold 0.05
```

The script reports $\max |d|$, $\text{rel-L2}$, and $\text{cosine}$ similarity per stage and immediately isolates the exact layer and operator where the numerical divergence starts.
