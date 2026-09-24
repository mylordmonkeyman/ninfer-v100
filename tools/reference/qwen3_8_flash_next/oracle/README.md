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
  --incremental-fp32 /srv/ninfer/precision-reference/phase11-14/fp32 \
  --out-dir /path/to/comparison
```

The frozen full-sequence oracle does not contain raw router scores. The
incremental FP32 trace supplies those scores as an explicitly labeled
supplement; its logits must first pass the frozen oracle parity check.

The first real-model 14-position CPU run succeeded on September 24, 2026
([workflow](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36014366398),
[compact report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36015048672)):

| Measurement against independent FP32 oracle | Result |
| --- | ---: |
| Worst incremental FP32 decode KL | 2.99e-11 |
| Storage-profile mean KL | 0.038748 |
| Storage-profile top-1 agreement | 14 / 14 |
| Storage-profile expert-set flips | 82 / 672 layer-position cells |
| Flips in layers 00–19 / 20–47 | 6 / 76 |
| Largest per-position KL (position 13) | 0.340838 |

This supports the precision-compounding hypothesis as a **CPU storage-profile
experiment**, while leaving the V100 cause unresolved. The profile is
uncalibrated: it models more expert-set flips at depth than near the input, but
does not reproduce the observed V100 top-1 flip at position 13. Stage parity
against a V100 candidate trace is required before treating these values as a
precision floor or revising Phase 11 acceptance thresholds.

### V100 calibration result

A single 14-position V100 candidate trace was collected
([GPU workflow](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36015694540),
[compact comparison](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36016286612)).
All 2,744 selected stages were compared with both the CPU profile and the
independent oracle. The GPU diagnostic CTest exited nonzero because the
teacher-forced qualification gate failed; the stage export and comparison
completed successfully.

| Measurement | CPU profile | V100 candidate |
| --- | ---: | ---: |
| Mean logits KL against FP32 oracle | 0.038748 | 0.057752 |
| Top-1 agreement across 14 positions | 14 / 14 | 13 / 14 |
| Expert-set flips across 672 layer-position cells | 82 | 97 |

The router flips overlap in only 57 cells: 25 are CPU-only and 40 are
V100-only. At position 2, the CPU profile has no flips while the V100 has
seven. At position 13, the CPU profile predicts the oracle top-1 token, but
the V100 predicts a different token. The embedding agrees exactly across all
three traces, and the initial layer differences are small; later divergences
depend strongly on position and router trajectory. For example, at position 2
the `L40_mlp_block_input` NRMSE is 0.00354 for CPU versus oracle and 0.04466
for V100 versus oracle; at position 10 those values are 0.21268 and 0.12543.

This **does not establish a precision floor** or exonerate all kernels.
Further work should first explain the missing V100-only flips and the excess
CPU-only flips at the earliest divergent layers, then rerun the same stage
comparison before using the profile to change acceptance thresholds.

### Corrected profile and raw-score comparison

The first CPU experiment above BF16-rounded the GDN QKV projection output.
The selected V100 diagnostic uses `NINFER_FLASH_NEXT_FP32_GDN_PROJECTION=1`
and `NINFER_FLASH_NEXT_FP32_GDN_CONV=1`; the corresponding CPU profile now
feeds the unrounded QKV output into convolution, while preserving its other
materialization boundaries. The revised CPU run
([workflow](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36017907847))
passed incremental FP32 parity at worst KL 2.99e-11 and emitted 3,416 stage
rows and 672 complete router-score vectors. The V100 trace
([workflow](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36018385552))
and [compact score comparison](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36019037272)
used the same 14 frozen positions.

| Measurement against FP32 reference | Corrected CPU profile | V100 candidate |
| --- | ---: | ---: |
| Mean logits KL | 0.0259998 | 0.057752 |
| Top-1 agreement | 14 / 14 | 13 / 14 |
| Expert-set flips / 672 layer-position cells | 54 | 97 |
| Median router-score NRMSE / 672 cells | 0.0003662 | 0.0004753 |
| P95 router-score NRMSE / 672 cells | 0.0086096 | 0.0131782 |

Of the expert-set flips, 39 are shared, 15 occur only on CPU, and 58 only
on V100. The first V100-only flip at position 2, layer 20 exchanges experts
489 and 388, whose FP32 cutoff gap is just 0.0002737; CPU and GPU router-score
RMS errors are 0.001506 and 0.002002 respectively. At position 13, layer 39,
the gap is 0.04668, while CPU and GPU router-score RMS errors are 0.02987
and 0.06394. The median GPU/CPU score-error ratio is 1.196 across cells.
These scores show how near ties amplify divergence but also show that the CPU
profile does not quantitatively match the V100 candidate.

A controlled V100 probe changed only
`NINFER_FLASH_NEXT_FP32_ROUTER_INPUT=1`
([workflow](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36019645369)).
It still flips the position 2, layer 20 expert set, has 93 expert-set flips,
13 / 14 top-1 agreement, and mean KL 0.06444803. The diagnostic CTest
continues to fail its original Phase 11 acceptance gate; the probe workflow
completed successfully and uploaded its trace. This result does not support
BF16 router-input materialization as the dominant source of the mismatch.

The precision-matched reference remains a calibration effort. Before
relaxing the gate, isolate the earliest V100-only score divergence upstream
of the layer 20 near tie and verify it against an independently reproduced
arithmetic path or a closer GPU-matched reference. Neither this CPU profile
nor the router-input probe establishes a mathematical lower bound on KL.

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
