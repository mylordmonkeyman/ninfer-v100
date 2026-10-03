# V100 27B forward port

The `forwardport/v100-flash-next` branch restores the dense 27B SM70 operation paths from
`geoffwatts/ninfer-v100` at `b37d0dd3e1163b9d802d8bccfa89918bf68d793e`, while retaining the
current branch's Qwen3.8 Flash-Next implementation. This is a source port; runtime qualification
is recorded separately below.

## Implementation

- Build the complete public Engine operation graph with CUDA 12.8 and SM70, including the
  CLI, server, scoring and benchmark targets. CUTLASS is pinned to v4.4.2; the original fork's
  vendored llama.cpp attention implementation retains its license.
- Reorder 27B NVFP4 code and scale planes and row-scaled FP8 code planes at load time into
  Volta QPN layout. Token embeddings retain their checkpoint layout. The model file is unchanged.
- Use the working fork's FP16 tensor-core paths for wide prefill and QPN paths for small
  batches and decode, including shared activation staging for paired FP8 projections.
- Restore the fused projection, SwiGLU, residual, output selection, recurrent-state and
  BF16/INT8/FP8 attention implementations needed by the dense runtime.
- Keep Flash-Next's FP8 weights with FP32 row scales on their existing separate implementation;
  27B weights with BF16 row scales use the restored prepacked implementation. Retain Flash-Next's
  registered BF16 shapes, portable NVFP4 codec and selected-block attention.
- Set the V100 default desktop memory reserve to zero. Callers can request another value explicitly.
- Permit 27B MTP windows from one to seven drafts on SM70. Flash-Next retains its own limit.
  CUDA graph control remains available for dense 27B.

NVFP4 and K8V4 **KV cache storage** remain unavailable on Volta. This restriction does not
prevent loading NVFP4 **model weights**, which use the restored Volta dequantization paths.
The original fork's later context-lookup MTP shortcut has not been integrated into the current
runtime. DFlash2 performance is not qualified by this regression.

## Verification

`.github/workflows/v100-27b-regression.yml` runs on the user's V100 with the existing
`/models/qwen3_8_27b_nvfp4.ninfer` artifact (23,719,496,192 bytes). It builds the public CLI,
server and benchmark; checks real-model scoring repeatability; measures eager, graph and
MTP K=1/3/7 generation with INT8 KV; exercises speculative checkpoints and vision; checks
real Flash-Next grouped prefill; and compares the original fork using the same model and
benchmark corpus. Synthetic general Q4/NVFP4 linear tests are not part of this workflow.

Qualification is in progress. No measured 27B performance or passing runtime claim is made
until the real-artifact job completes. The original fork's published numbers describe its
own workloads and are not results for this branch.
