# V100 NInfer GPU expert-stream slot reuse: opt-in experiment

**Branch:** `perf/v100-strata-derived`  
**Implementation:** `src/targets/qwen3_8_flash_next/impl/expert_stream.cpp` and `expert_stream.h`  
**Hardware A/B workflow (prepared but not yet launched):** `.github/workflows/v100-ninfer-stream-reuse-ab.yml`.

## The bottleneck

The production four-slot GPU expert-stream ring currently waits synchronously for a slot's **previous GPU compute consumer** before reusing its pinned host staging buffers. A slot has two ownership constraints:

1. The previous H2D copy must finish before the host overwrites the slot's pinned `host_weights` and `host_groups`.
2. The previous GPU expert kernel must finish before another H2D overwrites the slot's **device buffers**.

The original `cudaEventSynchronize(s.consumed)` enforces both, but stalls the host until GPU compute completes. In long prefill, many distinct experts are streamed; this can delay CPU staging and shorten the overlap window between transfer and GPU execution.

## New optional policy

```bash
export NINFER_V100_EXPERT_STREAM_SLOT_REUSE=pipelined
```

Instead of synchronously waiting for the previous GPU consumer:

- Wait for **previous transfer-completion event** `s.ready` before repacking its host buffers. This guards against writing pinned buffers while DMA is reading them.
- Add `cudaStreamWaitEvent(transfer_, s.consumed, 0)` so the next H2D cannot overwrite device buffers until the previous kernel has finished, **without blocking the CPU thread** on GPU computation.
- Enqueue the usual H2D copies, record a new `s.ready`, and let the compute stream wait for that transfer before executing the next expert group.

The default remains `blocking`; the ring stays at **four slots** and requires **no additional VRAM or pinned memory**. Runtime still enforces one producer/consumer sequence and `finish()` drains outstanding GPU work.

## Verification and performance decision

The existing GPU streaming oracle submits 11 expert groups through a four-slot ring, forcing slot reuse, and checks each expert output against the CPU reference. It now saves/restores the inherited slot-reuse environment, so the second fixture actually runs all mathematical checks in `pipelined` mode rather than silently resetting to blocking.

The prepared A/B compares three serial same-model cases:

1. `lru-prefill-auto256`: default blocking reuse, 4 slots.
2. `lru-auto256-pipelined-reuse`: same setup, pipelined reuse, 4 slots.
3. `lru-auto256-repeat`: default blocking reuse, 4 slots.

The workload is 585 input / 64 output and 63 input / 64 output, five long warmups and three measured repetitions of each size per case. Compare HTTP medians, native prefill/decode and baseline drift. **Do not start this while the busy-first ordering A/B is running;** they share V100 and must not compete. Hosted tests and CUDA compilation must pass first. If the pipelined variant loses or is within drift, keep the default blocking implementation.

This is a bounded performance experiment, not a production default or a reason to run further model-quality campaigns.
