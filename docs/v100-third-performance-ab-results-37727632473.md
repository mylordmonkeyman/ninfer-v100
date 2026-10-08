# Third V100 A/B: GPU prefill expert streaming is a measured NInfer win

**Run:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37727632473  
**Evidence:** `v100-ninfer-prefill-ab-37727632473-1`, artifact ID `11529276985` (19,612,962 bytes).  
**Branch:** `perf/v100-strata-derived`; frozen source SHA `0e62d4ac6c912dc820d4a272abb037485e778604`.  
**Validation:** hosted software and CUDA checks run `37727632390`, both successful.

All six requested cells completed. Same NInfer model, quantization, context,
GPU cache slot budget (156 slots per layer), seed profile and 64-token greedy
decode. Each cell had 5 warmup long prompts, 3 measured long and 3 measured
short. No other GPU job overlapped.

| NInfer configuration | Long total HTTP median | Long prefill native | Long decode native | Short total HTTP median |
|---|---:|---:|---:|---:|
| Static seeded cache | 14.026 s | 9.217 s | 4.727 s | 6.454 s |
| LRU adaptive cache | 12.371 s | 8.768 s | 3.578 s | 5.015 s |
| LRU + 2 cache admissions/round | 12.467 s | 8.878 s | 3.584 s | 5.268 s |
| **LRU + GPU expert prefill streaming** | **9.714 s** | **6.668 s** | **2.948 s** | **5.322 s** |
| Profile-prior adaptive policy weight 50 | 11.549 s | 7.747 s | 3.985 s | 5.182 s |
| Static baseline repeated | 13.865 s | 9.269 s | 4.624 s | 6.349 s |

The long-prompt GPU-streaming result is **30.7% lower wall time** than static,
or 1.444x as many 585-input/64-output requests per unit wall time. Against
ordinary LRU, it is **21.5% less time**. Native prefill falls 27.7% from
static and 24.0% from LRU. The prefill-stream policy did not explicitly
enable decode expert streaming; its decode improvement likely reflects
different cache residency/route history, not a new decode GPU streaming path.

On the **short 63-token input**, always-streaming was **6.1% slower**
than ordinary LRU (5.322 versus 5.015 s), even though it remained faster
than static. This is the expected tradeoff for small batches where expert
transfer/setup outweighs GPU execution benefit.

The static baseline repeated within 1.2% long and 1.6% short, so the
large measured streaming improvement is unlikely to be run-order drift.
All outputs contained nonempty answers, but **output hashes differ**
across caching/streaming configurations and some adaptive-cache repeats.
The 64-token single-topic smoke test is not an accuracy qualification.
Do not change the production default until a task-level quality suite
confirms correctness.

## Next single targeted test: auto prefill threshold

The current engine already implements `NINFER_V100_PREFILL_EXPERT_POLICY=auto`
with `NINFER_V100_PREFILL_STREAM_MIN_TOKENS`. In
`src/targets/qwen3_8_flash_next/impl/text_decode.cpp`, streaming is
selected only for a prefill batch whose `tokens` count reaches the
threshold. A threshold of **256 tokens** should use GPU streaming for
the 585-token long request and CPU-cache for the 63-token short request.
A/B **LRU**, **always-stream**, and **LRU+auto256** in one sequential
run, with a static baseline and end repeat. Do not re-run Strata or
unrelated configurations. Preserve model and evidence safeguards.

The performance mechanism is no longer just speculative: NInfer's
existing GPU expert-stream path reduces the cost of CPU expert fallback
during long prefill. Remaining improvement should target transfer/
kernel efficiency, not additional telemetry perfection. A
quantization-matched cross-engine experiment is still needed before
attributing all remaining NInfer-vs-Strata speed difference to engine
implementation rather than model representation.
