# V100 3K MoE threshold sweep: minroutes14/20/28

**October 9, 2026.** Protected [hardware run #37878973416](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37878973416) **SUCCESS**, source commit `5ff1a1e97567635346a7635161f08ae132b7a183`, [artifact #11593504643](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37878973416/artifacts/11593504643), 7,113,010 bytes compressed. Exactly **18 fresh-process timing servers** (3 repeats × 3 thresholds × MTP on/off) and three separate traffic diagnostics passed. All served four HTTP calls (cold, read-only prefix replay, continuation, continuation replay) with consistent within-policy output/finish accounting. No runner/model/Strata alterations or overlapping protected jobs.

**Controlled configuration:** 3,111-token cold input, 2,048+1,056-token prefill chunks, 64 output tokens; 4,096-token context, 64 fixed GPU expert slots/layer (static identity-bound expert profile), 32 host CPU expert workers, BF16 KV/projection, mmap PLE, 1 active request, no CUDA graphs, SV7 TensorOp OFF, auto256 GPU prefill with grouped CPU fallback. Experimental `NINFER_V100_PREFILL_EXPERT_STREAM_MIN_ROUTES` set to 14, 20 or 28. Streaming is eligible only for missed experts routed that many times within their layer/chunk. Production defaults unchanged.

## End-to-end HTTP medians, 3 fresh processes per cell

| Threshold | MTP | Cold TTFT s (range) | Cold prefill tok/s | Prefix replay TTFT s | Continuation TTFT s | Continuation decode tok/s |
|---|---|---:|---:|---:|---:|---:|
| min14 | off | 25.190 (25.166–25.222) | 123.524 | **0.396** | **1.754** | **8.995** |
| min20 | off | **25.026** (24.991–25.069) | **124.331** | 0.435 | 1.802 | 8.766 |
| min28 | off | 25.099 (25.039–25.201) | 123.970 | 0.403 | 1.767 | 8.797 |
| min14 | on | 25.712 (25.712–25.802) | 121.016 | 0.398 | 1.789 | 7.905 |
| min20 | on | **25.594** (25.481–25.625) | **121.573** | **0.394** | **1.721** | **8.345** |
| min28 | on | 25.746 (25.660–25.752) | 120.857 | 0.403 | 1.816 | 8.286 |

- Versus min14, min20 cold TTFT is only **0.65% faster without MTP** and **0.46% faster with MTP**; these are small effects. Without MTP it slows short prefix-replay TTFT by ~9.8% (0.396→0.435s) and continuation by ~2.7% (1.754→1.802s). With MTP min20 accelerates continuation by ~3.8% (1.789→1.721s), but the continuation text differs across policies and the MTP accepted-token count also differs.
- Min28 provides **no further cold latency benefit** and is slower than min20 at cold TTFT with either MTP setting. Prefix replay stays ~0.4s, continuation similar to min14 without MTP and worse with MTP.
- All cold prefill ranges and requested actual fresh tokens are in the artifact. All 18 requests have identical initial prompt token counts (3,111). The first 3,111 input tokens crossed two prefill chunks under the fixed 2,048-token chunk limit.

## Actual expert traffic (separate diagnostic runs, 96 large-chunk expert-layer records per threshold)

| Hardware counter, both cold prefill chunks | min14 | min20 | min28 |
|---|---:|---:|---:|
| GPU expert-weight H2D | **36.029 GiB** | **31.033 GiB** | **26.352 GiB** |
| GPU-streamed expert routes | 1,171,416 | 1,138,975 | 1,097,138 |
| Grouped-CPU missed routes | 61,274 | 93,424 | 136,086 |
| CPU compact-expert weight reads | 55.557 GiB | 78.195 GiB | 107.444 GiB |

Raising 14→28 saves **26.9%** further *expert-weight* H2D but nearly **doubles CPU compact-weight reads**. The cold HTTP latency plateaus around 25 seconds: the data support a host-CPU work vs PCIe transfer balance rather than unlimited gains from further cutting H2D. CPU weight reads are DRAM, not PCIe; the H2D metric does not count all GPU transfers.

## Output controls and numerical caveat

All three repeats of each policy were identical to one another and each within-process readonly prefix/continuation replay matched. **Cross-policy exact output equivalence failed** at all six comparisons: with MTP off, even some cold outputs differ across thresholds despite sharing the opening text; with MTP on, all three cold outputs agree, but continuation outputs diverge. The cold MTP draft/accept counts were consistently 78/36 across policies; the continuation MTP counts were 79/36 for min14, 74/38 for min20 and min28. These indicate different computation/output paths and prevent treating all decoding throughput numbers as equal-work measurements. None of these comparisons independently qualifies accuracy under Phase11; its numerical gates remain unchanged.

## Decision and next performance experiment

**Keep min14 as conservative experimental performance reference; min20 is a promising alternative, but not a clearly superior all-round setting.** min28 saves PCIe bytes at an increased host CPU cost without cold TTFT improvements. Do not promote any setting to production default without independent Phase11 numerical acceptance.

An informative next bounded experiment is whether the 32-worker CPU expert fallback becomes a bottleneck at min20. Compare **32 vs 48 vs 64** workers, holding min20, GPU cache, exact model/prompt and MTP controls fixed, 3 fresh-process timing repetitions/arm/MTP mode. Existing older performance tests with 156 slots found a 64-worker control faster than 16, but not a matched 32/48/64 grouped CPU fallback under this policy. This is more diagnostic than further increasing the GPU stream threshold. Verify thread configuration and no oversubscription / missing thermal evidence. No Phase18/SV8, broad quality campaign, page-cache eviction, model writes or installed Strata changes.
