# V100 7K static GPU expert-cache budget: 64 vs 128 slots/layer

**October 9, 2026 — performance-only controlled screen.** [Protected hardware run 37991802825](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37991802825) **passed** (hosted gate and hardware); [artifact 11646466123](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37991802825/artifacts/11646466123), source `724ed96b3e85b5d8f84f99cdf2ab6dc7f7028eb6`. Independently read artifact `campaign/report.json`, `report.txt`, `observations.json`, `cache-budget-traffic.json`, and diagnostic server logs. No production change.

The only intentional independent variable is the number of **fully seeded static GPU expert slots per MoE layer**: 64 versus 128, using the same pinned 512-expert popularity ranking. Both use frozen read-only 113GB mixed-quant Flash-Next, 88 CPU expert workers on the same 32 physical-core taskset across sockets, MTP draft window 2, BF16 KV/projections, 7111-token cold prompt crossing four 2048-token prefill chunks, 8192 context, one active request, grouped AVX2 CPU fallback, auto256/minroutes20 streaming, mmap PLE, device combine, no CUDA graphs, no SV7 TensorOp. Inherited NUMA memory policy remains unchanged. Each arm: 3 alternating **fresh-server** HTTP repetitions plus an independent instrumented four-chunk traffic server recording 192 MoE expert-layer events.

## Timed HTTP medians (3 fresh servers per arm)

| Measure | Static 64 | Static 128 | 128 vs 64 |
|---|---:|---:|---:|
| Cold TTFT, seconds | 53.872 | 53.609 | **0.49% lower** |
| Cold prefill, tokens/s | 132.015 | 132.667 | **0.49% higher** |
| Cold decode, tokens/s | **16.540** | 15.619 | **5.57% lower** |
| Prefix replay TTFT, seconds | 0.216 | 0.210 | **2.75% lower** |
| Prefix replay decode, tokens/s | **16.744** | 15.584 | **6.93% lower** |
| Continuation TTFT, seconds | 1.311 | 1.179 | **10.04% lower** |
| Continuation decode, tokens/s | **18.034** | 16.471 | **8.67% lower** |
| Continuation replay TTFT, seconds | 0.214 | 0.214 | negligible |
| Continuation replay decode, tokens/s | **17.931** | 16.116 | **10.12% lower** |
| Sampled peak GPU memory, MiB | 18,790 | 26,890 | **+8,100 MiB** |

Sampling every 5 seconds is not an exact device-allocation high-water mark. Cache payload allocation was **8,494,252,032 versus 16,988,504,064 bytes** (actual seeded 3072 versus 6144 total layer slots). The larger cache therefore adds about **7.91 GiB of allocated GPU expert weights**.

## Independently instrumented expert traffic (192 layers per arm)

| Metric | Static 64 | Static 128 | 128 vs 64 |
|---|---:|---:|---:|
| Total routes | 3,409,920 | 3,409,920 | identical |
| Resident GPU hit routes | 575,120 | 993,223 | **+72.70%** |
| Streamed expert routes | 2,653,326 | 2,262,714 | **−14.72%** |
| Streamed expert H2D bytes | 71,158,716,160 | 59,451,469,056 | **−16.45%** |
| CPU miss routes | 181,474 | 153,983 | **−15.15%** |
| CPU miss H2D bytes | 1,858,293,760 | 1,576,785,920 | **−15.15%** |
| CPU compact weight-read bytes | 162,222,344,592 | 138,384,170,016 | **−14.70%** |
| Inclusive CPU branch time | 5.215 s | 4.663 s | **−10.60%** |

The reduction in expert transfer/CPU fallback is real, but it **does not translate into a cold prefill or decode win proportional to its extra VRAM**. The decode regression may be due to increased GPU-resident execution/dispatch pressure; this is a hypothesis, not a measured causal conclusion. No claim that GPU H2D traffic explains the entire TTFT.

## Output and qualification gates

Within each arm, fresh-process HTTP response signatures are repeatable; finish reason and completion-token counts are preserved between arms. **All 3 cross-cache paired response-text comparisons differ**, including greedy requests with identical seed/sampling parameters. Cache residency changes execution arithmetic (GPU vs grouped host expert path), so cross-cache exactness was intentionally **diagnostic**, not a pass gate. Do not treat this run as independent numerical or Phase 11 qualification; no numerical thresholds were relaxed. Source/configuration remain frozen and baseline numerical issues remain separate. No GPUs were thermally throttled according to captured evidence; the 5-second sample does not exclude all short transients.

## Decision and next bounded performance experiment

**Retain 64 GPU expert slots/layer as the decode-oriented experimental reference; do not promote 128 or alter production settings.** The +8.1 GiB sampled VRAM burden buys only ~0.5% cold prefill gain, at a 5.6–10.1% decode penalty (depending on request). For a meaningful next single controlled test, compare **32 vs 64 fully seeded static slots/layer** at unchanged 88 workers/MTP2/physical-core mask/7K workload; determine whether smaller residency actually improves decode or merely increases streaming/CPU misses. Check planner headroom, actual seeded slots, traffic, exact same-arm outputs, GPU memory, numerical cross-arm difference diagnostics, and alternate ordering; retain defaults regardless of outcome. Do not launch speculative wide sweeps or quality campaigns.

## Software-only infrastructure status

[Software run 37991802757](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37991802757): Python/telemetry checks **passed**. One isolated CUDA syntax job failed on initial Docker Hub container pull and **failed again at container initialization after one specifically requested job retry**; checkout and compiler did not execute. This is not evidence of a CUDA compiler or kernel regression. Do not repeatedly retry unchanged pulls; do not modify source or start an extra hardware job to address an external Docker Hub error.
