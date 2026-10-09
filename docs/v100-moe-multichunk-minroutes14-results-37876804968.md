# V100 MoE multi-chunk minroutes14: controlled production HTTP A/B

**October 9, 2026.** Protected [GitHub Actions run #37876804968](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37876804968) **SUCCESS**, source `81cf60e76e8b01d79198a023f1d601621ea10c62`, [artifact #11593405741](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37876804968/artifacts/11593405741), 4,750,534 bytes compressed. All 12 fresh-process HTTP timing servers passed (3 repetitions per policy × MTP off/on), plus 2 independent diagnostic server processes, after hosted CI passed. The previous aborted attempt #37875821807 revealed a *single-chunk harness validation bug* when given the intended 3,111-token prompt; this was fixed with dedicated 2,560–3,500 token checks and regression tests before the successful rerun. No kernel/model changes were needed.

This is a performance-only screen comparing existing adaptive stream-all `NINFER_V100_PREFILL_EXPERT_POLICY=auto`, `NINFER_V100_PREFILL_STREAM_MIN_TOKENS=256` with opt-in `NINFER_V100_PREFILL_EXPERT_STREAM_MIN_ROUTES=14`, which streams a nonresident expert to the GPU only if that expert receives >=14 routes in the current layer/chunk. The rest go to the grouped AVX2 CPU pool. The first request used **3,111 tokens, two prefill chunks of 2,048 and 1,056 tokens**; fixed 4,096-token model context, 64 resident GPU expert slots/layer, 32 CPU workers, BF16 KV/projections, SV7 TensorOp OFF, mmap PLE, no CUDA graphs, one active request.

## Uninstrumented production HTTP median (three fresh processes per row)

| Policy | MTP | Cold TTFT s | Cold prefill tok/s | Prefix replay TTFT s | Continuation TTFT s | Continuation replay TTFT s | Continuation decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| auto256 stream all | off | 27.088 | 114.867 | **0.400** | 1.814 | 0.413 | 8.774 |
| **auto256 minroutes14** | off | **25.067** | **124.134** | 0.403 | **1.746** | **0.412** | **9.016** |
| auto256 stream all | on | 27.727 | 112.222 | 0.406 | 1.899 | **0.406** | 7.274 |
| **auto256 minroutes14** | on | **25.721** | **120.975** | **0.402** | **1.760** | 0.410 | **7.853** |

Cold TTFT reduction is **7.46% without MTP** and **7.23% with MTP**, and first-request prefill increases **8.07% and 7.80%** respectively. Min14 continuation TTFT improves 3.74% and 7.30%, without material changes to short-prefix replay. No-MTP cold TTFT baseline spans 27.056–27.107s, min14 25.021–25.126s; MTP baseline 27.715–27.738s, min14 25.700–25.728s, non-overlapping three-process ranges.

## Independent expert traffic diagnosis: both large chunks, 96 MoE layer records/arm

| Counter, two chunks summed across 48 layers | Stream all | Minroutes14 |
|---|---:|---:|
| GPU expert-weight uploads | 75,743,179,008 B (**70.541 GiB**) | 38,685,898,496 B (**36.029 GiB**) |
| GPU expert-weight H2D reduction | — | **48.925%** |
| Routes executed by GPU stream | 1,232,353 | 1,171,416 |
| Routes executed by grouped CPU fallback | 0 | **61,274** |
| CPU compact expert weight reads | 0 | **59,653,497,408 B (55.557 GiB)** |

GPU H2D reduction is 34.512 GiB across both prefill chunks. CPU weight reads are host DRAM bytes, **not extra PCIe transfers**. The 96 layer records were collected in separate diagnostic processes and are not mixed into timed HTTP measurements. This confirms actual selectivity rather than merely setting an environment variable.

## Greedy output and speculative observations

All three fresh processes were **identical within their own policy** for both MTP settings; same-policy readonly-prefix and continuation replay checks passed. **Cold completion output was identical between the two policies in both MTP configurations.** Continuation responses differed between policies (64 output tokens and `length` finish in both arms), so `exact_compared_responses=false` across all pairs. MTP cold draft/accept were identical, **78/36** under both policies; continuation was stream-all **85/33** versus min14 **79/36**. Therefore, continuation decode-speed differences are not strictly equal-work comparisons. Cross-policy output differences are diagnostic, not independent numerical acceptance or proof of quality loss.

## Decision and safety

- The earlier 1.2K-token [HTTP screen #37873691020](v100-moe-minroutes14-http-results-37873691020.md) established ~15% cold TTFT improvement and 54% lower expert H2D on one prefill chunk. This larger screen confirms a smaller but robust **~7% cold improvement** on two chunks and **49% lower expert H2D**. Do not extrapolate these percentages to new prompts, larger contexts, or concurrency.
- Keep `minroutes14` **experimental opt-in**. Do not promote to production defaults until independent Phase11 numerical qualification with unchanged thresholds and broader operational confidence. Do not run broad quality campaigns as part of this performance iteration.
- No 113GB model write, copy, page-cache eviction, installed Strata change, unrelated process kill, or runner churn. Shared hardware concurrency `v100-sv0-hardware`; 28,000 MiB free-V100 requirement before each fresh server; evidence 4.75 MB compressed under 1 GiB cap.
- SV6 strict PLE direct I/O showed <1% cold latency gain and MTP replay regression; keep mmap. SV5 route handoff and SV7 experimental FP16 TensorOps have no meaningful demonstrated production gain; leave both off.

**Next controlled performance-only experiment:** sample a small *higher* per-expert route threshold (e.g., 20/28) alongside min14 in this 3K workload, keeping all prior cache/model/CPU controls fixed. The trend from 70.5 to 36.0 GiB expert uploads suggests further transfer reduction may help, but higher thresholds also add CPU work, so benefit must be measured rather than assumed. Use one guarded GPU job, three fresh processes/arm/MTP mode, and bounded telemetry for actual traffic. No default change.
