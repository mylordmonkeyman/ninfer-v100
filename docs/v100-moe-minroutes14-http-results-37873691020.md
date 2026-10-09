# V100 MoE minroutes14: production HTTP and PCIe traffic

**October 9, 2026.** Protected V100 [hardware run #37873691020](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37873691020) **SUCCESS**, source `df9eb149d1e37f8fc2c1dbafb8e57c18f857c7f1`, [artifact 11591369763](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37873691020/artifacts/11591369763), 4,643,797 compressed bytes. Hosted and hardware checks passed. 12 uninstrumented fresh-process HTTP tests (3 repeats per arm × 2 MTP settings) plus 2 separately instrumented large-prefill traffic checks; no resident-model writes, runner churn or unrelated process shutdowns.

## Production HTTP medians, three independent processes per cell

| Expert policy | MTP | Cold TTFT (s) | Cold prefill (tok/s) | Prefix replay TTFT (s) | Continuation TTFT (s) | Continuation decode (tok/s) |
|---|---|---:|---:|---:|---:|---:|
| Baseline auto256 / stream all missed | off | 12.845 | 95.543 | 0.408 | 1.819 | **8.956** |
| **Auto256 / stream only >=14 routes** | off | **10.839** | **113.223** | **0.403** | **1.709** | 8.855 |
| Baseline auto256 / stream all missed | on | 12.933 | 94.899 | 0.400 | 1.841 | 7.135 |
| **Auto256 / stream only >=14 routes** | on | **10.993** | **111.647** | 0.402 | **1.736** | **8.207** |

Minroutes14 improves cold TTFT **15.6% (MTP off)** and **15.0% (MTP on)**, and cold prefill throughput **18.5% and 17.6%**. Continuation TTFT improves **6.1% and 5.7%**; short replay stays around 0.4 s. Without MTP, continuation decode is ~1.1% slower; with MTP the measured continuation decode is ~15.0% faster, but *the two policies generate different tokens and have different MTP acceptance*, so this must not be treated as equal-work throughput.

Medians are genuinely separated across all three processes: no-MTP cold TTFT baseline **12.595–12.979 s**, min14 **10.790–11.011 s**; MTP baseline **12.833–13.064 s**, min14 **10.970–11.054 s**. The 1,227-token first request and 64-token completion are the same across modes; fresh-process prefix/continuation accounting and replay checks passed.

## Direct expert-weight transfer evidence

The two separate diagnostic first large prefill chunks each recorded **48 routed expert-layer rows**; no stage ledger in timing runs. Counters summed across 48 layers for one large first chunk, not across an entire user session:

| Hardware counter | Auto256 stream-all | Auto256 minroutes14 |
|---|---:|---:|
| Expert weights streamed GPU H2D | **35.609 GiB** | **16.257 GiB** |
| GPU-stream expert routes | 478,061 | 444,506 |
| Grouped CPU fallback miss routes | 0 | **33,640** |
| CPU compact expert weight reads | 0 | **30.796 GiB** |

Thus minroutes14 cuts GPU expert-weight H2D **54.35%** (19.35 GiB saved) while shifting low-traffic groups to AVX2 CPU experts. The first-chunk counters do not include all host-device copies, nor imply that 30.8 GiB of CPU reads are PCIe traffic; those bytes are host compact-weight memory accesses.

## Interpretation and admission limits

- Compare only against the exact static 64 GPU expert slots/layer, 32 CPU workers, BF16 KV and projection, grouped CPU fallback, mmap PLE, no CUDA graphs, 4096 context, max concurrency 1, prefill chunk 2048, auto streaming threshold 256 tokens, fixed model and identity-bound profile. SV7 tensor-core pilot remained OFF.
- Greedy outputs and MTP acceptance were **identical across repeats within the same policy** but not identical across the two policies for any repeat and either MTP setting. Cold MTP draft/accepted counts: stream-all **71/38**, min14 **74/37**; continuation **87/33** versus **75/37**. Cross-policy difference is diagnostic, not a Phase11 numerical success. **Do not change or relax independent Phase11 thresholds** and do not promote to defaults.
- No broad task-quality campaign, long-context, second prompt distribution or concurrent load was measured. A 1.2K-token cold prompt is not a long-prefix production qualification. Fixed cache hit accounting passed but device/host cache behavior may differ after longer sessions.
- All 12 timing runs passed, as did the separate proof that the intended H2D-versus-CPU routing split occurred. The uploaded artifact was **4.64 MB**, below the 1 GiB cap; GPU free headroom >=28000 MiB enforced.
- Earlier [SV6 strict direct I/O screen](v100-sv6-ple-io-production-http-results-37871669564.md) was not meaningfully faster than mmap, and SV5 handoff / SV7 FP16 TensorOp had no end-to-end performance win. Retain mmap, SV5 off, SV7 off.

**Decision:** Keep minroutes14 *experimental opt-in*. This production HTTP and hardware-traffic A/B supports a real latency win in this profile, but it does not constitute independent full-model numerical qualification. Next bounded performance test: confirm performance with >1 prefill chunk / ~3K tokens (and if safe a second prompt distribution) using the same one-job concurrency and read-only model; avoid overspending on queue-depth, telemetry perfection, Phase18/SV8 or broad quality sweeps.
