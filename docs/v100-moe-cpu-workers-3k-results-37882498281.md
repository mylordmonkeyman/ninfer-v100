# V100 NInfer 3K grouped expert CPU workers: 32 / 48 / 64

**October 9, 2026.** Protected [GitHub Actions run #37882498281](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37882498281) **SUCCEEDED**; source `ce155fd27c8ffe2a8aa6749ce7bcd750aa0090f4`, artifact [#11596031751](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37882498281/artifacts/11596031751), 7,060,598 bytes compressed.

**Controls:** same 113GB NInfer Flash-Next mixed-quant model, 3111-token cold input spanning 2048+1056-token prefill chunks, 4096 context, 64 persistent V100 GPU experts per layer, identity-bound static profile, `auto256` prefill, `NINFER_V100_PREFILL_EXPERT_STREAM_MIN_ROUTES=20`, grouped AVX2 CPU fallback, BF16 KV/projection, mmap PLE, no graphs or SV7 TensorOp. Three **fresh-process** measured repetitions for each CPU worker count and each MTP state, 18 timed processes; three separate traffic diagnostic processes. Cross-arm output signatures were identical in each repeat for all four requests (cold, prefix replay, continuation, continuation replay). Model remained read-only.

## Controlled HTTP performance medians

| CPU workers | MTP | Cold TTFT s | Cold prefill tok/s | Readonly-prefix TTFT s | Continuation TTFT s | Continuation decode tok/s |
|---|---|---:|---:|---:|---:|---:|
| 32 | off | 25.129 | 123.823 | 0.433 | 1.752 | 8.745 |
| 48 | off | 24.476 | 127.125 | 0.348 | 1.689 | 10.390 |
| **64** | **off** | **24.116** | **129.024** | **0.308** | **1.600** | **11.624** |
| 32 | on | 25.616 | 121.472 | 0.403 | 1.827 | 8.181 |
| 48 | on | 25.085 | 124.047 | 0.347 | 1.686 | 9.524 |
| **64** | **on** | **24.673** | **126.110** | **0.317** | **1.601** | **12.335** |

Measured first-cold-response decode rates without MTP: 32/48/64 workers = 9.004 / 10.768 / 11.606 tokens/s. With MTP: 7.676 / 9.364 / 11.709 tokens/s. Thus 64 workers cut cold TTFT by **4.03% off / 3.68% on** against 32, improved cold prefill throughput by **4.20% off / 3.82% on**, and improved cold decode by **28.9% off / 52.5% on**. Continuation decode improved **32.9% off / 50.8% on**. All four requests and both MTP states consistently favor 64.

## Hardware work conserved

Across 96 prefill MoE layer records per arm (both large chunks, separate diagnostic runs):

| Metric | 32 | 48 | 64 |
|---|---:|---:|---:|
| Streamed GPU expert weight bytes | 33,321,689,856 | 33,321,689,856 | 33,321,689,856 |
| Streamed GPU expert routes | 1,138,975 | 1,138,975 | 1,138,975 |
| Grouped CPU missed routes | 93,424 | 93,424 | 93,424 |
| Compact CPU expert weight reads | 83,961,689,344 | 83,961,689,344 | 83,961,689,344 |
| Cumulative CPU branch time, seconds | 4.077 | 3.523 | 3.274 |

The route work and GPU transfer volume are **identical** for all arms. Cumulative CPU branch time improves **19.7%** at 64 workers. This isolates worker throughput, not a change in workload or cache residency.

## Recommendation and limits

For the tested dual Xeon E5-2697A v4 host with 64 logical CPUs, use `NINFER_FLASH_NEXT_CPU_EXPERT_WORKERS=64` as the **experimental reference** when evaluating grouped CPU expert prefill and short decode. This is not a default for other hosts, topologies, concurrency, or longer contexts; no NUMA pinning, thermal placement, SMT-specific attribution, or 8K+ input was tested. Model-quality Phase11 numerical acceptance and production defaults remain unchanged.

**Next performance test:** Validate whether the 64-worker benefit persists at 8K+ input across more than two prefill chunks under the same auto256/minroutes20 policy and static GPU expert budget, comparing 32 vs64 with independent cold-process timing and identical weights/profile. Separately consider a bounded NUMA placement A/B if longer-context behavior warrants it. Do not change worker counts in installed llama-swap/Strata automatically.
