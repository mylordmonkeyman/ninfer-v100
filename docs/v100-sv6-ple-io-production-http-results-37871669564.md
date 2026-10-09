# SV6 PLE I/O production HTTP: strict direct reads do not beat warm mmap

**October 9, 2026.** Protected [GPU run 37871669564](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37871669564) **SUCCESS**, source `ef2f339c8b2ec11e6ab6ef0ad0ec0b552d518f4c`, [artifact 11591640042](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37871669564/artifacts/11591640042) (4,664,294 bytes ZIP, 6,365,998 bytes on disk). The protected workflow ran hosted checks then two standalone backend-verification servers and **12 production HTTP timing servers** (three independent server processes per backend for each MTP setting). No process failures or unwanted model writes.

## Medians from 3 fresh processes per arm (seconds except throughput)

| Policy | MTP | Cold TTFT | Cold prefill tok/s | Prefix replay TTFT | Continuation TTFT | Continuation replay TTFT | Continuation decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| **mmap** | off | 12.794 | 95.922 | 0.405 | 1.821 | 0.407 | 9.023 |
| strict O_DIRECT | off | 12.664 | 96.915 | 0.402 | 1.780 | 0.403 | 9.000 |
| **mmap** | on | 12.976 | 94.583 | **0.399** | **1.829** | **0.400** | **7.214** |
| strict O_DIRECT | on | 12.905 | 95.099 | 0.452 | 1.899 | 0.458 | 6.743 |

**Result:** Strict direct is only **1.0% faster without MTP** and **0.5% faster with MTP** on cold TTFT, well below a compelling end-to-end benefit and too small to justify a default change from this limited corpus. With MTP, strict direct slows prefix replay **13.3%**, continuation TTFT **3.8%**, and continuation decode throughput **6.5%**. No-MTP continuation is 2.3% faster (1.780 vs 1.821 s); do not extrapolate either way.

## Validation

- **Actual storage backend, no fallback:** The separate telemetry-on diagnostic records confirmed exactly seven compressed PLE gather calls per arm, max request chunk **1,220 tokens**. `mmap` reported 0 direct page reads; strict direct reported **8,483 coalesced 4 KiB page reads**. Both indicated the expected backend and `storage_fallback=false`.
- **Exact output parity:** Both backends produced identical greedy response signatures for the same MTP setting across all three fresh processes; no difference in first or continuation text/finish accounting. On MTP, the cold request drafted 71 tokens and accepted 38 tokens in every process and both backends. Consequently, MTP performance differences in this A/B are not explained by different MTP acceptance counts for the cold request.
- **Fixed controls:** Same model and identity-bound 64-slots/layer expert profile (8,494,252,032 bytes); adaptive stream for chunks >=256 tokens; grouped CPU expert misses; 32 CPU workers; BF16 KV and projections, SV7 FP16 tensor-core candidate disabled; no CUDA graphs; one concurrent request; 4096-token max context, `--prefill-chunk 2048`; fixed two-turn solar-station request, max 64 generated tokens.
- **Ranges for cold TTFT:** no MTP mmap **12.639–12.824 s**, strict direct **12.626–12.810 s**; MTP mmap **12.822–12.981 s**, strict direct **12.859–13.089 s**. These overlap significantly; median differences are not convincing.
- **Safety:** The workflow checked >=28,000 MiB V100 headroom before starting and before each server, held the `v100-sv0-hardware` concurrency lock, did not modify/copy the original ~113GB model or installed Strata, did not evict the host file cache, did not terminate unrelated processes, and refused artifacts >=1GiB. The output directory contains no model symlinks; evidence directory was 6.37 MB.

**Limitations:** The test compares **warm/resident mmap pages** with strict direct reads and does not equalize page residency. This is the relevant production-like hot-cache case, not an uncached block-device contest. The workload is one ~1.2K-token initial prefix plus short prefix replay and continuation, with static 64 expert slots and a single active request. It is not a 4K/8K prefill or concurrency matrix. No independent numerical Phase11 admission; unchanged thresholds.

**Decision:** Keep `NINFER_V100_PLE_IO=mmap` (existing production behavior). Keep strict direct an opt-in diagnostic; **do not spend more GPU time tuning its queue depth** absent evidence that PLE disk misses dominate the actual production workload. PLE compressed pipeline already overlaps gather with layer-zero computation and uses a separate CUDA transfer stream; target the verified MoE expert bottleneck instead of claiming direct I/O speeds up prefill. Follow the documented ~15% *experimental* adaptive MoE cold-TTFT advantage from [run 37869007809](v100-moe-policy-production-http-results-37869007809.md), but do not promote auto256 while its reproducible cross-policy greedy differences remain numerically unqualified against Phase11 thresholds.

**Next performance-only investigation:** concentrate on the MoE runtime (CPU expert grouping / streamed misses / scheduling and decode overhead), avoid duplicate PLE I/O sweeps, SV8/Phase18 or broad quality campaigns. SV5 early handoff and SV7 rounded FP16 TC remain opt-in after their negative full-model speed results.
