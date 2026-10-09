# V100 MoE stage attribution: grouped CPU saves 34% cold TTFT; GPU streaming trades replay speed for cold prefill

**October 9, 2026.** Protected [hardware run 37863525245](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37863525245) **SUCCESS** at source `65daad2acd0631d02db516de217b1e6ccc91aa80`, artifact [11586949278](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37863525245/artifacts/11586949278), 4.34 MB compressed / 91.0 MB uncompressed. All three arms passed production request/prefix checks. Fixed 64 resident experts/layer, device route combine, BF16 projections, MTP disabled, 32 CPU workers, same model and request, one fresh process/arm. GPU guard >=28000 MiB; no model writes or other process stops.

| Metric | CPU-cache single | CPU-cache grouped | GPU-stream grouped |
|---|---:|---:|---:|
| Cold 1227-token HTTP TTFT (instrumented) | 22.870 s | 15.141 s | **13.127 s** |
| Cold 1220-token stage ledger total | 22.597 s | 14.771 s | **12.486 s** |
| Cold 1220-token MoE reduce interval | 20.454 s | 12.640 s | **10.355 s** |
| All seven instrumented prefill chunks | 27.112 s | **19.272 s** | 21.237 s |
| All-chunk MoE reduce interval | 23.554 s | **15.726 s** | 17.759 s |
| Prefix replay HTTP TTFT | **0.391 s** | 0.455 s | 0.748 s |
| Continuation HTTP TTFT | 2.009 s | **1.742 s** | 4.446 s |
| Continuation replay HTTP TTFT | **0.398 s** | 0.451 s | 0.802 s |
| Cold 1220-token CPU expert weight reads | 1232.06 GiB | **323.40 GiB** | 0 |
| Cold 1220-token streamed expert-weight H2D | 0 | 0 | 35.61 GiB |

**Findings:** Grouping reduces cold instrumented TTFT **33.8%** vs ungrouped and reduces CPU expert weight-read traffic **73.8%**. This is a controlled opt-in policy comparison, **not a new optimization over the prior production SV7 screen**, which already enabled grouping. Streaming every nonresident prefill expert reduces cold TTFT another **13.3%** vs grouped CPU and MoE reduce interval another **18.1%**, but is substantially worse on short prefix replay and continuation (0.748 vs 0.455 seconds and 4.446 vs 1.742 seconds). All-chunk time is **10.2% worse** with streaming than grouped CPU. The data strongly favor a **token-count-dependent policy** rather than unconditional GPU streaming.

**Limitations:** Stage ledger inserts CUDA events and synchronizes at chunk boundaries; elapsed intervals include CPU stalls and stream waits, so neither stage totals nor single-process TTFT are production throughput claims. Four requests per process, seven prefill chunks (tokens 13, 1220, 7, 7, 86, 7, 7), and only one process per policy. No independent numerical Phase11 admission was performed; output signatures are not a substitute for quality qualification. Static 64-slot cache differs from prior LRU/156-slot streaming sweeps; do not generalize crossover thresholds across cache budgets.

**Next performance-only action:** Run three-process-per-arm production HTTP A/B (with and without MTP) using exactly these three policies, no stage ledger or expensive telemetry, preserving cache, model, context, and output parameters. Record cold TTFT, prefix replay, continuation, decode, and MTP. If long cold prefill improvement persists while replay regresses, pursue a bounded opt-in policy that streams only sufficiently large prefill chunks; leave defaults unchanged. Do not expand to Phase18/SV8 or general quality campaigns.
