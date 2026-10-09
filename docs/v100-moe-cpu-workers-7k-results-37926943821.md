# V100 7K four-chunk CPU worker results

**October 9, 2026.** [Hardware run 37926943821](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37926943821) succeeded; [artifact 11616035855](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37926943821/artifacts/11616035855) is 4,900,227 bytes compressed. Source `0f3364d1ecce2f19077d8d87ac180ae29a04ac6d`.

Same original mixed-quant NInfer weights, 7111-token prompt, four 2048-token chunks, 8192 context, static 64 GPU experts/layer, auto256/minroutes20, grouped CPU fallback, BF16 KV, mmap PLE, no CUDA graphs or SV7 TensorOp. Three fresh-process repetitions per worker/MTP combination; two separate telemetry processes. GPU: V100 32GB; host: dual Xeon E5-2697A v4.

| Workers | MTP | Cold TTFT s | Cold decode tok/s | Continuation TTFT s | Continuation decode tok/s |
|---|---|---:|---:|---:|---:|
| 32 | off | 55.610 | 9.085 | 1.770 | 8.876 |
| 64 | off | 53.885 | 11.662 | 1.580 | 11.653 |
| 32 | on | 56.837 | 8.131 | 1.836 | 8.409 |
| 64 | on | 55.195 | 12.221 | 1.606 | 12.477 |

64 workers reduced cold TTFT by **3.10% without MTP / 2.89% with MTP**, and increased cold decode by **28.37% / 50.30%**. All prefix and continuation metrics also improved. The 3K result therefore extends to this 7K workload.

Independent review of all 48 timed HTTP responses confirmed exact message/finish/completion-count equality across worker arms within every repeat and MTP state. Each separate telemetry arm contains 192 MoE-layer records with identical 71,158,716,160 GPU expert H2D bytes, 2,653,326 streamed routes, 181,474 CPU missed routes, and 162,222,344,592 CPU weight-read bytes. Cumulative CPU branch time fell from 7.484 to 6.346 seconds (15.2%); these instrumented times are excluded from HTTP timing.

Retain 64 workers as the experimental reference on this host; no production default or installed Strata change. This is a within-NInfer scheduling result, not cross-quant optimization evidence or independent numerical qualification.

Next: one controlled inherited-memory-policy versus `numactl --interleave=all` A/B at 64 workers on this same 7K workload. Keep CPU affinity, model, route policy, MTP states and cache fixed. Capture actual process NUMA placement at readiness. Shared file-cache pages remain warm and are not migrated, so interpret it as a launch-policy experiment rather than guaranteed full-model first-touch locality.

The interleave-memory experiment [37938856797](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37938856797) stopped in its preflight: `set_mempolicy: Operation not permitted`. No model build or inference ran, and no performance claim follows. The current runner cannot apply that launch policy. Do not retry it unchanged or modify container security automatically.

The next available experiment instead keeps 64 workers and inherited memory policy fixed, comparing all allowed logical CPUs against one logical CPU per physical core across both sockets. This tests scheduling under Hyper-Threading, including 64 workers sharing 32 CPUs in the physical-core arm; it is not an isolated measure of NUMA memory locality. The launcher derives the CPU mask from actual socket/core topology and preserves process affinity/placement evidence. One protected workflow `.github/workflows/v100-ninfer-moe-affinity-7k-http.yml` runs the same three-repeat 7K HTTP screen, requiring unchanged outputs and expert traffic.


## Physical-core affinity result — run 37939319091

[Workflow run 37939319091](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37939319091) completed successfully on the V100. It held 64 expert workers and inherited memory policy fixed, comparing all 64 logical CPUs with CPUs 0–31: one logical CPU per physical core across both sockets. Process evidence confirmed `Cpus_allowed_list: 0-63` versus `0-31`; both arms retained `Mems_allowed_list: 0-1`.

| Metric | All logical CPUs | One logical CPU/core | Change |
|---|---:|---:|---:|
| Cold TTFT, MTP off | 53.863 s | 53.000 s | 1.6% lower |
| Cold prefill, MTP off | 132.044 tok/s | 134.186 tok/s | 1.6% higher |
| Cold decode, MTP off | 11.640 tok/s | 12.371 tok/s | 6.3% higher |
| Cold decode, MTP on | 12.195 tok/s | 13.576 tok/s | 11.3% higher |
| Continuation decode, MTP off | 11.660 tok/s | 12.824 tok/s | 10.0% higher |
| Continuation decode, MTP on | 12.501 tok/s | 15.086 tok/s | 20.7% higher |
| CPU expert branch diagnostic | 6.312 s | 5.347 s | 15.3% lower |

All six cross-arm HTTP response comparisons matched exactly. Both 192-layer diagnostics conserved GPU H2D bytes (71,158,716,160), stream routes (2,653,326), CPU misses (181,474), and CPU weight reads (162,222,344,592). No thermal throttling was observed.

The physical-core mask is the better experimental scheduling policy on this host. This remains an opt-in within-NInfer result; no production default, installed Strata setting, quantization claim, or Phase 11 numerical gate changed.

Next: compare 32 versus 64 expert workers while pinning both arms to the identical 32-physical-core mask. That isolates whether the 64-worker pool itself helps when scheduling/SMT placement is held fixed.


## Fixed physical-core worker result — run 37946839739

[Workflow run 37946839739](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37946839739) completed successfully on the V100. Both arms used CPUs 0–31, retained both NUMA memory nodes, and differed only in expert-worker count.

| Metric | 32 workers | 64 workers | Improvement |
|---|---:|---:|---:|
| Cold TTFT, MTP off | 54.154 s | 53.010 s | 2.1% lower |
| Cold prefill, MTP off | 131.337 tok/s | 134.166 tok/s | 2.2% higher |
| Cold decode, MTP off | 9.328 tok/s | 12.655 tok/s | 35.7% higher |
| Cold decode, MTP on | 9.966 tok/s | 13.603 tok/s | 36.5% higher |
| Continuation decode, MTP off | 9.199 tok/s | 12.807 tok/s | 39.2% higher |
| Continuation decode, MTP on | 10.279 tok/s | 15.013 tok/s | 46.1% higher |
| CPU expert branch diagnostic | 7.074 s | 5.392 s | 23.8% lower |

All six cross-arm HTTP response comparisons matched exactly. Both 192-layer diagnostics conserved GPU H2D bytes (71,158,716,160), stream routes (2,653,326), CPU misses (181,474), and CPU weight reads (162,222,344,592). Both process snapshots reported `Cpus_allowed_list: 0-31` and `Mems_allowed_list: 0-1`.

This isolates worker-pool parallelism from CPU-affinity policy: even on the same 32 physical CPUs, 64 expert workers substantially outperform 32. The result supports 64 as the current experimental reference but does not establish that 64 is the optimum.

Next: a bounded 64/80/96-worker saturation sweep on the identical physical-core mask, with the same 7K workload, route policy, cache, BF16 KV, MTP states, traffic conservation, and three-repeat crossover.


## Physical-core worker saturation — run 37953230669

[Workflow run 37953230669](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37953230669) completed successfully. All arms used CPUs 0–31 and both NUMA memory nodes.

| Metric | 64 workers | 80 workers | 96 workers | Best |
|---|---:|---:|---:|---|
| Cold TTFT, MTP off | 52.961 s | 52.715 s | 52.561 s | 96 |
| Cold prefill, MTP off | 134.290 | 134.918 | 135.313 tok/s | 96 |
| Cold decode, MTP off | 12.421 | 12.708 | 12.573 tok/s | 80 |
| Cold decode, MTP on | 13.520 | 15.835 | 15.211 tok/s | 80 |
| Continuation decode, MTP on | 14.969 | 16.304 | 15.884 tok/s | 80 |
| CPU expert branch diagnostic | 5.480 s | 5.153 s | 5.013 s | 96 |

All six three-arm HTTP comparisons matched exactly. Each 192-layer diagnostic conserved GPU H2D bytes (71,158,716,160), stream routes (2,653,326), CPU misses (181,474), and CPU weight reads (162,222,344,592).

Eighty workers is the best overall setting tested: compared with 64, cold decode improves 2.3% without MTP and 17.1% with MTP, while continuation MTP decode improves 8.9%. Ninety-six workers reduces CPU-branch time and marginally improves prefill, but scheduling overhead lowers decode versus 80. Treat 80 as the current experimental reference; production defaults remain unchanged.

Next: one narrow 72/80/88 refinement on the identical physical-core mask. This is the final worker-count sweep unless it materially displaces 80.
