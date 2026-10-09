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
