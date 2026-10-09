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


## Physical-core worker peak refinement — run 37964348309 attempt 2

The first attempt stopped in the GPU-headroom preflight with only 2,291 MiB free and did not build or load the model. The retry completed on the same source and preserved the 32-physical-CPU mask, inherited memory policy, warm existing file cache, minroutes20/auto256 grouped CPU fallback, 64 fixed GPU expert slots/layer, BF16 KV, mmap PLE, CUDA Graph off, and SV7 TensorOp off.

| Workers | Cold decode, MTP off (tok/s) | Cold decode, MTP on (tok/s) | Continuation decode, MTP on (tok/s) | Cold prefill, MTP on (tok/s) | CPU branch (s) |
|---:|---:|---:|---:|---:|---:|
| 72 | 13.408 | 14.495 | 15.213 | 131.454 | 6.020 |
| 80 | 12.923 | 15.186 | 15.528 | 131.340 | 5.133 |
| 88 | 12.684 | 16.028 | 16.519 | 131.610 | 5.073 |

All six timed cross-arm response comparisons were exact. Each separate 192-layer traffic diagnostic conserved 71,158,716,160 GPU H2D bytes, 2,653,326 streamed routes, 181,474 CPU-miss routes, and 162,222,344,592 CPU weight-read bytes. Readiness evidence kept CPUs 0–31 and memory nodes 0–1 in every arm.

The optimum is mode-dependent: 72 workers led non-MTP cold decode, while 88 workers led MTP cold and continuation decode. Further worker-count subdivision is not warranted. Production defaults remain unchanged. The next bounded comparison fixes the MTP-oriented 88-worker peak and physical-core placement, varying only CUDA Graph off versus on.


## CUDA Graph off/on at the 88-worker MTP peak — run 37977137136

The guarded 4.9 MB artifact independently confirmed that every server used CPUs 0–31, memory nodes 0–1, 88 expert workers, inherited memory policy, BF16 KV, minroutes20/auto256 grouped CPU fallback, mmap PLE, and identical 7,111-token traffic. Server startup records reported CUDA Graph disabled in the off arm and enabled in the on arm.

| Metric | Graph off | Graph on | Graph-on change |
|---|---:|---:|---:|
| Cold prefill, MTP off | 135.095 | 134.903 tok/s | -0.1% |
| Cold decode, MTP off | 12.699 | 12.736 tok/s | +0.3% |
| Continuation decode, MTP off | 12.577 | 12.777 tok/s | +1.6% |
| Cold decode, MTP on | 15.124 | 15.189 tok/s | +0.4% |
| Continuation decode, MTP on | 15.696 | 15.703 tok/s | +0.0% |
| CPU expert branch | 5.200 | 5.048 s | -2.9% |

All six cross-arm response comparisons were exact. Both separate 192-layer diagnostics conserved 71,158,716,160 GPU H2D bytes, 2,653,326 streamed routes, 181,474 CPU-miss routes, and 162,222,344,592 CPU weight-read bytes.

CUDA Graph is effectively neutral in this CPU-expert-bound workload and does not justify changing the existing graph-off experimental baseline. Production defaults remain unchanged. The next bounded comparison fixes graph-off, 88 workers, placement and workload while testing the supported Flash-Next MTP draft windows 2, 3 and 4.


## Flash-Next MTP draft-window sweep — run 37980713153

The guarded 7.2 MB artifact verified actual MTP draft counts 2/3/4, CUDA Graph off, CPUs 0–31, memory nodes 0–1, 88 expert workers, and the unchanged four-chunk workload.

| Draft window | Cold decode (tok/s) | Continuation decode (tok/s) | Cold accepted / drafted | Continuation accepted / drafted |
|---:|---:|---:|---:|---:|
| 2 | **16.597** | **17.941** | 33 / 58 | 35 / 54 |
| 3 | 15.126 | 15.701 | 37 / 74 | 38 / 72 |
| 4 | 13.607 | 14.797 | 38 / 93 | 40 / 87 |

MTP2 was 9.7% faster than MTP3 for cold decode and 14.3% faster for continuation decode. MTP4 reduced verification rounds slightly but its additional draft verification cost outweighed the extra accepted tokens. All three timed cross-arm comparisons were exact, and each 192-layer diagnostic conserved 71,158,716,160 GPU H2D bytes, 2,653,326 streamed routes, 181,474 CPU-miss routes, and 162,222,344,592 CPU weight-read bytes.

MTP2 is the current performance leader, but draft window 1 remains untested. The final bounded draft-window comparison therefore holds the full workload fixed and tests MTP1 versus MTP2. Production defaults remain unchanged.

## Final MTP1/MTP2 confirmation — run 37987057100

[Run 37987057100](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37987057100) and hosted checks passed at `1a651a988d2079bd25f9e58773c7c57ad9744ae4`; artifact 11644158159 independently reviewed. At fixed 88 workers and physical-core affinity, MTP2 versus MTP1 gave cold decode 16.607 versus 16.299 tok/s (+1.9%), continuation 17.880 versus 16.300 (+9.7%), and cold TTFT 53.906 versus 53.908 seconds (unchanged). All three timed paired response signatures were exact. Each separate 192-layer prefill diagnostic conserved 71,158,716,160 expert-weight H2D bytes, 2,653,326 streamed routes, 181,474 CPU misses and 162,222,344,592 CPU weight-read bytes. Use MTP2 as the experimental reference for this workload; no default or numerical-qualification claim.

## Next isolated prefill comparison: static cache budget 64 versus 128

Hold the same 7111-token/four-chunk input, 8192 context, 88 workers, physical-core mask, MTP2, BF16 KV, auto256/minroutes20, grouped fallback, mmap PLE and graph-off fixed. Double only static GPU residency from 64 to 128 experts per layer using the same full per-layer profile ranking. Cache weights grow from 8,494,252,032 to 16,988,504,064 bytes (+7.91 GiB); the existing runtime reserve/planner remains authoritative. Require actual capacity and complete seeding, not merely the requested maximum. Three fresh timed servers per arm in alternating order plus two separate 192-layer diagnostics compare TTFT, decode, expert-weight H2D, CPU work and resident-hit routes. Sample GPU memory at 5-second intervals; do not call it an exact allocation high-water mark. Different cache residency can select different CPU/GPU arithmetic, so preserve cross-arm outputs as diagnostics; unchanged same-path replay/accounting and numerical thresholds remain required. No model copies, modified weights or installed Strata changes. This directly tests SV1/SV3 residency versus repeated staging before more kernel or policy sweeps.
