# Single-GPU serving performance

Tested Git revisions:

- Qwen3.8-27B NVFP4 MTP0 context-length serving:
  `f08597d6eaafce5b875934aaa85854fcd5426df8`;
- Qwen3.8-27B NVFP4 MTP3 single-request and concurrent fixed-corpus serving:
  `32c9881b6783949df4999422a764b3dcaa111b13`;
- Qwen3.8-27B groupwise-int MTP0 context-length serving:
  `5e4bf313cb2f8b0603e00bf3b42e7ab3ec6d927a`;
- Qwen3.8-27B groupwise-int MTP3 single-request and concurrent fixed-corpus serving:
  `d9dbe1ce4d1a53deec2349e669a429a000c54d01`;
- Concurrent MTP3 decode saturation for the three measured Qwen3.6 artifact profiles:
  `26da9df7c1b3d3c04ea7bbd730271aa01d00742a`;
- Refreshed Qwen3.6-35B-A3B and Qwen3.6-27B NVFP4 MTP3:
  `f4f21cc36bd1a83cbc046f668719d591dc9c1e2e`;
- Qwen3.6-35B-A3B stored MTP3 response audit:
  `b1a220f028aa750f75bceb3522ac00bbaab7e42d`;
- Qwen3.6-35B-A3B DFlash block=8 (`k=7`):
  `0dc94097e8ec5c5bcf59b9e13e9d1852f504eb61`;
- Qwen3.6-27B NVFP4 accuracy and MTP0:
  `b3d4d0f50b868711c62432bbd68e746217a2f49a`;
- Qwen3.6-27B groupwise-int MTP3: `5ea3242a206cdb0c4c1beaeb9d8a3048e6248423`;
- Qwen3.6-35B-A3B MTP0 and Qwen3.6-27B groupwise-int MTP0:
  `0795169393cab0f2c16246d4bac20dee735dc2a4`.

The Qwen3.6 measurements characterize its three registered artifact profiles independently on one
NVIDIA GeForce RTX 5090. They cover long-context prefill and baseline decode with speculative
decoding disabled, plus long-reasoning and cross-scenario decode with MTP and DFlash. The Qwen3.6
concurrent decode-saturation campaign measures all three profiles at C=1, 2, 4, and 8. Both
Qwen3.8-27B weight profiles cover the MTP0 long-context profile and the complete MTP3
speculative-decode corpus at C=1, 2, 4, and 8; each C=1 point also supplies the corresponding
single-request MTP3 results below.

The single-request corpus requests were submitted serially to a persistent `ninfer-serve` process
over the loopback OpenAI-compatible HTTP endpoint. Each reported corpus fixture used five fixed
seeds. Values are arithmetic mean ± sample standard deviation, and server warm-up completes before
the measured requests. The concurrent campaign has its own sustained-wave method below.

## RTX 5090 NVFP4 W4A4 prefill schedule

The desktop NVIDIA GeForce RTX 5090 selects token-fast CTA rasterization and one
activation-scale TMA fetch per K-tile pair for NVFP4 Linear and LinearSwiGLU.
Selection uses the device name and SM120 capability, not capability alone: the
RTX PRO 6000 retains weight-fast rasterization and per-tile scale fetches because
the combined change regressed its measured 7,680-token prefill from 741.4 to 749.9 ms.
Other devices retain that schedule as well. The existing full-tile TMA eligibility
is unchanged; decode and non-TMA prefill routes are unchanged.

[Issue #22](https://github.com/igorls/ninfer/issues/22#issuecomment-5741594802) and
its [raw reports and reproduction scripts](https://gist.github.com/patrickscd/a1f67f1b693962d6cd6a826748cc24a3)
record the September 19, 2026 RTX 5090 A/B: baseline `5e4a66d0` versus that baseline
plus the two-file patch from `1acfff7d`. Five alternating fresh-process pairs per
mode, one discarded warmup per prompt, and 256 timed decode tokens used
Qwen3.8-27B NVFP4, 32,768 context/KV capacity, FP8 KV, 1,024-token prefill chunks,
CUDA Graphs, disabled prefix reuse, and a 2 GiB desktop reserve. Windows Ninja
Release used nvcc 13.3.33, MSVC 14.50.35503, driver 616.92 and a 480 W power limit.
Desktop applications remained open.

| Prompt tokens | Prefill gain, MTP0 | Prefill gain, MTP3 |
|---:|---:|---:|
| 512 | 4.13% | 4.35% |
| 4,096 | 5.83% | 5.21% |
| 8,192 | 5.83% | 5.72% |
| 16,384 | 5.12% | 5.24% |

Mean decode differences ranged from -0.31% to +0.26%. The submitted Linear A4
and LinearSwiGLU numerical checks passed for both arms; a fixed greedy 512-token
prompt produced identical text and token IDs across arms in each speculation mode.
These gains apply to the measured workload, not all chunk sizes or other GPUs.
The optional 35B MoE cache-hint measurement was waived because the contributor
does not run that artifact.

## Qwen3.8-27B DFlash2 on RTX PRO 6000

Native Windows qualification on September 7–8, 2026 uses the RTX PRO 6000 Blackwell 96 GB,
MSVC 19.51, CUDA 13.3.33 and `sm_120a`. The target is the existing Qwen3.8-27B NVFP4 artifact
with every base payload preserved, extended with the
[incoai DFlash2 companion](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).
See the [artifact conversion](maintainer/qwen3.8-27b-artifact.md) and
[execution/state contract](maintainer/qwen3.8-27b-dflash2.md).

The matched serving campaign uses one final Release executable, FP8 KV, CUDA graphs, capacity
eight, a 32,768-token context limit and KV pool, 2,048-token prefill chunks, and no prefix reuse
or Vision allocation. Each process receives a 256-token warmup before two repetitions. The
Python and translation fixtures use non-thinking mode; the AIME fixture uses `xhigh` reasoning.
Sampling is greedy with no presence/frequency penalty and a 1,024-token output budget. Values
are committed output tokens divided by complete client wall time, including prefill. At eight
active requests, the numerator sums the wave and the denominator is its makespan. This is not
a steady-state decode-only measurement. Naturally completed translations remain valid samples.

| Active requests | Workload | Ordinary tok/s | MTP5 tok/s | DFlash2 K7 tok/s |
|---:|---|---:|---:|---:|
| 1 | Python | 64.7 | 183.7 | **195.6** |
| 1 | Translation | 64.5 | 169.0 | **175.7** |
| 1 | Mathematics reasoning | 65.1 | 159.4 | **187.0** |
| 8 | Python | 425.3 | **1,016.3** | 987.4 |
| 8 | Translation | 389.5 | **896.5** | 821.7 |
| 8 | Mathematics reasoning | 423.2 | 827.5 | **839.3** |

These are two-repetition means, not confidence intervals. DFlash2 improves the single-request
wall rate by 4–17% over MTP5 in these cases, and by 2.7–3.0x over ordinary decoding. At eight
requests, MTP5 leads on Python and translation; the mathematics difference is small. Start with
`--spec dflash2 --draft-tokens 7 --lm-head-draft` for interactive decode, and MTP5 with the
optimized head for concurrent Python/translation throughput. Backend selection remains fixed
at Engine startup; these observations do not establish a universal winner at other contexts.

Strict JSON reaches 64.0 / 61.2 / 61.9 tok/s with ordinary / MTP5 / DFlash2 at one active request,
and 383.8 / 374.9 / 377.5 aggregate tok/s at eight. Every result passes the schema validator;
the single-request JSON text is identical across the three modes. In a separate paired
capacity-one experiment, replacing padded zero-draft verification with the width-one target
route improved DFlash2 JSON from 52.9 to 63.0 tok/s, about 19%, without a clear unconstrained
throughput regression. Backend state maintenance still has a small cost.

For the 7,680-token long-context execution probe, mean prefill / server TTFT is 764.8 / 770.0 ms
with ordinary decoding, 811.6 / 816.5 ms with MTP5, and 812.6 / 817.6 ms with DFlash2. DFlash2
does not improve prefill in this comparison; its benefit is generated-token decode.

| Startup mode, capacity eight | Materialized weights GiB | Runtime reservation GiB |
|---|---:|---:|
| Ordinary | 18.976 | 2.578 |
| MTP5, optimized head | 19.729 | 3.390 |
| DFlash2 K7, optimized head | 21.383 | 6.128 |

The runtime reservation includes the KV pool, state/workspace and unallocated CUDA-graph
headroom. These values describe this 32K configuration; they are not measured peak process VRAM.

The fixtures are `scenario_code_python`, `scenario_translation_markdown`, and
`long_decode_aime26_15` under `examples/cli/messages/`. A separate strict JSON-schema case asks
for twelve job-runner implementation steps and is independently validated. The `long_niah_8k`
fixture exercises the longer attention path; its short answer and explicit answer in the prompt
make it an execution/TTFT probe, not retrieval-quality evidence.

An earlier capacity-one sweep tested draft counts 2, 3, 5, 7, 11 and 15 and both proposal-head
routes. K=7 with the optimized head gave the best observed DFlash2 results for these workloads.
Larger draft blocks did not compensate for their additional verification work. Keep that sweep
separate from the final capacity-eight comparison.

Profiling a K=7, capacity-one native benchmark with 512 prompt tokens and 256 output tokens
shows 2,038 ms of kernel execution inside a 2,131 ms decode range. Of summed kernel time,
89.8% is target verification and 8.5% is drafting plus selection. The largest contributors
are target NVFP4 gate/up (455 ms), fused FP8 GDN input (377 ms), and NVFP4 down (288 ms).
These measurements identify target verification as the next optimization focus; they do not
measure hardware occupancy or bandwidth saturation. A forced A8 GDN input route was slower,
and materialized A16 did not improve its fused baseline; both experiments were reverted.
Nsight Compute counters were unavailable (`ERR_NVGPUCTRPERM`).

Qualification passes the independent numerical Op checks, variable-width sparse acceptance,
graph and eager/full-head execution, K=7/K=15 with eight requests, partial terminal blocks,
cancellation, page/ring boundaries, Host restore, and image/video input. A reproduced concurrent
admission failure now waits for an unfinished StateImage fork to settle; the CPU regression and
repeated real pressure runs pass. MTP5 and DFlash2 K7 each pass all 21 live structured-output
protocol checks. All-constrained batches use width-one target execution while maintaining the
selected backend state; mixed batches retain masked speculative execution.

Actual response text was read manually. Translations retain the required table, code and
identifiers, with prose differences; Python and mathematics hit their output budgets and cannot
establish completed-code or final-answer quality. The JSON plans are valid and relevant to the
request. Greedy text can differ across verification widths because qualified FP8/NVFP4 arithmetic
routes differ. This integration is not advertised as bitwise lossless, and these performance
fixtures do not establish Tribuno legal-workflow acceptance. The real checkpoint qualification
here is NVFP4; it does not qualify every supported weight profile or production context length.

Local request text, timings and counters are under `profiles/bench/dflash2-20260907/final-*`;
the phase breakdown is under `profiles/nsys/dflash2-20260907/`. Source changes are uncommitted.

## Qwen3.8-Flash-Next MTP on RTX PRO 6000

The September 6, 2026 Windows Release build was measured on an NVIDIA RTX PRO 6000
Blackwell Workstation Edition (96 GiB), CUDA 13.3, using the mixed Flash-Next artifact.
The loader quantizes this artifact's BF16 MTP expert bank to NVFP4; these measurements
use that existing path and require no replacement checkpoint.

Both modes used fresh persistent servers, completed startup warm-up, and received the
same three requests serially. Settings were 262,144 maximum context, 786,432 KV tokens,
eight maximum active requests, 8,192-token prefill chunks, FP8 KV, BF16 GDN state,
Vision enabled, and 3 GiB desktop reserve. Requests used temperature 0, seed 42,
top-p 0.8, top-k 20, presence penalty 1.5, reasoning disabled, and no streaming.
Every measured request selected a root prefix and reached its stated output limit.

MTP used `--spec mtp --draft-tokens 4`, eager draft steps, and captured sequential
verification. Decode throughput below is `(completion_tokens - 1) / decode_seconds`;
prefill and transport time are excluded. Each row is one matched observation,
not a repeated corpus average or a general speedup guarantee.

| Workload | Prompt / output tokens | Speculation off tok/s | MTP4 tok/s | Decode speedup | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| CPU execution explanation | 43 / 320 | 85.53 | 92.94 | 1.09x | 193 / 502 (38.4%) |
| Detailed CPU tutorial | 54 / 1,200 | 82.03 | 97.66 | 1.19x | 803 / 1,581 (50.8%) |
| CPU question after 600 synthetic catalog records | 18,537 / 320 | 55.82 | 79.56 | 1.43x | 204 / 460 (44.3%) |

Accepted drafts by position were `[91,56,28,18]`, `[318,230,154,101]`, and
`[81,58,40,25]`, respectively. All four draft positions contribute. A repeated
18.5K-context request reused 18,530 prompt tokens and preserved the output and
acceptance counts, including a production recheck with captured verification.
A chart-image request and a
three-request concurrent wave also completed successfully. Flash-Next currently
speculates only at decode batch size one; larger batches use ordinary batched
decode while keeping the MTP teacher state current.

The MTP cache, rollback slots, and graph captures consume part of the runtime memory
budget. At these settings the planner selected 16 private continuation slots from
the requested maximum of 48. The maximum active batch and KV token capacity remained
eight and 786,432. These results do not qualify full real-checkpoint FP32 oracle
parity; stem mathematics, state transactions, and captured verification have focused
numerical and integration checks described in the model reference and oracle tools.

### Device-resident MTP drafts

On the same RTX PRO 6000 and mixed artifact, six greedy code-generation requests
produced 640 tokens each: two prompts at each context length below. The requests were
identical across fresh servers, with MTP4, concurrency capacity eight, a 262,144-token
context limit, and 655,360 KV tokens. The smaller KV pool left desktop headroom during
this test window. Values are the mean of the two requests at each length.

| Prompt tokens | Serialized eager drafts tok/s | Device-resident captured drafts tok/s | Observed change |
|---:|---:|---:|---:|
| 74 | 190.29 | 191.51 | +0.6% |
| 33,348 | 120.90 | 122.86 | +1.6% |
| 77,767 | 115.49 | 116.76 | +1.1% |

All response text and draft-acceptance counters were identical across the six requests.
Removing intermediate readbacks without graph capture gave no clear improvement. The
captured result is a small observed gain from a limited sample, not evidence of a large
CPU bottleneck. Root TTFT was effectively unchanged: about 4.8 seconds at 33K tokens and
11.5 seconds at 78K. Draft GPU waits are now classified as device-wait time; the earlier
host-exposure bucket included those waits and must not be interpreted as CPU execution
or GPU idle time.

### Flash-Next phase profiling: September 6, 2026

Nsight Systems 2026.1.3 on the RTX PRO 6000 (188 SMs, 128 MiB L2,
advertised memory bandwidth 1.792 TB/s), driver 596.86, CUDA 13.3, Windows
Release build. This campaign profiles Flash-Next alone; it is not a cross-model
comparison. The mixed artifact uses NVFP4 routed experts, FP8 attention projections,
and a BF16 output head. Runtime settings: concurrency capacity eight, one active
request, 262,144 maximum context, 655,360 KV tokens, 8,192-token prefill chunks,
FP8 KV, BF16 GDN state, Vision allocated, 3 GiB desktop reserve. MTP uses four drafts.

The workload requests a Python LRU-cache implementation with expiration, locking,
and tests, either alone or after a 125,000-character snapshot of engine source.
Requests are greedy, seed 42, presence penalty zero, reasoning disabled, with 256
output tokens. Unique leading labels ensure root admission; every request asserts
zero cached tokens and the complete output budget. A short and long eight-token
request warm the relevant routes. Two untraced observations per length establish
the baseline; collection is stopped during those observations, though the process
is still launched through Nsight. Label differences can change generated content
and MTP acceptance, so this is not an exact-output profiler-overhead experiment.

| Context | Ordinary decode tok/s | MTP4 decode tok/s | Ordinary root TTFT | MTP4 root TTFT |
|---|---:|---:|---:|---:|
| 54 tokens | 132.56–132.59 | 168.20–187.29 | 59–60 ms | 64–103 ms |
| 30,293 tokens | 65.76–65.80 | 116.80–123.29 | 4.379–4.380 s | 4.320–4.325 s |

Decode rates use 255 post-first-token outputs divided by the logged decode time.
The `timings_seconds.prefill` field was zero in these recordings, even on the long
requests. Prefill attribution below uses NVTX ranges; the later timing correction
does not retroactively supply phase measurements for these runs.

Software CUDA tracing from process start, with graph-node tracing enabled, captures
the complete ordinary and MTP decode kernels. An initial hardware trace started
after graph creation recorded graph launches but omitted decode nodes; it is not
used for decode conclusions. The measured requests are `generate` ranges 3 and 4
in the complete traces, after startup and the two warm-up requests.

| Phase | NVTX wall time | Kernel-active fraction of wall | Largest kernel costs (% of summed kernel time) |
|---|---:|---:|---|
| Ordinary short decode, 255 rounds | 1.939 s | 94.3% | FP8 GDN input 13.7%; MoE gate/up 12.2%; BF16 output head 10.9%; hyper norm 10.3% |
| Ordinary 30K decode, 255 rounds | 3.877 s | 96.2% | sparse attention 53.4%; FP8 GDN input 6.8%; MoE gate/up 6.0% |
| MTP4 short decode, 78 rounds | 1.327 s | 94.9% | BF16 output head GEMV 19.1% plus verification head 4.8%; MoE gate/up + down 26.7% |
| MTP4 30K decode, 84 rounds | 2.241 s | 95.4% | sparse attention 35.1%; both output-head paths 15.3%; MoE gate/up + down 17.1% |
| Ordinary 30K prefill | 4.283 s | 96.8% | sparse-attention MMA 35.6%; MoE gate/up MMA 19.9%; MoE down MMA 7.7% |

The MTP long-prefill breakdown is similar. Graph-launch correlation IDs separate
drafts from the final target launch in each MTP round: draft kernels account for
25.6% and 26.2% of decode GPU kernel time at short and long context, respectively.
The remaining graph kernel time is target verification. The traced requests accept
177/308 and 171/336 drafts and emit 255 decode tokens in 78 and 84 rounds.

These measurements distinguish three limits:

- **Long-context decode is primarily limited by the sparse-attention implementation.**
  `sparse_attention_kernel` takes 0.182 s at short context and 1.990 s at 30K,
  explaining about 93% of the 1.937-second increase in ordinary decode time.
  Its single-token launch is 24 CTAs of 256 threads, one CTA per query head, on
  188 SMs. That permits at most 24 simultaneously participating SMs for this kernel.
  Each CTA walks selected-token chunks and performs a serial 256-feature dot
  product per thread, with block-wide reductions and barriers. Splitting selected
  tokens across more CTAs, with a stable softmax merge, is the first implementation
  direction to qualify. MTP verification launches up to five token rows together,
  but the same kernel remains the largest long-context cost.
- **The output head is already consistent with a bandwidth limit.** Its
  `[248320,2560]` BF16 matrix is 1.271 GB, versus 128 MiB L2. The approximately
  0.779 ms GEMV implies a one-pass weight rate of 1.63 TB/s, about 91% of advertised
  device bandwidth. This is an estimate from known weight bytes and measured time,
  not a DRAM-counter measurement. MTP pays this cost repeatedly for draft tokens;
  a large improvement here would likely require reducing transferred head bytes
  or head evaluations, with separate numerical and acceptance qualification.
- **Prefill needs attention and MoE work; CPU launch cleanup has a small ceiling.**
  The QSA and two MoE MMA kernels together consume 63.1% of prefill kernel time.
  Across measured phases, kernels occupy 94–97% of wall time. This is GPU activity,
  not SM utilization: a 24-CTA kernel can keep the timeline busy while leaving most
  SMs unused. CUDA API duration also includes queue backpressure and device waits;
  it must not be added to GPU duration or described as pure CPU work.

As an attribution bound, halving only the long-context attention kernel would
improve ordinary decode by about 1.35x and MTP decode by about 1.20x, if all other
costs remain fixed. Halving only the dominant prefill attention kernel gives about
1.21x. These are Amdahl estimates, not achieved speedups. No inference implementation
was changed by this profiling campaign.

The initial Nsight Compute attempt failed with `ERR_NVGPUCTRPERM`; the elevated
follow-up below resolves counter access. Neither campaign establishes an absolute
hardware throughput ceiling. Concurrent-request throughput is outside this C=1 sample.
The separately reported alternating-conversation retention defect is excluded by
the root-only workload and remains unresolved.

Local evidence is under `profiles/nsys/flash-next-20260906/`: `off-full.nsys-rep`,
`mtp-full.nsys-rep`, their SQLite exports and phase summaries, request JSONL, and
the source-context snapshot. Reproduction helpers are
`tools/bench/profile_flash_next.py --context-file <snapshot> --label <unique-run>
--output <json>` and `tools/bench/nsys_phase_summary.py <trace.sqlite>`.
Launch the isolated server through Nsight with
`--trace=cuda-sw,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none`,
capture from process start, warm both lengths, then issue the measured requests.
Export with `nsys export --type=sqlite`. The summary tool rejects traces with
decode ranges but no decode kernel nodes.

#### Elevated hardware counters

Nsight Compute 2026.2.0 ran elevated on the same GPU and artifact. The public CLI
used the same source snapshot, 30,301 prompt tokens, eight generated tokens, zero
prefix reuse, a 32,768-token context/KV capacity, 8,192-token prefill chunks, FP8 KV,
BF16 GDN state, Vision allocations, and ordinary decoding. This smaller C=1
allocation preserves the relevant operator shapes while leaving profiler headroom;
it is not the production concurrency/capacity configuration.

Kernel replay attempted to back up the model's memory and exceeded temporary disk
capacity. The successful capture uses application replay, graph-node profiling,
NVTX phase filters, and two samples per launch configuration. Ten application
passes collected the selected counters. Cache flushing and clock control are
disabled: the application recreates its own cache state on each replay, and timing
variation remains possible. These are representative operator samples, not a new
end-to-end speed measurement.

| Prefill kernel | Achieved occupancy | Tensor-pipe activity | DRAM throughput (% peak) | L2 sector hit rate |
|---|---:|---:|---:|---:|
| QSA, 8,192-token chunk | 8.33% | 1.54–1.55% | 0.44–0.45% | 98.56% |
| QSA, final 5,725-token chunk | 8.33% | 1.46–1.47% | 0.45% | 98.52–98.53% |
| MoE gate/up | 16.21% | 5.59–5.64% | 13.56–14.24% | 84.01–84.17% |
| MoE down | 24.26–24.62% | 7.63–8.02% | 28.94–33.31% | 72.05–72.47% |

Occupancy is the active-warp percentage relative to the SM's supported warp
capacity; it is not the fraction of SMs in use. Tensor activity and DRAM throughput
use the elapsed-time peak percentages reported by Nsight Compute. The requested
`dram__bytes_read.sum` metric was unavailable; no transferred-byte total is inferred
from it.

The prefill QSA launch has 132 registers per thread and 77,824 bytes of static
shared memory, plus 1,024 driver bytes. Shared memory limits it to **one 128-thread
block per SM**, explaining the four active warps out of 48 and the measured 8.33%
occupancy. Registers alone would permit three blocks. Smaller K/V staging tiles or
a different staging schedule are concrete candidates to admit more blocks and
hide latency; their extra loop/synchronization cost must be measured. The 98.5%
L2 hit rate and sub-1% DRAM throughput show that device-memory bandwidth is not
saturated in these attention samples.

MoE gate/up likewise permits only one resident block because it allocates 92,416
dynamic shared-memory bytes plus driver overhead. MoE down permits three blocks,
also limited by shared memory. The measured eligible warps per scheduler cycle are
approximately 0.087 for gate/up and 0.127–0.130 for down. Together with their low
tensor activity, these favor investigation of staging, load latency and scheduling
before treating either kernel as close to the tensor-compute ceiling.

For decode, the initial graph-node capture returned only 6–7 microseconds and
13,296 global-load requests per attention invocation, inconsistent with the
long-context timeline. Those two records are excluded from decode conclusions;
matching the kernel name and grid was insufficient to establish representative
work. A separate application-replay capture disabled CUDA graphs, selected only
the decode NVTX range, skipped the first 12 matching launches, and collected two
attention invocations. It also completed ten replay passes.

The eager samples take **648–653 microseconds**, matching the approximately
650-microsecond production-trace mean. They execute the same sparse-attention
kernel at grid `(24,1,1)` and block `(256,1,1)` and report:

- SM throughput approximately **1.41%** and DRAM throughput **0.19%** of peak;
- **16.67%** achieved active-warp occupancy, with only 24 blocks available for
  188 SMs;
- approximately **0.08 eligible warps per scheduler cycle**;
- a long-scoreboard stall ratio of approximately **20.5 per issue-active**, versus
  approximately **0.38** for barriers; these are profiler ratios, not percentages
  of kernel wall time;
- approximately **1.99 million global-load requests** and **7.15 sectors per
  request**, with an L2 sector hit rate of **91.5–92.9%**.

This combination supports insufficient parallel work and memory-dependency
latency as the first decode problems to address. It does not support device-memory
bandwidth saturation. The source's serial key dot products and strided key loads
give a concrete mechanism to change: distribute feature work across lanes and
selected-token work across blocks, then merge the partial softmax results. Reuse
across the 12 query heads sharing a KV head is also worth qualifying, but repeated
source-level loads must not be described as 12 times the DRAM traffic: caches can
serve them. The selected-block attention implementation below qualifies the new
arithmetic against an independent oracle and measures the Engine with graphs and MTP enabled.

Local evidence: `profiles/ncu/flash-next-20260906/targeted-app.ncu-rep`, its raw CSV
and details export, `decode-eager.ncu-rep` and its exports, `counter-summary.json`,
and the exact argument vectors in the corresponding `*.argv.json` files.

Operationally, the first production restoration failed its runtime-memory budget
check. The shared KV pool was reduced from 737,280 to 589,824 tokens to restore
service, retaining the 262,144 per-request context limit, concurrency eight, MTP4,
and the 3 GiB desktop reserve. This reduces aggregate KV capacity; it is not a
kernel optimization or a measured throughput improvement. The user subsequently restored
737,280 tokens; the September 7 optimization measurements use that restored capacity.

#### Selected-block attention optimization (September 7)

The profiler-directed change replaces serial decode attention with a coalesced,
16-partition softmax reduction and merge. Target decode and MTP share this Op and
reserve its complete caller-owned workspace. The default prefill route reuses
one shared-memory tile for K and then V, separated by synchronization. Its static
shared memory falls from 77,824 to 45,056 bytes; FP8 compilation retains 132
registers per thread and no local-memory allocation. This permits two blocks per
SM under the measured hardware's resource limits; achieved occupancy was not
remeasured for this candidate.

Paired Engine runs used RTX PRO 6000 Blackwell, CUDA 13.3/sm_120a, the mixed
Flash-Next artifact, one active request with capacity for eight, a 262,144-token
context limit, 737,280-token shared KV capacity, 8,192-token prefill chunks, FP8 KV,
BF16 GDN state, Vision allocations, a 3 GiB desktop reserve, and CUDA graphs.
Separate processes tested ordinary decoding and MTP4. Each process warmed both
lengths with eight outputs, then ran two measured requests per length with 256
outputs. The identical prompts contained 63 or 30,302 tokens. Every request
confirmed zero prefix reuse. Rates use 255 post-first-token outputs divided by
the logged decode time. These two-observation ranges are not confidence intervals.

| Mode and context | Baseline decode tok/s | Optimized decode tok/s | Baseline root TTFT | Optimized root TTFT |
|---|---:|---:|---:|---:|
| Ordinary, short | 134.44–134.55 | 147.23–147.63 | 0.062–0.063 s | 0.062–0.063 s |
| Ordinary, 30K | 67.08–67.14 | 132.34–133.22 | 4.302–4.304 s | 3.759–3.761 s |
| MTP4, short | 167.92–175.79 | 171.78–175.49 | 0.065–0.066 s | 0.067 s |
| MTP4, 30K | 118.70–122.01 | 168.50–176.85 | 4.333–4.339 s | 3.793–3.795 s |

Ordinary long-context decode improves approximately 98%, and short-context
ordinary decode approximately 10%. MTP long-context decode improves approximately
43% by the means of these observations; short-context MTP shows no clear gain.
MTP acceptance varies between runs, so these measurements do not establish a
universal speculation speedup. Root TTFT falls approximately 13% for both long
workloads. TTFT includes preparation, prefill, and first-token work; the existing
Flash-Next `prefill` timing field was zero in these runs and is not used as a phase measurement.
An intermediate decode-only binary retained approximately 4.31–4.35 s long TTFT,
isolating the later TTFT reduction to the prefill change.

Qualification uses an independent FP64 full-softmax oracle over represented BF16
and FP8 inputs, with a named normwise criterion, finite gross pointwise cap, and
exact empty-output checks. Coverage includes permuted pages and selection order,
causal boundaries, full 512-block selections, heterogeneous decode batches through
eight, prefill through 128 query tokens, amplified logits, and replay with changed
inputs. QSA integration tests pass for both cache formats and cache updates. The
decode change also passes the full Text executor, including small-context
workspace/capture checks and graph batches 1/2/4/8, continuation lifecycle, and CPU
resource-manager tests. Independent standards and numerical reviews were completed;
their workspace and contract findings were corrected.

Local evidence is under `profiles/bench/flash-next-20260907/`: `summary.json`,
paired request JSONL, numerical/integration logs, and compiled resource usage.
The root request helper is `tools/bench/profile_flash_next.py`; exact server
arguments are saved in each run's command JSON. These results describe the stated
single-active-request workloads, not concurrent throughput or a general hardware
ceiling.

#### Half-width MoE gate/up staging (September 7)

The routed NVFP4 gate/up prefill kernel now stages two K=1,280 tiles instead of
retaining K=2,560. The output tile remains 16 gate/up pairs by 32 tokens, and both
halves feed the same accumulators in the original forty K64 MMA steps. Dynamic
shared memory falls from 92,416 to 46,336 bytes and compiled registers from 106 to
57, with no spills. CUDA's occupancy query reports two resident 256-thread CTAs
instead of one on the RTX PRO 6000. This is a residency limit, not measured achieved
occupancy. Weight tiles are reloaded for each token chunk; the experiment measures
whether additional parallel work repays those copies.

The isolated kernel gate uses signed E2M1 codes, independently varied per-group
E4M3 scales, all 512 experts, and uniform routing or a skew that places 75% of tokens
among 32 experts. At T=512, 513, 2,048 and 8,192, every routed output is written and
finite, the unused shared-expert output remains a sentinel, and all routed BF16
values match the original kernel bitwise. The odd T=513 case exercises partial
token pairs. A no-op candidate fails the sentinel check. An independent FP64 oracle
decodes the represented packed inputs and stored scales, evaluates both complete
dot products and SiLU(gate)*up, and checks 256 selected outputs across these cases.
The maximum relative error is 0.003785, within the 0.0041 BF16 rounding criterion
(denominator floor 1e-4). Bitwise implementation comparison supplements that oracle.

Six alternating A/B observations per case time a graph containing twenty kernel
nodes. Median kernel-time reductions are 18.5%, 17.6%, 13.4% and 4.8% for uniform
routing, and 17.2%, 11.3%, 12.7% and 9.8% for skewed routing at those four extents.
These are operator measurements, not request speedups.

Public Engine qualification uses native Windows Release, MSVC 19.51, CUDA 13.3.33,
the same mixed Flash-Next artifact and BF16 output head, FP8 KV, BF16 GDN state,
Vision allocations, CUDA graphs, an 8,192-token prefill chunk, a 65,536-token context
limit, a 196,608-token KV pool, capacity for eight requests and one active request.
The unchanged kernel and candidate are built from the same source/toolchain and
run in separate processes, with ordinary decoding and MTP4 tested independently.
An initial comparison against the older production executable also changed
untouched decode throughput substantially; it is excluded from attribution.

Each process warms a short and long prompt, then runs three fixed root prompts per
length (52, 3,952 or 30,282 tokens), with 256 outputs, temperature zero, seed 42 and
reasoning disabled. Root requests assert zero prefix reuse. Each long request is
followed immediately by a continuation with 128 outputs. All 60 requests across
the four processes pass, including one schema-constrained request per process.

| Mode and root prompt | Baseline TTFT | Candidate TTFT | Mean TTFT reduction |
|---|---:|---:|---:|
| Ordinary, 3,952 tokens | 0.515–0.520 s | 0.505–0.508 s | 2.2% |
| Ordinary, 30,282 tokens | 4.074–4.094 s | 4.022–4.042 s | 1.3% |
| MTP4, 3,952 tokens | 0.525–0.532 s | 0.515–0.524 s | 1.7% |
| MTP4, 30,282 tokens | 4.192–4.230 s | 4.111–4.135 s | 2.2% |

The primary production-mode result is about 92 ms less root TTFT on the 30K MTP4
workload. Unchanged short MTP decode averages 165.22 versus 165.02 committed tok/s;
long MTP decode averages 135.04 versus 135.94 tok/s. Ordinary decode also varies
slightly between processes. No decode-speed improvement is attributed to this
prefill change. Ranges describe three observations, not confidence intervals or a
universal gain. The Flash-Next `prefill` timer was zero in these runs, so TTFT is
reported rather than relabelled as isolated prefill time.

Baseline and candidate return identical text in every paired request and identical
MTP acceptance counts at every draft position. This is a regression observation,
not a legal-quality evaluation. Continued prompts reuse 30,275 of 30,561 prompt
tokens; both variants deliver approximately 0.20–0.21 s TTFT for that short suffix,
without a material reuse gain. The actual-launcher MoE integration test passes.
Production was restored to its original executable after isolated qualification; this result
does not by itself claim deployment of the candidate.

Local evidence is in the isolated worktree's `profiles/moe-staging/`: the corrected
oracle/graph harness and negative control, `moe-test.log`, paired request/response
records, exact commands, VRAM snapshots and `paired-aggregate.json`. This campaign
does not measure concurrent-request throughput or the vendor's 90% prefix-hit point.

The initial upstream RMSNorm candidate `9954867a` is deferred for Flash-Next.
Its changed D=2,560 embedding route accounts for only 1.19–1.30 ms across the complete
short/long MTP decode ranges in the retained September 6 traces (0.09%/0.06% of
summed kernel time). Flash-Next's D=10,240 hidden norm uses the generic kernel,
which that commit does not change, and its gated 170-SM crossover is not reached.
Those traces predate the selected-block attention improvement above; the tiny
absolute time bound does not justify a full Engine experiment for this cherry-pick.

### Request-owned prefill timing

Flash-Next now accumulates the execution time of each `advance_prefill` call in
the admitted request and exports completed work on finish, abort and cancellation.
The sum includes host submission and device completion waits, while excluding
scheduling gaps and checkpoint materialization outside those calls. Vision uses
the current call's encode-time delta and is reported separately. Timings reset at
admission and do not travel with reusable checkpoints. TTFT remains wall time.

Focused reuse and Vision fixtures check chunk accumulation, eight reused turns,
admission reset, partial-work cancellation, and image/text request isolation. A
real-artifact check on Windows/RTX PRO 6000 reported 0.4164 s prefill for 2,290 cold
tokens and 0.0750 s for a continuation computing 48 tokens after reusing 2,283.
Both values were positive and below TTFT. An active background upload makes these
accounting checks unsuitable for a performance comparison. For reused requests,
divide computed prompt tokens by prefill time, rather than counting the cached
prefix as newly computed work.

### Cold host PLE and repeated-turn prefill

On the same hardware and serving settings with MTP4 enabled, a root request containing
about 43.7K tokens of C++/CUDA source took 51.74 seconds to return one token. A temporary
host timer attributed 46.1 seconds to synchronous compressed PLE gathering. After startup
warmed the 32.0 GB mapped PLE codes/scales, the same source workload took 6.51 seconds;
a previously unrequested source region (41.8K tokens) took 6.17 seconds. Each request had
zero prefix reuse. These are individual client wall-time observations, with a changed
request label to prevent reuse; they isolate a cold host-page problem, not MTP decode cost.
The warm pass took 25.52 seconds during startup and does not pin pages against OS reclamation.

A separate 48-turn synthetic conversation began at 18.5K prompt tokens and ended at 52.8K.
With exact checkpoint ownership and consumption, every follow-up advanced the reuse frontier,
computed only 735–736 tokens, and completed in 0.18–0.22 seconds. Before this correction,
reuse stopped advancing after turn 6; turn 17 recomputed 8,751 tokens and took 1.36 seconds.
The corrected run maintained two checkpoint state slots rather than accumulating old owners.
From an empty production cache, eight concurrent conversations completed three turns each:
all 24 requested integer sequences were correct, and all 16 follow-ups reused the expected
checkpoint (1,602 then 1,680 tokens). Cache capacity remains bounded: a separate run with
an older ninth conversation already cached had two misses among 16 follow-ups.

A controlled empty-cache conversation with a 33K-token initial prompt, two supplied
assistant reasoning/tool-call messages, and a new user message isolates another reuse
limit. Both modes reused 33,349 and then 33,393 tokens during the tool sequence. With
`preserve_thinking=false`, the new user message removed the earlier reasoning and caused
root prefill: 33,441 tokens, 4.799 seconds TTFT. With `preserve_thinking=true`, the new
user request reused 33,437 tokens and took 0.088 seconds TTFT. This is a reproduction of
the missing inherited turn-start checkpoint, not a claim that the default-mode miss is
fixed. The frontend test separately verifies the corresponding rendered-prefix change.

## Single-request serving performance method

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| CUDA compile/runtime | 13.1 / 13.1 |
| CUDA driver API | 13.3 for Qwen3.8, NVFP4, and refreshed 35B MTP3; 13.1 for the remaining single-request campaigns |
| Request mode | One active request, `stream=false` |
| Maximum context | 262,144 tokens; 131,072 for Qwen3.8 MTP3 and refreshed Qwen3.6-27B NVFP4 MTP3 |
| Prefill chunk | 1,024 tokens |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefix reuse | Disabled |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0 |
| Greedy profile | Exact argmax (`--sampling greedy` in the corpus runner) |
| MTP0 | no `--spec` |
| MTP3 | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| DFlash block=8 | `--spec dflash --draft-tokens 7 --lm-head-draft` |

The MTP0 profile uses four Long NIAH prompts with approximately 8K, 64K, 128K, and 256K tokens.
Thinking is disabled and the output budget is 128 tokens. These runs measure prefill throughput,
server-internal time to first token, and baseline decode throughput at each context length. Content
scenarios are not repeated with MTP disabled because they do not change the baseline decode path.

The speculative-decode corpus contains three long-reasoning fixtures with thinking enabled and a
65,536-token output limit, followed by twelve fixtures covering code, story, translation, and
structured output. The cross-scenario fixtures disable thinking and use a 4,096-token output limit.
The tables report actual completion lengths rather than assuming that every request reaches its
limit.

Metrics are computed from the server's unrounded phase timings and speculative-decode counters:

```text
prefill_tok_s = prompt_tokens / prefill_seconds
server_ttft_ms = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)
decode_tok_s = (completion_tokens - 1) / decode_seconds
spec_acceptance = accepted_tokens / drafted_tokens
spec_tokens_per_round = 1 + accepted_tokens / speculative_rounds
```

Decode throughput is a transport/execution measurement, not a correctness score. The response text,
finish reason, and fixture-level structural requirements are audited separately below. A request
that exhausts its output budget or enters a repetition loop remains useful as a sustained-decode
stress sample, but is not presented as a successfully completed task.

## Qwen3.8-27B concurrent MTP3 corpus makespan

Both weight profiles use the complete speculative-decode corpus described above: three
long-reasoning fixtures and twelve cross-scenario fixtures, each with five fixed seeds, for 75
requests per point. The runner shuffles that fixed request set once with seed `20260811` and
preserves the same ordered HTTP send sequence at every concurrency. Exactly C persistent client
workers each submit their next request only after receiving the current response. C=1 is therefore
a serial single-request corpus on one persistent server and supplies the per-fixture Qwen3.8
results in the final section.

Each point starts a fresh server on an RTX 5090 with CUDA 13.1 compile/runtime, CUDA driver API
13.3, stochastic sampling, INT8 group-64 KV, a 1,024-token prefill chunk, CUDA Graphs, prefix reuse
disabled, a 131,072-token per-request context ceiling, `--kv-capacity auto`, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Makespan begins when all client workers are released
and ends when the final complete HTTP response has been read. Prefill and decode rates divide the
corresponding server token totals by that full makespan; average batch includes the entire run,
including workload transitions and drain.

Sampling is stochastic: prompts, seeds, and send order are fixed, but weight-profile and
concurrency-specific numerical routes can change sampled continuations and their lengths. The
makespan speedups are therefore fixed-workload serving results rather than fixed-token
normalizations; the exact decode-token totals are retained in each table.

### `groupwise-int`

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Prefill tok/s | Decode tok/s | Avg batch | MTP acceptance | Speedup vs. C1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 15,460 | 747,295 | 4,622.59 | 0.0162 | 3.3 | 161.7 | 1.00 | 58.5% | 1.00× |
| 2 | 75 | 15,460 | 766,184 | 3,580.22 | 0.0209 | 4.3 | 214.0 | 1.91 | 59.5% | 1.29× |
| 4 | 75 | 15,460 | 739,692 | 2,864.48 | 0.0262 | 5.4 | 258.2 | 3.67 | 58.3% | 1.61× |
| 8 | 75 | 15,460 | 697,193 | 2,211.20 | 0.0339 | 7.0 | 315.3 | 4.76 | 58.9% | 2.09× |

All 300 requests completed without a request, CUDA, or out-of-memory failure. C=8 gives the
shortest complete-corpus makespan. `--kv-capacity auto` resolved to 131,072, 262,144, 341,952, and
313,984 tokens at C=1, 2, 4, and 8. C=8 reached a maximum of four waiting requests while admitting
long contexts into the shared pool; no spill, owner degradation/eviction, or search-exhaustion
event occurred.

### `nvfp4`

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Prefill tok/s | Decode tok/s | Avg batch | MTP acceptance | Speedup vs. C1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 15,460 | 752,160 | 4,670.27 | 0.0161 | 3.3 | 161.1 | 1.00 | 60.8% | 1.00× |
| 2 | 75 | 15,460 | 739,951 | 2,510.78 | 0.0299 | 6.2 | 294.7 | 1.98 | 59.2% | 1.86× |
| 4 | 75 | 15,460 | 713,384 | 1,647.74 | 0.0455 | 9.4 | 432.9 | 3.29 | 58.0% | 2.83× |
| 8 | 75 | 15,460 | 723,602 | 2,164.90 | 0.0346 | 7.1 | 334.2 | 2.36 | 57.6% | 2.16× |

All 300 requests completed without a request, CUDA, or out-of-memory failure. C=4 gives the
shortest complete-corpus makespan. C=8 is limited by memory pressure, which constrains effective
batching and makes the end-to-end result slower than C=4.

The groupwise-int weight arena is 16.672 GiB versus 19.729 GiB for NVFP4. At C=8, the smaller
resident profile permits 313,984 tokens of Device KV versus 187,712 and raises the full-corpus
average batch from 2.36 to 4.76. The best measured point is therefore C=8 for groupwise-int and C=4
for NVFP4.

## Concurrent MTP3 decode saturation

The concurrent campaign uses the `long_decode_aime26_15` fixture with thinking enabled. The
rendered prompt is 293 tokens, and every request has an 8,192-token output budget. For each
concurrency C, the runner starts a fresh `ninfer-serve` process with `max_concurrency=C`, releases
C non-stream requests together using distinct fixed seeds, and waits for every HTTP response.
Startup and server warmup occur before the measured wave.

All points use an RTX 5090, CUDA 13.1 compile/runtime, CUDA driver API 13.3, stochastic sampling
(temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0), INT8 group-64 KV, a 1,024-token
prefill chunk, CUDA Graphs, prefix reuse disabled, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Each request has a 16,384-token context ceiling.
`--kv-capacity auto` resolved to exactly `C * 16,384` tokens at every point.

Saturated throughput uses only complete one-second server intervals satisfying all of the following:

- computed prefill tokens are zero;
- `running=C`, `prefilling=0`, and `decode_ready=C`;
- at least one decode round completed;
- every decode round had exactly C rows.

Ramp-up, prefill, and drain intervals are excluded. The reported aggregate rate is:

```text
steady_decode_tok_s = sum(committed_decode_tokens) / sum(interval_seconds)
```

Wave makespan starts when the client threads are released and ends after the last complete HTTP
response. MTP acceptance is aggregated over the complete wave. Each row below is one sustained
wave rather than a repeated-sample mean.

| Model profile | C | Steady (s) | Avg batch | Aggregate decode tok/s | MTP acceptance | Speedup vs. C1 | Wave makespan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 1 | 43.01 | 1.00 | 185.8 | 68.2% | 1.00× | 44.23 |
| Qwen3.6-27B `groupwise-int` | 2 | 65.01 | 2.00 | 247.0 | 69.0% | 1.33× | 66.67 |
| Qwen3.6-27B `groupwise-int` | 4 | 102.02 | 4.00 | 309.5 | 68.4% | 1.67× | 107.49 |
| Qwen3.6-27B `groupwise-int` | 8 | 118.02 | 8.00 | 535.0 | 68.3% | 2.88× | 125.20 |
| Qwen3.6-27B `nvfp4` | 1 | 39.01 | 1.00 | 202.4 | 69.3% | 1.00× | 40.46 |
| Qwen3.6-27B `nvfp4` | 2 | 39.01 | 2.00 | 399.7 | 71.4% | 1.97× | 41.82 |
| Qwen3.6-27B `nvfp4` | 4 | 44.01 | 4.00 | 699.7 | 69.3% | 3.46× | 47.92 |
| Qwen3.6-27B `nvfp4` | 8 | 55.01 | 8.00 | 1,146.9 | 68.6% | 5.67× | 58.57 |
| Qwen3.6-35B-A3B `groupwise-int` | 1 | 12.00 | 1.00 | 593.0 | 67.2% | 1.00× | 13.75 |
| Qwen3.6-35B-A3B `groupwise-int` | 2 | 17.00 | 2.00 | 877.7 | 68.2% | 1.48× | 18.87 |
| Qwen3.6-35B-A3B `groupwise-int` | 4 | 26.01 | 4.00 | 1,166.0 | 69.8% | 1.97× | 28.43 |
| Qwen3.6-35B-A3B `groupwise-int` | 8 | 48.01 | 8.00 | 1,313.8 | 67.3% | 2.22× | 50.20 |

All 45 requests reached their output limit, producing 368,640 completion tokens. The campaign
contained 608 complete full-batch steady intervals and had no request, CUDA, or out-of-memory
failure. At C=8, available device memory after startup was 2.66 GiB for 27B groupwise-int,
2.18 GiB for 27B NVFP4, and 4.38 GiB for 35B-A3B.

## Reproduction

Build `ninfer-serve` and prepare the registered `.ninfer` artifacts. The refreshed per-target
serving tables use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 262144 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_35b_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 \
  --output profiles/bench/serve_corpus_27b_mtp3_20260724

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_27b_nvfp4_w8_20260731

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_qwen3_8_27b_groupwise_mtp0_20260831

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_qwen3_8_27b_groupwise_mtp3_20260830

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_qwen3_8_27b_nvfp4_mtp0_20260817

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_qwen3_8_27b_nvfp4_mtp3_20260817
```

The concurrent decode-saturation campaigns use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_35b_mtp3_20260811
```

Use `--mode dflash7` for the corresponding DFlash block=8 campaign; add `--sampling greedy` for
the exact-argmax profile.

Omit `--mode` and supply the two measured Qwen3.6 groupwise-int artifacts to run the complete
published Qwen3.6 MTP0/MTP3 campaign:

```bash
python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --output profiles/bench/serve_corpus_20260720
```

For the 27B NVFP4 accuracy run, start the model service with:

```bash
build/apps/ninfer-serve out/qwen3_6_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Then run the repository's full 27B reasoning suite in a separate shell:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/qwen3_6_27b_reasoning.yaml \
  --suite reasoning_full
```

## `qwen3_6_35b_a3b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 15,544.3 ± 242.4 | 500.2 ± 7.8 | 271.1 ± 3.6 |
| 64,512 | 5 | 10,809.0 ± 95.3 | 6,009.9 ± 52.6 | 242.9 ± 1.3 |
| 130,048 | 5 | 7,828.4 ± 34.1 | 16,693.3 ± 71.2 | 219.4 ± 1.6 |
| 260,096 | 5 | 5,157.1 ± 52.4 | 50,598.8 ± 519.7 | 188.2 ± 2.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,223.0 ± 2,224.1 | 726.2 ± 22.9 | 82.8% ± 3.4% | 3.48 ± 0.10 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 620.3 ± 8.1 | 72.7% ± 1.4% | 3.18 ± 0.04 |
| `long_decode_aime26_30` | 5 | 52,977.8 ± 11,849.6 | 671.9 ± 8.8 | 80.1% ± 2.7% | 3.40 ± 0.08 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 657.6 ± 34.3 | 70.3% ± 5.5% | 3.11 ± 0.16 |
| Story | 15 | 456.2 ± 36.6 | 38.0% ± 6.0% | 2.14 ± 0.18 |
| Translation | 15 | 649.7 ± 33.0 | 67.6% ± 5.1% | 3.03 ± 0.15 |
| Structured | 15 | 770.9 ± 29.3 | 89.1% ± 4.9% | 3.67 ± 0.15 |

### DFlash block=8 (`k=7`), stochastic sampling

The fixtures, five seeds, sampling parameters, and output limits are identical to MTP3. Different
speculative backends consume random values differently, so this is a fixed-workload comparison
rather than a token-identical paired-output comparison.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,495.4 ± 2,221.2 | 764.1 ± 55.6 | 65.2% ± 5.4% | 5.56 ± 0.38 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 584.0 ± 33.3 | 51.1% ± 3.7% | 4.58 ± 0.26 |
| `long_decode_aime26_30` | 5 | 53,330.4 ± 11,198.5 | 638.3 ± 15.8 | 56.4% ± 2.5% | 4.95 ± 0.17 |

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 562.3 ± 36.2 | 43.0% ± 3.7% | 4.01 ± 0.26 |
| Story | 15 | 261.7 ± 51.1 | 12.1% ± 5.3% | 1.85 ± 0.37 |
| Translation | 15 | 490.8 ± 62.6 | 34.8% ± 6.3% | 3.44 ± 0.44 |
| Structured | 15 | 786.4 ± 124.7 | 66.5% ± 13.5% | 5.66 ± 0.94 |

#### Decode throughput versus MTP3

| Workload | MTP3 tok/s | DFlash tok/s | DFlash change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 726.2 | 764.1 | +5.2% |
| `long_decode_aime26_15` | 620.3 | 584.0 | -5.9% |
| `long_decode_aime26_30` | 671.9 | 638.3 | -5.0% |
| Code | 657.6 | 562.3 | -14.5% |
| Story | 456.2 | 261.7 | -42.6% |
| Translation | 649.7 | 490.8 | -24.5% |
| Structured | 770.9 | 786.4 | +2.0% |

### DFlash block=8 (`k=7`), greedy sampling

Greedy uses exact argmax; all other corpus and server settings remain unchanged. The five seeds
repeat the same deterministic generation path, so within-fixture standard deviation measures
runtime variation rather than output variation.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 6,692.0 ± 0.0 | 872.4 ± 3.3 | 74.4% ± 0.0% | 6.21 ± 0.00 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 651.6 ± 0.6 | 58.6% ± 0.0% | 5.10 ± 0.00 |
| `long_decode_aime26_30` | 5 | 65,536.0 ± 0.0 | 994.9 ± 3.4 † | 98.0% ± 0.0% | 7.86 ± 0.00 |

† The generation is a deterministic repetition loop, not a valid AIME response. The raw rate is
retained to describe what was measured, but is excluded from performance comparisons.

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 599.8 ± 12.3 | 46.4% ± 1.4% | 4.25 ± 0.10 |
| Story | 15 | 291.5 ± 55.6 | 14.9% ± 5.7% | 2.04 ± 0.40 |
| Translation | 15 | 475.5 ± 50.6 | 33.0% ± 5.1% | 3.31 ± 0.36 |
| Structured | 15 | 869.0 ± 120.2 | 74.5% ± 13.1% | 6.21 ± 0.92 |

#### Decode throughput versus stochastic DFlash

| Workload | Stochastic tok/s | Greedy tok/s | Greedy change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 764.1 | 872.4 | +14.2% |
| `long_decode_aime26_15` | 584.0 | 651.6 | +11.6% |
| `long_decode_aime26_30` | 638.3 | 994.9 † | not comparable † |
| Code | 562.3 | 599.8 | +6.7% |
| Story | 261.7 | 291.5 | +11.4% |
| Translation | 490.8 | 475.5 | -3.1% |
| Structured | 786.4 | 869.0 | +10.5% |

### Speculative-decode output audit

The audit covers all 225 stored July responses from the 35B-A3B MTP3 stochastic-sampler, DFlash
stochastic-sampler, and DFlash greedy campaigns. It checks termination, exact repetition, and
fixture-specific mechanical constraints. AIME 1 was checked algebraically; the AIME 30 answer
(`393`) was checked by independent enumeration. This audit does not attempt to assign a subjective
quality score to prose or translations.

#### Long-reasoning answers

| Fixture | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| `long_decode_aime26_01` | 5/5 correct, natural stop | 5/5 correct, natural stop | 5/5 correct, natural stop |
| `long_decode_aime26_15` | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit |
| `long_decode_aime26_30` | 3/5 correct, 1 wrong, 1 no answer | 2/5 correct, 1 wrong, 2 no answer | 0/5 answers; all enter the same repetition loop |

The greedy AIME 30 response has an empty final-content field and fills its 65,536-token reasoning
budget. The exact line `Wait, $x_7 x_1 x_3$ is $x_7 x_1 x_3$.` occurs 2,406 times among 2,538
non-empty reasoning lines. Its 98.0% acceptance and 994.9 tok/s therefore characterize a highly
predictable pathological loop, not normal reasoning performance.

AIME 15 is also not a valid completion in any of the three campaigns: every sample exhausts the
budget without a boxed answer. Its output is long, non-convergent reasoning rather than the short
exact cycle seen in greedy AIME 30. The AIME 15 rates may be read only as sustained long-decode
throughput.

#### Cross-scenario outputs

| Category | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| Code | 1/15 natural stops; 0/15 prompt-complete | 2/15 natural stops; 0/15 prompt-complete | 0/15 natural stops |
| Story | 9/15 natural stops; the nine Chinese outputs pass requested division and minimum length | 8/15 natural stops; the eight Chinese outputs pass requested division and minimum length | 10/15 natural stops; five Chinese dialogue outputs are under length |
| Translation | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks |
| Structured | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract |

The code prompts require complete runnable multi-file deliverables, but almost all outputs end at the
4,096-token limit. The three natural-stop exceptions also contain decisive contract failures: the
MTP3 CUDA response substitutes CUDA 12.8 and an older architecture list; the DFlash CUDA response
copies FP32 input into a half-sized 16-bit allocation and passes raw `unsigned short` values to BF16
intrinsics; and the DFlash Python response never writes its advertised JSONL event stream to the
configured log file. Code throughput is therefore a truncated-generation stress result, not
successful code-generation throughput.

All English mystery samples reach the output limit with an unfinished ending. The naturally stopped
Chinese stories have the requested chapter/act counts; the MTP3 and stochastic-DFlash samples also
meet their requested Chinese-character minima. Greedy's five dialogue stories contain 3,239 Chinese
characters each, below the requested 3,500. Story results are consequently a mixed normal/truncated
workload.

All translation outputs stop naturally. Each plain-document result preserves six sections and
provides at least twenty glossary entries; each Markdown result preserves heading levels, the
six-line table, all required inline identifiers, and the exact fenced JSON object. Translation is
the cleanest cross-scenario normal-completion comparison in this corpus.

The structured prompts intentionally exceed what these generations fit into 4,096 tokens. MTP3,
stochastic DFlash, and greedy DFlash produce only 49–60, 49–58, and 57 valid JSONL records,
respectively, versus the requested 160. Their complete-width CSV ranges are 122–139, 121–143, and
133 rows versus the requested 220. No SQL output satisfies all four tables, two views, at least 80
rows, and six final analytical queries. These high-acceptance results describe predictable partial
record generation only.

The exact-line and repeated-token scan found no other response with a short-cycle collapse comparable
to greedy AIME 30. Output-limit and prompt-compliance failures above remain material even when no
repetition loop is present.

## `qwen3_6_27b`

### EvalScope reasoning accuracy

Both weight profiles were evaluated through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based
scoring, and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty
1.0, and seed 42. All 258 samples completed and were scored for each profile.

| Weights ID | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| `groupwise-int` | 86.67% (26 / 30) | 93.33% (28 / 30) | 86.87% (172 / 198) |
| `nvfp4` | 93.33% (28 / 30) | 93.33% (28 / 30) | 84.34% (167 / 198) |

These are single-sample results under the stated evaluation profile, not pass@k scores. Each
benchmark remains independently reportable; no combined score is computed.

### `groupwise-int`

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 5 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 5 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 5 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 10,686.2 ± 553.8 | 175.4 ± 1.0 | 77.9% ± 0.9% | 3.34 ± 0.03 |
| `long_decode_aime26_15` | 5 | 61,604.2 ± 5,677.9 | 161.9 ± 2.8 | 73.4% ± 1.7% | 3.20 ± 0.05 |
| `long_decode_aime26_30` | 5 | 47,339.8 ± 9,162.2 | 172.2 ± 0.9 | 78.8% ± 0.8% | 3.36 ± 0.02 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 167.0 ± 5.4 | 72.3% ± 3.5% | 3.17 ± 0.11 |
| Story | 15 | 112.6 ± 9.4 | 37.8% ± 5.9% | 2.13 ± 0.18 |
| Translation | 15 | 161.5 ± 11.3 | 68.3% ± 7.2% | 3.05 ± 0.22 |
| Structured | 15 | 193.0 ± 18.8 | 88.7% ± 11.7% | 3.66 ± 0.35 |

### `nvfp4`

The fixtures, seeds, sampling parameters, output limits, and runtime options are identical to the
groupwise-int serving campaign. Quantization can change sampled tokens, so the MTP3 results are a
fixed-workload comparison rather than a token-identical output comparison.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 5 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 5 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 5 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 12,053.4 ± 820.9 | 231.0 ± 3.0 | 80.2% ± 1.2% | 3.41 ± 0.04 |
| `long_decode_aime26_15` | 5 | 63,109.0 ± 5,426.9 | 213.1 ± 4.2 | 76.3% ± 2.0% | 3.29 ± 0.06 |
| `long_decode_aime26_30` | 5 | 57,166.4 ± 9,204.9 | 223.3 ± 1.8 | 81.1% ± 1.5% | 3.43 ± 0.04 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 220.3 ± 8.2 | 74.2% ± 4.0% | 3.23 ± 0.12 |
| Story | 15 | 148.8 ± 11.6 | 39.2% ± 5.7% | 2.18 ± 0.17 |
| Translation | 15 | 213.6 ± 12.2 | 70.5% ± 6.0% | 3.12 ± 0.18 |
| Structured | 15 | 252.2 ± 16.3 | 89.8% ± 8.0% | 3.69 ± 0.24 |

The baseline and speculative-decode suites intentionally measure different supported workloads.
No per-scenario baseline/speculative speedup is reported.

## `qwen3_8_27b`

The MTP0 tables come from the serial Long NIAH campaigns described by the single-request method.
The MTP3 tables come from the C=1 points of the fixed concurrent-corpus campaigns, which serially
run the same three long-reasoning and twelve cross-scenario fixtures. Each fixture has five fixed
seeds. Values are arithmetic mean ± sample standard deviation from the server's per-request phase
timings and speculative counters. The prompts, seeds, sampling parameters, output limits, and
runtime options are identical across weight profiles; quantization can change sampled tokens, so
MTP3 is a fixed-workload comparison rather than a token-identical output comparison.

### `groupwise-int`

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,274.7 ± 11.3 | 2,349.2 ± 8.4 | 79.8 ± 0.4 |
| 64,512 | 5 | 2,696.1 ± 12.7 | 23,952.3 ± 112.4 | 73.6 ± 0.4 |
| 130,048 | 5 | 2,183.5 ± 11.0 | 59,607.5 ± 299.8 | 66.3 ± 0.4 |
| 260,096 | 5 | 1,609.7 ± 5.3 | 161,674.5 ± 533.7 | 56.2 ± 0.6 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 1,537.4 ± 317.8 | 193.4 ± 5.5 | 72.5% ± 3.1% | 3.17 ± 0.09 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 150.1 ± 2.3 | 53.0% ± 1.3% | 2.59 ± 0.04 |
| `long_decode_aime26_30` | 5 | 46,245.4 ± 10,867.7 | 171.1 ± 27.7 | 63.6% ± 15.7% | 2.91 ± 0.47 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| Code | 15 | 200.3 ± 7.8 | 76.3% ± 4.2% | 3.29 ± 0.12 |
| Story | 15 | 130.4 ± 12.0 | 37.9% ± 6.5% | 2.14 ± 0.19 |
| Translation | 15 | 198.1 ± 10.2 | 74.9% ± 5.5% | 3.25 ± 0.17 |
| Structured | 15 | 224.4 ± 13.6 | 89.5% ± 7.4% | 3.68 ± 0.22 |

### `nvfp4`

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 8,340.4 ± 13.0 | 931.6 ± 1.6 | 71.2 ± 0.1 |
| 64,512 | 5 | 5,297.9 ± 259.2 | 12,281.1 ± 561.5 | 65.7 ± 0.8 |
| 130,048 | 5 | 3,544.7 ± 25.3 | 36,853.5 ± 259.4 | 59.6 ± 0.9 |
| 260,096 | 5 | 2,203.1 ± 13.4 | 118,354.8 ± 717.2 | 52.9 ± 2.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 1,465.4 ± 417.3 | 195.2 ± 4.6 | 76.0% ± 2.4% | 3.28 ± 0.07 |
| `long_decode_aime26_15` | 5 | 65,414.4 ± 271.9 | 151.4 ± 2.0 | 56.2% ± 1.1% | 2.69 ± 0.03 |
| `long_decode_aime26_30` | 5 | 50,023.4 ± 14,839.1 | 167.5 ± 23.7 | 64.6% ± 14.9% | 2.94 ± 0.45 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 194.3 ± 6.1 | 76.4% ± 3.9% | 3.29 ± 0.12 |
| Story | 15 | 126.1 ± 10.9 | 37.4% ± 5.8% | 2.12 ± 0.17 |
| Translation | 15 | 192.3 ± 11.9 | 75.0% ± 6.5% | 3.25 ± 0.19 |
| Structured | 15 | 219.8 ± 8.6 | 90.8% ± 5.1% | 3.72 ± 0.15 |

The baseline and speculative-decode suites intentionally measure different supported workloads.
No per-scenario baseline/speculative speedup is reported.


## OrcaRouter NVFP4 integration probe

On 2026-09-12, the separately registered `qwen3.8-27b-orcarouter/nvfp4` artifact was exercised
on RTX PRO 6000 Blackwell with ECC enabled, driver 616.92, CUDA 13.3 and a native Windows Release
build. It preserves the pinned OrcaRouter source's BF16 embeddings and output head. These are
initial integration measurements, not a benchmark suite or an Unsloth Studio comparison.

Each mode received the same 169-token Python queue-repair prompt, greedy non-thinking sampling,
neutral penalties and a 1536-token output budget. Startup used four active lanes, a 32K per-request
limit, 64K shared FP8 KV, 2048-token prefill chunks, CUDA graphs and Vision enabled. The timed
coding request ran alone; separate checks exercised two concurrent requests. Responses terminated
naturally and differed in content and length, so elapsed-time ratios are not fixed-output speedups.
Decode rates below use engine-reported decode time and exclude the first output token.

| Backend | Completion tokens | Decode tok/s | Accepted / drafted | Generated tests passing |
|---|---:|---:|---:|---:|
| Ordinary | 1296 | 70.8 | — | 1/2 |
| MTP K5, optimized head | 1465 | 196.0 | 1068/1985 (53.8%) | 2/2 |
| DFlash2 K7, optimized head | 1310 | 208.8 | 975/2338 (41.7%) | 2/2 |
| DFlash2 K7, full BF16 head | 1295 | 206.4 | 975/2240 (43.5%) | 1/2 |

Actual answers were reviewed and their Python unittest examples executed with Python 3.14. The
ordinary response incorrectly calls `Task.exception()` expecting a returned `CancelledError`;
that method raises it. The full-head DFlash2 response references a nonexistent public
`Queue.unfinished_tasks` attribute. The optimized DFlash2 response's own tests pass, but its
worker re-raises a job failure and dies, so remaining queued jobs can still make `join()` hang.
Its passing tests therefore do not establish a successful production repair. These observations
are too small a sample to rank model quality or attribute errors to speculation.

All four modes passed ordinary Chat Completions, constrained JSON Schema,
low-thinking output and two-request concurrency checks. Named tool calls also passed in the
installed build, which included a separate, pending forced-tool implementation; that observation
does not qualify forced-tool support in the standalone OrcaRouter change. The installed ordinary-decoding route
also correctly read a synthetic image's counts/shapes, streamed SSE through `[DONE]`, and reused
2764 tokens through a Responses `previous_response_id` continuation. Source-specific tokenizer
checks cover 17 independent reference cases; BF16 Linear and LinearTopK pass their independent
numerical oracles, including the existing FP8 top-k regression. These establish integration,
not long-horizon coding quality. The launcher defaults this derivative to ordinary decoding and
exposes both speculative backends for deliberate testing.
