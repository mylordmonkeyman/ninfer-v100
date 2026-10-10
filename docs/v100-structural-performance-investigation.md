# V100 structural performance investigation

Checkpoint: October 9, 2026. NInfer baseline `293e6bc0fe85bcf3f85c39958aab651adbe52895`;
Strata comparison source `0430d397d907032fc9ab83c8bb7acff48a935b53`.

## First execution: framing and launch correction

Run 38007509479 built both engines successfully, but the harness combined
`--no-prefix-reuse` with explicit context-cache capacity options. Native CLI
validation rejected the first NInfer launch before model loading or inference.
Artifact 11651544700 preserves the 66 KB compressed partial evidence. No request
performance result exists from that run. The corrected launch keeps prefix reuse
disabled and omits the incompatible capacity options. Thus those capacity settings
from the earlier prefix-reuse experiment are not part of this comparison's memory
budget; actual allocations are authoritative. Startup failures now print the native
error directly into Actions logs.

The artifact's actual native expert table and `.ninfer` directory establish:

| Engine / layers | Stored expert formats | Bytes per complete expert | All 512-expert banks |
|---|---|---:|---:|
| NInfer, 48 layers | NVFP4 fused gate/up + down | 2,764,808 | 67,947,921,408 B |
| Strata, 43 layers | Q4_K gate/up, Q5_1 down | 3,072,000 | Included below |
| Strata, 4 layers | Q4_K gate/up, Q8_0 down | 3,584,000 | Included below |
| Strata, 1 layer | Q5_K gate/up, Q8_0 down | 3,993,600 | Included below |
| Strata, all 48 layers | Mixed integer GGUF | Varies by layer | 77,017,907,200 B |

Strata's experts occupy about 13.35% more compact bytes, not fewer. This rules out
smaller compact expert payload as the explanation for its historical advantage;
it does not rule out format-specific compute efficiency, better device reuse,
different cache allocation, or other representation differences. The installed
Strata configuration is MTP spec4, INT8 KV, 262144 maximum context / 32768 resident
KV, auto expert cache, auto prefill, resident-budget71GiB. Those differ from NInfer's
MTP2/BF16/8192 controls and remain explicit descriptive-comparison confounds.
`nsys` is unavailable in the runner; existing bounded event ledgers are retained.

## Evidence retained from the completed chunk test

[Hardware 38003028002](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38003028002)
and software 38003028017 passed; artifact 11650009607. Same 7,111-token prompt,
8192 context, static64, MTP2, 88 workers on 32 physical cores, BF16 KV,
auto256/minroutes20 streaming, grouped CPU fallback, device combine, mmap PLE,
graphs and SV7 TensorOp off. Three fresh timed servers per arm; diagnostics separate.

| Measurement | Chunk2048 | Chunk4096 |
|---|---:|---:|
| Cold TTFT | 53.861 s | 51.826 s |
| Cold prefill | 132.047 tok/s | 137.230 tok/s |
| Cold decode | 16.645 tok/s | 15.236 tok/s |
| Continuation decode | 18.033 tok/s | 16.218 tok/s |
| Sampled GPU memory | 18,790 MiB | 19,764 MiB |
| Expert-weight H2D | 71,158,716,160 B | 47,796,758,016 B |
| Modeled CPU weight reads | 162,222,344,592 B | 76,507,766,976 B |
| Inclusive CPU branch | 5.157 s | 2.443 s |

Diagnostics cover equal 340,992 layer-tokens / 3,409,920 routes in 192/96 layer rows.
32.8% less expert H2D and 52.8% less modeled CPU reads yielded only 3.9% more
prefill throughput, with 8.5–10.1% worse decode. This disfavors further chunk tuning.
CPU branch intervals overlap GPU work; neither these intervals nor modeled weight
reads establish physical DRAM bandwidth or an additive critical path. Outputs differ
across chunk arms; same-arm replay passed. No numerical gate was waived.

## Source audit and live hypotheses

| Difference | Evidence | Mechanism to test | Evidence still needed |
|---|---|---|---|
| NInfer expert format | `expert_bank.cpp`, `expert_cache.h`: fused gate/up NVFP4 E2M1 codes, E4M3 scales per 16 values, per-bank expert divisor; 2560→640→2560 | Four-bit floating point decode/scaling costs differ from integer GGUF dots | Resident artifact directory, actual formats/bytes by object |
| Compact payload remains compact in NInfer | `expert_stream.cpp`: 2,764,808 B per pair, padded slot; host planes memcpy into pinned staging, unchanged compact device weights | Repacking copies exist, but there is no full FP16 expert expansion in this path | Actual cache allocation, staging traffic and time |
| NInfer resident/stream GPU compute | `moe_kernels.cu`: `cached_grouped_gate_up_kernel` / `cached_grouped_down_kernel`, FP32 SIMT FMA, at most four tokens per decoded weight group; BF16 intermediate | High-reuse prefill repeats compact decode and weight loads across many four-token groups; no expert TensorOp GEMM | Current MoE interval and GPU trace/kernel share |
| Strata prefill fallback | `prefill.cpp:compute`: native compact blocks dequantized into two rotating FP16 expert buffers, then `Gemm::f16` across all routed rows of one expert | Expand once, amortize dequantization, compute with FP16 GEMM; potentially much larger prefill gain than projection tuning | Actual MMQ versus FP16 dispatch on V100 and user's layer types |
| Strata alternative prefill routes | `mmq_plan()` checks type/geometry/device eligibility; MMQ or fused routes can supersede fallback | Do not attribute baseline speed to the FP16 source branch until runtime eligibility is established | Startup eligibility messages, native pack formats, phase timing; profiler if needed |
| CPU execution representations | NInfer AVX2 decodes NVFP4 into float vectors; Strata native CPU uses GGUF vec_dot and quantized activation buffers, specialized multi-token routes only for covered formats | Arithmetic and activation representation can change miss cost even at equal residency | Actual per-layer formats, miss work, CPU timing, short-request normalization |
| Residency is insufficient explanation | Prior static156/LRU156/Strata request 13.833/12.290/4.536 s, verify residency 52.45/88.99/89.35% | Similar route residency still leaves compute, copies, schedules, frontend/MTP and representation differences | Current adaptive reference at 7K, actual allocation and accepted work |

The Strata source paths are alternatives, not proof of which executed. The original
UD-Q4_K_XL pack and mixed `.ninfer` artifact remain unchanged. Matching raw messages
does not match quantization, tokenization, precision boundaries, KV memory or MTP width.

## Consolidated experiment

`tools/telemetry/structural_v100.py`, registered workflow `v100-strata-compare.yml`.
Explicit push marker `[v100:structural-compare]`; hosted software checks gate one
protected V100 job. Each engine builds once. No repeated worker/cache/MTP sweep.

Three arms: current static64 isolation reference; NInfer LRU156 adaptive reference
with the same auto256/minroutes20/grouped CPU controls; Strata native installed
configuration using frozen instrumented executable. NInfer keeps evidenced 88 workers
and physical-core placement; Strata keeps its native placement. Actual process CPU
sets, model directory, pack metadata, launch flags and memory samples are recorded.
Requested cache limits are not actual allocations or equal-memory controls.

Same 7K raw messages, 128 output-token limit, telemetry off for all timings.
Three fresh servers per arm, reverse order in the middle repetition. Each records
loading, first-request cold state, four additional warmup requests, then one measured
request after five residency-building requests. Retain all warmup rows to assess
ongoing drift; this does not assume global cache convergence. A final fresh adaptive
reference checks temporal drift.

Separate diagnostic servers use one 7K and one short request, 32 output tokens and
bounded level-2 telemetry. Existing NInfer chunk stage ledger and Strata prefill
phase timing observe waits/compute without rebuilding instrumentation. Preserve
prefill/verify/draft phase counters and actual native MTP result accounting. Strata's
native HTTP route requires speculation, so this campaign does not claim a matched
non-speculative full-model comparison. The workflow records profiler availability
without installing tools or changing permissions; full decode GPU attribution remains
a limitation if the existing observations do not resolve it.

Logs compressed per cell, partial evidence preserved, aliases outside artifacts,
uploads refuse symlinks and >=1GiB. No page-cache eviction, model conversion, process
stops outside owned server groups, runner security changes or Phase18/SV8.

## Implementation decision

If current profiling confirms routed expert GPU compute as the leading prefill
cost, implement a bounded Volta FP16 expert GEMM route at the existing streaming
boundary: decode the same represented NVFP4 pair once into reusable device buffers,
gather its routed inputs, fused gate/up GEMM + activation, down GEMM, scatter into
the existing route-output buffer. Keep compact persistent cache storage and account
for the FP16 staging/workspace budget; do not duplicate every cached expert in FP16.
Select a measured route-count crossover rather than use GEMM for tiny decode groups.

This is an arithmetic-changing operator: qualify against the independent FP32
formula decoding the exact stored NVFP4 codes/scales/divisor, at real 2560/640
shapes, including activation casts and unusual finite values. Preserve existing
criteria; pairwise SIMT agreement is supplementary. Then rerun the identical
request workload to establish whole-request benefit. If CPU miss execution or
synchronization instead dominates, prioritize that measured boundary. Do not commit
to a kernel rewrite merely because the source difference looks promising.

The older non-grouped stage result was about 87% MoE; the later grouped 1220-token
stream diagnostic still spent 10.355 of 12.486 s in the MoE reduce interval. These
are older inclusive measurements, not the current 7K GPU-compute fraction. There is
no justified numerical whole-request speedup prediction yet. A 2× acceleration of
a measured fraction f yields ideal speedup 1/(1-f/2), before added staging costs.
