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

## Completed structural comparison

[Run 38008780320](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38008780320)
and software run 38008780221 passed at `a8711f9`. Artifact 11655316722
(18.4 MB compressed) contains all ten timing servers and three separate diagnostics.
The compact machine-readable result is `v100-structural-comparison-38008780320.json`.
Every timed response reports 7,111 prompt tokens, zero reused prompt tokens and 128
completion tokens. These are observed frontend counts, not proof of equal accepted
internal work, quantization, outputs or arithmetic.

| Arm | Cold HTTP median | Warmed HTTP median | Warmed native prefill | Warmed native decode | Peak sampled GPU |
|---|---:|---:|---:|---:|---:|
| NInfer static64 | 70.218 s | 62.022 s | 53.491 s | 7.549 s | 18,844 MiB |
| NInfer adaptive156, including fresh end reference | 72.217 s | 60.373 s | 53.270 s | 6.186 s | 30,488 MiB |
| Strata native | 16.431 s | 11.551 s | 8.583 s | 2.301 s | 32,470 MiB |

NInfer medians are stable after the first request; Strata retains drift and changes
outputs between servers and some requests. Static/adaptive NInfer each replay their
own output exactly; their cross-arm outputs differ. The three main adaptive warmed
HTTP measurements are 60.541, 60.257, 59.617 s; the independent end reference is
60.489 s with 53.277 s prefill. This supports the prefill execution gap without
claiming global convergence or matched cross-engine correctness. Warmed NInfer
static/adaptive MTP accept 68/67 drafted tokens from 116/118 drafted, respectively;
Strata accepts 68/73/67 from 119/112/110 drafted. Equal output limits do not make
verifier/draft schedules equal.

Actual diagnostic cache capacities: static64 allocates 8,494,252,032 compact bytes
(3,072 experts), adaptive156 allocates 20,704,739,328 bytes (7,488 experts).
At the final verify snapshot adaptive has 7,442 ready and 46 uploading experts.
Strata logs an actual 8,082-slot cache, 23.59 GiB, despite an earlier auto estimate
of 6,342 slots; its prompt path borrows 1,144 slots (3.33 GiB). It also allocates
48.14 GiB of page-locked, mapped host cache complement. Its telemetry cache-window
fields are absent: their zero aggregate is **missing observation**, not zero cache.
Native allocation messages are authoritative. Child RSS coverage is incomplete.

Separate diagnostics aggregate 7K, short and warmup/setup work; NInfer prefill has
344,976 layer-tokens versus Strata 341,280, so do not compare raw totals as equal
work. Static/adaptive NInfer prefill expert H2D is 71.214/54.618 GB, modeled CPU
weight reads 193.617/147.879 GB; inclusive CPU expert intervals 10.152/9.835 s.
Verify residency is 27.0%/70.2%, versus Strata 66.0% with another 4.2% nonresident
GPU routes. Diagnostic verify work differs (40,800/40,800/68,160 routes).
The increased cache saves traffic and improves decode, but barely changes prefill.

The four long-prompt NInfer chunk ledgers put 78.4%/78.0% of their timelines in
`MoE: reduce`; this includes stream waits and host gaps. It is not yet a kernel-only
measurement. Strata's **actual** prefill log reports 1,472 ms dequantization,
2,145 ms gate/up GEMM and 1,071 ms down GEMM, establishing that its dequantize +
FP16 GEMM fallback executes on these actual layer types. No MMQ attribution is
needed for this run. Its 8,864 ms GPU timeline includes 277 ms waiting for copies
and 244 ms combine. Host spans and concurrent stream intervals are not additive.

The next job is one static64 diagnostic server with two 7K/32-output requests,
not another comparison or parameter sweep. Opt-in `NINFER_V100_EXPERT_STREAM_TIMING=1`
reuses a bounded five-event set per ring slot, recording H2D, compute-stream wait
and the two-kernel interval after the copy dependency. Collection occurs after
slot consumption; normal serving allocates no extra events. The kernel interval
can include launch submission gaps and overlap with transfer; it is not an
isolated instruction profiler. This job tests the existing real-shape independent
expert oracle with timing enabled and verifies that timed route/upload counts
exactly cover native streamed work. Its result decides whether to implement the
bounded FP16 routed-expert GEMM replacement. Numerical thresholds remain unchanged.

## Stream attribution and selected implementation

[Run 38013686130](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38013686130)
passed, including the independent expert stream oracle (2.60 s). Artifact
11655442811 is 4.48 MB compressed. Both requests conserve 25,755 streamed experts,
2,654,546 routed inputs and the exact corresponding padded compact H2D bytes.
Machine-readable evidence: `v100-stream-attribution-38013686130.json`.

| Static64 7K diagnostic | Cold | Repeated |
|---|---:|---:|
| Streamed expert kernel interval | 26.568 s | 26.665 s |
| Concurrent H2D interval sum | 7.033 s | 6.929 s |
| Compute-stream copy-dependency wait | 0.386 s | 0.155 s |

These sums overlap and must not be added. The repeated request's four long-prompt
chunk MoE intervals total 38.912 s; streamed kernels alone account for about 68.5%
of that scope. Event placement excludes copy dependencies; it can still include
small launch submission gaps. Combined with the prior traffic/cache experiments,
this supports replacing streamed expert compute before changing CPU placement,
cache capacity, or transfer policy.

The selected opt-in implementation is `NINFER_V100_PREFILL_EXPERT_GEMM=fp16`
(default `simt`): decode each submitted compact expert once into reusable FP16
weights, then gather, gate/up GEMM, FP32 SiLU/product, down GEMM and FP32 scatter.
One scratch allocation is reused in compute-stream order. A 256-route tile bounds
activation buffers even at maximum planned route capacity. Owned device scratch
is 19,597,312 bytes (18.69 MiB), plus library/context allocations observed at runtime.
There is no persistent expansion of all cached experts or model conversion.
The initial dispatch threshold is 32 routes; the hardware fixture also records
SIMT/GEMM costs at 8,16,20,32,64,128,256 routes to qualify that crossover.

Unscaled E2M1 × E4M3 finite weight products fit exactly in FP16; the expert divisor
is applied in FP32 after each GEMM. Inputs and SiLU products use per-token
power-of-two normalization before FP16 conversion to avoid range overflow on
large finite inputs. Accumulation is FP32 with cuBLAS reduced-precision reduction
disallowed. This changes arithmetic and intermediate casts: no numerical
qualification is claimed from compilation, pairwise comparison, or plausibility.

The independent gate uses the existing scalar host decoder and sequential
FP32 represented-weight formula with FP32 intermediate, without candidate FP16
casts. Real shapes, all finite nonnegative E4M3 scale codes, three divisors,
large/tiny/zero BF16 inputs, 1–512 routes including partial and multiple tiles,
and output guard regions are covered. NRMSE ≤0.002 and cosine ≥0.99999 remain
unchanged; the BF16-boundary stream/coexistence oracle is supplementary.

## Qualified streamed GEMM result and resident extension

[Run 38016688665, attempt 2](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38016688665)
passed on V100. Attempt 1 stopped before uploading evidence; its interruption
cause is unknown. The successful retry produced artifact 11658204890. Compact
evidence is in `v100-expert-gemm-38016688665.json`.

| Static64, same 7111 prompt / 128 emitted tokens | SIMT | Streamed FP16 GEMM |
|---|---:|---:|
| Cold HTTP median | 63.250 s | 49.861 s |
| Warmed HTTP median | 61.767 s | 47.397 s |
| Warmed prefill median | 53.442 s | 39.955 s |
| Warmed decode median | 8.159 s | 7.384 s |
| Observed peak GPU memory | 18843.75 MiB | 18873.75 MiB |

Each arm used three fresh servers, cold + four warmups + one warmed request,
with reversed middle order and prefix reuse disabled. All requests conserved
7111 frontend prompt tokens, 128 emitted tokens and zero cached tokens. Each arm
replayed one consistent output across its own trials, but outputs differ across
arms. SIMT accepted 68/116 drafted tokens in 59 rounds; FP16 accepted 70/114 in
57 rounds. Thus the observed 23.3% whole-request and 25.2% prefill reductions are
supported; the 9.5% decode reduction is confounded by changed output/MTP work.
The first SIMT cold prefill was 70.096 s versus 54.7–54.9 s thereafter; host file
cache was preserved, so cold startup and first-request timing are not symmetric
page-cache controls. No Strata rerun or cross-quantization claim is involved.

The direct independent FP32 formula gate passed every case with worst NRMSE
0.000225084. Cosine passed ≥0.99999 at full precision; the log rounds its printed
value to 1. Candidate-enabled stream/coexistence also passed. The expert-cost
fixture includes expansion, excludes H2D and measured 3 warmups + 5 spans: at
8 routes SIMT/GEMM was 0.0897/0.1251 ms; at 16, 0.1919/0.1251; at 32,
0.3412/0.1325; at 128, 1.2993/0.1565. These support the conservative 32-route
dispatch; tiny groups stay SIMT. This is qualification, not another parameter sweep.

Both arms allocated 8,494,252,032 bytes of compact cache (64 slots per layer).
The 30 MiB observed peak increment includes the 18.69 MiB explicit scratch and
library/context extras. Separate two-request 7K/32-output diagnostics show streamed
kernel interval totals falling 53.170→10.164 s (80.9%), while concurrent copies
remain 13.698→13.882 s and copy-dependency waits rise 0.466→11.989 s. The new
compute speed exposes transfer waits; these overlapping sums are not additive.
Streamed expert counts 51,510/51,644 and routes 5,309,092/5,309,866 differ slightly
with changed routing. Repeated-request MoE intervals fall 38.800→25.875 s;
CPU inclusive time and remaining resident SIMT work still matter.

The next additive opt-in extension is `NINFER_V100_PREFILL_RESIDENT_GEMM=1`
(default 0). Resident prefill groups with ≥32 routes reuse the same qualified
operator: expand one compact Ready expert, process bounded 256-route tiles and
scatter directly into existing destinations. A separate 18.69 MiB scratch instance
is included in cache transfer/operating-budget accounting; there is no persistent
FP16 cache expansion, extra expert H2D, or decode change. Slot leases are retained
through the existing completion boundary. Smaller groups keep fused grouped SIMT.

The protected job directly reruns the independent FP32 gate, then candidate-enabled
cache/stream coexistence and cache integration fixtures, including mixed 31/32/33
route boundaries. Any skip or numerical failure fails before inference. Only then
`--resident-gemm-ab` compares qualified stream-only FP16 against stream+resident
FP16 with the same three-server cold/warmed workload and separate diagnostics.
Diagnostic resident-dispatch rows must be present only in the resident arm. This
isolates the extension without repeating Strata or changing the measured streamed
threshold. Actual memory, emitted/accepted tokens and output changes remain required.
Both arithmetic-changing paths remain opt-in while this extension is qualified.

Resident extension run 38023836226 compiled and passed the direct FP32 operator
and resident/stream coexistence gates (32,730 resident GEMM routes). Its cache
fixture passed all numeric checks reached, including mixed dispatch boundaries,
then stopped on a legacy grouped-versus-scalar bitwise equality assertion at
128 tokens. This is an arithmetic-profile assertion, not an independent-formula
failure. Artifact 11659926204 preserves the partial logs; no inference timings
exist for this run.

The corrected fixture retains exact grouped/scalar parity for SIMT and small
fallback groups. Eligible resident GEMM routes are checked individually against
the existing independent represented-weight FP32 formula, without candidate
casts, at the same NRMSE ≤0.002 / cosine ≥0.99999. Mixed 32/31, 32/32 and 33/32
expert counts verify both dispatch conservation and each route's applicable
formula. The protected job runs the entire cache fixture with resident GEMM off
first, then on; neither the SIMT assertion nor the numerical thresholds are
removed. Rerun the isolated resident A/B only after all these gates pass.
