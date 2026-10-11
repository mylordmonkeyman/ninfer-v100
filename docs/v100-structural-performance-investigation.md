# V100 structural performance investigation

Current status: six-step performance screening completed October 11, 2026.
See the final checkpoint below for retained opt-ins, long-context feasibility and limits.

Initial checkpoint: October 9, 2026. NInfer baseline `293e6bc0fe85bcf3f85c39958aab651adbe52895`;
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


## Resident GEMM qualified; next CPU/stream schedule

[Run 38024956818](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38024956818)
passed all direct operator, SIMT exact-parity, resident cache and coexistence
gates, then the complete resident A/B. Artifact 11660616475 contains the measured
workload; `v100-resident-gemm-38024956818.json` preserves compact evidence.

| Same static64 workload | Stream-only FP16 | Stream + resident FP16 |
|---|---:|---:|
| Warmed HTTP median | 47.947 s | 43.249 s |
| Warmed prefill median | 40.549 s | 35.567 s |
| Warmed decode median | 7.390 s | 7.663 s |
| GPU peak | 18873.75 MiB | 18905.75 MiB |

Resident GEMM reduces warmed HTTP by 9.8% and prefill by 12.3%. The original
SIMT baseline was 61.767 s / 53.442 s: combined reductions are about 30.0% /
33.4% across successive runs, rather than one simultaneous three-arm experiment.
Both arms conserve 7111 prompt, 128 emitted and zero cached tokens; each replays
one output within its own trials. Outputs differ across arms. Stream-only accepts
70/114 drafts in 57 rounds; resident accepts 68/115 in 59 rounds, confounding the
3.7% decode increase. Compact cache remains 8,494,252,032 bytes; the resident
scratch is included in the 19,597,312-byte transfer-budget increment and observed
GPU peak rises 32 MiB including library/context effects.

The independent FP32 operator gate still has worst NRMSE 0.000225084. Resident
integration passes individual FP32-formula checks for 32/31, 32/32 and 33/32
expert groups with unchanged limits. Separate diagnostics actually dispatch 7,334
resident GEMMs for 1,050,616 routed inputs across two long requests; streamed
copy/kernel/wait totals remain about 13.65 / 10.12 / 11.66 s. CPU inclusive work
remains 10.66 s across the two requests. All simultaneous stream intervals and
inclusive host scopes remain non-additive.

Source inspection establishes a remaining scheduling boundary: the CPU fallback
pool starts only after the host has submitted all streamed experts. Four-slot
blocking reuse waits for prior GPU consumption during that submission loop, so
most CPU work cannot overlap the streaming phase. Previous pipelined-slot tests
provided little gain; do not repeat slot/chunk parameter sweeps.

New opt-in `NINFER_V100_PREFILL_CPU_STREAM_OVERLAP=1` (default 0) fully classifies
misses and assigns pinned CPU destinations before starting the existing pool on
one coordinator thread. The inference owner then submits the same GPU experts
in the same order. The owner joins before inspecting pool results, merging,
releasing host storage or advancing the round; exceptions also join before
unwinding. Decode and explicit serial diagnostics retain the old schedule. No
quantization, arithmetic, membership, route accumulation or cache capacity changes.
During asynchronous submission the owner makes no round-ledger mutations;
existing worker observations complete before the owner resumes ledger updates.
Diagnostic `early_cpu_stream` rows preserve actual CPU routes, CPU duration,
submission duration and their temporal overlap, without summing them as latency.

The protected job qualifies existing operator/cache/coexistence gates plus early
CPU/cache join parity before `--cpu-stream-overlap-ab`: qualified stream+resident
FP16 versus identical settings with early CPU misses. Three fresh servers per arm
use the same cold/warmed workload and separate long diagnostics. Every timing
request across both arms must have identical output hash, usage and native MTP
accounting. New schedule benefit is unclaimed until this experiment passes.

Run 38028087489 completed all timing arms with identical outputs, usage and MTP
accounting. Warmed HTTP fell 43.659→40.290 s (7.7%) and prefill fell
35.994→32.667 s (9.2%); decode remained effectively unchanged at
7.707/7.639 s. The diagnostic observed 11.509 s of real CPU/stream-submission
overlap across two long requests. The job failed only after measurement because
its route-conservation assertion included the initial 13-token prefill (4,839 CPU
routes) while the diagnostic emitted rows only for device-combine chunks. Artifact
11662081424 preserves the complete measurements.

The corrected diagnostic emits CPU route/timing rows for every requested prefill
scope and marks whether that chunk was eligible for early execution. Ineligible
small chunks report zero temporal overlap but remain in route conservation. This
does not change execution or relax the strict cross-arm output/MTP equality gate;
one protected rerun is required before promotion.

Run 38034582637 repeated the complete A/B and all build/numerical gates, but exposed
that the first correction still defined a requested scope as `stream_experts`.
The 13-token initial prefill does not enable expert streaming, so its 4,839 CPU
routes still had no row. The candidate again preserved exact output, usage and MTP
accounting while reducing the three warmed medians: HTTP 42.877→40.114 s (6.4%)
and native prefill 35.233→31.848 s (9.6%); decode medians were 7.543/7.977 s.
The final accounting fix defines telemetry request scope from prefill plus the
explicit overlap option, then separately marks execution eligibility from streaming,
device combine and non-serial scheduling. It changes no work placement.

## CPU/stream overlap qualified

[Run 38038027788](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38038027788)
passed every direct operator, SIMT, cache, resident, coexistence and early-join
gate, then the complete corrected A/B. Artifact 11665148807 contains the full
evidence; `v100-cpu-stream-overlap-38038027788.json` preserves the compact record.

| Same 7111-prompt / 128-output workload | Stream + resident FP16 | + early CPU overlap |
|---|---:|---:|
| Warmed HTTP median | 43.391 s | 40.719 s |
| Warmed prefill median | 35.505 s | 32.617 s |
| Warmed decode median | 7.659 s | 7.870 s |
| GPU peak | 18905.75 MiB | 18905.75 MiB |

The schedule reduces warmed HTTP by 6.2% and prefill by 8.1%. Every timing
request across both arms has the same output hash, usage and native MTP accounting:
7111 prompt, 128 emitted, zero cached, 68/115 accepted/drafted tokens and 59
rounds. The 2.8% decode increase therefore reflects run/phase drift rather than
different generated work, and no decode benefit is claimed.

The corrected diagnostic emits 432 early-CPU rows totaling 370,509 routes,
exactly equal to the native prefill ledger. Of these, 384 eligible streaming
rows cover 365,670 routes; 48 initial-scope rows cover the previously missing
4,839 routes and correctly report zero overlap. Actual CPU/submission temporal
overlap is 11.685 s across two long requests. Streamed and resident work is
unchanged at 5,311,550 and 1,050,616 routes respectively. Inclusive CPU,
submission and overlap intervals are not additive.

The independent represented-NVFP4 FP32 gate remains at worst NRMSE 0.000225084;
cache integration and coexistence maxima are 0.00169476 and 0.00173574, within
the unchanged NRMSE 0.002 and cosine 0.99999 limits. The three structural paths
are now qualified together. The next action is to select streamed FP16 GEMM,
resident GEMM and CPU/stream overlap by default for Volta builds while retaining
explicit legacy rollback values and validating the default-selected path once.

## Qualified Volta defaults

The qualified combination is now selected by default only when
`NINFER_VOLTA_BUILD` is active. Unset selectors choose streamed FP16 expert GEMM,
resident FP16 expert GEMM and early CPU/stream overlap. Explicit rollback remains
available with `NINFER_V100_PREFILL_EXPERT_GEMM=simt`,
`NINFER_V100_PREFILL_RESIDENT_GEMM=0` and
`NINFER_V100_PREFILL_CPU_STREAM_OVERLAP=0`; non-Volta builds retain the legacy
defaults unless explicitly opted in. Kernel arithmetic, dispatch thresholds,
numeric gates, compact cache payload and transfer-budget accounting are unchanged.

The protected validation preserves explicit legacy and qualified fixture gates,
then starts one fresh server with all three selectors removed from the environment.
[Run 38041684008](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38041684008)
passed this deployment-selection smoke. Compact evidence is in
`v100-qualified-defaults-38041684008.json`.

Neither the manifest nor recorded launch environment contains a selector override.
Cold and repeated 7111-prompt/32-output requests replay the exact output, usage and
native MTP work (16/27 accepted/drafted in 15 rounds). The default path dispatches
5,311,550 streamed routes and 1,050,616 resident-GEMM routes. Its 432 early-CPU
rows conserve all 370,509 native prefill CPU routes; the 48 ineligible rows cover
4,839 routes and report zero overlap, while eligible work overlaps submission by
11.812 s. Peak GPU use is 18905.75 MiB.

The independent FP32, explicit legacy rollback, streamed/resident coexistence,
cache integration and CPU/cache join fixtures all pass. Default-path cache and
coexistence maxima are NRMSE 0.00169476 and 0.00173574 under the unchanged 0.002
and cosine 0.99999 gates. This smoke is not a new performance comparison.

The authorized structural investigation is complete: all three evidence-backed
improvements are integrated as coherent Volta defaults, explicit rollback is
preserved, the compact expert payload is unchanged, and no unresolved in-scope
decision warrants another runner experiment. The remaining gap to Strata is not
an apples-to-apples kernel claim because quantization, KV, MTP and model arithmetic
differ; the same-work NInfer improvements are the supported deployment result.

## Authorized five-step continuation

On October 10 the user authorized executing the following performance plan, with
each step identified and reported on completion. The prior structural-default
milestone remains complete; this is a new investigation of the remaining gap.

1. **Residency with qualified GEMM/overlap — completed.** Compare static64 with
   adaptive LRU156, both explicitly selecting streamed/resident FP16 GEMM and early
   CPU overlap. Earlier cache-size conclusions used SIMT and need not apply after
   the compute change. This is a combined capacity/policy comparison, not isolated
   attribution to either variable. Retain the 7111-token prompt, 128 emitted tokens,
   zero prefix reuse, MTP2, BF16 KV, 2048-token chunks and 88 workers on physical
   cores. Three alternating fresh servers per arm each run cold + four warmups +
   one measured request. Observe actual allocation/seeding and sampled memory;
   preserve allocator reserves. Record outputs and MTP work instead of asserting
   numerical equivalence from equal frontend counts. No default changes yet.
2. **Remaining critical path — completed.** The same single protected
   job collects separate two-request 7K/32-output diagnostics for both arms with
   existing stage, streamed-expert and CPU-overlap ledgers. Use the winning arm's
   evidence first. Inclusive/overlapping intervals are not additive latency. Add
   another bounded diagnostic only if a missing observation changes the next design.
3. **Expert pipeline — completed; two negative experiments.** Implement the strongest measured remedy:
   batching, staged dequantization/GEMM overlap, justified expanded-weight reuse,
   or recalibrated CPU/GPU routing. Qualify the affected operator/lifetime contract
   and measure request-level benefit; do not repeat old ring/chunk sweeps unchanged.
4. **Decode — completed; row-budget experiment negative.** Investigate small routed groups and MTP
   verification separately; the qualified prefill GEMM crossover is 32 routes.
   Choose small-group GPU execution or CPU-miss improvements from observed costs.
   Different >=4-bit representations require independent numerical assessment.
5. **Strata comparison and longer-context feasibility — completed.** Rebenchmark both
   engines contemporaneously, disclose quantization/KV/MTP/memory differences, and
   evaluate residency, selected-block attention and speculation at larger contexts.
   Exceeding historical Strata time alone is not proof of a matched quality win.

Step 1 uses `--residency-gemm-ab` in `tools/telemetry/structural_v100.py` and the
existing protected structural workflow. It builds NInfer once, preserves the
original model and installed Strata, and records `residency-report.json` alongside
partial-safe evidence. Numerical thresholds are unchanged. No Phase18/SV8,
unrelated process termination, security changes or model conversion is authorized
by this campaign. Resume through GitHub after the result, advancing the numbered
steps without duplicate healthy jobs.

## Five-step continuation: Steps 1–2 completed, Step 3 experiment

Run [38060490788](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38060490788)
completed successfully on October 10. Three alternating fresh servers per arm,
each cold + four warmups + measured request, used the same 7111/128 frontend workload.
Both arms explicitly enabled qualified streamed/resident FP16 GEMM and CPU overlap.

| Warmed median | Static64 | Adaptive156 | Reduction |
|---|---:|---:|---:|
| HTTP request | 40.1527 s | 36.3596 s | 9.45% |
| Native prefill | 32.4406 s | 30.2795 s | 6.66% |
| Sampled peak GPU memory | 18.46 GiB | 29.83 GiB | — |

Actual fully seeded resident capacity was 3072 versus 7488 experts, using
8,494,252,032 versus 20,704,739,328 bytes; each arm retained the 2 GiB allocator
reserve. Both cold and measured outputs repeat within each arm, but differ between
arms. Adaptive warmed decode fell from 7.6980 to 5.7491 s with different MTP
acceptance/work, so this does not isolate a decode operator speedup. Adaptive156
is the next screening reference at 8192 context, not a universal deployment default.
The existing independent FP32 GEMM oracle and stream/cache join fixtures passed;
this does not close historical full-model drift qualification.

Step 2 used separate two-request 7K/32-output diagnostics. In the adaptive arm,
MoE reduction accounted for 18,604.261 of 33,692.276 ms (55.22%) of the repeated
four-chunk stage timeline. Streamed experts contributed aggregate copy 10,316.933,
compute-wait 8,725.422 and kernel 7,671.732 ms across both requests; these overlap
and are not additive wall latency. CPU work totaled 18,174.475 ms, with 10,074.805
ms overlapping streamed submission. Source inspection establishes a remaining
schedule gap: all resident consumers are submitted before CPU misses start. These
measurements identify the expert pipeline as the next target; they do not establish
how much the newly identified CPU-start gap costs.

Step 3's first attempt, CPU-before-resident dispatch, was rejected by run
[38063563801](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38063563801).
All numerical fixtures and exact corresponding-request output/usage/MTP checks
passed. Warmed HTTP medians were 35.6758 s resident-first and 35.7326 s CPU-first
(+0.16%); prefill was 29.8730 versus 29.8941 s. Cold HTTP was 39.4639 versus
39.7297 s. Extra CPU/resident overlap did not improve request throughput, so its
experimental selector, scheduling branch and campaign mode have been removed.
The earlier qualified CPU/stream overlap remains enabled.

Across the two diagnostics, eligible resident-first layers had 905.577 ms host
resident submission, 17,616.550 ms CPU work and 7,470.313 ms CPU join wait. CPU-first
had 1,078.410 ms additional resident overlap and 6,722.192 ms join wait. These
inclusive sums explain why simply starting CPU work sooner is insufficient;
they do not add to request latency. Verification remained CPU-heavy (about
1.81 s CPU in a 2.99 s inclusive MoE scope), retained for Step 4.

Step 3's second attempt, staged dequantization, was rejected by run
[38066265273](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38066265273).
The independent FP32 prepared-weight oracle, exact prepared/baseline parity,
multiple-wrap stream lifetime/coexistence fixtures and exact corresponding-request
output/usage/MTP checks passed. Warmed HTTP was 36.1270 s compute-dequant versus
36.4944 s staged-dequant (+1.02%); prefill was 30.2523 versus 30.3965 s. Cold HTTP
was 39.3714 versus 39.8020 s. Across two diagnostics, compute kernel intervals
fell 7690.888 -> 5588.926 ms, but compute waits rose 8919.324 -> 10026.323 ms,
with staged preparation 2821.063 ms. These intervals overlap and are not additive.
Moving expansion off compute did not improve request throughput. The experimental
stream, expanded-slot memory, prepared launch API and corresponding campaign were
removed, retaining the qualified bounded shared expansion. **Step 3 completed
with no additional deployment change:** both schedule experiments were negative.

## Step 4: small-batch CPU row scheduling

Verification diagnostics had 12,294 CPU routes over 31 rounds, with 1.8024 s CPU
work inside 3.0191 s inclusive MoE time. Small-batch CPU groups commonly trigger
two-phase row sharding using the whole configured 88-worker budget on the fixed
32-physical-core mask. Large prefill's 88-worker choice has separate measured
support; it does not establish the best small verification schedule.

Run [38070802684](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38070802684)
completed successfully with numerical/cache fixtures, exact CPU partition tests,
matching output/usage/MTP at corresponding request ordinals, and observed job
budgets. **Step 4 completed with a negative result:** restricting row sharding to
32 jobs for <=8-token batches was slower and has been removed along with its
selector, statistics and campaign mode. The original worker schedule is retained.

| Warmed median, 7111 prompt / 512 output | Full row budget | 32-row-job budget |
|---|---:|---:|
| HTTP request | 55.2376 s | 56.2639 s |
| Native prefill | 30.6047 s | 30.5371 s |
| Native decode | 24.3942 s | 25.6913 s |

The cap worsened decode 5.32% and HTTP 1.86%; all warmed HTTP sample ranges were
54.4870–55.3948 versus 56.1699–56.7868 s. Diagnostic verification CPU time grew
1.8138 -> 2.2770 s while host GPU wait stayed 0.6637 -> 0.6577 s. Fewer row jobs
increased CPU critical time despite the physical-core count. Neither this result
nor the unchanged large-prefill timing supports altering the 88-worker reference.

## Step 5: contemporary Strata and larger-context comparison

`--final-comparison` compares qualified adaptive156 NInfer against the installed
Strata executable `/opt/ai/strata/engine/strata` with its existing packs/config,
without rebuilding or editing Strata. Three fresh servers per engine run in
alternating order, each cold + four warmups + measured request, using identical
prompt text, 128-output-token limit and zero prompt reuse. Actual frontend counts,
outputs, native/response timings, startup and sampled GPU peak remain in evidence.

This is a practical configuration comparison, not equal representation or equal
memory: NInfer uses the original mixed artifact, BF16KV, MTP2 and 8192 capacity;
installed Strata uses its UD-Q4_K_XL pack, int8KV, spec4/min-p0.5, 262144 maximum
context, 32768 resident KV and native auto expert cache/prefill. Its working
directory, expert profile, resident CPU budget and placements remain as previously
configured; only generated request/log configs disable prompt/profile saving.
No numerical equivalence or matched quality is inferred from cross-engine times.

After preserving the 7K comparison, bounded longer-request checks use the same
larger prompt text (1400 records; actual token counts recorded), 64-output-token
limit, and cold + warmup + measured requests. NInfer is tested with 32768 and
262144 context/KV capacity, still requesting at most 156 experts/layer with the
existing allocator reserve authoritative; actual allocation/seeding is observed.
Strata uses its native 262144/int8/32768-resident profile. A startup/capacity failure
is recorded without erasing the completed comparison. Original weights/packs and
installed profiles remain unchanged; no unrelated process is stopped.

These checks measure larger prompt behavior and the memory cost of reserving
longer context. They do not measure a full 262K prompt, isolate selected-block
attention speed, demonstrate matched-quality superiority, or establish an untested
long-context optimization advantage. The final comparison is retained even when
requested capacity is infeasible. No deployment defaults change in this step.


## Step 5 completed: remaining gap and capacity tradeoff

[Run 38074752215](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38074752215)
completed successfully on October 10, 2026. Artifact 11678572331 retains the full
comparison and capacity evidence. All software checks and the unchanged independent
FP32 GEMM, stream/coexistence and cache/join hardware gates passed. Worst independent
GEMM NRMSE was 0.000225084; stream/coexistence remained within 0.002 NRMSE and
0.99999 cosine. This does not close historical full-model drift qualification.

Three alternating fresh servers per engine each ran cold + four warmups + measured
requests. Every timing request used 7111 prompt / 128 output / zero cached tokens.
Phase medians below are calculated independently, so their sum need not equal the
HTTP median.

| Warmed median | Qualified adaptive156 NInfer | Installed native Strata |
|---|---:|---:|
| HTTP request | 36.4183 s | 10.5422 s |
| Native prefill / response prompt time | 30.2255 s | 8.1898 s |
| Native decode / response predicted time | 5.7399 s | 2.2574 s |
| Sampled peak GPU memory | 29.83 GiB | 31.71 GiB |

NInfer HTTP was 3.45 times Strata's (warmed ranges 36.3498–36.4491 versus
10.4966–10.7430 s). Cold HTTP medians were 39.7822 versus 16.5350 s; one NInfer
cold request took 56.3483 s, so cold startup/request variability is retained in
raw evidence. NInfer fully seeded 7488 experts (156/layer), 20,704,739,328 compact
cache bytes, with the 2 GiB allocator reserve preserved.

Measured NInfer requests repeat one output and MTP work: 71/112 accepted/drafted
in 56 rounds. Measured Strata requests have two output hashes and 70/117 or 73/112
accepted/drafted; cross-engine outputs differ. NInfer BF16KV/MTP2/original mixed
weights and Strata int8KV/spec4/UD-Q4_K_XL/native placements are different practical
configurations. Equal frontend counts do not imply equal arithmetic, model quality,
speculative work or memory. **NInfer has not exceeded Strata in this comparison.**

All three larger-request cases completed cold + warmup + measured requests with
28311 prompt / 64 output / zero cached tokens. The table reports the single final
measured request per case, not a three-server performance estimate.

| Configuration | HTTP | Prefill | Decode | NInfer resident experts/layer | Peak GPU |
|---|---:|---:|---:|---:|---:|
| NInfer context/KV 32768 | 124.9823 s | 121.6967 s | 3.2509 s | 149 | 29.99 GiB |
| NInfer context/KV 262144 | 128.3533 s | 124.9241 s | 3.2113 s | 101 | 29.92 GiB |
| Strata native context 262144, resident KV 32768 | 31.0369 s | 29.5200 s | 1.3099 s | — | 31.71 GiB |

NInfer retained its reserve while clipping requested 156-slot residency to 149
(19,775,680,512 bytes, 7152 seeded experts) at 32K and 101 (13,404,991,488 bytes,
4848 seeded experts) at 262K. The two NInfer measured outputs match; no cross-case
arithmetic or work-equivalence claim follows. Both capacities are feasible for the
observed 28K request. **A full 262K prompt and isolated selected-block attention
performance remain untested.** Reserving larger KV displaces expert cache and
provides no demonstrated long-context speed advantage here.

All five authorized steps are complete. The qualified Volta GEMM/CPU-overlap
defaults remain; adaptive156 remains an 8K screening reference. Both Step 3
schedule experiments and the Step 4 CPU row cap were rejected and removed.
No production defaults, model representations, installed Strata assets or original
forwardport branch changed in this final comparison.

The evidence directs subsequent work toward reducing actual expert execution and
transfer/CPU-miss cost, with a qualified >=4-bit Volta expert representation and
small-route execution as concrete candidates. Merely rearranging the same schedule
has exhausted its tested benefit. Long-context work should address KV residency
cost before assuming a 262K performance advantage; full-length measurement and
selected-block attribution require a separate bounded investigation. These are
next design directions, not measured wins or additional campaigns launched here.


## Authorized six-step follow-up

On October 10 the user authorized the following follow-up after the completed
five-step investigation. Preserve the qualified deployment and original assets;
new arithmetic paths remain opt-in until their affected independent operator and
full-model numerical gates support promotion. Do not repeat rejected scheduling
experiments. No Phase18/SV8, below-4-bit weights, cache eviction, unrelated process
termination, security changes or installed Strata modifications are authorized.

1. **Non-MoE prefill attribution — completed.** Use qualified adaptive156 at
   8192 context, BF16KV, MTP2, 2048 chunks and 88 workers on 32 physical cores.
   One uninstrumented fresh server and one diagnostic fresh server each execute
   cold + four warmups + measured 7111/128/zero-reuse requests. Byte-delimited
   per-request native-log windows isolate prefill chunks from verification and
   later requests. Require complete prompt coverage, stage reconciliation and
   exact corresponding output/usage/MTP; record actual residency, native phase
   timings and telemetry overhead. Existing CUDA event/SV0 ledgers separate QSA
   projections/indexer/attention, GDN projections/recurrence/controls, PLE,
   hyper/norm boundaries, embedding/staging, final norm/head and MoE. Stage
   intervals include host gaps and dependencies; they are not pure GPU kernel
   times or exclusively BF16. Native-minus-ledger is an accounting difference,
   not a measured launch-gap category. Request-wide expert-stream intervals and
   inclusive host scopes remain non-additive. Nsight Systems was unavailable in
   the prior runner; record availability again, without installation or security
   changes. A missing distinction is grounds for a bounded targeted observation
   only if it changes the next design choice.
2. **Conditional SV7 revisit — pending attribution.** Dense projection TensorOp
   and QSA attention are separate candidates. Re-screen only a measured material
   contribution in the new baseline; independent operator gates precede timing,
   full-model oracle qualification precedes default promotion. Earlier isolated
   kernel gains and the old negative HTTP screen are not new end-to-end evidence.
3. **MoE work/bytes — pending.** Compare achieved FP16 GEMM throughput with a
   same-shape device-resident reference; distinguish actual GPU idle dependency
   gaps from merely recorded compute-stream waits. Investigate grouped expert
   launches and fewer dequant/staging passes, with scratch/residency tradeoffs and
   affected oracle/lifetime checks. Select one evidence-backed candidate rather
   than another ordering sweep.
4. **Context-aware cache planning — pending.** Retain static64 default and
   adaptive residency opt-in. Existing allocator already clips to the available
   budget; compare policies at the same context/KV capacity, including feasible
   adaptive101 at 262K. If unused KV reservation dominates, consider lazy device
   allocation with coherent cache budgeting; paged addressing alone does not
   imply lazy physical allocation. No host-tiering/SV8 implementation.
5. **Full-length FP8 KV — pending.** Existing Flash-Next/Volta code includes FP8
   KV storage and attention consumers despite no native FP8 arithmetic. Qualify
   the affected storage/attention contract and measure a manageable prompt first;
   advance to actual 128K and 262K prompts only after correctness/capacity gates.
   Record peak memory, resident slots, throughput and output/work, preserving
   allocator reserves. Prior 29.8 GiB peak already includes active GEMM scratch.
6. **Controlled Strata comparison — pending, baseline moved earlier.** After
   attribution, establish controlled prefill with speculation disabled where
   the installed serving route supports it; otherwise record the concrete
   limitation and use an applicable native route without editing installed
   assets. Match prompts, capacity, KV, cache budgets and placement as supported,
   then compare decode/speculation separately. Exact quantization equivalence
   requires represented-weight equivalence, not equal nominal bits. Reuse
   existing valid measurements rather than rebuild or repeat unchanged arms.

The execution order is attribution, controlled baseline, the largest justified
kernel/work reduction, KV/residency comparison and full-length checks. Numbered
steps retain the user's six proposals; conditional steps may be rejected by
measured evidence rather than forcing an unhelpful implementation. The initial
protected run uses `--prefill-breakdown` and produces
`prefill-breakdown-report.json`; original models/packs and installed Strata remain
unchanged. All GPU jobs share `v100-sv0-hardware`, cancel-in-progress false, with
CMake parallel level 32. Evidence aliases remain outside uploaded bundles; uploads
reject symlinks and bundles >=1 GiB.


## Six-step Step 1 completed; Step 6 serving controls started

Run [38086716479](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38086716479)
passed all gates and paired output/usage/MTP checks; artifact 11682880397 retains
per-request evidence. Uninstrumented warmed HTTP/native prefill/decode were
36.4945 / 30.0390 / 5.7993 s. Diagnostic native prefill was 38.7175 s, of which
38.6655 s was covered by four chunks (all 7111 prompt tokens); the accounting
difference was 0.0521 s. Instrumentation increased warmed prefill by 8.6785 s
(28.9%), principally complicating the inclusive MoE/host timeline. Cold reference
was also variable (49.1702 s prefill vs diagnostic 38.1132 s). These are two
fresh servers, not a precision timing estimate or pure GPU kernel profile.

| Warmed diagnostic category | Seconds |
|---|---:|
| MoE inclusive | 24.4489 |
| GDN projections | 4.7744 |
| Hyper/norm boundaries | 3.2098 |
| QSA projections | 1.6026 |
| QSA attention/indexer | 1.9482 |
| Final norm/head | 1.3685 |
| GDN recurrence/controls | 0.6601 |
| PLE inclusive | 0.6504 |

Non-MoE totaled 14.2165 s in this diagnostic, with similar cold intervals. This
identifies GDN projections and hyper boundaries as substantial targets, while
QSA main attention itself was 1.7629 s. It does not establish a production lower
bound or attribute all projection time to BF16. Nsight Systems remains unavailable.
SV7's PLE projection path alone is a small part of current prefill; a new dense
candidate must be selected from actual dispatch ownership, not isolated old gains.
Inclusive host spans and request-wide stream timings remain non-additive.

Step 6 starts with `--controlled-comparison`: three alternating fresh servers per
engine, each cold + four warmups + measured 7111/128/zero-reuse requests. Match
physical-core placement, 8192 maximum context, requested 2048 prefill chunks,
16-bit KV storage width and requested two-token MTP windows. Strata uses FP16 KV;
NInfer BF16 KV. Generated Strata config removes the host-backed KV-resident option,
sets explicit 6606 expert slots (approximate 20.7 GB from average expert size),
and preserves installed assets/profile/min-p/resident CPU budget. Actual allocation
and lending/fallback logs govern; equal cache bytes are not assumed from requested
counts. No changes to weights or production defaults.

Pinned installed Strata source (`src/program/generate.cpp`) explicitly rejects
serving when `spec < 2`, so this is an initial serving control, not a no-spec
comparison. A separate supported native no-spec route remains pending. Weight
quantization, BF16/FP16 KV values, speculative algorithms and acceptance still
prevent matched-quality/equal-work claims. Each report retains phase timings,
usage, output hashes, accepted work and observed memory/cache evidence.


## Step 6 initial result and corrected follow-up

Run [38090951809](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38090951809)
passed all gates, completing three fresh servers per engine with six requests each.
Artifact 11684975614 retains the original controls and phase/output/work evidence.

| Warmed medians | NInfer | Strata serving controls |
|---|---:|---:|
| HTTP | 36.9234 s | 21.2035 s |
| Native prefill / response prompt | 31.0262 s | 18.9097 s |
| Native decode / response predicted | 5.7742 s | 2.2482 s |

These changed several Strata controls simultaneously; the reduced practical gap
(1.74x HTTP) does not isolate a single cause or quantization quality. NInfer warmed
MTP accepted/drafted 71/112 in 56 rounds; Strata 48/66 or 47/64, with different
outputs. All requests retained 7111/128/zero-reuse frontend counts.

The requested Strata count 6606 did **not** match the intended byte budget. Native
cache planning multiplies the count by maximum blob size (3,993,600 B), then packs
smaller ranked expert blobs into that budget: actual cache was 8405 slots / 24.53
GiB, versus NInfer 7488 experts / 20,704,739,328 B (19.28 GiB). Nominal Strata spec2
also includes the anchor, and default suffix lookup can increase the window.

A focused `--controlled-strata-followup` therefore measures only three changed
Strata fresh servers; valid NInfer measurements are reused from the preceding run.
Request 5184 max-blob units = 20,702,822,400 B (0.0093% below NInfer), inspect actual
ranked-cache allocation, set spec3/mtp-max-t3 for at most two MTP drafts, disable
suffix lookup and probability truncation. Keep context8192, FP16KV, requested2048
chunks and physical-core placement. This is a separately measured follow-up, not
another contemporary alternating comparison or a matched-quality claim.

Source inspection corrected the promised native no-spec route: the pinned engine's
`native_pack` check in `src/program/generate.cpp` rejects `spec < 2` for native
CLI as well as serving. A no-speculation comparison is unavailable with the
preserved installed engine/pack. Do not launch a known-invalid CLI or claim draft
acceptance disabled is speculation disabled. The initial assumption that CLI could
provide this baseline was wrong; this concrete limitation bounds Step 6.

## Six-step Step 6 completed; Step 2 GDN operator screening

Corrected follow-up [38094038959](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38094038959)
passed all gates. Three changed Strata servers (six requests each) allocated 6608
ranked expert slots / 19.28 GiB, matching NInfer's 19.28 GiB byte budget to the
native log's precision. Generated config limits MTP to two drafts and disables
suffix drafting/min-p truncation. Installed assets remain unchanged.

| Warmed medians | NInfer reference | Corrected Strata |
|---|---:|---:|
| HTTP | 36.9234 s | 24.4306 s |
| Native prefill / response prompt | 31.0262 s | 21.0356 s |
| Native decode / response predicted | 5.7742 s | 3.5515 s |
| Sampled GPU peak | 29.83 GiB | 26.70 GiB |

The HTTP gap is 1.51x, not the unmatched native-profile 3.45x gap. This does not
establish equal quality or isolate any changed control: NInfer's valid reference
is reused from run38090951809, BF16/FP16 KV values and expert representations differ,
and native accepted/drafted work remains unequal (NInfer71/112, Strata69/118 or
73/112). Strata warmed outputs also vary across fresh servers. The preserved
installed engine cannot disable speculation for this native pack. Step 6 is
complete as a bounded controlled comparison, with that unresolved limitation;
NInfer has not beaten Strata.

Step 2 source attribution identifies the largest dense component, GDN's FP8/F32
projections (16384x2560 input, 2560x6144 output). Their current Volta QPN route
already uses fused FP16 Tensor Core MMA, but breaks every large prefill into at
most 32-token calls, repeatedly reading packed weights. The old SV7 BF16 selector
does not own these projections. Re-enabling it would not address this cost.

The next protected job screens a directly invoked alternative operator: expand
E4M3 weights once per full chunk, convert BF16 activations to FP16, CUTLASS Sm70
full-chunk GEMM with FP32 accumulation/output scratch, then multiply the represented
FP32 row scale before final BF16 rounding. This preserves F32 scales rather than
reusing the older BF16-scale operator's early output rounding. It is deliberately
not integrated into dispatch or a production selector yet. Repacking is temporary,
not a model/source-pack conversion. At T2048 the larger input projection requires
218 MiB temporary scratch (80 MiB weights, 10 MiB inputs, 128 MiB FP32 outputs),
which must enter the planner before a request-level candidate is considered.

Qualification checks the baseline and candidate directly against the independent
sequential FP32 mathematical dot product from E4M3 codes, BF16 inputs and stored
F32 row scales. Full outputs at both model shapes and T128/257/2048 are checked;
129-token extreme fixtures include signed finite top/subnormal codes, zeros,
large/small finite activations and scales. Repeated independently generated
row/token prototypes bound oracle cost without reducing device shape or output
coverage. Input values are within finite FP16 range; arbitrary BF16 overflow and
sub-FP16 values are not qualified by this screening. NRMSE<=0.002 and cosine>=0.99999,
output/workspace guards and scope reuse must all pass before event-timed operator
screening (three warmups/seven alternating repetitions including all expansion/cast/scaling
work). Operator gains alone will not establish request gains or authorize a default
change; full-model qualification remains required before arithmetic promotion.


### Step 2 first qualification stopped on a BF16 output-criterion error

Run [38096951500](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38096951500)
compiled successfully. Baseline and candidate passed the complete 16384x2560
fixtures (T128/257/2048 and extreme129), with identical reported oracle errors.
The 2560x6144 baseline at T257 passed aggregate NRMSE0.00167733/cosine0.999999,
but the additional per-token unrounded-oracle gate stopped at 0.00203029.
No timings were collected and the candidate was not evaluated on that final cell.

A host calculation using the same independent oracle and represented inputs
confirmed **ideal nearest-BF16 output alone** has worst-token NRMSE0.00203029
on this fixture. That extra per-token requirement was impossible even for ideal
BF16 rounding. Correct it to compare each token against the independently rounded
public BF16 oracle output; retain the original aggregate unrounded FP32 oracle
gate (NRMSE<=0.002/cosine>=0.99999), all inputs/shapes/extremes and lifetime guards.
Report the ideal rounding floor as well as per-token BF16-reference error. No
production arithmetic or aggregate thresholds change. Re-run only this bounded
operator qualification/screen; a serving comparison remains conditional on its
result.


### Step 2 operator screening passed; opt-in request comparison started

Run [38098702122](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38098702122)
passed every baseline/candidate numerical and lifetime gate after the output
criterion correction. Aggregate NRMSE ranged 0.00160–0.00170, cosine>=0.999999;
maximum per-token BF16-reference NRMSE was 0.00156828. Operator medians include
weight expansion, activation conversion, GEMM and row scaling:

| GDN shape / tokens | QPN | Full-chunk GEMM |
|---|---:|---:|
| 16384x2560 / 128 | 2.02957 ms | 0.616416 ms |
| 16384x2560 / 257 | 4.30182 ms | 0.834560 ms |
| 16384x2560 / 2048 | 29.5281 ms | 2.64499 ms |
| 2560x6144 / 128 | 0.536576 ms | 0.355328 ms |
| 2560x6144 / 257 | 1.17146 ms | 0.353280 ms |
| 2560x6144 / 2048 | 8.53504 ms | 0.931840 ms |

This establishes an operator case, not an end-to-end result. The implementation
now has **opt-in** `NINFER_V100_FP8_PREFILL_GEMM=1` (default0), restricted to these
two F32-scale projection shapes at T>=128. This includes GDN input and GDN/QSA
output; QSA input/vocabulary projections, smaller batches,
and non-Volta routes retain their existing path. The public Linear workspace
capacity includes the candidate's maximum scratch over the requested interval;
GDN and runtime planners consume that capacity after the composition correction below. No hidden allocation or
persistent expanded weights are introduced. The public dispatcher qualification
also covers T127 below the crossover and planned arena scope/guard reuse.

The protected request comparison uses three alternating fresh servers per arm,
each cold + four warmups + measured 7111 prompt/128 output/zero reuse requests.
Both request adaptive156, qualified expert paths, BF16KV, MTP2, 8K context,
2048 chunks and 88 workers on 32 physical cores. Actual clipped expert capacity,
profile seeding and sampled GPU peak are observed; the extra scratch may change
residency and cannot be assumed free. Report HTTP/native prefill/decode and
outputs/usage/accepted work separately. Arithmetic changes can alter routes/output
and MTP acceptance, so cross-arm exact output/work parity is not a gate or a
quality claim. Two separate diagnostic requests per arm validate real GDN dispatch
and input36/output48-layer/7111-token conservation without instrumenting timing.
The opt-in remains experimental; full-model qualification is required before a
production arithmetic default change.


### Step 2 serving integration correction and partial-run reuse

Run [38098932339](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38098932339)
passed public dispatch/operator and unchanged expert numerical/lifetime gates.
The first QPN control completed all six requests (warmed HTTP36.6983 s); the first
candidate request terminated with an arena `std::bad_alloc`, so no candidate
request timing or end-to-end benefit is established.

The full text-prefill planner independently composed GDN tensors plus recurrence
scratch, omitting the newly required input-projection Linear scratch. Updating
Linear and the standalone GDN planner was insufficient. Correct the full-prefill
composition to consume GDN's authoritative complete workspace capacity, including
projection and recurrence maxima under the existing shared arena scopes. This
also places the extra bytes in the runtime/expert-cache budget before allocation.
The failure occurred with 2.17 GiB device memory free after startup; it was not
evidence that adaptive156 necessarily exhausts total VRAM.

Source inspection also corrected the earlier **GDN-only** scope claim: QSA uses
the same 2560x6144 output projection. The qualified two-shape Op selector therefore
also accelerates QSA output. Rename the experimental selector to
`NINFER_V100_FP8_PREFILL_GEMM` (default0); no old selector alias remains. Keep
16384x2560 GDN input and 2560x6144 GDN/QSA output qualified, with QSA input and
vocabulary unchanged. Diagnostics now require 36 input and 48 output projections
per complete prompt, excluding narrow verification batches.

Resume the bounded request comparison after this correction, reusing only the
completed first QPN server from artifact11686872868. Validate its complete
cold/four-warmup/measured schedule, fixed counts, zero reuse, native timings,
explicit baseline selector, actual allocation and profile seeding before copying
its evidence. Measure the remaining two QPN and three candidate fresh servers,
then two diagnostic requests per arm. The report identifies the reused source
run; it is not claimed to be wholly contemporaneous. No additional tuning sweep,
weight conversion, cache eviction or default promotion is involved.


### Step 2 request screening complete: retain dense projection opt-in

Run [38099919154](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38099919154)
passed the public-dispatch/independent operator, expert arithmetic, stream lifetime
and cache join gates, and completed the resumed three-server-per-arm comparison.
Each fresh server received cold + four warmups + measured 7111 prompt/128 output,
zero reuse requests. The first unchanged QPN control was reused from the earlier
partial run; the other five timing servers were measured after the planner fix.

| Warmed median | QPN control | Full-chunk dense GEMM | Change |
|---|---:|---:|---:|
| HTTP | 36.4730 s | 31.8961 s | -12.55% |
| Native prefill | 30.3650 s | 25.6655 s | -15.48% |
| Native decode | 5.7614 s | 6.0977 s | +5.84% |
| Sampled peak GPU | 29.8337 GiB | 29.9177 GiB | +0.0840 GiB |
| Actual experts/layer | 156 | 155 | -1 |

HTTP ranges were 36.3911–36.6983 s and 31.7557–32.3774 s respectively.
Both arms produced 7111/128 tokens with no reuse. Within each arm the warmed
outputs and native MTP work were consistent across fresh servers; cross-arm
outputs differed. MTP acceptance changed from 71/112 drafted tokens (56 rounds)
to 68/116 (59 rounds), so decode differences are confounded by changed work.
The candidate seeded 7440 experts versus 7488 for the control. Separate diagnostic
requests passed input36/output48-layer and per-prompt7111-token conservation,
with the expected selector dispatch. Inclusive diagnostic spans remain unsuitable
for adding up exclusive GPU cost.

This is a measured prefill and request improvement with the same represented
weights, not a model-quality equivalence result. Keep
`NINFER_V100_FP8_PREFILL_GEMM=1` opt-in and default0; full-model numerical
qualification is still required before arithmetic default promotion. No further
SV7 attention sweep is justified before addressing the larger MoE share.
Step 2 performance screening is complete. Steps 3–5 remain open; the bounded
controlled Strata comparison in Step 6 is already complete with its documented
representation and speculative-work limits. NInfer has not been shown to beat Strata.

### Step 3 started: resident same-shape GEMM throughput reference

Measure the production expert pair (including compact-weight expansion, gather,
activation/scaling and scatter) alongside a resident synthetic FP16 reference
for its two GEMM geometries: gate/up1280x2560 and down2560x640. The reference
uses the same cuBLAS transpose/layout, FP32 accumulation/math mode, algorithm,
4 MiB workspace, 256-route tile and eight-route padding. Its two inputs are
independent resident operands; it does not reproduce the expert pair's SiLU or
route joins and is not a production implementation.

Before timing, qualify all reference outputs against independent sequential FP32
dots from represented FP16 operands at the relevant padded sizes, with output
canaries. Retain NRMSE<=0.002/cosine>=0.99999. Existing expert-pair qualification
also retains extreme represented inputs/divisors, guard checks and exact zero
behavior before its timing. Report pure GEMM gate/up, down and pair medians with
three warmups/seven measurements at32/33/64/128/256/512 routes, both useful and
padding-inclusive TFLOPs. The production pair's established mean-cost screen is
reported separately; differences are headroom evidence, not exclusive additive
phase attribution or an end-to-end speedup.

This bounded job changes no production math, scheduling, model assets or defaults
and starts no servers. Actual prefill GPU idle attributable to H2D remains an open
measurement: existing request-wide overlapping copy/wait/kernel event sums cannot
establish whole-device idle or the prefill critical path, and nsys was unavailable.
Use this result to decide whether grouped launches, fewer expansion/staging bytes,
or a better GEMM kernel is worth an implementation, rather than repeat scheduling
experiments. Step 3 is not complete merely because this reference is measured.


The initial reference job [38102068248](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38102068248)
built both targets and passed every production expert gate before its existing
cost screen. The new resident reference stopped on nonfinite output before any
reference timing. Its host input generator mixed an unsigned token index with
subtraction, turning intended negative inputs into large positive values that
overflowed FP16 conversion. Correct the generator to signed token arithmetic;
its intended input range is [-0.875,0.875]. No production arithmetic, numerical
threshold or benchmark geometry changes. Re-run only the bounded operator job.


### Step 3 resident reference complete; work/bytes optimization remains open

Corrected run [38102172529](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38102172529)
succeeded on Tesla V100-PCIE-32GB (80 SMs). Production expert independent-oracle,
extreme/divisor, guard and zero gates passed; every synthetic resident GEMM output
matched the independent FP32 dot exactly on this dyadic fixture. All timing below
excludes H2D transfers and CPU staging.

| Routes | Full expert pair mean | Resident GEMM pair median | Resident useful TFLOPs |
|---|---:|---:|---:|
| 32 | 0.143565 ms | 0.048128 ms | 6.54 |
| 33 (40 padded) | 0.150118 ms | 0.052224 ms | 6.21 |
| 64 | 0.156058 ms | 0.062464 ms | 10.07 |
| 128 | 0.171616 ms | 0.069632 ms | 18.07 |
| 256 | 0.229568 ms | 0.120832 ms | 20.83 |
| 512 (two tiles) | 0.380512 ms | 0.233472 ms | 21.56 |

The same-shape resident reference is a practical comparison for these narrow
GEMMs, not a measured saturated device peak. At256 routes, gate/up achieved
18.00 TFLOPs (0.093184 ms), down28.25 TFLOPs (0.029696 ms). Separate phase medians
are not additive with the pair median. The full expert path adds weight expansion,
gather, activation/scaling and scatter, with different represented inputs; its
roughly threefold cost over the reference at32 routes and twofold at256 motivate
reducing launches/expansion/staging work. They do not isolate an exclusive
phase cost or prove compute saturation, H2D starvation or an end-to-end gain.
The production pair remains substantially faster than SIMT at these sizes.

Next, establish phase-owned compute-stream dependency-wait evidence and evaluate
one grouped/fused work reduction if justified by actual active-expert geometry and
scratch constraints. Keep hardware evidence bounded and avoid another ordering
sweep. Step 3 remains open; Steps 4 and 5 are pending. No arithmetic default was
promoted and no claim of beating Strata follows from this reference.


### Step 3 phase-owned stream work/dependency measurement started

The next bounded job uses one uninstrumented and one diagnostic fresh server,
each cold + four warmups + measured 7111 prompt/128 output/zero reuse requests.
Both explicitly select the successful dense projection opt-in, qualified expert
paths, requested adaptive156, BF16KV, MTP2, 8K context, 2048 chunks and 88 workers
on 32 physical cores. Observe actual clipped capacity/seeding and GPU peak; require
exact corresponding output/usage/native MTP work between reference and telemetry.

Stream timing now captures phase/executor/transaction at submission, retaining
ownership across ring reuse and later finish. A multi-wrap fixture submits five
32-route prefill groups and one seven-route verification group, then finishes
inside verification; its emitted counters must keep those owners separate.
Existing event placement, ring reuse, synchronization and arithmetic are unchanged.
Only enabled telemetry collects the additional host metadata. Independent dense
and expert gates plus stream lifetime/cache join gates precede serving measurement.

The byte-window report reconciles each request's prefill token coverage and each
execution owner's streamed routes/compact-weight bytes against native SV0 counters.
It reports copy, compute dependency wait and whole expert-kernel intervals separately
for prefill/verification, plus actual streamed group-size buckets and modeled GEMM
expansion writes/useful FLOPs. The stream wait interval spans the compute-stream
wait for the transfer-ready event. It is **not whole-GPU idle**, an exclusive
unperturbed critical path or a value to add to overlapping CPU/copy intervals.
Stage/event/SV0 diagnostics perturb timing; paired per-ordinal overhead is recorded.
The phase-specific evidence is intended to select one grouped/fused work reduction,
not launch another ordering sweep. Step 3 remains open after this measurement.


### Step 3 phase-owned evidence complete; fused expansion candidate selected

Run [38102677923](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38102677923)
passed numerical/lifetime/cache gates, the mixed-scope ring ownership fixture,
complete request/owner route and weight-byte conservation, and exact corresponding
output/usage/native MTP work between reference and instrumentation. Both servers
actually allocated155 experts/layer, seeded7440, and peaked29.9177 GiB.

The warmed uninstrumented request took32.4921 s HTTP/25.6397 s native prefill.
Instrumentation took41.9032/35.1561 s: +9.5164 s prefill overhead. In that diagnostic
prefill, streamed compact weights totaled55,677,167,616 bytes over20,136 expert
pairs. Copy intervals summed5.6343 s; compute-stream transfer dependency waits
summed4.6878 s; streamed expert execution intervals summed4.0095 s. These overlap
other scopes and are not a production latency decomposition or whole-GPU idle.
No streamed verification work was observed for this policy; phase ownership and
coverage still reconciled with all native execution owners.

Of those streamed pairs,16,166 used GEMM, covering2,054,365 routes. Their modeled
FP16 expansion writes totaled158,918,246,400 bytes. GEMM group buckets were5925
at32–63 routes,5463 at64–127,3080 at128–255,1321 at256–511 and377 at512+;3970
smaller groups used SIMT. Most GEMM groups therefore remain narrow. Materializing
many active experts as expanded FP16 simultaneously would add substantial scratch
to an already near30 GiB footprint and can waste padded work in a uniform grouped
GEMM. A single grouped launch for every active expert is not selected without a
bounded memory/layout case.

Select one bounded **work reduction**: fuse the two gate/up and down expansion
launches into one kernel, decode both E2M1 values from each packed byte together,
load their common E4M3 scale once, and store the same two FP16 weights together.
This halves expansion threads and duplicate code/scale/address decode work and
removes one expansion launch per GEMM expert pair. It does **not** reduce expanded
weight bytes, H2D bytes, GEMM work or CPU misses. Existing compact storage, FP32
divisors, shared scratch, ring ownership, tile256/pad8 and32-route crossover remain.
No additional device memory is planned.

The experimental `NINFER_V100_EXPERT_PAIR_EXPAND=1` is default0 on all builds and
applies to both streamed and resident expert GEMMs. The independent represented
NVFP4 FP32 oracle, extreme inputs/scales/divisors, guards and zero gates precede
timing; candidate fixtures additionally require bit-exact baseline expert outputs.
Stream multi-wrap/destructor/coexistence and cache join gates run on the candidate.
Request screening compares three fresh-server cells per arm, cold/four-warmup/
measured7111/128/zero reuse, with dense projection opt-in1 in both. Reuse only the
completed uninstrumented control from this phase-owned run after validating source
revision, selectors, frontend/native work, full schedule, placement/context flags,
actual cache and seeding. The other two controls and all three candidate servers
are new; the report identifies reuse and is not wholly contemporaneous. Require
exact corresponding output/usage/native MTP work across arms and request ordinals;
observe dispatch, cache capacity and GPU peak. Retain only if the request-level
result justifies it, otherwise remove the experimental route and advance. Step 3
is still open pending this one implementation test; Steps 4 and 5 remain pending.


### Step 3 complete: fused expansion retained as an opt-in

Run [38104175339](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38104175339)
passed independent numerical/extreme/guard gates, exact scalar operator parity,
stream lifetime/ownership and cache joins. All corresponding request outputs, usage
and native MTP work matched across the six server cells. One scalar control was
reused from38102677923; the other five cells were fresh and alternated.

| Warmed median | Scalar expansion | Fused pair expansion | Change |
|---|---:|---:|---:|
| HTTP | 32.4921 s | 32.0157 s | -1.47% |
| Native prefill | 25.6397 s | 25.2293 s | -1.60% |
| Native decode | 6.1138 s | 6.0832 s | -0.50% |

HTTP ranges overlap (31.6339–32.6828 s scalar,31.5468–32.0456 s fused). This is a
small positive screening result, not a statistically established universal win.
Independent expert-pair mean cost improved0.132096→0.0995392 ms at32 routes and
0.208896→0.175117 ms at256. Actual residency stayed155 slots/layer,7440 seeded,
and peak29.9177 GiB. Retain `NINFER_V100_EXPERT_PAIR_EXPAND=1` as an opt-in with
default0; do not promote arithmetic defaults or repeat unchanged timing sweeps.
Step3's bounded work reduction is complete. Whole-device H2D idle was not measured;
phase-owned dependency intervals remain instrumented evidence with the stated limits.

### Step 4 started: residency benefit under 262K allocated context

Existing runtime planning sizes KV by configured physical pages; expert allocation
already clips the requested maximum against remaining memory and its2 GiB reserve.
That clipping is a memory constraint, not a measured cache-policy recommendation.
Keep static64 default and adaptive LRU opt-in. Compare static64 against requested
adaptive156 at262144 max-context/KV capacity, recording actual clipped slots
(previously101 without the new dense scratch), startup seeding and sampled GPU peak.
The allocator remains authoritative; no hard-coded101 capacity or reserve changes.

Use the same7111 prompt/128 output/zero reuse workload, BF16KV/MTP2/2048 chunks,
88 workers on32 physical cores, dense GEMM opt-in1 and fused expansion opt-in1.
Three alternating fresh servers per arm each receive cold/four warmups/measured.
Independent dense/expert and lifetime/cache gates precede timing. Record output
and native MTP differences rather than require cross-policy exact parity, since
residency can change CPU/GPU arithmetic. This is a262K allocation test, **not**
a full-length prompt test. Decide whether lazy/paged device KV is justified from
this result; Step5 separately qualifies FP8KV and tests actual long prompts.


### Step 4 complete: adaptive residency still helps at262K capacity

Run [38105697912](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38105697912)
passed all retained numerical/lifetime/cache gates and all36 fixed7111/128/no-reuse
requests. Three alternating fresh servers per arm used dense/fused opt-ins, BF16KV,
MTP2,262144 allocated context,2048 chunks and88 workers on32 physical cores.

| Warmed median | Static64 | Adaptive (actual101) | Change |
|---|---:|---:|---:|
| HTTP | 36.6172 s | 34.8458 s | -4.84% |
| Native prefill | 27.6540 s | 26.7536 s | -3.26% |
| Native decode | 8.4167 s | 6.4425 s | -23.46% |
| Sampled peak GPU | 25.3435 GiB | 29.9158 GiB | +4.5723 GiB |

Actual capacity/seeding stayed64/3072 versus101/4848, with the2 GiB reserve intact.
Cold HTTP medians43.8947 versus37.7933 s. Each arm had stable measured outputs,
but cross-policy output and native MTP work differ: decode gain is confounded.
The combined policy/capacity prefill benefit survives clipping at262K. Keep
static64 default; adaptive remains an opt-in whose actual capacity is selected
after context/scratch allocation. No hard-coded101 rule or reserve change.
Because the gain persists, defer lazy/paged device-KV redesign; first qualify
FP8 KV and measure its actual long-context feasibility/memory. This screen uses
a7K prompt and cannot establish full262K-prompt performance or quality.

### Step 5 started: software FP8 KV and actual long prompts

Inspection confirms Flash-Next QSA stores raw, unscaled E4M3 key/value codes via
software Volta conversion; the generic scaled row256 KV codec is a different
consumer contract despite the shared storage selector name. Qualify the affected
software E4M3 codec (every finite BF16 input against an independent nearest-even/
saturation exact oracle), the QSA prepared-query independent FP64 attention oracle
(including represented FP8/BF16, causal tails, noncontiguous pages, extreme scores
and serving batch sizes), and full QSA FP8 append/decode/prefill integration.
Retain independent dense/expert and stream/cache gates before serving. Existing
QSA oracle criteria are preserved; no arithmetic code/default or thresholds change.

Use262144 context/KV capacity, FP8, requested adaptive156 (actual reserve-clipped
capacity observed), dense/fused opt-ins, MTP2/64output,2048chunks/88workers/physical32.
Two manageable repeated-prose prompts (360 and361 identical records), each cold
and measured, calibrate this artifact frontend's token increment and template
overhead. Then construct one actual prompt at or just above131072 tokens and
one at or just above262016, leaving64 output tokens and context headroom. Verify
reported frontend counts against calibration; record each request/native MTP/work,
residency/seeding and sampled peak GPU. Large requests are single cold feasibility
observations, not a throughput comparison or long-document quality qualification.

The serving harness keeps its normal900-second timeout but this mode allows5400
seconds per request; hardware budget180minutes. Partial reports are saved after
each cell and on failure; no completed result is erased. No prefix reuse/page-cache
eviction or foreign tokenizer is used. Defaults remain unchanged.


### Step 5 qualification stop: cross-schedule BF16 fixture correction

Run [38109293993](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38109293993)
built successfully and passed the exact unscaled-E4M3 conversion oracle for all
65,280 finite BF16 inputs. The independent QSA FP64 oracle passed FP8/BF16,
extreme scores and all tested batches through8192; maximum reported normwise
error was2.42387e-6 against the independently rounded public BF16 oracle.

The full QSA integration fixture then stopped before any server/timing cell at
T=2: sequential value-cache0xbeeb versus batched0xbeec. That fixture required
bit-exact equality for independently reduced floating-point projections at T<16,
in addition to its numerical gate. Such cross-schedule equality is not the
floating-point Op contract. Remove only that redundant bitwise assertion for
key/value projections; retain finite/nonvacuous checks and all existing numerical
tolerances (including1e-3 for these small cases). The exact conversion oracle and
independent attention oracle are unchanged. No production arithmetic or default
changes. Resume qualification and long requests; no completed serving cells exist
to reuse or rerun from this stopped job.

### Step 5 complete: actual 128K and near-262K prompts fit with FP8 KV

Run [38111959506](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/38111959506)
passed every qualification gate and all six requests. Exact software conversion
covered all65,280 finite BF16 inputs; independent FP64 QSA attention, full QSA
append/decode/prefill, dense/expert numerical, stream lifetime/ownership and cache
join gates passed. The corrected supplementary BF16 cross-schedule fixture retained
its numerical criteria: T=2 value-cache relative-L2 was1.499333e-6 and output
2.963199e-5, both below the existing1e-3 limits. No production arithmetic changed.

Serving calibrated14 tokens per repeated prose record plus31 frontend/template
tokens. Actual large-request counts exactly matched calibration. Each used a fresh
server,262144 allocated context/KV capacity, software FP8 KV, requested adaptive156,
dense/fused opt-ins, MTP2,2048 chunks,88 workers on32 physical cores, no prefix reuse
and no QSA MMA/CUDA graphs. Both generated64 tokens with zero cached prompt tokens.

| Actual prompt tokens | Cold HTTP | Native prefill | Native decode | Sampled GPU peak |
|---|---:|---:|---:|---:|
| 131,085 | 617.8754 s | 612.7882 s | 4.8700 s | 30.0037 GiB |
| 262,027 | 1,359.5693 s | 1,354.5581 s | 4.6027 s | 30.0037 GiB |

All four servers actually allocated128 experts/layer (16,988,504,064 bytes), seeded
6144 experts and preserved the2 GiB reserve. At the largest request, prompt plus
output totaled262091, leaving53 tokens below262144. Native MTP work was35 rounds/
68 drafted/28 accepted at128K and33/65/30 at near262K. Calibration prompts5071
and5085 also completed cold and repeated requests. FP8 thus permits more actual
resident slots than the separate BF16262K screen's101; that is not a matched
FP8-versus-BF16 latency or quality experiment.

This establishes real long-prompt memory/execution feasibility on the32GB V100,
not merely allocation feasibility. Inputs were synthetic repeated prose, each
large size had one cold request, and there was no matched Strata long-prompt arm.
It does not qualify long-document quality, establish general throughput, close
historical full-model drift or justify arithmetic default promotion.

### Six-step final checkpoint

| Step | Result / decision |
|---|---|
| 1. Non-MoE attribution | Complete; dense GDN projections motivated implementation. Instrumented ledgers are not exclusive unperturbed kernel times. |
| 2. Conditional dense/SV7 revisit | Dense opt-in improved warmed HTTP12.55% and prefill15.48%; outputs/MTP changed. Retain opt-in, not a default promotion. Further attention-only tuning was not justified by attribution. |
| 3. MoE work/bytes | Fused expansion improved HTTP1.47%/prefill1.60% in a small overlapping-range screen, with exact corresponding work/output parity and no extra VRAM. Retain opt-in; whole-GPU H2D idle remains unmeasured. |
| 4. Context-aware cache | Existing reserve-aware clipping produced101 adaptive slots at262K BF16 capacity; HTTP4.84%/prefill3.26% better than static64. Keep static64 default/adaptive opt-in; defer lazy device-KV redesign. |
| 5. Full FP8 long prompts | Qualified operators and completed actual131085/262027-token prompts plus64 outputs; peak30.0037 GiB,128 slots/layer. Feasibility, not matched quality or throughput. |
| 6. Controlled Strata comparison | Bounded controls completed; installed Strata rejects no-spec operation, and representations/native work differ. No matched-quality win or claim that Strata was beaten. |

Retained experimental flags are `NINFER_V100_FP8_PREFILL_GEMM=1` and
`NINFER_V100_EXPERT_PAIR_EXPAND=1`, both default0; unset or0 disables each.
FP8 KV is selected explicitly with `--kv-dtype fp8`. Cache policy remains explicit
static64 or adaptive LRU with actual capacity determined by context/scratch and
the allocator reserve. The original model, source packs, forwardport branch and
installed Strata were preserved. No Phase18/SV8, lower-bit weights, unrelated
process kills or cache eviction were introduced.

All authorized six-step screens are complete. Stop this test campaign and its
automatic continuation instead of starting unrequested wider tuning. Arithmetic
default promotion still requires full-model qualification; historical drift and
matched-representation/quality comparison remain unresolved limitations.
