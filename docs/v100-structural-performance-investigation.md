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

1. **Non-MoE prefill attribution — in progress.** Use qualified adaptive156 at
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
