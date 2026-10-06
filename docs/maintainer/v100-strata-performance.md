# Strata-derived V100 performance project

Implementation branch: `perf/v100-strata-derived`.
Frozen source: `forwardport/v100-flash-next` at
`e184de12772999f8d336ea395cedcc0cf0543c52`.
Reference inspected: `jmnargi/Strata-V100` `9d7774919e26d235359bc2c3001f61f607eb288d`.
Milestones are SV0–SV8 in the supplied specification; they do not advance Phase 18.

## Current milestone: SV1 expert residency

On 2026-10-04 the user explicitly accepted the frozen baseline and authorized
continuation. SV0 is accepted by measured baseline parity. The independent
Phase 11 oracle and its numerical thresholds remain unchanged and its existing
failure remains reported; this decision does not qualify that numerical gate.

`NINFER_V100_TELEMETRY=1` emits schema-1 JSONL on stderr for host-backed MoE
layer calls and PLE gathers. It also enables the existing expert-cache CUDA event
measurements. No routing, arithmetic, cache admission/replacement, or production execution
policy is changed. Enabled diagnostics add CUDA events and waits when reading
completed transfer/projection timings; use them only for attribution. With telemetry off, histogram construction, JSON formatting,
cache snapshots, and additional clock reads are skipped.

Layer records contain all 512 routing counts, distinct expert counts, hit/CPU
route counts, router rendezvous wall time, CPU branch wall time, existing GPU hit
window and join time, and actual submitted transfer bytes. Ready/uploading/leased
counts are local to the reported layer. Cache fill/admission/eviction counters
are **global cumulative snapshots**, including background fills; do not sum them
across layers or treat them as per-request counters. A fresh process or cache
reset restarts those cumulative counters. `cache_transfer_budget_bytes` is the
planner reservation, not a measured allocation category.

The baseline's GPU hit window includes dispatch gaps, kernels, and result D2H;
it is not isolated GPU compute time. Router rendezvous includes pending compute
on the stream and route/input transfers; it is not pure router time. A mixed-hit
layer downloads the entire `[T][10][2560]` FP32 output buffer. Accounting only
hit slots would understate baseline traffic. CPU routes include routes that
bypass cache execution under the existing precision diagnostics.

Set `NINFER_FLASH_NEXT_STAGE_LEDGER=1` separately to expose the existing prefill
stage ledger in JSON as well as its current human-readable output. Stage intervals
include inter-stage stream gaps, and its completion synchronization is existing
opt-in diagnostic behavior. Neither stage intervals nor host branch times should
be summed into an end-to-end throughput estimate.

PLE records report successful host gather wall time and output payload bytes;
page-fault/read time is unavailable and explicitly null. Decode gather includes
host dequantization, while compressed prefill gather does not.

Generate a diagnostic report from an explicitly bounded run log:

```bash
python3 tools/diagnostics/v100_sv0_report.py run.log \
  --candidate-sha "$(git rev-parse HEAD)" \
  --provenance provenance.json --output sv0-report.json
```

`provenance.json` should identify the build command, CUDA/driver, GPU clocks and
thermal state, NUMA placement, CPU workers, memory pressure, model/artifact hash,
runtime flags, and cache/profile state. Run instrumented attribution separately
from throughput measurements. Logs aggregate by layer and prefill/decode phase;
every eager execution record carries an executor/transaction context and
lane/epoch/column/token-index mapping. These are lane incarnations and execution
transactions, not external HTTP request IDs. Direct operator tests have null
context, and graph replays do not emit BF16 dispatch records. Reports preserve unavailable
metrics and always mark qualification pending.

## Projection and memory attribution

BF16 dispatch collects reusable CUDA event pairs per executor, classified by
implementation and `(N,K,T)`. Event collection skips graph capture, and event
results are read only after all round work has been submitted. Prefill attribution
therefore waits for measured projections before returning its pending round;
normal telemetry-disabled execution keeps its asynchronous return behavior.
These records cover the BF16 linear dispatcher, not every fused BF16 operation.
The existing stage ledger remains the source for QSA prefill attention intervals.

Memory records expose the artifact device arena, extra embedding/output-head,
MTP expert/proposal buffers, runtime persistent/workspace allocations, KV/state,
round tensors, sampling workspace, and graph allowance. Artifact arena bytes may
include dense MTP and vision tensors. Planner categories and aggregate allocations
must not be summed together. Graph allowance is a reservation, not an observed
CUDA allocation; diagnostic event storage is not included. Expert cache bytes
remain reported separately, after the cache has actually been configured.

## Hardware gate

`.github/workflows/v100-sv0.yml` triggers on this performance branch. It first
builds SM70 instrumentation and correctness targets on a CUDA 12.8 hosted runner.
Then the self-hosted V100 runner builds both the fixed baseline and candidate,
runs existing operator/executor tests with diagnostics off and on, compares
64-position routing/input traces exactly, and executes the unchanged full-manifest
Phase 11 oracle gate on both builds. A busy GPU causes a clear failure;
the workflow never stops another workload. Failed gates prevent later steps.

Operator/executor fixtures run with their default precision profile. The strict
whole-model oracle and throughput steps retain the frozen Phase 17 FP32 profile;
that profile requires host experts absent from some synthetic executor fixtures.
The first hardware run stopped at this fixture/profile mismatch. The corrected
run passed those stages, then exposed an existing executor-test assumption that
whole-model graphs are enabled. The frozen Volta runtime explicitly disables
these graphs. The shared test harness now verifies the disabled capture state and
retains bit-exact eager-fallback tokens/logits, workspace bounds, and lane churn.
Graph topology/footprint/timing tests run only on builds supporting whole-model
graphs. The workflow overlays only this same test source onto the frozen baseline,
records the overlay and matching harness hashes, and leaves its engine sources
unchanged. Neither run reached candidate GPU tests or oracle/throughput results;
qualification remains pending the corrected run.

The next run passed all baseline/candidate correctness tests and the telemetry-on
suite. Its frozen baseline then reached and failed the unchanged 64-position
Phase 11 numerical gate (mean KL `0.004876`, top-1 agreement `0.96875`). The SV0
workflow therefore records oracle exit status instead of aborting immediately,
requires candidate and telemetry routing/input traces to match the baseline
exactly, compares selected full-manifest metrics exactly, and runs the disabled-
instrumentation throughput screen. At that point the final step still required both unchanged oracle invocations
to pass. The subsequent user acceptance replaces this SV0 condition with exact
frozen-baseline parity and checks that all 4,096 positions remain finite. Oracle
qualification remains a separate reported result.

Self-hosted checkouts retain build caches, so each run now clears only its exact
`sv0-results` contents before recording evidence. The shared executor test also
drains any CUDA work submitted before its intentional invalid-model exception
before directly overwriting recurrent state for the subsequent lane-reuse test.
This removes an ordering race in the test without changing runtime behavior.
Baseline, candidate, and telemetry test logs are all refreshed even if one suite
fails, preventing an earlier run's log from being mistaken for current evidence.
The reuse assertion dirties and reads both exact PLE state slots owned by the
released lane on the executor stream, and verifies that the allocator returns
that lane. It no longer relies on default-stream ordering or an implicit first-
slot assumption.

The first full-manifest evidence attempt incorrectly passed `4096` through the
smoke-count override, whose valid range ends at `4095`; the executable rejected
the request before model load. The workflow now leaves that override unset, as
required to select the unchanged 4,096-position manifest. Metric extraction is
also non-short-circuiting so both baseline and candidate logs are preserved if a
future executable exits before emitting the selected summary fields.

The separate throughput screen performs two process-level ABBA cycles for cache
off and cache on, with telemetry/cache timing/stage ledger disabled. It reports
process medians and complete ranges, checks equal cache capacity, and rejects
missing thermal evidence or observed thermal throttling. The existing harness
uses short teacher-forced prefill/decode and includes logit downloads; it is not
production request throughput. Filesystem/PLE access is warm in this screen.
Full cold/warm, context-length, continuation, and MTP request benchmarks remain
required before optimization promotion. Reports always distinguish these limits.

Run [37185578367](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37185578367)
completed the SV0 evidence path at `9a21095d`. Baseline, candidate, and
telemetry-enabled correctness suites passed; the 64-position routing/input traces
and exit statuses matched exactly. The frozen baseline and candidate also emitted
identical full-manifest metrics over 4,096 positions: mean KL `0.10880759`, P99 KL
`1.88303806`, top-1 agreement `0.91357422`, and first divergence at position 13.
Both returned failure from the unchanged oracle gate.

With all instrumentation disabled, the candidate median changes relative to the
frozen baseline ranged from `-0.28%` to `+3.30%` across decode/prefill with cache
off, overlap, and serial schedules. This is consistent with ordinary screening
variance and provides no evidence of disabled-telemetry overhead. It also does
not establish a production throughput improvement.

SV0 is therefore behaviorally parity-validated but not specification-qualified:
the required unchanged Phase 11 oracle already fails on the frozen source and
fails identically on the candidate. The user subsequently accepted exact
baseline parity for SV0 and authorized SV1.

Host-only checks pass locally. CUDA builds and hardware correctness gates must
pass before promotion of SV1 policies; no performance improvement is claimed
until request-level evidence supports it.

## Subsequent order

SV1 profile-seeded/adaptive residency; SV2 device route merge; SV3 prefill
streaming; SV4 grouped AVX2 misses; SV5 early route handoff; SV6 PLE queued I/O;
SV7 Volta tensor-core projections/attention; SV8 optional KV tiering only after
SV1–SV7 measurements. Every feature remains switchable until its numerical,
resource, and request-level acceptance requirements are met.

## SV3 measured streaming boundary and containment

The final isolated route-cost campaign completed on V100 in run 37424809406.
With one canonical represented expert already warm in host memory, grouped GPU
staging had separate timing ranges from the 32-worker CPU expert path at two or
more routes. This is an isolated synthetic result: it excludes whole-model
overlap, routing distribution, persistent hits, and request scheduling, and does
not select an automatic route threshold.

Repeated whole-model cache-off measurements favored CPU execution at 128 and
256 prefill tokens and streaming at 512 through 2048. With the fixed identity-
bound 64-slot profile, 512-token ranges overlapped or favored CPU while streaming
had separate faster ranges at 1024 and 2048. These results support experimental
cache-off and cached token thresholds of 512 and 1024 respectively for the
measured workload only.

The strict production response comparison still diverges between CPU and stream
paths after the same 1,227-token prompt. Bounded same-input samples found finite
outputs and bit-exact isolated GPU replay, within the existing expert numerical
criterion, but do not explain or eliminate the autoregressive divergence.
Streaming therefore remains opt-in, `cpu-cache` remains the default, and no
general automatic threshold or production gain is claimed.

## SV4 grouped CPU expert primitive

SV4 begins with an independently selectable grouped AVX2/FMA kernel for widths
2, 3, and 4. Each packed NVFP4 K16 weight block and scale is decoded once and
applied to independent token accumulators. Every token retains the single-token
FMA, horizontal reduction, and BF16 intermediate order. Width one continues to
use the existing kernel. The initial hardware gate requires bit-exact grouped
versus single-token output for random represented BF16 activations including
finite extremes before the grouped task is integrated into runtime dispatch.

Run 37433116151 passed that primitive gate on V100. Widths 2, 3, and 4 were
bit-exact with the single-token AVX2 kernel. Their grouped times were 1,830.85,
2,298.55, and 2,670.56 microseconds, versus 2,685.92, 4,000.20, and 5,318.56
microseconds for independent execution. These are isolated primitive timings,
not whole-model or production throughput.

The next opt-in integration groups same-expert CPU misses inside the existing
persistent worker pool. Groups retain original route/output slots; widths 2--4
share packed-weight decode, while singleton remainders use the unchanged
single-token kernel. `NINFER_V100_CPU_EXPERT_GROUP=1` enables grouping only for
multi-token Volta execution with the production BF16-input AVX2 profile. Unset
or `0` preserves existing behavior. Telemetry reports grouped task count,
grouped route count, and effective compact weight bytes read. The SV4 workflow
builds the primitive and represented full-model executable in one CMake build
invocation, then requires exact final BF16 logits and route provenance across
single and grouped arms before reporting a screening result.

## SV1 profile preparation

`tools/diagnostics/v100_expert_profile.py` builds an offline static residency
proposal from validated SV0 JSONL. Supply separate training and evaluation logs,
the actual artifact model/weights identity, its SHA-256 provenance, and a JSON array of 48 slot capacities from the
intended cache allocation. Ranking uses descending routing count with expert
index as the deterministic tie-breaker. Decode, prefill, or combined observations
can be selected explicitly. Identical trace contents are rejected.

```bash
python3 tools/diagnostics/v100_expert_profile.py \
  --train-log train.log --evaluation-log held-out.log \
  --artifact-sha256 "$ARTIFACT_SHA256" --slots layer-slots.json \
  --model-id "$MODEL_ID" --weights-id "$WEIGHTS_ID" \
  --phase decode --profile-output expert-profile.json \
  --evaluation-output profile-coverage.json
```

The profile includes geometry, artifact identity, complete per-layer ranking,
counts, and training trace identity. Outputs are written through `.tmp` and rename.
Coverage is a static projection, not a simulation of dynamic fills or leases;
the runtime optionally consumes schema v2 for startup seeding. No training-only or synthetic coverage result
qualifies a residency policy.


## SV1 adaptive replacement implementation

`NINFER_V100_EXPERT_POLICY=lru|heat|decay|static|profile` selects replacement; `lru` remains
the default. Heat policies accumulate selected route IDs once per layer call,
including hits, before background admission. Victims remain restricted to their
layer and uploads/leases remain protected. A heat-policy admission must be hotter
than its victim; equal heat retains the resident and reduces churn. Equal-score
victims use the existing access epoch as a tie-breaker.

Decay experiments require explicit `NINFER_V100_EXPERT_DECAY` in `(0,1]` and
`NINFER_V100_EXPERT_DECAY_INTERVAL`, measured in calls to each layer. Each interval
applies `heat = decay * old_heat + interval_count`; pending counts participate in
victim scores. No decay coefficient is promoted by default. Cache reset clears
heat as well as residency. LRU allocates no heat table and performs no heat update.

Adaptive replacement passed V100 correctness and default-policy parity in run
37203161634. Held-out request performance qualification remains SV1 work. No adaptive policy is qualified or enabled by default.

### SV1 startup profile seeding

`NINFER_V100_EXPERT_PROFILE=/path/profile.json` optionally seeds each layer's allocated
slots with its highest-ranked experts. Profile schema v2 binds `model_id` and `weights_id`
to the actual artifact Reader identity. The offline builder now requires those metadata
values via `--model-id` and `--weights-id`; its `--artifact-sha256` remains provenance,
not a runtime hash verification claim. Schema, identity, and all 48 complete expert
permutations are validated before fills. Startup uses the existing canonical packing,
H2D stream, and Ready publication path, with at most four outstanding jobs and a drain
before the runtime exposes the cache. Seeding reports resident count and startup time.

Default behavior stays LRU with no profile. Seeding can accompany opt-in heat or decay;
seeding supplies the original ranking for profile-prior replacement and learned persistence.
Host parser checks and hardware ranked-seed/Ready/numerical checks cover this path.
Held-out real-model policy throughput and resource qualification remain pending.

### SV1 learned profiles and prior replacement

`static` keeps the seeded Ready set fixed and restores it after a diagnostic reset. `profile` combines measured cumulative
heat with `NINFER_V100_EXPERT_PRIOR_WEIGHT * (512 - rank) / 512`; the positive, finite
weight must be chosen explicitly. Both require a startup profile in the runtime.
Uploads and leases remain protected, victims remain within their layer, and equal
scores retain residents. No coefficient or policy is enabled by default.

`NINFER_V100_EXPERT_PROFILE_SAVE=/path/learned.json` enables heat collection and
atomic profile saving on clean cache destruction after successful runtime startup. Explicit inference-owner calls
to `save_profile` also save at a drained boundary when heat collection is enabled.
Learned ordering puts current Ready residents first, then measured heat, original
profile ranking, and expert index. Saved frequencies exclude the artificial prior.
The write closes `<file>.tmp` successfully before rename. Explicit saves throw on
failure; shutdown saves log `v100.profile.save_failed` and preserve the old file.
With LRU plus saving, heat observation does not change replacement behavior.

Startup seeding passed V100 run 37209468461 (artifact 11307454288), including
canonical output checks and default-policy trace/full-metric parity. Static/prior
and learned persistence passed V100 run 37215774635 (artifact 11310560177):
6/6 candidate correctness tests, 5/5 telemetry checks, and default-policy exact
trace/full-metric parity. Held-out performance qualification remains pending. The independent Phase11
oracle remains a separate, unchanged numerical failure accepted by the user as
the frozen baseline.

### SV1 real-artifact held-out screen

`.github/workflows/v100-sv1.yml` runs the actual artifact executor with separate
384-token corpus segments: source positions 0–383 for training and 2048–2431 for
held-out evaluation. Each segment starts with fresh recurrent state and uses
128 prefill plus 256 teacher-forced decode tokens. Original oracle logits are
not reused for these changed prefixes. The Phase 11 gate is unchanged.

`tools/diagnostics/v100_sv1_residency.py` trains with real routing, verifies the
clean-shutdown profile's measured frequencies against all 48 layers of telemetry,
and reloads the runtime-produced profile with static residency before inference.
A separate frequency-ranked training profile seeds static/heat/decay/profile arms.
The evaluation-only route histogram never changes the supplied ranking.
Cache-off and empty dynamic LRU are controls. The planner's training capacity is
fixed for all cached evaluation arms; cache/transfer/runtime allocation categories
must match. Actual CUDA-used memory and startup latency are recorded separately.

Each policy receives one diagnostic pass and three fresh-process timing passes
in alternating order. Timing disables telemetry, cache events, and stage ledgers.
Diagnostics preserve hits by layer/phase, CPU miss branch wall time, fills,
evictions, queue bounds, and complete route counts. Static actual hit counts must
match projection onto that arm's actual routes. A separate CPU-only projection
is retained because CPU/GPU arithmetic can change downstream routing. Decay 0.9
at 32 layer calls and prior weight 32 are explicit screening coefficients only.

The report keeps the final BF16 logits and per-position decode top-1 results for
supplementary pairwise screening, checks finite outputs and the committed state
frontier, and samples GPU thermal status. It never promotes a policy. This
teacher-forced executor screen includes logits downloads and finite scans, uses
warm PLE/filesystem access, and does not replace independent numerical
qualification or production request/MTP/continuation/long-context measurements.

Run 37226714270 (2026-10-04, artifact 11313372633) passed this screen on V100
SM70/CUDA 12.8. All cached policies used 165 slots per layer, 21,899,243,520 cache
bytes, and identical CUDA-used memory (29,982,851,072 bytes). Learned profile
save/reload, complete routing, finite outputs, committed frontier, and thermal
checks passed. Three telemetry-disabled process observations gave:

| Policy | Decode t/s median | Decode hit rate | Diagnostic CPU miss time |
|---|---:|---:|---:|
| Cache off | 6.926 | 0% | 25.527 s |
| LRU | 10.756 | 67.47% | 12.659 s |
| Static | 10.953 | 69.03% | 12.080 s |
| Heat | 12.158 | 83.02% | 9.036 s |
| Decay | 12.275 | 84.03% | 8.682 s |
| Profile prior | 11.640 | 77.12% | 10.234 s |

Decay's median decode rate was 14.1% above LRU with nonoverlapping observed
ranges, and its diagnostic CPU miss branch was 31.4% shorter. Seeded startup
took about 7.5–7.8 seconds. These are workload-specific screening results.
Pairwise outputs differ: decode top-1 agreement versus cache-off was 98.05%
for LRU, 97.66% for heat, and 94.14% for decay; final-position KL was
0.001926, 0.001787, and 0.006228 respectively. Each arm reproduced its own
outputs across diagnostic/timing observations. No policy is promoted.

The workflow next collects all six policies' numerical metrics against the
unchanged full 4096-position Phase11 manifest. `--oracle-only` consumes the
current screen's identity-bound training profile and capacity, records exact
oracle exit status, and requires complete finite metrics and route coverage.
A known numerical gate failure remains explicitly reported as failed; an early
execution failure is rejected. This evidence step does not alter thresholds or
declare independent qualification. Full-manifest numerical evaluation is
separate from the disjoint held-out performance screen above.

Run 37229616603 (2026-10-04, artifact 11316495073) completed all six full
4096-position independent-oracle evaluations with zero nonfinite positions and
complete routed-layer coverage. All still failed the unchanged numerical gate:

| Policy | Mean KL | P99 KL | Top-1 agreement |
|---|---:|---:|---:|
| Cache off | 0.10880759 | 1.88303806 | 91.3574% |
| LRU | 0.09505473 | 2.09959815 | 92.9199% |
| Static | 0.12204488 | 1.97835612 | 89.8682% |
| Heat | 0.12541878 | 2.35374574 | 89.9902% |
| Decay | 0.11572333 | 2.00828330 | 91.7236% |
| Profile prior | 0.10245882 | 1.93551218 | 91.5283% |

Cache-off reproduced the accepted frozen-baseline metrics. Relative numerical
results are mixed; they do not establish independent qualification or justify a
new tolerance. The repeated held-out screen measured decay at 12.410 t/s versus
LRU at 10.793 t/s, with equal cached capacity. SV1 implementation and required
held-out hit/miss/throughput/fill/eviction/resource evidence are available; LRU
remains default and production/MTP/continuation qualification is pending.

### SV2 device route reduction

`ops::expert_route_combine` consumes FP32 `[T][10][2560]` route outputs and
FP32 `[T][10]` weights. It executes ten ascending-path round-to-nearest FMAs,
starting at +0, and supports a pitched destination that preserves the shared
expert's existing slab. This is an allocation-free, stream-ordered Op supporting
SM70. The independent naive FP64 sum is checked with the standard gamma-10 FP32
roundoff bound; exact host FMA bits are supplementary ordering evidence. Tests
cover T=1–5, 128, 256, and 4096, 0/50/100% route provenance, output guards,
shared-slab preservation, and compact output.

The opt-in `NINFER_V100_DEVICE_ROUTE_COMBINE=1` runtime path now gives the MoE
workspace one FP32 device route matrix. Scalar/grouped/batched cache hits write
directly to their route rows; CPU misses write reusable pinned host storage and
only contiguous miss runs cross H2D. Deferred hit work completes before leases
are released, and the device reduction writes the routed sum into the existing
pitched activation slab before the unchanged shared/final merge. The legacy
host reduction remains the default.

`.github/workflows/v100-sv2.yml` checks the standalone operator, direct cache
output and lease release, then compares cache-off and mixed-LRU legacy/device
execution on a disjoint real-model workload. It requires exact held-out final
BF16 logits and decode top-1, zero hit-result D2H, zero full routed-sum H2D, and
exact miss-only H2D byte accounting. Run `37240970132` on CUDA12.8/SM70 passed
on 2026-10-05 UTC. Three fresh-process timing observations, alternating order,
on the 128-prefill/256-decode held-out workload gave:

| Cache | Legacy decode t/s median (range) | Device decode t/s median (range) |
|---|---:|---:|
| Off | 6.870 (6.833–6.897) | 7.650 (7.630–7.707) |
| LRU | 10.602 (10.394–10.701) | 12.041 (11.795–12.068) |

Final BF16 logits and all decode top-1 decisions were exact within each cache
arm, including all timing observations. Both LRU diagnostics used 165 slots per
layer, 21,899,243,520 cache bytes and 29,982,851,072 observed CUDA-used bytes;
both counted 82,905 hits and 101,415 misses over 184,320 routes.

| LRU diagnostic payload | Legacy bytes | Device bytes |
|---|---:|---:|
| Hit-result D2H | 1,229,619,200 | 0 |
| Routed-sum H2D | 188,743,680 | 0 |
| CPU-miss route H2D | 0 | 1,038,489,600 |

Cache-off device execution uploaded 1,887,436,800 miss-route bytes, ten times
the legacy combined-sum payload; despite this, this screen measured faster
decode. These results establish the held-out transfer/arithmetic screen, not
production performance or default promotion. Artifact `11319068181` preserves
the operator/cache checks, reports, finite/state checks and thermal samples.
Run `37248276974` then compared the legacy and device paths over the unchanged
4096-position manifest. Cache-off and LRU each had exact selected metrics and
the same exit status between paths. Cache-off reproduced mean KL 0.10880759,
P99 KL 1.88303806 and 91.3574% top-1 agreement; LRU reproduced mean KL
0.09505473, P99 KL 2.09959815 and 92.9199% top-1 agreement. The independent
oracle gate remained false in every arm, as it was for the accepted baseline;
no threshold was changed. Artifact `11321129468` preserves this full-prefix
evidence.

The focused cache test also drives the actual direct-output cache and device
combine together at 0%, 50% and 100% hit layouts for T=1–5 and the current
2048-token MoE prefill test extent. CPU-miss rows use the independent CPU expert
reference and cross H2D only in contiguous missing-route runs; every case
reports zero hit-result D2H and zero routed-sum H2D. It additionally verifies
that owner-side exception cleanup completes submitted hit work, releases the
lease and makes a single cache slot reusable. MTP, continuation and
production-server qualification remain pending. The runtime path remains
opt-in.

Run `37254910291` passed the expanded focused cache coverage, including all
2048-token 0/50/100% layouts and the exception/lease cleanup case. Its held-out
screen then exposed a qualification-harness error: detailed telemetry delayed
asynchronous LRU filling enough to change one hit/miss boundary, after which the
CPU-versus-GPU expert arithmetic changed later routing. All three telemetry-off
legacy/device observations remained bitwise identical. The harness therefore
uses telemetry only for route/transfer attribution, requires cache-off
telemetry to remain exact, and accepts a telemetry-on LRU output difference
only when the recorded route/cache provenance also differs. Equal provenance
still requires bitwise output and top-1 identity; all telemetry-off timing
observations remain exact arithmetic controls. No numerical tolerance or
independent-oracle threshold was changed.

Run `37258860325` passed compilation, every focused integration case, and the
held-out screen. Cache-off full-prefix legacy/device metrics were again exact.
Its adaptive-LRU full-prefix processes independently crossed different
asynchronous fill boundaries, so their mean KL and other trajectory metrics
were not an arithmetic control. Full-prefix cached parity now uses one learned,
artifact-identity-bound profile from the held-out LRU diagnostic and the
`static` policy in both processes. That holds the resident set fixed while
still exercising mixed CPU/GPU expert arithmetic. The rerun requires exact
selected metrics and status with the independent oracle thresholds unchanged;
adaptive LRU remains the separate throughput screen.

Run `37270512213` completed all SV2 checks on 2026-10-05 (artifact
`11331205267`). The identity-bound static full-prefix control passed exact
legacy/device selected metrics and status over 4096 positions: mean KL
`0.11192840`, P99 `2.26605414`, top-1 `0.91625977`. Cache-off also stayed exact
at the accepted frozen-baseline metrics. Both independent numerical gates
remain false, with thresholds unchanged. Focused 0/50/100% integrated cache
coverage and all held-out telemetry-off output controls passed. The repeated
LRU executor screen measured legacy `10.421` versus device `11.944` median
decode t/s, with ranges `10.366–10.506` and `11.874–12.025`. This supports the
bounded screen only; the device path remains opt-in.

### SV2 production HTTP prefix and MTP screen

`.github/workflows/v100-sv2-serve.yml` builds the actual `ninfer-serve` binary,
consumes the fixed identity-bound profile from the explicitly named successful
SV2 artifact, and runs `tools/diagnostics/v100_sv2_serve.py`. It compares legacy
and device route modes with no speculation and with MTP draft window 3. Each
arm receives three fresh server processes in alternating order. The workload
includes a cold two-turn prompt, read-only prefix replay, a follow-up using the
assistant output, and read-only continuation replay.

The screen requires equal fixed cache allocation (64 slots per layer), complete
startup seeding, exact greedy response/finish/output-token signatures between
legacy and device modes, matching HTTP/Engine token accounting, and positive
prefix reuse on each replay/follow-up. MTP arms must record actual drafted
tokens rather than merely load MTP weights. Request JSONL retains acceptance,
backend, prefix path, TTFT, prefill, decode and Engine timings. Diagnostics are
disabled for throughput; thermal status is sampled separately. Servers bind
loopback only and cleanup terminates only the subprocess started by the test.

This is a short production-path screen using one active request, BF16 KV,
4096 context, and bounded cache capacity. It does not replace the broader
concurrency/cancellation/long-context/vision matrix, independent numerical
qualification, or a default-promotion decision. Hardware results are pending.

Run `37282415209` compiled both server builds but stopped before inference:
the public Engine requires a `.ninfer` suffix, whereas the validated runner
mount `/models/qwen3-8b-flash-next` has none. The workflow now checks that exact
mount is a readable file and gives it an explicit `model.ninfer` symlink in the
source directory. The harness accepts this alias explicitly without resolving
away its suffix. Readiness polling also retries connection resets while checking
for server exit on each iteration. No inference or performance result was
produced by the failed run.

Run `37317924275` reached the public Engine and completed four HTTP requests.
Cold/replay and follow-up/replay greedy outputs matched, and replays reused
308/393 tokens, but the intervening follow-up fell back to root. Flash-Next
ignored `allow_prefix_publication` in its base plan and consumed the original
anonymous owner during the read-only replay, replacing its endpoint without
retaining the prior turn-closure checkpoint. The planner now suppresses
publication/write opportunities for read-only requests and retains their
selected private source. The same production regression check still requires
positive reuse on the follow-up; no output, reuse, or MTP gate was relaxed.
Hardware validation of this lifecycle fix is pending.

Run `37321786662` passed the production screen at `81a61b9b`. All 12 fresh
processes and 48 HTTP responses passed exact within-arm legacy/device signatures,
HTTP/Engine accounting, positive prefix/follow-up reuse, complete 3072-expert
seeding, equal 8,494,252,032-byte cache allocation, and thermal checks. Each MTP
process drafted 298 tokens and accepted 148 across its four requests. Non-MTP
cold decode medians were 8.581/9.353 t/s (legacy/device); continuation medians
8.482/8.871. MTP cold medians were 7.677/8.405; continuation 7.184/8.173.
These are bounded production-path measurements; MTP itself was slower on this
short workload, and no default or independent oracle qualification changed.
Together with the focused route/lease tests and fixed-profile full-prefix
parity, this satisfies the SV2 opt-in implementation gate for building SV3 on
its device route layout. Broader production qualification remains separate.

### SV3 bounded expert staging primitive

`FlashNextExpertStream` explicitly owns four canonical device expert slots and
four pinned slots, a nonblocking transfer stream, bounded activation and grouped
launch descriptors, and ready/consumed events. It uploads each selected distinct
expert once, submits groups of up to four routes through the existing canonical
expert kernels, and writes caller-owned device route destinations. Slot reuse
waits for the prior consumer; packing/upload of later slots can overlap compute.
No persistent cache is referenced or evicted. Device/pinned capacity and expert
transfer bytes are reported. The caller must retain inputs/outputs until finish.

Run `37333323795` passed SM70 compilation and the V100 primitive test across
multiple ring wraps, 1–5 and 8–2048 route sizes, rejected-group cleanup,
destination guards, and the unchanged independent CPU expert mathematical
criterion. Worst observed NRMSE was `3.86488e-7` with cosine `1.0`.

The ring is now Program-owned and included in the runtime capacity curve when
`NINFER_V100_PREFILL_EXPERT_POLICY=stream|auto`; its separately allocated bytes
are subtracted from the main persistent arena and checked against the plan.
`stream` uses it for every prefill chunk. `auto` requires an explicit measured
`NINFER_V100_PREFILL_STREAM_MIN_TOKENS` threshold. Both require the already
qualified device route combine. Streamed prefill uploads each distinct selected
nonresident expert once, writes the SV2 route matrix directly, and copies only
route IDs to the host. Ready persistent hits use the existing grouped cache
kernels and write that same matrix without restaging. Streamed prefill does not
admit or evict persistent residents, and completion releases all hit leases.
Decode remains
unchanged. Telemetry separately records streamed routes, distinct experts, and
expert H2D bytes.

Run `37345994876` passed the real-model integration on all 48 layers at a
bounded 32-token prefill: all 15,360 routes used the staging ring, CPU/cache
expert work was zero, and 15,655,747,072 expert H2D bytes exactly matched the
5,662 distinct layer-local experts. Run `37345994709` simultaneously passed the
complete SV2 regression matrix.

The workflow next measures fresh-process `cpu-cache` and `stream` arms at 128,
256, 512, 1024, and 2048 tokens, with three alternating timing observations per
cell. A separate telemetry observation checks exact route and transfer bytes,
thermal status, repeated numerical output, and equal KV/state/workspace
capacity; the staging-ring reservation is reported separately. Persistent
expert caching is disabled in both calibration arms so this isolates the
CPU-versus-staged-GPU miss decision. The collector reports evidence without
changing defaults.

Calibration run `37365646019` passed compile, ring tests, and real integration,
then exposed a harness allocation mismatch: the oracle test allocated only a
128-token prefill chunk and rejected the 256-token probe before inference.
Prefill probes now allocate the supported 2048-token chunk consistently for
all calibration sizes and both arms; ordinary full-oracle runs retain their
128-token allocation. The partial diagnostics did not establish a crossover.

Corrected run `37368840887` passed the complete calibration with three fresh
processes per cell, finite outputs, exact repeated numerical metrics within
each arm, equal final top-1 across arms, thermal checks, equal logical runtime
capacity, and exact transfer accounting. Artifact `11370913476` preserves the
observations. Median prefill throughput on V100 was:

| Tokens | CPU t/s | Stream t/s | Stream change |
|---:|---:|---:|---:|
| 128 | 27.195 | 17.887 | -34.23% |
| 256 | 33.938 | 29.752 | -12.33% |
| 512 | 39.706 | 48.087 | +21.11% |
| 1024 | 44.152 | 71.240 | +61.35% |
| 2048 | 46.933 | 107.275 | +128.57% |

Ranges were separate at every size. `512` is an evidence-backed experimental
token threshold for this cache-off workload. It is not a qualified default or
a calibrated per-expert reuse policy. Resident-hit coexistence is now
implemented and awaits V100 validation: focused 0/50/100% resident layouts,
independent expert/combine oracle checks, and a real identity-bound 64-slot
LRU profile probe that requires both hits and streamed misses while holding
admissions, fills, evictions, and leases unchanged. Cached throughput remains
to be measured before selecting a general automatic policy.

The real-model integration requires:
cache-off routes must all use the ring; cached routes must partition into
Ready hits and streamed misses with zero CPU expert work. Activation D2H is
forbidden, and exact expert transfer accounting is required. `auto` continues
to require an explicitly supplied threshold; defaults remain unchanged.

SV3 resident coexistence validation run `37391259007` passed SM70 compilation,
all independent 0/50/100% resident operator cases at T=1..5,128,2048,
and the real-model two-pass residency/accounting check with exact repeated
selected numerical metrics against all-stream. The repeated cache-off calibration
measured CPU/stream medians 27.446/18.223,34.077/29.320,40.600/48.329,
43.897/71.497,46.732/107.476 t/s at 128,256,512,1024,2048 tokens.
This supports the experimental cache-off crossover of 512; defaults stay unchanged.

The next calibration uses the same identity-bound profile and fixed static
64-slot resident set in both arms at 512,1024,2048 tokens. Three fresh processes
per arm measure the first prefill; a second pass checks exact numerical and route
repeatability. Telemetry separately verifies no resident admissions/evictions,
zero leases after completion, mixed hit/miss partition, miss-only staging/CPU
transfers, equal cache capacity and separately planned ring overhead.
The existing independent oracle thresholds remain unchanged; this calibration
does not claim independent-oracle or production/default qualification.

Fixed-cache SV3 calibration run `37397409904` passed all hardware steps,
including three fresh processes per arm and stable 64-slot identity-bound
resident sets. CPU/stream median prefill t/s at 512,1024,2048 were
50.652/50.423,58.857/72.267,66.320/94.955. The 512-token ranges overlap;
1024 and 2048 favor streaming with separate ranges (+22.78%,+43.18%).
This suggests an experimental cached token crossover of 1024 for this workload,
not a universal per-expert cost policy or default promotion.

`v100-sv3-serve.yml` uses the shared production HTTP driver with
`--prefill-screen` to compare cpu-cache and stream with device route combine
in both arms, a fixed 64-slot profile, and an actual 1024..2048-token cold
prompt in one supported 2048-token chunk. It retains the three fresh processes,
prefix/continuation replay, MTP drafting, exact greedy response accounting,
equal cache capacity and thermal checks. It records cold prefill throughput
and decode throughput separately. The existing SV2 production regression
continues to exercise legacy/device modes through the same driver.

Production large-prefill run `37402261711` reached eight HTTP requests across
the first cpu-cache and stream processes. Both passed prefix reuse and exact
within-process replay, but their greedy responses differed after the same
1227-token cold prompt. The strict cross-path response gate failed; this run
does not qualify production streaming or establish a production speedup.
The staging and cached-calibration regression `37402261650` passed; the SV2
production regression hardware job was canceled and supplies no new evidence.

The active SV0–SV3 workflows now run lightweight Python checks on the hosted
runner and compile CUDA targets only on the V100 runner. SV0 also runs its
small independent host C++ checks without configuring a CUDA build. This
removes the previous hosted/self-hosted duplicate candidate compilation and
retains the runner's incremental build directories. SV0's frozen baseline and
candidate remain distinct necessary builds. All CMake builds retain the
user's 32-job environment setting.


Current-head SV2 production regression `37407560849` passed all 12 fresh
servers/48 HTTP requests and the strict replay/cross-path response gates after
the single-build workflow change. Non-MTP cold legacy/device decode medians
were 8.379/9.139 t/s; cold prefill medians were 30.880/44.690 t/s. These remain
scoped fixed-cache, short-context measurements, without default promotion.

The unresolved SV3 production response difference is now investigated with a
bounded same-input diagnostic (`v100-sv3-compare.yml`). With
`NINFER_V100_STREAM_COMPARE=1`, large streamed prefill samples eight
nonresident expert IDs spread across each layer's active list and early/late
routes within those experts. It reads the original GPU route output, replays
the same represented activation and weights through an isolated one-route
staging ring, and computes scalar/AVX2 CPU precision-profile controls. It
reports exact GPU replay differences, finite values, NRMSE/cosine and maximum
error. It never replaces an output or changes routing; diagnostic timing is
not throughput evidence. The existing focused CPU-profile criterion is
reported without relaxation, and remains supplementary to independent operator
and Phase11 mathematical-oracle checks. Eight samples per layer cannot exclude
unsampled corruption. No cause or production streaming qualification is claimed
before the hardware evidence is available. The failed response gate remains intact.

Same-input diagnostic run `37415223669` passed all 384 bounded samples (eight
per routed layer). Isolated GPU staging replay was bit exact; all sampled
outputs were finite. Against the scalar CPU profile, the maximum NRMSE was
0.000599579, minimum cosine was 0.999999822, and maximum absolute error was
0.000051875. The AVX2-reference and GPU-AVX2 maximum NRMSE values were
0.000599575 and 0.000298541. No sample failed the existing 0.002/0.99999
focused criterion. This rules out sampled staging/replay instability, but it
does not prove why autoregressive production responses diverged and cannot
exclude an unsampled fault. The strict production response gate remains
failed and unchanged.

The next SV3 campaign measures the specification's bounded per-expert route
costs at 1,2,4,8,16,32,64 routes. It alternates seven warm samples per point
using the production 32-worker CPU expert path plus result H2D versus staging
host packing, one expert H2D and grouped GPU execution. The fixture is the
canonical synthetic represented expert used by the independent stream tests;
shared route/control transfers, layer overlap and whole-model effects are
outside this measurement. The collector may report a separate-range observed
crossover, but always leaves the runtime route threshold unset and
unqualified. It does not alter `auto`, the default, or any oracle threshold.

To avoid simultaneous pending hardware jobs replacing each other in the shared
V100 concurrency group, a push commit can select one hardware campaign with
`[v100:sv0]`, `[v100:sv1]`, `[v100:sv2]`, `[v100:sv2-serve]`,
`[v100:sv3]`, `[v100:sv3-serve]`, or `[v100:sv3-compare]` in its message.
Other triggered workflows still run host checks but skip hardware. Untagged
pushes and manual dispatches retain their prior behavior. Select one campaign
and inspect its evidence before launching the next; do not publish over healthy
active same-head work. CUDA builds retain their incremental directories and
32-job environment setting.
