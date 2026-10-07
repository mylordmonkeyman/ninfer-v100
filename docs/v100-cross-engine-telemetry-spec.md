# NInfer / Strata V100 telemetry v1

This is the shared instrumentation contract for the comparative telemetry phase.
It implements the attached 2026-10-07 handoff and supersedes ad hoc optimization
screens. Instrumentation must not change precision, routing, cache policy,
worker selection, graph use, or speculation settings.

## Controls and fidelity

Both engines accept `V100_COMPARE_TELEMETRY_LEVEL=0|1|2|3` at process startup.
Absent the variable, instrumentation is off. NInfer's existing
`NINFER_V100_TELEMETRY=1` retains its detailed legacy behavior when the new control
is absent. The campaign supplies `V100_COMPARE_RUN_ID` to label every process.

| Level | Measurement | Valid use |
|---|---|---|
| 0 | Existing request metrics only | Authoritative throughput |
| 1 | Owner-local counters and host durations, one serialization per round | Aggregate diagnostic; overhead must measure <2% |
| 2 | Layer/stage details and deferred CUDA event timing | Attribution; never headline throughput |
| 3 | Level 2 plus NVTX and Nsight Systems | Scheduling/overlap attribution |

Level 1 must not add CUDA allocation, event recording, synchronization, or switch
from graph replay to eager execution. Host spans measure elapsed host time,
including waits already present in execution. They do not measure GPU compute.
Level 2 CUDA events describe their specific stream interval, not necessarily GPU
busy time; no CPU/GPU sum may be called a critical path. NVTX GPU attribution must
account for overlapping intervals and graph replay. Unobserved operations during
graph replay remain unknown, never zero. Invalid level values fail startup.

## JSONL envelope

Schema: `ninfer-strata-v100-telemetry-v1`. Round records carry `schema`, `engine`,
`kind=round`, `level`, `run_id`, `round_id`, `phase`, `status`, `host_wall_us`,
`timing_semantics=host_observed_not_gpu_execution`, and `layers`.
`round_id` is process-local owner ID plus transaction/sequence. It is not a
request ID. A request may span many prefill chunks and speculative rounds; the
campaign must associate lane/epoch, request ID, rounds and exact token traces.
Only request records establish request throughput or accepted-token counts.
Failed/capture/warmup records cannot enter measured throughput summaries.

Each layer has `layer` (0..47), `counters` and `host_us`. Counter values are
nonnegative integers. Host durations are nonnegative finite microseconds.
Missing fields mean **unavailable**. An observed zero means no work took place.
Counters are deltas within the record except fields suffixed `_snapshot` or
`_total`; cumulative cache values must be differenced per owner/layer, including
reset boundaries, before summing. Durations are inclusive; nested stage times
must not be summed into a latency decomposition.

## Native input context and cache windows

Optional `context` describes native host-visible round input, not a request record.
`input_columns` and contiguous `spans` cover all submitted columns. Each span has
`first_column`, `columns` and `first_token_index`; NInfer also records its actual
`lane` and `epoch`, plus context `executor` and `transaction`. Strata positions
are available, but caller request identity remains unavailable at these boundaries.
`execution_mode` is `eager`, `cuda_graph` or `unknown` from the actual chosen path.
At levels 2/3, available `input_token_ids` and `sampled_token_ids` preserve column
order. They are absent at level 1. Sampled candidates are neither accepted drafts
nor emitted text; no acceptance/throughput denominator may be derived from them.
Strata's session loop receives device-owned embeddings, so its token IDs remain
unavailable; PLE's token field is not substituted for an unobserved ingress.
Teacher forcing, request IDs and the accepted/emitted trace still require caller
instrumentation. A run ID must identify a single process in a frozen campaign.

Optional layer `cache_windows` records before/after snapshots around each actual
NInfer host-routed layer call. Snapshot counters are **per layer**, unlike the old
SV0 global cache totals. `generation` increments on reset; never subtract totals
across different generations. `capacity_experts`/`capacity_bytes` are the layer's
allocated slots/payload capacity. `ready`, `uploading`, `leased` are slot counts.
`hits_total`, `misses_total`, `admissions_total`, `fills_total`, `evictions_total`
and `fill_bytes_total` are cumulative in that layer and generation. Fill bytes
are actual padded H2D payload bytes. Levels 2/3 include `resident_ids` for Ready
slots; level 1 does not allocate/serialize the IDs. Startup seeding is included
in cumulative totals, not counted as current-round admissions.

The preliminary report differences matched-generation snapshots only. These
are observed-window deltas: an asynchronous fill may complete for an earlier
round, and fills between layer-call windows remain unassigned. Reset windows
make aggregate deltas unknown. Do not treat window deltas as all cache traffic
or as a causal attribution to current-token requests. Snapshot reads take the
existing cache mutex; they do not drain fills or add CUDA work. Cache overhead,
source/page reads, adaptation timing and Strata dynamic-residency observations
still require qualification/instrumentation.

## Stage taxonomy

Common stage names are `request`, `tokenize`, `prefill`, `decode`, `layer`,
`normalization`, `mixer`, `qsa`, `gdn`, `ple`, `dense`, `router`, `moe`,
`shared_expert`, `route_combine`, `residual`, `head`, `sampling`, `cpu_expert`,
`gpu_resident`, `gpu_nonresident`, `expert_transfer`, `host_wait`,
`cache_admission`, `cache_adaptation`, `mtp_draft`, `verify`, `commit`.
Native substages retain their native name beneath these common names.

## Routed work and memory units

All sizes are bytes, times microseconds, throughput tokens/sec or routes/sec,
and bandwidth decimal GB/sec. Record actual payload traffic, not allocation
capacity or compressed file size substituted for bytes transferred.

| Counter | Definition |
|---|---|
| routed_tokens | Token columns submitted to routed MoE, including verified drafts |
| total_routes | All routed token/expert pairs |
| resident_routes | Pairs served from persistent GPU cache |
| cpu_routes | Pairs computed on CPU |
| nonresident_gpu_routes | Pairs computed using transient GPU experts |
| distinct_experts | Unique requested IDs within this layer call |
| resident_distinct_experts | Unique IDs served persistently within this layer call |
| distinct_missed_experts | Unique IDs not served persistently within this layer call |
| cpu_groups / cpu_grouped_routes | Multi-token CPU groups / their route count |
| cpu_weight_read_bytes | Logical packed weights traversed, not measured DRAM transactions |
| expert_h2d_bytes | Transient expert payload transferred |
| route_d2h_bytes | Activations/IDs/weights transferred for host dispatch |
| output_d2h_bytes / output_h2d_bytes | Actual routed output transfers |

The invariant is `total_routes = resident_routes + cpu_routes +
nonresident_gpu_routes`. A persistent cache miss may execute on CPU or transient
GPU. Distinct counts across repeated layer calls are a sum of per-call distinct
work, not the unique set over the request. Request-unique IDs require a bitset.

Required later cache fields include capacity/occupancy in experts and bytes,
resident IDs, admissions, evictions, swaps, adaptation duration, incoming weight
source, page-cache reads and wait duration. Worker fields include configured and
actual workers, affinity, tasks/shards by worker, active/idle time and gate/up,
intermediate/down costs. Speculation records include actual verify width,
proposed/accepted drafts, acceptance histogram, emitted tokens, rounds,
draft/verify/commit/head/sampling time and memory. Never derive acceptance from
verify token count alone. CUDA API/launch/transfer counts and busy/idle interval
unions require explicit measurement coverage.

## CPU worker observations

Observed layers may contain `workers`, keyed by engine/process-local `pool_id`,
`worker_id` and `role=worker|host`. `configured_workers` is the pool's setting,
not the number that claimed jobs; actual participants have a positive job count.
Idle configured workers in an observed batch are included with zero job counts.
Strata's extra host-drainer slot is separate from its configured worker count.
Neither worker IDs nor pool IDs are OS thread IDs or CPU affinity measurements.

`full_jobs`, `gate_up_jobs`, and `down_jobs` count actual scheduler claims: whole
expert/group jobs or gate/up and down row-shard jobs. They are not routed-token
counts, and different partition strategies produce different job counts. Repeated
layer calls accumulate these counts. Level 1 records no per-job clocks. Level 2
adds `full_us`, `gate_up_us`, and `down_us`, sums of elapsed host job intervals
including preemption. A zero phase value means no separate job of that kind was
claimed; a whole job's internal gate/up and down split remains unavailable.
Worker interval sums overlap and cannot establish CPU busy time or critical path.
Idle/wait durations, affinity, NUMA placement and intermediate quantization time
remain unavailable until separately measured.

Each worker writes a separate aligned slot before the existing completion signal;
the inference owner merges after the existing completion/park barrier. Telemetry
adds no locks or atomics to per-job accounting and does not change job allocation,
barriers, precision, worker count or scheduling. Failed dispatch observations are
not qualified by the success-only fixture checks. Real CPU fixtures check output
bits and job conservation with levels 0/1/2; full native-model numerical equality
and Level-1 overhead still require hardware qualification.

The preliminary comparison reports jobs and timed host activity per observed
`(run_id, pool_id, worker_id, role)`, with observation/timing coverage. Partial
coverage leaves summed activity unknown. It preserves idle configured workers
with zero jobs, but does not infer their idle duration or combine overlapping
thread activity into request latency.

## Native lifecycle records

Separate schema `ninfer-strata-v100-lifecycle-v1` records successful native
`verified`, `commit_returned`, `emitted` and `aborted` observations. It uses the
same `run_id`/`round_id` as execution records, plus `sequence_trace_id` and a
native token position. NInfer sequence correlation is executor/lane/epoch;
Strata caller loops allocate an explicitly process-local trace label. These are
**not external request IDs**. Strata's linked caller ID propagates through its
native verifier scope without changing nested scope ownership.

`proposed_drafts` is the actual draft count offered to this verification call,
including suffix/oracle proposals where configured, after native selection and
constraint trimming. It does not count all drafts generated and later discarded.
The input width is one anchor plus offered drafts, also observed in the linked
round context. `accepted_drafts` comes from the caller's actual comparisons;
`verified_tokens` counts resulting output candidates (accepted drafts plus one
target output), not all executed verification input columns. `proposal_source`
is `mtp`, `suffix`, `oracle`, `none` or `unknown`. The record's `token_count` and,
at levels 2/3, exact `token_ids` describe only its stated `token_semantics`.
Level 1 emits counts/correlation only.

NInfer Program records verified candidates and actual possibly truncated commit
outputs. Strata records the input-state prefix selected by `ver.commit(a+1)`
separately from actual output-loop tokens after EOS/max-new trimming. A native
commit return can precede CUDA completion: no event here claims GPU completion
and no additional wait is inserted. `first_token_index` is the native input
anchor for verified/Strata events and the first committed output position for
NInfer commit-return events. The linked execution context supplies input spans.

`tools/telemetry/lifecycle.py` validates and reports observed acceptance histograms,
input work, commits by token semantics and emission coverage. It checks duplicate
identity, prefix consistency and conservation; partial/cancelled streams stay
incomplete. Missing NInfer external-emission observations are unknown, not zero.
With `--round-logs`, the report also joins native execution records by engine,
run, level and round, selects the actual lane/epoch span where applicable, and
checks offered input width and exact sampled output prefixes when available.
Missing execution records, ambiguous spans and unavailable exact IDs remain
unknown. Strata ends the caller linkage immediately after verification; later
draft production cannot inherit the preceding verifier's round identity.
Initial outputs outside these loops, complete request boundaries/IDs, draft
production/cost, cancellation paths outside Program commit and confirmed GPU
commit completion remain outstanding. This report cannot establish complete
request throughput or qualify an optimization.

## Campaign artifacts and reproducibility

Actual MTP calls produce separate `ninfer-strata-v100-draft-v1` records. Their
process-local draft ID and sequence label correlate production without assigning
the next verifier's round before it exists. Strata additionally records the
preceding verified round where available. Successful calls record actual produced
counts and, at levels 2/3, exact IDs; failed calls leave production unknown.
Level 1 adds no timing calls. Level 2 host call duration includes existing native
waits and concurrent work, so it is not an isolated GPU interval. The observations
include Strata CLI's initial draft and subsequent serve/CLI draft calls, and
NInfer Program's native MTP call before constraint trimming. Prompt MTP prefill,
history setup and alternative suffix/oracle production costs remain uncovered.

`tools/telemetry/draft.py --lifecycle-logs ...` reports successful production and
offered verification drafts by actual proposal source separately. It rejects
duplicate draft identities, invalid timing/trace coverage and fabricated failed
production. Missing call timing remains unknown. It does not infer discarded
tokens from aggregate subtraction or claim complete request costs.

The manifest freezes full engine SHAs, model/tokenizer and quant identity,
configuration and manifest SHA256, exact prompt/continuation IDs, experimental
flags, cache policy/bytes, worker/affinity/NUMA settings, graph mode, actual
speculation semantics, page-cache warmup and process order. `environment.json`
records host/kernel/CUDA/driver/GPU topology, affinity/memory policy and model
allocation breakdown. Sample NVML and /proc process state at 100–200 ms.

The bundle contains `manifest.json`, `environment.json`, `summary.json`,
`requests.jsonl`, `rounds.jsonl`, `layers.jsonl`, `experts.jsonl`, `memory.jsonl`,
GPU/process samples, token traces, logs and config snapshots. Level 3 adds one
Nsight trace per engine. Empty/unavailable files are documented as coverage
gaps. No fabricated values may satisfy a campaign gate.

## Gates and matrix

Hardware jobs are dispatch-only and `cancel-in-progress: false`; software checks
use hosted CPU runners. No push acquires the V100. Preserve the installed
Strata checkout; instrument a user-controlled fork of verified installed SHA
`ad5206fba4914b4ab0ffb5e1d17bae4cd77ba778`.

Before the first hardware campaign: validate known routes/bytes, nesting,
CUDA-event boundaries, telemetry off/on numerical equivalence and schema output.
Build each frozen engine once. Execute the full matrix sequentially in one job:
levels 0/1/2 overhead, speculation off, matched actual verification, native MTP,
1K/4K/16K prefill, static and adaptive residency. Use fixed-token teacher-forced
decode for work comparison, normal generation separately, 1–2 warmups and at
least three measured repeats, alternating engine order. Measure <2% level-1
overhead; investigate variation >5%. Quantization differs, so normalize subsystem
work by routes/bytes/launches; raw throughput alone cannot attribute a kernel gap.

Only after the attribution report may one opt-in A/B feature be tested. Preserve
strict exact-transform checks and paired baseline-relative oracle qualification
for arithmetic changes. This phase makes no new optimization/accuracy claim.

## Implementation coverage

Coverage is reported per engine and level in `v100-telemetry-progress.md`.
The presence of a schema or collector is not evidence that every requested
metric has been wired or hardware-qualified. Missing GPU/MTP/request linkage
must block a comprehensive campaign and critical-path attribution.
