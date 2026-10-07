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

## Campaign artifacts and reproducibility

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
