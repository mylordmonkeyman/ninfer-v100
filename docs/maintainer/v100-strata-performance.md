# Strata-derived V100 performance project

Implementation branch: `perf/v100-strata-derived`.
Frozen source: `forwardport/v100-flash-next` at
`e184de12772999f8d336ea395cedcc0cf0543c52`.
Reference inspected: `jmnargi/Strata-V100` `9d7774919e26d235359bc2c3001f61f607eb288d`.
Milestones are SV0–SV8 in the supplied specification; they do not advance Phase 18.

## Current milestone: SV0 instrumentation, first tranche

`NINFER_V100_TELEMETRY=1` emits schema-1 JSONL on stderr for host-backed MoE
layer calls and PLE gathers. It also enables the existing expert-cache CUDA event
measurements. No routing, arithmetic, cache admission/replacement, or execution
policy is changed. With telemetry off, histogram construction, JSON formatting,
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
they do not infer request IDs from layer numbers. Reports preserve unavailable
metrics and always mark qualification pending.

## Remaining SV0 requirements

Before SV1 execution changes, add explicit request correlation, isolated transfer
timing, BF16 timing by implementation, and full VRAM category accounting. Run
SM70 builds and existing Flash-Next tests, the unchanged Phase 11 independent
oracle checks, and alternating baseline/candidate throughput measurements with
telemetry **disabled**. The local host-only histogram/JSON/report checks do not
satisfy that hardware gate. No optimization is promoted by this first tranche.

## Subsequent order

SV1 profile-seeded/adaptive residency; SV2 device route merge; SV3 prefill
streaming; SV4 grouped AVX2 misses; SV5 early route handoff; SV6 PLE queued I/O;
SV7 Volta tensor-core projections/attention; SV8 optional KV tiering only after
SV1–SV7 measurements. Every feature remains switchable until its numerical,
resource, and request-level acceptance requirements are met.
