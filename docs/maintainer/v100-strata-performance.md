# Strata-derived V100 performance project

Implementation branch: `perf/v100-strata-derived`.
Frozen source: `forwardport/v100-flash-next` at
`e184de12772999f8d336ea395cedcc0cf0543c52`.
Reference inspected: `jmnargi/Strata-V100` `9d7774919e26d235359bc2c3001f61f607eb288d`.
Milestones are SV0–SV8 in the supplied specification; they do not advance Phase 18.

## Current milestone: SV0 instrumentation and hardware qualification

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
instrumentation throughput screen. A final step still fails unless both unchanged
oracle invocations pass; collecting later evidence never converts a failed gate
into qualification.

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
fails identically on the candidate. SV1 execution-policy work remains blocked
until that pre-existing numerical gate is repaired on the frozen baseline or the
project explicitly accepts exact baseline parity in place of that requirement.

Host-only checks pass locally. CUDA builds and hardware gates must pass before
SV1 execution changes; no performance improvement or completed SV0 qualification
is claimed until the evidence supports it.

## Subsequent order

SV1 profile-seeded/adaptive residency; SV2 device route merge; SV3 prefill
streaming; SV4 grouped AVX2 misses; SV5 early route handoff; SV6 PLE queued I/O;
SV7 Volta tensor-core projections/attention; SV8 optional KV tiering only after
SV1–SV7 measurements. Every feature remains switchable until its numerical,
resource, and request-level acceptance requirements are met.
