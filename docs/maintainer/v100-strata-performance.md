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

The operator is not yet used by the runtime. `.github/workflows/v100-sv2.yml`
builds and checks it before integration. Next, scalar/grouped/batched GPU hits
must write into a single device route layout, only CPU misses must be uploaded
from pinned storage, leases must remain protected until hit completion, and the
device sum must feed the existing shared/final merge. Transfer-byte elimination
and full mixed-route integration remain unqualified.
