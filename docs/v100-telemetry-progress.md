# V100 comparative telemetry progress

The current authority is `v100-cross-engine-telemetry-spec.md`, implementing the
2026-10-07 comparative-telemetry handoff. Do not resume optimization screens.

## Frozen starting points

- NInfer: `e0e138f3dc3318468c8a8d8dec8eae4a9b136c8e`, branch `perf/v100-strata-derived`.
- Installed Strata: `ad5206fba4914b4ab0ffb5e1d17bae4cd77ba778`.
  Verified from `strata-sha.txt` in successful Actions run `37683421865`,
  artifact `11510907047`. This supersedes the handoff's older examined revision.
- Run #24 completed; no cancellation/restart was performed. Its worker-count screen
  remains context only, not cross-engine subsystem attribution.

## Workflow discipline

All workflows using the self-hosted runner are manual-only, with cancellation
disabled. Push-triggered telemetry software checks use GitHub-hosted Ubuntu only.
The old comparison workflow is retained for historical reproducibility; do not
use it for the new campaign. The controlled telemetry campaign must replace it
as the active experiment entry point after telemetry fidelity gates pass.

## Remaining gates

The user-controlled Strata fork now exists at the verified installed SHA. Finish
instrumenting both engines, validate counters and numerical off/on equivalence, build once,
then validate overhead and run the frozen matrix. Never dispatch a performance
campaign solely because a commit was made. No baseline attribution is yet available.

## First NInfer instrumentation increment

The shared schema and host-only round collector are implemented. Level 1 observes
per-round/per-layer route conservation, logical CPU weight traffic, transient
expert traffic, actual route/output transfer payloads and host time for routed
MoE, CPU-pool execution, existing router rendezvous, QSA/GDN/PLE and layers. It
adds no CUDA events or waits and does not force graph replay to eager execution.
Graph replay without host callbacks leaves layer measurements unavailable.
The old detailed event timing remains Level 2; it is not an aggregate overhead
measurement. Existing diagnostic tests and shared-schema/collector tests pass.

This increment does **not** complete T2/T4. Full request/lane linkage, actual
worker distribution, distinct hit/miss IDs, cache deltas, GPU stage events,
launch counts, NVTX, MTP acceptance/commit and hardware numerical/overhead
qualification remain to be wired. No <2% overhead claim has been made.

## First Strata instrumentation increment

Strata commit `44efd143e9e574ddf40d38dd6145c0504ce8707d` is the first counterpart
increment on `mylordmonkeyman/Strata-V100:telemetry/v100-ninfer-compare`, based on
the verified installed revision. It contains the same host-only collector/spec,
scopes native prefill/verify/decode rounds and records native multi-token adapter
route classifications, CPU groups, logical weight bytes and host pool time.
Prefill expert subpaths and complete GPU/MTP/cache/worker instrumentation remain
missing. The installed checkout has not been changed.

Hosted CUDA syntax checks use an isolated checkout of the exact Strata telemetry
commit. They also compare the shared collector and spec bytes between engines.
These check host compilation only, not linking, CUDA kernels or hardware fidelity.
A missing target export include path in the first hosted NInfer syntax command
was identified and corrected.

The user provided `mylordmonkeyman/Strata-V100` during this session. Its
`telemetry/v100-ninfer-compare` branch is based on the verified installed SHA.
Use connector commits for this fork; no browser fallback is needed to create it.
GitHub's connected tools expose read-only Actions inspection and reruns, but no
workflow dispatch. Address that capability only once the concrete campaign is
ready; no hardware benchmark is authorized by a telemetry-development push.

The preliminary common comparison tool reports observed per-layer host costs,
coverage and work normalization, explicitly leaving critical-path contribution,
request throughput and A/B candidates unqualified. It must not be used to
justify an optimization before the remaining telemetry gates are met.

Hosted check `37691123945` passed all 65 Python tests and host collector checks,
but its CUDA syntax step never compiled: container default `sh` rejected the
Bash include array. The job now selects Bash explicitly and uses the compiler
already present in the CUDA image, avoiding redundant apt operations. This was
a hosted workflow-shell failure, not a self-hosted V100 interruption.

## Second instrumentation increment

Hosted run `37692062279` succeeded: all 65 Python tests, host collector checks,
and changed NInfer/Strata CUDA host syntax checks passed. It acquired no V100.

NInfer now records distinct resident and distinct persistently missed expert IDs
from each actual cache decision, including the hybrid streaming path. These are
per-call counts accumulated into a round, not a request-wide union. The validator
now bounds distinct work by observed routes instead of incorrectly capping a
round at 512; repeated verify groups/chunks can exceed that cap. Its regression
test raises the focused telemetry test count to eight (66 Python tests total).

Strata commit `8bfa5bc409b73d65ea475d546d73073660fb89da` adds native prefill
layer, PLE, GDN, QSA and MoE inclusive host spans. They add no CUDA events or
synchronization and measure host submission/waits, not GPU execution. The hosted
workflow now checks this frozen revision. GPU intervals, request/token/MTP
linkage, actual worker distributions, cache deltas and prefill expert counters
still need implementation; no comprehensive campaign is ready yet.

## Actual CPU worker increment

Hosted run `37692592309` passed all 66 Python tests, collector checks and both
engines' changed CUDA host syntax checks before this increment.

Both engines now report actual whole/group and gate/up/down shard job claims by
pool, worker and role, including Strata's separate host drainer. Level 1 adds
owner-local aligned counters with no per-job clocks. Level 2 adds elapsed host
job times. Counts/times are published through the existing completion protocol
and merged by the owner; job scheduling, barriers and arithmetic stay unchanged.
Observed idle workers have zero claims; affinity, native OS thread ID and
idle/wait intervals are still unavailable. Job counts are not route counts.

Strata worker commit: `f71fa0c41bf78aeb31e3656c5ce480a01d670ce3`.
Local real CPU fixtures passed at levels 0/1/2 for both engines, with exact serial
output equality, repeated batches and known phase-job conservation. NInfer
covers whole/group jobs, row sharding and recovery; Strata covers the Q2 path,
inline execution, host draining, sleeping workers and multi-token row phases.
These do not qualify the V100's native Q4 experts or full-model off/on equality.
The hosted workflow now runs these fixtures, reports unsupported ISA as explicit
unqualified skips, and includes both changed worker pools in CUDA host checks.

The common worker schema/coverage tests pass (nine focused telemetry tests,
67 Python tests total). Level-1 overhead, native-model equality, GPU intervals,
request/token/MTP linkage, worker affinity/idle time, cache deltas and memory
sampling remain outstanding. No new hardware campaign has been launched.

## External host/GPU sampler increment

Hosted run `37694357589` succeeded: all 67 Python tests, both real CPU pool
fixtures at levels 0/1/2, shared collector/spec parity and CUDA host compilation
for both engines (including the worker pools) passed without skips.

`tools/telemetry/sample_system.py` now samples an explicitly named NInfer or
Strata process at 100–200 ms, with optional explicit GPU UUID selection through
read-only NVML. Process CPU/fault/I/O/memory snapshots, host memory and optional
GPU memory/utilization/power/clocks/PCIe rates are available. More expensive
per-thread/NUMA snapshots default to 1000 ms. Missing metrics stay unavailable;
PID/TID reuse is checked, collector costs/timestamps are recorded, and terminal
records distinguish normal duration/process exit from a truncated stream.

The separate `ninfer-strata-v100-system-v1` validator checks stream identity,
ordering, finite values, coverage and terminal sample counts. The tool observes
both engines from the campaign checkout; it does not need to alter Strata's
installed source. Usage, units and qualification limits are in
`v100-system-sampling.md`. No CUDA context, benchmark or runner is started by it.

Six focused tests cover parsing/units, mixed page sizes, missing coverage,
PID reuse, the NVML C ABI via a compiled fixture, a real child process, bounded
collection, process exit and truncation. This environment mounts `/proc` from a
parent PID namespace; the real fixture supplies its explicitly visible proc PID.
All 15 telemetry tests and 58 diagnostic tests pass locally (73 total).

The real V100 NVML path and external sampler overhead remain unqualified. Device
utilization is not request/engine attribution, and process/NUMA snapshots are not
an allocation breakdown. Request/token/MTP linkage, native Q4/full-model off/on
qualification, cache deltas and deferred GPU intervals remain required before
the comprehensive hardware campaign. No new V100 run has been dispatched.


## Native context, cache windows and worker reporting increment

Hosted run `37695723107` succeeded for the external sampler increment, including
both real CPU fixtures and CUDA host syntax checks. No V100 was acquired.

NInfer common records now link executor/transaction, lane/epoch and token-position
spans to actual decode/prefill/verify inputs. The actual graph/eager choice is
recorded without changing it. Levels 2/3 copy exact already-host-visible input
IDs and sampled candidates; level 1 records ranges only. Strata prefill and
verify record positions and host input IDs, and verification records sampled
candidates after the existing completion boundary. Its session loop records the
position/graph path but cannot see device-owned embedding token IDs. A successful
split verifier now correctly marks its owner telemetry scope successful; native
return behavior is unchanged. Caller request IDs, teacher forcing, MTP acceptance,
commit and emitted token traces are still outstanding.

NInfer cache totals now have independent layer counters updated under existing
locks, including startup seeding, asynchronous fill publication and generation
changes on reset. Common records contain before/after occupancy/capacity and
cumulative totals for each observed host-routed layer call, with Ready IDs only
at levels 2/3. Global legacy totals are retained as global legacy diagnostics.
The preliminary report differences only matched-generation windows; asynchronous
fills between windows remain unassigned and reset windows remain unknown.

The report now includes observed worker distribution with process/pool identity,
job totals and timing coverage. It does not infer idle time or add overlapping
thread times to request latency. New regression coverage exercises exact token
traces, multiple lane epochs, cache reset boundaries/monotonicity, cumulative
versus delta reporting and partial worker timing coverage. The existing GPU cache
fixture also checks layer isolation, seed accounting and reset semantics; its
hardware execution is pending. The shared collector/spec remain byte-identical.

All 20 telemetry and 58 diagnostic Python tests pass locally (78 total). These
are software checks, not native Q4/full-model numerical equality, V100 cache
qualification or level-1 overhead. Deferred GPU intervals/NVTX, launch accounting,
caller request/MTP linkage and the frozen comprehensive hardware campaign remain
required. No optimization policy or V100 campaign has been started.


## Additional Strata adapter coverage

Hosted run `37698041978` succeeded for NInfer `9edcb12ea9d409118e922ebba0dc34c606a78ae2`
and Strata `0c6692f1e890ee3a12c327d1a77f92c94dd4ffcd`: all 78 Python tests,
round scope/trace fixtures, NInfer real CPU outputs/accounting at levels 0/1/2,
shared contract parity and both engines' CUDA host compilation, including the
new GPU cache fixture's syntax. Its hosted Strata CPU fixture explicitly skipped
because that machine lacked AVX512-VNNI/VBMI. The local Strata CPU fixture did run
and passed at levels 0/1/2; the hosted skip is not hardware qualification.

Strata `2753d4d1d326f296cc3cc484d316849a10088c7f` adds single-token adapter
routed/resident/CPU classifications, distinct work, logical CPU weight bytes,
MoE host spans and pool intervals. Classifications use the actual submitted CPU
job slots, not a second cache decision. Remote-GPU adapter classification remains
unavailable. Adapter/hit callback failures now mark the active round failed even
if an outer native loop otherwise returns success; native failure control flow
is unchanged.

Verification transient expert H2D bytes are recorded at successful native
`cudaMemcpyAsync` submissions. Explicit direct-mapped/no-copy calls observe zero
copy bytes; PCIe demand-read traffic remains unavailable. Split-group doorbell
steps map back to the actual model layer. Copy submission failures mark telemetry
failed, without adding waits or changing native error handling. The new adapter
and DMA observations still require actual native-model fidelity on V100.


## Caller speculation lifecycle increment

Hosted run `37698481145` fully succeeded: all 78 Python checks, both real CPU
fixtures at levels 0/1/2 without skips, caller-independent scope fixtures,
shared contract parity and both engines' CUDA host compilation passed.

NInfer Program now records actual speculative verification acceptance and
possibly trimmed committed output tokens, linked to native executor/transaction
and lane/epoch. Ordinary decode also records verified/committed results; zero
acceptance records cancellation at the commit boundary. Strata serve and CLI
verification loops correlate execution rounds with actual accepted drafts,
input-state-prefix commit returns and outputs after their native EOS/max-new
loops. MTP, suffix and oracle proposals retain their selected source. Caller
trace labels are explicitly process-local, not invented external request IDs.
No precision, proposal policy, output loop or synchronization behavior changes.

The separate lifecycle validator/report keeps offered verification drafts,
accepted outputs, state-input commits and emissions distinct. It rejects
conservation/identity/prefix mismatches, preserves partial coverage and leaves
NInfer external emissions unknown. Real C++ fixture records exercise levels
0/1/2, nested round linkage, trimmed commits and EOS-shortened emissions. These
fixtures validate telemetry contracts, not full-model acceptance or accuracy.
All 25 telemetry and 58 diagnostic Python tests pass locally (83 total).

This increment does not complete request attribution. Full external request
IDs/start/end, initial outputs outside the loops, NInfer engine output emission,
draft generation/production costs, additional cancellation paths, confirmed GPU
commit completion, deferred GPU timing and native-model hardware fidelity remain.
Actual caller source host compilation is included in the next hosted check.
