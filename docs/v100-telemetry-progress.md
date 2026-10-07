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

Create the user-controlled Strata fork at the verified installed SHA, instrument
both engines, validate counters and numerical off/on equivalence, build once,
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
