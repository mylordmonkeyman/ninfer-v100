# V100 first-qualification continuation: complete paired telemetry evidence

Frozen hardware run: https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37716164597
Hardware source: `7a0662e5ed61494bb3030b0d3d8ad38bc0564724`.
Strata source: `f9d8fc4decc68bf809dfabde410907fb170bb929`.
Frozen hardware artifact: `v100-telemetry-qualification-37716164597-1`, artifact ID `11524343285`, ~30.2 MB compressed.
Offline-only replay: https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37718164443.
Offline replay artifact: `v100-telemetry-offline-replay-37716164597`, ID `11524257909`, ~1.7 KB.
Installed Strata checkout `/opt/ai/strata` and the original NInfer forwardport branch were not modified.

## Status and what is established

Both engines and SM70 GPU expert fixtures compiled and passed, and **all six inference cells completed**.
Each cell uses **five warmups and three measured requests** for the same 64-output-token prompt, seeded greedy setting,
and fixed native MTP policy. Strata uses the correct
`chat_template_kwargs.enable_thinking=false` contract, and this run produced
answer content rather than reasoning-only completions.

The initial strict qualification **failed**, for two distinct reasons.
It is incorrect to label either of them a telemetry-induced change:

* NInfer returned exactly **one output hash across all 24 requests** and all levels.
  Level-1 median HTTP request overhead was **+1.4456%** (14.3891 vs 14.5971 seconds)
  but the max-to-min sample range of the three measured requests was
  **5.926% at level 0 and 5.073% at level 1**, exceeding the independent 5%
  timing-spread qualification requirement. Therefore its overhead remains **unqualified**
  despite the point estimate falling below 2%.
* Strata returned **five distinct output hashes among the eight baseline requests**
  (two distinct hashes among the three measured requests). The baseline was not
  repeatable even with telemetry disabled. Nevertheless, for **all eight request
  positions**, including warmups, the exact output hash, request payload and phase
  were identical at levels 0, 1 and 2. There was **no indexed cross-level mismatch**.
  Strata's level-1 median request overhead point estimate was **-1.6062%**
  (4.5791 vs 4.5055 seconds), with all sample spreads under 2%, passing the
  **HTTP timing-only** qualification gate. The negative point estimate does not
  establish that logging accelerates execution.

Thus we separately report:
1. **Indexed HTTP observer parity: PASS for both engines** (48 saved response
   fingerprints verified, 8 indexed positions per engine per level; zero mismatches).
2. **Strict baseline repeatability: FAIL for Strata**, independent of telemetry.
3. **Telemetry Level-1 overhead: HTTP timing gate PASS for Strata, NOT QUALIFIED
   for NInfer** due to sample variation.

Neither result establishes teacher-forced native numerical equivalence, that
the same internal tokens were traversed, or causality on the GPU.

## Evidence validation without hardware churn

`tools/telemetry/replay_qualification.py` re-analyzes the original saved HTTP
request/response JSONL in the frozen GitHub Actions ZIP. It verifies response
digests, request counts, warmup/measured sequencing, monotonic boundaries,
the old strict pass/fail result and the old timing metrics before computing
indexed cross-level parity. Its hosted replay on run `37718164443` passed.

The qualification harness's `report()` now computes both the **strict** baseline
repeatability gate and the **indexed observational** parity gate. The strict
gate is retained; it was **not** weakened to make this run appear successful.
Regression tests cover paired output variability, changed outputs with the
same output-hash distribution, mismatched request payloads and corrupted
archival responses. This analysis requires no additional V100 workload.

## Diagnostic payload and outstanding coverage

Native log size for eight requests was approximately:
NInfer level 1 **33.3 MB**, level 2 **244.4 MB**;
Strata level 1 **9.85 MB**, level 2 **50.75 MB** (the latter are
`native-engine.log`, not the short Python HTTP `server.log`).
Level-1 logs use lossless packed worker tuples; Level-2 logging intentionally
preserves much greater detail. Raw telemetry sizes are not isolated overhead
measurements.

Next, do not churn hardware by repeating the same six-cell matrix.
Investigate Strata's baseline request-to-request variability in a controlled
native fixed-token / no-speculation diagnostic, preserving production settings
for the normal performance run. Obtain native request ownership,
exact emitted/verified token correlation and GPU stream interval/launch coverage
before making per-layer causal performance claims. Separately tighten NInfer
overhead variance with a pre-registered balanced run, rather than declaring
the observed +1.4456% point estimate qualified.

The artifact-upload guard remains fixed: model alias only under
`$RUNNER_TEMP`, symlinks removed from evidence directories, and upload
refused for evidence >=1 GiB. No model-file contents are uploaded.

## Subsequent instrumentation coverage correction (not yet V100 qualified)

Reading the frozen Level-1 native logs, NInfer emitted route counts for
**15,120/15,120** observed layer calls; Strata emitted all route categories
for **12,384/12,768**. Strata's missing 384 observations were exactly
its eight prefill rounds times 48 layers. Accordingly, the frozen
~39% resident route share for NInfer and ~91.9% for Strata are **not
comparable**: NInfer includes prefill routing while Strata's classified
routing was limited to verify/decoding. The difference is a coverage gap,
not proof of better expert-cache efficiency.

A subsequent **unbenchmarked** change in
`mylordmonkeyman/Strata-V100:telemetry/v100-ninfer-compare`
instruments the Strata prefill GPU router's already-known T tokens and
T*K route total without any GPU copy/synchronization; the resident/CPU/
nonresident-GPU breakdown remains absent because those values are not
available on the fused GPU prefill path. A common-schema CPU regression
tests these known-only counters and confirms unknown classes stay absent.
The offline replay tool reports classified-route coverage explicitly and
uses *classified* route totals, never unclassified prefill totals, as the
denominator for observed residency fractions. This new instrumentation
requires actual V100 measurement in a later **consolidated** campaign
before the missing prefill classification can be evaluated further.
