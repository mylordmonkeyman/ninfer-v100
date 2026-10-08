# First controlled V100 performance A/B campaign

Use the **single** workflow `.github/workflows/v100-performance-ab.yml` on
`perf/v100-strata-derived`. It has opt-in `[v100:ab-campaign]`, a shared
concurrency group with other V100 hardware workflows, and cannot run on an
ordinary source commit. Build the two engines once, run the thirteen cases
**serially**, terminate only each case's own process group, verify GPU0
has at least 28,000 MiB free before each case, and preserve partial evidence
with the 1 GiB source artifact cap and symlink removal.

## Controlled cases

| Engine | Variant | Change from that engine's existing baseline |
|---|---|---|
| NInfer | baseline | static seeded profile, 156 expert cache slots, default 64 host expert workers on Volta, 2048 prefill chunk, native MTP width 1 |
| Strata | baseline | existing Q4_K_XL GGUF / IQ pack, expert cache auto + expert-profile, pool worker auto, prefill auto, spec=4 |
| NInfer | cache-off | NINFER_FLASH_NEXT_EXPERT_CACHE=0 |
| Strata | cache-off | --expert-cache 0 (and remove profile because no cache exists) |
| NInfer | workers-16 | NINFER_FLASH_NEXT_CPU_EXPERT_WORKERS=16 |
| Strata | workers-16 | --pool-workers 16 |
| NInfer | prefill-no-group | NINFER_FLASH_NEXT_EXPERT_CACHE_GROUPED_PREFILL=0 |
| Strata | prefill-256 | --prefill 256 instead of auto |
| Strata | mtp-window-1 | --spec 1 instead of 4 |
| NInfer | numa-node0 | numactl --cpunodebind=0 --preferred=0 |
| Strata | numa-node0 | same numactl policy on server and inherited child |
| NInfer | baseline-repeat | unmodified baseline again, for temporal drift |
| Strata | baseline-repeat | unmodified baseline again, for temporal drift |

The baseline models/quantizations **remain different**:
NInfer uses `/models/qwen3-8b-flash-next`, a 113.3 GB NInfer artifact
with a `.ninfer` alias only under `$RUNNER_TEMP`, and Strata uses
existing `unsloth-ud-q4_k_xl` pack and pinned GGUF shards.
The engines also use different initial MTP windows (NInfer 1, Strata 4).
**Do not interpret a raw NInfer/Strata speed ratio as a controlled
optimization verdict.** Within-engine baseline/variant ratios establish
the contribution of cache, workers, NUMA placement, and prefill settings.

## Shared workload

Every case uses 5 long-prompt warmups, 3 long-prompt measured requests
and 3 short-prompt measured requests, each with `max_tokens=64`,
greedy temperature 0, top_p=1, seed 42 and thinking disabled using each
engine's actual frontend contract. Prefix reuse is disabled, and Strata
prompt-cache is disabled. Only one HTTP request is in flight.
Save each request's exact response and usage, observer wall time,
output SHA, and native per-request timing where available.
NInfer native `request_done.timings_seconds` includes prefill/decode;
Strata's native HTTP response includes `timings.prompt_ms`,
`timings.predicted_ms` and draft acceptance counts.

The console and summary show median wall seconds, actual generated
tokens, total HTTP tok/s and **within-engine** speedup ratio.
Native times are for diagnosis; host overhead and some stages overlap,
so they are not necessarily additive. The difference between long
and short prompts is a workload probe, **not** by itself a clean
prefill-rate estimate if response lengths or execution paths change.

System samples use the existing GPU NVML recorder. Their Strata process
CPU fields observe the Python server PID, **not** the full C++ child
process/worker tree. Do not use those parent-PID CPU samples to assert
a complete cross-engine CPU utilization comparison. Native round
telemetry level 1 records available CPU expert jobs, with the
Strata prefill residency classification explicitly incomplete.
A/B output changes and missing metrics are reported; no experiment
is automatically called an optimization.

## Follow-up decision

First compare baseline-repeat against baseline to estimate drift.
Then rank differences **within each engine** by measured effect,
prefill/decode timing, GPU utilization/PCIe throughput, MTP acceptance,
and cache-route coverage. Changes with small/unstable effects remain
hypotheses. A second, **targeted** single-hardware run may be needed
for promising changes or a quantization-matched workload; do not
automatically launch it for each case.

Only one case runs at a time; never delete resident model data,
overwrite the installed `/opt/ai/strata`, create model symlinks
inside GitHub upload paths, or start another GPU workflow while this
campaign is running.
