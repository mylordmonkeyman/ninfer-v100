# First controlled NInfer / Strata V100 performance A/B results

Run: https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37719546391
Source commit: `811dfae49677dcc1d8bcf3c2afc0cc56570327be`.
Frozen Strata telemetry source: `0430d397d907032fc9ab83c8bb7acff48a935b53`.
Evidence: GitHub artifact `v100-performance-ab-37719546391-1`, ID `11526626035`, 17,743,934 compressed bytes.

Both engines built and six NInfer/Strata baseline/valid A/B configurations succeeded. **Nine
of 13 cases completed**; four optional cases failed startup, did not run requests,
and were not counted as successes. The workflow reported success because it is an
evidence-gathering campaign. A separate guarded runner was not needed.

## Measured results (585 input and 64 generated tokens)

| Engine | Case | HTTP wall median | Ratio to its OWN baseline | NInfer native prefill | NInfer native decode |
| --- | --- | ---: | ---: | ---: | ---: |
| NInfer | baseline, 64 CPU workers, static seeded GPU expert cache | 14.051 s | 1.000 | 9.343 s | 4.683 s |
| NInfer | expert GPU cache disabled | 18.698 s | 1.331 | 12.672 s | 5.991 s |
| NInfer | 16 CPU expert workers | 22.792 s | 1.622 | 15.697 s | 7.041 s |
| NInfer | grouped expert-cache prefill disabled | 17.413 s | 1.239 | 12.713 s | 4.682 s |
| NInfer | baseline repeat | 13.938 s | 0.992 | see request logs | see request logs |
| Strata | baseline (auto cache, native IQ pack, MTP spec4) | 4.567 s | 1.000 | 3.406 s† | 1.128 s† |
| Strata | 16 CPU expert-pool workers vs baseline 31 + host | 4.457 s | 0.976 | 3.376 s† | 1.063 s† |
| Strata | prefill chunk 256 vs auto | 7.426 s | 1.626 | 6.311 s† | 1.018 s† |
| Strata | baseline repeat | 4.691 s | 1.027 | see request logs | see request logs |

† Strata timings come from its HTTP response `timings.prompt_ms` /
`predicted_ms`, not NInfer's native request timing record.
The two engines have different quantization/kernel layouts, expert
cache representations and MTP window settings. Raw 14.05/4.57 s
is **descriptive**, NOT a matched-quant causal comparison.
Long-prompt baseline repeat drift is under 3%, while short-prompt
Strata baseline repeat drift is ~28% and cannot be used for reliable
small-effect assertions.

## Major identified performance mechanism

NInfer's GPU expert cache allocated **156 slots per layer** (7,488 slots over
48 routed layers, **20.70 GB**), with `NINFER_V100_EXPERT_POLICY=static`.
The NInfer cache constructor explicitly sets `admissions_enabled_=false`
when `static` is requested; seeded residency cannot adapt to new routing.
The frozen current NInfer Level-1 baseline accounted for 576/576 prefill
layer observations and 20,160/20,160 verify layer observations. Of
397,440 verify expert routes, **52.45%** were GPU-resident;
47.55% required CPU processing. Across prefill routes, only
**37.94%** were GPU-resident. The host-worker comparison is therefore
highly relevant: cutting NInfer CPU workers from 64 to 16 increased
long-prompt total time 62% and decode ~50% (4.68 to 7.04 s).

Strata baseline reports **8,082 GPU expert slots** using **23.59 GiB**
and a further ~48.14 GiB pinned host-memory expert complement, adaptive
GPU/host expert exchanges, default **31 CPU-pool workers + host thread**.
Strata's per-request decode GPU expert-cache hit rate progressed from
75.9% on the first request to ~97% after repeated long prompts.
The same NInfer/Strata native `verify` telemetry shows Strata's
observed classified verify routes ~**89.47% GPU-resident** and 9.51%
CPU, versus NInfer 52.45% GPU-resident and 47.55% CPU. These counts
span all requests and have different speculative verification widths,
so they are strong *diagnostic* evidence, not matched-token accuracy.

It is wrong to say NInfer had only 156 TOTAL expert slots; the
native `phase13.cache.slots_per_layer=156` is per layer.
The two engines use broadly similar VRAM budgets for expert cache,
but radically different cache policies and observed residency.
The leading controlled hypothesis is therefore that NInfer's **static
resident selection** is causing excess CPU expert work. This is more
specific than assuming all of NInfer's slowdown is intrinsic to its kernels
or quantization.

## Failed first-run cells and precise fixes

* Strata cache-off: the experiment harness removed `--expert-profile`,
  but the installed configuration's resident CPU expert mode requires a
  static profile and mmap experts. The next experiment keeps the profile.
* Strata spec1: native IQ pack expressly requires `--spec T` with T >= 2;
  follow-up uses `--spec 2`, retaining the same MTP artifact.
* NInfer and Strata NUMA preferred node0: the GitHub Actions container lacks
  privilege for `set_mempolicy` / `--preferred=0`. The follow-up will
  apply node0 *CPU affinity only* via `taskset`, intersecting node0 CPUs
  with the container's permitted cpuset. This is not a host-memory NUMA
  binding experiment.

## Follow-up single-campaign matrix

The next A/B campaign repeats both baselines, then compares:
1. **NInfer static-to-LRU admissions**, retaining the same profile,
   GPU cache size, quantization and workload. This is the highest-value
   currently supported explanation test.
2. Strata cache-off retaining the profile, and separately **4,096 slots**
   versus the automatic 8,082.
3. Strata `--spec 2` versus its original `--spec 4`.
4. CPU-only node0 affinity in both engines, not pretending to set
   unavailable NUMA memory policy.
5. Both baseline repeats at the end.

Each engine remains otherwise unchanged and only one GPU model runs
at a time. Individual unsupported cells are reported as failed rather
than silently omitted; do not restart completed cells merely to tweak
telemetry.
