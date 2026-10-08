# V100 second A/B: cache residency identified; prefill remains the major gap

Run https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37723429228
Commit `eddbc1dd6e40fdcb7358555549c6d4d9e0060f10`.
All 10 requested cells **started, completed, and emitted evidence**.
Artifact `v100-performance-followup-37723429228-1` (ID `11527563634`).
Software checks run `37723429111` passed.

## Matched, within-engine results

All long requests have 585 input and 64 output tokens. 5 warmups + 3
measured long + 3 measured short per cell. Values are medians.
Comparisons to a baseline of a different engine are *not* controlled
because model quantization, kernels and MTP differ.

| Engine | Control | Median long HTTP s | vs own baseline | Prefill s | Decode s |
|---|---|---:|---:|---:|---:|
| NInfer | static seeded profile, 64 CPU workers | 13.833 | reference | 9.160 | 4.621 |
| NInfer | LRU admission enabled, same 156 slots/layer | 12.290 | 11.2% less time | 8.740 | 3.545 |
| NInfer | CPU node0-only affinity | 18.061 | 30.6% more time | 14.674 | 3.487 |
| NInfer | baseline repeat | 14.003 | 1.2% more time | ~9.28 | ~4.71 |
| Strata | auto cache (~8,082 slots), spec4 | 4.536 | reference | 3.447 | 1.118 |
| Strata | cache arg `0` | 4.540 | no change, **INVALID cache-off experiment** | 3.370 | 1.150 |
| Strata | cache limit arg `4096` | 5.760 | 27% more time | 3.802 | 2.138 |
| Strata | spec2 vs spec4 | 4.430 | 2.3% less time (near noise) | 3.258 | 1.442 |
| Strata | CPU node0-only affinity | 4.779 | 5.4% more time | 3.627 | 1.209 |
| Strata | baseline repeat | 4.677 | 3.1% more time | ~3.52 | ~1.18 |

The Strata spec2 total is slightly faster despite slower decode because
prefill timing varied; it does **not** establish a speculative decoding
speedup. Use native phase times and baseline drift when judging effects.

## Cache residency: strongest supported finding

NInfer's 156 GPU slots *per layer* (7,488 total, 20.70 GB) were
unchanged between baseline and LRU; both loaded the same frozen seeded
profile (7,488 seeded experts). The static policy explicitly disables
new admissions. LRU enabled eviction/admission.

| Observed routed work across all requests | NInfer static | NInfer LRU |
|---|---:|---:|
| Prefill GPU resident / total | 888,986 / 2,343,360 = 37.94% | 1,031,767 / 2,343,360 = 44.03% |
| Verify GPU resident / total | 208,473 / 397,440 = 52.45% | 370,778 / 416,640 = 88.99% |
| Verify CPU expert routes | 188,967 | 45,862 |

Verify totals differ because speculative acceptance/verification differs,
so absolute route counts are not normalized. The within-engine residency
fractions and native phase times support a substantial reduction in CPU
expert work. NInfer's median decode time fell ~23.3%; prefill fell ~4.6%.
Adaptive GPU residency is a **real but partial** explanation for Strata's
speed advantage.

Strata baseline: ~8,082 resident GPU experts in 23.59 GiB and
~48.14 GiB pinned host expert complement. Its observed verify GPU
resident fraction was ~89.35%, close to NInfer LRU's ~88.99% after
warmup, yet Strata still completed the full long request ~2.7x faster.
This is evidence that **cache residency alone cannot explain the
remaining speed difference**.

### Strata cache flags: interpret actual allocation, not requested values

`--expert-cache 0` was parsed as **auto**, still allocated 8,082
GPU slots, with R4 GPU-hit path on, adaptive swaps and ~89.6% verify
resident routes. This experiment **did not disable the cache**.
`--expert-cache 4096` produced **5,223 actual GPU slots** (15.23 GiB),
not 4,096; it cut verify GPU residency to ~79.63% and increased CPU
fallback. The reported 27% slowdown is a valid *smaller-cache*
comparison but must be labeled with actual capacity.

### Prefill bottleneck, even after LRU

Long-prompt NInfer native prefill was ~9.16s static and ~8.74s LRU.
Strata HTTP prefill was ~3.45s. In NInfer's recent 585-token prefill
rounds, summed host-observed MoE time was ~9.05s static / ~8.59s LRU,
of which CPU expert host time was ~5.80s / ~5.28s. Strata's 584-token
prefill MoE host time was ~2.94s. Host stage times overlap; do not add
them as independent GPU timings.

Thus the next engineering focus is **MoE prefill CPU fallback and
expert execution**: investigate NInfer's NVFP4 CPU expert kernel,
grouping and GPU streaming of nonresident experts versus Strata's
pinned-memory host expert pipeline. The two quantizations differ;
do not attribute the entire prefill gap to CPU scheduling alone.

## Quality caveat before enabling LRU by default

Both baseline repetitions produced identical deterministic 64-token
long and short outputs, but NInfer LRU's response hashes differed from
static and sometimes differed across LRU repetitions. The outputs
remained on-topic in a quick qualitative review, but this **is not an
accuracy qualification**. Differences can result from GPU/CPU numerical
paths and speculative verification. Before changing the default cache
policy, compare a representative quality suite or teacher-forced
reference. Do not equate changed output hashes with quality regression
without a task-level check.

## Practical next step

Retain NInfer LRU as a supported opt-in now; do not change the default
before quality validation. Run **one targeted NInfer-only** comparison
of LRU baseline against the existing GPU expert-stream prefill policy
and profile-prior admission (same artifact, same cache budget), with
quality sanity prompts and a repeat baseline. Only implement a more
invasive Strata-derived MoE prefill optimization after those controls
identify a useful direction. Avoid repeated full cross-engine campaigns.
