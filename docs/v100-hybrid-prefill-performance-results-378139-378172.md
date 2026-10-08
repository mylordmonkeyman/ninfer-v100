# V100 hybrid CPU/GPU expert prefill: fraction and CPU grouping results

## Fraction sweep: hardware run 37813984867

[Full run](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37813984867), artifact `v100-ninfer-prefill-fraction-ab-37813984867-1` (ID `11567860321`). Identical NInfer model, quantization, 156 GPU slots/layer, LRU expert policy, auto256 GPU prefill, four-slot ring, MTP1; one GPU, serial cases. Five long warmups and 3 measured requests of each prompt length per case.

| Fraction of nonresident experts sent to GPU | Long HTTP median (s) | Native long prefill median (s) | Short HTTP median (s) | Long expert weight H2D (GiB) |
|---|---:|---:|---:|---:|
| 100% baseline | 10.153 | 6.647 | 5.175 | 22.58 |
| 75% | 9.737 | 6.649 | 6.543 | 17.04 |
| 70% | 9.905 | 5.965 | 4.436 | 15.79 |
| 60% | 9.788 | 5.976 | 4.492 | 13.62 |
| 50% | **9.119** | 6.077 | 4.151 | **11.21** |
| 100% repeat | 10.172 | 6.625 | 5.122 | 22.58 |

H2D bytes come from native level-1 `ninfer-strata-v100-telemetry-v1` prefill records with at least 256 input tokens, aggregated over 48 MoE layers and reported as the median across eight long-prefill rounds/case. They are expert *weights* transferred, not total PCIe traffic. The 50% GPU variant cuts expert staging from 22.58 to **11.21 GiB**, about a 50% reduction, but transfers an additional ~112 MB of CPU expert results to the GPU per long prompt.

**Caution on short latency:** This run had highly variable short-prefill times (the baseline native short prefill median was ~1.93 seconds, but some other cases were ~0.87 seconds). Do not attribute all short improvements to the fraction setting; these are not reliable isolated short-prompt comparisons.

## CPU grouping: hardware run 37817218038

[Full run](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37817218038), artifact `v100-ninfer-prefill-group-ab-37817218038-1` (ID `11568247637`). Same frozen model and GPU configuration. This isolates `NINFER_V100_CPU_EXPERT_GROUP=prefill` versus ungrouped fallback for fixed GPU fractions:

| GPU fraction / CPU fallback | Long HTTP median | Native long prefill | Short HTTP median | CPU expert weight read per long prompt |
|---|---:|---:|---:|---:|
| 60%, ungrouped | 10.544 s | 6.476 s | 4.526 s | 17.22 GiB |
| **60%, grouped** | **9.588 s** | **5.808 s** | 4.561 s | **9.32 GiB** |
| 50%, ungrouped | 9.758 s | 6.689 s | 4.244 s | 28.07 GiB |
| **50%, grouped** | **9.607 s** | **6.491 s** | 4.229 s | **12.80 GiB** |
| 50%, ungrouped repeat | 9.631 s | 6.597 s | 4.241 s | 28.07 GiB |

For 50% GPU streaming, grouping cuts CPU weight reads **54.4%** without changing the **11.21 GiB** GPU expert weight uploads or ~112 MB CPU-result H2D volume. At 60%, grouping cuts CPU weight reads **45.9%**, with GPU expert weight uploads fixed at **13.62 GiB**. The matched 60% long latency improves 9.1% in this run, while matched 50% improves 1.5%. The ungrouped 60% case has greater variability than the grouped control, so do not extrapolate an unconditional 9% win.

The groups are formed within the AVX2 CPU expert pool by expert ID, regardless of input task ordering. The implementation does not change quantization or the persistent GPU cache budget.

## Implications

1. The first large gains in the project came from LRU+auto256 instead of static caching (about 30% lower long latency, earlier run `37769284560`).
2. Hybrid prefill meaningfully reduces expert H2D weight transfers, but execution time depends on fallback CPU efficiency and repeated-request cache behavior. A fixed fractional cutoff need not be optimal for other prompt lengths or routing distributions.
3. CPU grouping is useful for **hybrid prefill**, unlike earlier all-GPU experiments where CPU grouping was not the bottleneck.
4. The next controlled test uses per-expert **absolute route-count thresholds** (10, 12, 14) versus 50%-GPU grouped fallback, with the same hardware and baseline repeat: [run 37835323235](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37835323235). Do not promote a new default until its wall latency and transfer volumes are measured.

No further model-quality campaign is proposed. Negative speed results are retained; preserve the original model and installed Strata.
