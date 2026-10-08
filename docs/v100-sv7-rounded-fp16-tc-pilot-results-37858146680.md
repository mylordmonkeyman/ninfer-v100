# SV7 rounded FP16 tensor-core pilot: isolated V100 results

**Source:** `8e4e5f1526fa8d369be25d6df6d3cffdb56b06de` (2026-10-08); hardware run [37858146680](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37858146680), artifact `11584862621`. Hardware and hosted checks succeeded. Prior first CUDA compile/FP64 oracle run [37843070069](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37843070069) also passed.

These are **isolated BF16 control projection** wall-clock measurements, not production HTTP prefill. Baseline was the existing Volta BF16 SIMT kernel; candidate was opt-in SM70 CUTLASS FP16 TensorOp with FP32 accumulation and BF16 output. Candidate includes per-call BF16→FP16 weight and activation conversion, nonfinite sentinel synchronization, and temporary workspace; it does **not** cache converted weights. The A/B/A test used 2 warmups and 5 measured calls per shape/token count per arm, reporting medians and averaging the two baseline medians. Independent sampled FP64 projection checks passed on both arms; no full-model numerical qualification was attempted. Production default unchanged.

| Projection | T | BF16 SIMT median ms | FP16 TC median ms | speedup |
|---|---:|---:|---:|---:|
| PLE key (10240×2560) | 128 | 9.2196 | 0.3789 | 24.34× |
| PLE key | 512 | 36.8578 | 0.5167 | 71.33× |
| PLE key | 1024 | 73.6946 | 0.8265 | **89.16×** |
| PLE value (2560×2560) | 128 | 2.3822 | 0.1877 | 12.69× |
| PLE value | 512 | 9.4928 | 0.1919 | 49.46× |
| PLE value | 1024 | 18.6667 | 0.2441 | **76.48×** |
| QSA indexer (640×2560) | 128 | 0.4360 | 0.1370 | 3.18× |
| QSA indexer | 512 | 1.8349 | 0.1479 | 12.40× |
| QSA indexer | 1024 | 3.7656 | 0.1677 | **22.45×** |
| Shared expert down (2560×640) | 128 | 0.9670 | 0.0683 | 14.16× |
| Shared expert down | 512 | 3.9370 | 0.0714 | 55.12× |
| Shared expert down | 1024 | 7.9625 | 0.0931 | **85.55×** |

**Interpretation:** BF16 SIMT is the main bottleneck on these larger isolated shapes. The pilot's one-shot host synchronization and repeated weight repack do not erase the TensorOp speedup for these T values. Do **not** extrapolate the speedups to full Flash-Next model: the projection's fraction of total time, model routing, CPU expert stalls, startup, real weight overflow/saturation, and memory budget remain unmeasured in this run.

**Next experiment:** paired real-model HTTP prefill and MTP with same resident artifact, fixed profile, and actual opt-in dispatch evidence. Use a separate protected V100 job, preserve original SIMT as fallback, record TTFT/prefill/decode, startup and VRAM. Independent Phase11 numerical gates must pass before any default promotion. No SV8/Phase18.
