# V100 NInfer 4-slot vs 8-slot expert-stream ring: no improvement

**Controlled V100 run:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37797479040  
**Result:** SUCCESS; all three serial cases completed.  
**Evidence:** artifact `v100-ninfer-stream-ring-ab-37797479040-1`, ID `11559962617`, 9,429,695 compressed bytes.  
**Build/hosted checks:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37797479145 — passed.  
**Hardware source:** `15286370fa7ec85b7c62143da479a7c8454281f1`.  
**Model:** identical NInfer Qwen3.8 Flash-Next mixed quant, 156 GPU cache slots/layer, adaptive LRU, `auto` prefill stream threshold 256. Only `NINFER_V100_EXPERT_STREAM_RING_SLOTS` differs.

Five warmup long requests, then three measured long (585 input, 64 output) and three short (63 input, 64 output) per case. Median times in seconds:

| Serial case | Long HTTP | Long native prefill | Long native decode | Short HTTP | Short native prefill |
|---|---:|---:|---:|---:|---:|
| **4-slot baseline** | **9.690** | ~6.774 | ~2.903 | 4.102 | ~0.884 |
| 8-slot stream ring | 10.045 | ~7.148 | ~2.897 | 3.994 | ~0.879 |
| **4-slot repeat** | **9.575** | ~6.680 | ~2.904 | 3.957 | ~0.884 |

Medians of per-request native phase timings are shown; the HTTP median and native phase medians do not necessarily belong to the same repetition.

Long latency is **3.7% slower** for eight slots than first baseline, and **4.9% slower** than repeat. Long native prefill rises by ~0.37–0.47 seconds while decode remains ~2.9 seconds. Short total latency appears 2.6% better versus first baseline, but baseline repeat improves 3.6%; do **not** attribute the short difference to eight slots.

**Conclusion:** Keep the four-slot GPU streaming ring (default). Extra scratch/pinned memory and eight in-flight expert groups do not measurably help under this workload and may increase prefill overhead. Do not enable `NINFER_V100_EXPERT_STREAM_RING_SLOTS=8` as a default, and do not pursue larger rings without new causal evidence.

**Next controlled performance test:** Change expert submission order *without changing ring size or cache budget*. Baseline expert ID order, opt-in `NINFER_V100_STREAM_EXPERT_ORDER=busy-first` (experts with most routed tokens submitted first, stable expert-ID tiebreak), baseline repeat. Workflow `.github/workflows/v100-ninfer-stream-order-ab.yml`, one GPU run at a time.
