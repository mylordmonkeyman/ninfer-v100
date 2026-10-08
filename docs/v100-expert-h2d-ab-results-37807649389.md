# V100 expert H2D reduction A/B — run 37807649389

**Completed October 8, 2026; success.** [GitHub hardware run](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37807649389), source commit `2f4ba0db6ad4449bb6663af7ea5cf78b07fefe75`, artifact ID `11565225614` (`v100-ninfer-profile-stream-ab-37807649389-1`). Companion [hosted checks](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37807649452) succeeded. All five serial cases completed. Same resident 113 GB mixed-NVFP4/FP8/PLE-INT4 model, V100, 156 expert-cache slots/layer, greedy 64-token decode, 5 warmups + 3 measured long (585 input) and 3 measured short (63 input) requests per case. Baseline repeated last. No other GPU workload was started.

| Policy | Long HTTP median s | Long native prefill s | Long native decode s | Long expert H2D GB/request | Long CPU routes | Short HTTP median s |
|---|---:|---:|---:|---:|---:|---:|
| LRU + auto256 baseline | 9.776 | 6.841 | 2.921 | 24.39 | 0 | 4.037 |
| Profile prior 50 + auto256 | 10.105 | 6.837 | 3.264 | 25.54 | 0 | 4.265 |
| LRU + auto256, GPU fraction 90% | 9.559 | 6.572 | 2.981 | 22.08 | 859 | 3.926 |
| **LRU + auto256, GPU fraction 75%** | **9.049** | **6.111** | **2.933** | **18.41** | **2,745** | **4.043** |
| LRU + auto256 baseline repeat | 9.708 | 6.790 | 2.927 | 24.39 | 0 | 4.018 |

Long-request wall time for GPU75 improved **7.4%** versus first baseline and **6.8%** versus repeated baseline, while native prefill improved **10.7%** and **10.0%**, respectively. The transfer volume fell by **5.98 GB/request (24.5%)**, with CPU processing a small fraction of routed tokens. Short latency remained within measurement variation. All median statistics are from the saved run artifact, not estimates.

**Profile-prior 50 did not help:** it actually raised long H2D from 24.39 to 25.54 GB and made long and short requests slower. Do not adopt this combination. GPU90 yielded a smaller ~2.2% long wall gain versus first baseline. The baseline repeat was stable (0.7% long, 0.5% short). Greedy output hashes sometimes vary; this is not an accuracy verdict, and no further quality campaign is requested.

**Next:** A single protected fraction crossover test at 75%, 70%, 60%, 50%, with full-GPU baseline before and after, has been started separately in [run 37813984867](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37813984867) (source `56528b4623`). Do not overlap another GPU job or change defaults before those results. If the crossover confirms gains, focus next on reducing expert H2D and CPU expert prefill overhead rather than expanding telemetry. Do not touch installed Strata, resident model or Phase18/SV8.
