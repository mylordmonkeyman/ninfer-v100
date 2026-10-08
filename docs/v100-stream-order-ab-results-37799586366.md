# V100 expert-stream submission order A/B: busy-first offers no gain

**Controlled hardware run:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37799586366  
**Evidence artifact:** `v100-ninfer-stream-order-ab-37799586366-1`, ID `11560198720` (9,420,997 bytes compressed).  
**Frozen hardware source:** `6ca3b7650b9d6234e96036e98a0f11a4b64589c3`.  
**Hosted checks:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37799586318 — both succeeded.

Three serial NInfer-only cells used the same Qwen3.8 Flash-Next mixed-quant weights, V100, verified profile, 156 persistent GPU expert slots per layer, MTP draft 1, **four** in-flight expert stream slots, and LRU+auto256 prefill policy. Only expert **submission order** changed: expert ID order (default), `NINFER_V100_STREAM_EXPERT_ORDER=busy-first` (descending routed token count), then default repeated.

Five long warmups, three long requests (585 input, 64 generated tokens), and three short requests (63 input, 64 generated tokens) per cell. Median request times (seconds):

| Configuration | Long HTTP | Long native prefill | Long native decode | Short HTTP |
|---|---:|---:|---:|---:|
| **ID-order baseline** | **9.7994** | 6.8633 | 2.9159 | 3.9924 |
| Busy-first order | 9.8604 | 6.9409 | 2.9018 | 4.0437 |
| **ID-order repeat** | **9.5999** | 6.6822 | 2.8881 | 3.9733 |

Results are medians over each phase's own three observations; HTTP and native medians need not refer to the same repetition. Busy-first is +0.6% long wall versus initial baseline and +2.7% versus repeated baseline, with prefill correspondingly unchanged or higher. It is +1.3% short versus initial baseline. The observed changes do **not** establish a throughput benefit and fit baseline drift.

**Conclusion:** Retain expert-ID submission order as the default. Busy-first remains an opt-in diagnostic experiment; do not promote it or add more scheduling order sweeps absent evidence.

**Next test:** A distinct bottleneck remains the blocking host wait on the four-slot GPU expert-stream ring's reused buffers. Experiment `NINFER_V100_EXPERT_STREAM_SLOT_REUSE=pipelined` keeps four slots and the original expert order, replacing CPU synchronization on the prior kernel with GPU-side event dependencies while protecting pinned DMA lifetime. Workflow `.github/workflows/v100-ninfer-stream-reuse-ab.yml` performs a full GPU mathematical fixture and baseline/pipelined/baseline timing A/B. Only one GPU job at a time.
