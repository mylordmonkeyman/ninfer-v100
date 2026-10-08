# V100 prefill per-expert minimum-route A/B — October 8, 2026

**Measured hardware run:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37835323235 (success), source `1c6113ee9f383fc87abb098f595c5372209ab3f8`, evidence artifact ID `11575638434`, name `v100-ninfer-prefill-minroutes-ab-37835323235-1`. Hosted checks https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37835323305 succeeded.

Five **serial NInfer-only** cases used the same resident mixed-quant 113 GB model, V100, LRU+auto256 prefill, 156 GPU expert slots/layer, four-slot expert stream ring, CPU prefill grouping, and fixed MTP depth. Per case: five long warmups followed by three measured long prompts (585 input, 64 output tokens) and three measured short prompts (63 input, 64 output tokens). The opt-in policy `NINFER_V100_PREFILL_EXPERT_STREAM_MIN_ROUTES=N` GPU-streams only missed experts receiving at least N routes; smaller groups fall back to the grouped CPU expert pool. The baseline streams 50% of missed expert groups instead. All cells completed.

| Policy | Long HTTP median (s) | Native long prefill median (s) | Native long decode median (s) | Short HTTP median (s) | Long expert-weight H2D (GiB) | Long CPU routed experts (routes) |
|---|---:|---:|---:|---:|---:|---:|
| GPU50 + grouped baseline | 9.812 | 6.139 | 2.979 | 4.207 | 11.38 | 11,275 |
| Minimum 10 routes + grouped | 9.397 | 5.488 | 2.998 | 3.855 | 8.98 | 17,803 |
| **Minimum 12 routes + grouped** | **8.430** | **5.372** | 3.048 | 4.105 | **8.04** | 21,638 |
| **Minimum 14 routes + grouped** | **8.389** | **5.396** | 2.949 | 4.173 | **7.29** | 25,462 |
| GPU50 + grouped baseline repeat | 9.664 | 6.445 | 3.006 | 4.216 | 11.38 | 11,275 |

H2D numbers are medians of last three long-prefill telemetry rounds per cell, summed over 48 MoE layers; `expert_h2d_bytes` counts expert weight uploads, not total PCIe traffic. HTTP and native medians are computed independently, so their sum need not match. CPU routes and bytes are aggregated in the same long-prefill records. Source script and raw records are in the linked GitHub artifact.

**Result:** Minimum 14 improves long HTTP median **14.5% vs first GPU50 baseline** and **13.2% vs repeated baseline**, with **36.0% fewer expert-weight H2D bytes** (7.29 vs 11.38 GiB/request). Minimum 12 improves long HTTP median **14.1% vs first baseline** and **12.8% vs repeat**, with 29.3% less weight H2D. Minimum 12 and 14 differ by only 0.041 s in median wall latency, too little to establish a clear winner. Minimum 14 shifts more work to CPU (~25,462 routes versus 11,275 on GPU50); summed CPU expert host time rises from ~0.684 to ~1.144 seconds/request, while lower GPU transfer volume improves total prefill.

Short requests stay near 4 seconds in both threshold-12/14 cells and baseline controls; threshold 10's 3.855 s result should not be credited solely to threshold policy because the short prompt is below the streaming threshold and observed short timings vary.

**Decision:** Keep all threshold modes opt-in, no production default change yet. The causal direction is supported by controlled within-engine A/B and native H2D counters, but baseline repeat drift and only three measured long calls/cell warrant another bounded confirmation before promoting either setting.

**Next performance-only test:** One protected serial A/B with GPU50 grouped baseline; min-routes 12 and 14 grouped; baseline repeat, and enough long observations to resolve the 0.041-second difference. Consider extending to distinct long-prompt routing distributions rather than expanding telemetry. Do not add quality campaigns, change installed Strata, modify the resident model, overlap GPU jobs, or proceed to Phase18/SV8.
