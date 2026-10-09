# V100 MoE adaptive expert policy: production HTTP crossover

**October 9, 2026.** Protected V100 [run #37869007809](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37869007809) **SUCCESS**, source `558410e389d8e17439ed6d6556dc5d837b2cd659`, [artifact #11590780942](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37869007809/artifacts/11590780942) (257,182 bytes compressed, 2,493,087-byte evidence directory). All 18 fresh-process experiments completed successfully and passed per-path prefix-replay, request accounting, thermal, and MTP drafting checks.

**Question:** Can existing `NINFER_V100_PREFILL_EXPERT_POLICY=auto` + `NINFER_V100_PREFILL_STREAM_MIN_TOKENS=256` retain GPU streaming's large-chunk advantage without its regression on short prefills?

## Uninstrumented production HTTP median (three fresh processes per policy/MTP; seconds)

| Policy | MTP | Cold TTFT | Cold prefill tok/s | Prefix replay TTFT | Continuation TTFT | Continuation replay TTFT | Continuation decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Grouped CPU cache | off | 14.873 | 82.515 | 0.401 | **1.680** | 0.405 | 8.897 |
| GPU stream grouped (all chunks) | off | 12.849 | 95.516 | 0.790 | 4.296 | 0.744 | 8.940 |
| **Auto256 + grouped CPU** | off | **12.635** | **97.128** | 0.404 | 1.816 | 0.407 | 8.953 |
| Grouped CPU cache | on | 15.097 | 81.293 | 0.398 | **1.666** | 0.396 | **8.242** |
| GPU stream grouped (all chunks) | on | 13.045 | 94.083 | 0.778 | 4.224 | 0.792 | 8.138 |
| **Auto256 + grouped CPU** | on | **12.818** | **95.749** | 0.400 | 1.867 | 0.398 | 7.048 |

## Interpretation

- Auto256 improves cold TTFT versus grouped CPU by **15.0% without MTP** (14.873 -> 12.635 s), and **15.1% with MTP** (15.097 -> 12.818 s). Cold prefill throughput improves 17.7% (82.515 -> 97.128 tok/s) without MTP. It is also approximately 1.7% faster cold than always-stream for both MTP settings; this small difference warrants caution.
- Auto256 preserves fast short-prefix replay (0.404 vs grouped CPU 0.401 s without MTP; 0.400 vs 0.398 s with MTP), whereas always-stream nearly doubles replay TTFT (0.790/0.778 s).
- Auto256 still incurs a **continuation penalty** of 8.1% without MTP (1.816 vs 1.680 s) and 12.1% with MTP (1.867 vs 1.666 s), but it avoids the much larger 4.296/4.224 s always-stream continuation TTFT.
- With MTP, auto continuation decode is slower (7.048 vs grouped CPU 8.242 tok/s, -14.5%). The adaptive arm also generated a **different continuation output** and saw 33 accepted speculative tokens out of 87 drafted, vs grouped CPU 37/75. The decode difference therefore cannot be attributed solely to kernel overhead.
- All three policies have identical reported per-layer expert cache allocation (8,494,252,032 bytes), 64 device expert slots/layer and 32 CPU expert workers; BF16 KV, BF16 projection/SV7 tensor-core OFF, no CUDA graphs, one concurrent request, 4096-token configured context, `--prefill-chunk 2048`, 1,227-token first user prompt and a 64-token completion. Every arm used the identical resident model and expert profile; no model writes.
- The three-process cold-TTFT ranges were: no-MTP grouped CPU 14.832–15.469 s, always stream 12.774–12.986 s, auto256 12.627–12.652 s; MTP grouped CPU 14.692–15.945 s, always stream 13.015–13.052 s, auto256 12.813–12.863 s.
- **Numerical caveat:** no-MTP grouped CPU and always-stream produced the same greedy response signatures; auto256 diverged on both cold and continuation outputs. With MTP, all three agreed on cold outputs but auto256 diverged on continuation. Differences were **reproducible within each path** over three processes; all cross-policy exact-signature comparisons failed. This is a diagnostic finding, **not an independent Phase11 qualification**; do not change numerical thresholds or promote auto256 to a default.
- MTP first request drafted/accepted: grouped CPU 77/36; stream 71/38; auto256 71/38. MTP continuation drafted/accepted: grouped CPU 75/37; stream 75/37; auto256 87/33. This supports treating the observed MTP decode gap as mixed numerical/sampling-path plus performance, not a pure throughput comparison.

## Experiment integrity and restrictions

`tools/diagnostics/v100_sv2_serve.py --moe-policy-screen --repeats 3`; arm order reversed on middle repeat; both MTP settings in each repeat. Per-arm same-path greedy/replay consistency, model/cache accounting, GPU free >=28,000 MiB at launch, thermal and artifact guards passed. No stage ledger or invasive telemetry. Production defaults remain unchanged, as do /opt/ai/strata and the resident model. No unrelated processes stopped, no runner churn. Artifact intentionally excludes model symlinks.

**Decision:** Retain auto256 as **experimental opt-in**. It is an end-to-end performance win for the measured 1.2K cold prompt but not established for long prefill, alternative cache capacities, real concurrency, or model numerical accuracy. Do not promote until independent full-model Phase11 acceptance.

**Next performance-only work:** Prioritize SV6 PLE I/O/overlap for the dominant MoE/CPU bottleneck while retaining grouped CPU and auto256 as paired performance controls. Any further adaptive scheduling sweep should be bounded to a handful of thresholds and at least 4K–8K prefill, using a safety-reviewed context/memory budget. Before default adoption, investigate mixed CPU/stream output divergence with existing independent Phase11 numerical gates (a narrow targeted qualification, not a general quality campaign). SV5 and SV7 remain opt-in.
