# V100 stream-slot reuse A/B — October 8, 2026

**Hardware:** [run 37803809277](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37803809277), artifact `v100-ninfer-stream-reuse-ab-37803809277-1` (ID `11562292699`), source `093917eb796e6f2c7b719d7754ffebe17fda997d`. Hosted checks [37803809078](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37803809078) passed.

Three serial NInfer-only cases use the same V100, mixed-quant 113 GB resident model, 156 persistent GPU expert slots/layer, LRU+auto256 prefill policy, four streaming slots and MTP draft1. Five warmups; three 585-input/64-output long requests and three 63-input/64-output short requests per cell; greedy output.

| Mode | Long HTTP s | Native prefill s | Native decode s | Short HTTP s | Short native prefill s |
|---|---:|---:|---:|---:|---:|
| Blocking reuse baseline | 9.5583 | 6.6709 | 2.9097 | 3.9944 | 0.8840 |
| Pipelined reuse | 9.4659 | 6.5712 | 2.8927 | 4.0060 | 0.8902 |
| Blocking reuse repeat | 9.6012 | 6.6827 | 2.9134 | 4.0079 | 0.8891 |

Pipelined reuse improves long HTTP time by only 0.97% versus initial and 1.41% versus repeat blocking controls, with ~0.10 second lower native prefill. Short latency is unchanged. **Do not promote to default**; keep four-slot blocking reuse as baseline. Mathematical GPU expert oracle passed under pipelined reuse (NRMSE approximately 1e-7 to 4e-7, cosine 1.0).

Each case transferred the same approximately 194.55 GB of expert weights across the instrumented prefill rounds and observed 82,277 distinct expert misses. Long prompt weight H2D ~24.3 GB/request. Pipelining reduces host sync pressure but does not eliminate weight traffic, the more promising remaining target.

## Next bounded experiment: profile-prior-50 + auto256

Compare three serial cells only: `ninfer/lru-prefill-auto256` baseline, `ninfer/profile-prior-50-auto256` candidate, `ninfer/lru-auto256-repeat` control; retain exact frozen model, quantization, 156 cache slots/layer, 5 warmups, 3 measured long and short requests and 64 output tokens. Existing verified profile is `sv2-static-profile.json`.

The candidate combines `NINFER_V100_EXPERT_POLICY=profile`, `NINFER_V100_EXPERT_PRIOR_WEIGHT=50`, `NINFER_V100_PREFILL_EXPERT_POLICY=auto`, `NINFER_V100_PREFILL_STREAM_MIN_TOKENS=256`, and `NINFER_V100_DEVICE_ROUTE_COMBINE=1`. Previous profile-prior alone showed ~58.6% prefill GPU residency (versus 43.3% LRU) and estimates suggest ~10% fewer distinct misses / ~2.5 GB less long-prompt H2D. **Neither the combined transfer saving nor its latency improvement has yet been demonstrated.** GPU streaming suppresses persistent-cache admissions; check actual H2D bytes and cache hit ratios as well as native prefill/decode and HTTP latency. Retain LRU+auto256 default unless benefit exceeds baseline drift without material short-prompt regression.

No new quality campaign, telemetry expansion, or changes to the installed Strata or resident model. Require successful hosted checks before launching a single protected hardware run.
