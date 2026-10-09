# V100 NInfer 7K (four prefill chunks) CPU worker-count performance screen

**Purpose:** Validate whether the reproducible 3K win from 64 versus 32 AVX2 CPU expert workers extends to larger, multi-chunk prefill; use performance data, not a broad model-quality campaign.

[Prior 3K worker A/B](v100-moe-cpu-workers-3k-results-37882498281.md) found cold TTFT improvements of **4.03% without MTP / 3.68% with MTP**, and **28.9% / 52.5%** higher cold decode tok/s. The V100 route counts and all GPU expert H2D work were identical. 64 workers is the experimental reference, not a global default.

**New isolated screen:** `.github/workflows/v100-ninfer-moe-cpu-workers-7k-http.yml` invokes `tools/diagnostics/v100_sv2_serve.py --cpu-workers-long-screen --repeats 3`. The measured first prompt uses 360 fixed records, expected 6,400–7,600 actual tokenizer tokens, across **four** 2,048-token prefill chunks, in an **8,192-token configured context**. Prefix replay, continuation and continuation replay follow the same four-request production HTTP sequence.

| Variable | Arm A | Arm B |
|---|---|---|
| `NINFER_FLASH_NEXT_CPU_EXPERT_WORKERS` | **32** | **64** |
| Cache | static identity-bound profile, 64 experts/layer | same |
| Prefill policy | `auto256`, `minroutes20`, grouped CPU fallback | same |
| Model/quantization | frozen original mixed NInfer Qwen3.8 Flash-Next | same |
| Context / prefill chunk / KV | 8192 / 2048 / BF16 | same |
| Decode | MTP off and MTP on (draft3), in separate arms | same |
| Repeat | 3 fresh servers for each worker/MTP setting | same |

There are 12 independent timed server processes and **two separate diagnostic servers** (one per worker count). The diagnostics must publish exactly **192 expert-layer records** (4 chunks × 48 MoE layers) per arm with prefill chunk length >=256. Confirm identical expert H2D bytes, CPU miss routes, CPU weight-read bytes and cache size before attributing differences to worker scheduling.

**Hard integrity gates:** verify 6,400–7,600 actual prompt tokens, prefix-replay accounting, same-policy response signatures, MTP backend/drafted tokens when enabled, fixed cache budget, thermal status and <=28GiB V100 headroom before launches. Different worker counts should preserve identical deterministic response signatures, but Phase11 model numerical qualification is separate. All results are self-hosted source-locked, with artifact upload refusing symlinks and >=1GiB folders.

**Success:** If 64 workers materially improves cold TTFT and/or decode on this four-chunk workload without important prefix/continuation regression, retain it as the experimental worker-count reference and consider a bounded NUMA locality investigation. If it loses, prefer a context-sensitive worker setting only after a distinct controlled test—not a heuristic based on these two points. Do **not** change production defaults, installed Strata or the original model.
