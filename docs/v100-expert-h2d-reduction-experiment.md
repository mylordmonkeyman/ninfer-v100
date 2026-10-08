# V100 expert H2D reduction: combined profile and hybrid CPU/GPU prefill A/B

**Objective:** Reduce nonresident expert transfers, currently **24,319,013,152 bytes (22.65 GiB) per 585-token prefill** on the LRU+auto256 benchmark, without reducing the persistent V100 cache or modifying quantization.

## Measured baseline

Parsed first-party `ninfer-strata-v100-telemetry-v1` records in the completed [stream reuse benchmark 37803809277](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37803809277), 8 repetitions of a 585-token prefill (5 warmup + 3 measured); 48 layers x 10 paths/token:

- `280,800` total routed paths.
- Mean `120,757` resident hits (**43.0%** of paths).
- Mean `160,043` nonresident GPU paths (**57.0%**).
- Mean `24,319,013,152` expert staging H2D bytes (**22.65 GiB**).
- Mean prefill host wall time 6.73 seconds blocking; 6.57 seconds pipelined reuse.

All-GPU prefill streams entire nonresident expert pairs, one canonical NVFP4 transfer per **distinct missing expert within each layer**. Each expert upload is `kExpertSlotBytes=2,764,800` bytes, not a tensor execution cache hit.

## New optional fraction policy

```bash
# No change to default behavior: missing env means 1.0 (all nonresidents on GPU).
export NINFER_V100_PREFILL_EXPERT_STREAM_FRACTION=0.9  # or 0.75
```

When prefill streaming is active (under existing `NINFER_V100_PREFILL_EXPERT_POLICY=auto` with 256-token threshold), group nonresident routes by expert ID; sort groups descending by number of routed tokens, with expert ID as deterministic tie-break. Stream the highest-work 90% or 75% of **distinct nonresident experts** on the V100, while the remaining groups use NInfer's existing grouped AVX2 worker pool. GPU cache-hit experts continue using the persistent resident GPU cache. All outputs retain the original token/path association for the GPU route-combine kernel.

The prefill fraction does **not** alter decode policy (`NINFER_V100_DECODE_EXPERT_STREAM_FRACTION` remains separate), model weights/quantization, persistent GPU expert slot budget, or four-slot staging ring. Default `1.0` retains original all-GPU prefill. The policy bounds parser rejects nonfinite, zero, negative, >1, and malformed inputs.

**Tradeoffs:** A smaller GPU fraction saves approximately the corresponding share of nonresident expert H2D *by distinct group count*, but adds CPU AVX2 computation, CPU input D2H, grouped result H2D, and likely more small result-copy launches. It might slow prefill; no gain is assumed. The 90% and 75% test points are intentionally limited and are not a basis for an unconstrained parameter sweep.

## Single five-cell hardware A/B

Workflow: `.github/workflows/v100-ninfer-profile-stream-ab.yml` (one protected V100 runner, sequential full server starts):

1. `lru-prefill-auto256`: original all-GPU (fraction 1.0).
2. `profile-prior-50-auto256`: profile-informed resident cache policy at fraction 1.0. This is a separately motivated way to reduce misses.
3. `lru-auto256-gpu90`: fraction 0.9, default LRU cache.
4. `lru-auto256-gpu75`: fraction 0.75, default LRU cache.
5. `lru-auto256-repeat`: original all-GPU control to bound drift.

Same validated resident expert profile, exact resident model and quantization, `156` cache slots/layer, MTP draft1, four-slot staging, device route combine, 5 long warmups, 3 measured long (585 input/64 generated) + 3 short (63 input/64 generated). Native JSON lifecycle records already include `expert_h2d_bytes`, `cpu_routes`, `cpu_weight_read_bytes`, `resident_routes`, and `nonresident_gpu_routes`; analyze actual bytes and **latency** rather than predicted savings.

**Prerequisites:** Hosted Python/test and CUDA host syntax checks pass for implementation. One V100 GPU run at a time; cross-workflow concurrency `v100-sv0-hardware`, no cancellation of healthy jobs, no unrelated process killing. Before each cell require at least 28 GiB free; model symlink outside uploaded artifact; 1GiB evidence-upload ceiling.

**Promotion:** Favor a variant only if measured long prefill/HTTP latency improves beyond baseline drift without a material short-prompt regression. Otherwise retain LRU+auto256 defaults. No quality-testing campaign is part of this performance experiment.
