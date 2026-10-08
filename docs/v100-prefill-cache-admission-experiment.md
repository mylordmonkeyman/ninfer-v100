# V100 prefill hot-expert cache adaptation (opt-in performance experiment)

The current LRU + auto256 streaming path intentionally suppresses cache admissions during long prefill. The cache starts from the validated 156-expert-per-layer resident profile, and 585-token prefill transfers about **22.6 GiB of nonresident expert weights** even after warmup. Those bytes are repeatedly transferred for the same prompt because the transient GPU streamer does not change persistent cache residency.

## New opt-in behavior

```bash
export NINFER_V100_PREFILL_STREAM_ADMIT=hot
```

For each prefill MoE layer in which GPU expert streaming is active:

1. Count only **nonresident** expert routes, which are already recorded in the per-expert `streamed_routes` groups.
2. Choose the expert with the largest miss count (stable lowest-ID tie break).
3. Call the existing bounded `FlashNextExpertCache::admit` for **one** expert after resident hit work completes. This uses the normal cache worker, LRU eviction safeguards and existing memory budget.
4. The admission can improve residency for later requests. Current prefill routes have already been assigned and do not change.

The default is `off`; the code does not modify quantization, resident cache slot count, stream ring, CPU backend, MTP or decode policy. It neither changes unrelated CPU/short-prefill cache admissions nor bypasses the existing cache worker's bounded queue, upload/lease safety or deduplication.

**Why opt-in:** These extra background cache fills compete with GPU expert streaming for PCIe bandwidth during initial warmup. Admission may replace some initially profiled expert weights that still matter to other prompts; performance may worsen. Only repeated measured long prompts with enough adaptation time can establish a win. This is *not* a new default or a tested performance claim.

## Follow-on controlled V100 run (prepared, not launched)

Workflow: `.github/workflows/v100-ninfer-prefill-admission-ab.yml`. It compares three serial NInfer-only cells under the same model/quantization, 156 GPU slots/layer, four-slot stream ring, MTP=1, five warmups, three 585+64 long and three 63+64 short measured requests:

| Cell | GPU/CPU prefill policy | New stream admissions |
|---|---|---|
| Baseline | `auto256`, GPU fraction `0.50`, `NINFER_V100_CPU_EXPERT_GROUP=prefill` | Off |
| Experiment | **identical** | `hot` |
| Repeat | Same baseline | Off |

Comparisons must include total HTTP time, native prefill/decode, resident GPU route share, per-request `expert_h2d_bytes`, `admissions_total` and queue-declined/fill counters, plus baseline-repeat drift. The candidate is useful only if uploads fall *and* wall time improves; more admissions alone are not success.

Host-only deterministic policy tests live in `tests/targets/qwen3_8_flash_next/test_stream_admission.cpp`, and the workflow builds and exercises the usual GPU cached/streamed expert fixtures before loading the full model.

**Sequencing:** Finish and analyze [route-count threshold run 37835323235](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37835323235) first. Require all hosted software/CUDA checks to pass. Only then trigger the single GPU workflow; retain the cross-workflow `v100-sv0-hardware` lock, no automatic job cancellations, 28GiB GPU free guard, model alias outside artifacts, and 1GiB upload ceiling. Preserve the original model and installed Strata.
