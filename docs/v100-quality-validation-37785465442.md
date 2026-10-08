# V100 NInfer expert-policy quality: initial smoke test passed

**Hardware run:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37785465442  
**Evidence artifact:** `v100-ninfer-quality-37785465442-1` (ID `11554328671`, 37,532 bytes).  
**Hosted checks:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37785465332 — software and CUDA-host syntax both passed.  
**Frozen source:** `e426c75afcf87f1f53d3dfe2212b237408f0c197` on `perf/v100-strata-derived`.

The existing resident NInfer Qwen3.8 Flash-Next mixed-quant model was used without alteration. Cache allocation remained 156 slots per layer. Each policy ran two warmup requests, then two repetitions of eight fixed objective tasks (four short, four long): 16 measured answers per policy, 48 total. Greedy decode, seed 42, 96 maximum generated tokens, no thinking, no prefix reuse. Long prompts were 941–955 tokens and short prompts 26–44 tokens.

| Policy | Correct / measured | Short | Long | Short median HTTP | Long median HTTP |
|---|---:|---:|---:|---:|---:|
| Static profile cache | 16/16 | 8/8 | 8/8 | 1.37 s | 14.99 s |
| LRU adaptive cache | 16/16 | 8/8 | 8/8 | 1.21 s | 13.61 s |
| **LRU + auto256 GPU prefill streaming** | **16/16** | **8/8** | **8/8** | **0.89 s** | **9.57 s** |

All **16 paired measured answers were byte-identical across all three policies**. There were no correctness regressions or output-hash differences on this particular suite. This is a more favorable result than the earlier 64-token general prose speed benchmark, where adaptive-cache output hashes sometimes differed. Different hashes alone would not constitute a correctness failure.

The policy's performance improvement agrees with the independent 64-token A/B benchmark [run 37769284560](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37769284560): original static 13.80 s long / 6.41 s short; LRU+auto256 9.65 s long / 4.08 s short. Those A/B figures are the primary performance result; this quality smoke suite uses variable output lengths and should not be treated as a precise throughput benchmark.

## Recommended **opt-in** V100 settings

Use on the measured NInfer V100 branch, with the same validated model, quantization, resident expert profile, and memory budget:

```bash
export NINFER_FLASH_NEXT_EXPERT_CACHE=1
export NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS=156
export NINFER_V100_EXPERT_PROFILE=/path/to/validated/sv2-static-profile.json
export NINFER_V100_EXPERT_POLICY=lru
export NINFER_V100_PREFILL_EXPERT_POLICY=auto
export NINFER_V100_PREFILL_STREAM_MIN_TOKENS=256
export NINFER_V100_DEVICE_ROUTE_COMBINE=1
export NINFER_V100_PLE_IO=mmap
```

The `auto` policy selects GPU expert streaming for prefill batches at or above 256 tokens, avoiding the short-prompt overhead of `stream` always. This is **not** a recommendation to change the default for other GPUs, other models, or arbitrary contexts.

## Remaining qualification

This eight-task smoke test does **not** establish general model-quality parity. It exercises arithmetic, literal retrieval, simple JSON and distractor resistance, but not sustained reasoning, code generation, long context, multi-turn conversations, vision, or adversarial inputs. Next, run a fixed broader held-out task suite under the same three policies, record task-level differences rather than require identical hashes, and only then consider a production-default change. Preserve one-GPU-run-at-a-time safeguards and avoid unrelated runner churn.
