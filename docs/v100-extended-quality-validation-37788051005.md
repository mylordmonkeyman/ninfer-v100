# V100 extended objective quality: cache-policy arithmetic divergence

**Run:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37788051005  
**Artifact:** `v100-ninfer-quality-extended-37788051005-1`, ID `11556860271`.  
**Commit:** `7a55a194a099f45ad59506711cd6fd6fbe037072` on `perf/v100-strata-derived`.  
**Hosted software/CUDA checks:** runs `37787927934` and `37787907373`, all passed.

20 objective tasks x 2 measured repeats x 3 policies = **120** answers. Same frozen Qwen3.8 Flash-Next model, quantization, seed, 156 expert cache slots/layer, and validated resident profile.

| Policy | Overall | Short (10 tasks x 2) | Long (10 tasks x 2) | Short median HTTP | Long median HTTP |
|---|---:|---:|---:|---:|---:|
| Static seeded cache | 38/40 | 18/20 | 20/20 | 1.178 s | 14.850 s |
| LRU adaptive cache | 36/40 | 16/20 | 20/20 | 0.993 s | 12.981 s |
| LRU + auto256 GPU prefill streaming | 37/40 | 17/20 | 20/20 | 0.727 s | 9.680 s |

**Shared failure:** all three policies answer `3` for the second smallest of 9, 3, 14, 5; correct answer is `5`. This is a common model/task failure, not evidence of policy regression.

**Policy-dependent failure:** `1001 % 9` (correct `2`). Static: `2, 2`; LRU: `5, 5`; LRU+auto256: `2, 5`. All other 18 tasks pass in all six repeated policy/task combinations. No failure occurred in the 20 long-prompt answers under any policy.

The modulo answer differences are an **actual correctness difference** on this small suite; changed hashes alone are not the criterion. They could reflect CPU vs GPU expert arithmetic, speculative decoding numerical differences, or cache state/order. They do **not** yet establish systematic regression or a general quality difference; only two repetitions per policy were measured, and the policies ran sequentially with distinct warmup/cache histories.

The previous eight-task suite [run 37785465442](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37785465442) had 48/48 correct across all three policies, with all 16 paired answers byte-identical.

## Next controlled qualification

Run a **small targeted repetition** of modulo and sorting questions under static, LRU and auto256, with a repeated static baseline, preserving same model/quantization and all GPU safeguards. Record the complete answer distribution and prompt token counts, not only binary scores. Consider a controlled reset of expert-cache state if differences depend on request order. If modulo regression persists, investigate GPU-vs-CPU expert numerical equivalence and verify logits or teacher-forced reference on the failing prompt. Do not change production default or claim broad accuracy parity before this is resolved. Continue to permit the measured faster policy as opt-in with explicit quality caveat.
