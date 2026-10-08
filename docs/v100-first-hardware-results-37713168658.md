# First real V100 NInfer / Strata telemetry qualification — run 37713168658

Source: https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37713168658
Artifact: `v100-telemetry-qualification-37713168658-1` (GitHub artifact ID 11523505207, 19,607,101 compressed bytes).
Frozen hardware code: NInfer `5a06ed65ca654bf7a91711905c8b25c8d3781258`; instrumented Strata `98cbf622124ab25f89529ff0f56eebbf9fdfd5fd`.
This is a **qualification** result, not a performance comparison across model quantizations.

## What actually passed

Both CUDA 12.8 / SM70 builds and NInfer's native expert-cache/stream fixtures at levels 0, 1 and 2 passed.
All six real inference cells completed: NInfer and Strata, each at telemetry levels 0, 1, 2, two warmups and three measured 64-token responses. The native round/lifecycle/draft validators completed; the bounded evidence upload succeeded without including the model.
Hosted software run 37713168684 passed its CPU checks and CUDA host syntax checks.

## Measurements: HTTP wall time, NOT matched-engine throughput

| Engine | Level 0 median (s) | Level 1 median (s) | Nominal level 1 overhead | Fidelity across levels | Overhead gate |
| --- | ---: | ---: | ---: | --- | --- |
| NInfer | 14.1026 | 14.4915 | +2.7574% | exact within and across 0/1/2 | unqualified (>2%) |
| Strata | 4.7325 | 5.2992 | +11.9752% | **baseline already variable** | unqualified (timing drift) |

NInfer output hashes were identical for all 15 warmup+measured responses. Strata produced two distinct 64-token reasoning-only completions even at level 0 (the alternative differs only in its last short reasoning span); the same two hashes were observed at level 1, while level 2 happened to produce just one. This **does not establish instrumentation-caused output drift**. It also does not prove instrumentation preserves outputs: an independent deterministic/fixed-token fidelity gate is still needed. The request specified greedy, seed 42, and `enable_thinking=false` for both, but Strata still returned `reasoning_content`, not `content` for all 64 tokens; request-level semantic parity must be resolved before claiming meaningful output comparisons.

The three measured request wall times, in seconds:

| Engine, level | First | Second | Third | First-to-last drift |
| --- | ---: | ---: | ---: | ---: |
| NInfer 0 | 14.04 | 14.30 | 14.10 | +0.47% |
| NInfer 1 | 14.49 | 14.59 | 14.36 | -0.88% |
| NInfer 2 | 15.54 | 15.06 | 15.52 | -0.12% |
| Strata 0 | 5.05 | 4.70 | 4.73 | -6.33% |
| Strata 1 | 5.97 | 5.30 | 5.17 | -13.47% |
| Strata 2 | 5.64 | 5.01 | 5.01 | -11.26% |

Strata's per-request expert GPU-cache hit rate kept growing rather than reaching equilibrium:
- Level 0: 75.0%, 86.2%, 90.9%, 93.1%, 93.3%
- Level 1: 74.5%, 86.2%, 90.0%, 93.0%, 94.3%
- Level 2: 75.0%, 86.4%, 91.0%, 93.5%, 94.0%

The two warmups were inadequate to eliminate residency/page-cache trends. In addition, Strata generated a different number of accepted MTP drafts in otherwise identical prompts. The apparent 12% overhead **cannot be isolated** from warming, work differences, timing variance and request semantics.

Native log volume for just five requests was 79,235,063 bytes NInfer level 1, 153,587,500 bytes NInfer level 2, 19,261,434 bytes Strata level 1, and 28,566,474 bytes Strata level 2. Emission / serialization / filesystem overhead requires isolation. The level 1 <2% target must not be relaxed to excuse this.

## Follow-up gates

1. Preserve strict fidelity gating. Distinguish nondeterministic level-0 output from newly observed output identities at levels 1/2; report per-cell measured trend and confounding instead of labeling every mismatch telemetry-induced. Added in diagnostic harness and regression tests after this run; **no GPU rerun was triggered**.
2. Validate fixed-token / deterministic (non-speculative control) output equivalence before attributing Strata's variations to instrumentation or MTP. Test native MTP separately; do not change production quantization, cache policies or worker counts to manufacture parity. State what is unobservable if a fixed-token replay API does not exist.
3. Stabilize Strata expert residency before measuring level 1 overhead, with measured plateau checks and a balanced order of levels on equivalent cache states, or report not-qualified. Compare matched request work rather than conflating 64 accepted tokens with 64 verification operations.
4. Reduce or isolate level-1 telemetry serialization/logging cost without losing required counters, then test offline invariants and real-machine overhead in one planned matrix.
5. After fidelity and overhead gates pass, proceed to a complete request/lane/token correlation and GPU launch/interval attribution. No optimization or Phase 18/SV8 is justified by these timings.

Original model and installed Strata checkout were untouched. Do not recreate the cancelled oversized artifact symlink inside any uploaded results folder.
