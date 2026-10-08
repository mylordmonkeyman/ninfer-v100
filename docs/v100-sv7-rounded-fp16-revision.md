# SV7 revision — BF16-to-FP16 Volta tensor-core eligibility

**Status:** revised specification, not permission to promote an unqualified kernel. 2026-10-08.

Supersedes the **whole-tensor exact-only eligibility rule** in §13.2 of the original Strata-V100-Derived Performance Enhancement Specification for NInfer. Retains the original §13.7 Phase 11 numerical gates without relaxation. The complete revised source specification is preserved separately.

## Conversion classes and candidate policy

On model load/candidate preparation, classify **each represented BF16 element** of intended weights, and separately the activation domain. Record per-tensor counts, maximum absolute and relative rounding error, and change in signed-zero count:

- **Exact finite:** BF16 and FP16 represent identical real values, including signed zero. Keep existing `exact_bf16_to_fp16` and exhaustive checker as an independent strict-exact control.
- **Rounded finite:** BF16 is within FP16 finite range but not exactly representable, including subnormal underflow and round-to-signed-zero. Convert with round-to-nearest, ties-to-even; record total changes and nonzero-to-zero.
- **Finite overflow:** magnitude >65504; a separate explicitly opt-in finite saturation policy may clamp to ±65504, and records every clamp. Never silently convert finite BF16 to FP16 infinity.
- **NaN/Inf:** reject the experimental candidate or use the existing BF16 fallback until a separately specified and qualified nonfinite policy exists. Do not silently clamp or reinterpret nonfinite values.

**A tensor containing rounded finite elements is no longer automatically rejected.** A tensor containing clamped finite elements is only eligible for an explicitly labeled saturating experiment. Conversion class statistics do **not** equal model accuracy qualification. Strict-exact mode remains available as a diagnostic control.

## Volta implementation constraints

Volta lacks native BF16 tensor cores. Use separate opt-in FP16 tensor-core GEMM (`CUTLASS OpClassTensorOp` / Volta FP16 MMA), FP32 accumulation, explicit BF16 output as required by downstream stages, and unchanged BF16/SIMT fallback. Prefer replacing a device weight copy to permanent duplicates; preserve original host artifact and other BF16 consumers, bound staging/activation/accumulator VRAM, and report conversion cost separately from prefill. Do not implement an SM80-only `cp.async`/`ldmatrix` dependency.

Candidate order: QSA indexer, PLE key/value, shared expert, MTP BF16 projections. Benchmark each shape independently, including T=1,2,4,8,32,128,512,1024,4096,8192, and representative 1K–32K prompts where feasible. Do not hardcode a dispatch crossover before measurements; keep tiny-T decode kernels where faster.

## Hard acceptance gates (unchanged)

1. Exhaustive 65536-pattern BF16 converter tests: RNE tie/subnormal boundaries, signed zero, overflow, NaN/Inf; keep existing strict-exact oracle unchanged.
2. Conversion inventory per real artifact tensor (exact/rounded/overflow/nonfinite counts; magnitude errors) with model/weights identity.
3. Independent numerical oracle/FP64 kernel checks, router decisions, full-model Phase 11 thresholds, long-prefix/teacher-forced and MTP tests where relevant. **Never relax acceptance thresholds to justify performance.**
4. Paired cold/warm performance against unchanged BF16 baseline: HTTP TTFT, prefill and decode, startup conversion cost, memory peak, repeated baseline, same model and quantization. One protected V100 hardware job at a time.
5. Production defaults remain BF16/SIMT until both numerical and repeatable end-to-end speed gates pass. No model artifact replacement, installed Strata modifications, Phase18/SV8, or unrelated process termination.

## Implementation sequence

First implement a deterministic finite converter policy/inventory and host test without changing dispatch. Next stage a narrowly scoped **opt-in** FP16 tensor-core candidate for a small real projection with unchanged baseline. Qualify against independent oracles before the first full-model performance comparison. The previously reported exact-FP16 eligibility inventory (only 8/749 tensors wholly exact) is now used to establish rounding/overflow prevalence, **not** to reject nearly every tensor.

References: `src/ops/common/exact_bf16_fp16.h`, `tests/ops/test_exact_bf16_fp16.cpp`, `tools/diagnostics/v100_sv7_weight_eligibility.cpp`, and `src/ops/linear/bf16/bf16_dispatch.cpp`. Strata-V100 `src/prefill/gemm.cu` rounds/caps on Volta; geoffwatts/ninfer-v100 uses BF16 SIMT for its SM70 BF16 linear path.
