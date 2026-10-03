# Later upstream changes relevant to Qwen3.6 27B NVFP4

Reviewed Neroued/ninfer master at d44ab58408aa389728cd8b1ee50179527e1f3e0d against the tested geoffwatts/ninfer-v100 reference b37d0dd3e1163b9d802d8bccfa89918bf68d793e. Their common ancestor is b88c0f6fc7e999f13eb2fcf7fc9105ed79a91868 (September 8, 2026, host-upload completion fix); 85 upstream commits follow that ancestor and are absent from the original reference's ancestry. Review used the upstream-only commit list, target-path history, and relevant implementation patches.

## Findings

No reviewed later upstream change establishes a fix for the intermittent single-request NVFP4 scoring/eager-repeat failure in the original reference. This is a negative finding from the reviewed history, not proof that no inherited defect exists.

| Change | Relevance to the reproduced failure |
|---|---|
| [Accurate NVFP4 SiLU restoration](https://github.com/Neroued/ninfer/commit/c4ae8a9c60f98b6f24731e0954c88fa30dfb8d5b) | Reverses approximation introduced by later commit 05507ab. Changes the W4A4 TMA epilogue. The tested original and candidate Volta paths use their existing SM70 projection/SiluMul path and do not contain that newer approximation. No backport needed for this failure. |
| [FP8 attention reorganization](https://github.com/Neroued/ninfer/commit/4e8939d646de1872e15e80d77448c0de2f72a1ae) and [INT8 attention reorganization](https://github.com/Neroued/ninfer/commit/20a36378ef977ebf32acb82c3d8dc1d5d40b8e6d) | Add explicit full FP8/INT8 KV append handling for multiple independent batch rows, among other tuning. Their added dispatch requires batch size greater than one. The reproduced scoring and ordinary generation checks use one request; this added branch does not explain those failures. |
| [BF16 attention graph stabilization](https://github.com/Neroued/ninfer/commit/98ba2dacd485b600d5e575dd5888a29de20c5670) and [attention/graph planning alignment](https://github.com/Neroued/ninfer/commit/a012e2bc5bb8d13f4c5b33b619ca954f8af701d9) | Restructure later attention schedules and graph profiles. The failure includes ordinary eager execution; generation uses INT8 KV and scoring uses FP8 KV. Not evidence of a direct fix for this reproduction. Importing their graph profiles without their corresponding kernels would be incorrect. |
| [DFlash prefill chunk bindings](https://github.com/Neroued/ninfer/commit/4201b5d2d0f6afe235f4ed8e70cd753800770eca) | Fixes draft state-slot/KV-row controls. Qwen3.6 27B NVFP4 is tested with ordinary execution and MTP, not DFlash. Keep this finding separate from the dense NVFP4 failure. |
| [Q4/Q5 projection bands and race correction](https://github.com/Neroued/ninfer/commit/9e163eee4b8acec21ab0ac765107b6a3f287b217) | Fixes a warp synchronization hazard in the row-block kernel introduced by that same later change. It does not establish that the original NVFP4 path has that race. |
| [Native NVFP4 A16 decoding](https://github.com/Neroued/ninfer/commit/1cfdb4d6b8de11fffcfc1d47e676dfe6a99431e9) | Adds CUDA 13.2/PTX 9.2 native pair conversions for the newer GPU path. Not usable as a V100/CUDA 12.8 fix. The current port restores the original CUDA software packed decoder instead. |
| [V3 bound model runtime](https://github.com/Neroued/ninfer/commit/04350ba94c203833598ba1a41943031c468208f1) | Broad artifact/loader/runtime restructuring, following v3 conversion and loading changes. Not a documented targeted NVFP4 repeatability fix and not a drop-in patch for the pinned v2 artifacts. |
| [Two-stage GDN prefill](https://github.com/Neroued/ninfer/commit/0784e76f647a62d548b7c398c297e99744a8c536) | Replaces the later chunked implementation, with numerical/performance qualification on RTX 5090. Potential future performance work, not a demonstrated correction of this V100 reference failure. |

The older [aliased-state ownership fix](https://github.com/Neroued/ninfer/commit/b87867513f65899cc6ebf5982a95bb69107ec4d9) and [speculative terminal KV-coverage fix](https://github.com/Neroued/ninfer/commit/03177b910e70f783b00c4f980ce0d1896a6b8592) are already ancestors of b37d0dd (verified by GitHub ancestry comparisons). They are not missing backports.

## Qualification approach

Preserve the live original's scoring and repeated-eager outputs and failures. Compare the current candidate independently with original exports from successful run 37141867858, checking identical published model metadata, the isolated ordered split-K reference patch, original/corrected archived scoring agreement, all same-mode generated tokens, and candidate eager repeatability. Use a freshly built unchanged original benchmark for performance. An archived parity result does not turn a live-reference failure into a live-reference pass.

The upstream findings do not justify changing production arithmetic merely to make an unstable reference repeat. Further fixes require a reproducible failing operation or relevant runtime transition.

## Current recheck result

[Run 37155769244](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37155769244) passed candidate scoring, all same-mode generation, and eager repetition against the archived qualified original. Fresh unchanged-original scoring also matched exactly. The live corrected original's graph output differed by 62 tokens; that diagnostic remains failed and is preserved separately. The upstream review does not establish a cause for that instability.

Fresh original/candidate generation measured 19.12/19.13 tok/s for ordinary graphs, 43.40/43.47 for MTP K=3, and 36.44/36.37 for MTP K=7. Prefill and generation remained within 0.6% in every tested mode. Restoring the original packed CUDA software decoder therefore resolves the measured candidate NVFP4 throughput regression on this workload. Workload: V100 32GB, CUDA 12.8, INT8 KV, one request, 1024 prompt tokens, 64 committed output tokens, one warmup and two repetitions.
