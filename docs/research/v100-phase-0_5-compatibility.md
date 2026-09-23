# V100 / SM70 Phase 0.5 compatibility manifest

Status: initial implementation audit against `mylordmonkeyman/ninfer-v100` branch `forwardport/v100-flash-next` at `6b147eefb7b0ef0517e78823fd75f1184032540d`.

This document records the CUDA-12.8 / SM70 compatibility findings needed before Phase 2 changes the build. It is deliberately separate from the later SM70 implementation: Phase 0.5 identifies the reachable incompatibilities and the intended replacement boundary; it does not emulate unsupported instructions with unsafe compatibility shims.

## Policy

For the V100 build:

```text
architecture: sm_70
CUDA toolkit: 12.8
CUDA Graph:   disabled for initial bring-up
```

The current Blackwell build remains `sm_120a` and retains its current toolkit policy. Do not lower the existing CUDA version requirement globally.

Compatibility classes used below:

```text
A  SM70-compatible as written
B  source is mostly architecture-neutral; dependency/API/runtime qualification required
C  high-level algorithm is reusable; SM70 device backend required
D  newer-architecture implementation must be excluded from the SM70 source set and replaced by dispatch
```

API disposition:

```text
OK12       expected to work unchanged with CUDA 12.8
GUARD      architecture/toolkit guard required
REPLACE12  CUDA-12.8-compatible replacement required
QUALIFY    API exists, but runtime/compile behavior still needs qualification on the V100 build
NO-SM70    implementation must be unreachable/excluded on SM70
```

## Build and source-set boundary

| Location | Finding | Class / disposition | Phase action |
|---|---|---|---|
| root `CMakeLists.txt` | Defaults to `120a`, rejects every other architecture, and rejects CUDA `< 13.1`. | GUARD | Phase 2: make architecture/toolkit policy explicit: `120a` keeps current requirement; `70` targets CUDA 12.8. Do not globally lower the version check. |
| `src/CMakeLists.txt` | Current source lists compile Blackwell-only kernels unconditionally, including a non-RDC TMA/NVFP4 archive. | D / NO-SM70 | Phase 2: architecture-condition source membership. SM70 must not compile TMA/setmaxnreg/native-NVFP4 implementations merely to discard them at runtime. |
| `src/ops/common/memory.cuh` | Includes `cuda_pipeline.h`; directly defines `cp.async`, commit/wait groups and `__pipeline_*` wrappers. | D as an SM70 implementation boundary | Phase 3: add a real synchronous Volta staging substrate with explicit barriers. Do **not** map async waits to no-ops. |
| `src/ops/common/mma.cuh` | Current helpers use `ldmatrix` and newer MMA shapes/types (`m16n8k16` FP16/BF16, s8, TF32, FP8, NVFP4). There is no Volta `m8n8k4` helper here. | D / NO-SM70 | Phase 3: separate Volta `m8n8k4` FP16×FP16→FP32 primitives and fragment layouts. |
| `src/ops/linear/bf16/bf16_gemm_mma.cuh` | Direct `cp_async`, `ldmatrix`, `mma_bf16` dependency. | C | Preserve GEMM semantics; replace backend with BF16-storage→FP16-compute Volta path. |
| `src/ops/common/rowsplit_mma.cuh` + `rowsplit_grouped_mma.cuh` | Staged `cp.async`, `ldmatrix`, BF16 MMA machinery used by grouped integer paths. | C | Add explicit Volta schedule/backend; do not reuse the current pipeline mechanically. |

## CUDA 12.8 / CCCL API audit

### QSA selection

`src/targets/qwen3_8_flash_next/impl/qsa_indexer_kernels.cu` directly includes:

```text
cub/device/device_segmented_radix_sort.cuh
cub/device/device_topk.cuh
cuda/__execution/determinism.h
cuda/__execution/output_ordering.h
cuda/__execution/require.h
cuda/__stream/stream_ref.h
```

Current decode selection uses:

```text
pack_topk_keys_kernel
cub::DeviceTopK::MaxPairs
bitonic_sort_512_descending
```

Disposition:

| Piece | CUDA 12.8 / SM70 disposition |
|---|---|
| `DeviceSegmentedRadixSort::SortPairsDescending` | OK12, compile-qualify under CUDA 12.8 |
| `cub/device/device_topk.cuh` | REPLACE12; unavailable in CUDA 12.8 / CCCL 2.7 |
| `cuda::std::execution::env` + `cuda::execution::*` used only to configure `DeviceTopK` | REPLACE12 with the TopK call; do not create a standalone compatibility layer for this path |
| `pack_topk_keys_kernel` | A; preserve |
| `bitonic_sort_512_descending` | A; preserve |

The packed key is already deterministic for distinct IDs:

```text
high 32 bits = monotonic score rank
low  32 bits = ~id
```

Therefore exact score ties prefer lower IDs. The SM70 selection fallback must preserve this ordering.

Phase-0.5 selection decision: **reuse the existing CUDA-12.8 segmented-radix machinery first**. A custom small TopK remains a later optimization candidate. Do not integrate a newer standalone CCCL by default.

### CUDA Graph API

`src/core/decode_graph.cpp` directly uses graph creation/destruction, instantiate/update, upload, and launch APIs. These core runtime APIs exist in CUDA 12.8.

Disposition: `QUALIFY`, not an API blocker. Initial SM70 planning must select eager/non-graph execution, so graph allocations are zero during bring-up. Graph qualification belongs after eager correctness and launch-overhead measurement.

### Other audited runtime APIs

No direct `cudaMallocAsync` / `cudaFreeAsync`, `cudaMemset2D`, `cudaMemcpy2D`, or stream-priority dependency was found in the audited Flash-Next target files listed below. The Phase-2 CUDA-12.8 compile remains the final proof of the reachable source set.

## Load-time conversion audit

### `impl/load/quantize_nvfp4_expert_bank.cu`

Direct dependencies:

```text
__nv_cvt_float_to_fp8(... __NV_E4M3)
ops::detail::pack_nvfp4_e2m1x16(...)
```

The included `src/ops/linear/nvfp4/nvfp4_codec.cuh` additionally depends on CUDA FP4/FP8 types and PTX E2M1 conversion. This is not a safe CUDA-12.8/SM70 header dependency.

Disposition: `C / REPLACE12` for the SM70 converter. Keep the exact persistent output contract:

```text
expert-blockscale-k16-m128x4-v1
row-major packed E2M1 code plane
per-expert swizzled E4M3FN K16 scale plane
one FP32 divisor per expert
```

The replacement encoder must be software-compatible and bit-tested against the canonical/current conversion semantics. This is load-time conversion only; it does not authorize a new persistent representation.

### `impl/load/quantize_output_head.cu`

Direct dependency:

```text
__nv_cvt_float2_to_fp8x2(... __NV_E4M3)
```

This path covers BF16→row-scaled E4M3 output-head/embedding/proposal-head conversion.

Disposition: `C / REPLACE12`. Provide a software-compatible E4M3FN encoder for the SM70 build, with exhaustive 256-code decode coverage and encoder parity tests for representative/tie/boundary inputs.

### `impl/load/materialized.cpp`

This file is architecture-neutral orchestration but makes the conversion paths reachable:

```text
BF16 token embedding -> FP8 (optional load-time materialization)
BF16 output head      -> FP8
BF16 proposal head    -> FP8
legacy MTP BF16 expert gate/up + down -> NVFP4
```

Disposition: `B`. The file can remain shared, but its called converters must have CUDA-12.8/SM70 implementations. Prefer a native-NVFP4 MTP artifact where available; do not make correctness depend on it.

## Direct instruction / pipeline audit

### Shared helpers

| Source | Direct newer-architecture dependency | Disposition |
|---|---|---|
| `src/ops/common/memory.cuh` | `cp.async`, async group commit/wait, `__pipeline_*` | D / NO-SM70 |
| `src/ops/common/mma.cuh` | `ldmatrix`, BF16/newer FP16/s8/TF32/FP8/NVFP4 MMA | D / NO-SM70 |
| `src/ops/linear/bf16/bf16_gemm_mma.cuh` | `cp_async`, `ldmatrix`, `mma_bf16` | C |
| `src/ops/common/rowsplit_mma.cuh` | explicit cp.async pipeline assumption | C |
| `src/ops/common/rowsplit_grouped_mma.cuh` | `cp_async`, `ldmatrix`, `mma_bf16` | C |

### Flash-Next target files

| Target source / path | Direct finding | Inherited finding | Class | SM70 action |
|---|---|---|---|---|
| `impl/frontend.cpp` | none in audit | orchestration only | A | keep |
| `impl/text_decode_kernels.cu` | none | simple BF16 replication | A | compile/correctness qualify |
| `impl/ple_decode_kernels.cu` | none | local reductions/shuffles | A/B | compile/correctness qualify |
| `impl/gdn.cpp`, `impl/gdn_kernels.cu` | no direct cp.async/ldmatrix/MMA hit in local kernels | projections route through FP8 linear backends | A/B locally, C downstream | preserve local state logic; Phase 4/8 projection backend |
| `impl/hyper_connection*.{cpp,cu}` | no direct hit in local file | generic BF16 GEMM -> cp.async + ldmatrix + BF16 MMA | C | preserve high-level semantics; Phase 6 Volta GEMM backend |
| `impl/moe_route.cu` decode route | no direct newer instruction hit | custom SIMT/warp selection | A/B | preserve deterministic higher-score/lower-ID rule |
| `impl/moe_route.cu` prefill projection | no direct hit in file | shared BF16 MMA backend | C | Volta backend |
| `impl/moe_shared_kernels.cu` grouping/compaction | no direct hit | none for grouping pieces | A/B | retain |
| `impl/moe_shared_kernels.cu` matrix routes | no direct hit | shared BF16 MMA backend | C | Volta backend |
| `impl/moe_kernels.cu` low-T decode | software NVFP4/SIMT portions exist | current NVFP4 codec/header still architecture-sensitive | B/C | reuse arithmetic where possible after software-codec split |
| `impl/moe_kernels.cu` native prefill | direct `cp_async`, `ldmatrix`, `mma_nvfp4_e4m3` | — | D | exclude on SM70; replace at dispatch/backend boundary |
| QSA indexer decode score | SIMT FP32 + warp reduction | no MMA requirement | A/B | retain |
| QSA indexer decode selection | `DeviceTopK` API only | packed-key + bitonic are portable | C-small | segmented-radix CUDA-12.8 fallback |
| QSA indexer prefill score | direct `cp_async_zfill`, `ldmatrix`, `mma_bf16` | — | C | real SM70 scorer rewrite |
| QSA indexer prefill selection | segmented radix sort | — | A/B | compile-qualify CUDA 12.8 |
| QSA attention file | prefill MMA schedule directly uses `cp_async`, `ldmatrix`, `mma_bf16`; BF16 helper also contains cp.async staging | decode/SIMT portions share TU | C/D source split required | architecture-condition implementations so unsupported device code is not emitted for SM70 |
| `impl/mtp_forward*.{cpp,cu}` | no direct newer instruction hit in audit | downstream QSA/GDN/MoE/linear dependencies | B | qualify after non-MTP vertical slice |
| `impl/text_executor.cpp` | no direct graph API hit | graph behavior delegated through core/runtime | B | eager SM70 plan first |

## Structural `cp.async` conclusion

Do not implement this compatibility shim:

```text
cp_async(...) -> synchronous copy
cp_commit()    -> no-op
cp_wait()      -> no-op
```

The current shared GEMM and prefill schedules use stage/ring lifetime assumptions. SM70 replacements need an explicit schedule:

```text
global vector load
  -> registers
  -> shared store
  -> __syncthreads()
  -> compute
```

Start with one synchronous stage. A software ping-pong schedule is a later optimization only after correctness.

## Load-time codec requirements created by this audit

Before the first CUDA-12.8 SM70 full build can pass, the source set needs architecture-safe codec boundaries for:

1. E4M3FN decode usable without CUDA FP8 device types;
2. E4M3FN encode for load-time row-FP8 materialization;
3. E2M1 decode usable without CUDA FP4 device types;
4. E2M1 encode/pack for legacy BF16→NVFP4 MTP conversion;
5. exact NVFP4 scale swizzle/layout preservation;
6. exhaustive code tests (E2M1 all 16, E4M3FN all 256) and converter parity tests.

The CPU expert-pair benchmark intentionally implements its own host-side decode from the documented persistent contract instead of including `nvfp4_codec.cuh`; this lets the architecture gate run before the CUDA-12.8 device codec port exists.

## Phase-0.5 exit checklist

Completed in this audit artifact:

- [x] verify development branch HEAD before changes;
- [x] identify architecture-specific root build blocker;
- [x] identify unconditional Blackwell source-set blocker;
- [x] classify shared `cp.async`/pipeline helpers;
- [x] classify current MMA helpers;
- [x] identify QSA `DeviceTopK` as the CUDA-12.8 CCCL blocker;
- [x] select existing segmented radix machinery as first QSA decode fallback design;
- [x] audit load-time NVFP4 and FP8 conversion reachability, including legacy MTP conversion;
- [x] classify direct vs inherited dependencies in the Flash-Next target surface;
- [x] confirm CUDA Graph is a later runtime qualification issue rather than the first toolkit blocker.

Still intentionally deferred to Phase 2 / implementation compile qualification:

- [ ] build the architecture-conditioned `sm_70` source set with CUDA 12.8;
- [ ] compile-test the software converter headers/implementations under CUDA 12.8;
- [ ] compile-qualify `DeviceSegmentedRadixSort` in the exact CUDA-12.8 build;
- [ ] prove no unsupported PTX/device helper remains reachable in the generated SM70 target;
- [ ] run CUDA-12.8 tests on the target host.

Those remaining items require the Phase-2 root CMake/source-selection changes. They should not be pulled forward into this patch solely to make the standalone CPU architecture probe build.
