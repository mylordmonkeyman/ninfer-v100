# Qwen3.8-Flash-Next NVFP4 on one RTX PRO 6000: feasibility for native NInfer on Windows

**Research snapshot:** 2026-08-28. Hugging Face repositories were inspected at the revisions shown below. The ecosystem was changing during the research window, so repository counts and sizes are a point-in-time inventory, not a permanent catalogue.

## Executive finding

This is **possible on one RTX PRO 6000, but not as an all-weights-resident 96 GB model and not with NInfer as it exists today**.

The credible single-card design is:

1. keep the approximately 88.8 GiB compute backbone resident in the GPU;
2. keep the 51.2-billion-parameter PLE lookup table outside VRAM;
3. gather only the selected PLE rows through an asynchronous, target-owned host/mmap path;
4. use a quantized, memory-mapped PLE table on this particular workstation, because its 125.6 GiB of host RAM is too tight for the approximately 95 GiB BF16 PLE plus Windows, the runtime, staging buffers, and page cache;
5. build a new compile-time NInfer target and `.ninfer` artifact rather than trying to load a community Hugging Face checkpoint directly.

The strongest starting point for **quality and provenance** is the official Qwen BF16/FP8 release followed by a deterministic NInfer-owned conversion. The strongest released **feasibility reference** is Primitive's expert-NVFP4 or mixed NVFP4/FP8 backbone paired with its 32 GB INT4 (or 28.8 GB NVFP4-style) mmap PLE. That reference is Linux/vLLM publisher evidence, not proof of native Windows/NInfer performance.

## Evidence labels

- **Verified artifact fact** means a value was read from a repository file, Hugging Face API inventory, local binary/repository state, or official hardware documentation.
- **Publisher claim** means a model author reports a benchmark, validation result, or runtime behavior. It has not been reproduced here.
- **Inference** means the conclusion follows from the verified layout, local machine state, and NInfer's current contracts, but has not yet been implemented or benchmarked.

## What “Qwen4 preview” means

Qwen itself describes Qwen3.8-Flash-Next as an **early preview of the architecture it is developing for Qwen4**, analogous to Qwen3-Next previewing Qwen3.5. This is not a community nickname and does not mean the checkpoint is a finished Qwen4 release. [Official Qwen repository](https://github.com/QwenLM/Qwen3.8-Flash-Next) and [official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

The released checkpoint is materially different from NInfer's existing `qwen3.8-27b` target:

| Fact | Qwen3.8-Flash-Next | Current NInfer Qwen3.8-27B |
|---|---:|---:|
| main model | 125B total, about 6B active/token | 27B dense |
| auxiliary parameters | 51B PLE + 4B MTP | one MTP layer, no PLE |
| decoder layers / hidden width | 48 / 2560 | 64 / 5120 |
| layer schedule | 12 × (3 GDN→MoE + 1 QSA→MoE) | 48 GDN + 16 full-attention dense-MLP layers |
| experts | 512 routed, top-10 + shared expert | dense MLP |
| attention | QSA sparse micro-block selection | ordinary full attention in 16 layers |
| native context | 262,144 | target-specific existing contract |

The Flash-Next values come from the official [architecture README](https://github.com/QwenLM/Qwen3.8-Flash-Next) and [`config.json`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json). The current NInfer values are recorded in [`qwen3.8-27b-artifact.md`](../maintainer/qwen3.8-27b-artifact.md).

Important verified Flash-Next details:

- the Transformers identity is `Qwen4ExpForConditionalGeneration` / `qwen4_exp`;
- the PLE table has a base vocabulary of 20,000,000 and width 2560, hence 51.2B parameters, split into 128 parts and injected at layer 2;
- the 48 layers use 12 QSA layers and 36 Gated DeltaNet layers;
- QSA selects at the micro-block level with a 512-block / 2,048-token budget;
- the MoE has 512 routed experts, top-10 routing, and a shared expert;
- four gated residual branches use rank 320;
- one MTP layer, vision, image, and video are part of the released model;
- the official design explicitly makes the PLE host-offloadable with asynchronous prefetch.

## RTX PRO 6000 and native Windows facts

The official RTX PRO 6000 Blackwell Workstation Edition specification is 96 GB GDDR7 with ECC, 1,792 GB/s memory bandwidth, PCIe Gen 5, 600 W maximum power, fifth-generation Tensor Cores, and a quoted 4,000 AI TOPS. The 4,000 figure is theoretical FP4 throughput with sparsity; it is not a model decode forecast. [NVIDIA product page](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/) and [RTX Blackwell architecture whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf).

NVIDIA lists the workstation card as compute capability 12.0. CUTLASS exposes the Blackwell `MmaMXF4NVF4Op` path with E2M1 data, UE4M3 scale, FP32 accumulation, vector size 16, and `(16,8,64)` atom support for `sm_120a/f` and `sm_121a/f`. [CUDA GPU table](https://developer.nvidia.com/cuda/gpus) and [CUTLASS CuTe DSL API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html).

**Verified on the target workstation during this research:** NInfer's registered Qwen3.8-27B route is already production-proven natively on Windows on this exact RTX PRO 6000. This proves the Windows/Blackwell platform lane, not compatibility with the new Flash-Next model. Commits `3ca81ca8` and `5a94cd1c` introduced the MSVC+CUDA build and native Windows Vision dependencies; `build-win/apps/ninfer.exe` and `ninfer-serve.exe` exist. The live machine reports:

| Component | Observed value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| usable VRAM reported by driver | 97,887 MiB |
| compute capability | 12.0 |
| CUDA toolkit | 13.3 (V13.3.33) |
| host link | PCIe 5 ×16 |
| host RAM | 134,908,497,920 bytes = 125.6 GiB |
| local storage | Samsung 9100 PRO NVMe |

NVIDIA's current Windows guide supports Windows 10/11 and Visual Studio 2022 for CUDA development. [CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).

One platform caveat remains: NVIDIA's TensorRT performance guidance recommends TCC rather than WDDM for stable low launch latency and notes that CUDA Graphs can mitigate enqueue-bound overhead. NInfer already uses CUDA Graphs, which is favorable, but TCC availability for this exact board/driver must be queried rather than assumed. [TensorRT hardware/software performance environment](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/hw-sw-environment.html).

## NVFP4 inventory: the 14 direct descendants in the user's model-tree filter

The exact Hugging Face filter in the screenshot — `base_model:quantized:Qwen/Qwen3.8-Flash-Next` plus `NVFP4` — returned these 14 direct entries. “Direct” only describes Hub metadata; it does **not** certify faithfulness, quality, or runtime compatibility. Hub sizes are the sum of repository file bytes at the shown revision, not expected VRAM residency.

| Repository and inspected revision | Hub files | Concrete format / compatibility | Assessment |
|---|---:|---|---|
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) `7b719225` | 125.96 GiB | ModelOpt NVFP4 W4A4, E2M1, group 16, dynamic activations; routed experts only. PLE is FP8 E4M3 with per-table scale and other components BF16. Card launches SGLang TP2 and reports GB300/B300 validation. | Best-documented conventional W4A4 artifact, but not a one-card RTX PRO recipe. Its card calls it a private candidate release. |
| [Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4) `103a7608` | 170.28 GiB | ModelOpt NVFP4 group 16 with input scales; 1,267 ignored patterns make it effectively expert-only. Config says calibration applied with 128 × 2,048-token samples. | Sparse documentation and no exact runtime/hardware validation in its card. |
| [primitive-ai/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4) `607e11cc` | 173.64 GiB | 120.8B routed-expert parameters in NVFP4 group 16; BF16 PLE and all other components. Author describes weights-only RTN/no calibration and vLLM PLE CPU offload. | **Publisher-reported one-card proof:** 88,828 MiB VRAM plus about 100 GB host RAM on one 96 GB RTX PRO 6000. |
| [orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4) `3a3b6316` | 165.48 GiB | Auto-gated during inspection; tags claim compressed-tensors NVFP4/FP8. | Modified “uncensored” derivative; exact files/card were not independently inspectable. Not a faithful-source candidate. |
| [primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8) `a4e813ed` | 171.16 GiB | Routed experts NVFP4 W4A4 group 16; QSA and GDN projections FP8 E4M3 per-channel; BF16 PLE and tail. Requires `VLLM_GDN_DECODE_KERNEL=triton` in the author's vLLM recipe. | **Publisher-reported fastest one-card option:** 84.4 tok/s at concurrency 1 and 526 tok/s aggregate at 32, same 88.8 GiB class of GPU residency. Card's rounded 183.7 GB differs from current Hub byte sum. |
| [lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8) `344f3a68` | 123.48 GiB | Radix-style expert NVFP4 plus FP8 E4M3 128×128 weight-only QSA/GDN projections (`FP8_PB_WO`); FP8 PLE. | Publisher says TP1 RTX PRO 6000 was verified, **but requires patched SGLang**; card warns stock SGLang silently interprets packed FP8 incorrectly and emits garbage. |
| [axiomofmind/Qwen3.8-Flash-Next-W4A16-NVFP4](https://huggingface.co/axiomofmind/Qwen3.8-Flash-Next-W4A16-NVFP4) `41ef3b02` | 173.63 GiB | ModelOpt NVFP4 group-16 expert weights with BF16 activations; all other components source precision. | W4A16 rather than native W4A4. No reported validation; requires a runtime that understands the exact checkpoint. |
| [PixelML/Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark) `b80180e3` | 125.96 GiB | Weight files are a Radix mirror; the deployment card targets SGLang TP2 across two GB10 DGX Sparks. | Useful SM121 deployment evidence, not a new quantization and not a one-discrete-GPU recipe. |
| [mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4](https://huggingface.co/mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4) `f2c21eb3` | 173.64 GiB | Primitive expert-only NVFP4 layout with BF16 PLE, applied to a refusal-modified derivative. | Card claims the same one-card offload route; quality/safety differs from official Qwen and the repo became auto-gated during inspection. |
| [starkweatherdigital/qwen3.8-flash-next-nvfp4](https://huggingface.co/starkweatherdigital/qwen3.8-flash-next-nvfp4) `1b304e5f` | 101.73 GiB | NVFP4 compute checkpoint plus a 4-bit PLE. Requires its author's custom vLLM loader (`VLLM_PLE_NVFP4=1`). | Smallest complete direct repository, but still larger than usable VRAM and tied to a custom loader. Card reports a DGX Spark result, not RTX PRO validation. |
| [mbehr90/Qwen3.8-Flash-Next-nvfp4](https://huggingface.co/mbehr90/Qwen3.8-Flash-Next-nvfp4) `d28eef6` | 170.26 GiB | compressed-tensors `nvfp4-pack-quantized`; routed experts are 4-bit group 16 with BF16 activations (NVFP4A16), PLE BF16 host-offloaded. | Author validated 4×H100; H100 falls through a Marlin W4A16 route. Not evidence of native Blackwell W4A4 speed. |
| [lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8-Pruned-RTXPRO-6000) `37ab979f` | 114.96 GiB | Same mixed NVFP4/FP8 scheme, but every MoE layer and MTP is pruned from 512 to 448 experts using the publisher's magnitude/RMS heuristic. | Altered architecture; the card does not provide a convincing quality evaluation of the pruning. Still not all-resident in 96 GB. |
| [dealignai/Qwen3.8-Flash-Next-UNCENSORED-NVFP4](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-UNCENSORED-NVFP4) `7470878a` | 125.96 GiB | Radix-style W4A4 expert/FP8-PLE layout, with refusal-modified BF16 tensors. | Modified model; publisher targets two DGX Sparks. |
| [dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4) `be794b99` | 125.96 GiB | Current model-weight hashes are identical to the “UNCENSORED” sibling; README/assets differ. | Separate listing, not a separate current quantized weight set. Modified model; publisher targets two DGX Sparks. |

### What the released formats actually amount to

There are four meaningful backbone families, not 14 independent quantization inventions:

1. **ModelOpt W4A4 NVFP4 experts**: E2M1 values, one E4M3 scale per 16 values, FP32 global scale, and dynamic NVFP4 activation quantization. Radix and mirrors use this.
2. **Weights-only / W4A16 expert NVFP4**: the packed expert weights are 4-bit, but activation compute is BF16. Axiom and mbehr are in this family.
3. **Mixed NVFP4 + FP8**: experts use NVFP4 while attention/GDN projections use FP8. Primitive and lovedheart use different FP8 metadata/layouts and are not interchangeable.
4. **PLE-quantized combinations**: the 51.2B lookup table is FP8 or 4-bit and needs a dedicated gather/dequant/offload path. Starkweather and companion PLE repositories are in this family.

No inspected release advertises a quantized KV cache as part of the model files. KV/state precision is a runtime choice.

## Wider search: why “all NVFP4 results” is noisier than the direct 14

A general Hugging Face search returned 26 entries whose **name or tags** mention NVFP4. It includes the 14 above plus 12 non-direct/noise entries:

| Extra result | What it is |
|---|---|
| [vcruz305/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/vcruz305/Qwen3.8-Flash-Next-NVFP4) | Empty/incomplete placeholder (effectively zero model bytes). |
| [windowsxp811203/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/windowsxp811203/Qwen3.8-Flash-Next-NVFP4) | Empty/incomplete placeholder. |
| [windowsxp811203/Qwen3.8-Flash-Next-Abliterated-NVFP4](https://huggingface.co/windowsxp811203/Qwen3.8-Flash-Next-Abliterated-NVFP4) | Quantization of an altered abliterated parent, not the official base. |
| [Blackfrost-AI/Qwen3.8-Flash-Next-DERISKED-NVFP4](https://huggingface.co/Blackfrost-AI/Qwen3.8-Flash-Next-DERISKED-NVFP4) | Quantization of an altered “derisked” parent. |
| [gorbatjovy/qwen3.8-flash-next-abliterated-NVFP4-plefp8](https://huggingface.co/gorbatjovy/qwen3.8-flash-next-abliterated-NVFP4-plefp8) | Altered abliterated model with FP8 PLE. |
| [Lewfkrad/Qwen3.8-Flash-Next-NVFP4-W4-PLE](https://huggingface.co/Lewfkrad/Qwen3.8-Flash-Next-NVFP4-W4-PLE) | Approximately 26.8 GiB companion 4-bit PLE files, not a full checkpoint. |
| [axiomofmind/Qwen3.8-Flash-Next-W4A16-NVFP4-GGUF](https://huggingface.co/axiomofmind/Qwen3.8-Flash-Next-W4A16-NVFP4-GGUF) | GGUF conversion for another runtime, not a native NVFP4 NInfer/vLLM artifact. |
| [local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89) | Non-direct 98.66 GiB repository; custom/incompletely documented option. |
| [provsalt/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4](https://huggingface.co/provsalt/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4) | Inferact-derived checkpoint with NVFP4 PLE and custom format assumptions. |
| [aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP](https://huggingface.co/aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP) | Quantization of the modified Mazin “uncensored” model. |
| [pocharlies/Qwen3.8-Flash-Next](https://huggingface.co/pocharlies/Qwen3.8-Flash-Next) | 62.9 KB deployment recipe for Radix on two DGX Sparks; no weights. |
| [randomllama/Qwen3.8-Flash-Next-DGX-Spark-field-notes](https://huggingface.co/randomllama/Qwen3.8-Flash-Next-DGX-Spark-field-notes) | 68.8 KB field notes/recipe; no weights. |

Thus, **14 is the stable answer to the user's exact direct-quantization screenshot; 26 is a discovery/search count, not 26 usable native NVFP4 checkpoints**. The broad query briefly returned 24 name matches during inspection; the two tag-only recipe entries explain the 26 count. This is normal search-index drift while repositories are being published.

## The PLE table is the decisive single-card issue

The PLE is an indexed embedding table, not a matrix scanned for every token. That makes offload viable: only selected rows need to cross PCIe. It also means repository size alone is a poor VRAM proxy.

[Primitive's companion PLE repository](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant) publishes three concrete mmap formats, each in 128 safetensors shards with 2,500,012 rows/shard, row width 160, plus `META.json`:

| PLE format | Published size | Stored tensors | Dequantization |
|---|---:|---|---|
| FP8 per row | 52.48 GB / 48.88 GiB (card rounds to 49 GB) | `weight_fp8[rows,160]` E4M3FN + `weight_scale[rows]` FP32 | `fp8 * row_scale` |
| INT4 group 16 | 32 GB | `weight_i4[rows,80]` packed nibbles + `weight_scale[rows,10]` FP16 | `(nibble - 8) * group_scale` |
| NVFP4-style group 16 | 28.8 GB | packed E2M1-like code + E4M3FN group scale + FP32 global scale | lookup(code) × group scale × global scale |

The overlay maps shards and dequantizes only gathered rows, approximately 100–200 KB per decoded token according to the publisher. The current stock vLLM worker cannot load these formats; Primitive ships replacement worker/model files for its exact image.

**Publisher-reported RTX PRO 6000 measurements** for the mixed backbone:

| PLE route | Host/mapped footprint | c1 decode | c32 aggregate | Reported quality note |
|---|---:|---:|---:|---|
| BF16 in RAM | 95.4 GB anonymous RAM | 84.4–84.5 tok/s | 516.8–523.6 tok/s | baseline |
| FP8 mmap | 52.6 GB mapped pages | 80.1–80.3 | 489.7–500.7 | same reported band |
| INT4 mmap | 32.9 GB mapped pages | 80.1–80.2 | 483.6–487.9 | same reported band |
| NVFP4 mmap | 29.8 GB mapped pages | 80.1–80.3 | 476.8–479.5 | same reported band |

The author further reports MTP at 129.6 tok/s with INT4 and 128.8 with NVFP4, versus 142.6 with BF16 in RAM and only 77.5–82.3 with BF16 merely disk-backed. These results are valuable architecture evidence: **quantized mmap PLE retains the useful working set in page cache; raw BF16 disk paging does not**. They were measured on Linux/vLLM with 176 GB host RAM, not reproduced on this 125.6 GiB Windows system.

## Recommended download/conversion source

**Recommendation: use a two-repository, revision-pinned source bundle — Primitive's mixed backbone without its BF16 PLE files, plus only the INT4 PLE directory from Primitive's companion repository.** This is complete for a native NInfer conversion and does not require any tensor from the 360 GB official BF16 repository.

| Source | Selective payload | Role |
|---|---:|---|
| [`primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8@a4e813ed`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/tree/a4e813ed3cfbbcc61e2929699eccb864a4dfa843), excluding `ple-bf16-*.safetensors` | 81.381 GB | Complete compute backbone and frontend: NVFP4 experts, FP8 QSA/GDN projections, BF16 sensitive tail, PLE support tensors, MTP, Vision, embeddings/head, tokenizer/config. |
| [`primitive-ai/Qwen3.8-Flash-Next-PLE-quant@da8b3958`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant/tree/da8b39586016d8325ac619be28ad77d6296625ec), including only `ples_int4/*` | 32.000 GB / 29.80 GiB | All 128 PLE table parts as packed INT4 group-16 values and FP16 scales, plus `META.json`. |
| [`Qwen/Qwen3.8-Flash-Next@de4b8e4d`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE), `LICENSE` only | 3.2 KB | Preserve the actual Qwen Community License with the converted artifact/source record; Primitive's repository metadata does not replace it. |
| **Combined** | **113.382 GB / 105.60 GiB** | Complete converter input. |

Suggested acquisition, with revisions pinned rather than mutable `main`:

```powershell
hf download primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8 `
  --revision a4e813ed3cfbbcc61e2929699eccb864a4dfa843 `
  --exclude "ple-bf16-*.safetensors" `
  --local-dir source/qwen38-flash-next-mixed

hf download primitive-ai/Qwen3.8-Flash-Next-PLE-quant `
  --revision da8b39586016d8325ac619be28ad77d6296625ec `
  --include "ples_int4/*" `
  --local-dir source/qwen38-flash-next-ple

hf download Qwen/Qwen3.8-Flash-Next LICENSE `
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 `
  --local-dir source/qwen38-flash-next-license
```

This is a **converter splice**, not a directory that Transformers can load after removing 43 files. The NInfer converter must intentionally replace the 128 `ngram_embedding.shard_N.weight` entries referenced by the mixed checkpoint's [safetensors index](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/blob/a4e813ed3cfbbcc61e2929699eccb864a4dfa843/model.safetensors.index.json) with the companion's shard-ordered INT4 rows described in its [format specification](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant/blob/da8b39586016d8325ac619be28ad77d6296625ec/README.md). The omitted 43 `ple-bf16-*` files contain only those 128 table tensors; the PLE convolution, key/value projections, norms, multipliers, offsets, and vocabulary sizes remain in the downloaded `carry-model-bf16-*` files.

### Coverage verified from the repository indexes

The mixed index has 296,630 entries. Its remaining non-PLE payload and the companion table jointly cover:

| Required area | Verified coverage |
|---|---|
| main backbone | layers `0..47`; every layer has experts `0..511`; routed expert projections, routers, shared experts, QSA/GDN, indexers, hyperconnections, and norms are present |
| PLE | nine support tensors remain in the mixed backbone; companion supplies 128 table shards `0..127` |
| MTP | all 31 semantic tensors, including packed BF16 expert `gate_up_proj`/`down_proj`, attention/indexer, hyperconnections, FCs, and pre-FC norms |
| Vision | 333 `model.visual` tensors: blocks `0..26` plus patch, position, and merger components |
| embeddings/output | `model.language_model.embed_tokens.weight` and `lm_head.weight` both present |
| frontend/config | `chat_template.jinja`, generation config, tokenizer JSON/config/vocabulary/merges, image and video preprocessors, mixed quant config, and model config |

The [mixed repository API inventory](https://huggingface.co/api/models/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8?blobs=true) shows that the chat template, generation config, tokenizer, tokenizer config, image preprocessor, and video preprocessor have the same Git blob IDs as the [official repository inventory](https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next?blobs=true). `config.json` differs intentionally to describe the quantized tensors.

### Complete single-repository alternatives

Several released repositories are genuinely complete according to their current safetensors indexes; “quantized checkpoint” does not imply that MTP or Vision is absent.

| Complete source | Download | Coverage / reason not preferred |
|---|---:|---|
| [RadixArk NVFP4 `7b719225`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594) | 135.254 GB | **Best one-repository fallback.** Complete 48×512 backbone, 128-shard FP8 PLE, 31-tensor BF16 MTP, 27-block Vision, embeddings/head, and official-identical frontend files. Better integrity reporting than most alternatives, but QSA/GDN remain BF16 and its exact-current quality comparison is limited. |
| [Primitive mixed `a4e813ed`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/tree/a4e813ed3cfbbcc61e2929699eccb864a4dfa843) | 183.782 GB | Complete by itself, including the original-value BF16 PLE. Choose this when NInfer should quantize the PLE independently instead of trusting a companion table. |
| [Primitive plain `607e11cc`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4/tree/607e11cce422007424bfd023b9d73c1d8379fde6) | 186.449 GB | Complete, but larger and leaves QSA/GDN BF16; weaker fit for the intended mixed-performance profile. |
| [lovedheart mixed `344f3a68`](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8/tree/344f3a6820275dfcbb07d9c2a9d8b7ca1f37b3af) | 132.582 GB | Complete and compact, but uses custom FP8 128×128 `FP8_PB_WO` projections. Its card documents that stock SGLang silently misloads them and emits garbage; NInfer would need exact dedicated decoding and stronger quality qualification. |
| [starkweatherdigital `1b304e5f`](https://huggingface.co/starkweatherdigital/qwen3.8-flash-next-nvfp4/tree/1b304e5f99de0faaf43c3a959f2b4000294bf65c) | **109.231 GB** | Smallest complete one-repository payload: main and MTP experts plus PLE are NVFP4. Not preferred because there is no task-quality evaluation, its input scales are derived from Radix rather than direct calibration, and its publisher reports regenerating about 0.2% of truncated shard bytes on CPU. |
| [provsalt NVFP4-PLE `48d98195`](https://huggingface.co/provsalt/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4/tree/48d98195ac8da8ff10d9ee497b5d52e7817f058d) | 109.238 GB | Complete, including 128 NVFP4 PLE parts and Vision. Backbone provenance inherits the sparsely documented Inferact quant; PLE evidence is a 236-position teacher-forced study plus smoke tests, not a task-quality benchmark. |

The complete-source assertion above was checked against each repository's `model.safetensors.index.json`: all six contain 48 main layers, all 512 routed experts per layer, 128 PLE shards, 27 Vision blocks, both embedding/output matrices, and complete MTP. Radix/Primitive/lovedheart store MTP experts as two packed semantic tensors; starkweather/provsalt expand the 512 MTP experts with quantization metadata, which explains their larger index-entry counts.

### Provenance/accuracy decision

- The recommended bundle has the best alignment with the intended NInfer execution profile and the strongest released same-publisher end-to-end evidence. Primitive reports its mixed checkpoint as weights-only RTN, with nonquantized tail tensors byte-identical to source, and reports the INT4 PLE within the same accuracy band as BF16 on its 1,370-item protocol. These remain publisher claims and must be requalified with NInfer's independent oracle and capability suite.
- Radix is the cleaner fallback when a single repository and more explicit integrity audits matter more than minimizing bytes. It is still a community “private candidate” and its BF16 comparison used an earlier source revision.
- The approximately 109 GB all-NVFP4 repositories save only about 4 GB versus the recommended 113.4 GB bundle while adding materially weaker provenance or evaluation. That is not a good trade for the canonical NInfer artifact.
- None of these community repositories is an official Qwen quantization. Pin revisions, retain the Qwen license, verify every tensor/scale, and do not mix modified “uncensored,” abliterated, or pruned derivatives into the canonical target.

### Why an all-resident 96 GB design does not fit

For 120.8B routed-expert parameters, NVFP4 with one byte-scale per 16 values is approximately:

```text
120.8B × (4 bits + 8 bits / 16) = 67.95 GB decimal
```

Adding even the 28.8 GB NVFP4-style PLE yields about 96.75 GB **before** attention/GDN weights, the shared expert, routers, embeddings, vision, MTP, KV/state, CUDA graphs, and workspaces. That exceeds the card's practical allocation envelope. Therefore no faithful, unpruned release can be all-resident merely by quantizing both the experts and PLE to the published NVFP4 layouts.

The released one-card route instead demonstrates about 88.8 GiB GPU residency and external PLE. On this machine:

- BF16 PLE wants about 95–100 GB of free host RAM, leaving only about 25–30 GiB for Windows, NInfer, staging, and page cache: operationally too tight;
- FP8 mmap is plausible but leaves a larger 49 GB page working set;
- INT4 or NVFP4-style mmap is the safest 125.6 GiB host-RAM fit;
- the Samsung 9100 PRO is favorable for cold-page refill, but steady decode still depends on locality and page-cache policy;
- pinned buffers, overlapped gathers, and a bounded prefetch queue should keep PCIe traffic asynchronous rather than synchronizing every token.

## What NInfer must add

The existing NInfer Windows/Blackwell path, NVFP4 primitives, GDN machinery, MTP, Vision, serving, scheduling, and CUDA Graph infrastructure are substantial reusable foundations. They do **not** make Flash-Next a checkpoint-only registration exercise.

A coherent implementation needs a new fixed target owning:

1. the exact 48-layer `Qwen4Exp` tensor inventory and `.ninfer` binding;
2. QSA micro-block indexer, sparse-selection state, page-size contract, and prefill/decode kernels;
3. 512-expert top-10 routing plus shared expert, fused expert dispatch, and routed NVFP4 GEMMs at the real shapes;
4. four-way gated residual/hyperconnection semantics;
5. PLE n-gram hashing/indexing, host/mmap table format, asynchronous gather/prefetch, Windows mapping/pinning, and MTP-aware speculation/commit semantics;
6. the target's MTP, KV, GDN-state, QSA-state, and continuation transactions;
7. the exact Vision/text frontend and output semantics for this checkpoint;
8. deterministic conversion from one pinned source and independent numerical parity at each quantized boundary.

NInfer's current NVFP4 product format is `blockscale-k16-m128x4-v1`, documented in [`tensor-formats.md`](../maintainer/tensor-formats.md) and [`storage-layouts.md`](../maintainer/storage-layouts.md). Community ModelOpt and compressed-tensors repositories are therefore **conversion sources**, not files the public Engine can load directly. The exact model is also absent from the registered identities in [`artifact-container.md`](../maintainer/artifact-container.md).

Recommended implementation profile:

- **GPU:** Primitive-style mixed backbone — expert W4A4 NVFP4, QSA/GDN projections FP8, BF16 only for sensitive/small tails — because it reduces recurrent decode traffic and has the strongest one-card publisher measurement.
- **PLE:** start with INT4 group-16 mmap as the quality/performance/host-memory reference; qualify NVFP4-style PLE as a second candidate, not automatically the winner.
- **Runtime ownership:** make PLE gather a semantic target operation, not an artifact-loader side effect. It must participate in round planning, MTP draft fetches, cancellation, and continuation rollback.
- **Scheduling:** preserve NInfer's fixed 1–8 request contract; qualify c1 and the supported concurrency points rather than chasing the publisher's c32 number.
- **Windows:** use explicit file mapping and page-working-set telemetry; pre-touch only metadata and hot rows, not the whole PLE. Verify TCC support; otherwise benchmark WDDM jitter with CUDA Graph replay.

## Feasibility verdict

| Question | Verdict |
|---|---|
| Can the architecture run on one RTX PRO 6000? | **Yes.** A community publisher reports it operating with ~88.8 GiB GPU-resident backbone plus host PLE on exactly this GPU class. |
| Can every weight and runtime allocation fit in 96 GB VRAM? | **No**, not for the faithful unpruned model with the released formats. Even experts + smallest published PLE exceed the practical budget before the rest of the model. |
| Is native Windows the blocker? | **No.** This checkout already has native Windows CUDA binaries and the exact card is live. Windows mmap/page-cache behavior and WDDM/TCC latency are engineering and qualification risks, not an architectural impossibility. |
| Can current NInfer load one of these HF repositories today? | **Partially.** The pinned source splice now converts to a native `.ninfer` artifact, and the exact target binder validates and plans all 1,566 objects. The Engine Program and execution leaves are not yet registered, so it cannot generate tokens yet. |
| Is this mostly a quantization/converter task? | **No.** It is a new model target with several new semantic operators and a heterogeneous-memory execution path. |
| Is a hyper-optimized native implementation credible? | **Yes**, because Blackwell has the required NVFP4 tensor path, NInfer already owns the relevant native execution machinery, and PLE access is sparse. It is still a substantial target implementation and must be benchmarked rather than projected from AI TOPS. |

## License, provenance, and trust caveats

The official model uses the **Qwen Community License Agreement 1.0**, not Apache 2.0. It requires retention of the copyright/license notice. Products above 100 million monthly active users or USD 20 million monthly revenue must prominently display the Qwen model name. Commercial Model-as-a-Service or “AI Work Assistant” use requires a separate Qwen license, except internal use whose outputs/capabilities are not made available to third parties. The license also imposes legal/IP use restrictions. Read the complete [official LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) before distribution or service deployment.

Several derivative cards tag themselves Apache 2.0 because their recipe or delta is Apache-licensed. That metadata does not relicense Qwen-derived model weights. The Qwen terms continue to matter.

Trust ranking for an NInfer artifact source:

1. **Preferred:** official Qwen checkpoint + NInfer-owned deterministic quantization and conversion, with an independent oracle and capability evaluation.
2. **Possible reference/source after audit:** Radix or Primitive faithful-base repositories pinned by revision, with tensor inventory, hash/provenance comparison, and exact quality re-evaluation.
3. **Do not use as the faithful base:** uncensored, abliterated, derisked, or pruned variants; auto-gated or incomplete repositories; mirror-only recipes; checkpoints whose card requires an unmerged or patched runtime but does not pin the patch.

Model cards and community benchmarks are publisher statements, not security or correctness certification. In particular, one card documents a silent wrong-loader failure that produces garbage rather than an error. A native converter must reject unsupported or ambiguous tensor metadata instead of guessing.

## Practical go/no-go gate

Proceed only if the target is explicitly added to NInfer's product contract and the first milestone proves these three things on this exact Windows workstation:

1. a pinned official/reference checkpoint converts into an exact `.ninfer` inventory and the mixed GPU-resident payload leaves enough VRAM for the supported context/concurrency/state/graphs;
2. an INT4 or NVFP4-style mmap PLE gather meets an independent row-dequant oracle and remains overlapped under c1 and c8 MTP workloads;
3. end-to-end Text parity and a representative Vision prompt pass before performance tuning.

If those gates pass, this is a defensible native-NInfer target. If the requirement changes to “everything in VRAM, no host table,” the faithful model is a **no-go on 96 GB** with the quantization formats released so far.
