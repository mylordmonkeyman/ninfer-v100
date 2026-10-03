# V100 Qwen forward port

The `forwardport/v100-flash-next` branch restores the dense 27B SM70 operation paths from
`geoffwatts/ninfer-v100` at `b37d0dd3e1163b9d802d8bccfa89918bf68d793e`, while retaining the
current branch's Qwen3.8 Flash-Next implementation. This is a source port; runtime qualification
is recorded separately below.

## Implementation

- Build the complete public Engine operation graph with CUDA 12.8 and SM70, including the
  CLI, server, scoring and benchmark targets. CUTLASS is pinned to v4.4.2; the original fork's
  vendored llama.cpp attention implementation retains its license.
- Reorder 27B NVFP4 code and scale planes and row-scaled FP8 code planes at load time into
  Volta QPN layout. Token embeddings retain their checkpoint layout. The model file is unchanged.
- Use the working fork's FP16 tensor-core paths for wide prefill and QPN paths for small
  batches and decode, including shared activation staging for paired FP8 projections.
- Restore the fused projection, SwiGLU, residual, output selection, recurrent-state and
  BF16/INT8/FP8 attention implementations needed by the dense runtime.
- Keep Flash-Next's FP8 weights with FP32 row scales on their existing separate implementation;
  27B weights with BF16 row scales use the restored prepacked implementation. Retain Flash-Next's
  registered BF16 shapes, portable NVFP4 codec and selected-block attention.
- Set the V100 default desktop memory reserve to zero. Callers can request another value explicitly.
- Restore the working fork's SM70 runtime hardware gate and reject unavailable KV storage before loading.
- Permit 27B and 35B-A3B MTP windows from one to seven drafts on SM70. Flash-Next retains its own limit.
  CUDA graph control remains available for dense 27B.

NVFP4 and K8V4 **KV cache storage** remain unavailable on Volta. This restriction does not
prevent loading NVFP4 **model weights**, which use the restored Volta dequantization paths.
The shared dense/35B runtime now includes the original fork's repeated-context MTP optimization:
copy a continuation after a matching 16-token suffix, require agreement with the learned drafts,
and verify up to 15 copied tokens using the target model. The larger replay records, workspace,
graph family, pending-row stride and acceptance counters are planned explicitly. Requests using
structured output or token log probabilities retain their ordinary decoding path.
DFlash2 performance is not qualified by this regression.

## Verification

Qualified on 2026-10-03 using the user's Tesla V100-PCIE-32GB and CUDA 12.8. The mounted
`/models/qwen3_8_27b_nvfp4.ninfer` artifact is 23,719,496,192 bytes.

The [candidate regression](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37098263741)
passed on code `dfa075179f8b426d4fa4f0169457753d5c79d80c`:
- CLI, server, public benchmark and MTP round benchmark builds.
- Real causal scoring: finite results, exact fresh-state repeatability, and maximum overlap error 0.
- Real eager generation and CUDA graph generation.
- Real graph-enabled MTP with K=1, 3 and 7.
- Speculative page-boundary, checkpoint-rewrite and vision regressions.
- Flash-Next expert-cache/text-decode checks and the real grouped-prefill smoke check.

The [original-fork reference](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37096296837)
also passed real scoring with maximum overlap error 0 and generated on the same artifact and
the same current-branch token corpus. The source fork was pinned to the commit above.

### Direct accuracy comparison

The [direct original-fork comparison](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37100631031)
passed on the actual Qwen3.8-27B NVFP4 artifact. The expanded
[comparison after the shared MTP changes](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37127552576)
also passed:

- 4,096 teacher-forced token log probabilities: maximum and mean difference **0**.
- 64 greedy generated tokens after each of 31-, 1,024- and 4,096-token prompts plus a
  repeated-text prompt (256 output tokens per mode), in eager,
  CUDA graph and MTP K=3 modes: **zero token differences** between implementations.
- MTP output also matched ordinary graph output in each implementation.
- Both implementations accepted 47 copied tokens beyond the configured K=3 window in each
  of two scenarios, establishing that the repeated-context lookup path actually ran.

Both binaries used the unchanged `tools/validation/v100_qwen_parity.cpp` public Engine probe,
the same explicit token corpus and the same model artifact. Scoring used FP8 KV and a 1024-token
prefill chunk; generation used INT8 group-64 KV. The criterion is exact input/greedy-token
agreement and a maximum log-probability error of `1e-5`. This compares actual token probabilities,
not the full vocabulary distribution. It establishes parity on these workloads, not universal
accuracy or an independent mathematical-oracle result.

The [updated runtime regression](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37127268924)
passed after restoring context lookup and separating dense wide-verification graph topology classes.
It includes real MTP K=1/3/7, state/checkpoint/vision regressions, and Flash-Next preservation.

### Other registered Qwen targets

| Artifact identity | Ported V100 route | Real artifact qualification |
|---|---|---|
| Qwen3.6-27B groupwise-int | Original Q4/Q5/W8 leaves and shared runtime | Real-artifact qualification pending |
| Qwen3.6-27B NVFP4 | Original NVFP4 prepacking and A16 leaves; shared runtime | Real-artifact qualification pending |
| Qwen3.6-35B-A3B groupwise-int | Original binder, Q4 experts/W8 shared projections, workspace-aware GDN, Volta graph topology and MTP K=1..7 | Real-artifact qualification pending |
| Qwen3.8-27B groupwise-int | Original groupwise leaves and shared runtime | Real-artifact qualification pending |
| Qwen3.8-27B NVFP4 | Original prepacked FP8/NVFP4 leaves and shared runtime | Expanded direct accuracy and updated runtime regression passed |

The 35B-A3B binder matches the original fork. Its GDN leaf now explicitly passes A16 policy and
caller workspace; its Volta MTP limit and wide-verification graph topology match the original
fork. Qwen3.6-27B uses the already restored 27B variant and its original profile-specific binder.
All peer variants are compiled into the public Engine. Compilation and source review do not
substitute for real-artifact accuracy/performance qualification. Published artifacts linked by the
original fork are available for the pending model matrix; downloads verify the pinned repository
revision, published size and SHA256 before use.

### 27B throughput

INT8 group-64 KV, one request, prefill chunk 1024, a 1024-token prompt followed by 64 timed
generated tokens, one discarded warmup and two measured repetitions. Generation rates count
committed output tokens. Baseline and candidate were separate runs on the same GPU.
Candidate rates below are from the updated runtime regression; original rates are from the
earlier unchanged-fork reference on the same artifact and corpus. The original fork's much higher
published rates use different prompts and acceptance rates.

| Mode | Candidate prefill tok/s | Original prefill tok/s | Candidate generation tok/s | Original generation tok/s |
|---|---:|---:|---:|---:|
| Eager, no MTP | 1073.4 | — | 29.40 | — |
| CUDA graphs, no MTP | 1072.7 | 1087.5 | 29.47 | 29.52 |
| CUDA graphs, MTP K=1 | 1058.4 | — | 42.75 | — |
| CUDA graphs, MTP K=3 | 1053.2 | 1067.4 | 46.40 | 46.25 |
| CUDA graphs, MTP K=7 | 1045.0 | — | 37.21 | — |

Graph-mode prefill is about 1.4% lower in this small sample; ordinary and K=3 generation
differ by less than 0.4%. K=3 was fastest among the tested windows on this prompt.
K=1/3/7 draft acceptance was 57.5%/30.6%/13.3%, respectively. These are workload-specific
measurements, not general generation guarantees.

The [saved-report summary](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37098963575)
extracts the earlier candidate rates from completed JSON artifacts without rerunning GPU work.
The table above uses the newer regression artifacts.

### Flash-Next preservation

The final real smoke check passed natural routing, finite logits, committed state and
fixed-policy exact replay. At 1024 warm prompt tokens, grouped prefill averaged 51.34 tok/s
versus 39.29 tok/s with grouping bypassed. The following 32-token teacher-forced decode
averaged 8.95 versus 8.92 tok/s. This preserves the previously accepted Phase 11 profile;
it does not establish the original specification's strict Section 7 numerical qualification.
The full 4096-position performance sweep remains manually available and was not completed
during this dense-port qualification.

### Workflows

`.github/workflows/v100-qwen-accuracy.yml` compares public scoring and eager/graph/MTP generation
against the original fork, including a repeated-text prompt for context lookup. Manual dispatch
accepts an exact `artifact_path` under the runner's mounted model storage, so the same comparison
can qualify each Qwen3.6/groupwise artifact once uploaded. Evidence is preserved as
`v100-qwen-accuracy`, including `parity.json` and the per-token comparison exports.

`.github/workflows/v100-qwen-model-matrix.yml` downloads four published peer artifacts into
the runner's writable `../v100-qualification-models` directory, validates pinned publication
metadata, size and SHA256, and compares scoring, eager/graph/MTP K=3/K=7 generation and
throughput with the unchanged original-fork binaries. DFlash and DFlash2 are checked for
the corresponding published artifacts. Models run serially on the V100. Qualification
is pending in [run 37128702414](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37128702414).

`.github/workflows/v100-27b-regression.yml` automatically checks the candidate and the real
Flash-Next smoke path. Scoring and dense-runtime logs are uploaded immediately so failures
can be diagnosed before later checks finish. The unchanged original fork can be rebuilt and
compared by manually dispatching the workflow with `compare_baseline=true`.

`.github/workflows/v100-prefill-policy.yml` retains the complete short/long/cold prefill sweep
through manual dispatch. Newer runs supersede obsolete ones and build directories survive
checkout cleanup. Synthetic general Q4/NVFP4 linear tests remain excluded from these workflows.

### Run the 27B server in the existing runner container

The qualification build is available at `/work/ninfer-v100/build-27b` inside
`ninfer-v100-runner`; the model is already mounted. For a local server check after the GPU job
has finished:

```bash
sudo docker exec -it ninfer-v100-runner \
  /work/ninfer-v100/build-27b/apps/ninfer-serve \
  /models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 2048 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

This listens on the container's loopback interface. CUDA graphs remain enabled by default for
27B. The V100 desktop reserve defaults to zero. This command uses the tested model, KV format,
prefill width and MTP window; HTTP protocol serving itself was built, but was not exercised by
the benchmark qualification.
