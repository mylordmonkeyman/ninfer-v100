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
DFlash2 is qualified separately with the Qwen3.8-27B groupwise artifact.

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
[comparison after the shared MTP changes](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37152861712)
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

The [updated runtime regression](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37152861694)
passed after restoring context lookup and separating dense wide-verification graph topology classes.
It includes real MTP K=1/3/7, state/checkpoint/vision regressions, and Flash-Next preservation.

### Other registered Qwen targets

| Artifact identity | Ported V100 route | Real artifact qualification |
|---|---|---|
| Qwen3.6-27B groupwise-int | Original Q4/Q5/W8 leaves and shared runtime | Real-artifact accuracy passed; measured throughput recorded below |
| Qwen3.6-27B NVFP4 | Original NVFP4 prepacking and A16 leaves; shared runtime | Current candidate matches qualified archived original; live-original graph diagnostic failed; fresh throughput passed |
| Qwen3.6-35B-A3B groupwise-int | Original binder, Q4 experts/W8 shared projections, workspace-aware GDN, Volta graph topology and MTP K=1..7 | Real-artifact accuracy passed; measured throughput recorded below |
| Qwen3.8-27B groupwise-int | Original groupwise leaves and shared runtime | Real-artifact accuracy passed; measured throughput recorded below |
| Qwen3.8-27B NVFP4 | Original prepacked FP8/NVFP4 leaves and shared runtime | Expanded direct accuracy and updated runtime regression passed |

The 35B-A3B binder matches the original fork. Its GDN leaf now explicitly passes A16 policy and
caller workspace; its Volta MTP limit and wide-verification graph topology match the original
fork. Qwen3.6-27B uses the already restored 27B variant and its original profile-specific binder.
All peer variants are compiled into the public Engine. Compilation and source review do not
substitute for real-artifact accuracy/performance qualification. The completed peer matrix used published artifacts linked by the original fork; downloads verified the pinned repository revision, published size and SHA256 before use.

### Groupwise reduction and NVFP4 decoding

Real Qwen3.6 artifacts exposed nondeterministic Q4/Q5 split-K reductions inherited from the
original fork. Each split now writes its own FP32 output plane, followed by a fixed-order
reduction before BF16 conversion. Split-K parallelism is retained. Workspace planning includes
every split plane and the bounded fused projection bands.

The [real-model arithmetic diagnostic](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37139276263)
passed seven actual projection shapes, using captured model activations, signed checkpoint
Q4/Q5 codes and stored FP16 scales. An independent naive FP64 calculation checked 992 sampled
outputs per shape, including residual additions. Relative L2 errors ranged from 0.159% to
0.180%, below the existing 1/256 criterion; the absolute-error checks also passed. Both
implementations produced the same measured errors after the isolated reduction correction.
In that diagnostic run, both Qwen3.6-27B artifacts also repeated the full four-request sequence three times, with
64 generated tokens per request, and matched the corrected reference exactly. A later original-reference NVFP4 run failed repetition; this earlier success did not establish universal reference stability.

The first peer benchmark exposed a separate Qwen3.6 NVFP4 regression: MTP K=3 generation
measured 20.47 tok/s against 43.39 tok/s in the original. The port had replaced the original's
packed CUDA software decode conversions with scalar lookup and exponent reconstruction in
hot A16/GEMV paths. Commit `a45105faf9c1b72540462f4e08cfbf9086414a3c` restores those
original decode conversions on SM70. The portable encoder used by Flash-Next remains intact.
The dedicated real-artifact recheck requires exact same-mode accuracy, ordinary repeatability,
and measured prefill and generation rates within 5% of the unchanged original for graph,
MTP K=3 and MTP K=7 profiles.

The [decoder recheck](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37155769244)
passed current candidate accuracy against qualified original exports from run 37141867858.
Model publication metadata and the reference correction were checked before comparison. All 4,096
scored probabilities matched exactly; eager, graph, MTP K=3/K=7 outputs matched exactly, and
candidate eager repetition passed. Fresh unchanged-original scoring also matched exactly.

The live corrected original remains intermittently unstable. In the latest run its eager,
MTP K=3/K=7, scoring and eager repetition checks passed, but graph generation differed by
62 tokens from the candidate and archived original. The previous attempt had scoring drift and
failed original eager repetition. Those failures are retained under `live-original/`; the current
`reference-provenance.json` explicitly records the live diagnostic failure. The successful
overall recheck means archived candidate accuracy and fresh throughput passed; it does not
mean every live-original accuracy check passed or that the inherited instability's cause is known.

Restoring packed software decoding recovered the measured throughput: ordinary graph generation
rose from 7.11 to 19.13 tok/s, and MTP K=3 rose from 20.47 to 43.47 tok/s on this workload.
Fresh unchanged-original rates were 19.12 and 43.40 tok/s. All three tested modes' prefill and
generation rates were within 0.6% of the fresh original, passing the 5% regression criterion.

### Peer accuracy reference

In the successful peer matrix run 37141867858, every peer profile passed 4,096 teacher-forced log probabilities against the unchanged
original, with maximum and mean differences **0**. Eager, graph, MTP K=3/K=7, and available
DFlash/DFlash2 exports matched their corresponding corrected original mode exactly, with
256 generated tokens per mode. Fresh ordinary request sequences repeated exactly in both forks in that matrix. The later NVFP4 reference failures described above remain separate limitations.

For peer generation, the reference is the original fork plus only the isolated ordered
split-K correction. Scoring and performance use the unchanged original. The preserved
performance binary links the original engine statically and is copied before that correction.

| Peer profile | MTP K=3 versus ordinary graph differing tokens: original / candidate |
|---|---:|
| Qwen3.6-27B groupwise | 34 / 34 |
| Qwen3.6-27B NVFP4 | 49 / 49 |
| Qwen3.6-35B-A3B groupwise | 47 / 47 |
| Qwen3.8-27B groupwise | 0 / 0 |

These inherited cross-mode differences remain explicit diagnostics. The peer matrix requires
same-mode accuracy and ordinary repeatability; it does not claim universal MTP-versus-ordinary
equality. The separate mounted Qwen3.8-27B NVFP4 comparison requires that equality and passes it.

### Later upstream fixes

The [upstream review](v100-qwen36-upstream-fixes.md) examined 85 commits in Neroued/ninfer after the shared ancestor. No reviewed change establishes a fix for the intermittent single-request original-reference failure. The newer SiLU restoration reverses a later TMA approximation absent from our Volta paths; the FP8/INT8 batch-append fixes require multiple independent requests; the DFlash chunk-binding fix does not apply to this 27B NVFP4 test. Older state-ownership and terminal KV-coverage fixes are already ancestors of the original reference.

### Artifact compatibility and pins

Both readers expect v2 container magic `4e494e4645520002` (hex). Latest v3 publications are
incompatible with the source fork's reader and are excluded. The matrix downloads the pinned
historical publications below and validates actual magic, published size and SHA256 before use.

| Artifact | Hugging Face revision | Bytes | SHA256 |
|---|---|---:|---|
| Qwen3.6-27B groupwise | `faaa0c140d0a92743872256a8b78a954b3984018` | 17,495,365,888 | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| Qwen3.6-27B NVFP4 | `300373c46a7911a6483111b0c8c6c8972f459937` | 18,324,064,000 | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| Qwen3.6-35B-A3B groupwise | `560f227e5a7104756d1a108201a8aa75654ea688` | 22,783,246,080 | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |
| Qwen3.8-27B groupwise | `dc370fb6295ae8b786e1af4f90d7142a16255c35` | 20,437,336,576 | `0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3` |

### 27B throughput

INT8 group-64 KV, one request, prefill chunk 1024, a 1024-token prompt followed by 64 timed
generated tokens, one discarded warmup and two measured repetitions. Generation rates count
committed output tokens. Baseline and candidate were separate runs on the same GPU.
Candidate rates below are from the updated runtime regression; original rates are from the
earlier unchanged-fork reference on the same artifact and corpus. The original fork's much higher
published rates use different prompts and acceptance rates.

| Mode | Candidate prefill tok/s | Original prefill tok/s | Candidate generation tok/s | Original generation tok/s |
|---|---:|---:|---:|---:|
| Eager, no MTP | 1084.6 | — | 29.39 | — |
| CUDA graphs, no MTP | 1084.0 | 1087.5 | 29.50 | 29.52 |
| CUDA graphs, MTP K=1 | 1070.6 | — | 42.72 | — |
| CUDA graphs, MTP K=3 | 1067.5 | 1067.4 | 46.31 | 46.25 |
| CUDA graphs, MTP K=7 | 1059.3 | — | 37.20 | — |

Graph-mode prefill is about 0.3% lower in this small sample; ordinary and K=3 generation
differ by less than 0.2%. K=3 was fastest among the tested windows on this prompt.
K=1/3/7 draft acceptance was 57.5%/30.6%/13.3%, respectively. These are workload-specific
measurements, not general generation guarantees.

The [saved-report summary](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37155916176)
extracts the current dense and NVFP4 recheck rates and previously completed groupwise-peer rates from JSON artifacts without rerunning GPU work. Dense/NVFP4 data use runtime a45105f; the three groupwise peer artifacts use the previously qualified a51dfec matrix.


### Peer throughput

Same V100, INT8 KV, one request, 1,024 prompt tokens plus 64 committed generated tokens, one warmup and two repetitions. Prefill and generation are separate rates. Measurements establish parity on this workload; they do not guarantee general serving throughput.

| Artifact | Mode | Original prefill tok/s | Candidate prefill tok/s | Original generation tok/s | Candidate generation tok/s |
|---|---|---:|---:|---:|---:|
| Qwen3.6 27B groupwise | graph | 1065.7 | 1066.2 | 29.74 | 29.81 |
| Qwen3.6 27B groupwise | mtp-3 | 1047.8 | 1047.0 | 46.66 | 46.67 |
| Qwen3.6 27B groupwise | mtp-7 | 1044.9 | 1039.7 | 36.86 | 36.71 |
| Qwen3.6 27B NVFP4 | graph | 224.6 | 224.3 | 19.12 | 19.13 |
| Qwen3.6 27B NVFP4 | mtp-3 | 223.1 | 224.3 | 43.40 | 43.47 |
| Qwen3.6 27B NVFP4 | mtp-7 | 224.0 | 223.0 | 36.44 | 36.37 |
| Qwen3.6 35B-A3B | graph | 691.1 | 693.1 | 135.37 | 134.10 |
| Qwen3.6 35B-A3B | mtp-3 | 685.9 | 685.2 | 118.63 | 118.66 |
| Qwen3.6 35B-A3B | mtp-7 | 684.1 | 683.7 | 71.76 | 71.81 |
| Qwen3.6 35B-A3B | dflash | 681.2 | 680.1 | 72.46 | 72.07 |
| Qwen3.8 27B groupwise | graph | 1065.2 | 1064.7 | 29.99 | 30.10 |
| Qwen3.8 27B groupwise | mtp-3 | 1045.8 | 1047.3 | 35.44 | 35.83 |
| Qwen3.8 27B groupwise | mtp-7 | 1041.4 | 1040.8 | 24.03 | 24.51 |
| Qwen3.8 27B groupwise | dflash2 | 1012.9 | 1012.6 | 27.25 | 27.56 |

On this prompt, ordinary graphs were faster than speculative modes for 35B-A3B; MTP K=3 was faster for the two Qwen3.6 27B artifacts. Mode rankings depend on draft acceptance and workload.

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
throughput with the references distinguished below. DFlash and DFlash2 are checked for
the corresponding published artifacts. Models run serially on the V100. Qualification
passed accuracy in [run 37141867858](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37141867858). The Qwen3.6 NVFP4 decoder correction is qualified separately in [run 37155769244](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37155769244).

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
  --max-context 2048 --kv-capacity 2048 --max-concurrency 1 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

This listens on the container's loopback interface. CUDA graphs remain enabled by default for
27B. The V100 desktop reserve defaults to zero. This command uses the tested model, KV format,
prefill width and MTP window; The [real HTTP smoke](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37146113118)
passed health, model listing and a chat request returning `READY`, with graphs prepared and
MTP K=3 configured. The temporary test server was stopped after the check.
