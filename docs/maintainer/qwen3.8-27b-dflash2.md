# Qwen3.8-27B DFlash2 execution and state contract

DFlash2 is a startup-selected speculative backend of the shared Qwen family Program. It uses
the public `.ninfer` Engine route, request scheduling, Frontend settlement and output publication.
It is available for canonical Qwen3.8-27B and the separately registered
`qwen3.8-27b-orcarouter/nvfp4` artifacts containing the complete companion bundle.
OrcaRouter's full proposal head uses its BF16 source weights through the BF16 LinearTopK route;
the optional optimized head is derived from those same weights. The companion was trained for
the canonical model, so useful acceptance on the derivative requires separate measurement.
Native Windows performance and qualification results belong in [performance.md](../performance.md).
The [artifact reference](qwen3.8-27b-artifact.md) owns tensor inventory and storage formats.

The source companion is [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`. Its weights match the upstream NInfer
reference's `z-lab` mirror. The integration follows upstream NInfer's completed DFlash2 series
through `a16b6442`, retaining this fork's Windows, state-ownership and constrained-decoding contracts.

## Geometry and computation

| Quantity | Value |
|---|---|
| Drafter layers / hidden width / MLP width | 5 / 5120 / 17408 |
| Query heads / KV heads / head dimension | 32 / 8 / 128 |
| Target feature layers | 5, 19, 33, 47, 61 |
| Concatenated feature width | 25600 |
| Context window | 2048 tokens |
| Dynamic convolution | two taps, groups of 16 hidden channels |
| Selector | 16 candidates per position, rank 256 |
| Valid vocabulary / physical rows | 248077 / 248320 |
| Mask token | 248070 |

The checkpoint recommends a block width of eight: one anchor and seven draft tokens. The Engine
supports startup-fixed draft count `K=1..15` and concurrency `B=1..8`. Query width is `K+1`.
Padded execution widths do not extend any request's live proposal or verification prefix.

The target's five feature layers condition the drafter. One parallel masked-block forward through
the five draft layers produces candidates at every draft position. A conditional selector traces
one candidate path; the target verifies that path. This is one parallel drafting pass, not repeated
diffusion denoising. Dynamic convolutions respect positions within each request's block, including
non-power-of-two widths and short terminal extents.

The target remains the output authority. Greedy verification accepts matching target choices;
stochastic verification uses the sparse proposal probabilities and target rejection correction.
The acceptance Op reads token counts and masks without committing persistent request state.
Numerical qualification uses the independent Op oracles; output parity with an existing route is
additional evidence and must identify the target's quantization and execution profile.

Greedy output is not guaranteed identical to ordinary width-one decoding. Verification width
selects different qualified FP8/NVFP4 kernel arithmetic, and complete response text can diverge
despite the same target weights and sampling parameters. Do not advertise this integration as
bitwise lossless or treat acceptance rate as an answer-quality score.

## State and settlement

Each StateImage owns five cyclic context caches and their frontiers. DFlash2 has no full paged
draft-KV pool. Its cyclic payload follows Device/Host capture and restore with the target state.
The companion adds about 2.07 GiB of W8/BF16 tensor payload, plus 40 MiB of cyclic context per
StateImage and the lane/graph workspace selected by the planner.

Target features stay pending until the Frontend selects the committed output prefix. The Program
then commits token counts, GDN/recurrent state, hidden features and draft context for that prefix.
Rejected suffixes and discarded tokens after stop/budget/cancellation cannot become retained
context. Main KV coverage is a lower bound during speculative staging; only explicit settlement
truncates mappings. DFlash2 must never dereference or allocate the legacy DFlash paged-KV pool.

Continuation reuse restores the same cyclic caches and metadata as fresh execution would create.
Qualification covers partial terminal blocks, page boundaries, replacement/wrap of the 2048-token
ring, and Host restore. Vision input uses the same target prefill and feature-capture path; DFlash2
accelerates subsequent generated-text decode, not media encoding or prefill.

## Constrained output

Structured-output matcher state belongs to each request. Allowed-token masks occupy stable
Program-owned addresses and refresh for every committed token. Constrained family-runtime lanes
use zero speculative extent while constrained; unconstrained neighbors retain speculation.
When every lane is constrained or has only one output/context position left, the Program executes
a width-one target round instead of the drafter and padded verification. DFlash2 still captures
and appends committed target features; MTP still aligns its state with the sampled target token.
The startup allocation and graph envelope include this ordinary route for every backend.
Mixed batches retain the speculative schedule with zero extent on their constrained lanes.
Zero extent still requires masked target sampling. Every raw-greedy shortcut must
exclude a non-null `allowed_tokens` mask, including sparse proposal acceptance.

The graph envelope must remain compatible with requests that become constrained after capture.
Grammar state cannot leak across lane reuse, prefix restore or mixed constrained/unconstrained
batches. Format validity is a runtime contract and does not establish legal-answer quality.

## Measurement

Compare ordinary decoding, MTP and DFlash2 with the same target weights, KV format, sampling,
prompts, token budgets and hardware. Keep cold/reused prefill and decode measurements distinct.
Report committed output tokens per wall second, first generated/content delta latency, acceptance
at every draft position, memory use and completed output text. Begin at K=7 and tune K, proposal
head and concurrency from measured results. Published H200 results are not RTX PRO 6000 results.
