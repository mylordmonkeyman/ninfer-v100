# V100 forward-port Phase 11: whole-model vertical slice

Status: implementation/compile qualification in progress; physical V100 execution deferred.

Phase 11 is the first whole-model numerical acceptance gate. It is deliberately a
correctness path, not a performance path.

## Contract

The Phase-11 runner fixes the following execution contract:

- SM70 / CUDA 12.8 build
- one active lane
- CUDA Graph disabled
- speculative decoding / MTP disabled
- Vision disabled
- QSA prefill MMA disabled
- BF16 KV cache
- FP32 GDN recurrent state
- all 48 main-text routed-expert layers host-backed in canonical compact NVFP4
- no routed-expert cache
- correctness-first synchronous CPU expert execution
- minimum 4,096 teacher-forced oracle positions

The main-text shared expert, routing, attention, GDN, hyper-connection, PLE, output
head and state machinery remain on the production GPU path. Only the routed expert
pairs use the Phase-10 CPU reference path.

## Whole-model gate

The opt-in executable is:

```text
ninfer_qwen3_8_flash_next_vertical_slice_real_test
```

It requires:

```bash
export NINFER_WEIGHTS=/path/to/qwen3_8_flash_next_mixed.ninfer
export NINFER_FLASH_NEXT_ORACLE_MANIFEST=/path/to/oracle/manifest.json
```

The oracle manifest uses the repository's existing reference-oracle dump format.
Every position must contain a contiguous FP32 `logits` tensor. Positions must be
contiguous from zero and at least 4,096 positions must be present.

The runner:

1. preflights the artifact with the Phase-11 runtime configuration;
2. asserts CUDA Graph, MTP, Vision and resident main-text routed experts are off;
3. asserts all 48 main-text expert layers are host-backed;
4. records CUDA free-memory snapshots before model load, after model load and after
   runtime allocation;
5. loads the real artifact and production text executor;
6. forces eager execution;
7. resets host-expert execution counters;
8. advances one token at a time using the oracle's teacher-forced token sequence;
9. copies production BF16 logits to the host and compares them against oracle FP32
   logits;
10. verifies greedy sampled-token output and committed state-frontier progression;
11. verifies every text layer traversed the host expert path;
12. reports the numerical and VRAM ledgers.

For `P` teacher-forced positions, the expected Phase-11 host-expert counters are:

```text
completed_layer_calls = 48 * P
routed_tokens         = 48 * P
expert_pairs          = 48 * P * 10
```

## Numerical acceptance

The whole-model gate uses the specification's Section 7 thresholds:

```text
positions                   >= 4096
unexpected NaN/Inf          = 0
top-1 agreement             >= 99.0%
mean KL                     <= 1e-3
P99 KL                      <= 1e-2
relative mean-NLL delta     <= 0.5%
```

The runner also reports mean top-5 overlap, mean top-10 overlap, maximum absolute
logit error, the first top-1 divergence and the worst-KL position.

The target token for NLL at position `p` is the oracle token at `p + 1`; the final
position is excluded from the NLL denominator.

## VRAM reconciliation

The static planned model residency is:

```text
embedding payload
+ output-head payload
+ other non-expert model payload
+ device-weight alignment padding
```

Main-text routed experts are intentionally absent from this device total.

Runtime residency comes from the finalized Flash-Next runtime plan. The runner
compares those planned bytes with `cudaMemGetInfo` deltas measured:

```text
before model load
after model load
after runtime allocation
```

The CUDA context/device-runtime baseline is therefore established before the first
model snapshot rather than being charged to model residency.

This is the measurement used to validate the non-expert V100 memory ledger before
later expert-cache sizing.

## CI versus physical qualification

`.github/workflows/sm70-phase11-compile.yml` uses the CUDA 12.8 development
container on a GitHub-hosted runner. It:

- configures `CMAKE_CUDA_ARCHITECTURES=70`;
- builds the Phase-11 host contract test;
- builds the real whole-model runner;
- builds its text-executor and load-plan dependencies;
- executes only the host-side contract/oracle-math test.

The real whole-model test intentionally skips unless both oracle/artifact environment
variables and a usable CUDA device are present.

No physical V100 run is part of compile CI. That gate remains deferred until the
forward port is far enough along to justify target-hardware qualification.

## Boundary

Passing compile CI does **not** close Phase 11. Phase 11 closes only after a physical
V100 run passes the teacher-forced whole-model oracle and the observed non-expert
VRAM ledger is reconciled closely enough with the static plan.

Phase 12 (MTP correctness) and Phase 13 (expert cache) must not be inferred from a
Phase-11 compile pass.
