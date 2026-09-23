# V100 forward-port Phase 11: whole-model vertical slice

Status: physical V100 execution established; non-expert VRAM ledger
reconciled (+5.6 MiB / 0.08%); 32-position smoke at 31/32 top-1 (mean KL
0.0274, single top-1 flip at position 13). The full 4,096-position
acceptance run is pending against the relaxed gate family documented in
"Documented error analysis".

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

## Physical qualification status

Runs execute on a self-hosted V100 runner (32 GB, CUDA 12.8, SM70 build)
through `.github/workflows/v100-phase11-runtime.yml`. The frozen precision
configuration for all numerical runs is: the full FP32 chain
(`NINFER_FLASH_NEXT_FP32_HYPER_STATE`, `..._GDN_CONV`, `..._GDN_PROJECTION`,
`..._GDN_READOUT`, `..._GDN_GATE`, `..._GDN_OUTPUT`, `..._GDN_OUTPUT_INPUT`,
`..._HYPER_INJECT_INPUT`, `..._HYPER_INJECT_APPLY`, `..._MLP_HYPER_OUTPUT`,
`..._MLP_HYPER_INPUT`, `..._MLP_HYPER_NORMALIZED`, `..._MLP_HYPER_LOW_RANK`,
`..._MLP_HYPER_APPLY`, `..._MLP_HYPER_INJECTION` all `1`), CPU reference
expert backend, 32 workers, CUDA Graph / MTP / Vision off, BF16 KV, FP32 GDN
recurrent state.

### VRAM reconciliation (measured 2026-09-23)

| component | planned GiB | observed GiB |
|---|---|---|
| non-expert weights (model load) | 6.729 | 6.730 |
| runtime allocation | 0.469 | 0.473 |
| total | 7.198 | 7.203 |

observed − planned = +5.639 MiB (0.078%); resident routed-expert layers = 0.
The non-expert ledger is reconciled. The 7.2 GiB device residency bounds the
Phase 13 expert-cache capacity derivation.

### 32-position smoke (commit 435690df, 2026-09-23)

| metric | measured | spec §7 initial gate | verdict |
|---|---|---|---|
| positions | 32 | ≥ 4096 | partial (smoke window) |
| NaN/Inf | 0 | 0 | pass |
| top-1 agreement | 31/32 (96.875%) | ≥ 99% | fail |
| mean KL | 0.027374 | ≤ 1e-3 | fail |
| P99 KL | 0.615417 | ≤ 1e-2 | fail |
| relative mean-NLL delta | 0.349% | ≤ 0.5% | pass |
| mean top-10 overlap | 0.928125 | — | — |

The single top-1 failure is position 13 (candidate 12688 vs oracle 37027;
KL 0.6154; max logit error 2.880). The prompt reaches EOS (248046) at
position 14; positions 15–31 are post-EOS with KL ≈ 0, so the effective
content window of the smoke is 15 positions. Per-position KL inside the
content window: pos 2–5 ≤ 0.0007, pos 6 = 0.0082, pos 7 = 0.0173, pos 8–9
≈ 0, pos 10 = 0.0746, pos 11 = 0.0135, pos 12 = 0.0781, pos 14 = 0.0232.


## Documented error analysis (spec §7 gate relaxation)

The spec §7 initial thresholds (mean KL ≤ 1e-3, P99 KL ≤ 1e-2) are
unreachable for the Phase 11 precision profile. This section is the
specification's documented error analysis justifying their relaxation. All
numbers are measured on the self-hosted V100 at commit 435690df with the
frozen precision configuration above: the 14-position stage-trace run
(12,516 candidate/oracle stage cells plus a PLE-replay pass) and the
32-position smoke.

### Audited components: none is the root cause

| component | evidence | verdict |
|---|---|---|
| GDN SSM recurrence | full L24 replay with oracle inputs at all 14 positions: recurrence NRMSE 0.00065 | within floor; kernel exonerated |
| L24 conv history | isolation: 91% of candidate BF16 words differ from the oracle and drive the post-conv Q/K/V error; restoring the oracle history removes it (key 0.058 → 0.006, value 0.035 → 0.003) | real, but downstream: the conv reads the drifted input stream |
| L24 as end-to-end cause | second-stage isolation: injecting exact L24 `gdn_recurrent_output` (and `gdn_zh`) at position 13 leaves the position-13 logits wrong (top-1 12688 vs 37027, KL 0.749) | L24 is not the end-to-end root cause |
| L0 MLP boundary | implementation error 1.3–1.7× the BF16 materialization floor | at/near floor |
| GDN projection (FP8) | error ≈ 0.0018 ≈ floor | at floor |
| conv given correct history | exact | at floor |
| router (FP32) | on the 213 mismatching cells: FP32 reference set 213/213, FP32 tree 213/213, BF16 tree 206/213 vs the oracle set | correct; flips are input-driven, margin-limited |
| expert mixing (association) | max error 2.6e-6 across all mismatching cells | exact |
| expert output materialization | BF16 rounding RMS 1.31e-3, max 3.96e-3 | at the BF16 floor |
| hyper-connection apply/inject | stage pass-through ≈ 1.0× (hyper_after_attn NRMSE ≈ its input error) | not an amplifier |
| PLE path | 14/14 oracle injections verified; candidate PLE NRMSE 0.0036; injecting exact PLE reduces downstream router drift (L44 pos-13 router NRMSE 0.631 → 0.246) | small contributor, not the cause |

### Mechanism: floor compounding plus discrete top-k amplification

1. **Per-layer base error.** With every audited component at its rounding
   floor, a layer adds ≈ 0.3–0.4% NRMSE to the BF16 activation stream
   (exact-id MoE cells: `mlp_block_output` 1.27% vs `mlp_block_input` 1.00%
   → ≈ 1.3× pass-through).
2. **Compounding.** Teacher-forced drift over the 14-position window (mean
   `mlp_block_output` NRMSE across L00–L47): 0.33% (pos 0/1) → 7.0% (pos 7)
   → 17.9% (pos 12), non-monotonic with partial cancellation.
3. **Discrete top-k flips.** Once the drifted router input error exceeds the
   typical top-10 boundary gap (median oracle top-10 gap ≈ 0.006, mean
   ≈ 0.012), the selected expert set changes. Over the 672 (position, layer)
   router cells: 459 exact-id, 116 order-only, 97 membership flips (14.4%).
   Flips are depth-localized: 4 flips in L00–L19 vs 93 in L20–L47. A flip
   cell's MoE block output error is 24.05% (mean) vs 9.25% for order-only
   cells on ≈ 2.4× the input error: the flip adds ≈ 2.9× discrete
   amplification on top of the ≈ 1.3× linear pass-through. In 87/97 flip
   cells the router score RMS error exceeds the top-10 cutoff margin, i.e.
   the flip is the expected consequence of drifted input, not router
   rounding.
4. **Tail.** Final-hidden error ≈ 10–18% at positions 9–13 → max logit
   error 2.9–3.1 → a top-1 flip occurs only where the oracle's top-1 margin
   is below the drift (position 13). The end-to-end error is a superposition
   across all 48 layers; making any single mid-depth layer exact does not
   restore the tail (the L24 exact-injection proof above).

Consequence: even a hypothetical all-components-at-floor implementation
cannot push the mean KL below the floor-compounding bound (per-position KL
0.008–0.078 on non-flipped tail content, measured), which is already an
order of magnitude above the initial 1e-3 gate. The top-1 failure rate and
the P99 KL are set by the discrete flip rate on drifted input, which the
precision profile (BF16 activations, top-k routing) does not suppress.

### Proposed relaxed gate

User-facing guarantees (top-1, NLL) stay hard; the KL statistics are
relaxed to values consistent with the mechanism above:

| metric | spec §7 initial | proposed | rationale |
|---|---|---|---|
| positions | ≥ 4096 | ≥ 4096 | unchanged |
| NaN/Inf | 0 | 0 | unchanged |
| top-1 agreement | ≥ 99% | ≥ 99% | unchanged; primary guarantee |
| relative mean-NLL delta | ≤ 0.5% | ≤ 0.5% | unchanged; currently 0.349% |
| mean KL | ≤ 1e-3 | ≤ 2e-2 | non-flipped content-window mean is ≈ 0.008–0.03 (measured); the corpus content fraction sets the overall mean, calibrated by the 4,096-position run |
| P99 KL | ≤ 1e-2 | ≤ 1e-1 | one top-1 flip contributes KL ≈ 0.3–0.6, so P99 ≤ 1e-2 is mathematically incompatible with a top-1 flip rate above ≈ 0.1%; 1e-1 is consistent with the hard top-1 ≥ 99% gate (≤ 41 flips in 4,096 positions) |

The 4,096-position acceptance run calibrates the final mean/P99 values
against the measured distribution. The calibration must remain consistent
with the per-layer floor profile above: a relaxation justified only by the
measured mean, not by the floor mechanism, is not acceptable.


## CI versus physical qualification

Two workflows cover the two surfaces:

- `.github/workflows/sm70-phase11-compile.yml` (GitHub-hosted, CUDA 12.8
  container): configures `CMAKE_CUDA_ARCHITECTURES=70`, builds the Phase-11
  host contract test, the real whole-model runner, and its text-executor and
  load-plan dependencies, and executes only the host-side
  contract/oracle-math test. The real whole-model test skips unless both
  oracle/artifact environment variables and a usable CUDA device are
  present, so this workflow performs no physical qualification.
- `.github/workflows/v100-phase11-runtime.yml` (self-hosted V100 runner):
  builds the SM70 targets, runs the contract and selected-block attention
  tests, the 14-position full-layer error decomposition, the 14-position PLE
  oracle-injection replay, the 32-position best-known smoke, and — on a
  `[full-phase11]` push marker or `full_phase11` dispatch input — the full
  ≥ 4,096-position teacher-forced qualification run. Results are uploaded as
  a `v100-phase11-<run>-<attempt>` artifact.

## Boundary

Phase 11 closes only after a physical V100 run passes the teacher-forced
whole-model oracle under the relaxed gate family above (the initial §7
thresholds are unreachable for this precision profile per the documented
error analysis) with the non-expert VRAM ledger reconciled — the latter is
already satisfied (+5.6 MiB / 0.08%).

Phase 12 (MTP correctness) and Phase 13 (expert cache) must not be inferred
from a Phase-11 compile pass or from the 32-position smoke.
