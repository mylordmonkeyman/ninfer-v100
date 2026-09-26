# Qwen3.8-Flash-Next Oracle & State Diagnostic Harness

This directory provides the authoritative CPU FP32 reference forward pass and state divergence analysis harness for Qwen3.8-Flash-Next.

## Phase 11 sequential storage-profile experiment

`run_precision_reference.py` is a CPU-only supplementary diagnostic. It first
decodes the frozen tokens with FP32 state and requires agreement with the
independent full-sequence oracle. It then repeats sequential decoding with
explicit BF16 materialization for selected projections, attention and MLP
inputs, expert intermediate values, convolution history, attention KV, and
logits, while retaining FP32 hyper and GDN SSM state. It
writes position-indexed FP32 stage dumps and KL/top-1 summaries.

The experiment is **not** an exact V100 precision oracle or a proven lower
bound. It currently computes CPU FP32 arithmetic between materialization
boundaries, and does not reproduce the reduction order of FP8 GEMV or fused
GDN readout. Only stage agreement with V100 can justify attributing final
errors to the precision profile. The authoritative mathematical oracle remains
`run_oracle.py`.

With the existing oracle environment and read-only source assets, the small
14-position experiment is invoked as follows:

```bash
python run_precision_reference.py \
  --model-dir /srv/ninfer/source/mixed \
  --ple-dir /srv/ninfer/source/ple/ples_int4 \
  --ids-file /srv/ninfer/oracle/phase11/token_ids.json \
  --fp32-oracle /srv/ninfer/oracle/phase11-stage-trace14 \
  --positions 14 \
  --out-dir /srv/ninfer/precision-reference/phase11-14
```

The V100 test can optionally write pre-injection selected tensors using
`NINFER_PHASE11_CANDIDATE_TRACE_ROOT`, with the existing all-position stage
trace enabled. Compare the three sets with:

```bash
python compare_precision_traces.py \
  --oracle /srv/ninfer/oracle/phase11-stage-trace14 \
  --cpu /srv/ninfer/precision-reference/phase11-14/v100-phase11-storage \
  --v100 /path/to/candidate-trace \
  --incremental-fp32 /srv/ninfer/precision-reference/phase11-14/fp32 \
  --out-dir /path/to/comparison
```

The frozen full-sequence oracle does not contain raw router scores. The
incremental FP32 trace supplies those scores as an explicitly labeled
supplement; its logits must first pass the frozen oracle parity check.

The first real-model 14-position CPU run succeeded on September 24, 2026
([workflow](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36014366398),
[compact report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36015048672)):

| Measurement against independent FP32 oracle | Result |
| --- | ---: |
| Worst incremental FP32 decode KL | 2.99e-11 |
| Storage-profile mean KL | 0.038748 |
| Storage-profile top-1 agreement | 14 / 14 |
| Storage-profile expert-set flips | 82 / 672 layer-position cells |
| Flips in layers 00–19 / 20–47 | 6 / 76 |
| Largest per-position KL (position 13) | 0.340838 |

This supports the precision-compounding hypothesis as a **CPU storage-profile
experiment**, while leaving the V100 cause unresolved. The profile is
uncalibrated: it models more expert-set flips at depth than near the input, but
does not reproduce the observed V100 top-1 flip at position 13. Stage parity
against a V100 candidate trace is required before treating these values as a
precision floor or revising Phase 11 acceptance thresholds.

### V100 calibration result

A single 14-position V100 candidate trace was collected
([GPU workflow](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36015694540),
[compact comparison](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36016286612)).
All 2,744 selected stages were compared with both the CPU profile and the
independent oracle. The GPU diagnostic CTest exited nonzero because the
teacher-forced qualification gate failed; the stage export and comparison
completed successfully.

| Measurement | CPU profile | V100 candidate |
| --- | ---: | ---: |
| Mean logits KL against FP32 oracle | 0.038748 | 0.057752 |
| Top-1 agreement across 14 positions | 14 / 14 | 13 / 14 |
| Expert-set flips across 672 layer-position cells | 82 | 97 |

The router flips overlap in only 57 cells: 25 are CPU-only and 40 are
V100-only. At position 2, the CPU profile has no flips while the V100 has
seven. At position 13, the CPU profile predicts the oracle top-1 token, but
the V100 predicts a different token. The embedding agrees exactly across all
three traces, and the initial layer differences are small; later divergences
depend strongly on position and router trajectory. For example, at position 2
the `L40_mlp_block_input` NRMSE is 0.00354 for CPU versus oracle and 0.04466
for V100 versus oracle; at position 10 those values are 0.21268 and 0.12543.

This **does not establish a precision floor** or exonerate all kernels.
Further work should first explain the missing V100-only flips and the excess
CPU-only flips at the earliest divergent layers, then rerun the same stage
comparison before using the profile to change acceptance thresholds.

### Corrected profile and raw-score comparison

The first CPU experiment above BF16-rounded the GDN QKV projection output.
The selected V100 diagnostic uses `NINFER_FLASH_NEXT_FP32_GDN_PROJECTION=1`
and `NINFER_FLASH_NEXT_FP32_GDN_CONV=1`; the corresponding CPU profile now
feeds the unrounded QKV output into convolution, while preserving its other
materialization boundaries. The revised CPU run
([workflow](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36017907847))
passed incremental FP32 parity at worst KL 2.99e-11 and emitted 3,416 stage
rows and 672 complete router-score vectors. The V100 trace
([workflow](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36018385552))
and [compact score comparison](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36019037272)
used the same 14 frozen positions.

| Measurement against FP32 reference | Corrected CPU profile | V100 candidate |
| --- | ---: | ---: |
| Mean logits KL | 0.0259998 | 0.057752 |
| Top-1 agreement | 14 / 14 | 13 / 14 |
| Expert-set flips / 672 layer-position cells | 54 | 97 |
| Median router-score NRMSE / 672 cells | 0.0003662 | 0.0004753 |
| P95 router-score NRMSE / 672 cells | 0.0086096 | 0.0131782 |

Of the expert-set flips, 39 are shared, 15 occur only on CPU, and 58 only
on V100. The first V100-only flip at position 2, layer 20 exchanges experts
489 and 388, whose FP32 cutoff gap is just 0.0002737; CPU and GPU router-score
RMS errors are 0.001506 and 0.002002 respectively. At position 13, layer 39,
the gap is 0.04668, while CPU and GPU router-score RMS errors are 0.02987
and 0.06394. The median GPU/CPU score-error ratio is 1.196 across cells.
These scores show how near ties amplify divergence but also show that the CPU
profile does not quantitatively match the V100 candidate.

A controlled V100 probe changed only
`NINFER_FLASH_NEXT_FP32_ROUTER_INPUT=1`
([workflow](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36019645369)).
It still flips the position 2, layer 20 expert set, has 93 expert-set flips,
13 / 14 top-1 agreement, and mean KL 0.06444803. The diagnostic CTest
continues to fail its original Phase 11 acceptance gate; the probe workflow
completed successfully and uploaded its trace. This result does not support
BF16 router-input materialization as the dominant source of the mismatch.

The precision-matched reference remains a calibration effort. Neither this
CPU profile nor the router-input probe establishes a mathematical lower bound
on KL. The following replay examines an earlier attention boundary upstream
of the layer-20 near tie.

### Sequential reference calibration at the first layer

The independent CPU decoder recomputes all 14 prefixes and carries rounded
cache and hyper state forward, rather than rounding completed FP32 traces.
An added BF16 attention-prepare shadow matches the V100 `L00_attn_block_input`
exactly at every position. The resulting
[three-way run](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36059468003)
has mean logits KL against FP32 of 0.03544772 for CPU and 0.05775240 for V100;
CPU versus V100 KL is 0.02770246. The CPU and V100 expert-set flips overlap
in 53 cells, with 17 CPU-only and 44 V100-only flips
([stage report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36060137188)).
These differences preclude attributing the V100 gate failure solely to the
precision profile.

A [focused layer-zero GDN trace](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36061122886)
compares the independent CPU recurrence with the V100 before the first MLP.
At position 0, CPU versus V100 NRMSE is 0 at the attention input, 2.03e-7
at recurrent output and 3.25e-7 at the gated output. The raw attention
projection trace differs by 0.001668 because the V100 diagnostic exposes
its FP32 accumulator while the CPU stage holds BF16. Rounding the V100
accumulator drops this error to 2.33e-6, with 99.92% of elements identical
([rounding report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36061643827)).
At position 12 the gated-output difference is 0.000266 and the rounded
projection difference is 0.000769; history and its readout need further
calibration. Despite the close position-zero projection, `L00_mlp_block_input`
still differs by 0.002997. The next focused test should capture the post-GDN
hyper update and MLP hyper prepare on both implementations before attempting
to change precision policy or the §7 gate. The expanded GDN tracing option
is diagnostic only; the default 14-prefix reference continues comparing the
original common set of V100 stages.

The [post-attention hyper trace](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36062874657)
then identified a rule missing from the CPU profile: the selected V100
diagnostic applies the **FP32 GDN output projection** to the FP32 hyper master,
not its BF16 mirror. After correcting that rule, the
[14-position three-way run](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36063434889)
has CPU mean KL 0.02179401 against the oracle, V100 mean KL 0.05775240,
and CPU-to-V100 mean KL 0.03587110. At position zero the CPU-to-V100
`L00_hyper_after_attn` NRMSE drops from 0.001488 to 4.82e-7;
`L00_mlp_block_input` drops from 0.002997 to 4.28e-5. This verifies the
specific precision rule locally, but the complete reference still does not
track V100: 47 router flips are shared, 22 CPU-only and 50 V100-only.
The [next-boundary report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36063853923)
shows that at position zero `L00_hyper_after_mlp` is already 0.003256
CPU-to-V100 NRMSE, before PLE and layer one. The next test should compare
`L00_mlp_block_output` to distinguish MoE output from MLP hyper injection.
Position 12 also has a residual GDN-history difference before this MLP
boundary. The §7 acceptance gate remains unchanged and unpassed.

The [layer-zero MoE output trace](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36064757282)
locates the first large position-zero discrepancy inside the MoE: the CPU-to-V100
`L00_mlp_block_input` NRMSE is 4.28e-5, `L00_mlp_block_output` is 0.004072,
and `L00_hyper_after_mlp` is 0.003256. At position two the corresponding
values are 9.87e-5, 0.002779, and 0.001785. The V100 MoE output at position
zero is closer to the independent oracle than the CPU profile (0.001715 versus
0.005091 NRMSE), so the CPU reference's weight dequantization, router alpha,
shared path, and expert arithmetic must be separated before blaming a V100
kernel. The V100 host-expert reference rounds BF16 input and intermediate
activations; the CPU reference models these casts but still uses different
arithmetic and accumulation. The precision floor has not been established.

The [layer-zero routing trace](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36072336918)
shows that router alpha and shared scale already match CPU-to-V100 at
position zero (NRMSE 8.58e-6 and 2.61e-6). Inspection of the selected V100
shared expert found that its FP32 gate/up projections multiply before a BF16
activation boundary; the CPU reference previously rounded both projections
separately. With that corrected, the
[14-prefix reference](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36073386852)
reduces position-zero CPU-to-V100 MoE output NRMSE from 0.004072 to 0.000245.
Its CPU mean KL against the independent FP32 oracle nevertheless rises from
0.021794 to 0.102301 because different rounded trajectories change router
membership. This local calibration does not establish a global precision
floor.

The [PLE boundary report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36073960605)
then locates a repeatable next mismatch: position-zero CPU-to-V100 NRMSE is
0.000196 after layer-zero MLP, but 0.004260 at PLE injection. The V100 PLE
reads the BF16 hyper shadow and materializes normalized query/key, gated
value, and normalized gated value in BF16. After independently recomputing
those boundaries in the CPU reference, the
[updated 14-prefix run](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36074175598)
and [before/after report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36074509683)
put the position-zero CPU-to-V100 PLE output NRMSE at 0.000326 and the
layer-one attention input at 0.000693. At position one PLE output matches
exactly; position two drops from 0.004028 to 0.000058. CPU mean oracle KL
falls from 0.102301 to 0.031902, while V100 oracle KL remains 0.057752;
CPU-to-V100 mean KL changes from 0.032852 to 0.039455. Router-set flips
across 672 layer-position cells are 49 shared, 19 CPU-only, and 48 V100-only,
compared with 65 shared, 18 CPU-only, and 32 V100-only before the PLE fix.
These nonmonotone whole-prefix results reflect discrete routing sensitivity,
and substantial V100-only divergence remains. The next focused boundary is
layer-one attention through its MLP input: at position zero, CPU-to-V100
NRMSE grows from 0.000693 at attention input to 0.001328 at MLP input and
0.002450 at layer-two attention input. The Phase 11 §7 gate is unchanged and
still fails; do not infer a precision-only explanation until further
CPU/V100 calibration and controlled routing comparisons agree.

The [layer-one GDN output trace](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36074950425)
records all 3,472 selected stage rows. At position zero, CPU-to-V100 NRMSE
is 0.000693 at GDN input and 0.000627 at GDN output, then 0.001328 at MLP
input. The [post-attention hyper trace](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36075722015)
adds the intermediate FP32 hyper state over 14 positions. At positions zero
and two, respectively, its CPU-to-V100 NRMSE is 0.000476 and 0.000187;
MLP input is 0.001328 and 0.001253. Thus the next local reference mismatch
arises during **layer-one MLP hyper preparation**, after the GDN output and
attention injection. At positions 12 and 13, the post-attention hyper gaps
are 0.000721 and 0.000905, growing to 0.002636 and 0.002639 at MLP input.
This comparison isolates the next CPU/V100 materialization or arithmetic
boundary; it does not alone establish which result is mathematically closer
or explain the remaining whole-prefix router divergence. The V100 diagnostic
test still returns failure under the unchanged Phase 11 acceptance gate;
the selected-stage capture and comparison completed successfully.

The [unrounded layer-one MLP trace](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36078905618)
compares the mixer's FP32 result before BF16 storage with both CPU and V100.
At position zero the CPU-to-V100 NRMSE is 0.000567 before storage and
0.001328 after storage; both unrounded implementations have similar error
against the independent oracle (0.002172 CPU, 0.002131 V100). The
[storage report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36087485926)
checks all 35,840 V100 values over 14 positions: the stored block input is
exactly the BF16 round-to-nearest-even of the captured FP32 result, with zero
bit mismatches. Position-zero V100 storage NRMSE against its own FP32 value
is 0.001660. The CPU/V100 gap across this boundary is sensitive to BF16
rounding of their already different FP32 results; this is not evidence of an
incorrect V100 storage kernel.

The [controlled full-prefix router replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36087743583)
and [KL report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36088144919)
hold expert membership fixed while **recomputing routing weights from each
CPU trajectory's own scores** and carrying all state forward. The ordinary
CPU reference has mean oracle KL 0.031902, P99 KL 0.287952 and 14/14 top-1;
V100 has mean 0.057752, P99 0.545568 and 13/14 top-1. Forcing the frozen
oracle membership in CPU lowers mean oracle KL to 0.000301 and P99 to
0.001579 with 14/14 top-1. Forcing V100 membership instead gives CPU mean
oracle KL 0.059429 and CPU-to-V100 KL 0.000221 (ordinary CPU-to-V100:
0.039455). Thus expert-set choices explain most of the observed logits
separation *in this controlled 14-position replay*. Oracle membership is
offline information and cannot qualify the deployable V100 implementation;
the original Phase 11 §7 gate remains unchanged and unpassed.

The [current router map](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36088243015)
compares both implementations with the incremental FP32 router trace (whose
logits separately reproduce the frozen FP32 oracle). Across 672 cells,
49 oracle-relative expert-set flips are shared, 19 CPU-only and 48 V100-only;
CPU and V100 differ in 91 cells. At position 12 the first V100-only flip is
at layer 2, where the CPU still selects the FP32 set. A targeted V100
position-12 boundary injection checks that early mismatch.

The [position-12 V100 stage-injection replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36088333449)
recomputes 13 sequential prefixes for each of three separate oracle inputs:
`L01_attn_block_input`, `L02_attn_block_input`, and
`L02_mlp_block_input`. Each injection restores the layer-two oracle expert
set, including the final injection immediately before routing. Without
injection, layer-two V100 replaces oracle expert 173 with 393; the CPU
profile selects the oracle set. In the ordinary position-12 trace the
layer-two MLP input NRMSE against oracle is 0.004211 on V100 and 0.004478
on CPU; CPU-to-V100 NRMSE is 0.003183. Therefore overall tensor NRMSE alone
does not predict this close top-10 decision. The latest-stage intervention
shows that this **specific V100-only expert-set flip depends on its upstream
MLP input** rather than incorrect router selection given oracle input. It
does not identify which earlier storage or arithmetic boundary is
responsible. The three 13-prefix replays have mean oracle KL 0.011139,
0.012815, and 0.014112, respectively; each still fails the unchanged
Phase 11 gate (the diagnostic workflow succeeds because the three injections
and measurements completed, while each CTest returns its gate-failure code).

The [independent CPU-input V100 replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36089911902)
then supplies the **CPU precision reference's own** position-12
`L02_mlp_block_input` to the real V100 router, while comparing subsequent
stages to the unchanged independent FP32 oracle. The V100 router now selects
the oracle/CPU expert set (zero layer-two ID mismatch); a separate oracle
input positive control also selects that set. Both runs recompute all 13
prefixes and retain the unchanged Phase 11 gate: mean oracle KL is 0.014877
for CPU input and 0.014112 for oracle input, and CTest fails the gate in
both cases. Together with the ordinary V100 position-12 flip, these
controlled interventions exclude a standalone router-selection defect at
this early cell for these two matched input tensors. They do not prove
the CPU and GPU trajectories agree globally, or determine whether an
upstream V100 kernel or the candidate precision profile causes the
natural-input difference.

The [frozen position-12 score report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36090386983)
quantifies that expert swap. The independently checked FP32 score for expert
173 exceeds expert 393 by 0.001560, and the CPU profile still ranks 173
tenth over 393 eleventh by 0.001398. Natural V100 reverses their order by
0.000284. CPU-to-V100 NRMSE is zero at the layer-zero attention input,
0.001124 at its MLP input, 0.002636 at layer-one MLP input, and 0.003183
at layer-two MLP input; it jumps to 0.016588 after the layer-two MLP and
hyper output. This shows how a close expert cutoff amplifies the divergent
trajectory, but does not identify a faulty earlier operation by itself.

A [second matched-input V100 replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36090530965)
tests the earlier token position 2 at deep layer 44, where natural V100
substitutes expert 395 for the oracle/CPU expert 359. Injecting the CPU
profile's own `L44_mlp_block_input` restores the oracle/CPU expert set;
injecting the independent oracle input does the same. Both three-prefix
diagnostics still return the unchanged Phase 11 CTest gate-failure code.
Their short-prefix mean oracle KL values are 0.000086 and 0.000064,
respectively, and do not represent the complete Phase 11 qualification.
These two matched-input experiments bracket one early and one deep
V100-only router flip: in both, the router accepts the independent CPU
input and chooses its expert set. The next investigation should isolate
upstream trajectory differences at the earliest shared input boundary
and evaluate a controlled precision change across full prefixes.

The [full-prefix FP32 MLP-input ablation](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36090985820)
and [distribution report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36091445721)
test a hypothetical precision upgrade: the independent CPU reference keeps
each MLP block input in FP32 while retaining the other candidate-profile
casts and recomputing all 14 tokens with persistent rounded state. This
**worsens** mean oracle KL from 0.031902 to 0.080842 and P99 KL from
0.287952 to 0.913398; oracle top-1 agreement falls from 14/14 to 13/14.
Oracle-relative expert-set flips rise from 68 to 81 across 672 cells.
Although the upgraded CPU reference differs from V100 in fewer expert sets
(54 instead of 91), it also adopts V100's incorrect position-2, layer-44
expert substitution. The blanket FP32 MLP-input change is therefore
rejected as a corrective precision setting for this sample. No deployable
V100 precision change has been qualified, and §7 thresholds are unchanged.

### Layer-zero GDN convolution-history rounding at token 10

A [current-profile 14-position GDN comparison](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36097496115)
finds that CPU and V100 layer-zero attention inputs match exactly at position
12. The recurrent outputs differ by 1.32e-5 NRMSE, which grows to 2.66e-4
at the gated output and 0.001124 at the MLP input. A separate
[gate recomputation](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36098095302)
matches the actual V100 FP32 gate output from its captured recurrent input,
BF16 `z`, and BF16 norm weights within 6.85e-8 NRMSE at position 12. A
CPU/V100 input swap attributes most of the gate gap to recurrent heads
24–26, rather than `z` or the gate formula.

The [history and projection trace](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36098533496)
shows that these heads match the CPU within about 1e-10 RMSE through position
10 and diverge at position 11. At position 10 the CPU FP32 key projection's
element 3097 is -29.0625019, near the BF16 midpoint: the CPU model stores
-29.125 and the V100 stores -29.0. The V100 `z` mirror separately equals
BF16 round-to-nearest-even of its captured FP32 accumulator. A
[controlled CPU replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36098696999)
changes only convolution history channel 3097 at position 10 to the observed
V100 value. Position-11 total recurrent NRMSE against V100 falls from
1.39e-5 to 2.47e-6; relative errors in heads 24–26 fall from about 0.1%
to below 5e-7. This identifies one causal format-conversion difference,
amplified by low-amplitude recurrent heads and normalization. It is not an
incorrect V100 gate kernel or an independently deployable correction.

The [extended 14-position replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36099417852)
confirms that the single key-history patch does not resolve qualification.
CPU mean oracle KL changes from 0.031902 to 0.030146, while CPU-to-V100
mean KL worsens from 0.040648 to 0.047474. The position-10 KL outlier
precedes the intervention and remains 0.115576. V100 remains at 0.057752
mean oracle KL and 13/14 top-1. This local rounding tie is causally real,
but is not the main source of complete-prefix KL failure.

An [FP32 convolution-history ablation over all 14 prefixes](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36099041700)
worsens CPU mean oracle KL from 0.031902 to 0.092104, P99 KL from
0.287952 to 0.828227, and top-1 agreement from 14/14 to 12/14. Reject this
broad precision change. The ordinary CPU profile already exceeds the §7 mean
KL limit on this sample, but the CPU profile still does not reproduce all
V100 arithmetic and router decisions. These experiments establish a local
rounding cause, not a proof that all Phase 11 failure is due to unavoidable
rounding or that no alternative precision policy can pass. The unchanged
qualification gate remains unpassed. The next decision should address the
position-10 and position-13 complete-prefix KL outliers while retaining the
measured convolution-history boundary, and validate on the required larger
teacher-forced sample.

The [per-position router map](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36132036935)
shows that position 10 first diverges by a **CPU-only** expert-set change at
layer 5; V100 still selects the independent oracle set there. At position
13, the CPU's first-only change is at layer 14, and the first V100-only
change is at layer 26. The [position-13 matched-input V100 replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36132670825)
supplies the independent CPU profile's own layer-26 MLP/router input. The
V100 router then selects the oracle expert set, as it does with an oracle
input positive control. CPU-input injection lowers full-prefix mean V100
oracle KL from 0.057752 to 0.046971, but position-13 KL remains 0.464482,
top-1 remains 13/14, and expert sets diverge again at layer 30. Oracle-input
injection yields 0.048437 mean KL and also fails. The candidate router is
working on these matched inputs; cumulative upstream differences and
subsequent discrete routing choices remain. Neither intervention qualifies
the original engine. Earlier full-prefix forced-membership CPU replays
demonstrate how much router sets affect logits, but their oracle IDs are
offline information unavailable to the serving engine.

### Layer 15 QSA projection replay and input boundary

The [upstream stage extraction](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36020859990)
locates a smaller position-2 jump before the layer-20 router flip. At layer
15, the oracle-relative `qsa_gated` input has 0.00303482 NRMSE, while the
BF16 `attn_block_output` reaches 0.01295645 (8.09 times its output-rounding
floor). The [V100 oracle injection replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36021150400)
isolated that projection with two independently verified injection points:

| Position-2 layer-15 replay | Attention output NRMSE | Layer-20 expert-set flip |
| --- | ---: | --- |
| Uninjected V100 baseline | 0.01295645 | Yes |
| Inject oracle `qsa_gated` before output projection | 0.00866809 | No |
| Inject oracle `attn_block_output` after projection | Stage replaced after 0.01295645 measurement | Yes |

The three-position injection runs reached the logits check; the diagnostic
CTest still exited nonzero under the original Phase 11 gate. The different
router decisions after the two injections show sensitivity to the pattern of
upstream errors, not a monotone relationship with attention-output NRMSE.

The [independent CPU projection replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36022091981)
used the frozen oracle `qsa_gated` tensor and the model's dequantized FP8
output weight. Its full-FP32 result reproduces the frozen attention output to
1.74e-6 NRMSE. Rounding only the output to BF16 yields 0.00160245 NRMSE;
rounding the **input and output** to BF16 yields 0.00866697, within about
0.0000011 NRMSE of the V100 oracle-input replay. This explains the local
projection error through input materialization and amplification; it does
not require a defective QSA output projection kernel.

The sequential CPU storage profile now rounds the QSA output projection
input as well. Its [14-position run](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36022333498)
passed incremental FP32 parity (worst KL 2.99e-11). A new
[V100 comparison](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36022821822)
and [compact score report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36023418921)
measured:

| Measurement against FP32 oracle | CPU with QSA input BF16 | V100 candidate |
| --- | ---: | ---: |
| Mean logits KL | 0.01925582 | 0.057752 |
| Top-1 agreement | 13 / 14 (flip at position 10) | 13 / 14 (flip at position 13) |
| Expert-set flips / 672 cells | 46 | 97 |
| Median router-score NRMSE / 672 cells | 0.00036674 | 0.00047534 |
| P95 router-score NRMSE / 672 cells | 0.00697020 | 0.01317816 |

Only 33 expert-set flips are shared (13 CPU-only, 64 V100-only). The layer-20
position-2 cutoff remains 0.0002737: the corrected CPU score RMS error is
0.0023354 and retains oracle expert 489, while V100 has 0.0020022 and picks
388. This shows why a smaller aggregate error does not predict the membership
at a near tie. The profile is closer on mean KL but still not a quantitative
precision match or an acceptance floor. Remaining calibration should compare
the QSA input path and other internal materialization boundaries before
changing the Phase 11 gate.

### Complete V100 oracle-membership replay (14 positions)

A [same-build V100 diagnostic](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36140377434)
ran the natural path and then replayed all 672 expert sets from the frozen
independent stage trace. At each router the replay retained the V100's own
scores, ordered the ten oracle-selected IDs by those scores, and recomputed
softmax weights from those scores; the shared expert and all other arithmetic
remained on the actual V100 engine. Each replayed ID and weight was checked on
the device before proceeding. Its CTest passed the original short-sample gate.

| V100 measurement | Natural routing | Oracle-membership replay |
| --- | ---: | ---: |
| Mean oracle logits KL | 0.05775208 | 0.00047948 |
| P99 oracle logits KL | 0.61541675 | 0.00462245 |
| Top-1 agreement | 13 / 14 | 14 / 14 |
| Relative mean NLL delta | 0.00714438 | 0.00121680 |
| Position-13 oracle logits KL | 0.61541675 | 0.00462245 |

This establishes that the real V100 engine's arithmetic reaches the §7
logit thresholds for these 14 prefixes **when expert membership is supplied
by the oracle**. No oracle IDs are available during serving. The uninjected
candidate still fails and the required longer teacher-forced qualification
has not run. This experiment isolates discrete membership selection as the
dominant cause of the observed short-sample failure; it does not establish
whether the upstream router-input differences arise exclusively from
rounding/format conversion or include an engine implementation error.

A [first boundary report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36141905854)
combined a hyper trace from an earlier CPU profile revision with a current
GDN trace. It incorrectly suggested that the CPU-to-V100 gap grew from
0.000220 after attention to 0.001768 after injection. A
[same-profile recomputation](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36145689677)
corrects the position-12 layer-zero boundary sequence:

| CPU-to-V100 stage | NRMSE |
| --- | ---: |
| Attention input | 0 |
| Attention output | 0.00021957 |
| Hyper state after attention injection | 0.00023048 |
| BF16 MLP/router input | 0.00112440 |

Directly captured CPU pre-injection state and all four gate values reproduce
its attention injection with less than 5e-9 normalized residual per stream;
the V100 update has the same accuracy. The earlier 5e-5 to 3e-4 CPU
reconstruction residual came from mixing profile revisions and is withdrawn.
The earliest material growth on this path is now localized to **layer-zero
MLP hyper preparation**, between the matched-profile hyper state and the
BF16 MLP input. This does not identify a defective V100 kernel: the CPU
profile still approximates the selected CUDA materialization and reduction
order. Compare the intermediate normalized state, low-rank result, and
final BF16 materialization with matched inputs before changing production
arithmetic.

A [same-prefix raw-mixer comparison](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36148885273)
adds the intermediate layer-zero FP32 MLP mixer output at position 12.
CPU-to-V100 NRMSE is 0.00048936 at that FP32 output, increasing to
0.00112440 after the BF16 materialization. Each implementation's own
raw-to-stored NRMSE is about 0.00159 (CPU 0.00158538, V100 0.00159533).
The [V100 raw/stored artifact](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36147753593)
contains 2,560 finite FP32 pairs. Independent bitwise round-to-nearest-even
recalculation exactly reproduces **all 2,560 V100 BF16 stored values**;
1,290 values round upward, 1,270 downward, and four FP32 values are within
32 low mantissa bits of a BF16 midpoint. The first GPU workflow failed only
because its post-run log printer invoked unavailable `rg`; both tensors and
the 14-position natural-path failure were captured. The
[corrected GPU workflow](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36148874618)
completed successfully while reporting the unchanged natural CTest failure.

At this boundary, the extra apparent BF16 input error is an expected
format-conversion amplification of a smaller upstream FP32 difference, not
an incorrect V100 BF16 conversion. This supports the rounding-driven routing
hypothesis but does not establish that *every* V100-only expert flip has the
same cause or that no upstream engine bug exists elsewhere.

Any candidate correction must run across complete prefixes without oracle
IDs before qualification can be claimed. The unchanged §7 gate remains
failed on natural routing, and the present evidence does not prove that
rounding is its only upstream cause.

### Earliest V100-only routing change: position 12, layer 2

The [14-position stage comparison](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36055511685)
found the first V100-only expert-set change at position 12, layer 2. The
oracle cutoff margin there is 0.00155973; the CPU storage profile retains
the oracle expert set, while V100 replaces one expert. The layer-2 input
already differs from the FP32 oracle by 0.00502061 NRMSE on V100 versus
0.00369965 on CPU, and its BF16 rounding floor is 0.00166790. Layer 2 uses
GDN; the first QSA layer is layer 3.

The [V100 early-boundary replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36056206949)
injected the independent FP32 oracle at position 12, rounded to each BF16
target stage. Each of these single-stage injections restored the oracle
layer-2 expert set:

| Injected stage | Layer-2 MLP input NRMSE before injection, if applicable | Layer-2 MLP output NRMSE |
| --- | ---: | ---: |
| None (14-position baseline) | 0.00421120 | 0.08765496 |
| Layer-1 attention input | 0.00370279 | 0.00419515 |
| Layer-2 attention input | 0.00400110 | 0.00475658 |
| Layer-2 MLP/router input | 0.00421120 | 0.00241739 |

The diagnostic records its stage comparison **before** injection. Direct
router-input injection therefore leaves the reported pre-injection input
NRMSE unchanged while changing the values consumed by the router. Injection
at layer-2 attention input reduces its subsequent GDN projection to 1.07
times the output BF16 rounding floor, although the recurrent output still
differs from the oracle by 0.00221696 NRMSE with the existing state history.
This supports accumulated upstream error and a near-tie router decision as
the immediate cause of this flip. It does not isolate every earlier rounding
boundary or establish that the V100 GDN state transition matches the CPU
storage profile. The 13-position injected runs still fail the original Phase
11 gate (mean KL 0.01113936, 0.01281500, and 0.01411181 respectively).
These injection runs are causal diagnostics, not substitute qualification
scores; the 14-position uninjected V100 mean KL remains 0.05775208.

A [paired GDN replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36057324550)
held the position-12 layer-2 attention input at oracle BF16 values and then
injected one more oracle stage. Injecting the FP32 recurrent output reduced
the gated-output NRMSE from 0.00340667 to 0.00180051 and the attention-output
NRMSE from 0.00328082 to 0.00155192. Injecting the FP32 gated output instead
reduced attention-output NRMSE to 0.00000024: the V100 output projection
reproduced the independent oracle when given the oracle's gated input. The
recurrent-output discrepancy contributes substantially to this layer's local
output error, while other upstream gate/input errors remain. These injections
cannot distinguish recurrence arithmetic from the pre-existing GDN state
history. Both runs retained the oracle expert set but still failed Phase 11
(13-position mean KL 0.01281534 and 0.01281434, respectively). Further
calibration should compare recurrence with matched input **and** matched
prior state before attributing the difference to a kernel or precision
profile; the output projection at this boundary needs no further replay.

### Prepared-weight parity at early divergent layers

The [prepared-artifact audit](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36172279707)
compared the pinned source safetensors with the actual `.ninfer` artifact on
the V100 host. All 16 selected planes matched byte for byte: at both layers
0 and 2, the BF16 router and three MLP hyper-connection weights, plus the
FP8 GDN fused projection and output codes and their FP32 row scales. The
comparison validated each object's artifact shape, format, and layout before
reading its payload. No weight bytes were published as Actions artifacts.

The [device-resident audit](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36173240610)
subsequently passed with 32,491 MiB free on the first check. All the same
16 selected planes matched source bytes after loading onto the V100. The
previous [attempt](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36170022463)
timed out in its memory-wait step, but that workflow did not print its
memory readings; its reason for seeing no passing reading is unknown. A
[focused runner probe](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36173113607)
also saw 32,491 MiB free on GPU 0, the V100. The probe itself exited early
because it queried a GPU 1 that this runner cannot see. The device audit now
logs each memory reading.

This rules out source-to-artifact and artifact-to-device byte differences
for these early objects. It does not verify every expert or later-layer
weight, or the reduction and rounding rules used when the V100 consumes
them. The next causal comparison should hold the position-12 layer-zero
attention output and hyper master state equal between CPU and V100, then
compare their FP32 MLP mixer before the BF16 storage cast. Retain the
unchanged Phase 11 qualification gate.

### Layer-zero position-12 matched-input GDN localization

The [fresh CPU precision trace](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36177462927)
completed all 14 positions. Its mean FP32-oracle KL was 0.031902. The
[matched hyper-state replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36176024960)
and [matched attention-output replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36176940353)
each ran the real V100 model through a 13-position prefix. CPU and V100
position-12 layer-zero attention inputs match exactly. Replacing the V100
attention output with the CPU reference's value reduces the next FP32 hyper
state NRMSE from 0.000230481 to 0.000000299 and the unrounded MLP mixer
NRMSE from 0.000489360 to 0.000000340. Replacing the hyper master directly
reduces the mixer NRMSE to 0.000000281. The natural difference at that point
arrives from GDN attention; these replays do not establish its cause.

The [GDN boundary report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36177837422)
records position-12 CPU-to-V100 NRMSE of 0.001305950 for the fused projection,
0.000021222 for z, 0.000013230 for the recurrent output, 0.000266225 for
the gated output, and 0.000219575 for the attention output. Exact equality
of the attention input does not guarantee bitwise equality of FP8 projection
accumulators, since CPU and CUDA reductions can have different arithmetic
orders. The earlier byte audit covers the relevant device-resident weights.

The [recurrent/gated replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36178050742)
supplied independent CPU position-12 recurrent output to V100 and reduced
gated-output NRMSE to 0.000041995. Supplying the CPU gated output instead
reduced attention-output NRMSE to 0.000000415. The
[z/recurrent replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36178778784)
then supplied CPU z alone, reducing gated-output NRMSE only to 0.000262892;
supplying both CPU z and CPU recurrent output reduced it to 0.000000092.
These stage files are captured *before* replacement, so the injected stage's
own recorded row still shows the baseline difference. The downstream values
show that, for these matched inputs, the V100 output gate closely reproduces
the CPU result. The remaining natural gate difference originates mainly
upstream of that gate, in the recurrence input/history or its arithmetic.

Every diagnostic V100 CTest above still exits 8 against the **unchanged**
Phase 11 qualification gate. Workflow success means its prefix completed and
the specified injection occurred, not that the model qualified. Next compare
the CPU and V100 layer-zero convolution query/key/value at position 12, then
hold the previous GDN state equal to distinguish accumulated storage/reduction
differences from a recurrence implementation error. No precision floor or
general kernel correctness conclusion follows from this one position.

The [matched GDN convolution-stage report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36182369693)
compares the CPU profile with the unmodified V100 13-position prefix at the
same position. The CPU query/key/value tensors are reconstructed from the
captured FP32 QKV projections and BF16-rounded prior convolution history;
the CPU decode uses an optimized state update that bypasses Conv1d hooks.
The V100 stage trace is captured directly from its FP32 convolution output.
Query NRMSE is 1.20e-7, key is 0.000676006, and value is 7.12e-8. Thus
the earliest large convolution-output difference on this path is localized
to key; this reconstruction alone cannot prove that the CPU implementation
uses the identical accumulation order at every element.

The [matched key replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36182817150)
captures and replaces the V100 key **before** its recurrent update, using
the independent CPU value at position 12. The CPU-to-V100 recurrent-output
NRMSE falls from 1.32302e-5 to 4.65489e-6, gated-output NRMSE from
0.000266225 to 0.0000465767, and attention-output NRMSE from 0.000219575
to 0.0000452091. The diagnostic CTests both exit 8 against the unchanged
Phase 11 gate. This intervention connects the local key mismatch to most of
the position-12 recurrent and gate mismatch. It is consistent with the
previously observed position-10 BF16 midpoint tie in a key-history channel;
it does not rule out additional recurrent-state, reduction-order, or later
implementation differences over full prefixes. Next hold prior recurrent
state and convolution history equal when assessing the remaining gap.

The [CPU prior-state probe](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36183463774)
found a finite FP32 layer-zero recurrent state of shape (1,48,128,128)
just before position 12. Its raw flattening is transposed relative to the
V100 source-state tensor: the [layout report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36184878400)
measures raw NRMSE 1.41145, falling to 1.37176e-5 after transposing each
128x128 matrix. The [hosted conversion](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36184956447)
validates this orientation and exports a separate converted CPU artifact.
An earlier untransposed [injection](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36184223882)
read back the supplied bytes but destroyed the recurrent result; those
numbers reflect an invalid state layout and are excluded from attribution.

The [validated state and key replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36185004001)
runs three complete 13-position V100 prefixes. The natural CPU-to-V100
recurrent-output NRMSE is 1.32302e-5. Replacing only the prior FP32 state
reduces it slightly to 1.22752e-5; replacing both prior state and key
reduces it to 8.19e-8. The state, key, and recurrent stage files are captured
before their diagnostic replacement; downstream differences measure the
interventions. With both inputs matched, gated-output NRMSE remains
4.20e-5, consistent with the separate z-input difference: supplying both
CPU z and CPU recurrent output had reduced gated-output NRMSE to 9.21e-8
in the earlier replay. This local agreement rules out a sizable recurrence
arithmetic discrepancy at this boundary under the matched-input profile.
The state and key interventions require offline CPU values and are not
candidate implementation changes. Mean oracle KL over the short prefixes
was 0.01485480 for the natural run, 0.01377233 with state injected, and
0.01281531 with state plus key; all underlying CTests still failed the
unchanged Phase 11 gate. The remaining complete-prefix KL and router-set
errors need a deployable precision or implementation correction.

### Position-ten CPU calibration is not a V100 precision floor

The [frozen position-ten router report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36186062340)
finds the first CPU-only expert-set change at layer 5: FP32 ranks expert
120 over 136 by 0.001184, the CPU storage profile reverses them by
0.000965, and V100 still ranks 120 above 136 by 0.000445. The CPU and V100
layer-zero attention and MLP inputs are bitwise identical at position 10;
their layer-zero MLP outputs differ by only 6.32e-8 NRMSE. At the layer-five
MLP input, CPU-to-V100 NRMSE is 0.003046, and the CPU layer-five post-MLP
hyper output differs from the oracle by 0.020302 versus V100's 0.003207.
Among 48 position-ten router cells, CPU differs from the FP32 set in 18
and V100 in 7; the first V100-only expert change occurs at layer 27.

The [three-way per-position KL report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36186513521)
shows position-ten oracle KL 0.115576 for CPU but 0.074579 for V100.
At position 13 CPU has KL 0.313710 and retains the oracle top token, while
V100 has KL 0.615420 and changes the top token. Consequently the CPU
profile's position-ten outlier is not a demonstrated lower bound for the
V100 implementation. These measurements prioritize calibrating early
CPU-only router decisions and testing whether a deployable precision change
improves the full prefixes; they do not relax the §7 gate.

The [single-layer FP32 MLP-input ablation](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36186371670)
recomputes all 14 CPU prefixes with only layer one's MLP input kept in
FP32. It restores the oracle layer-five expert set at position 10 and lowers
that position's oracle KL from 0.115576 to 0.104652. Nevertheless, mean
oracle KL worsens from 0.031902 to 0.089724, top-1 agreement falls from
14/14 to 12/14, and position-13 KL rises from 0.313710 to 1.101900.
This narrow materialization change is rejected as a corrective precision
policy for the sample, just as the earlier all-layer FP32 input ablation was.
A locally restored expert set does not establish a full-prefix pass.

The [14-position V100 state-and-key continuation](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36186612932)
repeats the position-12 diagnostic and includes position 13. Mean oracle KL
improves from 0.057752 to 0.050645, but top-1 remains 13/14 and both
underlying CTests fail the unchanged Phase 11 gate. The intervention uses
offline independent CPU state and key values, so even its KL improvement is
not a deployable correction. It also does not explain the final position's
incorrect top token by itself.

The [frozen position-13 router report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36188655073)
finds the first V100-only expert-set swap at layer 26: V100 selects 488
where the FP32 oracle and CPU profile select 252. Their 252-minus-488
score margins are +0.025909, +0.020459, and -0.029846, respectively;
CPU-to-V100 MLP-input NRMSE there is 0.067561. CPU-only changes already
occur by layer 14 at that position, and 6 of V100's 13 oracle-relative
router-set changes are V100-only. A matched CPU-input replay at layer 26
is the next causal check, while the complete-prefix gate remains unchanged.

A [matched position-13 replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36188766095)
supplies the independent CPU layer-26 MLP input to the actual V100 router
over all 14 positions. Its selected expert set changes from the V100-only
expert 488 to the CPU/oracle expert 252. Mean oracle KL improves from
0.057752 to 0.046971, but top-1 remains 13/14; the unchanged CTest
qualification gate still fails. The intervention uses offline CPU values.
It shows that the router chooses the expected experts on the matched input,
while upstream differences and subsequent choices still require diagnosis.


### Position-13 upstream map and scoped membership experiment

A [frozen stage report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36208272784)
maps the complete position-13 path through layer 26. CPU and V100
layer-zero attention inputs are identical. Their MLP inputs differ by
0.000912 NRMSE at layer zero and 0.008951 by layer 14, before the
CPU-only swap of oracle expert 46 for 70. The CPU's post-MLP hyper
output then differs from the oracle by 0.018615 NRMSE while V100
remains at 0.005894. The CPU has another oracle-relative expert swap
at layer 25 (457 to 140), which V100 does not. By layer 26 the CPU and
V100 MLP inputs differ by 0.067561 NRMSE, but that distance contains
CPU-only routing errors; it is not a measure of an isolated V100 defect.
The first V100-only swap at layer 26 still warrants a causal test on
the V100's own path.

The [scoped V100 membership replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36208527262)
uses the V100's own router scores and the frozen oracle IDs. The natural
14-position path has mean oracle KL 0.057752, P99 0.615417, and 13/14
top-1. Replacing only position-13 memberships (48 routers) yields
0.040423 mean KL and 0.372807 P99, while replacing only layers 26–47
there (22 routers) yields 0.040028 mean KL and 0.367278 P99. Both
remain 13/14 top-1 and **both CTests fail the unchanged gate**. This
rejects position-13-only routing correction as sufficient even with
offline oracle IDs. Earlier full-prefix oracle-membership replay over
all 672 routers reached 14/14 and passed the short gate, so the next
diagnostic isolates the earlier-token routing history carried into
position 13. All offline replays are unavailable at serving time.

The [prior-prefix V100 replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36208937548)
narrows the causal history. Oracle expert IDs only at positions 0–12
(624 routers), with completely natural position-13 routing, restore
14/14 top-1 and reduce mean KL to 0.015182 and P99 to 0.210462;
the unchanged CTest still fails on KL. Replaying those prior 624
routers plus position-13 layers 26–47 (646 in total) reduces mean KL
to 0.00022874 and P99 to 0.00111214; its CTest **passes the unchanged
14-position short-sample gate**. Neither result qualifies the natural
engine, and neither proves whether the upstream differences were
caused exclusively by rounding. In contrast, replays restricted to
position 13 retained the wrong top token. Routing decisions in earlier
tokens therefore affect the state reaching the last token materially.
The next diagnostic splits the prior history to find which positions
carry the strongest effect.

The [split history replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36209250525)
keeps the position-13 layer-26–47 oracle memberships constant in both
cases. Adding oracle memberships only for prior positions 0–6 yields
mean KL 0.036602 and 13/14 top-1. Adding them instead for positions
7–12 yields 0.00108463 mean KL, 0.00823096 P99, and 14/14 top-1.
Both CTests still fail the unchanged short gate; the 7–12 case also
has relative mean NLL delta 0.00793208. The later prior positions
carry most of the observed final-token routing-history effect in this
controlled replay. Correcting *all* previous positions still matters
for the short gate's aggregate requirements.

The [three-token split](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36209558173)
holds the same position-13 layer-26–47 membership replay fixed.
Correcting only prior positions 7–9 gives mean KL 0.038098,
P99 0.501937, and 13/14 top-1. Correcting only positions 10–12
gives mean KL 0.008119, P99 0.046193, and 13/14 top-1.
Both CTests fail. The earlier combined 7–12 replay reached 14/14
top-1 and 0.001085 mean KL, so neither three-position window alone
is sufficient; their effects interact through the running model state.

The [frozen prior-position router map](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36209860166)
finds 38 V100-only, 16 CPU-only, and 35 shared oracle-relative
changed cells across positions 7–12. Their earliest V100-only swaps
are position 7/layer 31 (V100 cutoff score gap 0.000587),
position 9/layer 20 (0.001047), position 10/layer 27 (0.002636),
position 11/layer 15 (0.004623), and position 12/layer 2
(0.000284). Position 8 has no V100-only swap, though it has a
shared oracle-relative layer-40 swap. The input differences before
these first decisions are much smaller than after their downstream
cascades: for example position 9/layer 20 V100-oracle MLP input
NRMSE is 0.004960 and the next layer is 0.049818.
This map is descriptive; cutoff gaps on the V100 path alone cannot
establish whether each upstream difference is solely format
conversion rather than an implementation difference.

### Precision-profile calibration at the first position-seven flip

The [five-boundary three-way report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36210246925)
compares FP32 oracle, independent storage profile, and V100 at their
first V100-only router changes. For position 7/layer 31, oracle expert
220 leads 385 by only 0.0002222 score units; the CPU profile leads by
0.0061741, while V100 reverses them by 0.0005865. The CPU and V100
MLP inputs already differ by 0.0098006 NRMSE before routing. Similar
pre-router differences occur at positions 9, 10, 11, and 12. The
reference's agreement on the expert set alone does not establish that
it matches the V100 precision trajectory.

A [matched-input V100 replay](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36210352857)
injects the CPU profile's independent position-7/layer-31 input into
the real router. V100 then selects the same ten experts, and its 512
scores differ from the CPU profile by 5.03e-7 RMS (7.62e-8 normalized).
The natural V100 score RMS difference from the CPU profile is 0.005995.
This verifies the local router arithmetic on the matched profile
input and places this expert swap upstream of the router. The injected
complete-prefix mean oracle KL **worsens** from 0.057752 to 0.060953;
both underlying CTests fail the unchanged gate. This offline injection
is a calibration check, not a precision upgrade.

The first visible separation on the natural position-7 path is much
earlier. The [corrected upstream report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36210663754)
shows bitwise-identical CPU and V100 layer-zero attention and MLP inputs
and layer-zero router scores agreeing within 8.71e-8 NRMSE. An
[early-boundary report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36210712498)
measures position-7 PLE injection at 1.73e-6 CPU-to-V100 NRMSE, but
layer-one stored attention input at 0.0001160 and layer-one MLP input
at 0.0015369. A separate
[sum check](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36210798630)
reconstructs the post-PLE master state from exported layer-zero master
and PLE injection at 1.27e-6 NRMSE; 16 of the 2,560 stored layer-one
attention input values differ. This sum is a diagnostic approximation
of the internal update, not a direct capture of both live master states.
An initial upstream workflow
([invalid report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36210629008))
mistakenly read position-13 CSV rows while labeling them position 7;
its numbers are excluded.

The [independent CPU raw-attention run](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36211013815)
preserves the modeled profile's mean oracle KL at 0.031902. Its
[position-7 conversion report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36211671912)
shows every one of the 2,560 stored layer-one attention-input values
matches BF16 round-to-nearest-even of the profile's raw FP32 mixer
output. Exactly 16 stored values differ from V100. Some span more
than one BF16 step; without V100's raw mixer values they cannot be
attributed solely to near-midpoint rounding. The first two
[V100 raw-capture jobs](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36211500054)
completed the natural CTest but did not upload the new stage: the
initial hook covered only decode, and the next run omitted the
stage from the test dumper's explicit selection list. The corrected
GPU trace is pending. These are capture errors, not evidence of a
V100 arithmetic fault or a change in the natural model result.

The [completed V100 raw capture](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36212350682)
stores all 2,560 values after the prefill hook was added to the
test dumper's allowlist. All stored values exactly match BF16
round-to-nearest-even of the captured V100 FP32 output, and natural
CTest still fails with mean KL 0.057752. The
[three-way raw comparison](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36212649875)
measures CPU-profile-to-V100 **raw** attention mixer NRMSE
9.6164e-5, increasing to 1.1601e-4 after BF16 storage. Both paths
round all 2,560 captured values correctly, but their FP32 mixer
outputs already differ. The largest raw difference, at output index
750, is 0.0019360; output index 2519 differs by 0.00022493.
Conversion amplifies part of the gap and does not cause all of it.

The [preparation-input reconstruction](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36212703851)
sums each implementation's captured FP32 layer-zero master and BF16
PLE injection. Its CPU–V100 FP32 state NRMSE is 1.2702e-6, yet
rounding those reconstructed states to the BF16 shadow yields **six
different values out of 10,240**. This is consistent with an upstream
near-boundary materialization feeding a different attention mixer
calculation, but the reconstruction is not a direct matched-input
experiment. Inject the independently reconstructed CPU FP32 master
at the V100 post-PLE boundary before attention preparation, then
compare raw mixer outputs on complete prefixes. If the large raw
differences persist on matched shadows, the independent CPU
arithmetic model or the V100 mixer implementation needs further
calibration.

The [matched post-PLE state experiment](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36212825856)
injects the independently reconstructed CPU-profile FP32 master into
the V100 at position 7, after PLE and before the layer-one attention
preparation, while evaluating all 14 prefixes. The natural V100 raw
mixer differs from the CPU profile by 9.61637e-5 NRMSE; with matched
post-PLE master, it differs by only 4.91144e-8. The local V100 mixer
therefore reproduces the independent CPU calculation on matched input,
and the measured natural raw divergence comes from its upstream state
on this position. The intervention worsens whole-prefix mean oracle KL
from 0.057752 to 0.094542 and top-1 from 13/14 to 12/14; both CTests
fail the unchanged Phase 11 gate. This is a causal calibration test,
not a serving-time correction. The [frozen split-state report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36213965871)
uses the four complete-prefix V100 traces from
[the split run](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36213259787).
Replacing only the post-layer-zero FP32 master with the CPU-profile
master while keeping V100 PLE reduces position-7 raw attention NRMSE
from 9.61637e-5 to 4.91144e-8, gives exactly matching BF16 attention
storage, and removes all six reconstructed BF16 shadow mismatches.
Replacing only PLE while keeping the V100 master leaves the raw NRMSE
at 9.61637e-5 and all six shadow mismatches. Both individual-swap
complete-prefix CTests fail the unchanged gate: mean KL becomes
0.108451 with CPU master and 0.118027 with CPU PLE, versus the natural
0.057752. A first split workflow had a wrong path for its final
matched-mode diagnostic and failed after the individual swaps;
the corrected run and hosted report provide the comparable results.
The [layer-zero master report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36214461947)
compares newly captured CPU-profile and V100 stages at the first
position-7 divergence. Both attention and MLP input vectors match
exactly (2,560/2,560 values each). The post-attention FP32 hyper
master already differs by 6.24941e-7 NRMSE (largest absolute
difference 1.00583e-7). The MLP output differs in only 2 of 2,560
values, with NRMSE 1.23360e-7; the subsequent post-MLP FP32 master
differs by 3.66576e-7. The [layer-zero GDN report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36245498853)
places the first visible difference before the hyper update: GDN
attention block output differs by 5.74873e-7 NRMSE on bitwise-identical
attention inputs. Its query/key/value tensors differ only on the order
of 1e-7 to 4e-7; recurrent output differs by 3.53406e-6 and final
attention output by 5.74873e-7. The raw CPU versus BF16 V100
`gdn_projected` comparison gives a misleading 0.001250 NRMSE:
[the projection-split report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36245598894)
shows that after BF16 rounding only two of the 10,240 convolution
projection mirror values differ, and the 6,144-value z segment matches
exactly. At index 4978, CPU FP32 0.070556640625 falls exactly halfway
between BF16 0.0703125 and 0.07080078125; CPU ties to even down and
V100 stores up. In this diagnostic path the V100 fused convolution
uses its FP32 accumulator for the current token and writes the BF16
mirror separately for history, so those mirror differences alone do
not explain the current-token GDN output. The [matched GDN-output experiment](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36245710491)
injects the CPU-profile FP32 attention block output into V100 at
position 7/layer 0. The post-attention hyper-state difference falls
from 6.24941e-7 to 2.09458e-7 NRMSE, post-MLP master from
3.66576e-7 to 2.57289e-7, and layer-one raw attention from
9.61637e-5 to 1.34904e-5. A remaining matched-output difference
at the hyper operation therefore contributes independently of the
GDN output. The 14-prefix mean KL worsens from 0.057752 to
0.093440, top-1 falls from 13/14 to 12/14, and both CTests fail
the unchanged gate. The [CPU post-attention master injection](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36246000944)
sets position-7/layer-zero V100 hyper state to the independent CPU
profile after attention. The downstream post-MLP master still differs
by 1.17502e-7 NRMSE and layer-one raw attention by 8.19131e-6.
The 14-position top-1 rises to 14/14, but mean KL 0.054267 still
fails the unchanged gate. The test dumper exports the pre-injection
stage, so its reported post-attention NRMSE remains the natural
6.24941e-7. The [two-boundary matched run](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36246240789)
supplies the CPU post-attention master and BF16 MLP output. The
post-MLP master difference drops only from 1.17502e-7 to
8.48047e-8 NRMSE; the layer-one raw attention NRMSE stays exactly
8.19131e-6. Its 14-position top-1 remains 14/14, but mean KL
0.057063 fails the unchanged gate. The
[shadow residual report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36246577156)
reconstructs six differing BF16 shadow entries on the natural path,
versus just one (index 5609) after either matched post-attention
experiment. At that index CPU post-PLE sum is 0.000844955444336;
the matched V100 sum is about 4.66e-10 lower and rounds to the
adjacent BF16 value. This tiny residual is upstream of the layer-one
attention mixer and survives matching the MLP block output.
The [matched-input gate report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36249192885)
compares four CPU-profile and V100 FP32 MLP injection scales using
[the corrected CPU](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36247796256)
and [V100](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36247843768)
complete-prefix traces. For stream 2, CPU uses 0.734752237797
and V100 0.734752178192, one FP32 ULP apart. At index 5609 both
receive matched FP32 pre-MLP state -0.00347781414166 and BF16 MLP
output 0.0069580078125. V100 post-MLP state
0.00163459731266 exactly equals FP32 fused multiply-add of those
inputs and its captured gate. CPU post-MLP state 0.00163459777832
equals an FP32 multiply rounded before the add. With even the CPU
gate, substituting FP32 fused multiply-add yields
0.0016345976619 and the lower BF16 post-PLE shadow word 14941;
the separate CPU arithmetic yields upper word 14942. The V100
gate's one-ULP difference adds to the master-state gap, but the
unmodeled fused update alone is sufficient for this shadow flip.
This is an explicit CPU precision-profile arithmetic mismatch;
the captured V100 update behaves as its implemented fused operation
specifies. It does not explain every later routing decision or
qualify the natural engine. The optional CPU fused-hyper reference
[completed a direct independent-oracle run](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36250285355)
with mean KL 0.0623346, versus 0.031902 for the standard CPU
storage profile and 0.057752 for natural V100. Neither CPU variant
qualifies natural V100. The
[frozen 14-position V100 comparison](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36251007048)
shows why the FMA arithmetic matters to **calibration** despite the
worse oracle KL: mean KL between V100 and the CPU profile drops from
0.040648 to 0.013224, and matching V100 expert sets rise from
581/672 to 623/672. Position 7 has no residual expert-set mismatch
under the FMA profile, yet its V100-to-profile logits KL is 0.002506
and the layer-one raw attention input remains about 9.58e-5 NRMSE
apart. These are complete-prefix tests; a local FMA correction is
insufficient to make the reference match V100 quantitatively or pass
the oracle gate.

The [residual router map](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36251051145)
finds 49 different router sets; its earliest position-zero mismatch
is layer 10. At that cell CPU FMA swaps oracle/V100 expert 429 for
321, with 0.002406 input NRMSE against V100. The
[position-zero stage report](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/36251155315)
shows identical CPU FMA and V100 layer-zero attention inputs, but
0.00004283 MLP-input NRMSE and 0.00019583 post-MLP master NRMSE
before the layer-10 expert change. Their individual distances from
the independent FP32 oracle at this early boundary are similar.
A dedicated position-zero GDN/hyper trace tests where the CPU FMA
profile still diverges before routing. Natural V100 Phase 11
remains unqualified.

## 1. Environment Setup

Run the setup script using Python 3.14 to create the isolated virtual environment:

```powershell
.\setup_env.ps1
```

This creates the isolated venv at `E:\NInfer\venv-qwen4exp` with PyTorch (CPU), safetensors, numpy, and transformers.

## 2. Generating State Dumps

### Deliverable A (NInfer C++ Engine Dump)
Generate raw stage tensor dumps for single token or prompt execution using the reference tool:

- **Single Token Execution (e.g. `<|im_start|>` token 248045):**
  ```powershell
  $env:PATH = "P:\third_party\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;" + $env:PATH
  .\build-win\tools\reference\qwen3_8_flash_next\ninfer_qwen3_8_flash_next_reference.exe `
    -m E:\NInfer\qwen3_8_flash_next.ninfer `
    --execute-token --token-id 248045 `
    --dump-states P:\dumps\ninfer_tok0
  ```

- **Chat Diagnostic Prompt Prefill:**
  ```powershell
  .\build-win\tools\reference\qwen3_8_flash_next\ninfer_qwen3_8_flash_next_reference.exe `
    -m E:\NInfer\qwen3_8_flash_next.ninfer `
    --chat-diagnostic --prompt "Hello" `
    --dump-states P:\dumps\ninfer_chat
  ```

### Deliverable B (Python CPU Oracle Dump)
Generate reference tensors for the same token(s):

```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe run_oracle.py `
  --token-id 248045 `
  --dump-states P:\dumps\oracle_tok0
```

For multi-token sequence:
```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe run_oracle.py `
  --ids "248045,846,198,20206" `
  --dump-states P:\dumps\oracle_seq
```

### Phase 11 logits-only acceptance oracle

For the frozen >=4096-position Phase 11 teacher-forced oracle, use the dedicated generator rather than `--dump-states`. The generator creates a deterministic 4096-position corpus with explicit instruction/chat, code, ordinary prose, reasoning/math, and long-context ranges; runs the independent CPU FP32 oracle without intermediate-state hooks; validates every logits file; and only then publishes the finished artifact.

On the physical V100 host, from a checkout containing these oracle tools:

```bash
tools/reference/qwen3_8_flash_next/oracle/generate_phase11_oracle.sh
```

Defaults are the pinned host assets:

```text
Python: /home/$USER/ninfer-v100/venv-oracle/bin/python
mixed:  /srv/ninfer/source/mixed
PLE:    /srv/ninfer/source/ple/ples_int4
output: /srv/ninfer/oracle/phase11
```

The final directory contains `token_ids.json`, `manifest.json`, one FP32 logits file per teacher-forced position, and `provenance.json`. The validator requires contiguous positions, exact token-ID agreement with the frozen corpus, one FP32 logits tensor per position, and exact file sizes.

If a previously published `/srv/ninfer/oracle/phase11` must intentionally be replaced, rerun with `--replace`; the old artifact is renamed to a timestamped backup before the new one is published.

For lower-level use, the equivalent manual oracle invocation is:

```bash
~/ninfer-v100/venv-oracle/bin/python \
  tools/reference/qwen3_8_flash_next/oracle/run_oracle.py \
  --model-dir /srv/ninfer/source/mixed \
  --ple-dir /srv/ninfer/source/ple/ples_int4 \
  --ids-file /srv/ninfer/oracle/phase11/token_ids.json \
  --dump-logits /srv/ninfer/oracle/phase11 \
  --logits-chunk-size 8
```

Keep the existing `--dump-states` mode for small detailed divergence investigations; it is intentionally not the full-oracle format.

### MTP reference qualification

`mtp_reference.py` implements the MTP stem and wraps one Transformers decoder layer,
following the [vLLM MTP implementation](https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/nvidia/mtp.py).
The hidden normalization covers all 10,240 features, its shared projection preserves
four distinct streams, and the sole mixer follows the decoder layer. The first draft
pairs the last target hidden with the target's next token; later drafts carry MTP hidden.

Run the focused comparison from the repository root with the environment above:

```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe tests/targets/qwen3_8_flash_next/test_mtp_oracle_parity.py `
  --ninfer-exe build-win/tests/Release/ninfer_qwen3_8_flash_next_mtp_test.exe `
  --dump-dir profiles/bench/flash-next-mtp/oracle-parity
```

This compares the five stem stages against the FP32 reference under shared synthetic
inputs and weights, and checks the remaining C++ stages for finite, nonzero values.
It does not compare decoder-layer or head values: their synthetic weights differ.
Zero indexer counts are legitimate diagnostics. `--stem-only` runs the pure tensor
semantics check; full mode fails explicitly if Qwen4Exp Transformers is unavailable.

The September 6 C++ comparison passed with maximum stem relative L2 error 0.002428.
The synthetic Transformers MTP forward also ran successfully. `run_oracle.py --mtp-real`
supports real-weight teacher seeding and chained drafts, but full real-checkpoint CPU
MTP parity has not been run and is not implied by those focused checks.

## 3. Comparing States and Finding First Divergence

Run `compare_states.py` to compare stage tensors across positions:

```powershell
E:\NInfer\venv-qwen4exp\Scripts\python.exe compare_states.py `
  P:\dumps\ninfer_tok0 `
  P:\dumps\oracle_tok0 `
  --threshold 0.05
```

The script reports $\max |d|$, $\text{rel-L2}$, and $\text{cosine}$ similarity per stage and immediately isolates the exact layer and operator where the numerical divergence starts.
