# SV7 stage attribution: MoE dominates, but prior screen disabled grouped CPU experts

Run [37862479691](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37862479691) succeeded at SHA `7ba0e0670f6130a4a6c358b15cbdc9074ebc5d6f`, artifact `11586852374`. Stage ledger is invasive and its CUDA event intervals include host stalls and stream waits. It spans multiple request chunks (336 MoE stage calls), **not solely cold-prefill TTFT**.

| Stage (aggregate instrumented elapsed) | BF16 control | FP16 TensorOp |
|---|---:|---:|
| MoE reduce (includes host wait and CPU expert work) | 23297.79 ms | 23722.34 ms |
| Embedding/staging | 813.92 ms | 813.67 ms |
| GDN qkvz projection | 721.07 ms | 715.61 ms |
| QSA projection | 241.68 ms | 240.53 ms |
| All stage intervals | 26856.35 ms | 27162.07 ms |

MoE reduce is approximately 87% of stage time in both arms. The earlier production HTTP run [37858456604](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37858456604) found no meaningful cold-prefill speedup with FP16 tensor cores despite 3–89× isolated BF16 projection speedups. This stage attribution explains why.

**Important qualification discovered by source audit:** `tools/diagnostics/v100_sv2_serve.py` previously enabled `NINFER_V100_CPU_EXPERT_GROUP=1` in the SV7 production timing screen, **but not** in the SV7 stage-attribution screen. Thus the 23-second MoE observation is specifically a **non-grouped CPU expert** workload, and cannot be treated as the grouped production baseline or used to quantify how much MoE remains after grouping. The new stage-attribution screen now explicitly compares `cpu-cache-single`, `cpu-cache-grouped`, and `stream-grouped` while holding BF16 projections fixed and retaining 64 GPU cache slots/layer and device combine.

**Next:** Run one protected three-arm V100 stage attribution, compare the MoE reduce interval and total, then select a representative end-to-end HTTP A/B configuration based on actual improvement. Avoid further SV7 projection micro-optimization, leave production defaults unchanged, do not claim independent Phase11 quality qualification.
