# SV7 production HTTP BF16 SIMT versus opt-in FP16 TensorOp: no prefill gain

Hardware run [37858456604](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37858456604), SHA `918dd265b31e4386356141963880d85cc8488e64`, artifact [11586070975](https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37858456604/artifacts/11586070975). **Success**, 12/12 server arms (three fresh processes per mode and MTP setting), same 64-slot/layer identity-bound resident expert profile, grouped CPU experts, one active request, 1220-token cold prompt, four HTTP requests per process (cold, prefix replay, continuation, continuation replay). No model writes; original production defaults unchanged. Actual opt-in dispatch verified in every FP16 candidate server log, e.g. `sv7.fp16_tc.dispatch=1 n=10240 k=2560 t=1220`; no dispatch marker in control.

| Metric (medians across fresh processes) | BF16 SIMT | FP16 TensorOp | Difference |
|---|---:|---:|---:|
| No MTP cold prefill tok/s | 81.445 | 81.315 | -0.16% |
| No MTP cold TTFT s | 15.068 | 15.092 | +0.024 s |
| No MTP cold decode tok/s | 8.957 | 9.121 | +1.83% |
| MTP cold prefill tok/s | 79.181 | 79.920 | +0.93% |
| MTP cold TTFT s | 15.499 | 15.356 | -0.143 s |
| MTP cold decode tok/s | 7.844 | 8.341 | +6.34% |

**Finding:** The previously observed isolated BF16 projection acceleration (up to 89×) **does not yield a measurable cold-prefill improvement** in this real-model screen. Candidate executed, so absence of improvement is not explained by complete dispatch failure. The benchmark does not attribute where the remaining time is spent; plausible factors include other kernels, expert memory/CPU execution, and synchronization, but those are hypotheses until measured. The 1–6% decode differences need targeted repeat to separate speed from run-to-run noise.

**Numerical limitation:** Exact greedy response signatures **differ between BF16 control and FP16 candidate on every paired fresh-process observation**; MTP draft/accept counts also differ (e.g. control 77 drafted/36 accepted vs candidate 74 drafted/38 accepted on the cold request). Same-path replay and accounting passed. This A/B deliberately treats cross-arm text changes as diagnostic, not a quality qualification. **Do not promote FP16 TensorOp to the default** without independent Phase11 full-model numerical qualification.

**Next performance experiment:** Use bounded stage-level attribution (PLE projection versus CPU MoE, PCIe, GDN, QSA and other stages) on a production cold prefill, avoiding unbounded traces or large model artifacts. Focus on the dominant measured stage before another kernel micro-optimization. Keep SV7 opt-in, one protected V100 run at a time; no Phase18/SV8 or broad quality campaign.
