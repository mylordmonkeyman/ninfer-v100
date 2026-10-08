# V100: MTP draft depth and hybrid decode A/B

**Completed run:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37792322111  
**Evidence:** `v100-ninfer-mtp-ab-37792322111-1`, artifact ID `11559000470` (46,704,143 bytes compressed).  
**Hosted software checks:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37792322080 — passed.  
**Frozen source:** `9af92d60355b33f64eb0242240a0803213b531fd` on `perf/v100-strata-derived`.

Five serial NInfer-only cases used the **same resident model, mixed quantization, expert profile and 156 GPU slots per layer**, GPU prefill policy `auto` threshold 256, adaptive `lru`. Five warmups, three measured long (585 input) and three short (63 input), with **192 generated tokens per request**. Native timings and HTTP wall in seconds (medians):

| Case | Long wall | Long native prefill | Long native decode | Short wall | Short native prefill | Short native decode |
|---|---:|---:|---:|---:|---:|---:|
| **MTP draft 1 (reference)** | **16.298** | 6.784 | 9.509 | 10.868 | 0.951 | 9.952 |
| MTP draft 2 | 16.283 | 6.680 | 9.500 | **10.324** | 0.968 | **9.332** |
| MTP draft 3 | 17.871 | 6.654 | 11.205 | 11.200 | 0.948 | 10.168 |
| Draft 1 + decode expert `hybrid` | 16.788 | 6.678 | 10.028 | 11.206 | 0.972 | 10.224 |
| Draft 1 baseline repeat | 16.263 | 6.720 | 9.483 | 10.675 | 0.965 | 9.705 |

MTP counters demonstrate the alternate draft depths were **actually engaged**:
- Draft 1: median 116 speculative verification rounds, 75 accepted draft tokens, 65.2% acceptance among drafted tokens (across 11 requests).
- Draft 2: median 93 rounds, 98 accepted tokens, 53.3% acceptance.
- Draft 3: median 89 rounds, 102 accepted tokens, 38.6% acceptance.

Depth 2 reduces verification rounds by ~20% and improves short-request latency ~5% versus reference (and ~3% versus reference repeat); **long total latency is unchanged** (16.283 vs 16.298 seconds). Depth 3 reduces rounds further but increases long latency ~9.7%; extra draft execution outweighs verification savings. Enabling `NINFER_V100_DECODE_EXPERT_POLICY=hybrid` with draft 1 degrades long and short latency ~3–5%.

**Recommendation:** Keep draft 1 as the controlled V100 reference for broad workloads. Depth 2 is an optional case for predominantly short inputs and longer outputs, not a demonstrated overall win. Do not default to depth 3 or hybrid decode streaming. Additional speculative-window tuning is lower priority than improving MoE prefill/streaming.

**Next controlled experiment:** Compare 4 versus 8 in-flight pinned-host/GPU expert-stream ring slots under the successful `lru` + `auto256` policy. This directly tests whether extra H2D/compute overlap improves long prompt prefill. Workflow `.github/workflows/v100-ninfer-stream-ring-ab.yml`; ensure only one V100 job at a time. The Strata cross-engine speed gap remains *descriptive*, not quantization-controlled.
