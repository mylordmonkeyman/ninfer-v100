# V100 performance-only continuation: controlled decode and stream-pipeline A/B

**Scope:** performance first. Do not spend GPU cycles on additional quality suites unless a measured change raises a concrete correctness failure. Do not stop unrelated GPU processes, rewrite the resident model, or alter installed Strata.

## Frozen performance baseline

Completed https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37769284560:

| NInfer policy | 585-token input + 64 output latency | 63-token input + 64 output latency |
|---|---:|---:|
| Static expert-cache profile | 13.80 s | 6.41 s |
| LRU expert cache | 12.18 s | 4.90 s |
| LRU + always GPU prefill streaming | 9.73 s | 5.23 s |
| **LRU + auto256 GPU streaming** | **9.65 s** | **4.08 s** |

These are within-engine comparisons with identical NInfer weights/quantization; Strata's ~4.54 s long baseline used different quantization and speculative decoding, so direct speed ratio is descriptive only.

## Campaign A: native MTP draft depth and GPU miss decode

**Run:** https://github.com/mylordmonkeyman/ninfer-v100/actions/runs/37792322111  
**Workflow:** `.github/workflows/v100-ninfer-mtp-ab.yml`  
**Trigger SHA:** `9af92d60355b33f64eb0242240a0803213b531fd`.

Sequential controls with identical NInfer model, profile, max 156 expert slots per layer, GPU and NUMA host:
1. Current performance reference: LRU+auto256, MTP draft1.
2. Same configuration, MTP draft2.
3. Same configuration, MTP draft3.
4. Same draft1, decode expert policy `hybrid` (stream misses with at least two routes).
5. Reference draft1 repeat.

Five long warmups and three measured long + three short per case; maximum 192 output tokens to make decode timing and MTP acceptance statistically informative. Record HTTP median, native prefill and decode seconds, native MTP proposed/accepted counters, cache GPU resident route share, failure and VRAM usage. A higher draft depth might reduce decode rounds, but extra verify work can offset acceptance; **select using measured output latency, not just speed of one stage**.

## Campaign B: bounded 4-versus-8 slot H2D/compute pipeline

**Prepared but not yet launched:** `.github/workflows/v100-ninfer-stream-ring-ab.yml`.

NInfer's nonresident expert streamer previously used **four** pinned-host / GPU scratch slots. A bounded `NINFER_V100_EXPERT_STREAM_RING_SLOTS=8` experiment is now supported; default remains **four**. Changes affect:
- Same number of in-flight asynchronous expert transfers and launch descriptors, with leases/events preventing buffer reuse before prior GPU consumers finish.
- Actual memory budget computed from ring size in `flash_next_expert_stream_device_bytes` and checked against allocated bytes.
- No change in weights, numerical kernels, expert cache policy, CPU worker count, or MTP draft depth.
- Eight-slot GPU correctness fixture before running a full model.

One hardware run should compare auto256 ring4, auto256 ring8, then ring4 repeat with 64-output-token long and short prompts. If ring8 does not improve median prefill beyond baseline drift, retain ring4 by default; do not proliferate tuning knobs or repeat campaigns.

**Sequencing:** Complete MTP run first. Hosted Python/CUDA syntax test must pass for ring code; after MTP run releases the V100, launch at most one ring-size comparison (same cross-workflow hardware concurrency group; no run cancellation). Interpret unchanged performance as a negative result worth recording, not a reason for more ad hoc parameter sweeps.

## Subsequent engineering direction

If MTP2/3 reduces decode substantially but prefill remains ~6.7 s vs Strata ~3.4 s (different quant), prioritize GPU expert-transfer cost and CPU staging bandwidth. If the eight-slot ring helps, investigate pipelining further using measured blocked-wait fraction; if it does not, focus on bytes moved and tensor kernels rather than queue depth. Carry forward only proven improvements.
