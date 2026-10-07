# External V100 system sampling

`tools/telemetry/sample_system.py` observes one explicitly identified engine
process and optionally one explicitly selected GPU. It works for either engine;
the sampler code need not be built into Strata. It does not launch/stop inference,
create a CUDA context, modify affinity or change GPU settings. Use it during
controlled diagnostics only until its separate overhead has been measured.

```bash
python3 tools/telemetry/sample_system.py \
  --pid "$COMPARE_ENGINE_PROC_PID" --engine ninfer \
  --run-id "$COMPARE_RUN_ID" --level 2 --gpu-uuid "$COMPARE_GPU_UUID" \
  --interval-ms 200 --detail-interval-ms 1000 \
  --output "$COMPARE_SYSTEM_LOG"
python3 tools/telemetry/system_schema.py "$COMPARE_SYSTEM_LOG"
```

Supply the engine PID as visible in the sampler's mounted `/proc`; PID namespaces
can make a host/container PID differ from a subprocess API's PID. Supply the GPU
UUID from the frozen campaign manifest, not a guessed default GPU index. Omit
`--gpu-uuid` only for CPU-only software checks: GPU fields then remain unavailable.
`--engine strata` changes the owner label, not sampling behavior. A run ID must
match the engine round/request logs. `--level` labels the observed configuration;
it does not enable engine telemetry. The campaign wrapper is still to be wired.

The output path must be new. The sampler stops on process exit/zombie, PID reuse,
SIGINT/SIGTERM or optional `--duration-seconds`. It never signals the observed
engine. A terminal record distinguishes the stop reason from a truncated file.
Startup/permission errors fail the collector; individual optional read/API failures
are recorded in `unavailable`, with absent/null measurements rather than zeros.

## Data and timing

The separate schema is `ninfer-strata-v100-system-v1`, with `system_sample` and
`system_end` records. It is intentionally separate from inference round JSONL.
The validator checks owner consistency, process start identity, ordered sample
IDs/timestamps, terminal totals, units/coverage and finite nonnegative values.

Samples carry monotonic and wall-clock start timestamps. Process counters and
memory are sampled every 100–200 ms (default 200). NUMA mappings and per-thread
stat/status snapshots are collected at the explicit detail interval (default
1000 ms); unavailable/unsampled details remain distinct from measured zeros.
The observation is not atomic: its duration bounds the sequential read window.
`sampling_duration_us` excludes output serialization/flush; `previous_output_us`
records that cost for the preceding sample. Overruns skip catch-up bursts; actual
timestamp gaps, rather than nominal cadence, determine observation coverage.

| Source | Measurements | Interpretation |
|---|---|---|
| Process stat/status | CPU user/system microseconds, fault/context-switch totals, threads, RSS/virtual/swap/locked/pinned bytes, allowed CPU/NUMA masks | Cumulative counters or OS snapshots; RSS accounting can be approximate |
| Process I/O | Native Linux `rchar`, `wchar`, syscall and storage byte counters | Cumulative process-wide I/O, not expert-specific DRAM traffic |
| NUMA mappings | Mapped resident bytes by node, using each mapping's kernel page size | Mapping accounting, not unique owned physical memory; aliases/shared pages can overlap |
| Per-thread stat/status | Native TID/start identity, CPU times, last CPU and allowed masks | Not yet joined to engine worker IDs; allowed masks are not measured residency |
| Host meminfo | Total/available/cache/dirty/writeback and swap bytes | Host snapshots, not model allocation breakdown |
| NVML | Device memory bytes, GPU/memory-interface utilization percent, power mW, temperature C, SM/memory clocks MHz | Whole-device observations, potentially including unrelated processes |
| NVML PCIe | TX/RX `*_kb_per_s_nvml` | Preserve NVML's native KB/s and 20 ms measurement window; not actual expert transfer payload bytes |

NVML's own utilization windows need not match the polling interval. Utilization
is not a per-engine critical path, and memory-interface utilization is not
measured DRAM bandwidth. System samples do not establish per-buffer GPU allocation,
per-worker idle time, cache events or request/MTP linkage. Never turn their time
averages into a precise overlap/busy/idle decomposition.

API/OS semantics follow the primary [NVML device query reference](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)
and [Linux process stat reference](https://www.man7.org/linux/man-pages/man5/proc_pid_stat.5.html).
The sampler uses stable NVML v1 memory/utilization structures and read-only calls.

## Qualification

Software tests cover `/proc` names containing parentheses, unit conversion,
mixed normal/huge-page NUMA mappings, missing-file coverage, PID reuse, a real
live child, bounded collection, process exit and truncated streams. A compiled
C fixture checks NVML argument/structure ABI writes and unsupported-call handling.
That fixture is not evidence that the V100 driver or every optional API works.

Before the comprehensive campaign, qualify the real GPU UUID/API coverage and
sampler off/on end-to-end overhead. Keep the sampler configuration/order fixed
across paired arms. Neither its presence nor the engine's Level-1 counters yet
supports a <2% combined overhead claim. No hardware benchmark is dispatched by
adding this utility.
