# CPU NVFP4 expert-pair viability probe

This directory is the Phase-1 architecture-gate benchmark for the Flash-Next V100 forward-port. It is intentionally standalone: it does **not** depend on the root CUDA build, CUDA Graphs, or any expert cache implementation.

The timed operation is one complete routed expert pair at `T=1`:

```text
input[2560]
  -> NVFP4 gate[640,2560]
  -> NVFP4 up[640,2560]
  -> SiLU(gate) * up
  -> NVFP4 down[2560,640]
  -> output[2560]
```

Persistent weights stay in the exact compact `expert-blockscale-k16-m128x4-v1` NVFP4 representation. The CPU path decodes E2M1 codes and swizzled E4M3FN K16 scales directly into AVX2/FMA operations. It never creates a persistent FP16/FP32 expert copy.

## Build

```bash
cmake -S bench/cpu_nvfp4_expert_pair \
      -B build/cpu_nvfp4_expert_pair \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu_nvfp4_expert_pair -j

build/cpu_nvfp4_expert_pair/ninfer_cpu_nvfp4_expert_pair --self-test
```

`--self-test` creates a one-expert synthetic compact bank and checks the AVX2 full-pair output against an independent scalar decoder. It is a correctness smoke test only; it is **not** an architecture-gate performance result.

## Extract actual compact expert bytes

Use one real Flash-Next layer so the timed working set spans all 512 expert IDs and is far larger than LLC:

```bash
python bench/cpu_nvfp4_expert_pair/extract_layer.py \
  /path/to/qwen3_8_flash_next.ninfer \
  --layer 0 \
  --out-dir /mnt/bench/nvfp4-layer0
```

This copies the exact artifact payloads for:

```text
text/layers/0/mlp/experts/gate_up
text/layers/0/mlp/experts/down
```

No decode or repack occurs during extraction.

## Run the architecture gate

Example for the conservative 36-host-layer / 52.7% hit-rate case:

```bash
build/cpu_nvfp4_expert_pair/ninfer_cpu_nvfp4_expert_pair \
  --gate-up /mnt/bench/nvfp4-layer0/layer00-gate_up.nvfp4.bin \
  --down    /mnt/bench/nvfp4-layer0/layer00-down.nvfp4.bin \
  --threads 32 \
  --pairs-per-batch 256 \
  --batches 20 \
  --target-tok-s 16.49 \
  --host-layers 36 \
  --hit-rate 0.527
```

The report includes:

```text
pairs_per_s
us_per_pair
compact_GiB_per_s
useful_GFLOP_per_s
batch_p50_us
batch_p95_us
batch_p99_us
gate_required_pairs_per_s
gate_preferred_pairs_per_s
gate_minimum=PASS|FAIL
gate_preferred=PASS|FAIL
```

The minimum gate is:

```text
required pairs/s = target_tok_s * 10 * L_host * (1 - hit_rate)
measured pairs/s >= required pairs/s
```

The preferred engineering gate multiplies the required rate by `--headroom` (default `1.3`).

For the current reference scenarios:

```text
36 host layers, h=0.527: required ~= 2808 pairs/s
36 host layers, h=0.622: required ~= 2243 pairs/s
48 host layers, h=0.530: required ~= 3720 pairs/s
```

Run all scenarios explicitly rather than treating `L_host=36` as fixed.

## Avoid an LLC benchmark

The benchmark defaults to uniform random IDs over the whole expert bank. A real recorded expert-ID stream can be supplied with `--ids FILE`, one integer ID per line. The sequence is repeated if it is shorter than the requested run.

Do not extract or benchmark a single expert repeatedly. One 512-expert layer already gives a compact working set of roughly 1.3 GiB across gate/up plus down, comfortably larger than Broadwell LLC.

The benchmark prefaults both mappings before warmup so major page faults do not dominate timed batches. For NUMA placement experiments, run under `numactl` and consider `--drop-file-cache` before the prefault so the process can re-fault pages under the selected policy.

## NUMA and worker sweep

Run the same bytes and sequence under several placements and worker counts. Example commands for the target dual-socket Broadwell host:

```bash
# Socket 0 local
numactl --cpunodebind=0 --membind=0 \
  build/cpu_nvfp4_expert_pair/ninfer_cpu_nvfp4_expert_pair ... --threads 16 --drop-file-cache

# Socket 1 local
numactl --cpunodebind=1 --membind=1 \
  build/cpu_nvfp4_expert_pair/ninfer_cpu_nvfp4_expert_pair ... --threads 16 --drop-file-cache

# Both sockets, interleaved pages
numactl --cpunodebind=0,1 --interleave=0,1 \
  build/cpu_nvfp4_expert_pair/ninfer_cpu_nvfp4_expert_pair ... --threads 32 --drop-file-cache
```

Suggested worker sweep:

```text
1, 2, 4, 8, 16, 24, 32, 48, 64
```

Record the complete command line, CPU affinity/memory policy, `pairs_per_s`, compact GiB/s, P50/P95/P99 batch latency, and the gate result. Hyperthreads may or may not help this memory-heavy workload; measure rather than assume.

## Scope

This probe is disposable/reference code for the early architecture decision. If the CPU gate passes, the later production host-expert backend can reuse its codec/arithmetic semantics while improving scheduling and NUMA placement. If straightforward AVX2/FMA plus NUMA tuning cannot meet the required pair rate, stop before investing in the compressed GPU cache/hybrid scheduler and revisit the architecture.
