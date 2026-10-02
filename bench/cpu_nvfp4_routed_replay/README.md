# Production routed CPU expert replay

Phase 12 uses the runtime AVX2/FMA kernel and persistent worker pool, rather than
a copy of the Phase 1 probe. Canonical NVFP4 weights stay file-backed and compact;
only activation/result vectors are expanded.

Build this CPU-only tool independently:

    cmake -S bench/cpu_nvfp4_routed_replay -B build-cpu -DCMAKE_BUILD_TYPE=Release
    cmake --build build-cpu -j
    ctest --test-dir build-cpu --output-on-failure

Capture natural routing with the real-model test using
NINFER_PHASE12_CPU_TRACE=/path/natural-routing.bin and a prefix of at least 128
teacher-forced positions. This captures represented BF16 inputs, the actual ten
ordered expert IDs and routing weights for all 48 layers. Stage-oracle diagnostics
are rejected during capture. The existing final numerical gate remains unchanged;
a complete capture does not claim Phase 11 passed.

    python3 bench/cpu_nvfp4_routed_replay/prepare.py MODEL.ninfer offsets.txt
    numactl --physcpubind=0-31 --interleave=0,1       build-cpu/ninfer_cpu_nvfp4_routed_replay       MODEL.ninfer offsets.txt natural-routing.bin 32 32 48 16.49 3

Arguments after the trace are workers, provisional slots per layer, host layers,
target committed tokens/s, and timed repetitions. A reduced host-layer scenario
explicitly selects the first L layers; it is not evidence for an arbitrary
residency policy. The initial workflow uses all 48 host-backed layers.

The offline cache model uses uniform LRU and admits at most one missed expert per
layer/token, after determining all current misses. It models instantaneous
completion of that admission for the next token; real asynchronous upload delays
can lower future hit rates. It creates no GPU cache and commits no VRAM capacity.
Zero-slot replay supplies the conservative no-hit bound.

Each actual layer miss batch executes and reduces before the next layer begins.
There is no flattening into a large throughput batch. Timings include worker
rendezvous, complete gate/up/SiLU/down computation and ordered CPU reduction;
they exclude CUDA transfers, GPU hits and GPU/CPU overlap. Required pair rate is
target tokens/s times the measured misses/token. Minimum and preferred gates are
1.0x and 1.3x. Reported rates do not establish whole-server throughput.

Before timing, scalar mathematical evaluation checks 48 real routed pairs spread
across the layers and prefix, using the represented BF16 SiLU boundary, with
cosine >= 0.99999, NRMSE <= 0.002 and finite outputs. The routed working set must
exceed 256 MiB. A complete pass warms actual miss pages and workers before timed
repetitions. Latency percentiles include each layer batch, including zero misses.

NUMA runs record CPU binding and the model mapping's actual page residency.
File-backed pages may already be resident from capture or earlier replay, so
membind alone cannot establish local expert pages. Interpret residency alongside
the placement results; do not label a socket result NUMA-local without evidence.
