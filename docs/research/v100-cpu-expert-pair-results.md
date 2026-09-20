# V100 forward-port Phase 1 CPU expert-pair result

Status: **PASS**

This records the target-host architecture-gate measurements for the standalone
`bench/cpu_nvfp4_expert_pair` probe after commit
`2638f8d054450b7979e4959c4d78fee477badd04` added the production-like BF16
round-to-nearest-even activation boundary between `SiLU(gate) * up` and the
down projection.

## Target host

```text
CPU:              2 x Intel Xeon E5-2697A v4 @ 2.60 GHz
physical cores:   32 total, 16/socket
logical CPUs:     64 total, SMT2
NUMA nodes:       2
node 0 CPUs:      0-15,32-47
node 1 CPUs:      16-31,48-63
NUMA distance:    local 10, remote 21
ISA:              AVX2 + FMA
```

The extracted real Flash-Next layer-0 routed-expert banks were:

```text
gate/up: 943,720,448 bytes
down:    471,861,248 bytes
total: 1,415,581,696 bytes (~1.318 GiB)
```

This working set is intentionally much larger than LLC. The benchmark used the
exact compact `expert-blockscale-k16-m128x4-v1` payloads extracted from the
real mixed Flash-Next artifact, not a decoded persistent copy.

The per-expert pair contract remained:

```text
pair_bytes=2,764,808
activation_boundary=BF16_RNE
```

## Correctness

Standalone synthetic self-test:

```text
verify.cosine=1
verify.nrmse=2.84912316e-07
verify.max_abs=144
self_test=PASS
pair_bytes=2764808
```

Real layer verification for both final runs:

```text
verify.cosine=1
verify.nrmse=3.95921072e-07
verify.max_abs=7.4505806e-09
```

The absolute value from the synthetic test is not used as the numerical gate;
the relative criteria are cosine and NRMSE.

## Gate definition

For the conservative reference scenario:

```text
target decode rate = 16.49 tok/s
host layers        = 36
cache hit rate     = 0.527

required pairs/s  = 16.49 * 10 * 36 * (1 - 0.527)
                  = 2,807.917 pairs/s

preferred pairs/s = 1.3 * required
                  = 3,650.292 pairs/s
```

The architecture gate requires measured CPU throughput to meet the required
rate. The preferred engineering target is 1.3x the required rate.

## Final BF16-corrected measurements

### 32 workers, one hardware thread from each physical core

Placement:

```text
numactl --physcpubind=0-31 --interleave=0,1
```

Result:

```text
workers=32
timed_seconds=0.513
pairs_per_s=9981.790
us_per_pair=100.182
compact_GiB_per_s=25.702
useful_GFLOP_per_s=98.125
batch_p50_us=25300.622
batch_p95_us=26876.318
batch_p99_us=28455.842
gate_required_pairs_per_s=2807.917
gate_preferred_pairs_per_s=3650.292
gate_minimum=PASS
gate_preferred=PASS
```

Margins:

```text
3.555x required rate
2.735x preferred rate
```

### 64 workers, SMT enabled

Placement:

```text
numactl --physcpubind=0-63 --interleave=0,1
```

Result:

```text
workers=64
timed_seconds=0.440
pairs_per_s=11632.975
us_per_pair=85.963
compact_GiB_per_s=29.954
useful_GFLOP_per_s=114.357
batch_p50_us=22388.497
batch_p95_us=23499.230
batch_p99_us=24162.859
gate_required_pairs_per_s=2807.917
gate_preferred_pairs_per_s=3650.292
gate_minimum=PASS
gate_preferred=PASS
```

Margins:

```text
4.143x required rate
3.187x preferred rate
```

SMT improved throughput by about 16.5% over the 32-worker physical-core result.

## Earlier physical-core sweep

The pre-BF16-boundary sweep established the scaling shape:

| Workers | Pairs/s | Compact GiB/s | Minimum gate | Preferred gate |
|---:|---:|---:|:---:|:---:|
| 8  | 2,806.998 | 7.228  | FAIL by ~0.03% | FAIL |
| 16 | 5,475.968 | 14.100 | PASS | PASS |
| 24 | 7,930.944 | 20.422 | PASS | PASS |
| 32 | 9,913.394 | 25.526 | PASS | PASS |

Those measurements used the same real compact layer payloads but precede the
BF16 activation-boundary correction. They are retained only as scaling evidence;
the final architecture decision uses the corrected 32- and 64-worker runs above.

## Architecture decision

The Phase-1 CPU expert-pair viability gate **passes decisively**.

Even the BF16-corrected 32-physical-core result supplies more than 3.5x the
minimum required pair rate for the conservative 36-host-layer / 52.7% hit-rate
case and more than 2.7x the preferred engineering target.

For the more demanding reference case of 48 host layers at approximately 53%
hit rate (about 3,720 required pairs/s), the same 32-worker measurement provides
about 2.68x throughput headroom.

This result supports continuing the forward-port architecture with host-side
compact NVFP4 routed-expert execution. It does **not** size or authorize a GPU
expert cache yet.

## Next phase

Proceed to Phase 2:

1. explicit SM70 / CUDA 12.8 root build selection;
2. `NINFER_VOLTA_BUILD` or equivalent explicit architecture definition;
3. architecture-conditioned source membership so Blackwell-only device code is
   not emitted for SM70;
4. eager/non-CUDA-Graph initial runtime planning;
5. static VRAM ledger and actual host-backed layer-count calculation.

The expert cache and hybrid admission/scheduling policy remain deferred until
the later cache phase specified by the forward-port plan.
