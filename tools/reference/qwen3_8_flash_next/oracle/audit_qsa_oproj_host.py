#!/usr/bin/env python3
"""Host-only decomposition of the Phase 11 QSA o_proj stage error.

Pure stdlib (the runner system python3 has no numpy/torch) and no GPU.

For one (layer, position) the script loads:

  - the FP8 E4M3 o_proj codes and F32 row scale from the mixed source
    checkpoint. The converter's Fp8Recipe copies these bytes verbatim
    into the .ninfer artifact, so this is exactly the weight the device
    kernel decodes;
  - the FP32 stage traces <L>_qsa_gated, <L>_attn_block_output and
    <L>_attn_block_input from the 14-position stage trace.

It then answers, in a pure FP64 GEMM with the FP8-decoded weight:

  1. SANITY  Does W @ x reproduce the oracle attn_block_output stage
     (pure o_proj output, or only after adding the block input)?
  2. DECISIVE  Replace x by RNE-BF16(x) -- exactly what the device test
     injects (the production boundary storage) -- and report the GEMM
     output NRMSE against the oracle stage. Compare it with the
     device-measured stage error for the same position.
  3. GAIN  ||W d||/||d|| relative to ||W x||/||x|| for d = xb - x.
     A well-conditioned GEMM amplifies the BF16 rounding at ~1x the
     input NRMSE; a large ratio means the stage's excess error is a
     data property of this weight/input pair, not an implementation
     defect.

Usage:
  python3 audit_qsa_oproj_host.py [layer] [position] [--device-nrmse 0.00867]

Environment overrides:
  NINFER_AUDIT_MODEL   mixed source checkpoint dir (default /srv/ninfer/source/mixed)
  NINFER_AUDIT_STAGES  stage trace root           (default /oracle/phase11-stage-trace14)
"""

import glob
import json
import math
import os
import struct
import sys
from operator import mul


def e4m3_values():
    """Exact E4M3 value for each of the 256 byte patterns (float64)."""
    lut = []
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = (m / 8.0) * 2.0 ** -6  # subnormal: 2^(1-7) * m/8
        elif e == 15 and m == 7:
            v = float("nan")  # NaN encoding; weights never contain it
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        lut.append(sign * v)
    return lut


def round_bf16(x: float) -> float:
    """RNE float32 -> BF16 -> float32 (the test's round_fp32_to_bf16)."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    if (u >> 23) & 0xFF == 0xFF:
        return x  # inf/NaN pass through
    u += 0x7FFF + ((u >> 16) & 1)
    u &= 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", u))[0]


def read_safetensors_tensor(model_dir: str, name: str):
    for fn in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(fn, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
            if name not in header:
                continue
            meta = header[name]
            off0, off1 = meta["data_offsets"]
            f.seek(8 + hlen + off0)
            return f.read(off1 - off0), meta
    raise SystemExit(f"tensor {name} not found under {model_dir}")


def load_stage(stage_root: str, pos: int, name: str):
    path = os.path.join(stage_root, f"pos{pos:04d}", f"{name}.bin")
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) % 4:
        raise SystemExit(f"{path} is not a whole number of float32s")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def nrmse(a, b) -> float:
    """Device-test metric: ||a - b|| / ||b|| (b = expected/oracle)."""
    nb = math.sqrt(sum(map(mul, b, b)))
    err = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return err / max(nb, 1e-30)


def cosine(a, b) -> float:
    dot = sum(map(mul, a, b))
    na = math.sqrt(sum(map(mul, a, a)))
    nb = math.sqrt(sum(map(mul, b, b)))
    return dot / max(na * nb, 1e-30)


def main() -> None:
    args = sys.argv[1:]
    layer, pos = 15, 2
    device_nrmse = 0.00867
    if "--device-nrmse" in args:
        i = args.index("--device-nrmse")
        device_nrmse = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    if len(args) >= 1:
        layer = int(args[0])
    if len(args) >= 2:
        pos = int(args[1])

    model_dir = os.environ.get("NINFER_AUDIT_MODEL", "/srv/ninfer/source/mixed")
    stage_root = os.environ.get("NINFER_AUDIT_STAGES", "/oracle/phase11-stage-trace14")
    for base in (
            f"model.language_model.layers.{layer}.self_attn.o_proj",
            f"model.layers.{layer}.self_attn.o_proj",
            f"model.language_model.layers.{layer}.self_attn.qsa_o_proj",
            f"model.layers.{layer}.self_attn.qsa_o_proj"):
        try:
            codes, code_meta = read_safetensors_tensor(model_dir, base + ".weight")
            scale_raw, scale_meta = read_safetensors_tensor(model_dir, base + ".weight_scale")
            break
        except SystemExit:
            continue
    else:
        raise SystemExit(f"o_proj weight not found under {model_dir} (tried self_attn.o_proj / self_attn.qsa_o_proj, with and without the language_model prefix)")
    print(f"o_proj tensor {base}.weight")
    if code_meta["dtype"] != "F8_E4M3":
        raise SystemExit(f"unexpected o_proj weight dtype {code_meta['dtype']}")
    n, k = code_meta["shape"]
    if code_meta["dtype"] == "F8_E4M3" and len(codes) != n * k:
        raise SystemExit(f"o_proj code byte count {len(codes)} != {n} * {k}")
    if scale_meta["dtype"] != "F32":
        raise SystemExit(f"unexpected o_proj scale dtype {scale_meta['dtype']}")
    n_scale = len(scale_raw) // 4
    scale = struct.unpack(f"<{n_scale}f", scale_raw)
    if n_scale not in (n,):
        raise SystemExit(f"o_proj scale length {n_scale} != {n} output rows")

    ll = f"L{layer:02d}"
    x = load_stage(stage_root, pos, f"{ll}_qsa_gated")
    y = load_stage(stage_root, pos, f"{ll}_attn_block_output")
    r = load_stage(stage_root, pos, f"{ll}_attn_block_input")
    if len(x) != k:
        raise SystemExit(f"stage qsa_gated length {len(x)} != weight k {k}")
    if len(y) != n or len(r) != n:
        raise SystemExit(f"stage lengths y={len(y)} r={len(r)} != n={n}")

    xb = [round_bf16(v) for v in x]
    d = [xb[i] - x[i] for i in range(k)]
    lut = e4m3_values()

    # One pass over the weight: decode each row once, take the three dot
    # products (x, xb, d) and the row/column norm accumulators.
    y0 = [0.0] * n
    y1 = [0.0] * n
    yd = [0.0] * n
    col_sq = [0.0] * k
    for j in range(n):
        row = (struct.unpack(f"<{k}B", codes[j * k:(j + 1) * k]))
        s = float(scale[j])
        w = [lut[b] * s for b in row]
        y0[j] = sum(map(mul, w, x))
        y1[j] = sum(map(mul, w, xb))
        yd[j] = sum(map(mul, w, d))
        col_sq = [c + wv * wv for c, wv in zip(col_sq, w)]

    print(f"model dir   {model_dir}")
    print(f"stage root  {stage_root}  layer {layer}  position {pos}")
    print(f"o_proj      {n} x {k}  dtype={code_meta['dtype']} scale dtype={scale_meta['dtype']}")
    print(f"scale       min={min(scale):.6g}  max={max(scale):.6g}")
    print()
    print("1. SANITY (pure FP64 GEMM with the FP8-decoded weight)")
    print(f"   nrmse(W@x, y)               = {nrmse(y0, y):.3e}   (want ~1e-7: y is pure o_proj output)")
    y_plus_r = [y0[j] + r[j] for j in range(n)]
    print(f"   nrmse(W@x + block_input, y) = {nrmse(y_plus_r, y):.3e}   (want ~1e-7 if y includes the residual add)")
    print()
    print("2. DECISIVE (x replaced by RNE-BF16(x), the production boundary storage)")
    print(f"   input floor   nrmse(xb, x)     = {nrmse(xb, x):.3e}")
    y_bf = [round_bf16(v) for v in y]
    print(f"   output floor  nrmse(bf16(y),y) = {nrmse(y_bf, y):.3e}")
    y1r = [y1[j] + r[j] for j in range(n)]
    host_pure = nrmse(y1, y)
    host_resid = nrmse(y1r, y)
    host = host_pure if host_pure < host_resid else host_resid
    composition = "pure o_proj output" if host_pure < host_resid else "o_proj output + residual"
    print(f"   nrmse(W@xb, y)      = {host_pure:.3e}   (stage is pure o_proj output)")
    print(f"   nrmse(W@xb + r, y)  = {host_resid:.3e}   (stage includes the residual add)")
    print(f"   DECISIVE [{composition}]  = {host:.3e}   <-- compare with device: {device_nrmse:.5f}")
    print(f"   cosine(W@xb(+r), y) = {cosine(y1r if host_resid < host_pure else y1, y):.8f}")
    print(f"   residual  ||W@xb(+r) - bf16(y)||/||y|| = {nrmse(y1r if host_resid < host_pure else y1, y_bf):.3e}   (device 'residual beyond the floor')")
    print()
    print("3. GAIN (does this weight amplify the rounding direction?)")
    gain_x = math.sqrt(sum(map(mul, y0, y0))) / math.sqrt(sum(map(mul, x, x)))
    gain_d = math.sqrt(sum(map(mul, yd, yd))) / max(math.sqrt(sum(map(mul, d, d))), 1e-30)
    cn = [math.sqrt(v) for v in col_sq]
    cn_sorted = sorted(cn)
    median = cn_sorted[k // 2]
    print(f"   ||W x||/||x||          = {gain_x:.3f}")
    print(f"   ||W d||/||d||          = {gain_d:.3f}")
    print(f"   relative gain          = {gain_d / gain_x:.3f}   (a well-conditioned GEMM is ~1)")
    print(f"   column-norm max/median = {max(cn) / median:.3f}   (weight anisotropy proxy)")
    print()
    if 0.8 <= host / device_nrmse <= 1.25:
        print("VERDICT: the device stage error is reproduced by the pure GEMM with the")
        print("BF16-rounded input. The FP8 decode/scale/MMA path is exonerated; the error")
        print("is the input BF16 rounding amplified by this layer's o_proj weight (a")
        print("data-specific boundary floor of ~%.0fx the naive BF16 floor)."
              % (host / max(nrmse(xb, x), 1e-30)))
    else:
        print("VERDICT: the pure GEMM does NOT reproduce the device stage error. The")
        print("excess lives in the device path itself (kernel, stage emission, or the")
        print("boundary tensor the kernel actually reads) -- next step is a device-side")
        print("A/B on the same position.")


if __name__ == "__main__":
    main()
