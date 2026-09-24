#!/usr/bin/env python3
"""Replay one QSA output projection from frozen FP32 tensors and FP8 weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import safetensors
import torch


def nrmse(reference, value):
    difference = (value.double() - reference.double()).square().mean().sqrt()
    scale = reference.double().square().mean().sqrt().clamp_min(1e-12)
    return float(difference / scale)


def read_stage(root, position, layer, name, size):
    path = root / f"pos{position:04d}" / f"L{layer:02d}_{name}.bin"
    data = np.fromfile(path, dtype="<f4")
    if data.size != size or not np.isfinite(data).all():
        raise ValueError(f"invalid frozen stage: {path}")
    return torch.from_numpy(data.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--position", type=int, default=2)
    parser.add_argument("--layer", type=int, default=15)
    args = parser.parse_args()

    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    base = f"model.language_model.layers.{args.layer}.self_attn.o_proj"

    def weight(name):
        file = args.model_dir / index["weight_map"][name]
        with safetensors.safe_open(str(file), framework="pt", device="cpu") as reader:
            return reader.get_tensor(name)

    codes = weight(base + ".weight")
    scale = weight(base + ".weight_scale").float()
    if codes.shape != (2560, 6144) or scale.numel() != 2560:
        raise ValueError(f"unexpected projection shape: {codes.shape}, {scale.shape}")
    matrix = codes.float() * scale.reshape(-1, 1)
    gate = read_stage(args.oracle, args.position, args.layer, "qsa_gated", 6144)
    expected = read_stage(args.oracle, args.position, args.layer, "attn_block_output", 2560)

    torch.set_num_threads(16)
    with torch.no_grad():
        original = torch.mv(matrix, gate)
        oracle_input_bf16 = gate.to(torch.bfloat16).float()
        input_rounded = torch.mv(matrix, oracle_input_bf16)
        output_rounded = original.to(torch.bfloat16).float()
        both_rounded = input_rounded.to(torch.bfloat16).float()

    report = {
        "position": args.position, "layer": args.layer,
        "oracle_projection_nrmse": nrmse(expected, original),
        "input_bf16_only_nrmse": nrmse(expected, input_rounded),
        "output_bf16_only_nrmse": nrmse(expected, output_rounded),
        "both_bf16_nrmse": nrmse(expected, both_rounded),
        "input_bf16_rms": float((gate.double()-oracle_input_bf16).square().mean().sqrt()),
        "oracle_output_rms": float(expected.double().square().mean().sqrt()),
        "gate_nonzero": int(torch.count_nonzero(gate)),
    }
    print(json.dumps(report, indent=2))
    if report["oracle_projection_nrmse"] > 1e-4:
        raise ValueError("FP32 replay did not reproduce the frozen QSA projection")


if __name__ == "__main__":
    main()
