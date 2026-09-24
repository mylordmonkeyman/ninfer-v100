#!/usr/bin/env python3
"""CPU Phase 11 decode experiment with explicit persistent storage rounding.

The existing run_oracle.py remains the independent FP32 oracle.  This script
checks that incremental FP32 decoding agrees with that oracle before reporting
the storage-profile experiment.  It deliberately does not describe its output
as a precision floor: CUDA fused arithmetic and reduction order remain to be
calibrated with stage traces on the V100.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from precision_profile import (
    PROFILES,
    install_projection_boundaries,
    materialize_persistent_states,
    round_to_bf16,
)
from precision_metrics import compare_logits
from compare_precision_traces import compare as compare_stages
from run_oracle import build_oracle


def read_ids(path, count):
    root = json.loads(path.read_text())
    ids = root["token_ids"] if isinstance(root, dict) else root
    if not isinstance(ids, list) or len(ids) < count or not all(
        isinstance(token, int) and 0 <= token for token in ids[:count]
    ):
        raise ValueError(f"expected at least {count} nonnegative token IDs in {path}")
    return ids[:count]


def read_oracle(root, token_ids):
    manifest = json.loads((root / "manifest.json").read_text())
    positions = manifest["positions"]
    if len(positions) < len(token_ids):
        raise ValueError("FP32 oracle has fewer positions than requested")
    result = []
    for position, token_id in enumerate(token_ids):
        record = positions[position]
        if record["position"] != position or record["token_id"] != token_id:
            raise ValueError(f"oracle token or position mismatch at {position}")
        tensors = {item["name"]: item for item in record["tensors"]}
        item = tensors["logits"]
        path = root / item["file"]
        logits = np.fromfile(path, dtype="<f4")
        if logits.size != int(np.prod(item["shape"])):
            raise ValueError(f"truncated oracle logits: {path}")
        result.append((logits, tensors))
    return result


def _stage_hooks(model, captured):
    handles = []

    def capture(name, select=lambda output: output):
        def hook(_module, _args, output):
            captured[name] = select(output).detach().to(torch.float32).cpu().clone()
        return hook

    handles.append(model.embed_tokens.register_forward_hook(capture("embedding")))
    for index, layer in enumerate(model.layers):
        prefix = f"L{index:02d}_"
        if layer.ple is not None:
            handles.append(layer.ple.register_forward_hook(capture("ple_injection")))
        handles.append(layer.attn_hyper_connection.register_forward_hook(
            capture(prefix + "attn_block_input", lambda output: output[0])
        ))
        handles.append(layer.mlp_hyper_connection.register_forward_hook(
            capture(prefix + "mlp_block_input", lambda output: output[0])
        ))
        handles.append(layer.register_forward_hook(capture(prefix + "hyper_after_mlp")))
        handles.append(layer.mlp.gate.register_forward_hook(
            capture(prefix + "moe_router_ids", lambda output: output[2])
        ))
        handles.append(layer.mlp.gate.register_forward_hook(
            capture(prefix + "moe_router_scores", lambda output: output[0])
        ))
    handles.append(model.hyper_connection_mixer.register_forward_hook(
        capture("final_hidden")
    ))
    return handles


def run_decode(model, head, token_ids, profile, out_root):
    # One token per forward, with one cache for the entire prefix.  Rounding a
    # cache tensor after the update changes all later positions, unlike
    # independently rounding tensors from the completed FP32 oracle.
    captured = {}
    handles = install_projection_boundaries(model, profile)
    handles.extend(_stage_hooks(model, captured))
    cache = None
    logits = []
    manifest = {"profile": profile.name, "positions": []}
    try:
        with torch.inference_mode():
            for position, token_id in enumerate(token_ids):
                captured.clear()
                output = model(input_ids=torch.tensor([[token_id]], dtype=torch.long),
                               past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                if cache is None:
                    raise RuntimeError("model did not return a persistent decode cache")
                materialize_persistent_states(cache, model, profile)
                hidden = output.last_hidden_state[0, -1].to(torch.float32)
                logit = F.linear(hidden, head).reshape(-1)
                if profile.logits_bf16:
                    logit = round_to_bf16(logit)
                logit = logit.detach().cpu().numpy().astype("<f4", copy=False)
                logits.append(logit)
                if out_root is not None:
                    folder = out_root / f"pos{position:04d}"
                    folder.mkdir(parents=True, exist_ok=True)
                    stages = dict(captured)
                    stages["logits"] = torch.from_numpy(logit)
                    records = []
                    for name, tensor in stages.items():
                        array = tensor.reshape(-1).numpy().astype("<f4", copy=False)
                        if not np.isfinite(array).all():
                            raise ValueError(f"nonfinite stage {name} at position {position}")
                        filename = f"pos{position:04d}/{name}.bin"
                        array.tofile(out_root / filename)
                        records.append({"name": name, "dtype": "FP32",
                                        "shape": list(tensor.shape), "file": filename,
                                        "bytes": array.nbytes})
                    manifest["positions"].append({"position": position,
                                                   "token_id": token_id,
                                                   "tensors": records})
    finally:
        for handle in handles:
            handle.remove()
    if out_root is not None:
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ple-dir", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--fp32-oracle", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=14)
    parser.add_argument("--max-fp32-kl", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--v100-trace", type=Path,
                        help="optional selected-stage dump from the V100 test")
    args = parser.parse_args()
    if args.positions < 1:
        parser.error("--positions must be positive")
    token_ids = read_ids(args.ids_file, args.positions)
    oracle = read_oracle(args.fp32_oracle, token_ids)
    model, head = build_oracle(str(args.model_dir), str(args.ple_dir))
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fp32 = run_decode(model, head, token_ids, PROFILES["fp32"], args.out_dir / "fp32")
    baseline = [compare_logits(item[0], logits)
                for item, logits in zip(oracle, fp32)]
    worst = max(row["kl"] for row in baseline)
    if worst > args.max_fp32_kl:
        (args.out_dir / "fp32_decode_mismatch.json").write_text(
            json.dumps({"worst_kl": worst, "positions": baseline}, indent=2)
        )
        raise RuntimeError(f"incremental FP32 decode does not reproduce the oracle: "
                           f"worst KL {worst:.6g} > {args.max_fp32_kl}; "
                           "precision-profile conclusions are blocked")

    for layer in model.layers:
        layer.mlp.experts.round_activations_to_bf16 = True
    matched = run_decode(model, head, token_ids, PROFILES["v100-phase11-storage"],
                         args.out_dir / "v100-phase11-storage")
    comparison = [compare_logits(item[0], logits)
                  for item, logits in zip(oracle, matched)]
    report = {
        "profile": "v100-phase11-storage",
        "status": "uncalibrated; not a lower bound or Phase 11 qualification",
        "positions": args.positions,
        "fp32_decode_worst_kl": worst,
        "top1_agreement": sum(row["top1_agree"] for row in comparison) / len(comparison),
        "mean_kl": sum(row["kl"] for row in comparison) / len(comparison),
        "per_position": comparison,
        "limitations": ["FP32 CPU arithmetic between materialization boundaries",
                        "FP8 GEMV reduction order and fused GDN readout unmodeled",
                        "prepared V100 expert and output-head weights not independently checked",
                        "PLE and QSA internal materializations not fully represented",
                        "V100 stage parity still required"],
    }
    report["stage_comparison"] = compare_stages(
        args.fp32_oracle, args.out_dir / "v100-phase11-storage",
        args.out_dir / "comparison", args.v100_trace,
        incremental_fp32_root=args.out_dir / "fp32"
    )
    (args.out_dir / "precision_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in
                      ("status", "top1_agreement", "mean_kl")}, indent=2))


if __name__ == "__main__":
    main()
