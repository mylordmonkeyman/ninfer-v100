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
from types import MethodType

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
from compare_precision_traces import (
    candidate_entries,
    compare as compare_stages,
    manifest_entries,
)
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


def read_router_ids(entries, positions, layers):
    """Read represented top-k expert IDs for a complete prefix trace."""
    expected = layers[0].mlp.gate.top_k
    result = {}
    for position in range(positions):
        for layer in range(len(layers)):
            key = (position, f"L{layer:02d}_moe_router_ids")
            if key not in entries:
                raise ValueError(f"router replay lacks stage {key}")
            values = np.fromfile(entries[key], dtype="<f4")
            if (values.size != expected or not np.isfinite(values).all() or
                not np.array_equal(values, values.astype(np.int64).astype(np.float32)) or
                (values < 0).any() or (values >= layers[layer].mlp.gate.num_experts).any() or
                len(set(values.tolist())) != expected):
                raise ValueError(f"invalid router replay expert IDs at {key}")
            result[position, layer] = torch.from_numpy(values.astype(np.int64))
    return result


def force_router_membership(model, ids, position):
    """Use saved expert sets, retaining the recomputed candidate router scores."""
    originals = []
    for layer, module in enumerate(model.layers):
        gate = module.mlp.gate
        original = gate.forward
        originals.append((gate, original))

        def replay(self, hidden, _original=original, _layer=layer):
            scores, _alpha, _selected = _original(hidden)
            chosen = ids[position[0], _layer].to(device=scores.device)
            chosen = chosen.expand(scores.shape[0], -1)
            alpha = torch.softmax(scores.float(), dim=-1).gather(-1, chosen)
            if self.norm_topk_prob:
                alpha = alpha / alpha.sum(dim=-1, keepdim=True)
            return scores, alpha.to(scores.dtype), chosen

        gate.forward = MethodType(replay, gate)
    return originals


def _stage_hooks(model, captured, trace_gdn_internals=False,
                 trace_hyper_layer0=False, trace_mlp_layer0=False,
                 trace_moe_routing_layer0=False, trace_gdn_layer1=False,
                 trace_hyper_layer1=False, trace_moe_components_layer0=False):
    handles = []

    def capture(name, select=lambda output: output):
        def hook(_module, _args, output):
            captured[name] = select(output).detach().to(torch.float32).cpu().clone()
        return hook

    def capture_input(name):
        def hook(_module, args):
            captured[name] = args[0].detach().to(torch.float32).cpu().clone()
        return hook

    handles.append(model.embed_tokens.register_forward_hook(capture("embedding")))
    for index, layer in enumerate(model.layers):
        prefix = f"L{index:02d}_"
        if layer.ple is not None:
            handles.append(layer.ple.register_forward_hook(capture("ple_injection")))
        if index == 1 and trace_gdn_layer1:
            handles.append(layer.linear_attn.register_forward_hook(
                capture(prefix + "attn_block_output", lambda output: output[0])))
        handles.append(layer.attn_hyper_connection.register_forward_hook(
            capture(prefix + "attn_block_input", lambda output: output[0])
        ))
        if trace_gdn_internals and hasattr(layer, "linear_attn"):
            gdn = layer.linear_attn
            handles.append(gdn.register_forward_hook(capture(prefix + "attn_block_output")))
            if index == 0:
                def capture_conv(_module, _args, output, stem=prefix):
                    # Conv1d returns [batch, channels, time]; decode's SiLU
                    # and split feed the GDN recurrence at this boundary.
                    conv = F.silu(output).transpose(1, 2)
                    if conv.shape[-1] != 10240 or conv.shape[1] != 1:
                        raise ValueError(f"unexpected layer-zero GDN convolution shape {tuple(conv.shape)}")
                    for name, chunk in zip(("gdn_query", "gdn_key", "gdn_value"),
                                           conv.split((2048, 2048, 6144), dim=-1)):
                        captured[stem + name] = chunk.detach().float().cpu().clone()
                handles.append(gdn.conv1d.register_forward_hook(capture_conv))
            for module, name in ((gdn.in_proj_qkv, "qkv"),
                                 (gdn.in_proj_z, "z"),
                                 (gdn.in_proj_a, "a"),
                                 (gdn.in_proj_b, "b")):
                handles.append(module.register_forward_hook(capture(prefix + "_raw_" + name)))
            handles.append(gdn.norm.register_forward_pre_hook(
                capture_input(prefix + "gdn_recurrent_output")))
            handles.append(gdn.out_proj.register_forward_pre_hook(
                capture_input(prefix + "gdn_gated_output")))
        handles.append(layer.mlp_hyper_connection.register_forward_hook(
            capture(prefix + "mlp_block_input", lambda output: output[0])
        ))
        if index == 1 and trace_hyper_layer1:
            hyper = layer.attn_hyper_connection
            hyper._phase11_capture_attention_raw = True
            def capture_attention_raw(module, _args, output, stem=prefix):
                raw = getattr(module, "_phase11_attention_raw", output[0])
                captured[stem + "attn_block_input_fp32"] = (
                    raw.detach().to(torch.float32).cpu().clone()
                )
                if hasattr(module, "_phase11_attention_raw"):
                    delattr(module, "_phase11_attention_raw")
            handles.append(hyper.register_forward_hook(capture_attention_raw))
        if index == 0 and trace_hyper_layer0:
            handles.append(layer.attn_hyper_connection.register_forward_pre_hook(
                capture_input(prefix + "hyper_before_attn")))
            handles.append(layer.attn_hyper_connection.register_forward_hook(
                capture(prefix + "attn_injection", lambda output: output[2])))
        if (index == 0 and trace_hyper_layer0) or (index == 1 and trace_hyper_layer1):
            handles.append(layer.mlp_hyper_connection.register_forward_pre_hook(
                capture_input(prefix + "hyper_after_attn")))
        if index == 0 and trace_hyper_layer0:
            handles.append(layer.mlp_hyper_connection.register_forward_hook(
                capture(prefix + "mlp_injection", lambda output: output[2])))
        if index == 0 and trace_mlp_layer0:
            handles.append(layer.mlp.register_forward_hook(
                capture(prefix + "mlp_block_output")))
        handles.append(layer.register_forward_hook(capture(prefix + "hyper_after_mlp")))
        handles.append(layer.mlp.gate.register_forward_hook(
            capture(prefix + "moe_router_ids", lambda output: output[2])
        ))
        handles.append(layer.mlp.gate.register_forward_hook(
            capture(prefix + "moe_router_scores", lambda output: output[0])
        ))
        if index == 0 and trace_moe_components_layer0:
            handles.append(layer.mlp.experts.register_forward_hook(
                capture(prefix + "moe_routed_sum")))
            # Registered after the profile's BF16 shared-down input hook.
            handles.append(layer.mlp.shared_expert.down_proj.register_forward_pre_hook(
                capture_input(prefix + "moe_shared_activation")))
        if index == 0 and trace_moe_routing_layer0:
            handles.append(layer.mlp.gate.register_forward_hook(
                capture(prefix + "moe_router_alpha", lambda output: output[1])
            ))
            handles.append(layer.mlp.shared_expert_gate.register_forward_hook(
                capture(prefix + "moe_shared_scale", torch.sigmoid)
            ))
    handles.append(model.hyper_connection_mixer.register_forward_hook(
        capture("final_hidden")
    ))
    return handles


def run_decode(model, head, token_ids, profile, out_root,
               trace_gdn_internals=False, trace_hyper_layer0=False,
               trace_mlp_layer0=False, trace_moe_routing_layer0=False,
               trace_gdn_layer1=False, trace_hyper_layer1=False,
               trace_mlp_raw_layer1=False, forced_router_ids=None,
               mlp_block_input_fp32=False, mlp_block_input_fp32_layers=(),
               fused_hyper_updates=False, trace_moe_components_layer0=False):
    # One token per forward, with one cache for the entire prefix.  Rounding a
    # cache tensor after the update changes all later positions, unlike
    # independently rounding tensors from the completed FP32 oracle.
    captured = {}
    handles = []
    replay_position = [-1]
    originals = (force_router_membership(model, forced_router_ids, replay_position)
                 if forced_router_ids is not None else [])
    for layer_index, enabled in ((0, trace_hyper_layer0), (1, trace_mlp_raw_layer1)):
        if enabled:
            # Register before the profile's BF16 output hook so the raw
            # mixer result is compared separately from stored block_input.
            def capture_unrounded_mlp(_module, _args, output, index=layer_index):
                captured[f"L{index:02d}_mlp_block_input_fp32"] = (
                    output[0].detach().to(torch.float32).cpu().clone()
                )
            handles.append(model.layers[layer_index].mlp_hyper_connection.register_forward_hook(
                capture_unrounded_mlp))
    handles.extend(install_projection_boundaries(
        model, profile, mlp_block_input_fp32=mlp_block_input_fp32,
        mlp_block_input_fp32_layers=mlp_block_input_fp32_layers,
        fused_hyper_updates=fused_hyper_updates))
    handles.extend(_stage_hooks(model, captured, trace_gdn_internals,
                                trace_hyper_layer0, trace_mlp_layer0,
                                trace_moe_routing_layer0, trace_gdn_layer1,
                                trace_hyper_layer1, trace_moe_components_layer0))
    cache = None
    logits = []
    conv_projection_history = {}
    manifest = {"profile": profile.name, "positions": []}
    try:
        with torch.inference_mode():
            for position, token_id in enumerate(token_ids):
                replay_position[0] = position
                captured.clear()
                output = model(input_ids=torch.tensor([[token_id]], dtype=torch.long),
                               past_key_values=cache, use_cache=True)
                # Match the independent oracle's controls and projection
                # materializations without replaying the recurrent kernel.
                for index, layer in enumerate(model.layers if trace_gdn_internals else ()):
                    if not hasattr(layer, "linear_attn"):
                        continue
                    prefix = f"L{index:02d}_"
                    qkv = captured.pop(prefix + "_raw_qkv")
                    z = captured.pop(prefix + "_raw_z")
                    a = captured.pop(prefix + "_raw_a")
                    b = captured.pop(prefix + "_raw_b")
                    gdn = layer.linear_attn
                    if index == 0 and prefix + "gdn_query" not in captured:
                        # Optimized one-token decode updates convolution state
                        # directly, bypassing Conv1d.forward and its hook.
                        # Reconstruct the same depthwise window from the raw
                        # current projection and the prior stored projections.
                        weight = gdn.conv1d.weight.detach().float().cpu()
                        width = weight.shape[-1]
                        if (weight.shape[:2] != (10240, 1) or width != 4 or
                            tuple(qkv.shape) != (1, 1, 10240)):
                            raise ValueError("unexpected GDN convolution dimensions")
                        prior = conv_projection_history.setdefault(index, [])
                        zero = torch.zeros_like(qkv)
                        frames = ([zero] * max(0, width - 1 - len(prior)) +
                                  prior[-(width - 1):] + [qkv])
                        window = torch.stack([frame.reshape(10240) for frame in frames], dim=-1)
                        bias = (None if gdn.conv1d.bias is None else
                                gdn.conv1d.bias.detach().float().cpu())
                        conv = F.silu(F.conv1d(window[None], weight, bias,
                                               groups=10240)).transpose(1, 2)
                        for name, chunk in zip(("gdn_query", "gdn_key", "gdn_value"),
                                               conv.split((2048, 2048, 6144), dim=-1)):
                            captured[prefix + name] = chunk.detach().clone()
                        prior.append((round_to_bf16(qkv) if profile.conv_state_bf16
                                      else qkv).clone())
                        del prior[:max(0, len(prior) - (width - 1))]
                    captured[prefix + "gdn_projected"] = torch.cat((qkv, z), dim=-1)
                    captured[prefix + "gdn_z"] = z
                    captured[prefix + "gdn_g"] = (-torch.exp(gdn.A_log.detach().cpu().float())
                        * F.softplus(a + gdn.dt_bias.detach().cpu().float()))
                    captured[prefix + "gdn_beta"] = torch.sigmoid(b)
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
        if trace_hyper_layer1:
            hyper = model.layers[1].attn_hyper_connection
            for name in ("_phase11_capture_attention_raw", "_phase11_attention_raw"):
                if hasattr(hyper, name):
                    delattr(hyper, name)
        for gate, original in originals:
            gate.forward = original
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
    parser.add_argument("--trace-gdn-internals", action="store_true",
                        help="collect extended GDN stages; requires matching V100 trace coverage")
    parser.add_argument("--trace-hyper-layer0", action="store_true",
                        help="collect post-attention layer-zero hyper state")
    parser.add_argument("--trace-mlp-layer0", action="store_true",
                        help="collect layer-zero MoE output before hyper injection")
    parser.add_argument("--trace-moe-routing-layer0", action="store_true",
                        help="collect layer-zero expert weights and shared scale")
    parser.add_argument("--trace-moe-components-layer0", action="store_true",
                        help="capture layer-zero shared activation and routed FP32 sum")
    parser.add_argument("--trace-gdn-layer1", action="store_true",
                        help="collect layer-one GDN output before hyper injection")
    parser.add_argument("--trace-hyper-layer1", action="store_true",
                        help="collect layer-one hyper state after attention injection")
    parser.add_argument("--trace-mlp-raw-layer1", action="store_true",
                        help="collect layer-one MLP mixer output before BF16 storage")
    parser.add_argument("--fused-routed-layer0", action="store_true",
                        help="use selected-path FP32 FMA for layer-zero routed expert accumulation")
    parser.add_argument("--fused-hyper-updates", action="store_true",
                        help="model CUDA fmaf in attention and MLP hyper updates for CPU storage profile")
    parser.add_argument("--replay-router-ids", action="store_true",
                        help="recompute complete prefixes with frozen oracle and V100 expert IDs")
    parser.add_argument("--ablate-mlp-block-input-bf16", action="store_true",
                        help="recompute complete prefixes with FP32 MLP input, retaining all other storage boundaries")
    parser.add_argument("--ablate-mlp-layer1-input-bf16", action="store_true",
                        help="recompute complete prefixes with only layer-one MLP input in FP32")
    args = parser.parse_args()
    if args.positions < 1:
        parser.error("--positions must be positive")
    token_ids = read_ids(args.ids_file, args.positions)
    oracle = read_oracle(args.fp32_oracle, token_ids)
    model, head = build_oracle(str(args.model_dir), str(args.ple_dir))
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fp32 = run_decode(model, head, token_ids, PROFILES["fp32"], args.out_dir / "fp32",
                      trace_gdn_internals=args.trace_gdn_internals,
                      trace_hyper_layer0=args.trace_hyper_layer0,
                      trace_mlp_layer0=args.trace_mlp_layer0,
                      trace_moe_routing_layer0=args.trace_moe_routing_layer0,
                      trace_moe_components_layer0=args.trace_moe_components_layer0,
                      trace_gdn_layer1=args.trace_gdn_layer1,
                      trace_hyper_layer1=args.trace_hyper_layer1,
                      trace_mlp_raw_layer1=args.trace_mlp_raw_layer1)
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
    if args.fused_routed_layer0:
        model.layers[0].mlp.experts.route_order_fma = True
    matched = run_decode(model, head, token_ids, PROFILES["v100-phase11-storage"],
                         args.out_dir / "v100-phase11-storage",
                         trace_gdn_internals=args.trace_gdn_internals,
                         trace_hyper_layer0=args.trace_hyper_layer0,
                         trace_mlp_layer0=args.trace_mlp_layer0,
                         trace_moe_routing_layer0=args.trace_moe_routing_layer0,
                         trace_moe_components_layer0=args.trace_moe_components_layer0,
                         trace_gdn_layer1=args.trace_gdn_layer1,
                         trace_hyper_layer1=args.trace_hyper_layer1,
                         trace_mlp_raw_layer1=args.trace_mlp_raw_layer1,
                         fused_hyper_updates=args.fused_hyper_updates)
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
    if args.v100_trace is not None:
        candidate = candidate_entries(args.v100_trace)
        candidate_logits = []
        for position, (reference, _) in enumerate(oracle):
            path = candidate.get((position, "logits"))
            if path is None:
                raise ValueError(f"V100 trace lacks logits at position {position}")
            logits = np.fromfile(path, dtype="<f4")
            if logits.shape != reference.shape or not np.isfinite(logits).all():
                raise ValueError(f"invalid V100 logits at position {position}")
            candidate_logits.append(logits)
        v100_rows = [compare_logits(item[0], logits)
                     for item, logits in zip(oracle, candidate_logits)]
        cpu_v100_rows = [compare_logits(cpu, gpu)
                         for cpu, gpu in zip(matched, candidate_logits)]
        report["three_way"] = {
            "oracle_vs_cpu_mean_kl": report["mean_kl"],
            "oracle_vs_v100_mean_kl": sum(r["kl"] for r in v100_rows) / len(v100_rows),
            "cpu_vs_v100_mean_kl": sum(r["kl"] for r in cpu_v100_rows) / len(cpu_v100_rows),
            "oracle_vs_v100_top1_agreement": sum(r["top1_agree"] for r in v100_rows) / len(v100_rows),
            "cpu_vs_v100_top1_agreement": sum(r["top1_agree"] for r in cpu_v100_rows) / len(cpu_v100_rows),
            "per_position_v100": v100_rows,
            "per_position_cpu_v100": cpu_v100_rows,
        }
    if args.replay_router_ids:
        if args.v100_trace is None:
            parser.error("--replay-router-ids requires --v100-trace")
        oracle_entries = {(p, item["name"]): args.fp32_oracle / item["file"]
                          for p, (_logits, tensors) in enumerate(oracle)
                          for item in tensors.values()}
        v100_entries = candidate_entries(args.v100_trace)
        replays = {}
        for source, entries in (("oracle", oracle_entries), ("v100", v100_entries)):
            ids = read_router_ids(entries, args.positions, model.layers)
            logits = run_decode(model, head, token_ids,
                                PROFILES["v100-phase11-storage"], None,
                                forced_router_ids=ids)
            oracle_rows = [compare_logits(item[0], result)
                           for item, result in zip(oracle, logits)]
            v100_rows = [compare_logits(result, gpu)
                         for result, gpu in zip(logits, candidate_logits)]
            replays[source] = {
                "oracle_mean_kl": sum(r["kl"] for r in oracle_rows) / len(oracle_rows),
                "cpu_vs_v100_mean_kl": sum(r["kl"] for r in v100_rows) / len(v100_rows),
                "oracle_top1_agreement": sum(r["top1_agree"] for r in oracle_rows) / len(oracle_rows),
                "per_position_oracle": oracle_rows,
                "per_position_cpu_vs_v100": v100_rows,
            }
        report["router_replay"] = {
            "meaning": "forced expert membership; weights recomputed from each CPU trajectory's scores",
            "source": replays,
        }
    if args.ablate_mlp_block_input_bf16:
        upgraded_root = args.out_dir / "mlp-block-input-fp32"
        upgraded_logits = run_decode(
            model, head, token_ids, PROFILES["v100-phase11-storage"], upgraded_root,
            mlp_block_input_fp32=True)
        upgraded_rows = [compare_logits(item[0], logits)
                         for item, logits in zip(oracle, upgraded_logits)]
        upgraded_stages = manifest_entries(upgraded_root)
        fp32_stages = manifest_entries(args.out_dir / "fp32")
        candidate_stages = (candidate_entries(args.v100_trace)
                            if args.v100_trace is not None else {})
        router_counts = {"oracle_flip": 0, "v100_difference": 0}
        for position in range(args.positions):
            for layer in range(len(model.layers)):
                key = (position, f"L{layer:02d}_moe_router_ids")
                chosen = set(np.fromfile(upgraded_stages[key], dtype="<f4").tolist())
                baseline_ids = set(np.fromfile(fp32_stages[key], dtype="<f4").tolist())
                router_counts["oracle_flip"] += chosen != baseline_ids
                if candidate_stages:
                    v100_ids = set(np.fromfile(candidate_stages[key], dtype="<f4").tolist())
                    router_counts["v100_difference"] += chosen != v100_ids
        report["mlp_block_input_fp32_ablation"] = {
            "meaning": "hypothetical FP32 MLP input in CPU reference; all other storage rules retained",
            "oracle_mean_kl": sum(row["kl"] for row in upgraded_rows) / len(upgraded_rows),
            "oracle_top1_agreement": sum(row["top1_agree"] for row in upgraded_rows),
            "per_position_oracle": upgraded_rows,
            "router_set_counts": router_counts,
        }
    if args.ablate_mlp_layer1_input_bf16:
        upgraded_root = args.out_dir / "mlp-layer1-input-fp32"
        upgraded_logits = run_decode(
            model, head, token_ids, PROFILES["v100-phase11-storage"], upgraded_root,
            mlp_block_input_fp32_layers=(1,))
        upgraded_rows = [compare_logits(item[0], logits)
                         for item, logits in zip(oracle, upgraded_logits)]
        upgraded_stages = manifest_entries(upgraded_root)
        baseline_stages = manifest_entries(args.out_dir / "v100-phase11-storage")
        fp32_stages = manifest_entries(args.out_dir / "fp32")
        candidate_stages = (candidate_entries(args.v100_trace)
                            if args.v100_trace is not None else {})
        counts = {"oracle_flip": 0, "baseline_difference": 0, "v100_difference": 0}
        for position in range(args.positions):
            for layer in range(len(model.layers)):
                key = (position, f"L{layer:02d}_moe_router_ids")
                chosen = set(np.fromfile(upgraded_stages[key], dtype="<f4").tolist())
                oracle_ids = set(np.fromfile(fp32_stages[key], dtype="<f4").tolist())
                baseline_ids = set(np.fromfile(baseline_stages[key], dtype="<f4").tolist())
                counts["oracle_flip"] += chosen != oracle_ids
                counts["baseline_difference"] += chosen != baseline_ids
                if candidate_stages:
                    v100_ids = set(np.fromfile(candidate_stages[key], dtype="<f4").tolist())
                    counts["v100_difference"] += chosen != v100_ids
        report["mlp_layer1_input_fp32_ablation"] = {
            "meaning": "hypothetical FP32 layer-one MLP input only; other storage retained",
            "oracle_mean_kl": sum(row["kl"] for row in upgraded_rows) / len(upgraded_rows),
            "oracle_top1_agreement": sum(row["top1_agree"] for row in upgraded_rows),
            "per_position_oracle": upgraded_rows,
            "router_set_counts": counts,
        }
    (args.out_dir / "precision_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in
                      ("status", "top1_agreement", "mean_kl")}, indent=2))
    if "three_way" in report:
        print("Three-way mean KL:", json.dumps({
            key: report["three_way"][key] for key in (
                "oracle_vs_cpu_mean_kl", "oracle_vs_v100_mean_kl",
                "cpu_vs_v100_mean_kl"
            )
        }))


if __name__ == "__main__":
    main()
