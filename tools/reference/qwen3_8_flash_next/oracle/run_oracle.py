import argparse
import ctypes
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextModel,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpTextDecoderLayer,
)

if os.environ.get("NINFER_ORACLE_DETERMINISTIC") == "1":
    # Oracle runs are correctness artifacts, not throughput benchmarks. Pin the
    # worker count explicitly so reduction order is stable across repeated runs.
    oracle_threads = int(os.environ.get("NINFER_ORACLE_THREADS", "1"))
    if oracle_threads < 1:
        raise RuntimeError("NINFER_ORACLE_THREADS must be >= 1")
    torch.manual_seed(0)
    torch.set_num_threads(oracle_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

class LazyPLEEmbedding(nn.Module):
    def __init__(self, ple_dir: str):
        super().__init__()
        self.ple_dir = ple_dir
        self.shards = {}
        self.weight = torch.empty(0, device="cpu")

    def get_shard(self, shard_idx: int):
        if shard_idx not in self.shards:
            path = os.path.join(self.ple_dir, f"shard_{shard_idx}.safetensors")
            self.shards[shard_idx] = safetensors.safe_open(path, framework="pt", device="cpu")
        return self.shards[shard_idx]

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        # ngram_ids: [batch, seq_len, 16]
        orig_shape = ngram_ids.shape
        flat_ids = ngram_ids.reshape(-1)
        out_list = []
        for r_tensor in flat_ids:
            r = int(r_tensor.item())
            shard_idx = r // 2500012
            local_row = r % 2500012
            shard = self.get_shard(shard_idx)
            i4 = shard.get_tensor("weight_i4")[local_row].to(torch.int32)  # [80] bytes
            scale = shard.get_tensor("weight_scale")[local_row].float()  # [10]
            low = (i4 & 0x0F).float() - 8.0
            high = (i4 >> 4).float() - 8.0
            unpacked = torch.stack([low, high], dim=-1).reshape(160)
            scale_exp = scale.repeat_interleave(16)
            out_list.append(unpacked * scale_exp)
        res = torch.stack(out_list, dim=0).reshape(*orig_shape, 160)
        return res

class LazyExperts(nn.Module):
    def __init__(self, config, layer_idx: int, model_dir: str):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = layer_idx
        self.model_dir = model_dir
        self.exp_f = None
        self.cache_gate_up = {}
        self.cache_down = {}
        # Disabled for the authoritative FP32 oracle. The supplementary
        # storage-profile experiment enables the two materialization boundaries
        # used by the CPU NVFP4 expert path.
        self.round_activations_to_bf16 = False
        # Supplementary CPU precision profile only; the FP32 oracle keeps the
        # independent model's original expert reduction.
        self.route_order_fma = False

    def get_exp_file(self):
        if self.exp_f is None:
            path = os.path.join(self.model_dir, f"ct-experts-layer{self.layer_idx:02d}.safetensors")
            self.exp_f = safetensors.safe_open(path, framework="pt", device="cpu")
        return self.exp_f

    def dequant_matrix(self, prefix: str) -> torch.Tensor:
        f = self.get_exp_file()
        packed = f.get_tensor(prefix + ".weight_packed")
        scale = f.get_tensor(prefix + ".weight_scale").float()
        glob = f.get_tensor(prefix + ".weight_global_scale").item()
        rows, half_cols = packed.shape
        cols = half_cols * 2
        low = (packed & 0x0F).to(torch.long)
        high = (packed >> 4).to(torch.long)
        val_low = FP4_LUT[low]
        val_high = FP4_LUT[high]
        unpacked = torch.stack([val_low, val_high], dim=-1).reshape(rows, cols)
        full_scale = scale.repeat_interleave(16, dim=-1) / glob
        return unpacked * full_scale

    def get_expert_weights(self, expert_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if expert_idx not in self.cache_gate_up:
            w_gate = self.dequant_matrix(f"model.language_model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.gate_proj")
            w_up = self.dequant_matrix(f"model.language_model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.up_proj")
            w_down = self.dequant_matrix(f"model.language_model.layers.{self.layer_idx}.mlp.experts.{expert_idx}.down_proj")
            gate_up = torch.cat([w_gate, w_up], dim=0)  # gate rows FIRST
            self.cache_gate_up[expert_idx] = gate_up
            self.cache_down[expert_idx] = w_down
        return self.cache_gate_up[expert_idx], self.cache_down[expert_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        pair_outputs = {} if self.route_order_fma else None
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0].item())
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            if self.round_activations_to_bf16:
                current_state = current_state.to(torch.bfloat16).float()
            gate_up, down = self.get_expert_weights(expert_idx)
            gate, up = F.linear(current_state, gate_up).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            if self.round_activations_to_bf16:
                current_hidden_states = current_hidden_states.to(torch.bfloat16).float()
            current_hidden_states = F.linear(current_hidden_states, down)
            if pair_outputs is not None:
                # Preserve the unweighted expert pair output. The V100 host
                # applies alpha in selected path order with one FP32 FMA.
                for row, token, path in zip(current_hidden_states,
                                            token_idx.tolist(), top_k_pos.tolist()):
                    pair_outputs[token, path] = row.detach().numpy().astype(np.float32, copy=True)
            else:
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        if pair_outputs is not None:
            if hidden_states.ndim != 2 or top_k_index.shape != top_k_weights.shape or \
                    top_k_index.shape != (hidden_states.shape[0], 10) or \
                    len(pair_outputs) != hidden_states.shape[0] * 10:
                raise RuntimeError("route-order FMA requires ten selected paths per token")
            fmaf = ctypes.CDLL("libm.so.6").fmaf
            fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
            fmaf.restype = ctypes.c_float
            result = np.zeros(tuple(hidden_states.shape), dtype=np.float32)
            for token in range(hidden_states.shape[0]):
                for path in range(10):
                    alpha = float(top_k_weights[token, path])
                    pair = pair_outputs[token, path]
                    for row in range(self.hidden_dim):
                        result[token, row] = fmaf(alpha, float(pair[row]),
                                                  float(result[token, row]))
            final_hidden_states = torch.from_numpy(result)

        # Each expert is used at most once per forward; a persistent FP32 cache (~20 MB per
        # expert) exhausts host memory on multi-token prompts.
        self.cache_gate_up.clear()
        self.cache_down.clear()
        return final_hidden_states

def build_oracle(model_dir: str, ple_dir: str):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        root_cfg = json.load(f)
    text_cfg_dict = root_cfg["text_config"]
    cfg = Qwen4ExpTextConfig(**text_cfg_dict)

    with torch.device("meta"):
        model = Qwen4ExpTextModel(cfg)

    # Replace heavy layers before allocating memory on CPU
    for l in range(48):
        model.layers[l].mlp.experts = LazyExperts(cfg, l, model_dir)
    model.layers[1].ple.ple_embedding.ngram_embedding = LazyPLEEmbedding(ple_dir)

    model.to_empty(device="cpu")
    model.float()
    model.eval()

    # Qwen4ExpTextRotaryEmbedding stores inv_freq/original_inv_freq as
    # persistent=False buffers. They are constructed on the meta device above,
    # so to_empty() allocates uninitialized CPU storage for them and
    # load_state_dict() cannot restore them from the checkpoint. Recreate those
    # buffers from config explicitly before the first forward pass.
    rotary_modules = [
        module for module in model.modules()
        if isinstance(module, Qwen4ExpTextRotaryEmbedding)
    ]
    if not rotary_modules:
        raise RuntimeError("reference model has no Qwen4ExpTextRotaryEmbedding")
    for module in rotary_modules:
        fresh = Qwen4ExpTextRotaryEmbedding(module.config, device="cpu")
        module.inv_freq = nn.Buffer(fresh.inv_freq.detach().clone(), persistent=False)
        module.original_inv_freq = nn.Buffer(
            fresh.original_inv_freq.detach().clone(), persistent=False
        )
        module.attention_scaling = fresh.attention_scaling

    # Load weights
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    open_safetensors = {}
    def get_raw_tensor(name: str):
        fn = weight_map[name]
        if fn not in open_safetensors:
            p = os.path.join(model_dir, fn)
            open_safetensors[fn] = safetensors.safe_open(p, framework="pt", device="cpu")
        return open_safetensors[fn].get_tensor(name)

    sd = {}
    lm_head_weight = None

    fp8_targets = [
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]

    for orig_key, fn in weight_map.items():
        if fn.startswith("ple-bf16-") or "ngram_embedding.weight" in orig_key:
            continue
        if orig_key == "lm_head.weight":
            lm_head_weight = get_raw_tensor(orig_key).float()
            continue
        if not orig_key.startswith("model.language_model."):
            continue
        
        target_key = orig_key[len("model.language_model."):]

        # Skip expert weights
        if ".mlp.experts." in target_key:
            continue

        # Check if FP8 row scale
        is_fp8 = False
        for tgt in fp8_targets:
            if target_key.endswith(tgt + ".weight"):
                is_fp8 = True
                base_name = orig_key[:-len(".weight")]
                w_packed = get_raw_tensor(base_name + ".weight")
                w_scale = get_raw_tensor(base_name + ".weight_scale").float()
                w_f32 = w_packed.to(torch.float32)
                if w_scale.ndim == 1:
                    w_f32 = w_f32 * w_scale.unsqueeze(1)
                else:
                    w_f32 = w_f32 * w_scale
                sd[target_key] = w_f32
                break
            elif target_key.endswith(tgt + ".weight_scale"):
                is_fp8 = True  # handled with .weight
                break

        if not is_fp8:
            t = get_raw_tensor(orig_key)
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.float()
            sd[target_key] = t

    res = model.load_state_dict(sd, strict=False)
    
    missing_set = set(res.missing_keys)
    unexpected_set = set(res.unexpected_keys)

    assert len(missing_set) == 0, f"Missing keys found: {missing_set}"
    assert len(unexpected_set) == 0, f"Unexpected keys found: {unexpected_set}"

    # Assert PLE index buffers
    ple_emb = model.layers[1].ple.ple_embedding
    expected_multipliers = [23703573157769, 20109073645365, 8052911324071]
    expected_offsets = [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]
    expected_vocab_sizes = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                            20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

    assert ple_emb.layer_multipliers.tolist() == expected_multipliers, "PLE layer_multipliers mismatch!"
    assert ple_emb.ngram_heads_offsets.tolist() == expected_offsets, "PLE ngram_heads_offsets mismatch!"
    assert ple_emb.ngram_heads_vocab_sizes.tolist() == expected_vocab_sizes, "PLE ngram_heads_vocab_sizes mismatch!"

    return model, lm_head_weight

def register_hooks(model: nn.Module):
    stage_outputs = {}

    def save_output(name: str):
        def hook(module, input, output):
            val = output[0] if isinstance(output, (tuple, list)) else output
            stage_outputs[name] = val.detach().clone()
        return hook

    def save_input(name: str):
        def hook(module, input):
            val = input[0] if isinstance(input, (tuple, list)) else input
            stage_outputs[name] = val.detach().clone()
        return hook

    def save_router(prefix: str):
        def hook(module, input, output):
            _, router_alpha, router_ids = output
            stage_outputs[prefix + "moe_router_alpha"] = router_alpha.detach().clone()
            stage_outputs[prefix + "moe_router_ids"] = router_ids.detach().to(torch.int32).clone()
        return hook

    def save_shared_scale(prefix: str):
        def hook(module, input, output):
            stage_outputs[prefix + "moe_shared_scale"] = torch.sigmoid(output.detach()).clone()
        return hook

    def save_gdn_norm_input(prefix: str):
        def hook(module, input):
            stage_outputs[prefix + "gdn_recurrent_output"] = input[0].detach().clone()
        return hook

    # Embedding
    model.embed_tokens.register_forward_hook(save_output("embedding"))

    # Layer 1 PLE injection
    model.layers[1].ple.register_forward_hook(save_output("ple_injection"))

    for l in range(48):
        prefix = f"L{l:02d}_"
        # attn_hyper_connection input & output
        model.layers[l].attn_hyper_connection.register_forward_pre_hook(save_input(prefix + "hyper_in"))
        model.layers[l].attn_hyper_connection.register_forward_hook(save_output(prefix + "attn_block_input"))

        # linear_attn or self_attn
        if hasattr(model.layers[l], "linear_attn"):
            gdn = model.layers[l].linear_attn
            gdn.register_forward_hook(save_output(prefix + "attn_block_output"))
            gdn.in_proj_qkv.register_forward_hook(save_output(prefix + "gdn_qkv_projected"))
            gdn.in_proj_z.register_forward_hook(save_output(prefix + "gdn_z"))
            gdn.in_proj_a.register_forward_hook(save_output(prefix + "gdn_a"))
            gdn.in_proj_b.register_forward_hook(save_output(prefix + "gdn_b"))
            gdn.norm.register_forward_pre_hook(save_gdn_norm_input(prefix))
            gdn.out_proj.register_forward_pre_hook(save_input(prefix + "gdn_gated_output"))
        elif hasattr(model.layers[l], "self_attn"):
            attn = model.layers[l].self_attn
            attn.register_forward_hook(save_output(prefix + "attn_block_output"))
            attn.q_proj.register_forward_hook(save_output(prefix + "qsa_q_proj"))
            attn.k_proj.register_forward_hook(save_output(prefix + "qsa_k_proj"))
            attn.v_proj.register_forward_hook(save_output(prefix + "qsa_v_proj"))
            attn.q_norm.register_forward_hook(save_output(prefix + "qsa_query_normed"))
            attn.k_norm.register_forward_hook(save_output(prefix + "qsa_key_normed"))
            attn.o_proj.register_forward_pre_hook(save_input(prefix + "qsa_gated"))

        # mlp_hyper_connection input & output
        model.layers[l].mlp_hyper_connection.register_forward_pre_hook(save_input(prefix + "hyper_after_attn"))
        model.layers[l].mlp_hyper_connection.register_forward_hook(save_output(prefix + "mlp_block_input"))

        # mlp
        model.layers[l].mlp.register_forward_hook(save_output(prefix + "mlp_block_output"))
        if hasattr(model.layers[l].mlp, "gate"):
            model.layers[l].mlp.gate.register_forward_hook(save_router(prefix))
            model.layers[l].mlp.shared_expert_gate.register_forward_hook(
                save_shared_scale(prefix)
            )

        # layer output
        model.layers[l].register_forward_hook(save_output(prefix + "hyper_after_mlp"))

    # Final mixer
    model.hyper_connection_mixer.register_forward_hook(save_output("final_hidden"))

    return stage_outputs

def derive_gdn_stages(gdn: nn.Module, stage_outputs: dict, prefix: str):
    qkv = stage_outputs[prefix + "gdn_qkv_projected"]
    z = stage_outputs[prefix + "gdn_z"]
    a = stage_outputs[prefix + "gdn_a"]
    b = stage_outputs[prefix + "gdn_b"]

    qkv_channels = qkv.transpose(1, 2)
    weight = gdn.conv1d.weight
    conv = F.conv1d(
        qkv_channels.to(weight.dtype),
        weight=weight,
        bias=gdn.conv1d.bias,
        padding=weight.shape[-1] - 1,
        groups=qkv_channels.shape[1],
    )[:, :, : qkv_channels.shape[-1]]
    conv = F.silu(conv).transpose(1, 2)
    query, key, value = torch.split(conv, [2048, 2048, 6144], dim=-1)

    beta = torch.sigmoid(b.float())
    g = -torch.exp(gdn.A_log.float()) * F.softplus(a.float() + gdn.dt_bias.float())
    projected = torch.cat([qkv, z], dim=-1)
    return projected, query, key, value, z, g, beta


def apply_text_qsa_rope(x: torch.Tensor, position: int, theta: float = 1.0e7) -> torch.Tensor:
    """Apply Qwen4Exp text MRoPE for equal T/H/W positions."""
    if x.shape[-1] != 256:
        raise RuntimeError(f"expected QSA head_dim=256, got {x.shape[-1]}")
    half = 32
    pair = torch.arange(half, dtype=torch.float32, device=x.device)
    inv_freq = torch.exp((-2.0 * pair / 64.0) * math.log(theta))
    angle = float(position) * inv_freq
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    rotary = x[..., :64].float()
    first = rotary[..., :half]
    second = rotary[..., half:]
    rotated = torch.cat(
        [first * cos - second * sin, second * cos + first * sin], dim=-1
    )
    return torch.cat([rotated, x[..., 64:].float()], dim=-1)


def parse_token_list(ids: str, ids_file: str, token_id: int) -> List[int]:
    if ids and ids_file:
        raise SystemExit("--ids and --ids-file are mutually exclusive")
    if ids_file:
        with open(ids_file, "r", encoding="utf-8") as f:
            root = json.load(f)
        if isinstance(root, dict):
            if "token_ids" not in root:
                raise SystemExit("--ids-file JSON object must contain a token_ids array")
            root = root["token_ids"]
        if not isinstance(root, list) or not root:
            raise SystemExit("--ids-file must contain a non-empty JSON token ID array")
        return [int(t) for t in root]
    if ids:
        parsed = [int(t.strip()) for t in ids.split(",") if t.strip()]
        if not parsed:
            raise SystemExit("--ids did not contain any token IDs")
        return parsed
    return [token_id]


def dump_logits_only(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    token_list: List[int],
    dump_root: str,
    chunk_size: int,
):
    if chunk_size <= 0:
        raise SystemExit("--logits-chunk-size must be positive")
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise RuntimeError("logits-only oracle expects hidden states shaped [1, seq, hidden]")
    if hidden_states.shape[1] != len(token_list):
        raise RuntimeError("logits-only oracle hidden/token length mismatch")

    os.makedirs(dump_root, exist_ok=True)
    manifest = {
        "oracle": {
            "kind": "qwen3_8_flash_next_cpu_fp32_logits_only",
            "teacher_forced": True,
            "dtype": "FP32",
        },
        "positions": [],
    }

    total = len(token_list)
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        with torch.no_grad():
            logits_chunk = F.linear(hidden_states[:, start:end, :], lm_head_weight)[0]

        for local_idx in range(end - start):
            pos = start + local_idx
            logits_pos = logits_chunk[local_idx].detach().contiguous().cpu()
            if not torch.isfinite(logits_pos).all():
                bad = int((~torch.isfinite(logits_pos)).sum().item())
                raise RuntimeError(
                    f"CPU oracle logits position {pos} contains {bad} non-finite "
                    "values; refusing to publish oracle"
                )
            data = logits_pos.numpy().astype(np.float32, copy=False).tobytes()
            bin_file = f"pos{pos:06d}_logits.bin"
            with open(os.path.join(dump_root, bin_file), "wb") as out:
                out.write(data)

            manifest["positions"].append({
                "position": pos,
                "token_id": int(token_list[pos]),
                "tensors": [{
                    "name": "logits",
                    "dtype": "FP32",
                    "shape": [int(logits_pos.numel())],
                    "file": bin_file,
                    "bytes": len(data),
                }],
            })

            if pos < 4 or (pos + 1) % 128 == 0 or pos + 1 == total:
                top1 = int(torch.argmax(logits_pos).item())
                print(
                    f"[logits-only {pos + 1:04d}/{total:04d}] "
                    f"token={token_list[pos]} top1={top1}"
                )

        del logits_chunk

    manifest_tmp = os.path.join(dump_root, "manifest.json.tmp")
    manifest_path = os.path.join(dump_root, "manifest.json")
    with open(manifest_tmp, "w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2)
    os.replace(manifest_tmp, manifest_path)
    print(f"Dumped {total} FP32 logits-only positions to {manifest_path}")


# The MTP reference lives in mtp_reference.py, faithful to vLLM's
# Qwen4ExpMultiTokenPredictor. The class that used to sit here mixed the four
# hyper streams down in the stem and repeated the result, omitted
# pre_fc_norm_hidden, and invented a second final mixer; see that module's
# docstring for the corrected contract and its sources.
from mtp_reference import (  # noqa: E402
    ALL_STAGES,
    Qwen4ExpMTPReference as Qwen4ExpMTP,
    dump_stages,
    load_mtp_weights,
    make_mtp_config,
)


def run_mtp_real(model_dir: str, ple_dir: str, token_list, draft_steps: int, dump_root: str):
    """Real-weight MTP reference over a teacher-forced prompt.

    Scheme A: the target's post-combine multi-stream hidden at every position is
    the draft's backbone input. Draft slot t = hidden[t] + embedding(token[t+1]) at
    position t. The first N-1 pairs seed the draft KV. The first draft then uses
    the last target hidden and the target's greedy next token at position N-1.
    Later drafts use the previous MTP multi_hidden and its own argmax token,
    advancing all three MRoPE rows."""
    from transformers.cache_utils import DynamicCache

    print(f"Building target reference from {model_dir} ...")
    model, lm_head_weight = build_oracle(model_dir, ple_dir)
    stage_outputs = register_hooks(model)
    input_ids = torch.tensor([token_list], dtype=torch.long)
    with torch.no_grad():
        target_output = model(input_ids=input_ids, use_cache=False)
        anchor = int(F.linear(target_output.last_hidden_state[:, -1, :],
                              lm_head_weight).argmax(-1).item())
    # Layer-47 output is the multi-stream state the final mixer consumes: vLLM's
    # `multi_hidden`, captured into `_mtp_hidden_buffer` for the drafter.
    backbone_multi = stage_outputs["L47_hyper_after_mlp"]  # [1, N, hc*H]
    embeds = stage_outputs["embedding"]  # [1, N, H]
    n = len(token_list)
    if n < 2:
        raise SystemExit("--mtp-real needs at least two prompt tokens (hidden[t] pairs with token[t+1])")

    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = make_mtp_config(json.load(f)["text_config"])
    mtp = Qwen4ExpMTP(cfg)
    mtp.float().eval()
    load_mtp_weights(mtp, model_dir)

    manifest = {"positions": []}
    cache = DynamicCache()
    slots = n - 1
    pos_ids = torch.arange(slots, dtype=torch.long).view(1, 1, slots).expand(3, 1, slots)
    mask = torch.ones((1, 1, slots, slots), dtype=torch.bool).tril()
    with torch.no_grad():
        stages = mtp(embeds[:, 1:n, :], backbone_multi[:, 0:slots, :], pos_ids, mask, cache)
    for t in range(slots):
        per_pos = {k: v[0, t] for k, v in stages.items()}
        fed = token_list[t + 1]
        got = int(per_pos["mtp_draft_tokens"].item())
        if t + 2 < n:
            verdict = "HIT" if got == token_list[t + 2] else "miss"
            print(f"[prefill slot {t:04d}] hidden[{t}] + tok[{t + 1}]={fed} -> draft {got}  next={token_list[t + 2]} {verdict}")
        else:
            print(f"[prefill slot {t:04d}] hidden[{t}] + tok[{t + 1}]={fed} -> draft {got}")
        if dump_root:
            dump_stages(dump_root, t, fed, (t, t, t), per_pos, manifest)

    # The teacher's last hidden anchors the first draft; only later steps carry MTP hidden.
    multi = backbone_multi[:, -1:, :]
    tok = anchor
    for k in range(draft_steps):
        p_ = slots + k
        emb = model.embed_tokens(torch.tensor([[tok]], dtype=torch.long)).float()
        pid = torch.full((3, 1, 1), p_, dtype=torch.long)
        vis = cache.get_seq_length() + 1
        m = torch.ones((1, 1, 1, vis), dtype=torch.bool)
        with torch.no_grad():
            st = mtp(emb, multi, pid, m, cache)
        per_pos = {kk: v[0, 0] for kk, v in st.items()}
        nxt = int(per_pos["mtp_draft_tokens"].item())
        print(f"[draft step {k}] fed tok {tok} at pos {p_} -> draft {nxt}")
        if dump_root:
            dump_stages(dump_root, p_, tok, (p_, p_, p_), per_pos, manifest)
        multi, tok = st["mtp_multi_hidden"], nxt

    if dump_root:
        with open(os.path.join(dump_root, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Dumped MTP reference states to {dump_root}/manifest.json")



def main():
    parser = argparse.ArgumentParser(description="Authoritative HF Qwen4Exp CPU FP32 reference oracle for Qwen3.8-Flash-Next")
    parser.add_argument("--model-dir", default=r"E:\NInfer\qwen3_8_flash_next\source\mixed", help="Path to mixed source model dir")
    parser.add_argument("--ple-dir", default=r"E:\NInfer\qwen3_8_flash_next\source\ple\ples_int4", help="Path to PLE INT4 shards")
    parser.add_argument("--token-id", type=int, default=248045, help="Single token ID to execute")
    parser.add_argument("--ids", type=str, default="", help="Comma-separated token IDs to execute in sequence")
    parser.add_argument("--ids-file", type=str, default="", help="JSON file containing a token ID array or {\\\"token_ids\\\": [...]}")
    parser.add_argument("--dump-states", type=str, default="", help="Directory to dump state tensors and manifest")
    parser.add_argument("--dump-logits", type=str, default="", help="Directory to dump only FP32 logits and a Phase 11-compatible manifest")
    parser.add_argument("--logits-chunk-size", type=int, default=8, help="LM-head positions per chunk in --dump-logits mode")
    parser.add_argument("--mtp-synthetic", action="store_true", help="Run MTP synthetic architecture parity step")
    parser.add_argument("--mtp-real", action="store_true", help="Run the real-weight MTP reference over --ids (scheme A) and chained draft steps")
    parser.add_argument("--draft-steps", type=int, default=3, help="Chained draft steps after the prompt in --mtp-real")
    args = parser.parse_args()

    if args.dump_states and args.dump_logits:
        raise SystemExit("--dump-states and --dump-logits are mutually exclusive")

    if args.mtp_real:
        ids = parse_token_list(args.ids, args.ids_file, args.token_id)
        run_mtp_real(args.model_dir, args.ple_dir, ids, args.draft_steps, args.dump_states)
        return

    if args.mtp_synthetic:
        print("Running Qwen4ExpMTP synthetic architecture step (vLLM-faithful stem) ...")
        with open(os.path.join(args.model_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = make_mtp_config(json.load(f)["text_config"])
        mtp = Qwen4ExpMTP(cfg)
        mtp.eval()
        dim = 2560
        with torch.no_grad():
            for p in mtp.parameters():
                p.zero_()
            mtp.fc_embedding.weight.copy_(torch.eye(dim))
            mtp.fc_hidden.weight.copy_(torch.eye(dim))
            for i in range(100):
                mtp.lm_head.weight[i, 0] = float(i + 1)
        # Distinct per-stream input so the four streams are visibly different
        # after the stem; an all-ones backbone could not tell a correct stem from
        # the old mix-and-repeat one.
        input_emb = torch.ones(1, 1, dim)
        backbone_h = torch.arange(1, 5, dtype=torch.float32).repeat_interleave(dim).view(1, 1, 4 * dim)
        with torch.no_grad():
            stages = mtp(input_emb, backbone_h)
        stages = {k: v[0, 0] for k, v in stages.items()}
        print(f"MTP Reference Output: Draft Token = {int(stages['mtp_draft_tokens'].item())}")
        if args.dump_states:
            manifest = {"positions": []}
            dump_stages(args.dump_states, 0, 0, (0, 0, 0), stages, manifest)
            with open(os.path.join(args.dump_states, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"Dumped MTP oracle states to {args.dump_states}/manifest.json")
        return

    print(f"Building authoritative Transformers Qwen4ExpTextModel from {args.model_dir} ...")
    model, lm_head_weight = build_oracle(args.model_dir, args.ple_dir)
    stage_outputs = register_hooks(model) if args.dump_states else None

    token_list = parse_token_list(args.ids, args.ids_file, args.token_id)

    input_ids = torch.tensor([token_list], dtype=torch.long)
    if args.dump_logits:
        print(f"Running logits-only teacher-forced forward pass for {len(token_list)} tokens ...")
    else:
        print(f"Running teacher-forced forward pass for {len(token_list)} tokens: {token_list} ...")

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    if args.dump_logits:
        dump_logits_only(
            out.last_hidden_state,
            lm_head_weight,
            token_list,
            args.dump_logits,
            args.logits_chunk_size,
        )
        return

    with torch.no_grad():
        logits_all = F.linear(out.last_hidden_state, lm_head_weight)  # [1, seq_len, vocab_size]
    if not torch.isfinite(logits_all).all():
        bad = int((~torch.isfinite(logits_all)).sum().item())
        raise RuntimeError(
            f"CPU oracle produced {bad} non-finite logits; refusing to publish trace"
        )

    manifest = {"positions": []}

    for pos in range(len(token_list)):
        tok = token_list[pos]
        logits_pos = logits_all[0, pos]
        top5_vals, top5_ids = torch.topk(logits_pos, 5)

        print(f"\n[Position {pos:04d}] Token {tok}:")
        print(f"  Argmax: {top5_ids[0].item()} (logit: {top5_vals[0].item():.2f})")
        print("  Top-5: ", ", ".join(f"{tid.item()}:{val.item():.2f}" for val, tid in zip(top5_vals, top5_ids)))

        if args.dump_states:
            assert stage_outputs is not None
            pos_dir = os.path.join(args.dump_states, f"pos{pos:04d}")
            os.makedirs(pos_dir, exist_ok=True)
            pos_records = []

            # Format stage names
            stages = []
            
            # embedding & hyper_init
            emb_pos = stage_outputs["embedding"][0, pos]
            stages.append(("embedding", emb_pos))
            hyper_init_pos = torch.cat([emb_pos, emb_pos, emb_pos, emb_pos], dim=-1)
            stages.append(("hyper_init", hyper_init_pos))

            if "ple_injection" in stage_outputs:
                ple_pos = stage_outputs["ple_injection"][0, pos]
                stages.append(("ple_injection", ple_pos))

            for l in range(48):
                prefix = f"L{l:02d}_"
                if l == 1:
                    # hyper_after_ple is the input to layer 1 attn_hyper_connection
                    stages.append(("hyper_after_ple", stage_outputs[prefix + "hyper_in"][0, pos]))

                stages.append((prefix + "attn_block_input", stage_outputs[prefix + "attn_block_input"][0, pos]))
                if prefix + "gdn_qkv_projected" in stage_outputs:
                    gdn = model.layers[l].linear_attn
                    projected, query, key, value, z, g, beta = derive_gdn_stages(
                        gdn, stage_outputs, prefix
                    )
                    recurrent = stage_outputs[prefix + "gdn_recurrent_output"].reshape(
                        1, len(token_list), 48, 128
                    )
                    stages.append((prefix + "gdn_projected", projected[0, pos]))
                    stages.append((prefix + "gdn_query", query[0, pos]))
                    stages.append((prefix + "gdn_key", key[0, pos]))
                    stages.append((prefix + "gdn_value", value[0, pos]))
                    stages.append((prefix + "gdn_z", z[0, pos]))
                    stages.append((prefix + "gdn_g", g[0, pos]))
                    stages.append((prefix + "gdn_beta", beta[0, pos]))
                    stages.append((
                        prefix + "gdn_recurrent_output",
                        recurrent[0, pos].reshape(-1),
                    ))
                    stages.append((
                        prefix + "gdn_gated_output",
                        stage_outputs[prefix + "gdn_gated_output"][0, pos],
                    ))
                if prefix + "qsa_q_proj" in stage_outputs:
                    q_proj = stage_outputs[prefix + "qsa_q_proj"][0, pos]
                    k_proj = stage_outputs[prefix + "qsa_k_proj"][0, pos]
                    v_proj = stage_outputs[prefix + "qsa_v_proj"][0, pos]
                    q_heads = q_proj.reshape(24, 512)
                    stages.append((
                        prefix + "qsa_projected",
                        torch.cat([q_proj, k_proj, v_proj], dim=-1),
                    ))
                    gate = q_heads[:, 256:].reshape(-1)
                    query_normed = stage_outputs[prefix + "qsa_query_normed"][0, pos]
                    key_normed = stage_outputs[prefix + "qsa_key_normed"][0, pos]
                    query = apply_text_qsa_rope(query_normed, pos)
                    key = apply_text_qsa_rope(key_normed, pos)
                    gated = stage_outputs[prefix + "qsa_gated"][0, pos]
                    attended = gated.float() / torch.sigmoid(gate.float())
                    stages.append((prefix + "qsa_gate", gate))
                    stages.append((prefix + "qsa_query", query))
                    stages.append((prefix + "qsa_key", key))
                    stages.append((prefix + "qsa_value", v_proj))
                    stages.append((prefix + "qsa_attended", attended))
                    stages.append((prefix + "qsa_gated", gated))
                stages.append((prefix + "attn_block_output", stage_outputs[prefix + "attn_block_output"][0, pos]))
                stages.append((prefix + "hyper_after_attn", stage_outputs[prefix + "hyper_after_attn"][0, pos]))
                stages.append((prefix + "mlp_block_input", stage_outputs[prefix + "mlp_block_input"][0, pos]))
                if prefix + "moe_router_ids" in stage_outputs:
                    stages.append((
                        prefix + "moe_router_ids",
                        stage_outputs[prefix + "moe_router_ids"][pos],
                    ))
                    stages.append((
                        prefix + "moe_router_alpha",
                        stage_outputs[prefix + "moe_router_alpha"][pos],
                    ))
                    stages.append((
                        prefix + "moe_shared_scale",
                        stage_outputs[prefix + "moe_shared_scale"][pos],
                    ))
                stages.append((prefix + "mlp_block_output", stage_outputs[prefix + "mlp_block_output"][0, pos]))
                stages.append((prefix + "hyper_after_mlp", stage_outputs[prefix + "hyper_after_mlp"][0, pos]))

            stages.append(("final_hidden", stage_outputs["final_hidden"][0, pos]))
            stages.append(("logits", logits_pos))

            for name, tensor in stages:
                array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                if not np.isfinite(array).all():
                    bad = int((~np.isfinite(array)).sum())
                    raise RuntimeError(
                        f"CPU oracle stage {name} at position {pos} contains "
                        f"{bad} non-finite values; refusing to publish trace"
                    )
                f32_bytes = array.tobytes()
                bin_file = f"{name}.bin"
                with open(os.path.join(pos_dir, bin_file), "wb") as f:
                    f.write(f32_bytes)
                pos_records.append({
                    "name": name,
                    "dtype": "FP32",
                    "shape": list(tensor.shape),
                    "file": f"pos{pos:04d}/{bin_file}",
                    "bytes": len(f32_bytes),
                })

            manifest["positions"].append({
                "position": pos,
                "token_id": tok,
                "tensors": pos_records,
            })

    if args.dump_states:
        with open(os.path.join(args.dump_states, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nDumped {len(token_list)} positions to {args.dump_states}/manifest.json")

if __name__ == "__main__":
    main()
