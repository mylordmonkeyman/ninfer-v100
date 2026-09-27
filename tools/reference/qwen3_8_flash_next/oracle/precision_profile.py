"""Explicit CPU storage profile for the Phase 11 Flash-Next decode experiment.

This is a *supplementary implementation-profile reference*, not the independent
FP32 mathematical oracle in run_oracle.py.  It models observable BF16 tensors
and persistent-state casts while using CPU FP32 arithmetic between boundaries.
It must be calibrated against the real V100 stages before its output can be
used to attribute a qualification failure to precision alone.
"""

from dataclasses import dataclass
import ctypes
import math
from types import MethodType

import numpy as np


@dataclass(frozen=True)
class PrecisionProfile:
    name: str
    activation_bf16: bool
    conv_state_bf16: bool
    ssm_state_bf16: bool
    attention_kv_bf16: bool
    hyper_state_bf16: bool
    logits_bf16: bool


# The selected Phase 11 V100 workflow keeps hyper and GDN SSM master state in
# FP32, and uses FP32 diagnostic readouts for several GDN and hyper operations.
# The model's conv histories, QSA KV, projection materializations and output
# logits remain BF16.  These are *storage* facts, not a claim about the order
# or precision of arithmetic inside fused CUDA kernels.
PROFILES = {
    "fp32": PrecisionProfile("fp32", False, False, False, False, False, False),
    "v100-phase11-storage": PrecisionProfile(
        "v100-phase11-storage", True, True, False, True, False, True
    ),
}


def round_to_bf16(value):
    """Round to BF16 storage and return FP32 for CPU arithmetic."""
    import torch

    if not value.is_floating_point():
        raise TypeError("BF16 boundary requires floating-point input")
    return value.to(torch.bfloat16).to(torch.float32)


class _RestoreForward:
    def __init__(self, module, forward):
        self.module = module
        self.forward = forward

    def remove(self):
        self.module.forward = self.forward


_LIBM_FMAF = ctypes.CDLL("libm.so.6").fmaf
_LIBM_FMAF.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_LIBM_FMAF.restype = ctypes.c_float


def _fused_hyper_update(master, block_output, gates):
    """Reproduce CUDA FP32 fused block*gate+master, retaining FP32 master."""
    import torch

    if master.dtype != torch.float32 or block_output.dtype != torch.float32:
        raise TypeError("Phase 11 fused hyper update expects FP32 CPU tensors")
    if master.shape[-1] != 10240 or block_output.shape[-1] != 2560 or gates.shape[-1] != 4:
        raise ValueError("unexpected hyper injection dimensions")
    source = master.detach().cpu().reshape(-1, 4, 2560).numpy()
    block = block_output.detach().cpu().reshape(-1, 2560).numpy()
    scale = gates.detach().cpu().reshape(-1, 4).numpy()
    result = np.empty_like(source)
    for token in range(source.shape[0]):
        for stream in range(4):
            gate = float(scale[token, stream])
            for feature in range(2560):
                result[token, stream, feature] = _LIBM_FMAF(
                    float(block[token, feature]), gate,
                    float(source[token, stream, feature]))
    return torch.from_numpy(result.reshape(master.shape))


def _decoder_forward_fused_hyper(self, hidden_states, position_embeddings,
                                 attention_mask=None, conv_mask=None,
                                 past_key_values=None, ple_input_ids=None,
                                 **kwargs):
    """Mirror the model decoder while modeling V100 fused hyper updates."""
    if self.ple is not None:
        hidden_states = hidden_states + self.ple(
            hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask)
    block_input, master, gates = self.attn_hyper_connection(hidden_states)
    if self.layer_type == "linear_attention":
        block_output = self.linear_attn(
            block_input, cache_params=past_key_values,
            attention_mask=conv_mask, **kwargs)
    else:
        block_output, _ = self.self_attn(
            block_input, position_embeddings, attention_mask=attention_mask,
            past_key_values=past_key_values, **kwargs)
    hidden_states = _fused_hyper_update(master, block_output, gates)
    block_input, master, gates = self.mlp_hyper_connection(hidden_states)
    block_output = self.mlp(block_input)
    return _fused_hyper_update(master, block_output, gates)


def _attention_hyper_from_bf16_shadow(self, hyper_input):
    """Decode attention prepare with the selected V100 storage boundaries.

    The V100 diagnostic retains the master hyper state in FP32, but the
    attention prepare consumes a BF16 shadow. Its normalized streams and
    SiLU low-rank mixer output are also BF16. Return the original master as
    the pass-through value so the CPU model does not round persistent state.
    This does not reproduce the CUDA dot-product reduction association.
    """
    import torch
    import torch.nn.functional as F

    shadow = round_to_bf16(hyper_input)
    normalized = round_to_bf16(self.hc_norm(shadow))
    low_rank = round_to_bf16(F.silu(
        self.input_mix_weight_down(normalized) / self.hc_count
    ))
    mix_gate = torch.sigmoid(self.input_mix_weight_up(low_rank))
    mix_gate = mix_gate.unflatten(-1, (self.hc_count, self.hidden_size))
    mixed = (mix_gate * normalized.unflatten(
        -1, (self.hc_count, self.hidden_size))).mean(dim=-2)
    if getattr(self, "_phase11_capture_attention_raw", False):
        # Diagnostic only: expose the same mathematical mixer boundary before
        # the profile's BF16 output materialization.
        self._phase11_attention_raw = mixed.detach().to(torch.float32).cpu().clone()
    if self.block_inject_weight is None:
        return round_to_bf16(mixed)
    injection = 2 * torch.sigmoid(
        self.block_inject_weight(normalized) / self.hc_count
    )
    return round_to_bf16(mixed), hyper_input, injection


def _ple_from_bf16_shadow(self, hidden_states, input_ids, past_key_values,
                          conv_mask=None):
    """Decode PLE with the V100 BF16 query, gate and convolution inputs."""
    import torch
    embedding = self.ple_embedding(input_ids, past_key_values)
    key = self.norm_key(self.key_proj(embedding)).unflatten(
        -1, (self.hc_count, self.hidden_size))
    value = self.value_proj(embedding)
    query = self.norm_query(round_to_bf16(hidden_states)).unflatten(
        -1, (self.hc_count, self.hidden_size))
    gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = round_to_bf16(torch.sigmoid(gate) * value.unsqueeze(-2))
    gated = gated.flatten(-2)
    normalized = self.norm_conv(gated)
    if conv_mask is not None:
        gated = (gated * conv_mask[:, :, None]).to(gated.dtype)
        normalized = (normalized * conv_mask[:, :, None]).to(normalized.dtype)
    return gated + self._short_conv(normalized, past_key_values)


def install_projection_boundaries(model, profile, *, mlp_block_input_fp32=False,
                                  mlp_block_input_fp32_layers=(),
                                  fused_hyper_updates=False):
    """Materialize selected projection outputs; return removable hook handles.

    This only models projected activation storage.  In particular it does not
    claim to reproduce FP8 GEMV reduction order or fused GDN readout rounding.
    """
    import torch

    if not profile.activation_bf16:
        return []
    handles = []
    handles.append(model.embed_tokens.register_forward_hook(
        lambda _module, _args, output: round_to_bf16(output)
    ))
    handles.append(model.hyper_connection_mixer.register_forward_hook(
        lambda _module, _args, output: round_to_bf16(output)
    ))
    for layer_index, layer in enumerate(model.layers):
        if fused_hyper_updates:
            handles.append(_RestoreForward(layer, layer.forward))
            layer.forward = MethodType(_decoder_forward_fused_hyper, layer)
        projections = []
        hyper = layer.attn_hyper_connection
        handles.append(_RestoreForward(hyper, hyper.forward))
        hyper.forward = MethodType(_attention_hyper_from_bf16_shadow, hyper)
        if hasattr(layer, "linear_attn"):
            # The selected V100 diagnostic's FP32_GDN_PROJECTION path feeds
            # unrounded QKV directly into its fused convolution. It still
            # writes a BF16 projection mirror and uses BF16 for z. Rounding
            # QKV here would change the conv input before that fused step.
            projections.append(layer.linear_attn.in_proj_z)
            # With FP32_GDN_OUTPUT_INPUT and FP32_HYPER_INJECT_APPLY enabled,
            # the selected V100 decode applies the FP32 projection accumulator
            # to the FP32 hyper master. Its separate BF16 output mirror is
            # emitted for the normal kernel path but is not the injected value.
            # Keep the independent CPU projection unrounded at this boundary.
        if hasattr(layer, "self_attn"):
            # QSA gates its selected-attention value before the output
            # projection. V100 stores that 6144-wide input as BF16; rounding
            # only o_proj's output misses the amplified input error.
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda _module, args: (round_to_bf16(args[0]),)
            ))
            projections.extend((layer.self_attn.q_proj, layer.self_attn.k_proj,
                                layer.self_attn.v_proj, layer.self_attn.o_proj))
            handles.append(layer.self_attn.register_forward_hook(
                lambda _module, _args, output:
                    (round_to_bf16(output[0]), output[1])
            ))
        if layer.ple is not None:
            ple = layer.ple
            handles.append(_RestoreForward(ple, ple.forward))
            ple.forward = MethodType(_ple_from_bf16_shadow, ple)
            handles.append(ple.ple_embedding.register_forward_hook(
                lambda _module, _args, output: round_to_bf16(output)
            ))
            for norm in (ple.norm_key, ple.norm_query, ple.norm_conv):
                handles.append(norm.register_forward_hook(
                    lambda _module, _args, output: round_to_bf16(output)
                ))
            projections.extend((layer.ple.key_proj, layer.ple.value_proj))
            handles.append(layer.ple.register_forward_hook(
                lambda _module, _args, output: round_to_bf16(output)
            ))
        handles.append(layer.attn_hyper_connection.register_forward_hook(
            lambda _module, _args, output:
                (round_to_bf16(output[0]), output[1], output[2])
        ))
        if not mlp_block_input_fp32 and layer_index not in mlp_block_input_fp32_layers:
            handles.append(layer.mlp_hyper_connection.register_forward_hook(
                lambda _module, _args, output:
                    (round_to_bf16(output[0]), output[1], output[2])
            ))
        handles.append(layer.mlp.register_forward_hook(
            lambda _module, _args, output: round_to_bf16(output)
        ))
        # The V100 host-backed shared path keeps gate/up dot products in FP32
        # and rounds only SiLU(gate)*up to BF16 before shared_down. Its FP32
        # down result joins the routed FP32 sum before the final BF16 output.
        # Rounding gate/up outputs and shared_down separately at these points
        # creates extra, non-candidate precision loss in the CPU reference.
        handles.append(layer.mlp.shared_expert.down_proj.register_forward_pre_hook(
            lambda _module, args: (round_to_bf16(args[0]),)
        ))
        for module in projections:
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"expected a projection, got {type(module)}")
            handles.append(module.register_forward_hook(
                lambda _module, _args, output: round_to_bf16(output)
            ))
    return handles


def materialize_persistent_states(cache, model, profile):
    """Round cache values *after each token*, so errors persist in the prefix."""
    if cache is None or profile.name == "fp32":
        return
    if len(cache.layers) != len(model.layers):
        raise RuntimeError("incomplete cache: cannot model sequential precision")
    for layer_idx, (cache_layer, model_layer) in enumerate(zip(cache.layers, model.layers)):
        if hasattr(model_layer, "linear_attn"):
            if not hasattr(cache_layer, "conv_states") or not hasattr(cache_layer, "recurrent_states"):
                raise RuntimeError(f"GDN cache layout unavailable at layer {layer_idx}")
            if profile.conv_state_bf16:
                for state in cache_layer.conv_states.values():
                    # The HF cache also stores integer position/history
                    # metadata. Only floating histories have BF16 storage.
                    if state is not None and state.is_floating_point():
                        state.copy_(round_to_bf16(state))
            if profile.ssm_state_bf16:
                for state in cache_layer.recurrent_states.values():
                    if state is not None:
                        state.copy_(round_to_bf16(state))
        else:
            # DynamicCache's attention layers expose keys/values.  Explicitly
            # reject an unfamiliar cache instead of silently omitting KV casts.
            if not hasattr(cache_layer, "keys") or not hasattr(cache_layer, "values"):
                raise RuntimeError(f"QSA cache layout unavailable at layer {layer_idx}")
            if profile.attention_kv_bf16:
                for name in ("keys", "values"):
                    state = getattr(cache_layer, name)
                    if state is not None:
                        state.copy_(round_to_bf16(state))
        if model_layer.ple is not None and not hasattr(model_layer, "linear_attn"):
            if not hasattr(cache_layer, "conv_states"):
                raise RuntimeError(f"PLE cache layout unavailable at layer {layer_idx}")
            if profile.conv_state_bf16:
                for state in cache_layer.conv_states.values():
                    if state is not None and state.is_floating_point():
                        state.copy_(round_to_bf16(state))
