"""Explicit CPU storage profile for the Phase 11 Flash-Next decode experiment.

This is a *supplementary implementation-profile reference*, not the independent
FP32 mathematical oracle in run_oracle.py.  It models observable BF16 tensors
and persistent-state casts while using CPU FP32 arithmetic between boundaries.
It must be calibrated against the real V100 stages before its output can be
used to attribute a qualification failure to precision alone.
"""

from dataclasses import dataclass
from types import MethodType


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
    if self.block_inject_weight is None:
        return round_to_bf16(mixed)
    injection = 2 * torch.sigmoid(
        self.block_inject_weight(normalized) / self.hc_count
    )
    return round_to_bf16(mixed), hyper_input, injection


def install_projection_boundaries(model, profile):
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
    for layer in model.layers:
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
            projections.extend((layer.ple.key_proj, layer.ple.value_proj))
            handles.append(layer.ple.register_forward_hook(
                lambda _module, _args, output: round_to_bf16(output)
            ))
        for hyper in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
            handles.append(hyper.register_forward_hook(
                lambda _module, _args, output:
                    (round_to_bf16(output[0]), output[1], output[2])
            ))
        handles.append(layer.mlp.register_forward_hook(
            lambda _module, _args, output: round_to_bf16(output)
        ))
        projections.extend((layer.mlp.shared_expert.gate_proj,
                            layer.mlp.shared_expert.up_proj,
                            layer.mlp.shared_expert.down_proj))
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
