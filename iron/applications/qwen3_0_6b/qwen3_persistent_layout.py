#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from iron.applications.qwen3_0_6b.qwen3_cpu import Qwen3ForCausalLM
from iron.applications.qwen3_0_6b.qwen3_persistent_refs import rope_lut_for_position


def host_owned_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().contiguous()


def pack_layer_cache(state, layer_idx: int) -> torch.Tensor:
    return torch.cat(
        [state.keys[layer_idx].flatten(), state.values[layer_idx].flatten()]
    ).contiguous()


def build_full_layer_weight_inputs_for_layer(
    model: Qwen3ForCausalLM,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    layer = f"model.layers.{layer_idx}"
    attn = f"{layer}.self_attn"
    mlp = f"{layer}.mlp"
    return {
        "input_norm_weight": model.w(f"{layer}.input_layernorm.weight")
        .flatten()
        .contiguous(),
        "W_q": model.w(f"{attn}.q_proj.weight").contiguous(),
        "W_k": model.w(f"{attn}.k_proj.weight").contiguous(),
        "W_v": model.w(f"{attn}.v_proj.weight").contiguous(),
        "W_o": model.w(f"{attn}.o_proj.weight").contiguous(),
        "W_q_norm": model.w(f"{attn}.q_norm.weight").flatten().contiguous(),
        "W_k_norm": model.w(f"{attn}.k_norm.weight").flatten().contiguous(),
        "post_norm_weight": model.w(f"{layer}.post_attention_layernorm.weight")
        .flatten()
        .contiguous(),
        "W_gate": model.w(f"{mlp}.gate_proj.weight").contiguous(),
        "W_up": model.w(f"{mlp}.up_proj.weight").contiguous(),
        "W_down": model.w(f"{mlp}.down_proj.weight").contiguous(),
    }


def build_full_layer_inputs_for_layer(
    model: Qwen3ForCausalLM,
    layer_idx: int,
    hidden: torch.Tensor,
    state,
) -> dict[str, torch.Tensor]:
    return {
        "hidden": hidden.flatten().contiguous(),
        **build_full_layer_weight_inputs_for_layer(model, layer_idx),
        "rope_angles": rope_lut_for_position(
            model.config.head_dim,
            model.config.rope_theta,
            state.position,
        ),
        "initial_cache": pack_layer_cache(state, layer_idx),
        "initial_keys_cache": state.keys[layer_idx].contiguous(),
        "initial_values_cache": state.values[layer_idx].contiguous(),
    }


def pack_full_layer_weights(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            inputs["input_norm_weight"].flatten(),
            inputs["W_q"].flatten(),
            inputs["W_k"].flatten(),
            inputs["W_v"].flatten(),
            inputs["W_q_norm"].flatten(),
            inputs["W_k_norm"].flatten(),
            inputs["W_o"].flatten(),
            inputs["post_norm_weight"].flatten(),
            inputs["W_gate"].flatten(),
            inputs["W_up"].flatten(),
            inputs["W_down"].flatten(),
        ]
    ).contiguous()


def pack_full_layer_weights_for_layer(
    model: Qwen3ForCausalLM,
    layer_idx: int,
) -> torch.Tensor:
    return pack_full_layer_weights(
        build_full_layer_weight_inputs_for_layer(model, layer_idx)
    )


def unpack_full_layer_outputs(
    op, packed_outputs: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {
        "v_context_stream": packed_outputs[
            op.v_context_stream_output_base : op.attn_context_output_base
        ],
        "attn_context": packed_outputs[
            op.attn_context_output_base : op.attn_context_flat_output_base
        ],
        "attn_residual": packed_outputs[
            op.attn_residual_output_base : op.mlp_x_norm_output_base
        ],
        "ffn_hidden": packed_outputs[
            op.ffn_hidden_output_base : op.ffn_out_output_base
        ],
        "ffn_out": packed_outputs[
            op.ffn_out_output_base : op.layer_residual_output_base
        ],
        "layer_residual": packed_outputs[op.layer_residual_output_base :],
    }


def layer_residual_from_packed_output(
    op,
    packed_outputs: torch.Tensor,
) -> torch.Tensor:
    return packed_outputs[op.layer_residual_output_base :]
