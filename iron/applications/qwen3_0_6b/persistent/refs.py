#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from iron.applications.qwen3_0_6b.qwen3_cpu import (
    Qwen3ForCausalLM,
    rms_norm,
)
from iron.applications.qwen3_0_6b.qwen3_decode_reference import Qwen3CachedReference
from iron.applications.qwen3_0_6b.qwen3_decode_reference import clone_decode_state
from iron.applications.qwen3_0_6b.full_elf.debug import (
    one_layer_reference_tensors,
)


def build_reference_input(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, _ = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    hidden = model.embed(torch.tensor([[next_token]], dtype=torch.long)).flatten()
    weight = model.w("model.layers.0.input_layernorm.weight").flatten()
    expected = rms_norm(
        hidden.view(1, 1, -1),
        weight,
        model.config.rms_norm_eps,
    ).flatten()
    return next_token, hidden.contiguous(), weight.contiguous(), expected.contiguous()


def build_reference_qkv(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
):
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    hidden = model.embed(torch.tensor([[next_token]], dtype=torch.long)).flatten()
    references = one_layer_reference_tensors(model, next_token, state, max_seq_len)
    attn = "model.layers.0.self_attn"
    return (
        next_token,
        {
            "hidden": hidden.contiguous(),
            "input_norm_weight": model.w("model.layers.0.input_layernorm.weight")
            .flatten()
            .contiguous(),
            "W_q": model.w(f"{attn}.q_proj.weight").contiguous(),
            "W_k": model.w(f"{attn}.k_proj.weight").contiguous(),
            "W_v": model.w(f"{attn}.v_proj.weight").contiguous(),
        },
        {
            "x_norm": references["x_norm"].flatten().contiguous(),
            "queries_raw": references["queries_raw"].flatten().contiguous(),
            "keys_raw": references["keys_raw"].flatten().contiguous(),
            "values": references["values"].flatten().contiguous(),
        },
    )


def rope_lut_for_position(
    head_dim: int, rope_theta: float, position: int
) -> torch.Tensor:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = position * inv_freq
    lut = torch.empty(head_dim, dtype=torch.bfloat16)
    lut[::2] = freqs.cos().to(torch.bfloat16)
    lut[1::2] = freqs.sin().to(torch.bfloat16)
    return lut.contiguous()


def build_reference_mlp_gate_up(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
):
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    references = one_layer_reference_tensors(model, next_token, state, max_seq_len)
    layer = "model.layers.0"
    mlp = f"{layer}.mlp"
    attn_residual = references["attn_residual"].flatten().contiguous()
    post_norm_weight = model.w(f"{layer}.post_attention_layernorm.weight").flatten()
    mlp_x_norm = rms_norm(
        attn_residual.view(1, 1, -1),
        post_norm_weight,
        model.config.rms_norm_eps,
    ).flatten()
    w_gate = model.w(f"{mlp}.gate_proj.weight").contiguous()
    w_up = model.w(f"{mlp}.up_proj.weight").contiguous()
    ffn_gate = F.linear(mlp_x_norm.view(1, 1, -1), w_gate).flatten()
    ffn_up = F.linear(mlp_x_norm.view(1, 1, -1), w_up).flatten()
    ffn_gate_silu = F.silu(ffn_gate)
    ffn_hidden = ffn_gate_silu * ffn_up
    return (
        next_token,
        {
            "attn_residual": attn_residual.contiguous(),
            "post_norm_weight": post_norm_weight.contiguous(),
            "W_gate": w_gate,
            "W_up": w_up,
        },
        {
            "mlp_x_norm": mlp_x_norm.contiguous(),
            "ffn_gate": ffn_gate.contiguous(),
            "ffn_up": ffn_up.contiguous(),
            "ffn_gate_silu": ffn_gate_silu.contiguous(),
            "ffn_hidden": ffn_hidden.contiguous(),
        },
    )


def build_reference_mlp_down_residual(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
):
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    references = one_layer_reference_tensors(model, next_token, state, max_seq_len)
    mlp = "model.layers.0.mlp"
    ffn_hidden = references["ffn_hidden"].flatten().contiguous()
    attn_residual = references["attn_residual"].flatten().contiguous()
    w_down = model.w(f"{mlp}.down_proj.weight").contiguous()
    ffn_out = F.linear(ffn_hidden.view(1, 1, -1), w_down).flatten()
    layer_residual = attn_residual + ffn_out
    return (
        next_token,
        {
            "ffn_hidden": ffn_hidden,
            "attn_residual": attn_residual,
            "W_down": w_down,
        },
        {
            "ffn_out": ffn_out.contiguous(),
            "layer_residual": layer_residual.contiguous(),
        },
    )


def build_reference_full_mlp(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
):
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    references = one_layer_reference_tensors(model, next_token, state, max_seq_len)
    layer = "model.layers.0"
    mlp = f"{layer}.mlp"
    attn_residual = references["attn_residual"].flatten().contiguous()
    post_norm_weight = model.w(f"{layer}.post_attention_layernorm.weight").flatten()
    w_gate = model.w(f"{mlp}.gate_proj.weight").contiguous()
    w_up = model.w(f"{mlp}.up_proj.weight").contiguous()
    w_down = model.w(f"{mlp}.down_proj.weight").contiguous()
    mlp_x_norm = rms_norm(
        attn_residual.view(1, 1, -1),
        post_norm_weight,
        model.config.rms_norm_eps,
    ).flatten()
    ffn_gate = F.linear(mlp_x_norm.view(1, 1, -1), w_gate).flatten()
    ffn_up = F.linear(mlp_x_norm.view(1, 1, -1), w_up).flatten()
    ffn_gate_silu = F.silu(ffn_gate)
    ffn_hidden = ffn_gate_silu * ffn_up
    ffn_out = F.linear(ffn_hidden.view(1, 1, -1), w_down).flatten()
    layer_residual = attn_residual + ffn_out
    return (
        next_token,
        {
            "attn_residual": attn_residual,
            "post_norm_weight": post_norm_weight.contiguous(),
            "W_gate": w_gate,
            "W_up": w_up,
            "W_down": w_down,
        },
        {
            "mlp_x_norm": mlp_x_norm.contiguous(),
            "ffn_gate": ffn_gate.contiguous(),
            "ffn_up": ffn_up.contiguous(),
            "ffn_gate_silu": ffn_gate_silu.contiguous(),
            "ffn_hidden": ffn_hidden.contiguous(),
            "ffn_out": ffn_out.contiguous(),
            "layer_residual": layer_residual.contiguous(),
        },
    )


def build_reference_multi_layer_full_layer(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
    num_layers: int,
    prefill_num_layers: int | None = None,
):
    prefill_layers = num_layers if prefill_num_layers is None else prefill_num_layers
    prefill_ref = Qwen3CachedReference(model, max_seq_len, num_layers=prefill_layers)
    decode_ref = Qwen3CachedReference(model, max_seq_len, num_layers=num_layers)
    prefill_logits, state = prefill_ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    hidden = model.embed(torch.tensor([[next_token]], dtype=torch.long)).flatten()
    expected_hidden, expected_state = decode_ref.decode_hidden(
        next_token, clone_decode_state(state)
    )
    return (
        next_token,
        state.position,
        hidden.contiguous(),
        state,
        expected_hidden.flatten().contiguous(),
        expected_state,
    )
