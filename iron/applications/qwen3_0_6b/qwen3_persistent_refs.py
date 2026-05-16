#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F

from iron.applications.qwen3_0_6b.qwen3_cpu import (
    Qwen3ForCausalLM,
    apply_rope,
    repeat_kv,
    rms_norm,
)
from iron.applications.qwen3_0_6b.qwen3_decode_reference import Qwen3CachedReference
from iron.applications.qwen3_0_6b.qwen3_megakernel_debug import (
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


def build_reference_qkv_rope_cache(
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
    initial_cache = torch.cat(
        [state.keys[0].flatten(), state.values[0].flatten()]
    ).contiguous()
    return (
        next_token,
        state.position,
        {
            "hidden": hidden.contiguous(),
            "input_norm_weight": model.w("model.layers.0.input_layernorm.weight")
            .flatten()
            .contiguous(),
            "W_q": model.w(f"{attn}.q_proj.weight").contiguous(),
            "W_k": model.w(f"{attn}.k_proj.weight").contiguous(),
            "W_v": model.w(f"{attn}.v_proj.weight").contiguous(),
            "W_o": model.w(f"{attn}.o_proj.weight").contiguous(),
            "W_q_norm": model.w(f"{attn}.q_norm.weight").flatten().contiguous(),
            "W_k_norm": model.w(f"{attn}.k_norm.weight").flatten().contiguous(),
            "rope_angles": rope_lut_for_position(
                model.config.head_dim, model.config.rope_theta, state.position
            ),
            "initial_cache": initial_cache,
            "initial_keys_cache": state.keys[0].contiguous(),
            "initial_values_cache": state.values[0].contiguous(),
        },
        {
            "x_norm": references["x_norm"].flatten().contiguous(),
            "queries_raw": references["queries_raw"].flatten().contiguous(),
            "keys_raw": references["keys_raw"].flatten().contiguous(),
            "values": references["values"].flatten().contiguous(),
            "queries_norm": references["queries_norm"].flatten().contiguous(),
            "keys_norm": references["keys_norm"].flatten().contiguous(),
            "queries": references["queries"].flatten().contiguous(),
            "keys": references["keys"].flatten().contiguous(),
            "attn_context": references["attn_context"].flatten().contiguous(),
            "attn_out": references["attn_out"].flatten().contiguous(),
            "attn_residual": references["attn_residual"].flatten().contiguous(),
        },
    )


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


def build_qk_pair_reference(
    queries: torch.Tensor,
    current_keys: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    q_by_head = queries.view(q_heads, head_dim)
    k_by_head = current_keys.view(kv_heads, head_dim)
    q_per_kv = q_heads // kv_heads
    if q_per_kv != 2:
        raise ValueError(f"expected q_per_kv=2 for qk_pair debug, got {q_per_kv}")
    pairs = torch.empty((kv_heads, 3, head_dim), dtype=queries.dtype)
    for kv_head in range(kv_heads):
        pairs[kv_head, 0, :] = q_by_head[kv_head * q_per_kv]
        pairs[kv_head, 1, :] = q_by_head[kv_head * q_per_kv + 1]
        pairs[kv_head, 2, :] = k_by_head[kv_head]
    return pairs.flatten().contiguous()


def print_structured_attention_error(
    name: str,
    errors: list[int],
    output: torch.Tensor,
    expected: torch.Tensor,
    op,
) -> None:
    if not errors:
        return
    first = int(errors[0])
    out_flat = output.flatten().to(torch.float32)
    exp_flat = expected.flatten().to(torch.float32)
    if name == "qk_pair":
        slot_names = ("q0", "q1", "current_k")
        elems_per_pair = 3 * op.head_dim
        kv_head = first // elems_per_pair
        rem = first % elems_per_pair
        slot = rem // op.head_dim
        dim = rem % op.head_dim
        print(
            "qk_pair_first_error: "
            f"kv_head={kv_head} slot={slot_names[slot]} dim={dim} "
            f"expected={float(exp_flat[first]):.6f} got={float(out_flat[first]):.6f}"
        )
    elif name in {"attn_scores", "attn_weights"}:
        q_head = first // op.max_seq_len
        pos = first % op.max_seq_len
        region = "valid" if pos <= op.position else "future"
        heads = sorted({int(idx) // op.max_seq_len for idx in errors})
        head_counts = {
            head: sum(1 for idx in errors if int(idx) // op.max_seq_len == head)
            for head in heads
        }
        counts = ", ".join(f"h{head}:{count}" for head, count in head_counts.items())
        print(
            f"{name}_first_error: "
            f"q_head={q_head} pos={pos} region={region} "
            f"expected={float(exp_flat[first]):.6f} got={float(out_flat[first]):.6f}"
        )
        print(f"{name}_error_heads: {counts}")
        if name == "attn_scores":
            got = out_flat[first]
            matches = (exp_flat == got).nonzero(as_tuple=False).flatten().tolist()
            formatted = []
            for idx in matches[:8]:
                match_q = int(idx) // op.max_seq_len
                match_pos = int(idx) % op.max_seq_len
                formatted.append(f"h{match_q}:p{match_pos}")
            if formatted:
                print(
                    f"attn_scores_first_got_matches_expected_at: {', '.join(formatted)}"
                )
            else:
                print("attn_scores_first_got_matches_expected_at: none")
    elif name in {"attn_context", "attn_context_flat"}:
        q_head = first // op.head_dim
        dim = first % op.head_dim
        print(
            f"{name}_first_error: "
            f"q_head={q_head} dim={dim} "
            f"expected={float(exp_flat[first]):.6f} got={float(out_flat[first]):.6f}"
        )
        got = out_flat[first]
        matches = (exp_flat == got).nonzero(as_tuple=False).flatten().tolist()
        formatted = []
        for idx in matches[:8]:
            match_q = int(idx) // op.head_dim
            match_dim = int(idx) % op.head_dim
            formatted.append(f"h{match_q}:d{match_dim}")
        if formatted:
            print(f"{name}_first_got_matches_expected_at: {', '.join(formatted)}")
        else:
            print(f"{name}_first_got_matches_expected_at: none")
    elif name in {"attn_o_proj", "attn_residual"}:
        dim = first % op.hidden_size
        print(
            f"{name}_first_error: "
            f"dim={dim} expected={float(exp_flat[first]):.6f} "
            f"got={float(out_flat[first]):.6f}"
        )
    elif name in {
        "mlp_x_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_gate_silu",
        "ffn_hidden",
        "ffn_out",
        "layer_residual",
    }:
        dim = first % output.numel()
        print(
            f"{name}_first_error: "
            f"dim={dim} expected={float(exp_flat[first]):.6f} "
            f"got={float(out_flat[first]):.6f}"
        )
