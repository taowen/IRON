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
from iron.applications.qwen3_0_6b.qwen3_decode_reference import Qwen3DecodeState

DEFAULT_DEBUG_OUTPUTS = (
    "x_norm",
    "queries_raw",
    "queries_norm",
    "queries",
    "keys_raw",
    "keys_norm",
    "keys",
    "values",
    "attn_scores",
    "attn_weights",
    "attn_context",
    "attn_out",
    "ffn_out",
    "x",
    "logits",
)

DEBUG_STAGE_OUTPUTS = {
    "qkv": (
        "x_norm",
        "queries_raw",
        "queries_norm",
        "queries",
        "keys_raw",
        "keys_norm",
        "keys",
        "values",
    ),
    "attention": (
        "queries",
        "keys",
        "values",
        "attn_scores",
        "attn_weights",
        "attn_context",
        "attn_out",
    ),
    "all": DEFAULT_DEBUG_OUTPUTS,
}


def one_layer_reference_tensors(
    model: Qwen3ForCausalLM,
    token_id: int,
    state: Qwen3DecodeState,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    cfg = model.config
    layer = "model.layers.0"
    attn = f"{layer}.self_attn"
    mlp = f"{layer}.mlp"
    position = state.position

    x = model.embed(torch.tensor([[token_id]], dtype=torch.long)).to(dtype=model.dtype)
    residual = x
    input_x_norm = rms_norm(
        x,
        model.w(f"{layer}.input_layernorm.weight"),
        cfg.rms_norm_eps,
    )
    q_raw = F.linear(input_x_norm, model.w(f"{attn}.q_proj.weight")).view(
        1, 1, cfg.num_attention_heads, cfg.head_dim
    )
    q_raw = q_raw.transpose(1, 2)
    k_raw = F.linear(input_x_norm, model.w(f"{attn}.k_proj.weight")).view(
        1, 1, cfg.num_key_value_heads, cfg.head_dim
    )
    k_raw = k_raw.transpose(1, 2)
    values = F.linear(input_x_norm, model.w(f"{attn}.v_proj.weight")).view(
        1, 1, cfg.num_key_value_heads, cfg.head_dim
    )
    values = values.transpose(1, 2)

    q_norm = rms_norm(q_raw, model.w(f"{attn}.q_norm.weight"), cfg.rms_norm_eps)
    k_norm = rms_norm(k_raw, model.w(f"{attn}.k_norm.weight"), cfg.rms_norm_eps)
    queries, keys = apply_rope(
        q_norm, k_norm, torch.tensor([position]), cfg.head_dim, cfg.rope_theta
    )

    cache_keys = state.keys[0].clone()
    cache_values = state.values[0].clone()
    cache_keys[:, position : position + 1, :] = keys.squeeze(0)
    cache_values[:, position : position + 1, :] = values.squeeze(0)
    repeats = cfg.num_attention_heads // cfg.num_key_value_heads
    k_ctx = repeat_kv(cache_keys[:, : position + 1, :].unsqueeze(0), repeats)
    v_ctx = repeat_kv(cache_values[:, : position + 1, :].unsqueeze(0), repeats)
    scores = torch.matmul(queries, k_ctx.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
    weights = torch.softmax(scores.to(torch.float32), dim=-1).to(dtype=model.dtype)
    context = torch.matmul(weights, v_ctx)
    context_flat = context.transpose(1, 2).contiguous().view(1, 1, -1)
    attn_out = F.linear(context_flat, model.w(f"{attn}.o_proj.weight"))

    x = residual + attn_out
    residual = x
    mlp_x_norm = rms_norm(
        x,
        model.w(f"{layer}.post_attention_layernorm.weight"),
        cfg.rms_norm_eps,
    )
    gate = F.linear(mlp_x_norm, model.w(f"{mlp}.gate_proj.weight"))
    up = F.linear(mlp_x_norm, model.w(f"{mlp}.up_proj.weight"))
    ffn_hidden = F.silu(gate) * up
    ffn_out = F.linear(ffn_hidden, model.w(f"{mlp}.down_proj.weight"))
    x = residual + ffn_out
    x = rms_norm(x, model.w("model.norm.weight"), cfg.rms_norm_eps)
    logits = F.linear(x, model.w("model.embed_tokens.weight"))

    padded_scores = torch.zeros(cfg.num_attention_heads, max_seq_len, dtype=model.dtype)
    padded_weights = torch.zeros_like(padded_scores)
    padded_scores[:, : position + 1] = scores.squeeze(0).squeeze(1)
    padded_weights[:, : position + 1] = weights.squeeze(0).squeeze(1)

    return {
        "x": x.flatten(),
        "x_norm": input_x_norm.flatten(),
        "queries_raw": q_raw.squeeze(0).squeeze(1),
        "queries_norm": q_norm.squeeze(0).squeeze(1),
        "queries": queries.squeeze(0).squeeze(1),
        "keys_raw": k_raw.squeeze(0).squeeze(1),
        "keys_norm": k_norm.squeeze(0).squeeze(1),
        "keys": keys.squeeze(0).squeeze(1),
        "values": values.squeeze(0).squeeze(1),
        "attn_scores": padded_scores,
        "attn_weights": padded_weights,
        "attn_context": context.squeeze(0).squeeze(1),
        "attn_out": attn_out.flatten(),
        "ffn_gate": gate.flatten(),
        "ffn_up": up.flatten(),
        "ffn_hidden": ffn_hidden.flatten(),
        "ffn_out": ffn_out.flatten(),
        "logits": logits.flatten(),
    }


def local_reference_tensors(
    model: Qwen3ForCausalLM,
    npu_tensors: dict[str, torch.Tensor],
    position: int,
) -> dict[str, torch.Tensor]:
    cfg = model.config
    attn = "model.layers.0.self_attn"
    refs = {}

    if "queries_raw" in npu_tensors:
        refs["queries_norm"] = rms_norm(
            npu_tensors["queries_raw"].view(
                1, cfg.num_attention_heads, 1, cfg.head_dim
            ),
            model.w(f"{attn}.q_norm.weight"),
            cfg.rms_norm_eps,
        ).view(cfg.num_attention_heads, cfg.head_dim)
    if "keys_raw" in npu_tensors:
        refs["keys_norm"] = rms_norm(
            npu_tensors["keys_raw"].view(1, cfg.num_key_value_heads, 1, cfg.head_dim),
            model.w(f"{attn}.k_norm.weight"),
            cfg.rms_norm_eps,
        ).view(cfg.num_key_value_heads, cfg.head_dim)
    if "queries_norm" in npu_tensors and "keys_norm" in npu_tensors:
        q_rope, k_rope = apply_rope(
            npu_tensors["queries_norm"].view(
                1, cfg.num_attention_heads, 1, cfg.head_dim
            ),
            npu_tensors["keys_norm"].view(1, cfg.num_key_value_heads, 1, cfg.head_dim),
            torch.tensor([position]),
            cfg.head_dim,
            cfg.rope_theta,
        )
        refs["queries"] = q_rope.view(cfg.num_attention_heads, cfg.head_dim)
        refs["keys"] = k_rope.view(cfg.num_key_value_heads, cfg.head_dim)
    return refs


def print_tensor_diff(name: str, got: torch.Tensor, ref: torch.Tensor, label: str):
    got = got.detach().to(torch.float32).flatten()
    ref = ref.detach().to(torch.float32).flatten()
    if got.numel() != ref.numel():
        print(
            f"diff {name} {label}: shape_mismatch got={got.numel()} ref={ref.numel()}"
        )
        return
    diff = (got - ref).abs()
    max_i = int(torch.argmax(diff).item()) if diff.numel() else 0
    print(
        f"diff {name} {label}: "
        f"max_abs={float(diff.max()):.6f} "
        f"mean_abs={float(diff.mean()):.6f} "
        f"got_mean={float(got.mean()):.6f} "
        f"ref_mean={float(ref.mean()):.6f} "
        f"max_i={max_i} got={float(got[max_i]):.6f} ref={float(ref[max_i]):.6f}"
    )


def parse_debug_outputs(debug_stage: str | None, debug_outputs: str | None):
    outputs = []
    if debug_stage:
        outputs.extend(DEBUG_STAGE_OUTPUTS[debug_stage])
    if debug_outputs:
        outputs.extend(
            output.strip() for output in debug_outputs.split(",") if output.strip()
        )
    return tuple(dict.fromkeys(outputs))
