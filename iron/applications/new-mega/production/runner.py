#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from ml_dtypes import bfloat16
import torch.nn.functional as F
from transformers import AutoTokenizer

from iron.applications.qwen3_0_6b.qwen3_cpu import (
    Qwen3ForCausalLM,
    apply_rope,
    encode_prompt,
    repeat_kv,
    resolve_model_dir,
    rms_norm,
)
from iron.applications.qwen3_0_6b.qwen3_decode_reference import Qwen3CachedReference
from iron.applications.qwen3_0_6b.qwen3_preflight import (
    PersistentPreflightResult,
    run_persistent_artifact_preflight,
)
from iron.common.context import AIEContext

from ops import (
    ATTENTION_SCORE_PV_PHASES,
    ATTENTION_SCORE_PV_SECONDARY_PHASES,
    PHASE_LABELS,
    NewMegaPhaseOwnedDecode,
)

K_PHASE = PHASE_LABELS.index("attention_chunk_0")
V_PHASE = PHASE_LABELS.index("attention_chunk_1")
Q_ROPE_PHASE = PHASE_LABELS.index("attention_chunk_2")
K_ROPE_PHASE = PHASE_LABELS.index("attention_chunk_3")
ATTENTION_SCORE_PV_PHASE_INDICES = [
    PHASE_LABELS.index(label) for label in ATTENTION_SCORE_PV_PHASES
]
ATTENTION_SCORE_PV_SECONDARY_PHASE_INDICES = [
    PHASE_LABELS.index(label) for label in ATTENTION_SCORE_PV_SECONDARY_PHASES
]
O_PHASE = PHASE_LABELS.index("o_proj")
GATE_UP_PHASE = PHASE_LABELS.index("gate_up")
DOWN_PHASE = PHASE_LABELS.index("down_proj")


def _kernel_normed_row_dot(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    row_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Scalar reference for the current BF16 AIE row-shard kernels."""

    x_f32 = x.flatten().to(torch.float32)
    weight_f32 = norm_weight.flatten().to(torch.float32)
    row_f32 = row_weight.flatten().to(torch.float32)
    mean_square = (x_f32 * x_f32).mean(dtype=torch.float32)
    inv_rms = torch.rsqrt(mean_square + eps)
    acc = torch.zeros((), dtype=torch.float32)
    for i in range(x_f32.numel()):
        xnorm = x_f32[i] * inv_rms * weight_f32[i]
        acc += xnorm * row_f32[i]
    return acc.to(torch.bfloat16)


def _kernel_down_residual_row(
    ffn_hidden: torch.Tensor,
    down_weight_row: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    ffn_f32 = ffn_hidden.flatten().to(torch.float32)
    weight_f32 = down_weight_row.flatten().to(torch.float32)
    acc = torch.zeros((), dtype=torch.float32)
    for i in range(ffn_f32.numel()):
        acc += ffn_f32[i] * weight_f32[i]
    acc += residual.to(torch.float32)
    return acc.to(torch.bfloat16)


def _rope_cos_sin(
    *,
    position: int,
    head_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(torch.tensor([position], dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).flatten()
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _kernel_norm_rope_row(
    raw_head: torch.Tensor,
    norm_weight: torch.Tensor,
    cos_values: torch.Tensor,
    sin_values: torch.Tensor,
    row_index: int,
) -> torch.Tensor:
    raw_f32 = raw_head.flatten().to(torch.float32)
    weight_f32 = norm_weight.flatten().to(torch.float32)
    cos_f32 = cos_values.flatten().to(torch.float32)
    sin_f32 = sin_values.flatten().to(torch.float32)
    head_dim = raw_f32.numel()
    half_dim = head_dim // 2
    mean_square = (raw_f32 * raw_f32).mean(dtype=torch.float32)
    inv_rms = torch.rsqrt(mean_square + 1.0e-6)
    pair_index = row_index + half_dim if row_index < half_dim else row_index - half_dim
    x = raw_f32[row_index] * inv_rms * weight_f32[row_index]
    pair_x = raw_f32[pair_index] * inv_rms * weight_f32[pair_index]
    rotated = -pair_x if row_index < half_dim else pair_x
    out = x * cos_f32[row_index] + rotated * sin_f32[row_index]
    return out.to(torch.bfloat16)


def _kernel_norm_rope_head(
    raw_head: torch.Tensor,
    norm_weight: torch.Tensor,
    cos_values: torch.Tensor,
    sin_values: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            _kernel_norm_rope_row(raw_head, norm_weight, cos_values, sin_values, idx)
            for idx in range(raw_head.numel())
        ]
    ).to(torch.bfloat16)


def _make_decode_mask(max_seq_len: int, position: int) -> torch.Tensor:
    mask = torch.zeros((max_seq_len,), dtype=torch.bfloat16)
    mask[: position + 1] = 1
    return mask


def _attention_context_head(
    q_head: torch.Tensor,
    k_cache_head: torch.Tensor,
    v_cache_head: torch.Tensor,
    mask: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    q = q_head.flatten().to(torch.float32)
    k = k_cache_head.to(torch.float32)
    v = v_cache_head.to(torch.float32)
    valid = mask.to(torch.float32) > 0.5
    scores = torch.matmul(k, q) / math.sqrt(head_dim)
    scores = scores.masked_fill(~valid, float("-inf"))
    weights = torch.softmax(scores, dim=0).to(torch.float32)
    return torch.matmul(weights.unsqueeze(0), v).flatten().to(torch.bfloat16)


def _pack_attention_chunk_packet(
    lane_packets: torch.Tensor,
    start: int,
    *,
    q_head: torch.Tensor,
    k_cache_head: torch.Tensor,
    v_cache_head: torch.Tensor,
    mask: torch.Tensor,
    chunk_start: int,
    chunk_size: int,
    head_dim: int,
) -> None:
    k_start = start + head_dim
    v_start = k_start + chunk_size * head_dim
    mask_start = v_start + chunk_size * head_dim
    chunk_end = chunk_start + chunk_size
    lane_packets[start : start + head_dim] = q_head
    lane_packets[k_start:v_start] = k_cache_head[chunk_start:chunk_end].flatten()
    lane_packets[v_start:mask_start] = v_cache_head[chunk_start:chunk_end].flatten()
    lane_packets[mask_start : mask_start + chunk_size] = mask[chunk_start:chunk_end]


def _attention_decode_context_and_output(
    model: Qwen3ForCausalLM,
    x: torch.Tensor,
    layer_idx: int,
    state,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = model.config
    prefix = f"model.layers.{layer_idx}.self_attn"
    batch, seq_len, _ = x.shape
    if batch != 1 or seq_len != 1:
        raise ValueError("decode attention expects batch=1 and seq_len=1")

    q = F.linear(x, model.w(f"{prefix}.q_proj.weight"))
    k = F.linear(x, model.w(f"{prefix}.k_proj.weight"))
    v = F.linear(x, model.w(f"{prefix}.v_proj.weight"))

    q = q.view(batch, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)

    q = rms_norm(q, model.w(f"{prefix}.q_norm.weight"), cfg.rms_norm_eps)
    k = rms_norm(k, model.w(f"{prefix}.k_norm.weight"), cfg.rms_norm_eps)
    position_ids = torch.tensor([state.position])
    q, k = apply_rope(q, k, position_ids, cfg.head_dim, cfg.rope_theta)

    state.keys[layer_idx][:, state.position : state.position + 1, :] = k.squeeze(0)
    state.values[layer_idx][:, state.position : state.position + 1, :] = v.squeeze(0)

    k_ctx = state.keys[layer_idx][:, : state.position + 1, :].unsqueeze(0)
    v_ctx = state.values[layer_idx][:, : state.position + 1, :].unsqueeze(0)
    repeats = cfg.num_attention_heads // cfg.num_key_value_heads
    k_ctx = repeat_kv(k_ctx, repeats)
    v_ctx = repeat_kv(v_ctx, repeats)
    scores = torch.matmul(q, k_ctx.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
    probs = torch.softmax(scores.to(torch.float32), dim=-1).to(dtype=x.dtype)
    context = torch.matmul(probs, v_ctx)
    context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    output = F.linear(context, model.w(f"{prefix}.o_proj.weight"))
    return context.flatten().contiguous(), output.flatten().contiguous()


@dataclass(frozen=True)
class PhaseOwnedCase:
    shared_packets: torch.Tensor
    lane_packets: torch.Tensor
    expected: torch.Tensor
    qwen3_reference: torch.Tensor
    model_dir: Path
    prompt_tokens: int
    next_token: int


@dataclass(frozen=True)
class PhaseOwnedRunResult:
    op: NewMegaPhaseOwnedDecode
    preflight: PersistentPreflightResult
    case: PhaseOwnedCase | None
    npu_time_us: float | None
    max_abs: float | None
    mean_abs: float | None
    errors: int | None
    qwen3_max_abs: float | None
    qwen3_mean_abs: float | None
    qwen3_errors: int | None


def load_model_and_prompt(
    *,
    model_name: str,
    revision: str | None,
    prompt: str,
    raw_prompt: bool,
    enable_thinking: bool,
) -> tuple[Path, Qwen3ForCausalLM, torch.Tensor]:
    model_dir = resolve_model_dir(model_name, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = encode_prompt(
        tokenizer,
        prompt,
        raw_prompt=raw_prompt,
        enable_thinking=enable_thinking,
    )
    model = Qwen3ForCausalLM(model_dir)
    return model_dir, model, input_ids


def phase_owned_reference(
    shared_packets: torch.Tensor,
    lane_packets: torch.Tensor,
    op: NewMegaPhaseOwnedDecode,
) -> torch.Tensor:
    expected = torch.zeros((op.output_elements,), dtype=torch.float32)
    shared_f32 = shared_packets.to(torch.float32)
    lane_f32 = lane_packets.to(torch.float32)
    lane_span = op.total_phase_packets * op.packet_elements
    q_stride = op.q_output_values_per_lane
    k_stride = op.k_output_values_per_lane
    v_stride = op.v_output_values_per_lane
    q_rope_stride = op.q_rope_output_values_per_lane
    k_rope_stride = op.k_rope_output_values_per_lane
    context_stride = op.context_output_values_per_lane
    attention_stride = op.attention_output_values_per_lane
    gate_up_stride = op.gate_up_output_values_per_lane
    out_lane_stride = op.output_values_per_lane
    out_layer_span = op.output_values_per_layer
    for lane in range(op.num_lanes):
        acc = torch.zeros((), dtype=torch.float32)
        hidden_state = torch.zeros((op.hidden_size,), dtype=torch.float32)
        lane_start = lane * lane_span
        for layer in range(op.num_layers):
            shared_start = layer * op.shared_packet_elements
            shared = shared_f32[shared_start : shared_start + op.shared_packet_elements]
            if layer == 0:
                hidden_state = shared[: op.hidden_size].clone()
            weight = shared[op.hidden_size : 2 * op.hidden_size]
            mean_square = (hidden_state * hidden_state).mean(dtype=torch.float32)
            inv_rms = torch.rsqrt(mean_square + 1.0e-6)
            q_packet_index = layer * op.phase_packets_per_layer
            q_packet_start = lane_start + q_packet_index * op.packet_elements
            q_packet = lane_f32[q_packet_start : q_packet_start + op.packet_elements]
            output_start = layer * out_layer_span + lane * out_lane_stride
            for row in range(op.q_rows_per_packet):
                q_row_start = row * op.hidden_size
                q_acc = torch.zeros((), dtype=torch.float32)
                for i in range(op.hidden_size):
                    xnorm = hidden_state[i] * inv_rms * weight[i]
                    if row == 0:
                        acc += xnorm
                    q_acc += xnorm * q_packet[q_row_start + i]
                expected[output_start + row] = q_acc
                acc += q_acc

            k_packet_index = layer * op.phase_packets_per_layer + K_PHASE
            k_start = lane_start + k_packet_index * op.packet_elements
            k_packet = lane_f32[k_start : k_start + op.packet_elements]
            k_base = output_start + q_stride
            for row in range(op.q_rows_per_packet):
                row_start = row * op.hidden_size
                k_acc = torch.zeros((), dtype=torch.float32)
                for i in range(op.hidden_size):
                    xnorm = hidden_state[i] * inv_rms * weight[i]
                    k_acc += xnorm * k_packet[row_start + i]
                expected[k_base + row] = k_acc
                acc += k_acc

            v_packet_index = layer * op.phase_packets_per_layer + V_PHASE
            v_start = lane_start + v_packet_index * op.packet_elements
            v_packet = lane_f32[v_start : v_start + op.packet_elements]
            v_base = output_start + q_stride + k_stride
            for row in range(op.q_rows_per_packet):
                row_start = row * op.hidden_size
                v_acc = torch.zeros((), dtype=torch.float32)
                for i in range(op.hidden_size):
                    xnorm = hidden_state[i] * inv_rms * weight[i]
                    v_acc += xnorm * v_packet[row_start + i]
                expected[v_base + row] = v_acc
                acc += v_acc

            q_rope_packet_index = layer * op.phase_packets_per_layer + Q_ROPE_PHASE
            q_rope_start = lane_start + q_rope_packet_index * op.packet_elements
            q_rope_packet = lane_f32[q_rope_start : q_rope_start + op.packet_elements]
            q_rope_row_base = int(float(q_rope_packet[0].item()))
            q_raw_head = q_rope_packet[1 : 1 + op.head_dim]
            q_norm_weight = q_rope_packet[1 + op.head_dim : 1 + 2 * op.head_dim]
            q_cos = q_rope_packet[1 + 2 * op.head_dim : 1 + 3 * op.head_dim]
            q_sin = q_rope_packet[1 + 3 * op.head_dim : 1 + 4 * op.head_dim]
            q_rope_base = output_start + q_stride + k_stride + v_stride
            for row in range(op.q_rows_per_packet):
                row_index = q_rope_row_base + row
                q_rope_acc = _kernel_norm_rope_row(
                    q_raw_head,
                    q_norm_weight,
                    q_cos,
                    q_sin,
                    row_index,
                ).to(torch.float32)
                expected[q_rope_base + row] = q_rope_acc
                acc += q_rope_acc

            k_rope_packet_index = layer * op.phase_packets_per_layer + K_ROPE_PHASE
            k_rope_start = lane_start + k_rope_packet_index * op.packet_elements
            k_rope_packet = lane_f32[k_rope_start : k_rope_start + op.packet_elements]
            k_rope_row_base = int(float(k_rope_packet[0].item()))
            k_raw_head = k_rope_packet[1 : 1 + op.head_dim]
            k_norm_weight = k_rope_packet[1 + op.head_dim : 1 + 2 * op.head_dim]
            k_cos = k_rope_packet[1 + 2 * op.head_dim : 1 + 3 * op.head_dim]
            k_sin = k_rope_packet[1 + 3 * op.head_dim : 1 + 4 * op.head_dim]
            k_rope_base = q_rope_base + q_rope_stride
            for row in range(op.q_rows_per_packet):
                row_index = k_rope_row_base + row
                k_rope_acc = _kernel_norm_rope_row(
                    k_raw_head,
                    k_norm_weight,
                    k_cos,
                    k_sin,
                    row_index,
                ).to(torch.float32)
                expected[k_rope_base + row] = k_rope_acc
                acc += k_rope_acc

            context_base = (
                output_start
                + q_stride
                + k_stride
                + v_stride
                + q_rope_stride
                + k_rope_stride
            )
            context_heads = []
            for slot, phase_group in enumerate(
                (
                    ATTENTION_SCORE_PV_PHASE_INDICES,
                    ATTENTION_SCORE_PV_SECONDARY_PHASE_INDICES,
                )
            ):
                q_head = None
                k_chunks = []
                v_chunks = []
                mask_chunks = []
                for phase in phase_group:
                    packet_index = layer * op.phase_packets_per_layer + phase
                    start = lane_start + packet_index * op.packet_elements
                    packet = lane_f32[start : start + op.packet_elements]
                    if q_head is None:
                        q_head = packet[: op.head_dim].clone()
                    k_start = op.head_dim
                    v_start = k_start + op.attention_chunk_size * op.head_dim
                    mask_start = v_start + op.attention_chunk_size * op.head_dim
                    k_chunks.append(
                        packet[k_start:v_start].view(
                            op.attention_chunk_size, op.head_dim
                        )
                    )
                    v_chunks.append(
                        packet[v_start:mask_start].view(
                            op.attention_chunk_size, op.head_dim
                        )
                    )
                    mask_chunks.append(
                        packet[mask_start : mask_start + op.attention_chunk_size]
                    )
                if q_head is None:
                    raise ValueError("attention score/PV phases are required")
                context_head = _attention_context_head(
                    q_head,
                    torch.cat(k_chunks, dim=0),
                    torch.cat(v_chunks, dim=0),
                    torch.cat(mask_chunks, dim=0),
                    op.head_dim,
                ).to(torch.float32)
                context_heads.append(context_head)
                slot_start = context_base + slot * op.head_dim
                expected[slot_start : slot_start + op.head_dim] = context_head
                acc += context_head.sum(dtype=torch.float32)

            o_packet_index = layer * op.phase_packets_per_layer + O_PHASE
            o_start = lane_start + o_packet_index * op.packet_elements
            o_packet = lane_f32[o_start : o_start + op.packet_elements]
            context_head_indices = [
                int(float(o_packet[0].item())),
                int(float(o_packet[1].item())),
            ]
            attention_context = o_packet[2 : 2 + op.attention_size].clone()
            for slot, context_head_index in enumerate(context_head_indices):
                context_head_start = context_head_index * op.head_dim
                context_head_end = context_head_start + op.head_dim
                attention_context[context_head_start:context_head_end] = context_heads[
                    slot
                ]
            attention_residual_shard = o_packet[
                2 + op.attention_size : 2 + op.attention_size + op.q_rows_per_packet
            ]
            o_block_start = 2 + op.attention_size + op.q_rows_per_packet
            attention_base = (
                output_start
                + q_stride
                + k_stride
                + v_stride
                + q_rope_stride
                + k_rope_stride
                + context_stride
            )
            for row in range(op.q_rows_per_packet):
                row_start = o_block_start + row * op.attention_size
                o_acc = torch.zeros((), dtype=torch.float32)
                for i in range(op.attention_size):
                    o_acc += attention_context[i] * o_packet[row_start + i]
                residual_acc = attention_residual_shard[row] + o_acc
                expected[attention_base + row] = residual_acc
                acc += residual_acc

            gate_packet_index = layer * op.phase_packets_per_layer + GATE_UP_PHASE
            gate_start = lane_start + gate_packet_index * op.packet_elements
            gate_packet = lane_f32[gate_start : gate_start + op.packet_elements]
            residual_row_base = int(float(gate_packet[0].item()))
            attn_residual = gate_packet[1 : 1 + op.hidden_size].clone()
            attention_base = (
                output_start
                + q_stride
                + k_stride
                + v_stride
                + q_rope_stride
                + k_rope_stride
                + context_stride
            )
            attn_residual[
                residual_row_base : residual_row_base + op.q_rows_per_packet
            ] = expected[attention_base : attention_base + op.q_rows_per_packet]
            post_weight = gate_packet[1 + op.hidden_size : 1 + 2 * op.hidden_size]
            gate_block = gate_packet[
                1 + 2 * op.hidden_size : 1 + (2 + op.q_rows_per_packet) * op.hidden_size
            ]
            up_block = gate_packet[
                1
                + (2 + op.q_rows_per_packet)
                * op.hidden_size : (2 + 2 * op.q_rows_per_packet)
                * op.hidden_size
                + 1
            ]
            mean_square = (attn_residual * attn_residual).mean(dtype=torch.float32)
            inv_rms = torch.rsqrt(mean_square + 1.0e-6)
            gate_base = (
                output_start
                + q_stride
                + k_stride
                + v_stride
                + q_rope_stride
                + k_rope_stride
                + context_stride
                + attention_stride
            )
            up_base = gate_base + op.q_rows_per_packet
            for row in range(op.q_rows_per_packet):
                gate_acc = torch.zeros((), dtype=torch.float32)
                up_acc = torch.zeros((), dtype=torch.float32)
                row_start = row * op.hidden_size
                for i in range(op.hidden_size):
                    xnorm = attn_residual[i] * inv_rms * post_weight[i]
                    gate_acc += xnorm * gate_block[row_start + i]
                    up_acc += xnorm * up_block[row_start + i]
                expected[gate_base + row] = gate_acc
                expected[up_base + row] = up_acc
                acc += gate_acc + up_acc

            down_packet_index = layer * op.phase_packets_per_layer + DOWN_PHASE
            down_start = lane_start + down_packet_index * op.packet_elements
            down_packet = lane_f32[down_start : down_start + op.packet_elements]
            ffn_row_base = int(float(down_packet[0].item()))
            ffn_hidden = down_packet[1 : 1 + op.intermediate_size].clone()
            local_gate = (
                expected[gate_base : gate_base + op.q_rows_per_packet]
                .to(torch.bfloat16)
                .to(torch.float32)
            )
            local_up = (
                expected[up_base : up_base + op.q_rows_per_packet]
                .to(torch.bfloat16)
                .to(torch.float32)
            )
            ffn_hidden[ffn_row_base : ffn_row_base + op.q_rows_per_packet] = (
                F.silu(local_gate) * local_up
            )
            residual_shard = down_packet[
                1
                + op.intermediate_size : 1
                + op.intermediate_size
                + op.q_rows_per_packet
            ]
            down_block_start = 1 + op.intermediate_size + op.q_rows_per_packet
            residual_base = (
                output_start
                + q_stride
                + k_stride
                + v_stride
                + q_rope_stride
                + k_rope_stride
                + context_stride
                + attention_stride
                + gate_up_stride
            )
            for row in range(op.q_rows_per_packet):
                row_start = down_block_start + row * op.intermediate_size
                down_acc = torch.zeros((), dtype=torch.float32)
                for i in range(op.intermediate_size):
                    down_acc += ffn_hidden[i] * down_packet[row_start + i]
                residual_acc = residual_shard[row] + down_acc
                expected[residual_base + row] = residual_acc
                acc += residual_acc

            next_packet_index = (layer + 1) * op.phase_packets_per_layer - 1
            next_start = lane_start + next_packet_index * op.packet_elements
            next_packet = lane_f32[next_start : next_start + op.packet_elements]
            hidden_state = next_packet[: op.hidden_size].clone()
            acc += hidden_state.sum(dtype=torch.float32)
    return expected.to(torch.bfloat16)


@torch.inference_mode()
def build_phase_owned_case(
    op: NewMegaPhaseOwnedDecode,
    *,
    model_dir: Path,
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
) -> PhaseOwnedCase:
    cfg = model.config
    if cfg.hidden_size != op.hidden_size:
        raise ValueError(
            f"operator hidden_size={op.hidden_size} does not match model "
            f"hidden_size={cfg.hidden_size}"
        )
    model_attention_size = cfg.num_attention_heads * cfg.head_dim
    if model_attention_size != op.attention_size:
        raise ValueError(
            f"operator attention_size={op.attention_size} does not match model "
            f"num_attention_heads*head_dim={model_attention_size}"
        )
    if cfg.head_dim != op.head_dim:
        raise ValueError(
            f"operator head_dim={op.head_dim} does not match model head_dim={cfg.head_dim}"
        )
    if cfg.intermediate_size != op.intermediate_size:
        raise ValueError(
            f"operator intermediate_size={op.intermediate_size} does not match "
            f"model intermediate_size={cfg.intermediate_size}"
        )
    if op.num_layers > cfg.num_hidden_layers:
        raise ValueError(
            f"operator num_layers={op.num_layers} exceeds model layers "
            f"{cfg.num_hidden_layers}"
        )
    if max_seq_len != op.max_seq_len:
        raise ValueError(
            f"case max_seq_len={max_seq_len} must match operator max_seq_len={op.max_seq_len}"
        )
    output_rows = op.num_lanes * op.q_rows_per_packet
    if output_rows > cfg.num_attention_heads * cfg.head_dim:
        raise ValueError("Q shard rows exceed Qwen3 q_proj output rows")
    if output_rows > cfg.head_dim:
        raise ValueError("q/k norm+RoPE shard currently covers only head 0")
    if op.num_lanes > cfg.num_attention_heads:
        raise ValueError("context handoff currently maps one attention head per lane")
    kv_rows = cfg.num_key_value_heads * cfg.head_dim
    if output_rows > kv_rows:
        raise ValueError("K/V shard rows exceed Qwen3 key/value projection rows")
    if output_rows > cfg.intermediate_size:
        raise ValueError("gate/up shard rows exceed Qwen3 intermediate_size")

    ref = Qwen3CachedReference(model, max_seq_len, num_layers=op.num_layers)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())

    shared_packets = torch.zeros((op.shared_input_elements,), dtype=torch.bfloat16)
    lane_packets = torch.zeros((op.input_elements,), dtype=torch.bfloat16)
    qwen3_reference = torch.zeros((op.output_elements,), dtype=torch.bfloat16)

    q_stride = op.q_output_values_per_lane
    k_stride = op.k_output_values_per_lane
    v_stride = op.v_output_values_per_lane
    q_rope_stride = op.q_rope_output_values_per_lane
    k_rope_stride = op.k_rope_output_values_per_lane
    context_stride = op.context_output_values_per_lane
    attention_stride = op.attention_output_values_per_lane
    gate_up_stride = op.gate_up_output_values_per_lane
    out_lane_stride = op.output_values_per_lane
    out_layer_span = op.output_values_per_layer
    lane_span = op.total_phase_packets * op.packet_elements
    x = model.embed(torch.tensor([[next_token]], dtype=torch.long)).to(
        dtype=model.dtype
    )

    for layer_idx in range(op.num_layers):
        layer = f"model.layers.{layer_idx}"
        attn = f"{layer}.self_attn"

        hidden = x.flatten().contiguous()
        input_norm_weight = model.w(f"{layer}.input_layernorm.weight").flatten()
        shared_start = layer_idx * op.shared_packet_elements
        if layer_idx == 0:
            shared_packets[shared_start : shared_start + op.hidden_size] = hidden
        shared_packets[
            shared_start + op.hidden_size : shared_start + 2 * op.hidden_size
        ] = input_norm_weight

        x_norm = rms_norm(
            x,
            model.w(f"{layer}.input_layernorm.weight"),
            cfg.rms_norm_eps,
        )
        q_weight = model.w(f"{attn}.q_proj.weight").contiguous()
        k_weight = model.w(f"{attn}.k_proj.weight").contiguous()
        v_weight = model.w(f"{attn}.v_proj.weight").contiguous()
        q_raw_flat = F.linear(x_norm, q_weight).flatten().contiguous()
        k_raw_flat = F.linear(x_norm, k_weight).flatten().contiguous()
        v_raw_flat = F.linear(x_norm, v_weight).flatten().contiguous()
        q_raw_heads = q_raw_flat.view(
            cfg.num_attention_heads, cfg.head_dim
        ).contiguous()
        k_raw_kv_heads = k_raw_flat.view(
            cfg.num_key_value_heads, cfg.head_dim
        ).contiguous()
        v_raw_kv_heads = v_raw_flat.view(
            cfg.num_key_value_heads, cfg.head_dim
        ).contiguous()
        cos_values, sin_values = _rope_cos_sin(
            position=state.position,
            head_dim=cfg.head_dim,
            rope_theta=cfg.rope_theta,
            dtype=model.dtype,
        )
        q_norm_weight = model.w(f"{attn}.q_norm.weight").flatten().contiguous()
        k_norm_weight = model.w(f"{attn}.k_norm.weight").flatten().contiguous()
        q_rope_heads = torch.stack(
            [
                _kernel_norm_rope_head(
                    q_raw_heads[head], q_norm_weight, cos_values, sin_values
                )
                for head in range(cfg.num_attention_heads)
            ]
        ).contiguous()
        k_rope_kv_heads = torch.stack(
            [
                _kernel_norm_rope_head(
                    k_raw_kv_heads[head], k_norm_weight, cos_values, sin_values
                )
                for head in range(cfg.num_key_value_heads)
            ]
        ).contiguous()
        k_cache_kv_heads = state.keys[layer_idx].clone()
        v_cache_kv_heads = state.values[layer_idx].clone()
        k_cache_kv_heads[:, state.position, :] = k_rope_kv_heads
        v_cache_kv_heads[:, state.position, :] = v_raw_kv_heads
        mask = _make_decode_mask(op.max_seq_len, state.position)
        repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        context_heads = torch.stack(
            [
                _attention_context_head(
                    q_rope_heads[head],
                    k_cache_kv_heads[head // repeats],
                    v_cache_kv_heads[head // repeats],
                    mask,
                    op.head_dim,
                )
                for head in range(cfg.num_attention_heads)
            ]
        ).contiguous()

        for lane in range(op.num_lanes):
            lane_start = lane * lane_span
            q_packet_index = layer_idx * op.phase_packets_per_layer
            q_packet_start = lane_start + q_packet_index * op.packet_elements
            k_packet_index = layer_idx * op.phase_packets_per_layer + K_PHASE
            k_packet_start = lane_start + k_packet_index * op.packet_elements
            v_packet_index = layer_idx * op.phase_packets_per_layer + V_PHASE
            v_packet_start = lane_start + v_packet_index * op.packet_elements
            q_rope_packet_index = layer_idx * op.phase_packets_per_layer + Q_ROPE_PHASE
            q_rope_packet_start = lane_start + q_rope_packet_index * op.packet_elements
            k_rope_packet_index = layer_idx * op.phase_packets_per_layer + K_ROPE_PHASE
            k_rope_packet_start = lane_start + k_rope_packet_index * op.packet_elements
            attention_packet_starts = [
                lane_start
                + (layer_idx * op.phase_packets_per_layer + phase) * op.packet_elements
                for phase in ATTENTION_SCORE_PV_PHASE_INDICES
            ]
            secondary_attention_packet_starts = [
                lane_start
                + (layer_idx * op.phase_packets_per_layer + phase) * op.packet_elements
                for phase in ATTENTION_SCORE_PV_SECONDARY_PHASE_INDICES
            ]
            row_base = lane * op.q_rows_per_packet
            context_head_index = lane
            secondary_context_head_index = lane + op.num_lanes
            kv_head_index = context_head_index // repeats
            secondary_kv_head_index = secondary_context_head_index // repeats
            lane_packets[q_rope_packet_start] = torch.tensor(
                row_base, dtype=torch.bfloat16
            )
            q_rope_base = q_rope_packet_start + 1
            lane_packets[q_rope_base : q_rope_base + op.head_dim] = q_raw_heads[0]
            lane_packets[q_rope_base + op.head_dim : q_rope_base + 2 * op.head_dim] = (
                q_norm_weight
            )
            lane_packets[
                q_rope_base + 2 * op.head_dim : q_rope_base + 3 * op.head_dim
            ] = cos_values
            lane_packets[
                q_rope_base + 3 * op.head_dim : q_rope_base + 4 * op.head_dim
            ] = sin_values
            lane_packets[k_rope_packet_start] = torch.tensor(
                row_base, dtype=torch.bfloat16
            )
            k_rope_base = k_rope_packet_start + 1
            lane_packets[k_rope_base : k_rope_base + op.head_dim] = k_raw_kv_heads[0]
            lane_packets[k_rope_base + op.head_dim : k_rope_base + 2 * op.head_dim] = (
                k_norm_weight
            )
            lane_packets[
                k_rope_base + 2 * op.head_dim : k_rope_base + 3 * op.head_dim
            ] = cos_values
            lane_packets[
                k_rope_base + 3 * op.head_dim : k_rope_base + 4 * op.head_dim
            ] = sin_values
            for row in range(op.q_rows_per_packet):
                q_row = row_base + row
                dst = q_packet_start + row * op.hidden_size
                lane_packets[dst : dst + op.hidden_size] = q_weight[q_row]
                out = layer_idx * out_layer_span + lane * out_lane_stride + row
                qwen3_reference[out] = _kernel_normed_row_dot(
                    hidden,
                    input_norm_weight,
                    q_weight[q_row],
                    cfg.rms_norm_eps,
                )
                k_dst = k_packet_start + row * op.hidden_size
                lane_packets[k_dst : k_dst + op.hidden_size] = k_weight[q_row]
                k_out = out + q_stride
                qwen3_reference[k_out] = _kernel_normed_row_dot(
                    hidden,
                    input_norm_weight,
                    k_weight[q_row],
                    cfg.rms_norm_eps,
                )
                v_dst = v_packet_start + row * op.hidden_size
                lane_packets[v_dst : v_dst + op.hidden_size] = v_weight[q_row]
                v_out = out + q_stride + k_stride
                qwen3_reference[v_out] = _kernel_normed_row_dot(
                    hidden,
                    input_norm_weight,
                    v_weight[q_row],
                    cfg.rms_norm_eps,
                )
                q_rope_out = out + q_stride + k_stride + v_stride
                qwen3_reference[q_rope_out] = _kernel_norm_rope_row(
                    q_raw_heads[0],
                    q_norm_weight,
                    cos_values,
                    sin_values,
                    q_row,
                )
                k_rope_out = q_rope_out + q_rope_stride
                qwen3_reference[k_rope_out] = _kernel_norm_rope_row(
                    k_raw_kv_heads[0],
                    k_norm_weight,
                    cos_values,
                    sin_values,
                    q_row,
                )
            for chunk_idx, attention_packet_start in enumerate(attention_packet_starts):
                _pack_attention_chunk_packet(
                    lane_packets,
                    attention_packet_start,
                    q_head=q_rope_heads[context_head_index],
                    k_cache_head=k_cache_kv_heads[kv_head_index],
                    v_cache_head=v_cache_kv_heads[kv_head_index],
                    mask=mask,
                    chunk_start=chunk_idx * op.attention_chunk_size,
                    chunk_size=op.attention_chunk_size,
                    head_dim=op.head_dim,
                )
            for chunk_idx, attention_packet_start in enumerate(
                secondary_attention_packet_starts
            ):
                _pack_attention_chunk_packet(
                    lane_packets,
                    attention_packet_start,
                    q_head=q_rope_heads[secondary_context_head_index],
                    k_cache_head=k_cache_kv_heads[secondary_kv_head_index],
                    v_cache_head=v_cache_kv_heads[secondary_kv_head_index],
                    mask=mask,
                    chunk_start=chunk_idx * op.attention_chunk_size,
                    chunk_size=op.attention_chunk_size,
                    head_dim=op.head_dim,
                )
            context_out = (
                layer_idx * out_layer_span
                + lane * out_lane_stride
                + q_stride
                + k_stride
                + v_stride
                + q_rope_stride
                + k_rope_stride
            )
            qwen3_reference[context_out : context_out + op.head_dim] = context_heads[
                context_head_index
            ]
            qwen3_reference[
                context_out + op.head_dim : context_out + 2 * op.head_dim
            ] = context_heads[secondary_context_head_index]

        residual = x
        attention_context, attention_out = _attention_decode_context_and_output(
            model,
            x_norm,
            layer_idx,
            state,
        )
        x = residual + attention_out.view_as(residual)
        attn_residual = x.flatten().contiguous()
        residual = x
        x_norm = rms_norm(
            x,
            model.w(f"{layer}.post_attention_layernorm.weight"),
            cfg.rms_norm_eps,
        )
        mlp = f"{layer}.mlp"
        gate = F.linear(x_norm, model.w(f"{mlp}.gate_proj.weight")).flatten()
        up = F.linear(x_norm, model.w(f"{mlp}.up_proj.weight")).flatten()
        ffn_hidden = F.silu(gate) * up
        gate_weight = model.w(f"{mlp}.gate_proj.weight").contiguous()
        up_weight = model.w(f"{mlp}.up_proj.weight").contiguous()
        down_weight = model.w(f"{mlp}.down_proj.weight").contiguous()
        o_weight = model.w(f"{attn}.o_proj.weight").contiguous()
        post_norm_weight = model.w(f"{layer}.post_attention_layernorm.weight").flatten()

        for lane in range(op.num_lanes):
            lane_start = lane * lane_span
            o_packet_index = layer_idx * op.phase_packets_per_layer + O_PHASE
            o_packet_start = lane_start + o_packet_index * op.packet_elements
            lane_packets[o_packet_start] = torch.tensor(lane, dtype=torch.bfloat16)
            lane_packets[o_packet_start + 1] = torch.tensor(
                lane + op.num_lanes, dtype=torch.bfloat16
            )
            lane_packets[
                o_packet_start + 2 : o_packet_start + 2 + op.attention_size
            ] = attention_context
            row_base = lane * op.q_rows_per_packet
            residual_base = o_packet_start + 2 + op.attention_size
            o_weight_base = residual_base + op.q_rows_per_packet
            out_base = (
                layer_idx * out_layer_span
                + lane * out_lane_stride
                + q_stride
                + k_stride
                + v_stride
                + q_rope_stride
                + k_rope_stride
                + context_stride
            )
            for row in range(op.q_rows_per_packet):
                proj_row = row_base + row
                lane_packets[residual_base + row] = hidden[proj_row]
                weight_dst = o_weight_base + row * op.attention_size
                lane_packets[weight_dst : weight_dst + op.attention_size] = o_weight[
                    proj_row
                ]
                lane_context = attention_context.clone()
                head_start = lane * op.head_dim
                lane_context[head_start : head_start + op.head_dim] = context_heads[
                    lane
                ]
                secondary_head_start = (lane + op.num_lanes) * op.head_dim
                lane_context[
                    secondary_head_start : secondary_head_start + op.head_dim
                ] = context_heads[lane + op.num_lanes]
                qwen3_reference[out_base + row] = _kernel_down_residual_row(
                    lane_context,
                    o_weight[proj_row],
                    hidden[proj_row],
                )

        for lane in range(op.num_lanes):
            lane_start = lane * lane_span
            gate_packet_index = layer_idx * op.phase_packets_per_layer + GATE_UP_PHASE
            gate_packet_start = lane_start + gate_packet_index * op.packet_elements
            row_base = lane * op.q_rows_per_packet
            lane_packets[gate_packet_start] = torch.tensor(
                row_base, dtype=torch.bfloat16
            )
            lane_packets[
                gate_packet_start + 1 : gate_packet_start + 1 + op.hidden_size
            ] = attn_residual
            lane_packets[
                gate_packet_start
                + 1
                + op.hidden_size : gate_packet_start
                + 1
                + 2 * op.hidden_size
            ] = model.w(f"{layer}.post_attention_layernorm.weight").flatten()
            gate_weight_base = gate_packet_start + 1 + 2 * op.hidden_size
            up_weight_base = gate_weight_base + op.q_rows_per_packet * op.hidden_size
            lane_attn_residual = attn_residual.clone()
            attention_base = (
                layer_idx * out_layer_span
                + lane * out_lane_stride
                + q_stride
                + k_stride
                + v_stride
                + q_rope_stride
                + k_rope_stride
                + context_stride
            )
            lane_attn_residual[row_base : row_base + op.q_rows_per_packet] = (
                qwen3_reference[attention_base : attention_base + op.q_rows_per_packet]
            )
            out_base = (
                layer_idx * out_layer_span
                + lane * out_lane_stride
                + q_stride
                + k_stride
                + v_stride
                + q_rope_stride
                + k_rope_stride
                + context_stride
                + attention_stride
            )
            for row in range(op.q_rows_per_packet):
                proj_row = row_base + row
                gate_dst = gate_weight_base + row * op.hidden_size
                up_dst = up_weight_base + row * op.hidden_size
                lane_packets[gate_dst : gate_dst + op.hidden_size] = gate_weight[
                    proj_row
                ]
                lane_packets[up_dst : up_dst + op.hidden_size] = up_weight[proj_row]
                qwen3_reference[out_base + row] = _kernel_normed_row_dot(
                    lane_attn_residual,
                    post_norm_weight,
                    gate_weight[proj_row],
                    cfg.rms_norm_eps,
                )
                qwen3_reference[out_base + op.q_rows_per_packet + row] = (
                    _kernel_normed_row_dot(
                        lane_attn_residual,
                        post_norm_weight,
                        up_weight[proj_row],
                        cfg.rms_norm_eps,
                    )
                )

        for lane in range(op.num_lanes):
            lane_start = lane * lane_span
            down_packet_index = layer_idx * op.phase_packets_per_layer + DOWN_PHASE
            down_packet_start = lane_start + down_packet_index * op.packet_elements
            row_base = lane * op.q_rows_per_packet
            lane_packets[down_packet_start] = torch.tensor(
                row_base, dtype=torch.bfloat16
            )
            lane_packets[
                down_packet_start + 1 : down_packet_start + 1 + op.intermediate_size
            ] = ffn_hidden
            residual_base = down_packet_start + 1 + op.intermediate_size
            down_weight_base = residual_base + op.q_rows_per_packet
            lane_ffn_hidden = ffn_hidden.clone()
            gate_base_for_lane = (
                layer_idx * out_layer_span
                + lane * out_lane_stride
                + q_stride
                + k_stride
                + v_stride
                + q_rope_stride
                + k_rope_stride
                + context_stride
                + attention_stride
            )
            local_gate = qwen3_reference[
                gate_base_for_lane : gate_base_for_lane + op.q_rows_per_packet
            ].to(torch.float32)
            local_up = qwen3_reference[
                gate_base_for_lane
                + op.q_rows_per_packet : gate_base_for_lane
                + 2 * op.q_rows_per_packet
            ].to(torch.float32)
            lane_ffn_hidden[row_base : row_base + op.q_rows_per_packet] = (
                F.silu(local_gate) * local_up
            )
            out_base = (
                layer_idx * out_layer_span
                + lane * out_lane_stride
                + q_stride
                + k_stride
                + v_stride
                + q_rope_stride
                + k_rope_stride
                + context_stride
                + attention_stride
                + gate_up_stride
            )
            for row in range(op.q_rows_per_packet):
                proj_row = row_base + row
                lane_packets[residual_base + row] = attn_residual[proj_row]
                weight_dst = down_weight_base + row * op.intermediate_size
                lane_packets[weight_dst : weight_dst + op.intermediate_size] = (
                    down_weight[proj_row]
                )
                qwen3_reference[out_base + row] = _kernel_down_residual_row(
                    lane_ffn_hidden,
                    down_weight[proj_row],
                    attn_residual[proj_row],
                )

        x = residual + ref._mlp(x_norm, layer_idx)

        next_hidden = x.flatten().contiguous()
        next_packet_index = (layer_idx + 1) * op.phase_packets_per_layer - 1
        for lane in range(op.num_lanes):
            lane_start = lane * lane_span
            next_start = lane_start + next_packet_index * op.packet_elements
            lane_packets[next_start : next_start + op.hidden_size] = next_hidden

    expected = phase_owned_reference(shared_packets, lane_packets, op)
    return PhaseOwnedCase(
        shared_packets=shared_packets,
        lane_packets=lane_packets,
        expected=expected,
        qwen3_reference=qwen3_reference,
        model_dir=model_dir,
        prompt_tokens=input_ids.shape[1],
        next_token=next_token,
    )


def compile_phase_owned_stage(
    *,
    num_lanes: int,
    num_layers: int,
    phase_packets_per_layer: int,
    packet_elements: int | None,
    hidden_size: int,
    attention_size: int,
    head_dim: int,
    intermediate_size: int,
    max_seq_len: int,
    attention_chunk_size: int,
    q_rows_per_packet: int,
    fabric_group_size: int,
    build_dir: Path,
) -> tuple[NewMegaPhaseOwnedDecode, PersistentPreflightResult]:
    context = AIEContext(build_dir=build_dir)
    if packet_elements is None:
        gate_up_elements = 1 + (2 + 2 * q_rows_per_packet) * hidden_size
        o_elements = (
            2 + attention_size + q_rows_per_packet + q_rows_per_packet * attention_size
        )
        down_elements = (
            1
            + intermediate_size
            + q_rows_per_packet
            + q_rows_per_packet * intermediate_size
        )
        norm_rope_elements = 1 + 4 * head_dim
        attention_chunk_elements = (
            head_dim + 2 * attention_chunk_size * head_dim + attention_chunk_size
        )
        packet_elements = max(
            o_elements,
            gate_up_elements,
            down_elements,
            norm_rope_elements,
            attention_chunk_elements,
        )
        packet_elements = ((packet_elements + 7) // 8) * 8
    op = NewMegaPhaseOwnedDecode(
        num_lanes=num_lanes,
        num_layers=num_layers,
        phase_packets_per_layer=phase_packets_per_layer,
        packet_elements=packet_elements,
        hidden_size=hidden_size,
        attention_size=attention_size,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
        attention_chunk_size=attention_chunk_size,
        q_rows_per_packet=q_rows_per_packet,
        fabric_group_size=fabric_group_size,
        context=context,
    )
    op.compile()
    preflight = run_persistent_artifact_preflight(
        mlir_path=Path(op.xclbin_artifact.mlir_input.filename),
        arg_specs=len(op.get_arg_spec()),
    )
    return op, preflight


def run_phase_owned_stage(
    *,
    op: NewMegaPhaseOwnedDecode,
    preflight: PersistentPreflightResult,
    case: PhaseOwnedCase | None,
    abs_tol: float,
    compile_only: bool = False,
) -> PhaseOwnedRunResult:
    if compile_only:
        return PhaseOwnedRunResult(
            op=op,
            preflight=preflight,
            case=None,
            npu_time_us=None,
            max_abs=None,
            mean_abs=None,
            errors=None,
            qwen3_max_abs=None,
            qwen3_mean_abs=None,
            qwen3_errors=None,
        )
    if case is None:
        raise ValueError("case is required unless compile_only=True")

    out_buf = XRTTensor((op.output_elements,), dtype=bfloat16)
    result = op.get_callable()(
        XRTTensor.from_torch(case.shared_packets),
        XRTTensor.from_torch(case.lane_packets),
        out_buf,
    )
    actual = out_buf.to_torch().detach().clone().contiguous()

    diff = (actual.to(torch.float32) - case.expected.to(torch.float32)).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    errors = int((diff > abs_tol).sum().item())
    qwen3_diff = (
        actual.to(torch.float32) - case.qwen3_reference.to(torch.float32)
    ).abs()
    qwen3_max_abs = float(qwen3_diff.max().item())
    qwen3_mean_abs = float(qwen3_diff.mean().item())
    qwen3_errors = int((qwen3_diff > abs_tol).sum().item())
    return PhaseOwnedRunResult(
        op=op,
        preflight=preflight,
        case=case,
        npu_time_us=result.npu_time / 1000.0,
        max_abs=max_abs,
        mean_abs=mean_abs,
        errors=errors,
        qwen3_max_abs=qwen3_max_abs,
        qwen3_mean_abs=qwen3_mean_abs,
        qwen3_errors=qwen3_errors,
    )


def print_preflight(result: PersistentPreflightResult) -> None:
    print(f"preflight_runtime_memrefs: {result.runtime_memrefs}")
    print(f"preflight_arg_specs: {result.arg_specs}")
    print(f"preflight_metadata_host_bos: {result.metadata_host_bos}")
    print(f"preflight_compute_cores: {result.compute_cores}")
    print(f"preflight_max_fifo_buffered_bytes: {result.max_fifo_buffered_bytes}")
    print(f"preflight_total_dma_tasks: {result.total_dma_tasks}")
    print(f"preflight_max_dma_tasks_per_fifo: {result.max_dma_tasks_per_fifo}")
    print(f"preflight_max_compute_tile_inputs: {result.max_compute_tile_inputs}")
    print(f"preflight_max_compute_tile_outputs: {result.max_compute_tile_outputs}")
    print(f"preflight_non_advancing_acquires: {result.non_advancing_acquires}")


def print_phase_owned_run(result: PhaseOwnedRunResult) -> None:
    op = result.op
    print(f"xclbin: {op.xclbin_artifact.filename}")
    print(f"runtime_bin: {op.insts_artifact.filename}")
    print(f"mlir: {op.xclbin_artifact.mlir_input.filename}")
    print(f"num_lanes: {op.num_lanes}")
    print(f"num_layers: {op.num_layers}")
    print(f"phase_packets_per_layer: {op.phase_packets_per_layer}")
    print(f"total_phase_packets: {op.total_phase_packets}")
    print(f"phase_labels: {','.join(PHASE_LABELS[: op.phase_packets_per_layer])}")
    print(f"hidden_size: {op.hidden_size}")
    print(f"attention_size: {op.attention_size}")
    print(f"attention_head_count: {op.attention_head_count}")
    print(
        f"npu_context_heads_per_layer: {min(op.attention_head_count, 2 * op.num_lanes)}"
    )
    print(f"head_dim: {op.head_dim}")
    print(f"intermediate_size: {op.intermediate_size}")
    print(f"max_seq_len: {op.max_seq_len}")
    print(f"attention_chunk_size: {op.attention_chunk_size}")
    print(f"attention_chunk_count: {op.attention_chunk_count}")
    print(f"q_rows_per_packet: {op.q_rows_per_packet}")
    print(f"fabric_group_size: {op.fabric_group_size}")
    print(f"shared_packet_elements: {op.shared_packet_elements}")
    print(f"shared_input_elements: {op.shared_input_elements}")
    print(f"tile_local_hidden_elements: {op.hidden_size}")
    print(f"q_output_values_per_lane: {op.q_output_values_per_lane}")
    print(f"k_output_values_per_lane: {op.k_output_values_per_lane}")
    print(f"v_output_values_per_lane: {op.v_output_values_per_lane}")
    print(f"q_rope_output_values_per_lane: {op.q_rope_output_values_per_lane}")
    print(f"k_rope_output_values_per_lane: {op.k_rope_output_values_per_lane}")
    print(f"context_output_values_per_lane: {op.context_output_values_per_lane}")
    print(f"attention_output_values_per_lane: {op.attention_output_values_per_lane}")
    print(f"gate_up_output_values_per_lane: {op.gate_up_output_values_per_lane}")
    print(f"residual_output_values_per_lane: {op.residual_output_values_per_lane}")
    print(f"output_values_per_lane: {op.output_values_per_lane}")
    print(f"output_values_per_layer: {op.output_values_per_layer}")
    print(f"packet_elements: {op.packet_elements}")
    print(f"packet_bytes: {op.packet_bytes}")
    print(f"lane_stream_bytes: {op.lane_stream_bytes}")
    print(f"input_elements: {op.input_elements}")
    print(f"output_elements: {op.output_elements}")
    print(f"packet_fits_l1_64k: {op.packet_bytes <= 64 * 1024}")
    print_preflight(result.preflight)
    if result.case is not None:
        print(f"model_dir: {result.case.model_dir}")
        print(f"prompt_tokens: {result.case.prompt_tokens}")
        print(f"next_token: {result.case.next_token}")
    if result.npu_time_us is None:
        print("decision: compile-only")
        return
    print(f"npu_time_us: {result.npu_time_us:.3f}")
    print(f"phase_owned_max_abs: {result.max_abs:.6f}")
    print(f"phase_owned_mean_abs: {result.mean_abs:.6f}")
    print(f"phase_owned_errors: {result.errors}")
    print(f"qwen3_phase_output_max_abs: {result.qwen3_max_abs:.6f}")
    print(f"qwen3_phase_output_mean_abs: {result.qwen3_mean_abs:.6f}")
    print(f"qwen3_phase_output_errors: {result.qwen3_errors}")
