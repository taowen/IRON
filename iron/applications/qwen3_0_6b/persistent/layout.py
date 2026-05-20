#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import numpy as np
import torch

from iron.applications.qwen3_0_6b.qwen3_cpu import Qwen3ForCausalLM
from iron.applications.qwen3_0_6b.persistent.refs import rope_lut_for_position

PACKED_WEIGHTS_FORMAT = "qwen3_iron_packed_weights_v1"
PACKED_WEIGHTS_BIN = "weights.bf16.bin"
PACKED_WEIGHTS_MANIFEST = "manifest.json"
FULL_LAYER_WEIGHT_ORDER = (
    "input_norm_weight",
    "W_q",
    "W_k",
    "W_v",
    "W_q_norm",
    "W_k_norm",
    "W_o",
    "post_norm_weight",
    "W_gate",
    "W_up",
    "W_down",
)

SEGMENT_MAJOR_WEIGHT_ORDER = (
    "input_norm_weight",
    "W_q",
    "W_k",
    "W_v",
    "W_qk_norm",
    "W_o",
    "post_norm_gate_up",
    "W_down",
)


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
        [inputs[name].flatten() for name in FULL_LAYER_WEIGHT_ORDER]
    ).contiguous()


def _segment_major_inputs_from_full_layer_inputs(
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "input_norm_weight": inputs["input_norm_weight"].flatten(),
        "W_q": inputs["W_q"].flatten(),
        "W_k": inputs["W_k"].flatten(),
        "W_v": inputs["W_v"].flatten(),
        "W_qk_norm": torch.cat(
            [inputs["W_q_norm"].flatten(), inputs["W_k_norm"].flatten()]
        ),
        "W_o": inputs["W_o"].flatten(),
        "post_norm_gate_up": torch.cat(
            [
                inputs["post_norm_weight"].flatten(),
                inputs["W_gate"].flatten(),
                inputs["W_up"].flatten(),
            ]
        ),
        "post_norm_weight": inputs["post_norm_weight"].flatten(),
        "W_gate": inputs["W_gate"].flatten(),
        "W_up": inputs["W_up"].flatten(),
        "W_down": inputs["W_down"].flatten(),
    }


def _pack_segment_major_full_layer_weights_default(
    segment_inputs_by_layer: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    return torch.cat(
        [
            segment_inputs[name].flatten()
            for name in SEGMENT_MAJOR_WEIGHT_ORDER
            for segment_inputs in segment_inputs_by_layer
        ]
    ).contiguous()


def _gate_up_shard(
    gate: torch.Tensor,
    up: torch.Tensor,
    col: int,
    *,
    columns: int = 2,
    pair_rows: bool = False,
    row_group: int = 4,
) -> torch.Tensor:
    hidden_size = gate.shape[1]
    if gate.shape[0] % columns != 0:
        raise ValueError(
            f"gate/up rows {gate.shape[0]} must be divisible by {columns} columns"
        )
    shard_rows = gate.shape[0] // columns
    start = col * shard_rows
    end = start + shard_rows
    gate_shard = gate[start:end].contiguous()
    up_shard = up[start:end].contiguous()
    if not pair_rows:
        return torch.cat([gate_shard.flatten(), up_shard.flatten()])
    if shard_rows % row_group != 0:
        raise ValueError(
            f"gate/up shard rows {shard_rows} must be divisible by {row_group}"
        )
    pieces = []
    for row_start in range(0, shard_rows, row_group):
        row_end = row_start + row_group
        pieces.append(gate_shard[row_start:row_end].reshape(row_group, hidden_size))
        pieces.append(up_shard[row_start:row_end].reshape(row_group, hidden_size))
    return torch.cat([piece.flatten() for piece in pieces])


def _interleave_same_size_chunks(
    tensors: list[torch.Tensor],
    *,
    chunk_size: int,
) -> torch.Tensor:
    if not tensors:
        raise ValueError("tensors must not be empty")
    first_numel = tensors[0].numel()
    if first_numel % chunk_size != 0:
        raise ValueError(
            f"tensor length {first_numel} must be divisible by chunk_size {chunk_size}"
        )
    num_chunks = first_numel // chunk_size
    for tensor in tensors[1:]:
        if tensor.numel() != first_numel:
            raise ValueError("all tensors must have the same number of elements")
    pieces = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        pieces.extend(tensor[start:end] for tensor in tensors)
    return torch.cat(pieces).contiguous()


def _pack_segment_major_full_layer_weights_mlp2(
    segment_inputs_by_layer: list[dict[str, torch.Tensor]],
    *,
    mlp_gate_up_pair_rows: bool = False,
    mlp_gate_up_row_group: int = 4,
) -> torch.Tensor:
    def gate_up_shard(
        segment_inputs: dict[str, torch.Tensor], col: int
    ) -> torch.Tensor:
        hidden_size = segment_inputs["post_norm_weight"].numel()
        gate = segment_inputs["W_gate"].view(-1, hidden_size)
        up = segment_inputs["W_up"].view(-1, hidden_size)
        return _gate_up_shard(
            gate,
            up,
            col,
            pair_rows=mlp_gate_up_pair_rows,
            row_group=mlp_gate_up_row_group,
        )

    def down_shard(segment_inputs: dict[str, torch.Tensor], col: int) -> torch.Tensor:
        hidden_size = segment_inputs["post_norm_weight"].numel()
        down = segment_inputs["W_down"].view(hidden_size, -1)
        shard_rows = down.shape[0] // 2
        start = col * shard_rows
        end = start + shard_rows
        return down[start:end].flatten()

    common = [
        segment_inputs[name].flatten()
        for name in (
            "input_norm_weight",
            "W_q",
            "W_k",
            "W_v",
            "W_qk_norm",
            "W_o",
        )
        for segment_inputs in segment_inputs_by_layer
    ]
    post_norm = [
        segment_inputs["post_norm_weight"].flatten()
        for segment_inputs in segment_inputs_by_layer
    ]
    gate_up = [
        gate_up_shard(segment_inputs, col)
        for col in range(2)
        for segment_inputs in segment_inputs_by_layer
    ]
    down = [
        down_shard(segment_inputs, col)
        for col in range(2)
        for segment_inputs in segment_inputs_by_layer
    ]
    return torch.cat(common + post_norm + gate_up + down).contiguous()


def _pack_segment_major_full_layer_weights_attention2_mlp2(
    segment_inputs_by_layer: list[dict[str, torch.Tensor]],
    *,
    mlp_gate_up_columns: int = 2,
    mlp_gate_up_pair_rows: bool = False,
    mlp_gate_up_row_group: int = 4,
) -> torch.Tensor:
    def row_shard(
        segment_inputs: dict[str, torch.Tensor], name: str, col: int
    ) -> torch.Tensor:
        hidden_size = segment_inputs["post_norm_weight"].numel()
        matrix = segment_inputs[name].view(-1, hidden_size)
        shard_rows = matrix.shape[0] // 2
        start = col * shard_rows
        end = start + shard_rows
        return matrix[start:end].flatten()

    def qk_interleaved_shard(
        segment_inputs: dict[str, torch.Tensor], col: int
    ) -> torch.Tensor:
        hidden_size = segment_inputs["post_norm_weight"].numel()
        head_dim = segment_inputs["W_qk_norm"].numel() // 2
        q_matrix = segment_inputs["W_q"].view(-1, hidden_size)
        k_matrix = segment_inputs["W_k"].view(-1, hidden_size)
        q_repeat = q_matrix.shape[0] // k_matrix.shape[0]
        kv_heads_per_col = (k_matrix.shape[0] // head_dim) // 2
        q_head_start = col * kv_heads_per_col * q_repeat
        k_head_start = col * kv_heads_per_col
        pieces = []
        for local_kv_head in range(kv_heads_per_col):
            k_head = k_head_start + local_kv_head
            q_head = q_head_start + local_kv_head * q_repeat
            pieces.append(
                k_matrix[k_head * head_dim : (k_head + 1) * head_dim].flatten()
            )
            pieces.append(
                q_matrix[q_head * head_dim : (q_head + q_repeat) * head_dim].flatten()
            )
        return torch.cat(pieces)

    def o_context_shard(
        segment_inputs: dict[str, torch.Tensor], col: int
    ) -> torch.Tensor:
        hidden_size = segment_inputs["post_norm_weight"].numel()
        matrix = segment_inputs["W_o"].view(hidden_size, -1)
        shard_cols = matrix.shape[1] // 2
        start = col * shard_cols
        end = start + shard_cols
        return matrix[:, start:end].contiguous().flatten()

    def gate_up_shard(
        segment_inputs: dict[str, torch.Tensor], col: int
    ) -> torch.Tensor:
        hidden_size = segment_inputs["post_norm_weight"].numel()
        gate = segment_inputs["W_gate"].view(-1, hidden_size)
        up = segment_inputs["W_up"].view(-1, hidden_size)
        row_group = 4 if col == 0 else mlp_gate_up_row_group
        return _gate_up_shard(
            gate,
            up,
            col,
            columns=mlp_gate_up_columns,
            pair_rows=mlp_gate_up_pair_rows,
            row_group=row_group,
        )

    def down_shard(segment_inputs: dict[str, torch.Tensor], col: int) -> torch.Tensor:
        hidden_size = segment_inputs["post_norm_weight"].numel()
        down = segment_inputs["W_down"].view(hidden_size, -1)
        shard_rows = down.shape[0] // 2
        start = col * shard_rows
        end = start + shard_rows
        return down[start:end].flatten()

    input_norm = [
        segment_inputs["input_norm_weight"].flatten()
        for segment_inputs in segment_inputs_by_layer
    ]
    qkv_shards = []
    for col in range(2):
        qkv_shards.extend(
            qk_interleaved_shard(segment_inputs, col)
            for segment_inputs in segment_inputs_by_layer
        )
        qkv_shards.extend(
            row_shard(segment_inputs, "W_v", col)
            for segment_inputs in segment_inputs_by_layer
        )
    qk_norm = [
        segment_inputs["W_qk_norm"].flatten()
        for segment_inputs in segment_inputs_by_layer
    ]
    o_shards = [
        o_context_shard(segment_inputs, col)
        for col in range(2)
        for segment_inputs in segment_inputs_by_layer
    ]
    gate_up_col0 = [
        torch.cat(
            [
                segment_inputs["post_norm_weight"].flatten(),
                gate_up_shard(segment_inputs, 0),
            ]
        )
        for segment_inputs in segment_inputs_by_layer
    ]
    if mlp_gate_up_columns == 3:
        gate_up_rest = [
            _interleave_same_size_chunks(
                [
                    gate_up_shard(segment_inputs, 1),
                    gate_up_shard(segment_inputs, 2),
                ],
                chunk_size=segment_inputs["post_norm_weight"].numel(),
            )
            for segment_inputs in segment_inputs_by_layer
        ]
    else:
        gate_up_rest = [
            gate_up_shard(segment_inputs, col)
            for col in range(1, mlp_gate_up_columns)
            for segment_inputs in segment_inputs_by_layer
        ]
    down = [
        down_shard(segment_inputs, col)
        for col in range(2)
        for segment_inputs in segment_inputs_by_layer
    ]
    return torch.cat(
        input_norm
        + qkv_shards
        + qk_norm
        + o_shards
        + gate_up_col0
        + gate_up_rest
        + down
    ).contiguous()


def _pack_segment_major_full_layer_weights_attention2_default_mlp(
    segment_inputs_by_layer: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    def row_shard(
        segment_inputs: dict[str, torch.Tensor], name: str, col: int
    ) -> torch.Tensor:
        hidden_size = segment_inputs["post_norm_weight"].numel()
        matrix = segment_inputs[name].view(-1, hidden_size)
        shard_rows = matrix.shape[0] // 2
        start = col * shard_rows
        end = start + shard_rows
        return matrix[start:end].flatten()

    def qk_interleaved_shard(
        segment_inputs: dict[str, torch.Tensor], col: int
    ) -> torch.Tensor:
        hidden_size = segment_inputs["post_norm_weight"].numel()
        head_dim = segment_inputs["W_qk_norm"].numel() // 2
        q_matrix = segment_inputs["W_q"].view(-1, hidden_size)
        k_matrix = segment_inputs["W_k"].view(-1, hidden_size)
        q_repeat = q_matrix.shape[0] // k_matrix.shape[0]
        kv_heads_per_col = (k_matrix.shape[0] // head_dim) // 2
        q_head_start = col * kv_heads_per_col * q_repeat
        k_head_start = col * kv_heads_per_col
        pieces = []
        for local_kv_head in range(kv_heads_per_col):
            k_head = k_head_start + local_kv_head
            q_head = q_head_start + local_kv_head * q_repeat
            pieces.append(
                k_matrix[k_head * head_dim : (k_head + 1) * head_dim].flatten()
            )
            pieces.append(
                q_matrix[q_head * head_dim : (q_head + q_repeat) * head_dim].flatten()
            )
        return torch.cat(pieces)

    def o_context_shard(
        segment_inputs: dict[str, torch.Tensor], col: int
    ) -> torch.Tensor:
        hidden_size = segment_inputs["post_norm_weight"].numel()
        matrix = segment_inputs["W_o"].view(hidden_size, -1)
        shard_cols = matrix.shape[1] // 2
        start = col * shard_cols
        end = start + shard_cols
        return matrix[:, start:end].contiguous().flatten()

    input_norm = [
        segment_inputs["input_norm_weight"].flatten()
        for segment_inputs in segment_inputs_by_layer
    ]
    qkv_shards = []
    for col in range(2):
        qkv_shards.extend(
            qk_interleaved_shard(segment_inputs, col)
            for segment_inputs in segment_inputs_by_layer
        )
        qkv_shards.extend(
            row_shard(segment_inputs, "W_v", col)
            for segment_inputs in segment_inputs_by_layer
        )
    qk_norm = [
        segment_inputs["W_qk_norm"].flatten()
        for segment_inputs in segment_inputs_by_layer
    ]
    o_shards = [
        o_context_shard(segment_inputs, col)
        for col in range(2)
        for segment_inputs in segment_inputs_by_layer
    ]
    mlp = [
        segment_inputs[name].flatten()
        for name in ("post_norm_gate_up", "W_down")
        for segment_inputs in segment_inputs_by_layer
    ]
    return torch.cat(input_norm + qkv_shards + qk_norm + o_shards + mlp).contiguous()


def pack_segment_major_full_layer_weights(
    inputs_by_layer: list[dict[str, torch.Tensor]],
    mlp_columns: int = 1,
    mlp_gate_up_columns: int = 0,
    attention_columns: int = 1,
    mlp_gate_up_pair_rows: bool = False,
    mlp_gate_up_row_group: int = 4,
) -> torch.Tensor:
    """Pack a layer chunk by segment, then by layer within each segment.

    The persistent n-layer graph consumes one ObjectFIFO stream per logical
    weight segment. Packing all layers for a segment contiguously lets one DMA
    task feed that FIFO for the whole chunk, instead of emitting one DMA task
    per layer.
    """
    if not inputs_by_layer:
        raise ValueError("inputs_by_layer must not be empty")
    segment_inputs_by_layer = [
        _segment_major_inputs_from_full_layer_inputs(inputs)
        for inputs in inputs_by_layer
    ]
    if mlp_gate_up_columns == 0:
        mlp_gate_up_columns = 2 if mlp_columns == 2 else 1
    if attention_columns == 2:
        if mlp_columns == 1:
            if mlp_gate_up_columns != 1:
                raise ValueError("mlp_columns=1 requires mlp_gate_up_columns=1")
            return _pack_segment_major_full_layer_weights_attention2_default_mlp(
                segment_inputs_by_layer
            )
        if mlp_columns != 2:
            raise ValueError("unsupported attention2 MLP column layout")
        if mlp_gate_up_columns == 1:
            return _pack_segment_major_full_layer_weights_attention2_default_mlp(
                segment_inputs_by_layer
            )
        if mlp_gate_up_columns not in {2, 3}:
            raise ValueError("unsupported attention2 gate/up column layout")
        return _pack_segment_major_full_layer_weights_attention2_mlp2(
            segment_inputs_by_layer,
            mlp_gate_up_columns=mlp_gate_up_columns,
            mlp_gate_up_pair_rows=mlp_gate_up_pair_rows,
            mlp_gate_up_row_group=mlp_gate_up_row_group,
        )
    if attention_columns != 1:
        raise ValueError(
            f"unsupported segment-major attention column layout: {attention_columns}"
        )
    if mlp_columns == 1:
        if mlp_gate_up_columns != 1:
            raise ValueError("mlp_columns=1 requires mlp_gate_up_columns=1")
        return _pack_segment_major_full_layer_weights_default(segment_inputs_by_layer)
    if mlp_columns == 2:
        if mlp_gate_up_columns == 1:
            return _pack_segment_major_full_layer_weights_default(
                segment_inputs_by_layer
            )
        if mlp_gate_up_columns != 2:
            raise ValueError("unsupported MLP gate/up column layout")
        return _pack_segment_major_full_layer_weights_mlp2(
            segment_inputs_by_layer,
            mlp_gate_up_pair_rows=mlp_gate_up_pair_rows,
            mlp_gate_up_row_group=mlp_gate_up_row_group,
        )
    raise ValueError(f"unsupported segment-major MLP column layout: {mlp_columns}")


def pack_segment_major_weights_for_layers(
    model: Qwen3ForCausalLM,
    layer_indices: list[int] | range,
    mlp_columns: int = 1,
    mlp_gate_up_columns: int = 0,
    attention_columns: int = 1,
    mlp_gate_up_pair_rows: bool = False,
    mlp_gate_up_row_group: int = 4,
) -> torch.Tensor:
    return pack_segment_major_full_layer_weights(
        [
            build_full_layer_weight_inputs_for_layer(model, layer_idx)
            for layer_idx in layer_indices
        ],
        mlp_columns=mlp_columns,
        mlp_gate_up_columns=mlp_gate_up_columns,
        attention_columns=attention_columns,
        mlp_gate_up_pair_rows=mlp_gate_up_pair_rows,
        mlp_gate_up_row_group=mlp_gate_up_row_group,
    )


def pack_qk_rope_metadata_for_layers(
    inputs_by_layer: list[dict[str, torch.Tensor]],
    *,
    position: int | None = None,
    valid_length: int | None = None,
) -> torch.Tensor:
    """Pack q_norm, k_norm, RoPE LUT, and optional decode metadata."""
    position_metadata = None
    if position is not None:
        if valid_length is None:
            valid_length = position + 1
        position_metadata = torch.tensor(
            [position, valid_length, 0, 0, 0, 0, 0, 0],
            dtype=torch.float32,
        ).to(dtype=inputs_by_layer[0]["rope_angles"].dtype)
    return torch.cat(
        [
            torch.cat(
                (
                    [
                        inputs["W_q_norm"].flatten(),
                        inputs["W_k_norm"].flatten(),
                        inputs["rope_angles"].flatten(),
                    ]
                    if position_metadata is None
                    else [
                        inputs["W_q_norm"].flatten(),
                        inputs["W_k_norm"].flatten(),
                        inputs["rope_angles"].flatten(),
                        position_metadata,
                    ]
                )
            )
            for inputs in inputs_by_layer
        ]
    ).contiguous()


def pack_full_layer_weights_for_layer(
    model: Qwen3ForCausalLM,
    layer_idx: int,
) -> torch.Tensor:
    return pack_full_layer_weights(
        build_full_layer_weight_inputs_for_layer(model, layer_idx)
    )


def default_packed_weights_dir(model_dir: Path) -> Path:
    return Path(model_dir) / "qwen3_iron_packed"


def _config_dict(model: Qwen3ForCausalLM) -> dict[str, object]:
    config = model.config
    if is_dataclass(config):
        raw = asdict(config)
    else:
        raw = dict(vars(config))
    keys = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "tie_word_embeddings",
        "torch_dtype",
    )
    return {key: raw[key] for key in keys if key in raw}


def _bf16_to_file(tensor: torch.Tensor, path: Path) -> None:
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    tensor = tensor.contiguous()
    raw = tensor.view(torch.uint16).cpu().numpy()
    raw.tofile(path)


def _bf16_from_file(path: Path, numel: int) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != numel:
        raise RuntimeError(
            f"packed weight file has {raw.size} bf16 elements, expected {numel}"
        )
    return torch.from_numpy(raw.copy()).view(torch.bfloat16)


def _byte_alignment(byte_offset: int) -> int | None:
    if byte_offset == 0:
        return None
    return byte_offset & -byte_offset


def build_full_layer_weight_manifest_for_layer(
    model: Qwen3ForCausalLM,
    layer_idx: int,
    layer_base_offset: int,
) -> tuple[list[dict[str, object]], int]:
    inputs = build_full_layer_weight_inputs_for_layer(model, layer_idx)
    offset = layer_base_offset
    segments: list[dict[str, object]] = []
    for name in FULL_LAYER_WEIGHT_ORDER:
        tensor = inputs[name]
        numel = tensor.numel()
        segments.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "bfloat16",
                "element_offset": offset,
                "numel": numel,
                "byte_offset": offset * 2,
                "byte_length": numel * 2,
                "byte_alignment": _byte_alignment(offset * 2),
            }
        )
        offset += numel
    return segments, offset


def build_packed_weights_manifest(
    model: Qwen3ForCausalLM,
    per_layer_numel: int,
) -> dict[str, object]:
    layers = []
    offset = 0
    for layer_idx in range(model.config.num_hidden_layers):
        segments, end_offset = build_full_layer_weight_manifest_for_layer(
            model,
            layer_idx,
            offset,
        )
        actual_numel = end_offset - offset
        if actual_numel != per_layer_numel:
            raise RuntimeError(
                f"layer {layer_idx} packed weight size {actual_numel} "
                f"!= expected per-layer size {per_layer_numel}"
            )
        layers.append(
            {
                "id": layer_idx,
                "element_offset": offset,
                "numel": per_layer_numel,
                "byte_offset": offset * 2,
                "byte_length": per_layer_numel * 2,
                "byte_alignment": _byte_alignment(offset * 2),
                "segments": segments,
            }
        )
        offset = end_offset

    return {
        "format": PACKED_WEIGHTS_FORMAT,
        "dtype": "bfloat16",
        "element_size_bytes": 2,
        "weight_order": list(FULL_LAYER_WEIGHT_ORDER),
        "model_config": _config_dict(model),
        "num_layers": model.config.num_hidden_layers,
        "per_layer_numel": per_layer_numel,
        "total_numel": offset,
        "total_bytes": offset * 2,
        "data_file": PACKED_WEIGHTS_BIN,
        "layers": layers,
    }


def pack_all_full_layer_weights(
    model: Qwen3ForCausalLM,
    expected_per_layer_numel: int | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    packed_layers = []
    for layer_idx in range(model.config.num_hidden_layers):
        packed = pack_full_layer_weights_for_layer(model, layer_idx).to(torch.bfloat16)
        if expected_per_layer_numel is not None and (
            packed.numel() != expected_per_layer_numel
        ):
            raise RuntimeError(
                f"layer {layer_idx} packed weight size {packed.numel()} "
                f"!= expected per-layer size {expected_per_layer_numel}"
            )
        packed_layers.append(packed)
    per_layer_numel = packed_layers[0].numel()
    weights = torch.cat(packed_layers).contiguous()
    manifest = build_packed_weights_manifest(model, per_layer_numel)
    return weights, manifest


def write_packed_weight_artifact(
    model: Qwen3ForCausalLM,
    output_dir: Path,
    expected_per_layer_numel: int | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights, manifest = pack_all_full_layer_weights(
        model,
        expected_per_layer_numel=expected_per_layer_numel,
    )
    bin_path = output_dir / PACKED_WEIGHTS_BIN
    manifest_path = output_dir / PACKED_WEIGHTS_MANIFEST
    tmp_bin = output_dir / f"{PACKED_WEIGHTS_BIN}.tmp"
    tmp_manifest = output_dir / f"{PACKED_WEIGHTS_MANIFEST}.tmp"
    _bf16_to_file(weights, tmp_bin)
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_bin.replace(bin_path)
    tmp_manifest.replace(manifest_path)
    validate_packed_weight_artifact(
        model,
        output_dir,
        expected_per_layer_numel=expected_per_layer_numel,
    )
    return manifest


def load_packed_weights_manifest(packed_dir: Path) -> dict[str, object]:
    manifest_path = Path(packed_dir) / PACKED_WEIGHTS_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing packed weight manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def validate_packed_weights_manifest(
    model: Qwen3ForCausalLM,
    manifest: dict[str, object],
    expected_per_layer_numel: int | None = None,
) -> None:
    if manifest.get("format") != PACKED_WEIGHTS_FORMAT:
        raise RuntimeError(
            f"unsupported packed weight format {manifest.get('format')!r}; "
            f"expected {PACKED_WEIGHTS_FORMAT}"
        )
    if manifest.get("dtype") != "bfloat16":
        raise RuntimeError(
            f"packed weight dtype must be bfloat16: {manifest.get('dtype')}"
        )
    if manifest.get("weight_order") != list(FULL_LAYER_WEIGHT_ORDER):
        raise RuntimeError("packed weight order does not match the runtime layout")
    config = _config_dict(model)
    manifest_config = manifest.get("model_config", {})
    for key, value in config.items():
        if key in manifest_config and manifest_config[key] != value:
            raise RuntimeError(
                f"packed weight config mismatch for {key}: "
                f"{manifest_config[key]!r} != {value!r}"
            )
    if int(manifest.get("num_layers", -1)) != model.config.num_hidden_layers:
        raise RuntimeError(
            f"packed weight layer count {manifest.get('num_layers')} "
            f"!= model layer count {model.config.num_hidden_layers}"
        )
    per_layer_numel = int(manifest.get("per_layer_numel", -1))
    if expected_per_layer_numel is not None and (
        per_layer_numel != expected_per_layer_numel
    ):
        raise RuntimeError(
            f"packed per-layer size {per_layer_numel} "
            f"!= expected {expected_per_layer_numel}"
        )
    expected_total = per_layer_numel * model.config.num_hidden_layers
    if int(manifest.get("total_numel", -1)) != expected_total:
        raise RuntimeError(
            f"packed total elements {manifest.get('total_numel')} "
            f"!= expected {expected_total}"
        )
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != model.config.num_hidden_layers:
        raise RuntimeError("packed manifest layer table is missing or incomplete")
    for layer_idx, layer in enumerate(layers):
        offset = layer_idx * per_layer_numel
        if int(layer["id"]) != layer_idx:
            raise RuntimeError(f"packed layer id mismatch at index {layer_idx}")
        if int(layer["element_offset"]) != offset:
            raise RuntimeError(f"packed layer {layer_idx} element offset mismatch")
        if int(layer["numel"]) != per_layer_numel:
            raise RuntimeError(f"packed layer {layer_idx} size mismatch")
        if int(layer["byte_offset"]) != offset * 2:
            raise RuntimeError(f"packed layer {layer_idx} byte offset mismatch")
        if int(layer["byte_offset"]) % 64 != 0:
            raise RuntimeError(
                f"packed layer {layer_idx} byte offset is not 64B aligned"
            )


def validate_packed_weight_artifact(
    model: Qwen3ForCausalLM,
    packed_dir: Path,
    expected_per_layer_numel: int | None = None,
) -> dict[str, object]:
    packed_dir = Path(packed_dir)
    manifest = load_packed_weights_manifest(packed_dir)
    validate_packed_weights_manifest(
        model,
        manifest,
        expected_per_layer_numel=expected_per_layer_numel,
    )
    bin_path = packed_dir / PACKED_WEIGHTS_BIN
    if not bin_path.exists():
        raise FileNotFoundError(f"missing packed weight data file: {bin_path}")
    expected_bytes = int(manifest["total_bytes"])
    actual_bytes = bin_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"packed weight file size {actual_bytes} bytes != expected {expected_bytes}"
        )
    return manifest


def load_packed_weight_tensor(
    packed_dir: Path,
    manifest: dict[str, object] | None = None,
) -> torch.Tensor:
    packed_dir = Path(packed_dir)
    if manifest is None:
        manifest = load_packed_weights_manifest(packed_dir)
    return _bf16_from_file(
        packed_dir / PACKED_WEIGHTS_BIN,
        int(manifest["total_numel"]),
    )


def _manifest_segment_slice(
    packed_weights: torch.Tensor,
    manifest: dict[str, object],
    layer_idx: int,
    segment_name: str,
) -> torch.Tensor:
    layer = manifest["layers"][layer_idx]
    for segment in layer["segments"]:
        if segment["name"] == segment_name:
            start = int(segment["element_offset"])
            end = start + int(segment["numel"])
            return packed_weights[start:end]
    raise KeyError(f"missing segment {segment_name!r} in layer {layer_idx}")


def pack_segment_major_weight_chunk_from_layer_major(
    packed_weights: torch.Tensor,
    manifest: dict[str, object],
    layer_start: int,
    layer_count: int,
    mlp_columns: int = 1,
    mlp_gate_up_columns: int = 0,
    attention_columns: int = 1,
    mlp_gate_up_pair_rows: bool = False,
    mlp_gate_up_row_group: int = 4,
) -> torch.Tensor:
    """Build a segment-major chunk from the existing layer-major artifact."""
    if layer_count < 1:
        raise ValueError(f"layer_count must be positive, got {layer_count}")
    layers = range(layer_start, layer_start + layer_count)
    if mlp_gate_up_columns == 0:
        mlp_gate_up_columns = 2 if mlp_columns == 2 else 1

    def segment_for_layer(layer_idx: int, name: str) -> torch.Tensor:
        if name == "W_qk_norm":
            return torch.cat(
                [
                    _manifest_segment_slice(
                        packed_weights, manifest, layer_idx, "W_q_norm"
                    ),
                    _manifest_segment_slice(
                        packed_weights, manifest, layer_idx, "W_k_norm"
                    ),
                ]
            )
        if name == "post_norm_gate_up":
            return torch.cat(
                [
                    _manifest_segment_slice(
                        packed_weights, manifest, layer_idx, "post_norm_weight"
                    ),
                    _manifest_segment_slice(
                        packed_weights, manifest, layer_idx, "W_gate"
                    ),
                    _manifest_segment_slice(
                        packed_weights, manifest, layer_idx, "W_up"
                    ),
                ]
            )
        return _manifest_segment_slice(packed_weights, manifest, layer_idx, name)

    if attention_columns == 1 and mlp_columns == 1:
        return torch.cat(
            [
                segment_for_layer(layer_idx, name)
                for name in SEGMENT_MAJOR_WEIGHT_ORDER
                for layer_idx in layers
            ]
        ).contiguous()
    if attention_columns == 2:

        def row_shard(layer_idx: int, name: str, col: int) -> torch.Tensor:
            hidden_size = segment_for_layer(layer_idx, "post_norm_weight").numel()
            matrix = segment_for_layer(layer_idx, name).view(-1, hidden_size)
            shard_rows = matrix.shape[0] // 2
            start = col * shard_rows
            end = start + shard_rows
            return matrix[start:end].flatten()

        def qk_interleaved_shard(layer_idx: int, col: int) -> torch.Tensor:
            hidden_size = segment_for_layer(layer_idx, "post_norm_weight").numel()
            head_dim = segment_for_layer(layer_idx, "W_qk_norm").numel() // 2
            q_matrix = segment_for_layer(layer_idx, "W_q").view(-1, hidden_size)
            k_matrix = segment_for_layer(layer_idx, "W_k").view(-1, hidden_size)
            q_repeat = q_matrix.shape[0] // k_matrix.shape[0]
            kv_heads_per_col = (k_matrix.shape[0] // head_dim) // 2
            q_head_start = col * kv_heads_per_col * q_repeat
            k_head_start = col * kv_heads_per_col
            pieces = []
            for local_kv_head in range(kv_heads_per_col):
                k_head = k_head_start + local_kv_head
                q_head = q_head_start + local_kv_head * q_repeat
                pieces.append(
                    k_matrix[k_head * head_dim : (k_head + 1) * head_dim].flatten()
                )
                pieces.append(
                    q_matrix[
                        q_head * head_dim : (q_head + q_repeat) * head_dim
                    ].flatten()
                )
            return torch.cat(pieces)

        def o_context_shard(layer_idx: int, col: int) -> torch.Tensor:
            hidden_size = segment_for_layer(layer_idx, "post_norm_weight").numel()
            matrix = segment_for_layer(layer_idx, "W_o").view(hidden_size, -1)
            shard_cols = matrix.shape[1] // 2
            start = col * shard_cols
            end = start + shard_cols
            return matrix[:, start:end].contiguous().flatten()

        def gate_up_shard(layer_idx: int, col: int) -> torch.Tensor:
            hidden_size = segment_for_layer(layer_idx, "post_norm_weight").numel()
            gate = segment_for_layer(layer_idx, "W_gate").view(-1, hidden_size)
            up = segment_for_layer(layer_idx, "W_up").view(-1, hidden_size)
            row_group = 4 if col == 0 else mlp_gate_up_row_group
            return _gate_up_shard(
                gate,
                up,
                col,
                columns=mlp_gate_up_columns,
                pair_rows=mlp_gate_up_pair_rows,
                row_group=row_group,
            )

        def down_shard(layer_idx: int, col: int) -> torch.Tensor:
            hidden_size = segment_for_layer(layer_idx, "post_norm_weight").numel()
            down = segment_for_layer(layer_idx, "W_down").view(hidden_size, -1)
            shard_rows = down.shape[0] // 2
            start = col * shard_rows
            end = start + shard_rows
            return down[start:end].flatten()

        input_norm = [
            segment_for_layer(layer_idx, "input_norm_weight") for layer_idx in layers
        ]
        qkv_shards = []
        for col in range(2):
            qkv_shards.extend(
                qk_interleaved_shard(layer_idx, col) for layer_idx in layers
            )
            qkv_shards.extend(row_shard(layer_idx, "W_v", col) for layer_idx in layers)
        qk_norm = [segment_for_layer(layer_idx, "W_qk_norm") for layer_idx in layers]
        o_shards = [
            o_context_shard(layer_idx, col) for col in range(2) for layer_idx in layers
        ]
        if mlp_columns == 1:
            if mlp_gate_up_columns != 1:
                raise ValueError("mlp_columns=1 requires mlp_gate_up_columns=1")
            mlp = [
                segment_for_layer(layer_idx, name)
                for name in ("post_norm_gate_up", "W_down")
                for layer_idx in layers
            ]
            return torch.cat(
                input_norm + qkv_shards + qk_norm + o_shards + mlp
            ).contiguous()
        if mlp_columns != 2:
            raise ValueError("unsupported attention2 MLP column layout")
        if mlp_gate_up_columns == 1:
            mlp = [
                segment_for_layer(layer_idx, name)
                for name in ("post_norm_gate_up", "W_down")
                for layer_idx in layers
            ]
            return torch.cat(
                input_norm + qkv_shards + qk_norm + o_shards + mlp
            ).contiguous()
        if mlp_gate_up_columns not in {2, 3}:
            raise ValueError("unsupported attention2 gate/up column layout")
        gate_up_col0 = [
            torch.cat(
                [
                    segment_for_layer(layer_idx, "post_norm_weight"),
                    gate_up_shard(layer_idx, 0),
                ]
            )
            for layer_idx in layers
        ]
        if mlp_gate_up_columns == 3:
            gate_up_rest = [
                _interleave_same_size_chunks(
                    [
                        gate_up_shard(layer_idx, 1),
                        gate_up_shard(layer_idx, 2),
                    ],
                    chunk_size=segment_for_layer(layer_idx, "post_norm_weight").numel(),
                )
                for layer_idx in layers
            ]
        else:
            gate_up_rest = [
                gate_up_shard(layer_idx, col)
                for col in range(1, mlp_gate_up_columns)
                for layer_idx in layers
            ]
        down = [down_shard(layer_idx, col) for col in range(2) for layer_idx in layers]
        return torch.cat(
            input_norm
            + qkv_shards
            + qk_norm
            + o_shards
            + gate_up_col0
            + gate_up_rest
            + down
        ).contiguous()
    if attention_columns != 1:
        raise ValueError(
            f"unsupported segment-major attention column layout: {attention_columns}"
        )
    if mlp_columns == 1:
        if mlp_gate_up_columns != 1:
            raise ValueError("mlp_columns=1 requires mlp_gate_up_columns=1")
        return torch.cat(
            [
                segment_for_layer(layer_idx, name)
                for name in SEGMENT_MAJOR_WEIGHT_ORDER
                for layer_idx in layers
            ]
        ).contiguous()
    if mlp_columns != 2:
        raise ValueError(f"unsupported segment-major MLP column layout: {mlp_columns}")
    if mlp_gate_up_columns == 1:
        return torch.cat(
            [
                segment_for_layer(layer_idx, name)
                for name in SEGMENT_MAJOR_WEIGHT_ORDER
                for layer_idx in layers
            ]
        ).contiguous()
    if mlp_gate_up_columns != 2:
        raise ValueError("unsupported MLP gate/up column layout")

    def gate_up_shard(layer_idx: int, col: int) -> torch.Tensor:
        hidden_size = segment_for_layer(layer_idx, "post_norm_weight").numel()
        gate = segment_for_layer(layer_idx, "W_gate").view(-1, hidden_size)
        up = segment_for_layer(layer_idx, "W_up").view(-1, hidden_size)
        return _gate_up_shard(
            gate,
            up,
            col,
            pair_rows=mlp_gate_up_pair_rows,
            row_group=mlp_gate_up_row_group,
        )

    def down_shard(layer_idx: int, col: int) -> torch.Tensor:
        hidden_size = segment_for_layer(layer_idx, "post_norm_weight").numel()
        down = segment_for_layer(layer_idx, "W_down").view(hidden_size, -1)
        shard_rows = down.shape[0] // 2
        start = col * shard_rows
        end = start + shard_rows
        return down[start:end].flatten()

    common = [
        segment_for_layer(layer_idx, name)
        for name in (
            "input_norm_weight",
            "W_q",
            "W_k",
            "W_v",
            "W_qk_norm",
            "W_o",
        )
        for layer_idx in layers
    ]
    post_norm = [
        segment_for_layer(layer_idx, "post_norm_weight") for layer_idx in layers
    ]
    gate_up = [
        gate_up_shard(layer_idx, col) for col in range(2) for layer_idx in layers
    ]
    down = [down_shard(layer_idx, col) for col in range(2) for layer_idx in layers]
    return torch.cat(common + post_norm + gate_up + down).contiguous()


def packed_weight_layer_slice(
    packed_weights: torch.Tensor,
    manifest: dict[str, object],
    layer_idx: int,
) -> torch.Tensor:
    layer = manifest["layers"][layer_idx]
    start = int(layer["element_offset"])
    end = start + int(layer["numel"])
    return packed_weights[start:end]
