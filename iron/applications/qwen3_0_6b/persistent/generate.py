#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch

from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from iron.applications.qwen3_0_6b.qwen3_cpu import Qwen3ForCausalLM
from iron.applications.qwen3_0_6b.persistent.layout import (
    host_owned_tensor,
    load_packed_weight_tensor,
    pack_full_layer_weights_for_layer,
    pack_layer_cache,
    pack_qk_rope_metadata_for_layers,
    pack_segment_major_weight_chunk_from_layer_major,
    pack_segment_major_weights_for_layers,
    packed_weight_layer_slice,
    validate_packed_weight_artifact,
)
from iron.applications.qwen3_0_6b.persistent.refs import rope_lut_for_position
from iron.common.utils import XRTSubBuffer


@dataclass
class FastGenerateSetupTiming:
    weight_pack_s: float = 0.0
    weight_disk_load_s: float = 0.0
    weight_xrt_s: float = 0.0
    cache_xrt_s: float = 0.0
    weight_source: str = "runtime_pack"


@dataclass
class FastGenerateStepTiming:
    hidden_sync_s: float = 0.0
    rope_sync_s: float = 0.0
    op_call_s: float = 0.0
    output_drain_s: float = 0.0
    layer_residual_clone_s: float = 0.0


@dataclass
class FastGenerateBuffers:
    chunk_weight_bufs: list[XRTTensor]
    chunk_cache_bufs: list[XRTTensor]
    hidden_buf: XRTTensor
    rope_angles_buf: XRTTensor
    chunk_outputs_buf: XRTTensor
    timing: FastGenerateSetupTiming
    chunk_hidden_bufs: list[XRTTensor] | None = None
    chunk_rope_bufs: list[XRTTensor] | None = None
    weight_parent_buf: XRTTensor | None = None
    packed_weight_manifest: dict[str, object] | None = None


def copy_tensor_to_xrt(buffer: XRTTensor, tensor: torch.Tensor) -> float:
    start = time.perf_counter()
    view = buffer.torch_view()
    view.copy_(tensor.reshape(view.shape))
    buffer.to("npu")
    return time.perf_counter() - start


def pack_attention2_runtime_inputs(
    model: Qwen3ForCausalLM,
    current_hidden: torch.Tensor,
    position: int,
    layer_start: int,
    chunk_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope_angles = rope_lut_for_position(
        model.config.head_dim,
        model.config.rope_theta,
        position,
    )
    metadata = pack_qk_rope_metadata_for_layers(
        [
            {
                "W_q_norm": model.w(
                    f"model.layers.{layer_idx}.self_attn.q_norm.weight"
                ),
                "W_k_norm": model.w(
                    f"model.layers.{layer_idx}.self_attn.k_norm.weight"
                ),
                "rope_angles": rope_angles,
            }
            for layer_idx in range(layer_start, layer_start + chunk_len)
        ],
        position=None,
    )
    metadata_rows = metadata.view(chunk_len, -1)
    runtime_hidden = torch.cat(
        [
            torch.cat([current_hidden.flatten(), metadata_rows[layer_idx]])
            for layer_idx in range(chunk_len)
        ]
    ).contiguous()
    return runtime_hidden, metadata


def prepare_fast_generate_buffers(
    model: Qwen3ForCausalLM,
    state,
    op,
    packed_weights_dir: Path | None = None,
    require_packed_weights: bool = False,
    layer_chunk_size: int = 1,
) -> FastGenerateBuffers:
    if layer_chunk_size < 1:
        raise ValueError(f"layer_chunk_size must be positive, got {layer_chunk_size}")

    timing = FastGenerateSetupTiming()
    mlp_columns = op.num_aie_columns if getattr(op, "num_aie_columns", 1) == 2 else 1
    mlp_gate_up_columns = getattr(op, "effective_mlp_gate_up_columns", 0)
    mlp_gate_up_pair_rows = getattr(op, "mlp_gate_up_pair_rows", False)
    mlp_gate_up_row_group = getattr(op, "mlp_gate_up_row_group", 4)
    attention_columns = getattr(op, "attention_columns", 1)

    weight_parent_buf = None
    packed_weight_manifest = None
    chunk_weight_bufs: list[XRTTensor]
    if packed_weights_dir is not None and Path(packed_weights_dir).exists():
        packed_weights_dir = Path(packed_weights_dir)
        start = time.perf_counter()
        packed_weight_manifest = validate_packed_weight_artifact(
            model,
            packed_weights_dir,
            expected_per_layer_numel=op.packed_weights_size,
        )
        packed_weights = load_packed_weight_tensor(
            packed_weights_dir,
            packed_weight_manifest,
        )
        timing.weight_disk_load_s = time.perf_counter() - start

        start = time.perf_counter()
        weight_parent_buf = XRTTensor.from_torch(packed_weights)
        chunk_weight_bufs = []
        for layer_idx in range(0, model.config.num_hidden_layers, layer_chunk_size):
            chunk_len = min(
                layer_chunk_size,
                model.config.num_hidden_layers - layer_idx,
            )
            layer_slice = packed_weight_layer_slice(
                packed_weights,
                packed_weight_manifest,
                layer_idx,
            )
            if layer_slice.numel() != op.packed_weights_size:
                raise RuntimeError(
                    f"layer {layer_idx} packed artifact slice size "
                    f"{layer_slice.numel()} != op.packed_weights_size "
                    f"{op.packed_weights_size}"
                )
            if chunk_len == 1 and mlp_columns == 1:
                layer = packed_weight_manifest["layers"][layer_idx]
                chunk_weight_bufs.append(
                    XRTSubBuffer.from_parent(
                        weight_parent_buf,
                        (op.packed_weights_size,),
                        int(layer["element_offset"]),
                        op.packed_weights_size,
                        weight_parent_buf.dtype,
                    )
                )
            else:
                chunk_weight_bufs.append(
                    XRTTensor.from_torch(
                        pack_segment_major_weight_chunk_from_layer_major(
                            packed_weights,
                            packed_weight_manifest,
                            layer_idx,
                            chunk_len,
                            mlp_columns=mlp_columns,
                            mlp_gate_up_columns=mlp_gate_up_columns,
                            attention_columns=attention_columns,
                            mlp_gate_up_pair_rows=mlp_gate_up_pair_rows,
                            mlp_gate_up_row_group=mlp_gate_up_row_group,
                        )
                    )
                )
        timing.weight_xrt_s = time.perf_counter() - start
        timing.weight_source = "packed_artifact"
    else:
        if require_packed_weights:
            raise FileNotFoundError(
                f"packed weights required but not found at {packed_weights_dir}"
            )
        start = time.perf_counter()
        packed_weights_by_layer = [
            pack_full_layer_weights_for_layer(model, layer_idx)
            for layer_idx in range(model.config.num_hidden_layers)
        ]
        for layer_idx, packed_weights in enumerate(packed_weights_by_layer):
            if packed_weights.numel() != op.packed_weights_size:
                raise RuntimeError(
                    f"layer {layer_idx} packed weight size {packed_weights.numel()} "
                    f"!= op.packed_weights_size {op.packed_weights_size}"
                )
        timing.weight_pack_s = time.perf_counter() - start

        start = time.perf_counter()
        chunk_weight_bufs = []
        for layer_idx in range(0, model.config.num_hidden_layers, layer_chunk_size):
            chunk_len = min(
                layer_chunk_size,
                model.config.num_hidden_layers - layer_idx,
            )
            if chunk_len == 1 and mlp_columns == 1:
                chunk_weight_bufs.append(
                    XRTTensor.from_torch(packed_weights_by_layer[layer_idx])
                )
            else:
                chunk_weight_bufs.append(
                    XRTTensor.from_torch(
                        pack_segment_major_weights_for_layers(
                            model,
                            range(layer_idx, layer_idx + chunk_len),
                            mlp_columns=mlp_columns,
                            mlp_gate_up_columns=mlp_gate_up_columns,
                            attention_columns=attention_columns,
                            mlp_gate_up_pair_rows=mlp_gate_up_pair_rows,
                            mlp_gate_up_row_group=mlp_gate_up_row_group,
                        )
                    )
                )
        timing.weight_xrt_s = time.perf_counter() - start

    start = time.perf_counter()
    packed_cache_by_layer = [
        pack_layer_cache(state, layer_idx).clone()
        for layer_idx in range(model.config.num_hidden_layers)
    ]
    for layer_idx, packed_cache in enumerate(packed_cache_by_layer):
        if packed_cache.numel() != op.packed_cache_size:
            raise RuntimeError(
                f"layer {layer_idx} packed cache size {packed_cache.numel()} "
                f"!= op.packed_cache_size {op.packed_cache_size}"
            )
    chunk_cache_bufs = [
        XRTTensor.from_torch(
            torch.cat(
                packed_cache_by_layer[layer_idx : layer_idx + layer_chunk_size]
            ).contiguous()
        )
        for layer_idx in range(0, model.config.num_hidden_layers, layer_chunk_size)
    ]
    timing.cache_xrt_s = time.perf_counter() - start

    dtype = chunk_weight_bufs[0].dtype
    chunk_hidden_bufs = None
    chunk_rope_bufs = None
    fused_attention_runtime = (
        getattr(op, "runtime_hidden_size", model.config.hidden_size)
        != model.config.hidden_size
    )
    if attention_columns == 2:
        chunk_hidden_bufs = []
        chunk_rope_bufs = []
        for layer_idx in range(0, model.config.num_hidden_layers, layer_chunk_size):
            chunk_len = min(
                layer_chunk_size,
                model.config.num_hidden_layers - layer_idx,
            )
            qk_rope_metadata_size = getattr(
                op,
                "qk_rope_metadata_size",
                3 * model.config.head_dim,
            )
            runtime_hidden_numel = (
                chunk_len * (model.config.hidden_size + qk_rope_metadata_size)
                if fused_attention_runtime
                else model.config.hidden_size
            )
            chunk_hidden_bufs.append(
                XRTTensor(
                    (runtime_hidden_numel,),
                    dtype=dtype,
                )
            )
            chunk_rope_bufs.append(
                XRTTensor(
                    (chunk_len * qk_rope_metadata_size,),
                    dtype=dtype,
                )
            )
    return FastGenerateBuffers(
        chunk_weight_bufs=chunk_weight_bufs,
        chunk_cache_bufs=chunk_cache_bufs,
        hidden_buf=XRTTensor((model.config.hidden_size,), dtype=dtype),
        rope_angles_buf=XRTTensor((model.config.head_dim,), dtype=dtype),
        chunk_outputs_buf=XRTTensor((op.packed_outputs_size,), dtype=dtype),
        timing=timing,
        chunk_hidden_bufs=chunk_hidden_bufs,
        chunk_rope_bufs=chunk_rope_bufs,
        weight_parent_buf=weight_parent_buf,
        packed_weight_manifest=packed_weight_manifest,
    )


def run_n_layer_decode_hidden_fast(
    model: Qwen3ForCausalLM,
    token_id: int,
    position: int,
    layer_chunk_size: int,
    chunk_ops: dict[int, tuple[object, object]],
    fast_buffers: FastGenerateBuffers,
) -> tuple[torch.Tensor, float, FastGenerateStepTiming]:
    if layer_chunk_size < 1:
        raise ValueError(f"layer_chunk_size must be positive, got {layer_chunk_size}")
    for chunk_len, (op, _op_func) in chunk_ops.items():
        if position != op.position:
            raise RuntimeError(
                f"compiled position for chunk {chunk_len} is {op.position}, "
                f"decode position is {position}"
            )
        if chunk_len != op.layer_iterations:
            raise RuntimeError(
                f"chunk key {chunk_len} != op.layer_iterations {op.layer_iterations}"
            )

    current_hidden = host_owned_tensor(
        model.embed(torch.tensor([[token_id]], dtype=torch.long)).flatten()
    )
    hidden_buf = fast_buffers.hidden_buf
    rope_angles_buf = fast_buffers.rope_angles_buf
    chunk_outputs_buf = fast_buffers.chunk_outputs_buf
    chunk_weight_bufs = fast_buffers.chunk_weight_bufs
    chunk_cache_bufs = fast_buffers.chunk_cache_bufs
    chunk_hidden_bufs = fast_buffers.chunk_hidden_bufs
    chunk_rope_bufs = fast_buffers.chunk_rope_bufs
    attention_columns = getattr(
        next(iter(chunk_ops.values()))[0], "attention_columns", 1
    )

    total_npu_time = 0.0
    timing = FastGenerateStepTiming()
    if attention_columns == 1:
        rope_angles = rope_lut_for_position(
            model.config.head_dim,
            model.config.rope_theta,
            position,
        )
        timing.rope_sync_s += copy_tensor_to_xrt(rope_angles_buf, rope_angles)
    elif chunk_hidden_bufs is None or chunk_rope_bufs is None:
        raise RuntimeError("attention2 fast generate requires per-chunk buffers")
    chunk_idx = 0
    layer_idx = 0
    while layer_idx < model.config.num_hidden_layers:
        chunk_len = min(layer_chunk_size, model.config.num_hidden_layers - layer_idx)
        op, op_func = chunk_ops[chunk_len]
        if getattr(op, "attention_columns", 1) == 2:
            fused_attention_runtime = (
                getattr(op, "runtime_hidden_size", model.config.hidden_size)
                != model.config.hidden_size
            )
            packed_hidden, rope_metadata = pack_attention2_runtime_inputs(
                model,
                current_hidden,
                position,
                layer_idx,
                chunk_len,
            )
            runtime_hidden = (
                packed_hidden if fused_attention_runtime else current_hidden
            )
            active_hidden_buf = chunk_hidden_bufs[chunk_idx]
            active_rope_buf = chunk_rope_bufs[chunk_idx]
            timing.hidden_sync_s += copy_tensor_to_xrt(
                active_hidden_buf, runtime_hidden
            )
            if not fused_attention_runtime:
                timing.rope_sync_s += copy_tensor_to_xrt(active_rope_buf, rope_metadata)
        else:
            active_hidden_buf = hidden_buf
            active_rope_buf = rope_angles_buf
            timing.hidden_sync_s += copy_tensor_to_xrt(hidden_buf, current_hidden)

        chunk_outputs_buf.device = "npu"
        cache_buf = chunk_cache_bufs[chunk_idx]
        start = time.perf_counter()
        result = op_func(
            active_hidden_buf,
            chunk_weight_bufs[chunk_idx],
            active_rope_buf,
            chunk_outputs_buf,
            cache_buf,
        )
        timing.op_call_s += time.perf_counter() - start
        total_npu_time += result.npu_time

        chunk_outputs_buf.device = "npu"
        cache_buf.device = "npu"
        start = time.perf_counter()
        packed_outputs = chunk_outputs_buf.to_torch()
        timing.output_drain_s += time.perf_counter() - start

        start = time.perf_counter()
        current_hidden = host_owned_tensor(packed_outputs)
        timing.layer_residual_clone_s += time.perf_counter() - start
        layer_idx += chunk_len
        chunk_idx += 1

    return current_hidden, total_npu_time, timing
