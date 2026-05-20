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
    weight_parent_buf: XRTTensor | None = None
    packed_weight_manifest: dict[str, object] | None = None


def copy_tensor_to_xrt(buffer: XRTTensor, tensor: torch.Tensor) -> float:
    start = time.perf_counter()
    view = buffer.torch_view()
    view.copy_(tensor.reshape(view.shape))
    buffer.to("npu")
    return time.perf_counter() - start


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
            if chunk_len == 1:
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
            if chunk_len == 1:
                chunk_weight_bufs.append(
                    XRTTensor.from_torch(packed_weights_by_layer[layer_idx])
                )
            else:
                chunk_weight_bufs.append(
                    XRTTensor.from_torch(
                        pack_segment_major_weights_for_layers(
                            model,
                            range(layer_idx, layer_idx + chunk_len),
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
    return FastGenerateBuffers(
        chunk_weight_bufs=chunk_weight_bufs,
        chunk_cache_bufs=chunk_cache_bufs,
        hidden_buf=XRTTensor((model.config.hidden_size,), dtype=dtype),
        rope_angles_buf=XRTTensor((model.config.head_dim,), dtype=dtype),
        chunk_outputs_buf=XRTTensor((op.packed_outputs_size,), dtype=dtype),
        timing=timing,
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

    total_npu_time = 0.0
    timing = FastGenerateStepTiming()
    rope_angles = rope_lut_for_position(
        model.config.head_dim,
        model.config.rope_theta,
        position,
    )
    timing.rope_sync_s += copy_tensor_to_xrt(rope_angles_buf, rope_angles)
    chunk_idx = 0
    layer_idx = 0
    while layer_idx < model.config.num_hidden_layers:
        chunk_len = min(layer_chunk_size, model.config.num_hidden_layers - layer_idx)
        op, op_func = chunk_ops[chunk_len]
        timing.hidden_sync_s += copy_tensor_to_xrt(hidden_buf, current_hidden)

        chunk_outputs_buf.device = "npu"
        cache_buf = chunk_cache_bufs[chunk_idx]
        start = time.perf_counter()
        result = op_func(
            hidden_buf,
            chunk_weight_bufs[chunk_idx],
            rope_angles_buf,
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
