#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from iron.applications.qwen3_0_6b.qwen3_cpu import Qwen3ForCausalLM
from iron.applications.qwen3_0_6b.qwen3_persistent_layout import (
    host_owned_tensor,
    layer_residual_from_packed_output,
    pack_full_layer_weights_for_layer,
    pack_layer_cache,
)
from iron.applications.qwen3_0_6b.qwen3_persistent_refs import rope_lut_for_position


@dataclass
class FastGenerateSetupTiming:
    weight_pack_s: float = 0.0
    weight_xrt_s: float = 0.0
    cache_xrt_s: float = 0.0


@dataclass
class FastGenerateStepTiming:
    hidden_sync_s: float = 0.0
    rope_sync_s: float = 0.0
    op_call_s: float = 0.0
    output_drain_s: float = 0.0
    layer_residual_clone_s: float = 0.0


@dataclass
class FastGenerateBuffers:
    weight_bufs: list[XRTTensor]
    cache_bufs: list[XRTTensor]
    hidden_buf: XRTTensor
    rope_angles_buf: XRTTensor
    packed_outputs_buf: XRTTensor
    timing: FastGenerateSetupTiming


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
) -> FastGenerateBuffers:
    timing = FastGenerateSetupTiming()

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
    weight_bufs = [
        XRTTensor.from_torch(packed_weights)
        for packed_weights in packed_weights_by_layer
    ]
    timing.weight_xrt_s = time.perf_counter() - start

    start = time.perf_counter()
    cache_bufs = []
    for layer_idx in range(model.config.num_hidden_layers):
        packed_cache = pack_layer_cache(state, layer_idx).clone()
        if packed_cache.numel() != op.packed_cache_size:
            raise RuntimeError(
                f"layer {layer_idx} packed cache size {packed_cache.numel()} "
                f"!= op.packed_cache_size {op.packed_cache_size}"
            )
        cache_bufs.append(XRTTensor.from_torch(packed_cache))
    timing.cache_xrt_s = time.perf_counter() - start

    dtype = weight_bufs[0].dtype
    return FastGenerateBuffers(
        weight_bufs=weight_bufs,
        cache_bufs=cache_bufs,
        hidden_buf=XRTTensor((model.config.hidden_size,), dtype=dtype),
        rope_angles_buf=XRTTensor((model.config.head_dim,), dtype=dtype),
        packed_outputs_buf=XRTTensor((op.packed_outputs_size,), dtype=dtype),
        timing=timing,
    )


def run_full_layer_decode_hidden_fast(
    model: Qwen3ForCausalLM,
    token_id: int,
    position: int,
    op,
    op_func,
    fast_buffers: FastGenerateBuffers,
) -> tuple[torch.Tensor, float, FastGenerateStepTiming]:
    if position != op.position:
        raise RuntimeError(f"compiled position {op.position} != decode {position}")

    current_hidden = host_owned_tensor(
        model.embed(torch.tensor([[token_id]], dtype=torch.long)).flatten()
    )
    hidden_buf = fast_buffers.hidden_buf
    rope_angles_buf = fast_buffers.rope_angles_buf
    packed_outputs_buf = fast_buffers.packed_outputs_buf
    weight_bufs = fast_buffers.weight_bufs
    cache_bufs = fast_buffers.cache_bufs

    total_npu_time = 0.0
    timing = FastGenerateStepTiming()
    rope_angles = rope_lut_for_position(
        model.config.head_dim,
        model.config.rope_theta,
        position,
    )
    for layer_idx in range(model.config.num_hidden_layers):
        timing.hidden_sync_s += copy_tensor_to_xrt(hidden_buf, current_hidden)
        timing.rope_sync_s += copy_tensor_to_xrt(rope_angles_buf, rope_angles)

        packed_outputs_buf.device = "npu"
        cache_buf = cache_bufs[layer_idx]
        start = time.perf_counter()
        result = op_func(
            hidden_buf,
            weight_bufs[layer_idx],
            rope_angles_buf,
            packed_outputs_buf,
            cache_buf,
        )
        timing.op_call_s += time.perf_counter() - start
        total_npu_time += result.npu_time

        packed_outputs_buf.device = "npu"
        cache_buf.device = "npu"
        start = time.perf_counter()
        packed_outputs = packed_outputs_buf.to_torch()
        timing.output_drain_s += time.perf_counter() - start

        start = time.perf_counter()
        current_hidden = host_owned_tensor(
            layer_residual_from_packed_output(op, packed_outputs)
        )
        timing.layer_residual_clone_s += time.perf_counter() - start

    return current_hidden, total_npu_time, timing
