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
    layer_residual_from_packed_output,
    load_packed_weight_tensor,
    pack_full_layer_weights_for_layer,
    pack_layer_cache,
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
    weight_bufs: list[XRTTensor]
    cache_bufs: list[XRTTensor]
    hidden_buf: XRTTensor
    rope_angles_buf: XRTTensor
    packed_outputs_buf: XRTTensor
    timing: FastGenerateSetupTiming
    weight_parent_buf: XRTTensor | None = None
    packed_weight_manifest: dict[str, object] | None = None
    weight_pair_bufs: list[XRTTensor] | None = None
    cache_pair_bufs: list[XRTTensor] | None = None
    two_layer_outputs_buf: XRTTensor | None = None


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
    two_layer_op=None,
) -> FastGenerateBuffers:
    if layer_chunk_size not in (1, 2):
        raise ValueError(f"layer_chunk_size must be 1 or 2, got {layer_chunk_size}")
    if layer_chunk_size == 2 and two_layer_op is None:
        raise ValueError("two_layer_op is required when layer_chunk_size=2")

    timing = FastGenerateSetupTiming()
    needs_single_layer_buffers = (
        layer_chunk_size == 1 or model.config.num_hidden_layers % 2 == 1
    )

    weight_parent_buf = None
    packed_weight_manifest = None
    weight_bufs: list[XRTTensor]
    weight_pair_bufs: list[XRTTensor] | None = None
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
        weight_bufs = []
        if needs_single_layer_buffers:
            for layer_idx in range(model.config.num_hidden_layers):
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
                layer = packed_weight_manifest["layers"][layer_idx]
                weight_bufs.append(
                    XRTSubBuffer.from_parent(
                        weight_parent_buf,
                        (op.packed_weights_size,),
                        int(layer["element_offset"]),
                        op.packed_weights_size,
                        weight_parent_buf.dtype,
                    )
                )
        if layer_chunk_size == 2:
            weight_pair_bufs = []
            for layer_idx in range(0, model.config.num_hidden_layers - 1, 2):
                layer = packed_weight_manifest["layers"][layer_idx]
                weight_pair_bufs.append(
                    XRTSubBuffer.from_parent(
                        weight_parent_buf,
                        (two_layer_op.packed_weight_pair_size,),
                        int(layer["element_offset"]),
                        two_layer_op.packed_weight_pair_size,
                        weight_parent_buf.dtype,
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
        weight_bufs = (
            [
                XRTTensor.from_torch(packed_weights)
                for packed_weights in packed_weights_by_layer
            ]
            if needs_single_layer_buffers
            else []
        )
        if layer_chunk_size == 2:
            weight_pair_bufs = [
                XRTTensor.from_torch(
                    torch.cat(
                        [
                            packed_weights_by_layer[layer_idx],
                            packed_weights_by_layer[layer_idx + 1],
                        ]
                    ).contiguous()
                )
                for layer_idx in range(0, model.config.num_hidden_layers - 1, 2)
            ]
        timing.weight_xrt_s = time.perf_counter() - start

    start = time.perf_counter()
    cache_bufs = []
    cache_pair_bufs = [] if layer_chunk_size == 2 else None
    if needs_single_layer_buffers:
        for layer_idx in range(model.config.num_hidden_layers):
            packed_cache = pack_layer_cache(state, layer_idx).clone()
            if packed_cache.numel() != op.packed_cache_size:
                raise RuntimeError(
                    f"layer {layer_idx} packed cache size {packed_cache.numel()} "
                    f"!= op.packed_cache_size {op.packed_cache_size}"
                )
            cache_bufs.append(XRTTensor.from_torch(packed_cache))
    if layer_chunk_size == 2:
        for layer_idx in range(0, model.config.num_hidden_layers - 1, 2):
            packed_cache_pair = torch.cat(
                [
                    pack_layer_cache(state, layer_idx).clone(),
                    pack_layer_cache(state, layer_idx + 1).clone(),
                ]
            ).contiguous()
            if packed_cache_pair.numel() != two_layer_op.packed_cache_pair_size:
                raise RuntimeError(
                    f"layer pair {layer_idx}/{layer_idx + 1} packed cache size "
                    f"{packed_cache_pair.numel()} != "
                    f"two_layer_op.packed_cache_pair_size "
                    f"{two_layer_op.packed_cache_pair_size}"
                )
            cache_pair_bufs.append(XRTTensor.from_torch(packed_cache_pair))
    timing.cache_xrt_s = time.perf_counter() - start

    dtype = weight_bufs[0].dtype if weight_bufs else weight_pair_bufs[0].dtype
    return FastGenerateBuffers(
        weight_bufs=weight_bufs,
        cache_bufs=cache_bufs,
        hidden_buf=XRTTensor((model.config.hidden_size,), dtype=dtype),
        rope_angles_buf=XRTTensor((model.config.head_dim,), dtype=dtype),
        packed_outputs_buf=XRTTensor((op.packed_outputs_size,), dtype=dtype),
        timing=timing,
        weight_parent_buf=weight_parent_buf,
        packed_weight_manifest=packed_weight_manifest,
        weight_pair_bufs=weight_pair_bufs,
        cache_pair_bufs=cache_pair_bufs,
        two_layer_outputs_buf=(
            XRTTensor((two_layer_op.packed_outputs_size,), dtype=dtype)
            if layer_chunk_size == 2
            else None
        ),
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


def run_full_layer_decode_hidden_fast_chunked(
    model: Qwen3ForCausalLM,
    token_id: int,
    position: int,
    single_op,
    single_op_func,
    two_layer_op,
    two_layer_op_func,
    fast_buffers: FastGenerateBuffers,
) -> tuple[torch.Tensor, float, FastGenerateStepTiming]:
    if position != two_layer_op.position or (
        single_op is not None and position != single_op.position
    ):
        single_pos = None if single_op is None else single_op.position
        raise RuntimeError(
            f"compiled positions single={single_pos} "
            f"two_layer={two_layer_op.position} != decode {position}"
        )
    if fast_buffers.weight_pair_bufs is None or fast_buffers.cache_pair_bufs is None:
        raise RuntimeError("fast buffers do not contain layer-pair buffers")
    if fast_buffers.two_layer_outputs_buf is None:
        raise RuntimeError("fast buffers do not contain a two-layer output buffer")

    current_hidden = host_owned_tensor(
        model.embed(torch.tensor([[token_id]], dtype=torch.long)).flatten()
    )
    hidden_buf = fast_buffers.hidden_buf
    rope_angles_buf = fast_buffers.rope_angles_buf
    single_outputs_buf = fast_buffers.packed_outputs_buf
    two_layer_outputs_buf = fast_buffers.two_layer_outputs_buf
    weight_bufs = fast_buffers.weight_bufs
    cache_bufs = fast_buffers.cache_bufs
    weight_pair_bufs = fast_buffers.weight_pair_bufs
    cache_pair_bufs = fast_buffers.cache_pair_bufs

    total_npu_time = 0.0
    timing = FastGenerateStepTiming()
    rope_angles = rope_lut_for_position(
        model.config.head_dim,
        model.config.rope_theta,
        position,
    )
    layer_idx = 0
    while layer_idx < model.config.num_hidden_layers:
        timing.hidden_sync_s += copy_tensor_to_xrt(hidden_buf, current_hidden)
        timing.rope_sync_s += copy_tensor_to_xrt(rope_angles_buf, rope_angles)

        if layer_idx + 1 < model.config.num_hidden_layers:
            pair_idx = layer_idx // 2
            two_layer_outputs_buf.device = "npu"
            cache_pair_buf = cache_pair_bufs[pair_idx]
            start = time.perf_counter()
            result = two_layer_op_func(
                hidden_buf,
                weight_pair_bufs[pair_idx],
                rope_angles_buf,
                two_layer_outputs_buf,
                cache_pair_buf,
            )
            timing.op_call_s += time.perf_counter() - start
            total_npu_time += result.npu_time

            two_layer_outputs_buf.device = "npu"
            cache_pair_buf.device = "npu"
            start = time.perf_counter()
            packed_outputs = two_layer_outputs_buf.to_torch()
            timing.output_drain_s += time.perf_counter() - start

            start = time.perf_counter()
            current_hidden = host_owned_tensor(packed_outputs)
            timing.layer_residual_clone_s += time.perf_counter() - start
            layer_idx += 2
        else:
            if single_op is None or single_op_func is None:
                raise RuntimeError("single-layer op is required for an odd layer count")
            single_outputs_buf.device = "npu"
            cache_buf = cache_bufs[layer_idx]
            start = time.perf_counter()
            result = single_op_func(
                hidden_buf,
                weight_bufs[layer_idx],
                rope_angles_buf,
                single_outputs_buf,
                cache_buf,
            )
            timing.op_call_s += time.perf_counter() - start
            total_npu_time += result.npu_time

            single_outputs_buf.device = "npu"
            cache_buf.device = "npu"
            start = time.perf_counter()
            packed_outputs = single_outputs_buf.to_torch()
            timing.output_drain_s += time.perf_counter() - start

            start = time.perf_counter()
            current_hidden = host_owned_tensor(
                layer_residual_from_packed_output(single_op, packed_outputs)
            )
            timing.layer_residual_clone_s += time.perf_counter() - start
            layer_idx += 1

    return current_hidden, total_npu_time, timing
