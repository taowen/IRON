# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def phase_owned_decode(
    dev,
    num_lanes,
    num_layers,
    phase_packets_per_layer,
    packet_elements,
    hidden_size,
    attention_size,
    head_dim,
    intermediate_size,
    max_seq_len,
    attention_chunk_size,
    q_rows_per_packet,
    fabric_group_size,
    kernel_object="phase_owned_kernels.o",
):
    if num_lanes <= 0:
        raise ValueError("num_lanes must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if phase_packets_per_layer <= 0:
        raise ValueError("phase_packets_per_layer must be positive")
    if packet_elements <= 0:
        raise ValueError("packet_elements must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if attention_size <= 0:
        raise ValueError("attention_size must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if attention_size % head_dim != 0:
        raise ValueError("attention_size must be divisible by head_dim")
    attention_head_count = attention_size // head_dim
    if num_lanes > attention_head_count:
        raise ValueError(
            "production currently maps one attention context head per lane"
        )
    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if attention_chunk_size <= 0:
        raise ValueError("attention_chunk_size must be positive")
    if max_seq_len % attention_chunk_size != 0:
        raise ValueError("max_seq_len must be divisible by attention_chunk_size")
    attention_chunk_count = max_seq_len // attention_chunk_size
    if attention_chunk_count != 4:
        raise ValueError("production currently expects exactly four attention chunks")
    if head_dim != 128:
        raise ValueError(
            "production attention score/PV kernel currently expects head_dim=128"
        )
    if q_rows_per_packet <= 0:
        raise ValueError("q_rows_per_packet must be positive")
    if q_rows_per_packet > 16:
        raise ValueError("q_rows_per_packet must be <= 16 for local FFN handoff")
    if num_lanes * q_rows_per_packet > head_dim:
        raise ValueError("current q/k norm+RoPE shard supports only the first head")
    if fabric_group_size <= 0:
        raise ValueError("fabric_group_size must be positive")
    if num_lanes % fabric_group_size != 0:
        raise ValueError("num_lanes must be divisible by fabric_group_size")
    fabric_group_count = num_lanes // fabric_group_size
    if fabric_group_count != 2:
        raise ValueError(
            "production O cross-group partial reduce currently expects two fabric groups"
        )
    o_projection_chunk_rows = num_lanes * q_rows_per_packet
    if hidden_size % o_projection_chunk_rows != 0:
        raise ValueError("hidden_size must be divisible by O projection chunk rows")
    o_projection_chunk_count = hidden_size // o_projection_chunk_rows
    expected_phase_packets = (
        1 + 4 + 2 * attention_chunk_count + o_projection_chunk_count + 3
    )
    if phase_packets_per_layer != expected_phase_packets:
        raise ValueError(
            "phase_packets_per_layer must match the production phase-owned body "
            f"({expected_phase_packets})"
        )
    q_phase_elements = (2 + q_rows_per_packet) * hidden_size
    gate_up_elements = 1 + (2 + 2 * q_rows_per_packet) * hidden_size
    o_elements = 2 + o_projection_chunk_rows + o_projection_chunk_rows * 2 * head_dim
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
    minimum_packet_elements = max(
        o_elements,
        q_phase_elements,
        gate_up_elements,
        down_elements,
        norm_rope_elements,
        attention_chunk_elements,
    )
    if packet_elements < minimum_packet_elements:
        raise ValueError(
            "packet_elements must hold o_proj, gate/up, down, q/k RoPE, "
            "and attention score/PV phase payloads"
        )

    dtype = bfloat16
    total_phase_packets = num_layers * phase_packets_per_layer
    shared_packet_elements = 2 * hidden_size
    input_elements = num_lanes * total_phase_packets * packet_elements
    shared_input_elements = num_layers * shared_packet_elements
    q_output_values_per_lane = ((q_rows_per_packet + 7) // 8) * 8
    k_output_values_per_lane = ((q_rows_per_packet + 7) // 8) * 8
    v_output_values_per_lane = ((q_rows_per_packet + 7) // 8) * 8
    q_rope_output_values_per_lane = ((q_rows_per_packet + 7) // 8) * 8
    k_rope_output_values_per_lane = ((q_rows_per_packet + 7) // 8) * 8
    context_output_values_per_lane = 2 * head_dim
    attention_output_values_per_lane = ((hidden_size + 7) // 8) * 8
    gate_up_output_values_per_lane = ((2 * q_rows_per_packet + 7) // 8) * 8
    residual_output_values_per_lane = ((q_rows_per_packet + 7) // 8) * 8
    output_values_per_lane = (
        q_output_values_per_lane
        + k_output_values_per_lane
        + v_output_values_per_lane
        + q_rope_output_values_per_lane
        + k_rope_output_values_per_lane
        + context_output_values_per_lane
        + attention_output_values_per_lane
        + gate_up_output_values_per_lane
        + residual_output_values_per_lane
    )
    output_values_per_layer = num_lanes * output_values_per_lane
    output_values_per_group = fabric_group_size * output_values_per_lane
    output_elements = num_layers * output_values_per_layer

    shared_l3_ty = np.ndarray[(shared_input_elements,), np.dtype[dtype]]
    packets_l3_ty = np.ndarray[(input_elements,), np.dtype[dtype]]
    outputs_l3_ty = np.ndarray[(output_elements,), np.dtype[dtype]]
    shared_packet_ty = np.ndarray[(shared_packet_elements,), np.dtype[dtype]]
    packet_ty = np.ndarray[(packet_elements,), np.dtype[dtype]]
    q_join_ty = np.ndarray[(output_values_per_group,), np.dtype[dtype]]
    lane_output_ty = np.ndarray[(output_values_per_lane,), np.dtype[dtype]]
    o_target_rows = o_projection_chunk_rows
    o_partial_values_per_lane = o_target_rows
    o_partial_values_per_group = fabric_group_size * o_partial_values_per_lane
    o_partial_ty = np.ndarray[(o_partial_values_per_lane,), np.dtype[np.float32]]
    o_partial_join_ty = np.ndarray[(o_partial_values_per_group,), np.dtype[np.float32]]
    o_source_target_ty = np.ndarray[(o_target_rows,), np.dtype[np.float32]]
    o_target_reduced_group_ty = np.ndarray[(o_target_rows,), np.dtype[np.float32]]
    hidden_state_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    state_ty = np.ndarray[(1,), np.dtype[np.float32]]
    attention_state_ty = np.ndarray[(2,), np.dtype[np.float32]]
    attention_acc_ty = np.ndarray[(head_dim,), np.dtype[np.float32]]

    init_kernel = Kernel(
        "new_mega_phase_state_init_f32",
        kernel_object,
        [state_ty],
    )
    q_shard_kernel = Kernel(
        "new_mega_phase0_q_shard_bf16",
        kernel_object,
        [
            packet_ty,
            hidden_state_ty,
            hidden_state_ty,
            state_ty,
            lane_output_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    gate_up_kernel = Kernel(
        "new_mega_phase_gate_up_shard_bf16",
        kernel_object,
        [
            packet_ty,
            state_ty,
            lane_output_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    projection_kernel = Kernel(
        "new_mega_phase_projection_shard_bf16",
        kernel_object,
        [
            packet_ty,
            hidden_state_ty,
            hidden_state_ty,
            state_ty,
            lane_output_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    norm_rope_kernel = Kernel(
        "new_mega_phase_norm_rope_shard_bf16",
        kernel_object,
        [
            packet_ty,
            state_ty,
            lane_output_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    attention_init_kernel = Kernel(
        "new_mega_phase_attention_init_f32",
        kernel_object,
        [attention_state_ty, attention_acc_ty, np.int32],
    )
    attention_update_kernel = Kernel(
        "new_mega_phase_attention_update_packed_bf16",
        kernel_object,
        [
            packet_ty,
            attention_state_ty,
            attention_acc_ty,
            np.int32,
            np.int32,
        ],
    )
    attention_finalize_kernel = Kernel(
        "new_mega_phase_attention_finalize_bf16",
        kernel_object,
        [
            attention_state_ty,
            attention_acc_ty,
            state_ty,
            lane_output_ty,
            np.int32,
            np.int32,
        ],
    )
    o_partial_kernel = Kernel(
        "new_mega_phase_o_partial_shard_bf16",
        kernel_object,
        [
            packet_ty,
            state_ty,
            lane_output_ty,
            o_partial_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    o_source_reduce_kernel = Kernel(
        "new_mega_phase_o_source_reduce_targets_f32",
        kernel_object,
        [
            o_partial_join_ty,
            o_source_target_ty,
            o_source_target_ty,
            np.int32,
            np.int32,
        ],
    )
    o_target_reduce_kernel = Kernel(
        "new_mega_phase_o_reduce_two_sources_f32",
        kernel_object,
        [
            o_source_target_ty,
            o_source_target_ty,
            o_target_reduced_group_ty,
            np.int32,
        ],
    )
    o_finalize_kernel = Kernel(
        "new_mega_phase_o_finalize_reduced_bf16",
        kernel_object,
        [
            packet_ty,
            o_target_reduced_group_ty,
            state_ty,
            lane_output_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    down_kernel = Kernel(
        "new_mega_phase_down_residual_shard_bf16",
        kernel_object,
        [
            packet_ty,
            state_ty,
            lane_output_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    next_hidden_kernel = Kernel(
        "new_mega_phase_next_hidden_bf16",
        kernel_object,
        [packet_ty, hidden_state_ty, state_ty, np.int32, np.int32, np.int32],
    )

    packet_fifos = [
        ObjectFifo(packet_ty, name=f"new_mega_phase_lane_{lane}_packets", depth=1)
        for lane in range(num_lanes)
    ]
    q_join_fifos = [
        ObjectFifo(
            q_join_ty,
            name=f"new_mega_phase_q_joined_g{group}",
            depth=1,
        )
        for group in range(fabric_group_count)
    ]
    lane_output_fifos = []
    for group in range(fabric_group_count):
        group_fifos = (
            q_join_fifos[group]
            .prod()
            .join(
                offsets=[
                    lane_in_group * output_values_per_lane
                    for lane_in_group in range(fabric_group_size)
                ],
                obj_types=[lane_output_ty] * fabric_group_size,
                names=[
                    "new_mega_phase_lane_"
                    f"{group * fabric_group_size + lane_in_group}_output"
                    for lane_in_group in range(fabric_group_size)
                ],
                depths=[1] * fabric_group_size,
            )
        )
        lane_output_fifos.extend(group_fifos)

    o_partial_join_fifos = [
        ObjectFifo(
            o_partial_join_ty,
            name=f"new_mega_phase_o_partial_joined_g{group}",
            depth=1,
        )
        for group in range(fabric_group_count)
    ]
    o_partial_lane_fifos = []
    for group in range(fabric_group_count):
        group_fifos = (
            o_partial_join_fifos[group]
            .prod()
            .join(
                offsets=[
                    lane_in_group * o_partial_values_per_lane
                    for lane_in_group in range(fabric_group_size)
                ],
                obj_types=[o_partial_ty] * fabric_group_size,
                names=[
                    "new_mega_phase_lane_"
                    f"{group * fabric_group_size + lane_in_group}_o_partial"
                    for lane_in_group in range(fabric_group_size)
                ],
                depths=[1] * fabric_group_size,
            )
        )
        o_partial_lane_fifos.extend(group_fifos)
    o_source_target_fifos = [
        [
            ObjectFifo(
                o_source_target_ty,
                name=(
                    "new_mega_phase_o_source_"
                    f"{source_group}_to_target_{target_group}"
                ),
                depth=1,
            )
            for target_group in range(fabric_group_count)
        ]
        for source_group in range(fabric_group_count)
    ]
    o_target_reduced_group_fifos = [
        ObjectFifo(
            o_target_reduced_group_ty,
            name=f"new_mega_phase_o_target_reduced_g{group}",
            depth=1,
        )
        for group in range(fabric_group_count)
    ]
    o_reduced_group_fifos = [
        o_target_reduced_group_fifos[target_group]
        .cons()
        .forward(
            name=f"new_mega_phase_o_target_reduced_broadcast_g{target_group}",
            depth=1,
        )
        for target_group in range(fabric_group_count)
    ]
    states = [
        Buffer(
            initial_value=np.zeros(shape=(1,), dtype=np.float32),
            name=f"new_mega_phase_lane_{lane}_state",
        )
        for lane in range(num_lanes)
    ]
    hidden_states = [
        Buffer(
            initial_value=np.zeros(shape=(hidden_size,), dtype=dtype),
            name=f"new_mega_phase_lane_{lane}_hidden_state",
        )
        for lane in range(num_lanes)
    ]
    norm_weight_states = [
        Buffer(
            initial_value=np.zeros(shape=(hidden_size,), dtype=dtype),
            name=f"new_mega_phase_lane_{lane}_norm_weight_state",
        )
        for lane in range(num_lanes)
    ]
    attention_states = [
        Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.float32),
            name=f"new_mega_phase_lane_{lane}_attention_state",
        )
        for lane in range(num_lanes)
    ]
    attention_accs = [
        Buffer(
            initial_value=np.zeros(shape=(head_dim,), dtype=np.float32),
            name=f"new_mega_phase_lane_{lane}_attention_acc",
        )
        for lane in range(num_lanes)
    ]

    def lane_worker_body(
        packet_fifo,
        lane_output_fifo,
        o_partial_fifo,
        o_reduced_fifo,
        hidden_state,
        norm_weight_state,
        state,
        attention_state,
        attention_acc,
        init_kernel,
        q_shard_kernel,
        projection_kernel,
        norm_rope_kernel,
        attention_init_kernel,
        attention_update_kernel,
        attention_finalize_kernel,
        o_partial_kernel,
        o_finalize_kernel,
        gate_up_kernel,
        down_kernel,
        next_hidden_kernel,
    ):
        init_kernel(state)
        for layer in range_(num_layers):
            packet = packet_fifo.acquire(1)
            lane_output = lane_output_fifo.acquire(1)
            layer_i32 = index.casts(T.i32(), layer)
            q_shard_kernel(
                packet,
                hidden_state,
                norm_weight_state,
                state,
                lane_output,
                packet_elements,
                hidden_size,
                q_rows_per_packet,
                q_output_values_per_lane,
                layer_i32,
            )
            packet_fifo.release(1)

            packet = packet_fifo.acquire(1)
            projection_kernel(
                packet,
                hidden_state,
                norm_weight_state,
                state,
                lane_output,
                packet_elements,
                hidden_size,
                q_rows_per_packet,
                q_output_values_per_lane,
                k_output_values_per_lane,
            )
            packet_fifo.release(1)

            packet = packet_fifo.acquire(1)
            projection_kernel(
                packet,
                hidden_state,
                norm_weight_state,
                state,
                lane_output,
                packet_elements,
                hidden_size,
                q_rows_per_packet,
                q_output_values_per_lane + k_output_values_per_lane,
                v_output_values_per_lane,
            )
            packet_fifo.release(1)

            packet = packet_fifo.acquire(1)
            norm_rope_kernel(
                packet,
                state,
                lane_output,
                packet_elements,
                head_dim,
                q_rows_per_packet,
                q_output_values_per_lane
                + k_output_values_per_lane
                + v_output_values_per_lane,
                q_rope_output_values_per_lane,
            )
            packet_fifo.release(1)

            packet = packet_fifo.acquire(1)
            norm_rope_kernel(
                packet,
                state,
                lane_output,
                packet_elements,
                head_dim,
                q_rows_per_packet,
                q_output_values_per_lane
                + k_output_values_per_lane
                + v_output_values_per_lane
                + q_rope_output_values_per_lane,
                k_rope_output_values_per_lane,
            )
            packet_fifo.release(1)

            attention_init_kernel(attention_state, attention_acc, head_dim)
            for _ in range_(attention_chunk_count):
                packet = packet_fifo.acquire(1)
                attention_update_kernel(
                    packet,
                    attention_state,
                    attention_acc,
                    attention_chunk_size,
                    head_dim,
                )
                packet_fifo.release(1)
            attention_finalize_kernel(
                attention_state,
                attention_acc,
                state,
                lane_output,
                q_output_values_per_lane
                + k_output_values_per_lane
                + v_output_values_per_lane
                + q_rope_output_values_per_lane
                + k_rope_output_values_per_lane,
                head_dim,
            )

            attention_init_kernel(attention_state, attention_acc, head_dim)
            for _ in range_(attention_chunk_count):
                packet = packet_fifo.acquire(1)
                attention_update_kernel(
                    packet,
                    attention_state,
                    attention_acc,
                    attention_chunk_size,
                    head_dim,
                )
                packet_fifo.release(1)
            attention_finalize_kernel(
                attention_state,
                attention_acc,
                state,
                lane_output,
                q_output_values_per_lane
                + k_output_values_per_lane
                + v_output_values_per_lane
                + q_rope_output_values_per_lane
                + k_rope_output_values_per_lane
                + head_dim,
                head_dim,
            )

            for _ in range_(o_projection_chunk_count):
                packet = packet_fifo.acquire(1)
                o_partial = o_partial_fifo.acquire(1)
                o_partial_kernel(
                    packet,
                    state,
                    lane_output,
                    o_partial,
                    packet_elements,
                    head_dim,
                    q_rows_per_packet,
                    num_lanes,
                    num_lanes,
                    q_output_values_per_lane
                    + k_output_values_per_lane
                    + v_output_values_per_lane
                    + q_rope_output_values_per_lane
                    + k_rope_output_values_per_lane,
                )
                o_partial_fifo.release(1)
                o_reduced = o_reduced_fifo.acquire(1)
                o_finalize_kernel(
                    packet,
                    o_reduced,
                    state,
                    lane_output,
                    packet_elements,
                    q_rows_per_packet,
                    num_lanes,
                    q_output_values_per_lane
                    + k_output_values_per_lane
                    + v_output_values_per_lane
                    + q_rope_output_values_per_lane
                    + k_rope_output_values_per_lane
                    + context_output_values_per_lane,
                    attention_output_values_per_lane,
                )
                o_reduced_fifo.release(1)
                packet_fifo.release(1)

            packet = packet_fifo.acquire(1)
            gate_up_kernel(
                packet,
                state,
                lane_output,
                packet_elements,
                hidden_size,
                q_rows_per_packet,
                hidden_size,
                q_output_values_per_lane
                + k_output_values_per_lane
                + v_output_values_per_lane
                + q_rope_output_values_per_lane
                + k_rope_output_values_per_lane
                + context_output_values_per_lane,
                q_output_values_per_lane
                + k_output_values_per_lane
                + v_output_values_per_lane
                + q_rope_output_values_per_lane
                + k_rope_output_values_per_lane
                + context_output_values_per_lane
                + attention_output_values_per_lane,
                output_values_per_lane,
            )
            packet_fifo.release(1)

            packet = packet_fifo.acquire(1)
            down_kernel(
                packet,
                state,
                lane_output,
                packet_elements,
                hidden_size,
                intermediate_size,
                q_rows_per_packet,
                q_output_values_per_lane
                + k_output_values_per_lane
                + v_output_values_per_lane
                + q_rope_output_values_per_lane
                + k_rope_output_values_per_lane
                + context_output_values_per_lane
                + attention_output_values_per_lane,
                q_output_values_per_lane
                + k_output_values_per_lane
                + v_output_values_per_lane
                + q_rope_output_values_per_lane
                + k_rope_output_values_per_lane
                + context_output_values_per_lane
                + attention_output_values_per_lane
                + gate_up_output_values_per_lane,
                output_values_per_lane,
            )
            packet_fifo.release(1)
            lane_output_fifo.release(1)

            packet = packet_fifo.acquire(1)
            layer_i32 = index.casts(T.i32(), layer)
            next_hidden_kernel(
                packet,
                hidden_state,
                state,
                packet_elements,
                hidden_size,
                layer_i32,
            )
            packet_fifo.release(1)

    def o_reduce_worker_body(
        o_partial_join_fifo,
        target0_fifo,
        target1_fifo,
        o_reduce_kernel,
    ):
        for _ in range_(num_layers):
            for _ in range_(o_projection_chunk_count):
                partials = o_partial_join_fifo.acquire(1)
                target0 = target0_fifo.acquire(1)
                target1 = target1_fifo.acquire(1)
                o_reduce_kernel(
                    partials,
                    target0,
                    target1,
                    fabric_group_size,
                    o_target_rows,
                )
                o_partial_join_fifo.release(1)
                target0_fifo.release(1)
                target1_fifo.release(1)

    def o_target_reduce_worker_body(
        source0_fifo,
        source1_fifo,
        target_reduced_fifo,
        o_target_reduce_kernel,
    ):
        for _ in range_(num_layers):
            for _ in range_(o_projection_chunk_count):
                source0 = source0_fifo.acquire(1)
                source1 = source1_fifo.acquire(1)
                reduced = target_reduced_fifo.acquire(1)
                o_target_reduce_kernel(
                    source0,
                    source1,
                    reduced,
                    o_target_rows,
                )
                source0_fifo.release(1)
                source1_fifo.release(1)
                target_reduced_fifo.release(1)

    workers = [
        Worker(
            lane_worker_body,
            [
                packet_fifos[lane].cons(),
                lane_output_fifos[lane].prod(),
                o_partial_lane_fifos[lane].prod(),
                o_reduced_group_fifos[lane // fabric_group_size].cons(),
                hidden_states[lane],
                norm_weight_states[lane],
                states[lane],
                attention_states[lane],
                attention_accs[lane],
                init_kernel,
                q_shard_kernel,
                projection_kernel,
                norm_rope_kernel,
                attention_init_kernel,
                attention_update_kernel,
                attention_finalize_kernel,
                o_partial_kernel,
                o_finalize_kernel,
                gate_up_kernel,
                down_kernel,
                next_hidden_kernel,
            ],
            stack_size=0xD00,
        )
        for lane in range(num_lanes)
    ]
    o_source_reduce_workers = [
        Worker(
            o_reduce_worker_body,
            [
                o_partial_join_fifos[group].cons(),
                o_source_target_fifos[group][0].prod(),
                o_source_target_fifos[group][1].prod(),
                o_source_reduce_kernel,
            ],
            stack_size=0x400,
        )
        for group in range(fabric_group_count)
    ]
    o_target_reduce_workers = [
        Worker(
            o_target_reduce_worker_body,
            [
                o_source_target_fifos[0][target_group].cons(),
                o_source_target_fifos[1][target_group].cons(),
                o_target_reduced_group_fifos[target_group].prod(),
                o_target_reduce_kernel,
            ],
            stack_size=0x400,
        )
        for target_group in range(fabric_group_count)
    ]

    packet_taps = [
        TensorAccessPattern(
            (input_elements,),
            lane * total_phase_packets * packet_elements,
            [1, 1, 1, total_phase_packets * packet_elements],
            [0, 0, 0, 1],
        )
        for lane in range(num_lanes)
    ]
    out_taps = [
        TensorAccessPattern(
            (output_elements,),
            group * fabric_group_size * output_values_per_lane,
            [num_layers, 1, 1, output_values_per_group],
            [output_values_per_layer, 0, 0, 1],
        )
        for group in range(fabric_group_count)
    ]

    rt = Runtime()
    with rt.sequence(
        shared_l3_ty,
        packets_l3_ty,
        outputs_l3_ty,
    ) as (shared_l3, packets_l3, outputs_l3):
        rt.start(*(workers + o_source_reduce_workers + o_target_reduce_workers))
        tg = rt.task_group()
        for lane in range(num_lanes):
            rt.fill(
                packet_fifos[lane].prod(),
                packets_l3,
                packet_taps[lane],
                task_group=tg,
            )
        for group in range(fabric_group_count):
            rt.drain(
                q_join_fifos[group].cons(),
                outputs_l3,
                out_taps[group],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
