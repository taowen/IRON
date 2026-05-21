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
    intermediate_size,
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
    if phase_packets_per_layer < 2:
        raise ValueError(
            "phase_packets_per_layer must include a next_layer_token phase"
        )
    if phase_packets_per_layer < 9:
        raise ValueError("phase_packets_per_layer must include a gate_up phase")
    if phase_packets_per_layer < 10:
        raise ValueError("phase_packets_per_layer must include a down_proj phase")
    if packet_elements <= 0:
        raise ValueError("packet_elements must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if attention_size <= 0:
        raise ValueError("attention_size must be positive")
    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    if q_rows_per_packet <= 0:
        raise ValueError("q_rows_per_packet must be positive")
    if fabric_group_size <= 0:
        raise ValueError("fabric_group_size must be positive")
    if num_lanes % fabric_group_size != 0:
        raise ValueError("num_lanes must be divisible by fabric_group_size")
    gate_up_elements = (2 + 2 * q_rows_per_packet) * hidden_size
    o_elements = attention_size + q_rows_per_packet + q_rows_per_packet * attention_size
    down_elements = (
        intermediate_size + q_rows_per_packet + q_rows_per_packet * intermediate_size
    )
    minimum_packet_elements = max(o_elements, gate_up_elements, down_elements)
    if packet_elements < minimum_packet_elements:
        raise ValueError(
            "packet_elements must hold o_proj, gate/up, and down phase payloads"
        )

    dtype = bfloat16
    total_phase_packets = num_layers * phase_packets_per_layer
    shared_packet_elements = 2 * hidden_size
    input_elements = num_lanes * total_phase_packets * packet_elements
    shared_input_elements = num_layers * shared_packet_elements
    q_output_values_per_lane = ((q_rows_per_packet + 7) // 8) * 8
    k_output_values_per_lane = ((q_rows_per_packet + 7) // 8) * 8
    v_output_values_per_lane = ((q_rows_per_packet + 7) // 8) * 8
    attention_output_values_per_lane = ((q_rows_per_packet + 7) // 8) * 8
    gate_up_output_values_per_lane = ((2 * q_rows_per_packet + 7) // 8) * 8
    residual_output_values_per_lane = ((q_rows_per_packet + 7) // 8) * 8
    output_values_per_lane = (
        q_output_values_per_lane
        + k_output_values_per_lane
        + v_output_values_per_lane
        + attention_output_values_per_lane
        + gate_up_output_values_per_lane
        + residual_output_values_per_lane
    )
    output_values_per_layer = num_lanes * output_values_per_lane
    fabric_group_count = num_lanes // fabric_group_size
    output_values_per_group = fabric_group_size * output_values_per_lane
    output_elements = num_layers * output_values_per_layer

    shared_l3_ty = np.ndarray[(shared_input_elements,), np.dtype[dtype]]
    packets_l3_ty = np.ndarray[(input_elements,), np.dtype[dtype]]
    outputs_l3_ty = np.ndarray[(output_elements,), np.dtype[dtype]]
    shared_packet_ty = np.ndarray[(shared_packet_elements,), np.dtype[dtype]]
    packet_ty = np.ndarray[(packet_elements,), np.dtype[dtype]]
    q_join_ty = np.ndarray[(output_values_per_group,), np.dtype[dtype]]
    lane_output_ty = np.ndarray[(output_values_per_lane,), np.dtype[dtype]]
    hidden_state_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    state_ty = np.ndarray[(1,), np.dtype[np.float32]]

    init_kernel = Kernel(
        "new_mega_phase_state_init_f32",
        kernel_object,
        [state_ty],
    )
    q_shard_kernel = Kernel(
        "new_mega_phase0_q_shard_bf16",
        kernel_object,
        [
            shared_packet_ty,
            packet_ty,
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
        ],
    )
    projection_kernel = Kernel(
        "new_mega_phase_projection_shard_bf16",
        kernel_object,
        [
            shared_packet_ty,
            packet_ty,
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
    o_kernel = Kernel(
        "new_mega_phase_o_residual_shard_bf16",
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
        ],
    )
    packet_kernel = Kernel(
        "new_mega_phase_packet_accum_bf16",
        kernel_object,
        [packet_ty, state_ty, np.int32, np.int32, np.int32],
    )
    next_hidden_kernel = Kernel(
        "new_mega_phase_next_hidden_bf16",
        kernel_object,
        [packet_ty, hidden_state_ty, state_ty, np.int32, np.int32, np.int32],
    )

    shared_l3_fifos = [
        ObjectFifo(
            shared_packet_ty,
            name=f"new_mega_phase_shared_packets_l3l2_g{group}",
            depth=2,
        )
        for group in range(fabric_group_count)
    ]
    shared_fifos = [
        shared_l3_fifos[group]
        .cons()
        .forward(
            name=f"new_mega_phase_shared_packets_broadcast_g{group}",
            depth=2,
        )
        for group in range(fabric_group_count)
    ]
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

    def lane_worker_body(
        shared_fifo,
        packet_fifo,
        lane_output_fifo,
        hidden_state,
        state,
        init_kernel,
        q_shard_kernel,
        projection_kernel,
        o_kernel,
        gate_up_kernel,
        down_kernel,
        packet_kernel,
        next_hidden_kernel,
    ):
        init_kernel(state)
        for layer in range_(num_layers):
            shared = shared_fifo.acquire(1)
            packet = packet_fifo.acquire(1)
            lane_output = lane_output_fifo.acquire(1)
            layer_i32 = index.casts(T.i32(), layer)
            q_shard_kernel(
                shared,
                packet,
                hidden_state,
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
                shared,
                packet,
                hidden_state,
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
                shared,
                packet,
                hidden_state,
                state,
                lane_output,
                packet_elements,
                hidden_size,
                q_rows_per_packet,
                q_output_values_per_lane + k_output_values_per_lane,
                v_output_values_per_lane,
            )
            shared_fifo.release(1)
            packet_fifo.release(1)

            for phase_tail in range_(2):
                packet = packet_fifo.acquire(1)
                layer_i32 = index.casts(T.i32(), layer)
                phase_i32 = index.casts(T.i32(), phase_tail)
                packet_kernel(
                    packet,
                    state,
                    packet_elements,
                    layer_i32,
                    phase_i32,
                )
                packet_fifo.release(1)

            packet = packet_fifo.acquire(1)
            o_kernel(
                packet,
                state,
                lane_output,
                packet_elements,
                attention_size,
                q_rows_per_packet,
                q_output_values_per_lane
                + k_output_values_per_lane
                + v_output_values_per_lane,
                attention_output_values_per_lane,
                output_values_per_lane,
            )
            packet_fifo.release(1)

            for phase_tail in range_(1):
                packet = packet_fifo.acquire(1)
                layer_i32 = index.casts(T.i32(), layer)
                phase_i32 = index.casts(T.i32(), phase_tail)
                packet_kernel(
                    packet,
                    state,
                    packet_elements,
                    layer_i32,
                    phase_i32,
                )
                packet_fifo.release(1)

            packet = packet_fifo.acquire(1)
            gate_up_kernel(
                packet,
                state,
                lane_output,
                packet_elements,
                hidden_size,
                q_rows_per_packet,
                q_output_values_per_lane
                + k_output_values_per_lane
                + v_output_values_per_lane
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
                + attention_output_values_per_lane
                + gate_up_output_values_per_lane,
                output_values_per_lane,
            )
            packet_fifo.release(1)
            lane_output_fifo.release(1)

            for phase_tail in range_(phase_packets_per_layer - 10):
                packet = packet_fifo.acquire(1)
                layer_i32 = index.casts(T.i32(), layer)
                phase_i32 = index.casts(T.i32(), phase_tail)
                packet_kernel(
                    packet,
                    state,
                    packet_elements,
                    layer_i32,
                    phase_i32,
                )
                packet_fifo.release(1)

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

    workers = [
        Worker(
            lane_worker_body,
            [
                shared_fifos[lane // fabric_group_size].cons(),
                packet_fifos[lane].cons(),
                lane_output_fifos[lane].prod(),
                hidden_states[lane],
                states[lane],
                init_kernel,
                q_shard_kernel,
                projection_kernel,
                o_kernel,
                gate_up_kernel,
                down_kernel,
                packet_kernel,
                next_hidden_kernel,
            ],
            stack_size=0xD00,
        )
        for lane in range(num_lanes)
    ]

    shared_tap = TensorAccessPattern(
        (shared_input_elements,),
        0,
        [1, 1, 1, shared_input_elements],
        [0, 0, 0, 1],
    )
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
        rt.start(*workers)
        tg = rt.task_group()
        for group in range(fabric_group_count):
            rt.fill(
                shared_l3_fifos[group].prod(),
                shared_l3,
                shared_tap,
                task_group=tg,
            )
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
