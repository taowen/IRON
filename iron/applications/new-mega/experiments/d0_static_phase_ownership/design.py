# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def d0_static_phase_ownership(
    dev,
    num_lanes,
    num_phase_packets,
    packet_elements,
    kernel_object="static_phase_skeleton.o",
):
    dtype = bfloat16

    packets_l3_ty = np.ndarray[
        (num_lanes * num_phase_packets * packet_elements,), np.dtype[dtype]
    ]
    outputs_l3_ty = np.ndarray[(num_lanes * 8,), np.dtype[dtype]]
    packet_ty = np.ndarray[(packet_elements,), np.dtype[dtype]]
    out_ty = np.ndarray[(8,), np.dtype[dtype]]
    state_ty = np.ndarray[(1,), np.dtype[np.float32]]

    init_kernel = Kernel(
        "d0_phase_state_init_f32",
        kernel_object,
        [state_ty],
    )
    accum_kernel = Kernel(
        "d0_phase_packet_accum_bf16",
        kernel_object,
        [packet_ty, state_ty, np.int32],
    )
    finalize_kernel = Kernel(
        "d0_phase_state_finalize_bf16",
        kernel_object,
        [state_ty, out_ty],
    )

    packet_fifos = [
        ObjectFifo(packet_ty, name=f"d0_lane_{lane}_packets", depth=1)
        for lane in range(num_lanes)
    ]
    out_fifos = [
        ObjectFifo(out_ty, name=f"d0_lane_{lane}_out", depth=1)
        for lane in range(num_lanes)
    ]
    states = [
        Buffer(
            initial_value=np.zeros(shape=(1,), dtype=np.float32),
            name=f"d0_lane_{lane}_state",
        )
        for lane in range(num_lanes)
    ]

    def worker_body(
        packet_fifo, out_fifo, state, init_kernel, accum_kernel, finalize_kernel
    ):
        init_kernel(state)
        for _ in range_(num_phase_packets):
            packet = packet_fifo.acquire(1)
            accum_kernel(packet, state, packet_elements)
            packet_fifo.release(1)

        out = out_fifo.acquire(1)
        finalize_kernel(state, out)
        out_fifo.release(1)

    workers = [
        Worker(
            worker_body,
            [
                packet_fifos[lane].cons(),
                out_fifos[lane].prod(),
                states[lane],
                init_kernel,
                accum_kernel,
                finalize_kernel,
            ],
            stack_size=0xD00,
        )
        for lane in range(num_lanes)
    ]

    packet_taps = [
        TensorAccessPattern(
            (num_lanes * num_phase_packets * packet_elements,),
            lane * num_phase_packets * packet_elements,
            [1, 1, 1, num_phase_packets * packet_elements],
            [0, 0, 0, 1],
        )
        for lane in range(num_lanes)
    ]
    out_taps = [
        TensorAccessPattern((num_lanes * 8,), lane * 8, [1, 1, 1, 8], [0, 0, 0, 1])
        for lane in range(num_lanes)
    ]

    rt = Runtime()
    with rt.sequence(packets_l3_ty, outputs_l3_ty) as (packets_l3, outputs_l3):
        rt.start(*workers)
        tg = rt.task_group()
        for lane in range(num_lanes):
            rt.fill(
                packet_fifos[lane].prod(), packets_l3, packet_taps[lane], task_group=tg
            )
        for lane in range(num_lanes):
            rt.drain(
                out_fifos[lane].cons(),
                outputs_l3,
                out_taps[lane],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
