# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def c1_two_phase_worker(dev, size, kernel_object="two_phase_worker.o"):
    dtype = bfloat16

    l3_ty = np.ndarray[(size,), np.dtype[dtype]]
    tile_ty = np.ndarray[(size,), np.dtype[dtype]]

    phase0_kernel = Kernel(
        "c1_phase0_bf16",
        kernel_object,
        [tile_ty, tile_ty, np.int32],
    )
    phase1_kernel = Kernel(
        "c1_phase1_bf16",
        kernel_object,
        [tile_ty, tile_ty, tile_ty, np.int32],
    )

    phase0_fifo = ObjectFifo(tile_ty, name="c1_phase0_in", depth=1)
    phase1_fifo = ObjectFifo(tile_ty, name="c1_phase1_in", depth=1)
    out_fifo = ObjectFifo(tile_ty, name="c1_out", depth=1)
    state = Buffer(
        initial_value=np.zeros(shape=(size,), dtype=dtype),
        name="c1_phase_state",
    )

    def worker_body(
        phase0_fifo,
        phase1_fifo,
        out_fifo,
        state,
        phase0_kernel,
        phase1_kernel,
    ):
        phase0 = phase0_fifo.acquire(1)
        phase0_kernel(phase0, state, size)
        phase0_fifo.release(1)

        phase1 = phase1_fifo.acquire(1)
        out = out_fifo.acquire(1)
        phase1_kernel(state, phase1, out, size)
        phase1_fifo.release(1)
        out_fifo.release(1)

    worker = Worker(
        worker_body,
        [
            phase0_fifo.cons(),
            phase1_fifo.cons(),
            out_fifo.prod(),
            state,
            phase0_kernel,
            phase1_kernel,
        ],
        stack_size=0xD00,
    )

    tap = TensorAccessPattern((size,), 0, [1, 1, 1, size], [0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(l3_ty, l3_ty, l3_ty) as (phase0_l3, phase1_l3, out_l3):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(phase0_fifo.prod(), phase0_l3, tap, task_group=tg)
        rt.fill(phase1_fifo.prod(), phase1_l3, tap, task_group=tg)
        rt.drain(out_fifo.cons(), out_l3, tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
