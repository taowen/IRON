# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def c2_inactive_fifo_skip(
    dev,
    size,
    use_optional_phase,
    kernel_object="inactive_fifo_skip.o",
):
    dtype = bfloat16

    l3_ty = np.ndarray[(size,), np.dtype[dtype]]
    tile_ty = np.ndarray[(size,), np.dtype[dtype]]

    phase0_kernel = Kernel(
        "c2_phase0_bf16",
        kernel_object,
        [tile_ty, tile_ty, np.int32],
    )
    skip_kernel = Kernel(
        "c2_finalize_skip_bf16",
        kernel_object,
        [tile_ty, tile_ty, np.int32],
    )
    active_kernel = Kernel(
        "c2_finalize_active_bf16",
        kernel_object,
        [tile_ty, tile_ty, tile_ty, np.int32],
    )

    phase0_fifo = ObjectFifo(tile_ty, name="c2_phase0_in", depth=1)
    out_fifo = ObjectFifo(tile_ty, name="c2_out", depth=1)
    state = Buffer(
        initial_value=np.zeros(shape=(size,), dtype=dtype),
        name="c2_phase_state",
    )

    tap = TensorAccessPattern((size,), 0, [1, 1, 1, size], [0, 0, 0, 1])
    rt = Runtime()

    if use_optional_phase:
        optional_fifo = ObjectFifo(tile_ty, name="c2_optional_in", depth=1)

        def active_worker_body(
            phase0_fifo,
            optional_fifo,
            out_fifo,
            state,
            phase0_kernel,
            active_kernel,
        ):
            phase0 = phase0_fifo.acquire(1)
            phase0_kernel(phase0, state, size)
            phase0_fifo.release(1)

            optional = optional_fifo.acquire(1)
            out = out_fifo.acquire(1)
            active_kernel(state, optional, out, size)
            optional_fifo.release(1)
            out_fifo.release(1)

        worker = Worker(
            active_worker_body,
            [
                phase0_fifo.cons(),
                optional_fifo.cons(),
                out_fifo.prod(),
                state,
                phase0_kernel,
                active_kernel,
            ],
            stack_size=0xD00,
        )

        with rt.sequence(l3_ty, l3_ty, l3_ty) as (phase0_l3, optional_l3, out_l3):
            rt.start(worker)
            tg = rt.task_group()
            rt.fill(phase0_fifo.prod(), phase0_l3, tap, task_group=tg)
            rt.fill(optional_fifo.prod(), optional_l3, tap, task_group=tg)
            rt.drain(out_fifo.cons(), out_l3, tap, wait=True, task_group=tg)
            rt.finish_task_group(tg)
    else:

        def skip_worker_body(
            phase0_fifo,
            out_fifo,
            state,
            phase0_kernel,
            skip_kernel,
        ):
            phase0 = phase0_fifo.acquire(1)
            phase0_kernel(phase0, state, size)
            phase0_fifo.release(1)

            out = out_fifo.acquire(1)
            skip_kernel(state, out, size)
            out_fifo.release(1)

        worker = Worker(
            skip_worker_body,
            [
                phase0_fifo.cons(),
                out_fifo.prod(),
                state,
                phase0_kernel,
                skip_kernel,
            ],
            stack_size=0xD00,
        )

        with rt.sequence(l3_ty, l3_ty) as (phase0_l3, out_l3):
            rt.start(worker)
            tg = rt.task_group()
            rt.fill(phase0_fifo.prod(), phase0_l3, tap, task_group=tg)
            rt.drain(out_fifo.cons(), out_l3, tap, wait=True, task_group=tg)
            rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
