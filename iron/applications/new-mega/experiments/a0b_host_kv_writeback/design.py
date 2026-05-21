# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def host_kv_writeback_blob(
    dev,
    max_seq_len,
    head_dim,
    chunk_size,
    kernel_object="host_kv_writeback.o",
):
    if max_seq_len % chunk_size != 0:
        raise ValueError("max_seq_len must be divisible by chunk_size")

    num_chunks = max_seq_len // chunk_size
    packed_chunk_elements = 2 * chunk_size * head_dim + chunk_size
    dtype = bfloat16

    current_l3_ty = np.ndarray[(head_dim,), np.dtype[dtype]]
    packed_cache_l3_ty = np.ndarray[
        (num_chunks * packed_chunk_elements,), np.dtype[dtype]
    ]
    out_l3_ty = np.ndarray[(3 * head_dim,), np.dtype[dtype]]

    current_ty = np.ndarray[(head_dim,), np.dtype[dtype]]
    packed_chunk_ty = np.ndarray[(packed_chunk_elements,), np.dtype[dtype]]
    out_ty = np.ndarray[(3 * head_dim,), np.dtype[dtype]]
    acc_ty = np.ndarray[(head_dim,), np.dtype[np.float32]]

    init_kernel = Kernel(
        "host_kv_writeback_acc_init_f32",
        kernel_object,
        [acc_ty, np.int32],
    )
    update_kernel = Kernel(
        "host_kv_writeback_acc_update_bf16",
        kernel_object,
        [packed_chunk_ty, acc_ty, np.int32, np.int32],
    )
    finalize_kernel = Kernel(
        "host_kv_writeback_finalize_bf16",
        kernel_object,
        [current_ty, acc_ty, out_ty, np.int32],
    )

    current_fifo = ObjectFifo(current_ty, name="host_kv_current", depth=1)
    cache_fifo = ObjectFifo(packed_chunk_ty, name="host_kv_cache", depth=1)
    out_fifo = ObjectFifo(out_ty, name="host_kv_out", depth=1)
    acc = Buffer(
        initial_value=np.zeros(shape=(head_dim,), dtype=np.float32),
        name="host_kv_acc",
    )

    def worker_body(
        current_fifo,
        cache_fifo,
        out_fifo,
        acc,
        init_kernel,
        update_kernel,
        finalize_kernel,
    ):
        current = current_fifo.acquire(1)
        init_kernel(acc, head_dim)

        for _ in range_(num_chunks):
            cache_chunk = cache_fifo.acquire(1)
            update_kernel(cache_chunk, acc, chunk_size, head_dim)
            cache_fifo.release(1)

        out = out_fifo.acquire(1)
        finalize_kernel(current, acc, out, head_dim)
        out_fifo.release(1)
        current_fifo.release(1)

    worker = Worker(
        worker_body,
        [
            current_fifo.cons(),
            cache_fifo.cons(),
            out_fifo.prod(),
            acc,
            init_kernel,
            update_kernel,
            finalize_kernel,
        ],
        stack_size=0xD00,
    )

    current_tap = TensorAccessPattern((head_dim,), 0, [1, 1, 1, head_dim], [0, 0, 0, 1])
    cache_tap = TensorAccessPattern(
        (num_chunks * packed_chunk_elements,),
        0,
        [1, 1, 1, num_chunks * packed_chunk_elements],
        [0, 0, 0, 1],
    )
    out_tap = TensorAccessPattern(
        (3 * head_dim,), 0, [1, 1, 1, 3 * head_dim], [0, 0, 0, 1]
    )

    rt = Runtime()
    with rt.sequence(current_l3_ty, packed_cache_l3_ty, out_l3_ty) as (
        current_l3,
        cache_l3,
        out_l3,
    ):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(current_fifo.prod(), current_l3, current_tap, task_group=tg)
        rt.fill(cache_fifo.prod(), cache_l3, cache_tap, task_group=tg)
        rt.drain(out_fifo.cons(), out_l3, out_tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
