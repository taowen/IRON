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


def fixed_cache_attention_context(
    dev,
    max_seq_len,
    q_heads,
    kv_heads,
    head_dim,
    chunk_size,
    kernel_object="fixed_attention.o",
):
    if max_seq_len % chunk_size != 0:
        raise ValueError("max_seq_len must be divisible by chunk_size")
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")

    num_chunks = max_seq_len // chunk_size
    q_size = q_heads * head_dim
    packed_chunk_elements = 2 * chunk_size * head_dim + chunk_size
    dtype = bfloat16

    q_l3_ty = np.ndarray[(q_size,), np.dtype[dtype]]
    packed_l3_ty = np.ndarray[
        (q_heads * num_chunks * packed_chunk_elements,), np.dtype[dtype]
    ]
    out_l3_ty = np.ndarray[(q_size,), np.dtype[dtype]]

    q_ty = np.ndarray[(q_size,), np.dtype[dtype]]
    packed_chunk_ty = np.ndarray[(packed_chunk_elements,), np.dtype[dtype]]
    out_ty = np.ndarray[(q_size,), np.dtype[dtype]]
    state_ty = np.ndarray[(2,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(head_dim,), np.dtype[np.float32]]

    init_kernel = Kernel(
        "new_mega_fixed_attention_init_f32",
        kernel_object,
        [state_ty, acc_ty, np.int32],
    )
    update_kernel = Kernel(
        "new_mega_fixed_attention_update_packed_bf16",
        kernel_object,
        [q_ty, packed_chunk_ty, state_ty, acc_ty, np.int32, np.int32, np.int32],
    )
    finalize_kernel = Kernel(
        "new_mega_fixed_attention_finalize_into_full_bf16",
        kernel_object,
        [state_ty, acc_ty, out_ty, np.int32, np.int32],
    )

    q_fifo = ObjectFifo(q_ty, name="new_mega_attn_q", depth=1)
    packed_fifo = ObjectFifo(packed_chunk_ty, name="new_mega_attn_cache", depth=1)
    out_fifo = ObjectFifo(out_ty, name="new_mega_attn_context", depth=1)

    state = Buffer(
        initial_value=np.zeros(shape=(2,), dtype=np.float32),
        name="new_mega_attn_state",
    )
    acc = Buffer(
        initial_value=np.zeros(shape=(head_dim,), dtype=np.float32),
        name="new_mega_attn_acc",
    )

    def worker_body(
        q_fifo,
        packed_fifo,
        out_fifo,
        state,
        acc,
        init_kernel,
        update_kernel,
        finalize_kernel,
    ):
        q = q_fifo.acquire(1)
        out = out_fifo.acquire(1)
        for q_head in range_(q_heads):
            q_head_i32 = index.casts(T.i32(), q_head)
            init_kernel(state, acc, head_dim)
            for _ in range_(num_chunks):
                packed = packed_fifo.acquire(1)
                update_kernel(
                    q,
                    packed,
                    state,
                    acc,
                    q_head_i32,
                    chunk_size,
                    head_dim,
                )
                packed_fifo.release(1)
            finalize_kernel(state, acc, out, q_head_i32, head_dim)
        out_fifo.release(1)
        q_fifo.release(1)

    worker = Worker(
        worker_body,
        [
            q_fifo.cons(),
            packed_fifo.cons(),
            out_fifo.prod(),
            state,
            acc,
            init_kernel,
            update_kernel,
            finalize_kernel,
        ],
        stack_size=0xD00,
    )

    q_tap = TensorAccessPattern((q_size,), 0, [1, 1, 1, q_size], [0, 0, 0, 1])
    packed_tap = TensorAccessPattern(
        (q_heads * num_chunks * packed_chunk_elements,),
        0,
        [1, 1, 1, q_heads * num_chunks * packed_chunk_elements],
        [0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(q_l3_ty, packed_l3_ty, out_l3_ty) as (q_l3, packed_l3, out_l3):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(q_fifo.prod(), q_l3, q_tap, task_group=tg)
        rt.fill(packed_fifo.prod(), packed_l3, packed_tap, task_group=tg)
        rt.drain(out_fifo.cons(), out_l3, q_tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
