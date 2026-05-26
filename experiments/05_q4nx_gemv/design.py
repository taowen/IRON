"""
Experiment 05: Q4NX Online Dequant+GEMV — Single Tile

Q4NX packed weights (int4 + scale + zero) are streamed to a single compute tile,
unpacked and dequantized inline, and MAC'd with the activation vector.
No intermediate bf16 weight tensor exists in memory.
"""

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def q4nx_gemv_pipeline(dev, M=32, K=256, group_size=32):
    groups_per_row = K // group_size
    chunk_bytes = M * groups_per_row * 2 + M * groups_per_row * 2 + M * K // 2
    # Buffer as bf16 elements: chunk_bytes / 2
    chunk_bf16_elems = chunk_bytes // 2

    L1_chunk_ty = np.ndarray[(chunk_bf16_elems,), np.dtype[bfloat16]]
    L1_act_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    L1_out_ty = np.ndarray[(M,), np.dtype[bfloat16]]

    L3_chunk_ty = np.ndarray[(chunk_bf16_elems,), np.dtype[bfloat16]]
    L3_act_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    L3_out_ty = np.ndarray[(M,), np.dtype[bfloat16]]

    q4_kernel = Kernel(
        "q4nx_matvec_bf16",
        "q4nx_matvec.o",
        [L1_chunk_ty, L1_act_ty, L1_out_ty, np.int32],
    )

    wt_fifo = ObjectFifo(L1_chunk_ty, name="wt_fifo", depth=2)
    act_fifo = ObjectFifo(L1_act_ty, name="act_fifo", depth=1)
    out_fifo = ObjectFifo(L1_out_ty, name="out_fifo", depth=2)

    def core_body(wt_fifo, act_fifo, out_fifo, kernel):
        for _ in range_(0xFFFFFFFF):
            act = act_fifo.acquire(1)
            wt = wt_fifo.acquire(1)
            out = out_fifo.acquire(1)
            kernel(wt, act, out, M)
            out_fifo.release(1)
            wt_fifo.release(1)
            act_fifo.release(1)

    worker = Worker(
        core_body,
        [wt_fifo.cons(), act_fifo.cons(), out_fifo.prod(), q4_kernel],
    )

    wt_tap = TensorAccessPattern(
        tensor_dims=L3_chunk_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, chunk_bf16_elems],
        strides=[0, 0, 0, 1],
    )
    act_tap = TensorAccessPattern(
        tensor_dims=L3_act_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, K],
        strides=[0, 0, 0, 1],
    )
    out_tap = TensorAccessPattern(
        tensor_dims=L3_out_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, M],
        strides=[0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(L3_chunk_ty, L3_act_ty, L3_out_ty) as (wt_buf, act_buf, out_buf):
        rt.start(worker)
        rt.fill(act_fifo.prod(), act_buf, act_tap)
        rt.fill(wt_fifo.prod(), wt_buf, wt_tap)
        rt.drain(out_fifo.cons(), out_buf, out_tap, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())
