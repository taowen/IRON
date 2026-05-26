"""
Experiment 11: Multi-Chunk Weight-Streaming Q4NX GEMV

Proves chunked weight streaming with lock-driven double-buffering:
- K=1024 processed as 4x256 chunks via weight FIFO depth=2
- Activation also streamed in 256-element chunks (in sync with weights)
- FP32 accumulation across chunks (global accumulator in kernel BSS)
- Matches MyLM's projection tile inner loop pattern
"""

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


M = 32
K = 1024
K_CHUNK = 256
NUM_CHUNKS = K // K_CHUNK
GROUP_SIZE = 32


def chunked_q4nx_pipeline(dev):
    groups_per_row = K_CHUNK // GROUP_SIZE
    chunk_bytes = M * groups_per_row * 2 + M * groups_per_row * 2 + M * K_CHUNK // 2
    chunk_bf16_elems = chunk_bytes // 2  # 2560

    # Per-chunk types (local tile buffers)
    L1_chunk_ty = np.ndarray[(chunk_bf16_elems,), np.dtype[bfloat16]]
    L1_act_chunk_ty = np.ndarray[(K_CHUNK,), np.dtype[bfloat16]]
    L1_out_ty = np.ndarray[(M,), np.dtype[bfloat16]]

    # L3 (DDR) buffer types
    L3_wt_ty = np.ndarray[(chunk_bf16_elems * NUM_CHUNKS,), np.dtype[bfloat16]]
    L3_act_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    L3_out_ty = np.ndarray[(M,), np.dtype[bfloat16]]

    accum_kernel = Kernel(
        "q4nx_chunk_accum",
        "q4nx_matvec_accum.o",
        [L1_chunk_ty, L1_act_chunk_ty, np.int32],
    )

    flush_kernel = Kernel(
        "q4nx_flush_output",
        "q4nx_matvec_accum.o",
        [L1_out_ty, np.int32],
    )

    wt_fifo = ObjectFifo(L1_chunk_ty, name="wt_fifo", depth=2)
    act_fifo = ObjectFifo(L1_act_chunk_ty, name="act_fifo", depth=2)
    out_fifo = ObjectFifo(L1_out_ty, name="out_fifo", depth=2)

    def core_body(wt_fifo, act_fifo, out_fifo, accum_kernel, flush_kernel):
        for _ in range_(0xFFFFFFFF):
            out = out_fifo.acquire(1)
            for _ in range_(NUM_CHUNKS):
                wt = wt_fifo.acquire(1)
                act = act_fifo.acquire(1)
                accum_kernel(wt, act, M)
                wt_fifo.release(1)
                act_fifo.release(1)
            flush_kernel(out, M)
            out_fifo.release(1)

    worker = Worker(
        core_body,
        [wt_fifo.cons(), act_fifo.cons(), out_fifo.prod(), accum_kernel, flush_kernel],
    )

    # TAP: send 4 weight chunks sequentially from L3 weight buffer
    # 2560 > 1023 (dma_bd size limit), so factor as 10 x 256
    wt_tap = TensorAccessPattern(
        tensor_dims=L3_wt_ty.__args__[0],
        offset=0,
        sizes=[1, NUM_CHUNKS, 10, 256],
        strides=[0, chunk_bf16_elems, 256, 1],
    )

    # TAP: send 4 activation slices (each K_CHUNK=256) from L3 activation buffer
    act_tap = TensorAccessPattern(
        tensor_dims=L3_act_ty.__args__[0],
        offset=0,
        sizes=[1, 1, NUM_CHUNKS, K_CHUNK],
        strides=[0, 0, K_CHUNK, 1],
    )

    # TAP: drain 1 output (M=32 bf16)
    out_tap = TensorAccessPattern(
        tensor_dims=L3_out_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, M],
        strides=[0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(L3_wt_ty, L3_act_ty, L3_out_ty) as (wt_buf, act_buf, out_buf):
        rt.start(worker)
        rt.fill(wt_fifo.prod(), wt_buf, wt_tap)
        rt.fill(act_fifo.prod(), act_buf, act_tap)
        rt.drain(out_fifo.cons(), out_buf, out_tap, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())
