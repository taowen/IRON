"""
Experiment 06: Single-Descriptor Multi-K Q4NX GEMV (32×4096, fp32 Accumulation)

Single tile processes 16 Q4NX chunks (K=4096) with fp32 on-tile accumulator.
One DMA descriptor per stream — chunks are core loop unit, not host task unit.
"""

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Buffer, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib import TensorAccessPattern


def q4nx_gemv_pipeline(dev, M=32, K=4096, K_chunk=256, group_size=32):
    K_chunks = K // K_chunk
    groups_per_row = K_chunk // group_size
    chunk_bytes = M * groups_per_row * 2 * 2 + M * K_chunk // 2
    chunk_bf16_elems = chunk_bytes // 2

    # L1 types (per-iteration)
    L1_chunk_ty = np.ndarray[(chunk_bf16_elems,), np.dtype[bfloat16]]
    L1_act_ty = np.ndarray[(K_chunk,), np.dtype[bfloat16]]
    L1_out_ty = np.ndarray[(M,), np.dtype[bfloat16]]
    L1_acc_ty = np.ndarray[(M,), np.dtype[np.float32]]

    # L3 types (full transfer)
    L3_wt_ty = np.ndarray[(K_chunks * chunk_bf16_elems,), np.dtype[bfloat16]]
    L3_act_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    L3_out_ty = np.ndarray[(M,), np.dtype[bfloat16]]

    # Kernels
    mac_kernel = Kernel(
        "q4nx_mac_f32", "q4nx_matvec.o",
        [L1_chunk_ty, L1_act_ty, L1_acc_ty, np.int32],
    )
    zero_kernel = Kernel(
        "q4nx_zero_f32", "q4nx_matvec.o",
        [L1_acc_ty, np.int32],
    )
    convert_kernel = Kernel(
        "q4nx_convert_f32_bf16", "q4nx_matvec.o",
        [L1_acc_ty, L1_out_ty, np.int32],
    )

    # ObjectFifos
    wt_fifo = ObjectFifo(L1_chunk_ty, name="wt_fifo", depth=2)
    act_fifo = ObjectFifo(L1_act_ty, name="act_fifo", depth=2)
    out_fifo = ObjectFifo(L1_out_ty, name="out_fifo", depth=2)

    # Local fp32 accumulator buffer
    acc_buffer = Buffer(L1_acc_ty, name="acc_buffer")

    def core_body(wt_fifo, act_fifo, out_fifo, mac_kernel, zero_kernel, convert_kernel, acc_buffer):
        for _ in range_(0xFFFFFFFF):
            zero_kernel(acc_buffer, M)
            for _ in range_(K_chunks):
                wt = wt_fifo.acquire(1)
                act = act_fifo.acquire(1)
                mac_kernel(wt, act, acc_buffer, M)
                wt_fifo.release(1)
                act_fifo.release(1)
            out = out_fifo.acquire(1)
            convert_kernel(acc_buffer, out, M)
            out_fifo.release(1)

    worker = Worker(
        core_body,
        [wt_fifo.cons(), act_fifo.cons(), out_fifo.prod(),
         mac_kernel, zero_kernel, convert_kernel, acc_buffer],
    )

    # TAPs — one descriptor per stream
    # BD size dimensions limited to 1023 on AIE2P; decompose chunk (2560) into 10×256
    wt_chunk_rows = 10
    wt_chunk_cols = chunk_bf16_elems // wt_chunk_rows  # 256
    wt_tap = TensorAccessPattern(
        tensor_dims=L3_wt_ty.__args__[0],
        offset=0,
        sizes=[1, K_chunks, wt_chunk_rows, wt_chunk_cols],
        strides=[0, chunk_bf16_elems, wt_chunk_cols, 1],
    )
    act_tap = TensorAccessPattern(
        tensor_dims=L3_act_ty.__args__[0],
        offset=0,
        sizes=[1, 1, K_chunks, K_chunk],
        strides=[0, 0, K_chunk, 1],
    )
    out_tap = TensorAccessPattern(
        tensor_dims=L3_out_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, M],
        strides=[0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(L3_wt_ty, L3_act_ty, L3_out_ty) as (wt_buf, act_buf, out_buf):
        rt.start(worker)
        rt.fill(act_fifo.prod(), act_buf, act_tap)
        rt.fill(wt_fifo.prod(), wt_buf, wt_tap)
        rt.drain(out_fifo.cons(), out_buf, out_tap, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())
