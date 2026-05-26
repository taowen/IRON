"""
Experiment 04: Numerically-Correct Fused RMSNorm → GEMV

First fused slice with REAL kernels and correctness verification.
The intermediate (normed hidden) flows tile-to-tile, never touches DDR.

Topology:
  DDR(hidden[K]) → [RMSNorm Tile] → normed[K] (tile-to-tile) → [GEMV Tile]
  DDR(W[M×K])   → weight FIFO ────────────────────────────────→ [GEMV Tile]
                                                                    ↓
                                                        DDR(output[M]) ← result
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def fused_norm_gemv_pipeline(
    dev,
    K,          # hidden dimension
    M,          # projection output dimension
    m_input,    # GEMV tile rows per kernel call
):
    assert K >= 16, "K must be at least 16 for RMSNorm vectorization"
    assert M % m_input == 0

    dtype = np.dtype[bfloat16]

    # L1 types
    L1_hidden_ty = np.ndarray[(K,), dtype]
    L1_normed_ty = np.ndarray[(K,), dtype]
    L1_W_ty = np.ndarray[(m_input, K), dtype]
    L1_C_ty = np.ndarray[(M,), dtype]

    # L3 (DDR) types
    L3_hidden_ty = np.ndarray[(K,), dtype]
    L3_W_ty = np.ndarray[(M * K,), dtype]
    L3_C_ty = np.ndarray[(M,), dtype]

    # Kernels — two different kernel objects on different tiles
    rms_norm_kernel = Kernel(
        "rms_norm_bf16_vector",
        "rms_norm.o",
        [L1_hidden_ty, L1_normed_ty, np.int32],
    )

    matvec_kernel = Kernel(
        "matvec_vectorized_bf16_bf16",
        "mv.o",
        [np.int32, np.int32, L1_W_ty, L1_normed_ty, L1_C_ty],
    )

    # === ObjectFifos ===

    # DDR → RMSNorm tile
    hidden_fifo = ObjectFifo(L1_hidden_ty, name="hidden_in", depth=2)

    # RMSNorm tile → GEMV tile (INTERNAL, no DDR)
    normed_fifo = ObjectFifo(L1_normed_ty, name="normed", depth=2)

    # DDR → GEMV tile (weight stream)
    W_fifo = ObjectFifo(L1_W_ty, name="W", depth=2)

    # GEMV tile → DDR (output)
    C_fifo = ObjectFifo(L1_C_ty, name="C", depth=2)

    # === Workers ===

    def norm_worker_body(hidden_in, normed_out, rms_norm_k):
        for _ in range_(0xFFFFFFFF):
            h = hidden_in.acquire(1)
            n = normed_out.acquire(1)
            rms_norm_k(h, n, K)
            normed_out.release(1)
            hidden_in.release(1)

    norm_worker = Worker(
        norm_worker_body,
        [hidden_fifo.cons(), normed_fifo.prod(), rms_norm_kernel],
    )

    def gemv_worker_body(normed_in, W_in, C_out, matvec_k):
        for _ in range_(0xFFFFFFFF):
            x = normed_in.acquire(1)
            c = C_out.acquire(1)
            for i_idx in range_(M // m_input):
                i_i32 = index.casts(T.i32(), i_idx)
                row_offset = i_i32 * m_input
                w = W_in.acquire(1)
                matvec_k(m_input, row_offset, w, x, c)
                W_in.release(1)
            C_out.release(1)
            normed_in.release(1)

    gemv_worker = Worker(
        gemv_worker_body,
        [normed_fifo.cons(), W_fifo.cons(), C_fifo.prod(), matvec_kernel],
    )

    # === Runtime Sequence ===

    hidden_tap = TensorAccessPattern(
        tensor_dims=L3_hidden_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, K],
        strides=[0, 0, 0, 1],
    )

    W_tap = TensorAccessPattern(
        tensor_dims=L3_W_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, M * K],
        strides=[0, 0, 0, 1],
    )

    C_tap = TensorAccessPattern(
        tensor_dims=L3_C_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, M],
        strides=[0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(L3_hidden_ty, L3_W_ty, L3_C_ty) as (hidden_buf, W_buf, C_buf):
        rt.start(norm_worker, gemv_worker)

        tg = rt.task_group()
        rt.fill(hidden_fifo.prod(), hidden_buf, hidden_tap, task_group=tg)
        rt.fill(W_fifo.prod(), W_buf, W_tap, task_group=tg)
        rt.drain(C_fifo.cons(), C_buf, C_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
