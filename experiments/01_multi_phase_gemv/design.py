"""
Experiment 01: Multi-Phase GEMV with On-Chip Input Reuse

Goal: Verify whether IRON's ObjectFifo + Worker model can express a single
kernel run that does two sequential matrix-vector multiplications (simulating
Q and K projections) while keeping the input vector on-chip between phases.

Fused Layer Engine equivalent:
  - hidden_norm is loaded once into memtile/core local
  - Q weights stream in, Q = W_q @ hidden_norm
  - K weights stream in, K = W_k @ hidden_norm  (hidden_norm NOT re-read from DDR)

This design tests:
  1. Can a Worker hold an ObjectFifo element across multiple compute phases?
  2. Can the Runtime fill a single weight ObjectFifo with data for two
     sequential projections (Q weights followed by K weights)?
  3. Does the resulting DMA sequence avoid re-transferring the input vector?
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def multi_phase_gemv(
    dev,
    cols,
    M_q,       # output dim of first projection (Q)
    M_k,       # output dim of second projection (K)
    K,         # shared input dim (hidden size)
    m_input,   # tile size for weight rows streamed per kernel call
    func_prefix="",
):
    """
    Single-graph design: two sequential GEMVs sharing the same input vector.

    Phase 1: y_q[M_q] = W_q[M_q, K] @ x[K]
    Phase 2: y_k[M_k] = W_k[M_k, K] @ x[K]

    Key constraint being tested: x is acquired ONCE and held across both phases.
    Weights for both projections stream through the SAME ObjectFifo sequentially.

    NOTE: For this experiment, M_q == M_k to avoid kernel type signature issues.
    The structural question (multi-phase with on-chip reuse) is independent of
    output dimension differences.
    """

    assert M_q == M_k, "This experiment uses equal output dims to isolate the multi-phase question"
    assert M_q % cols == 0
    M = M_q  # both phases have same output dim

    dtype_in = np.dtype[bfloat16]

    # Types for ObjectFifo tiles
    L1_W_ty = np.ndarray[(m_input, K), dtype_in]    # weight tile: m_input rows x K
    L1_X_ty = np.ndarray[(K,), dtype_in]            # input vector: K elements
    L1_C_ty = np.ndarray[(M // cols,), dtype_in]    # output per column (same for both phases)

    # DDR buffer types
    total_weight_rows = M_q + M_k
    L3_W_ty = np.ndarray[(total_weight_rows * K,), dtype_in]  # Q weights then K weights
    L3_X_ty = np.ndarray[(K,), dtype_in]                       # input vector
    L3_Cq_ty = np.ndarray[(M_q,), dtype_in]                    # output Q
    L3_Ck_ty = np.ndarray[(M_k,), dtype_in]                    # output K

    # Kernel: same matvec kernel used for both phases
    matvec = Kernel(
        f"{func_prefix}matvec_vectorized_bf16_bf16",
        f"{func_prefix}mv.o",
        [np.int32, np.int32, L1_W_ty, L1_X_ty, L1_C_ty],
    )

    # ObjectFifos
    # Weight FIFO: carries BOTH Q and K weight tiles sequentially
    W_fifos = [
        ObjectFifo(L1_W_ty, name=f"W_{i}", depth=2) for i in range(cols)
    ]
    # Input vector FIFO: filled ONCE, held across both phases
    X_fifos = [
        ObjectFifo(L1_X_ty, name=f"X_{i}", depth=1) for i in range(cols)
    ]
    # Output FIFOs: separate for Q and K results (same type)
    Cq_fifos = [
        ObjectFifo(L1_C_ty, name=f"Cq_{i}", depth=2) for i in range(cols)
    ]
    Ck_fifos = [
        ObjectFifo(L1_C_ty, name=f"Ck_{i}", depth=2) for i in range(cols)
    ]

    def core_body(W_fifo, X_fifo, Cq_fifo, Ck_fifo, matvec_kernel):
        """
        Critical test: acquire x ONCE, use it for BOTH projection phases.
        """
        # Acquire input vector — hold it for the entire multi-phase computation
        x = X_fifo.acquire(1)

        # Phase 1: Q projection
        # Consume (M / cols / m_input) weight tiles, accumulate output
        cq = Cq_fifo.acquire(1)
        for i_idx in range_((M // cols) // m_input):
            i_i32 = index.casts(T.i32(), i_idx)
            row_offset = i_i32 * m_input
            w = W_fifo.acquire(1)
            matvec_kernel(m_input, row_offset, w, x, cq)
            W_fifo.release(1)
        Cq_fifo.release(1)

        # Phase 2: K projection — x is STILL held, not re-acquired
        ck = Ck_fifo.acquire(1)
        for j_idx in range_((M // cols) // m_input):
            j_i32 = index.casts(T.i32(), j_idx)
            row_offset = j_i32 * m_input
            w = W_fifo.acquire(1)
            matvec_kernel(m_input, row_offset, w, x, ck)
            W_fifo.release(1)
        Ck_fifo.release(1)

        # Release input vector only after BOTH phases complete
        X_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [
                W_fifos[i].cons(),
                X_fifos[i].cons(),
                Cq_fifos[i].prod(),
                Ck_fifos[i].prod(),
                matvec,
            ],
        )
        for i in range(cols)
    ]

    # DMA patterns
    # Weight TAP: Q weights (M rows) followed by K weights (M rows),
    # distributed across columns
    rows_per_col = M // cols

    W_taps = []
    for col in range(cols):
        # Q weight region for this column, then K weight region
        q_offset = col * rows_per_col * K
        k_offset = M * K + col * rows_per_col * K
        tap_q = TensorAccessPattern(
            tensor_dims=L3_W_ty.__args__[0],
            offset=q_offset,
            sizes=[1, 1, 1, rows_per_col * K],
            strides=[0, 0, 0, 1],
        )
        tap_k = TensorAccessPattern(
            tensor_dims=L3_W_ty.__args__[0],
            offset=k_offset,
            sizes=[1, 1, 1, rows_per_col * K],
            strides=[0, 0, 0, 1],
        )
        W_taps.append((tap_q, tap_k))

    # Input vector TAP: broadcast to all columns (single read)
    X_tap = TensorAccessPattern(
        tensor_dims=L3_X_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, K],
        strides=[0, 0, 0, 1],
    )

    # Output TAPs
    Cq_taps = [
        TensorAccessPattern(
            tensor_dims=L3_Cq_ty.__args__[0],
            offset=col * rows_per_col,
            sizes=[1, 1, 1, rows_per_col],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]
    Ck_taps = [
        TensorAccessPattern(
            tensor_dims=L3_Ck_ty.__args__[0],
            offset=col * rows_per_col,
            sizes=[1, 1, 1, rows_per_col],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]

    # Runtime sequence
    rt = Runtime()
    with rt.sequence(L3_W_ty, L3_X_ty, L3_Cq_ty, L3_Ck_ty) as (W, X, Cq, Ck):
        rt.start(*workers)

        # Fill input vector to all columns (ONCE)
        tg_x = rt.task_group()
        for col in range(cols):
            rt.fill(X_fifos[col].prod(), X, X_tap, task_group=tg_x)

        # Fill weights: Q weights then K weights through same FIFO
        tg_w = rt.task_group()
        for col in range(cols):
            rt.fill(W_fifos[col].prod(), W, W_taps[col][0], task_group=tg_w)
            rt.fill(W_fifos[col].prod(), W, W_taps[col][1], task_group=tg_w)

        # Drain outputs
        tg_out = rt.task_group()
        for col in range(cols):
            rt.drain(Cq_fifos[col].cons(), Cq, Cq_taps[col], task_group=tg_out)
            rt.drain(Ck_fifos[col].cons(), Ck, Ck_taps[col], task_group=tg_out, wait=True)

        rt.finish_task_group(tg_x)
        rt.finish_task_group(tg_w)
        rt.finish_task_group(tg_out)

    return Program(dev, rt).resolve_program(SequentialPlacer())
