"""
Experiment 03: Phase Replay — 3-Phase GEMV with Tile-to-Tile Input Hold + Tile-to-Tile Outputs

Goal: Verify whether IRON can express the core fused layer projection pattern:
  1. Norm tile produces normed_hidden to an INTERNAL ObjectFifo
  2. Projection tile HOLDS normed_hidden across 3 sequential GEMV phases (Q/K/V)
  3. Each phase's output goes to a DIFFERENT tile-to-tile FIFO (not DDR)
  4. A collector tile consumes all 3 outputs and drains to DDR
  5. The same weight FIFO carries 3 different weight sets sequentially

This combines Exp 01 (multi-phase hold) + Exp 02 (tile-to-tile intermediate) + NEW:
  - Multiple tile-to-tile OUTPUT FIFOs from same worker
  - 3 sequential DMA fills to same weight FIFO
  - 4-tile heterogeneous pipeline (norm → projection → collector)

Topology:
  DDR(hidden) → [Norm Tile 0,2] → normed (tile-to-tile) → [Projection Tile 0,3]
  DDR(W_q,W_k,W_v) → W (same FIFO, 3 fills) ──────────────→ [Projection Tile 0,3]
  [Projection Tile 0,3] → Q_out (tile-to-tile) → [Collector Tile 0,4]
  [Projection Tile 0,3] → K_out (tile-to-tile) → [Collector Tile 0,4]
  [Projection Tile 0,3] → V_out (tile-to-tile) → [Collector Tile 0,4]
  [Collector Tile 0,4] → result → DDR(output)

Key test: if DMA channel exhaustion (only 2 MM2S per tile) blocks 3 output FIFOs,
IRON cannot express the fused layer pattern without manual BD management.
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def phase_replay_pipeline(
    dev,
    K,          # hidden dimension (input vector length)
    M_q,        # Q projection output rows
    M_k,        # K projection output rows
    M_v,        # V projection output rows
    m_input,    # GEMV tile size (rows per kernel call)
):
    assert M_q % m_input == 0
    assert M_k % m_input == 0
    assert M_v % m_input == 0

    dtype = np.dtype[bfloat16]

    # L1 types
    L1_hidden_ty = np.ndarray[(K,), dtype]
    L1_normed_ty = np.ndarray[(K,), dtype]
    L1_W_ty = np.ndarray[(m_input, K), dtype]
    L1_Cq_ty = np.ndarray[(M_q,), dtype]
    L1_Ck_ty = np.ndarray[(M_k,), dtype]
    L1_Cv_ty = np.ndarray[(M_v,), dtype]
    L1_result_ty = np.ndarray[(M_q + M_k + M_v,), dtype]

    # L3 (DDR) types
    L3_hidden_ty = np.ndarray[(K,), dtype]
    # All weights packed sequentially: W_q[M_q*K] + W_k[M_k*K] + W_v[M_v*K]
    total_weight_elems = (M_q + M_k + M_v) * K
    L3_W_ty = np.ndarray[(total_weight_elems,), dtype]
    L3_C_ty = np.ndarray[(M_q + M_k + M_v,), dtype]

    # Kernel
    matvec = Kernel(
        "matvec_vectorized_bf16_bf16",
        "mv.o",
        [np.int32, np.int32, L1_W_ty, L1_normed_ty, L1_Cq_ty],
    )

    # === ObjectFifos ===

    # DDR → norm tile
    hidden_fifo = ObjectFifo(L1_hidden_ty, name="hidden_in", depth=2)

    # norm tile → projection tile (INTERNAL, no DDR)
    normed_fifo = ObjectFifo(L1_normed_ty, name="normed", depth=2)

    # DDR → projection tile: weight stream (carries Q, K, V weights sequentially)
    W_fifo = ObjectFifo(L1_W_ty, name="W", depth=2)

    # projection tile → collector tile (INTERNAL, no DDR)
    Q_out_fifo = ObjectFifo(L1_Cq_ty, name="Q_out", depth=2)
    K_out_fifo = ObjectFifo(L1_Ck_ty, name="K_out", depth=2)
    V_out_fifo = ObjectFifo(L1_Cv_ty, name="V_out", depth=2)

    # collector tile → DDR
    result_fifo = ObjectFifo(L1_result_ty, name="result", depth=2)

    # === Workers ===

    def norm_worker_body(hidden_in, normed_out):
        for _ in range_(0xFFFFFFFF):
            h = hidden_in.acquire(1)
            n = normed_out.acquire(1)
            # Passthrough: structural test (real impl would do RMSNorm)
            normed_out.release(1)
            hidden_in.release(1)

    norm_worker = Worker(
        norm_worker_body,
        [hidden_fifo.cons(), normed_fifo.prod()],
    )

    def projection_worker_body(normed_in, W_in, q_out, k_out, v_out, matvec_kernel):
        for _ in range_(0xFFFFFFFF):
            # Acquire normed ONCE — hold across all 3 phases
            x = normed_in.acquire(1)

            # Phase Q
            cq = q_out.acquire(1)
            for i_idx in range_(M_q // m_input):
                i_i32 = index.casts(T.i32(), i_idx)
                row_offset = i_i32 * m_input
                w = W_in.acquire(1)
                matvec_kernel(m_input, row_offset, w, x, cq)
                W_in.release(1)
            q_out.release(1)

            # Phase K — x is STILL held
            ck = k_out.acquire(1)
            for i_idx in range_(M_k // m_input):
                i_i32 = index.casts(T.i32(), i_idx)
                row_offset = i_i32 * m_input
                w = W_in.acquire(1)
                matvec_kernel(m_input, row_offset, w, x, ck)
                W_in.release(1)
            k_out.release(1)

            # Phase V — x is STILL held
            cv = v_out.acquire(1)
            for i_idx in range_(M_v // m_input):
                i_i32 = index.casts(T.i32(), i_idx)
                row_offset = i_i32 * m_input
                w = W_in.acquire(1)
                matvec_kernel(m_input, row_offset, w, x, cv)
                W_in.release(1)
            v_out.release(1)

            # Release normed ONLY after all 3 phases complete
            normed_in.release(1)

    projection_worker = Worker(
        projection_worker_body,
        [
            normed_fifo.cons(),
            W_fifo.cons(),
            Q_out_fifo.prod(),
            K_out_fifo.prod(),
            V_out_fifo.prod(),
            matvec,
        ],
    )

    def collector_worker_body(q_in, k_in, v_in, result_out):
        for _ in range_(0xFFFFFFFF):
            # Collect Q, K, V results and pack into output buffer
            q = q_in.acquire(1)
            k = k_in.acquire(1)
            v = v_in.acquire(1)
            r = result_out.acquire(1)
            # The collector just passes data through (real impl would route to attention)
            result_out.release(1)
            v_in.release(1)
            k_in.release(1)
            q_in.release(1)

    collector_worker = Worker(
        collector_worker_body,
        [
            Q_out_fifo.cons(),
            K_out_fifo.cons(),
            V_out_fifo.cons(),
            result_fifo.prod(),
        ],
    )

    # === Runtime Sequence ===
    # Only DDR connections: hidden, weights, output
    # normed, Q_out, K_out, V_out are all INTERNAL

    # Weight TAPs: Q weights at offset 0, K weights after Q, V weights after K
    W_q_tap = TensorAccessPattern(
        tensor_dims=L3_W_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, M_q * K],
        strides=[0, 0, 0, 1],
    )
    W_k_tap = TensorAccessPattern(
        tensor_dims=L3_W_ty.__args__[0],
        offset=M_q * K,
        sizes=[1, 1, 1, M_k * K],
        strides=[0, 0, 0, 1],
    )
    W_v_tap = TensorAccessPattern(
        tensor_dims=L3_W_ty.__args__[0],
        offset=(M_q + M_k) * K,
        sizes=[1, 1, 1, M_v * K],
        strides=[0, 0, 0, 1],
    )

    hidden_tap = TensorAccessPattern(
        tensor_dims=L3_hidden_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, K],
        strides=[0, 0, 0, 1],
    )

    result_tap = TensorAccessPattern(
        tensor_dims=L3_C_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, M_q + M_k + M_v],
        strides=[0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(L3_hidden_ty, L3_W_ty, L3_C_ty) as (hidden_buf, W_buf, C_buf):
        rt.start(norm_worker, projection_worker, collector_worker)

        # Fill hidden → norm tile
        tg_h = rt.task_group()
        rt.fill(hidden_fifo.prod(), hidden_buf, hidden_tap, task_group=tg_h)

        # Fill weights sequentially: Q, then K, then V — all to same W FIFO
        tg_w = rt.task_group()
        rt.fill(W_fifo.prod(), W_buf, W_q_tap, task_group=tg_w)
        rt.fill(W_fifo.prod(), W_buf, W_k_tap, task_group=tg_w)
        rt.fill(W_fifo.prod(), W_buf, W_v_tap, task_group=tg_w)

        # Drain result from collector
        tg_c = rt.task_group()
        rt.drain(result_fifo.cons(), C_buf, result_tap, task_group=tg_c, wait=True)

        rt.finish_task_group(tg_h)
        rt.finish_task_group(tg_w)
        rt.finish_task_group(tg_c)

    return Program(dev, rt).resolve_program(SequentialPlacer())
