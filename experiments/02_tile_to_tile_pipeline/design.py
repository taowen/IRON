"""
Experiment 02: Tile-to-Tile Pipeline — Norm → Projection without DDR intermediate

Goal: Verify whether IRON can express a two-stage pipeline where:
  1. Worker A (norm tile) computes RMSNorm(hidden)
  2. Worker B (projection tile) consumes the norm output directly via
     tile-to-tile ObjectFifo, does a GEMV, and drains result to DDR.
  3. The intermediate (normalized hidden) NEVER touches DDR.

This is the fundamental pattern of the fused layer engine:
  - host sends hidden to AIE once
  - norm tile computes RMSNorm
  - norm output goes to memtile, replays to projection tiles
  - projection output either drains to DDR or feeds next stage

What this tests that Experiment 01 did NOT:
  - Two DIFFERENT Workers with different roles in the same graph
  - ObjectFifo connecting tile-to-tile (prod=compute tile, cons=compute tile)
  - The intermediate never appears as a runtime_sequence argument
  - Memtile forward/broadcast pattern (one norm → multiple projection tiles)
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


def norm_projection_pipeline(
    dev,
    cols,           # number of projection columns
    K,              # hidden dimension (input to norm and projection)
    M,              # projection output dimension
    m_input,        # GEMV tile size (rows per kernel call)
    func_prefix="",
):
    """
    Two-stage pipeline:
      Stage 1: Norm worker reads hidden[K] from DDR, computes elementwise norm,
               produces normalized[K] to an INTERNAL ObjectFifo.
      Stage 2: Projection workers consume normalized[K] from the internal FIFO
               (via memtile broadcast), compute GEMV, drain output to DDR.

    The internal FIFO (norm_output) is NOT connected to any DDR buffer in the
    runtime sequence — it is purely tile-to-tile.
    """

    assert M % cols == 0

    dtype = np.dtype[bfloat16]

    # L1 types
    L1_hidden_ty = np.ndarray[(K,), dtype]          # input to norm
    L1_normed_ty = np.ndarray[(K,), dtype]          # norm output = projection input
    L1_W_ty = np.ndarray[(m_input, K), dtype]       # weight tile
    L1_C_ty = np.ndarray[(M // cols,), dtype]       # projection output per col

    # L3 (DDR) types — note: normed hidden is NOT here
    L3_hidden_ty = np.ndarray[(K,), dtype]
    L3_W_ty = np.ndarray[(M * K,), dtype]
    L3_C_ty = np.ndarray[(M,), dtype]

    # Kernels
    # Norm kernel: simple elementwise scale (approximation of RMSNorm for this test)
    # We use the same matvec kernel with m_input=1 as a passthrough/scale,
    # but actually let's write a simpler "copy" to prove the pipeline.
    # For this structural test, the norm worker just does an acquire-release
    # forwarding pattern (the compute kernel doesn't matter for proving dataflow).

    matvec = Kernel(
        f"{func_prefix}matvec_vectorized_bf16_bf16",
        f"{func_prefix}mv.o",
        [np.int32, np.int32, L1_W_ty, L1_normed_ty, L1_C_ty],
    )

    # === ObjectFifos ===

    # DDR → norm worker: hidden input
    hidden_fifo = ObjectFifo(L1_hidden_ty, name="hidden_in", depth=2)

    # norm worker → projection workers: INTERNAL, no DDR connection
    # This is the key: producer is the norm tile, consumers are projection tiles
    # Using .forward() through memtile for broadcast to multiple consumers
    normed_fifo = ObjectFifo(L1_normed_ty, name="normed", depth=2)

    # DDR → projection workers: weight input
    W_fifos = [
        ObjectFifo(L1_W_ty, name=f"W_{i}", depth=2) for i in range(cols)
    ]

    # projection workers → DDR: output
    C_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_{i}", depth=2) for i in range(cols)
    ]

    # === Workers ===

    def norm_worker_body(hidden_in, normed_out):
        """
        Norm worker: reads hidden from DDR fifo, writes to internal normed fifo.
        For this structural test, it just copies (real impl would do RMSNorm).
        """
        for _ in range_(0xFFFFFFFF):
            h = hidden_in.acquire(1)
            n = normed_out.acquire(1)
            # In a real implementation, this would call an RMSNorm kernel.
            # For the structural test, we need some operation here.
            # We'll use a memcpy-like approach: the point is proving the FIFO connection.
            normed_out.release(1)
            hidden_in.release(1)

    norm_worker = Worker(
        norm_worker_body,
        [hidden_fifo.cons(), normed_fifo.prod()],
    )

    # Projection workers: consume normed from internal fifo, weights from DDR
    def projection_worker_body(normed_in, W_in, C_out, matvec_kernel):
        """
        Projection worker: reads normalized hidden from INTERNAL fifo (not DDR),
        reads weights from DDR fifo, computes GEMV, writes output to DDR fifo.
        """
        for _ in range_(0xFFFFFFFF):
            x = normed_in.acquire(1)
            c = C_out.acquire(1)
            for i_idx in range_((M // cols) // m_input):
                i_i32 = index.casts(T.i32(), i_idx)
                row_offset = i_i32 * m_input
                w = W_in.acquire(1)
                matvec_kernel(m_input, row_offset, w, x, c)
                W_in.release(1)
            C_out.release(1)
            normed_in.release(1)

    # Create multiple consumer handles for the normed_fifo (broadcast)
    projection_workers = [
        Worker(
            projection_worker_body,
            [
                normed_fifo.cons(),  # each projection worker gets a consumer handle
                W_fifos[i].cons(),
                C_fifos[i].prod(),
                matvec,
            ],
        )
        for i in range(cols)
    ]

    # === Runtime Sequence ===
    # Key: normed_fifo is NOT mentioned here — it's purely internal

    # Weight distribution TAPs
    rows_per_col = M // cols
    W_taps = [
        TensorAccessPattern(
            tensor_dims=L3_W_ty.__args__[0],
            offset=col * rows_per_col * K,
            sizes=[1, 1, 1, rows_per_col * K],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]

    # Hidden input TAP
    hidden_tap = TensorAccessPattern(
        tensor_dims=L3_hidden_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, K],
        strides=[0, 0, 0, 1],
    )

    # Output TAPs
    C_taps = [
        TensorAccessPattern(
            tensor_dims=L3_C_ty.__args__[0],
            offset=col * rows_per_col,
            sizes=[1, 1, 1, rows_per_col],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]

    rt = Runtime()
    with rt.sequence(L3_hidden_ty, L3_W_ty, L3_C_ty) as (hidden_buf, W_buf, C_buf):
        rt.start(norm_worker, *projection_workers)

        # Fill hidden from DDR to norm worker
        tg_h = rt.task_group()
        rt.fill(hidden_fifo.prod(), hidden_buf, hidden_tap, task_group=tg_h)

        # Fill weights from DDR to projection workers
        tg_w = rt.task_group()
        for col in range(cols):
            rt.fill(W_fifos[col].prod(), W_buf, W_taps[col], task_group=tg_w)

        # Drain outputs from projection workers to DDR
        tg_c = rt.task_group()
        for col in range(cols):
            rt.drain(
                C_fifos[col].cons(), C_buf, C_taps[col],
                task_group=tg_c, wait=True,
            )

        rt.finish_task_group(tg_h)
        rt.finish_task_group(tg_w)
        rt.finish_task_group(tg_c)

    return Program(dev, rt).resolve_program(SequentialPlacer())
