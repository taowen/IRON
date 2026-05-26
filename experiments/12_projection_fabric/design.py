"""
Experiment 12: Projection Fabric — Multi-Tile Phase-Reuse Q4NX

Proves the complete projection fabric pattern:
- 4 tiles (2 cols x 2 rows), all running identical Q4NX kernel
- Activation acquired ONCE, held across 2 projections (replay)
- Phase reuse: same tiles execute projA then projB via weight DMA queue
- Weight distribution: per-tile TAPs into shared weight BO
- FP32 accumulation across 4 chunks per projection per tile
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


M_PER_TILE = 32
NUM_TILES = 4
K = 1024
K_CHUNK = 256
NUM_CHUNKS = K // K_CHUNK
GROUP_SIZE = 32
NUM_PROJECTIONS = 2
TOTAL_OUTPUT = NUM_TILES * M_PER_TILE * NUM_PROJECTIONS


def projection_fabric_pipeline(dev):
    groups_per_row = K_CHUNK // GROUP_SIZE
    chunk_bytes = M_PER_TILE * groups_per_row * 2 + M_PER_TILE * groups_per_row * 2 + M_PER_TILE * K_CHUNK // 2
    chunk_bf16_elems = chunk_bytes // 2  # 2560

    # Per-chunk weight type (what arrives via weight FIFO)
    L1_wt_chunk_ty = np.ndarray[(chunk_bf16_elems,), np.dtype[bfloat16]]
    # Full activation held in tile (K=1024 bf16)
    L1_act_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    # Output per projection (M_PER_TILE=32 bf16)
    L1_out_ty = np.ndarray[(M_PER_TILE,), np.dtype[bfloat16]]

    # L3 (DDR) buffer types
    per_tile_wt_elems = NUM_PROJECTIONS * NUM_CHUNKS * chunk_bf16_elems  # 20480
    total_wt_elems = NUM_TILES * per_tile_wt_elems  # 81920
    L3_wt_ty = np.ndarray[(total_wt_elems,), np.dtype[bfloat16]]
    L3_act_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    L3_out_ty = np.ndarray[(TOTAL_OUTPUT,), np.dtype[bfloat16]]

    # Kernels (shared object file, all tiles use same binary)
    accum_kernel = Kernel(
        "q4nx_chunk_accum_offset",
        "q4nx_chunk_accum.o",
        [L1_wt_chunk_ty, L1_act_ty, np.int32, np.int32],
    )

    flush_kernel = Kernel(
        "q4nx_flush_output",
        "q4nx_chunk_accum.o",
        [L1_out_ty, np.int32],
    )

    # Per-tile FIFOs
    wt_fifos = [ObjectFifo(L1_wt_chunk_ty, name=f"wt_{i}", depth=2) for i in range(NUM_TILES)]
    act_fifos = [ObjectFifo(L1_act_ty, name=f"act_{i}", depth=2) for i in range(NUM_TILES)]
    out_fifos = [ObjectFifo(L1_out_ty, name=f"out_{i}", depth=2) for i in range(NUM_TILES)]

    # Worker body — identical for all 4 tiles
    def core_body(act_fifo, wt_fifo, out_fifo, accum_k, flush_k):
        for _ in range_(0xFFFFFFFF):
            # Acquire activation ONCE — hold across BOTH projections
            act = act_fifo.acquire(1)

            for _ in range_(NUM_PROJECTIONS):
                out = out_fifo.acquire(1)
                for chunk_idx in range_(NUM_CHUNKS):
                    chunk_i32 = index.casts(T.i32(), chunk_idx)
                    act_offset = chunk_i32 * K_CHUNK
                    wt = wt_fifo.acquire(1)
                    accum_k(wt, act, act_offset, M_PER_TILE)
                    wt_fifo.release(1)
                flush_k(out, M_PER_TILE)
                out_fifo.release(1)

            # Release activation ONLY after both projections complete
            act_fifo.release(1)

    # Create 4 workers with separate FIFOs
    workers = []
    for i in range(NUM_TILES):
        w = Worker(
            core_body,
            [act_fifos[i].cons(), wt_fifos[i].cons(), out_fifos[i].prod(),
             accum_kernel, flush_kernel],
        )
        workers.append(w)

    # --- TAPs ---

    # Activation TAP: single K=1024 transfer (same for all tiles)
    act_tap = TensorAccessPattern(
        tensor_dims=L3_act_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, K],
        strides=[0, 0, 0, 1],
    )

    # Weight TAPs: per-tile, different offsets into shared weight BO
    # Layout: [tile0_proj0_chunks | tile0_proj1_chunks | tile1_proj0_chunks | ...]
    # Each tile section: NUM_PROJ * NUM_CHUNKS * chunk_bf16_elems = 2 * 4 * 2560 = 20480 bf16
    # Per-tile TAP sends: NUM_PROJ * NUM_CHUNKS chunks = 8 chunks, factored as [2, 4, 10, 256]
    wt_taps = []
    for i in range(NUM_TILES):
        tap = TensorAccessPattern(
            tensor_dims=L3_wt_ty.__args__[0],
            offset=i * per_tile_wt_elems,
            sizes=[NUM_PROJECTIONS, NUM_CHUNKS, 10, 256],
            strides=[NUM_CHUNKS * chunk_bf16_elems, chunk_bf16_elems, 256, 1],
        )
        wt_taps.append(tap)

    # Output TAPs: per-tile, different offsets into shared output BO
    # Layout: [tile0_proj0(32) | tile0_proj1(32) | tile1_proj0(32) | ... | tile3_proj1(32)]
    out_taps = []
    for i in range(NUM_TILES):
        tap = TensorAccessPattern(
            tensor_dims=L3_out_ty.__args__[0],
            offset=i * M_PER_TILE * NUM_PROJECTIONS,
            sizes=[1, 1, NUM_PROJECTIONS, M_PER_TILE],
            strides=[0, 0, M_PER_TILE, 1],
        )
        out_taps.append(tap)

    # --- Runtime Sequence ---
    rt = Runtime()
    with rt.sequence(L3_wt_ty, L3_act_ty, L3_out_ty) as (wt_buf, act_buf, out_buf):
        rt.start(*workers)

        # Fill activation to each tile (same data, separate FIFOs)
        tg_act = rt.task_group()
        for i in range(NUM_TILES):
            rt.fill(act_fifos[i].prod(), act_buf, act_tap, task_group=tg_act)

        # Fill weights to each tile (different offsets)
        tg_wt = rt.task_group()
        for i in range(NUM_TILES):
            rt.fill(wt_fifos[i].prod(), wt_buf, wt_taps[i], task_group=tg_wt)

        # Drain outputs from each tile (different offsets)
        tg_out = rt.task_group()
        for i in range(NUM_TILES):
            rt.drain(out_fifos[i].cons(), out_buf, out_taps[i],
                     task_group=tg_out, wait=(i == NUM_TILES - 1))

        rt.finish_task_group(tg_act)
        rt.finish_task_group(tg_wt)
        rt.finish_task_group(tg_out)

    return Program(dev, rt).resolve_program(SequentialPlacer())
