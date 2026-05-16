# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, Buffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern


def main():
    module = sage_attention(dev=NPU2(), S_q=128, S_kv=128, d=64)
    Path("sage_attention.mlir").write_text(str(module))


def _legalize_tap(tap: TensorAccessPattern, max_dim_size: int):
    sizes = list(tap._sizes)
    if all(size <= max_dim_size for size in sizes):
        return tap

    for idx, stride in enumerate(tap._strides[:-1]):
        if stride != 0 and stride != tap._sizes[idx + 1]:
            raise ValueError("Cannot legalize non-contiguous DMA transfer")
    if tap._strides[-1] != 1:
        raise ValueError("Cannot legalize non-contiguous DMA transfer")

    tap._sizes = [1, 1, 1, math.prod(sizes)]
    tap._strides = [0, 0, 0, 1]
    return tap


def _legalize_tas(tas):
    for tap in tas:
        _legalize_tap(tap, 1023)


def sage_attention(
    dev,
    S_q: int,
    S_kv: int,
    d: int,
    B_q: int = 64,
    B_kv: int = 64,
    num_pipelines: int = 2,
    trace_size: int = 0,
    verbose: bool = False,
):
    if not isinstance(dev, NPU2):
        raise ValueError("SageAttention bring-up currently supports NPU2 only")
    if d != 64:
        raise ValueError(f"SageAttention bring-up currently supports d=64, got {d}")
    if B_q != 64 or B_kv != 64:
        raise ValueError("SageAttention bring-up currently supports 64x64 tiles")
    if num_pipelines < 1 or num_pipelines > 2:
        raise ValueError("SageAttention bring-up currently supports 1 or 2 pipelines")

    of_depth = 2
    dtype = bfloat16
    r = s = t = 8

    q_group = B_q * num_pipelines
    S_q_pad = ((S_q + q_group - 1) // q_group) * q_group
    S_kv_pad = ((S_kv + B_kv - 1) // B_kv) * B_kv
    num_q_blocks = S_q_pad // B_q
    num_kv_blocks = S_kv_pad // B_kv
    num_q_block_per_pipeline = num_q_blocks // num_pipelines

    if verbose:
        print(f"SageAttention S_q={S_q} S_kv={S_kv} d={d}")
        print(f"Padded S_q={S_q_pad} S_kv={S_kv_pad}")

    q_l3_ty = np.ndarray[(S_q_pad, d), np.dtype[np.int8]]
    k_l3_ty = np.ndarray[(S_kv_pad, d), np.dtype[np.int8]]
    v_l3_ty = np.ndarray[(S_kv_pad, d), np.dtype[dtype]]
    scale_l3_ty = np.ndarray[(num_q_blocks, num_kv_blocks), np.dtype[np.float32]]
    o_l3_ty = np.ndarray[(S_q_pad, d), np.dtype[dtype]]

    q_i8_ty = np.ndarray[(B_q, d), np.dtype[np.int8]]
    k_i8_ty = np.ndarray[(d, B_kv), np.dtype[np.int8]]
    v_bf16_ty = np.ndarray[(d, B_kv), np.dtype[dtype]]
    qk_bf16_ty = np.ndarray[(B_q, B_kv), np.dtype[dtype]]
    qk_i32_ty = np.ndarray[(B_q, B_kv), np.dtype[np.int32]]
    scale_ty = np.ndarray[(4 * B_q,), np.dtype[dtype]]
    dequant_scale_ty = np.ndarray[(num_kv_blocks,), np.dtype[np.float32]]
    q_group_i8_ty = np.ndarray[(num_pipelines * B_q, d), np.dtype[np.int8]]
    o_group_ty = np.ndarray[(num_pipelines * B_q, d), np.dtype[dtype]]
    o_tile_ty = np.ndarray[(B_q, d), np.dtype[dtype]]

    q_dims = [(B_q // r, r * d), (d // s, s), (r, d), (s, 1)]
    k_dims = [(B_kv // t, t * d), (d // s, s), (t, d), (s, 1)]
    v_dims = [(B_kv // s, s * B_kv), (B_kv // t, t), (s, B_kv), (t, 1)]
    a_dims = [(B_q // r, r * B_kv), (r, t), (B_kv // t, r * t), (t, 1)]

    inQ = ObjectFifo(q_group_i8_ty, name="sage_inQ", depth=of_depth)
    memQ = inQ.cons().split(
        offsets=[B_q * d * i for i in range(num_pipelines)],
        obj_types=[q_i8_ty] * num_pipelines,
        names=[f"sage_memQ{i}" for i in range(num_pipelines)],
        dims_to_stream=[q_dims] * num_pipelines,
        placement=Tile(col=6, row=1),
        depths=[of_depth] * num_pipelines,
    )

    inK = ObjectFifo(k_i8_ty, name="sage_inK", depth=of_depth)
    memK = inK.cons().forward(
        name="sage_memK",
        dims_to_stream=k_dims,
        placement=Tile(col=3, row=1),
        depth=of_depth,
    )

    inV = ObjectFifo(v_bf16_ty, name="sage_inV", depth=of_depth)
    memV = inV.cons().forward(
        name="sage_memV",
        dims_to_stream=v_dims,
        placement=Tile(col=4, row=1),
        depth=of_depth,
    )

    memA = []
    outA = []
    for i in range(num_pipelines):
        memA.append(ObjectFifo(qk_i32_ty, depth=of_depth, name=f"sage_memA{i}"))
        outA.append(
            memA[i]
            .cons()
            .forward(
                name=f"sage_outA{i}",
                dims_to_stream=a_dims,
                depth=of_depth,
            )
        )

    memP = []
    outP = []
    scaleOF = []
    inDequantScale = []
    for i in range(num_pipelines):
        memP.append(ObjectFifo(qk_bf16_ty, depth=of_depth, name=f"sage_memP{i}"))
        outP.append(
            memP[i]
            .cons()
            .forward(
                name=f"sage_outP{i}",
                dims_to_stream=q_dims,
                depth=of_depth,
            )
        )
        scaleOF.append(ObjectFifo(scale_ty, depth=of_depth, name=f"sage_scaleOF{i}"))
        inDequantScale.append(
            ObjectFifo(dequant_scale_ty, depth=of_depth, name=f"sage_inDequantScale{i}")
        )

    memO = ObjectFifo(
        o_group_ty,
        name="sage_memO",
        dims_to_stream=a_dims,
    )
    outO = memO.prod().join(
        offsets=[B_q * d * i for i in range(num_pipelines)],
        obj_types=[o_tile_ty] * num_pipelines,
        names=[f"sage_outO{i}" for i in range(num_pipelines)],
        depths=[of_depth] * num_pipelines,
        placement=Tile(col=6, row=1),
    )

    qk_kernel = Kernel(
        "matmul_i8_i32_wrapper",
        "sage_attention.o",
        [
            q_i8_ty,
            k_i8_ty,
            qk_i32_ty,
            np.ndarray[(2,), np.dtype[np.int32]],
        ],
    )
    zero_kernel = Kernel("zero_bf16", "sage_attention.o", [qk_bf16_ty])
    partial_softmax_kernel = Kernel(
        "partial_softmax_i32_dequant",
        "sage_attention.o",
        [
            qk_i32_ty,
            qk_bf16_ty,
            qk_bf16_ty,
            scale_ty,
            np.ndarray[(2,), np.dtype[np.int32]],
            dequant_scale_ty,
            dtype,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    scale_buffer_init_kernel = Kernel(
        "init_scale_buffer", "sage_attention.o", [scale_ty, np.int32]
    )
    memcopy_kernel_scale = Kernel(
        "passThroughLine",
        "sage_attention_passThrough.o",
        [scale_ty, scale_ty, np.int32],
    )
    matmul_PV = Kernel(
        "matmul_PV",
        "sage_attention.o",
        [
            qk_bf16_ty,
            v_bf16_ty,
            qk_bf16_ty,
            scale_ty,
            np.int32,
            np.int32,
            np.ndarray[(2,), np.dtype[np.int32]],
        ],
    )
    rescale_O = Kernel(
        "rescale_O",
        "sage_attention.o",
        [qk_bf16_ty, scale_ty, np.int32, np.ndarray[(2,), np.dtype[np.int32]]],
    )

    def qk_worker(
        of_q,
        of_k,
        of_a_out,
        qk,
        q_block_bias,
        idx_buffer,
    ):
        idx_buffer[1] = q_block_bias
        for _ in range_(num_q_block_per_pipeline):
            idx_buffer[0] = 0
            elem_q = of_q.acquire(1)
            for _ in range_(num_kv_blocks):
                elem_k = of_k.acquire(1)
                elem_a = of_a_out.acquire(1)
                qk(
                    elem_q,
                    elem_k,
                    elem_a,
                    idx_buffer,
                )
                of_k.release(1)
                of_a_out.release(1)
                idx_buffer[0] += 1
            of_q.release(1)
            idx_buffer[1] += num_pipelines

    def softmax_worker(
        of_in_a,
        of_dequant_scale,
        of_out_p,
        of_out_scale,
        partial_softmax,
        init_scale_buffer,
        memcopy_scale,
        q_block_bias,
        idx_buffer,
        logits,
        scale_buffer,
    ):
        idx_buffer[1] = q_block_bias
        for _ in range_(num_q_block_per_pipeline):
            idx_buffer[0] = 0
            init_scale_buffer(scale_buffer, B_q)
            elem_dequant_scale = of_dequant_scale.acquire(1)
            for _ in range_(num_kv_blocks):
                elem_p = of_out_p.acquire(1)
                elem_a = of_in_a.acquire(1)
                elem_scale = of_out_scale.acquire(1)
                partial_softmax(
                    elem_a,
                    logits,
                    elem_p,
                    scale_buffer,
                    idx_buffer,
                    elem_dequant_scale,
                    (1 / np.sqrt(d)) * 1.4453125,
                    B_q,
                    B_kv,
                    S_q,
                    S_kv,
                )
                memcopy_scale(scale_buffer, elem_scale, 4 * B_q)
                of_in_a.release(1)
                of_out_p.release(1)
                of_out_scale.release(1)
                idx_buffer[0] += 1
            of_dequant_scale.release(1)
            idx_buffer[1] += num_pipelines

    def pv_worker(
        of_p,
        of_v,
        of_scale,
        of_o_out,
        zero,
        matmul_pv,
        rescale_o,
        q_block_bias,
        idx_buffer,
    ):
        idx_buffer[1] = q_block_bias
        for _ in range_(num_q_block_per_pipeline):
            idx_buffer[0] = 0
            elem_o = of_o_out.acquire(1)
            zero(elem_o)

            elem_p = of_p.acquire(1)
            elem_v = of_v.acquire(1)
            elem_scale = of_scale.acquire(1)
            matmul_pv(elem_p, elem_v, elem_o, elem_scale, B_q, 0, idx_buffer)
            of_p.release(1)
            of_v.release(1)
            of_scale.release(1)
            idx_buffer[0] += 1

            if num_kv_blocks > 2:
                for _ in range_(num_kv_blocks - 2):
                    elem_p = of_p.acquire(1)
                    elem_v = of_v.acquire(1)
                    elem_scale = of_scale.acquire(1)
                    matmul_pv(elem_p, elem_v, elem_o, elem_scale, B_q, 1, idx_buffer)
                    of_p.release(1)
                    of_v.release(1)
                    of_scale.release(1)
                    idx_buffer[0] += 1

            if num_kv_blocks > 1:
                elem_p = of_p.acquire(1)
                elem_v = of_v.acquire(1)
                elem_scale = of_scale.acquire(1)
                matmul_pv(elem_p, elem_v, elem_o, elem_scale, B_q, 1, idx_buffer)
                rescale_o(elem_o, elem_scale, B_q, idx_buffer)
                of_p.release(1)
                of_v.release(1)
                of_scale.release(1)
                idx_buffer[0] += 1
            else:
                rescale_o(elem_o, elem_scale, B_q, idx_buffer)
                idx_buffer[0] += 1

            of_o_out.release(1)
            idx_buffer[1] += num_pipelines

    qk_workers = []
    softmax_workers = []
    pv_workers = []
    for i in range(num_pipelines):
        qk_idx = Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"sage_idx_qk_{i}",
        )
        qk_workers.append(
            Worker(
                qk_worker,
                [
                    memQ[i].cons(),
                    memK.cons(),
                    memA[i].prod(),
                    qk_kernel,
                    i,
                    qk_idx,
                ],
                stack_size=0xD00,
                placement=Tile(col=i, row=2),
            )
        )

        softmax_idx = Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"sage_idx_softmax_{i}",
        )
        softmax_scale = Buffer(
            initial_value=np.zeros(shape=(4 * B_q,), dtype=dtype),
            name=f"sage_scale_buffer_{i}",
        )
        softmax_logits = Buffer(
            initial_value=np.zeros(shape=(B_q, B_kv), dtype=dtype),
            name=f"sage_logits_buffer_{i}",
        )
        softmax_workers.append(
            Worker(
                softmax_worker,
                [
                    outA[i].cons(),
                    inDequantScale[i].cons(),
                    memP[i].prod(),
                    scaleOF[i].prod(),
                    partial_softmax_kernel,
                    scale_buffer_init_kernel,
                    memcopy_kernel_scale,
                    i,
                    softmax_idx,
                    softmax_logits,
                    softmax_scale,
                ],
                stack_size=0xD00,
                placement=Tile(col=i, row=3),
            )
        )

        pv_idx = Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"sage_idx_pv_{i}",
        )
        pv_workers.append(
            Worker(
                pv_worker,
                [
                    outP[i].cons(),
                    memV.cons(),
                    scaleOF[i].cons(),
                    outO[i].prod(),
                    zero_kernel,
                    matmul_PV,
                    rescale_O,
                    i,
                    pv_idx,
                ],
                stack_size=0xD00,
                placement=Tile(col=i, row=4),
            )
        )

    Q_tiles = TensorTiler2D.group_tiler((S_q_pad, d), (num_pipelines * B_q, d), (1, 1))
    K_tiles = TensorTiler2D.group_tiler((S_kv_pad, d), (S_kv_pad, d), (1, 1))
    V_tiles = TensorTiler2D.group_tiler((S_kv_pad, d), (S_kv_pad, d), (1, 1))
    scale_tiles = TensorTiler2D.group_tiler(
        (num_q_blocks, num_kv_blocks), (1, num_kv_blocks), (1, 1)
    )
    O_tiles = TensorTiler2D.group_tiler((S_q_pad, d), (num_pipelines * B_q, d), (1, 1))
    _legalize_tas(K_tiles)
    _legalize_tas(V_tiles)

    rt = Runtime()
    with rt.sequence(q_l3_ty, k_l3_ty, v_l3_ty, scale_l3_ty, o_l3_ty) as (
        Q,
        K,
        V,
        dequant_scale,
        O,
    ):
        for i in range(num_pipelines):
            rt.start(qk_workers[i], softmax_workers[i], pv_workers[i])

        for q_block_idx in range(num_q_block_per_pipeline):
            tg = rt.task_group()
            rt.fill(
                inQ.prod(),
                Q,
                tap=Q_tiles[q_block_idx],
                placement=Tile(col=4, row=0),
                task_group=tg,
            )
            rt.fill(
                inK.prod(),
                K,
                tap=K_tiles[0],
                placement=Tile(col=5, row=0),
                task_group=tg,
            )
            rt.fill(
                inV.prod(),
                V,
                tap=V_tiles[0],
                placement=Tile(col=6, row=0),
                task_group=tg,
            )
            for i in range(num_pipelines):
                logical_q_block = q_block_idx * num_pipelines + i
                rt.fill(
                    inDequantScale[i].prod(),
                    dequant_scale,
                    tap=scale_tiles[logical_q_block],
                    placement=Tile(col=i, row=0),
                    task_group=tg,
                )
            rt.drain(
                memO.cons(),
                O,
                tap=O_tiles[q_block_idx],
                wait=True,
                placement=Tile(col=7, row=0),
                task_group=tg,
            )
            rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())
