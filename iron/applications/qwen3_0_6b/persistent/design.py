#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer


def qwen3_persistent_input_rmsnorm(
    dev,
    hidden_size,
    trace_size,
    func_prefix="",
    kernel_object="rms_norm.o",
):
    """Single-token Qwen3 input RMSNorm persistent bring-up stage.

    This is intentionally not a FusedMLIROperator runlist. It is the first
    hand-authored IRON dataflow stage for the Qwen3 decode megakernel path:
    host hidden vector -> ObjectFIFO -> RMSNorm worker -> multiply worker ->
    ObjectFIFO -> host output.
    """
    dtype = bfloat16
    tensor_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    tile_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    weight_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]

    in_hidden = ObjectFifo(tile_ty, name="qwen3_hidden_in", depth=2)
    in_weight = ObjectFifo(weight_ty, name="qwen3_input_norm_weight", depth=2)
    normed = ObjectFifo(tile_ty, name="qwen3_input_norm_unweighted", depth=2)
    out_hidden = ObjectFifo(tile_ty, name="qwen3_input_norm_out", depth=2)

    rms_norm_kernel = Kernel(
        f"{func_prefix}rms_norm_bf16_vector",
        f"{func_prefix}{kernel_object}",
        [tile_ty, tile_ty, np.int32],
    )
    mul_kernel = Kernel(
        f"{func_prefix}eltwise_mul_bf16_vector",
        f"{func_prefix}mul.o",
        [tile_ty, weight_ty, tile_ty, np.int32],
    )

    def rmsnorm_worker(of_in, of_out, rms_norm):
        for _ in range_(1):
            hidden = of_in.acquire(1)
            tmp = of_out.acquire(1)
            rms_norm(hidden, tmp, hidden_size)
            of_in.release(1)
            of_out.release(1)

    def weight_worker(of_in, of_weight, of_out, mul):
        weight = of_weight.acquire(1)
        for _ in range_(1):
            tmp = of_in.acquire(1)
            out = of_out.acquire(1)
            mul(tmp, weight, out, hidden_size)
            of_in.release(1)
            of_out.release(1)
        of_weight.release(1)

    workers = [
        Worker(
            rmsnorm_worker,
            [
                in_hidden.cons(),
                normed.prod(),
                rms_norm_kernel,
            ],
        ),
        Worker(
            weight_worker,
            [
                normed.cons(),
                in_weight.cons(),
                out_hidden.prod(),
                mul_kernel,
            ],
        ),
    ]

    hidden_tap = TensorAccessPattern(
        (1, hidden_size),
        0,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, weight_ty, tensor_ty) as (hidden, weight, output):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(in_hidden.prod(), hidden, hidden_tap, task_group=tg)
        rt.fill(in_weight.prod(), weight, task_group=tg)
        rt.drain(out_hidden.cons(), output, hidden_tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


def qwen3_persistent_input_rmsnorm_qkv(
    dev,
    hidden_size,
    q_size,
    kv_size,
    num_columns,
    tile_size_input,
    tile_size_output,
    trace_size,
    func_prefix="",
    rms_kernel_object="rms_norm.o",
    gemv_kernel_object="mv.o",
):
    """Single-token Qwen3 input RMSNorm + Q/K/V projection stage."""
    dtype = bfloat16
    tensor_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    weights_size = hidden_size + q_size * hidden_size + 2 * kv_size * hidden_size
    outputs_size = hidden_size + q_size + 2 * kv_size
    weights_ty = np.ndarray[(weights_size,), np.dtype[dtype]]
    outputs_ty = np.ndarray[(outputs_size,), np.dtype[dtype]]
    tile_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    weight_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    gemv_a_ty = np.ndarray[(tile_size_input, hidden_size), np.dtype[dtype]]
    gemv_b_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    gemv_c_ty = np.ndarray[(tile_size_output,), np.dtype[dtype]]

    if q_size % num_columns != 0 or kv_size % num_columns != 0:
        raise ValueError("Q/KV output sizes must be divisible by num_columns")
    if tile_size_output % tile_size_input != 0:
        raise ValueError("tile_size_output must be a multiple of tile_size_input")

    in_hidden = ObjectFifo(tile_ty, name="qwen3_qkv_hidden_in", depth=2)
    in_weight = ObjectFifo(weight_ty, name="qwen3_qkv_input_norm_weight", depth=2)
    normed = ObjectFifo(tile_ty, name="qwen3_qkv_input_norm_unweighted", depth=2)
    xnorm = ObjectFifo(tile_ty, name="qwen3_qkv_xnorm", depth=2)

    q_weight_fifos = [
        ObjectFifo(gemv_a_ty, name=f"qwen3_q_weight_{col}", depth=2)
        for col in range(num_columns)
    ]
    k_weight_fifos = [
        ObjectFifo(gemv_a_ty, name=f"qwen3_k_weight_{col}", depth=2)
        for col in range(num_columns)
    ]
    v_weight_fifos = [
        ObjectFifo(gemv_a_ty, name=f"qwen3_v_weight_{col}", depth=2)
        for col in range(num_columns)
    ]
    q_out_fifos = [
        ObjectFifo(gemv_c_ty, name=f"qwen3_q_out_{col}", depth=2)
        for col in range(num_columns)
    ]
    k_out_fifos = [
        ObjectFifo(gemv_c_ty, name=f"qwen3_k_out_{col}", depth=2)
        for col in range(num_columns)
    ]
    v_out_fifos = [
        ObjectFifo(gemv_c_ty, name=f"qwen3_v_out_{col}", depth=2)
        for col in range(num_columns)
    ]

    rms_norm_kernel = Kernel(
        f"{func_prefix}rms_norm_bf16_vector",
        f"{func_prefix}{rms_kernel_object}",
        [tile_ty, tile_ty, np.int32],
    )
    mul_kernel = Kernel(
        f"{func_prefix}eltwise_mul_bf16_vector",
        f"{func_prefix}mul.o",
        [tile_ty, weight_ty, tile_ty, np.int32],
    )
    matvec = Kernel(
        f"{func_prefix}matvec_vectorized_bf16_bf16",
        f"{func_prefix}{gemv_kernel_object}",
        [np.int32, np.int32, gemv_a_ty, gemv_b_ty, gemv_c_ty],
    )

    def rmsnorm_worker(of_in, of_out, rms_norm):
        for _ in range_(1):
            hidden = of_in.acquire(1)
            tmp = of_out.acquire(1)
            rms_norm(hidden, tmp, hidden_size)
            of_in.release(1)
            of_out.release(1)

    def weight_worker(of_in, of_weight, of_out, mul):
        weight = of_weight.acquire(1)
        tmp = of_in.acquire(1)
        out = of_out.acquire(1)
        mul(tmp, weight, out, hidden_size)
        of_out.release(1)
        of_in.release(1)
        of_weight.release(1)

    def q_matvec_worker(weight_fifo, x_fifo, out_fifo, matvec_kernel):
        x = x_fifo.acquire(1)
        for i_idx in range_(q_size // tile_size_output // num_columns):
            c = out_fifo.acquire(1)
            for j_idx in range_(tile_size_output // tile_size_input):
                j_i32 = index.casts(T.i32(), j_idx)
                output_row_offset = j_i32 * tile_size_input
                w = weight_fifo.acquire(1)
                matvec_kernel(tile_size_input, output_row_offset, w, x, c)
                weight_fifo.release(1)
            out_fifo.release(1)
        x_fifo.release(1)

    def kv_matvec_worker(weight_fifo, x_fifo, out_fifo, matvec_kernel):
        x = x_fifo.acquire(1)
        for i_idx in range_(kv_size // tile_size_output // num_columns):
            c = out_fifo.acquire(1)
            for j_idx in range_(tile_size_output // tile_size_input):
                j_i32 = index.casts(T.i32(), j_idx)
                output_row_offset = j_i32 * tile_size_input
                w = weight_fifo.acquire(1)
                matvec_kernel(tile_size_input, output_row_offset, w, x, c)
                weight_fifo.release(1)
            out_fifo.release(1)
        x_fifo.release(1)

    workers = [
        Worker(
            rmsnorm_worker,
            [
                in_hidden.cons(),
                normed.prod(),
                rms_norm_kernel,
            ],
        ),
        Worker(
            weight_worker,
            [
                normed.cons(),
                in_weight.cons(),
                xnorm.prod(),
                mul_kernel,
            ],
        ),
    ]
    for col in range(num_columns):
        workers.append(
            Worker(
                q_matvec_worker,
                [
                    q_weight_fifos[col].cons(),
                    xnorm.cons(),
                    q_out_fifos[col].prod(),
                    matvec,
                ],
            )
        )
        workers.append(
            Worker(
                kv_matvec_worker,
                [
                    k_weight_fifos[col].cons(),
                    xnorm.cons(),
                    k_out_fifos[col].prod(),
                    matvec,
                ],
            )
        )
        workers.append(
            Worker(
                kv_matvec_worker,
                [
                    v_weight_fifos[col].cons(),
                    xnorm.cons(),
                    v_out_fifos[col].prod(),
                    matvec,
                ],
            )
        )

    hidden_tap = TensorAccessPattern(
        (1, hidden_size),
        0,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )

    norm_weight_tap = TensorAccessPattern(
        (weights_size,),
        0,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )
    q_weight_base = hidden_size
    k_weight_base = q_weight_base + q_size * hidden_size
    v_weight_base = k_weight_base + kv_size * hidden_size

    def weight_taps(total_rows, base_offset):
        return [
            TensorAccessPattern(
                (weights_size,),
                base_offset + col * (total_rows // num_columns) * hidden_size,
                [1, 1, 1, (total_rows // num_columns) * hidden_size],
                [0, 0, 0, 1],
            )
            for col in range(num_columns)
        ]

    xnorm_output_tap = TensorAccessPattern(
        (outputs_size,),
        0,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )
    q_output_base = hidden_size
    k_output_base = q_output_base + q_size
    v_output_base = k_output_base + kv_size

    def out_taps(total_rows, base_offset):
        return [
            TensorAccessPattern(
                (outputs_size,),
                base_offset + col * (total_rows // num_columns),
                [1, 1, 1, total_rows // num_columns],
                [0, 0, 0, 1],
            )
            for col in range(num_columns)
        ]

    q_weight_taps = weight_taps(q_size, q_weight_base)
    k_weight_taps = weight_taps(kv_size, k_weight_base)
    v_weight_taps = weight_taps(kv_size, v_weight_base)
    q_output_taps = out_taps(q_size, q_output_base)
    k_output_taps = out_taps(kv_size, k_output_base)
    v_output_taps = out_taps(kv_size, v_output_base)

    rt = Runtime()
    with rt.sequence(tensor_ty, weights_ty, outputs_ty) as (
        hidden,
        weights,
        outputs,
    ):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(in_hidden.prod(), hidden, hidden_tap, task_group=tg)
        rt.fill(in_weight.prod(), weights, norm_weight_tap, task_group=tg)
        for col in range(num_columns):
            rt.fill(
                q_weight_fifos[col].prod(),
                weights,
                q_weight_taps[col],
                task_group=tg,
            )
            rt.fill(
                k_weight_fifos[col].prod(),
                weights,
                k_weight_taps[col],
                task_group=tg,
            )
            rt.fill(
                v_weight_fifos[col].prod(),
                weights,
                v_weight_taps[col],
                task_group=tg,
            )
        rt.drain(xnorm.cons(), outputs, xnorm_output_tap, wait=True, task_group=tg)
        for col in range(num_columns):
            rt.drain(
                q_out_fifos[col].cons(),
                outputs,
                q_output_taps[col],
                wait=True,
                task_group=tg,
            )
            rt.drain(
                k_out_fifos[col].cons(),
                outputs,
                k_output_taps[col],
                wait=True,
                task_group=tg,
            )
            rt.drain(
                v_out_fifos[col].cons(),
                outputs,
                v_output_taps[col],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


def qwen3_persistent_post_attn_rmsnorm_mlp_gate_up(
    dev,
    hidden_size,
    intermediate_size,
    num_columns,
    tile_size_input,
    tile_size_output,
    trace_size,
    func_prefix="",
    rms_kernel_object="rms_norm.o",
    gemv_kernel_object="mv.o",
    silu_kernel_object="silu.o",
    mul_kernel_object="mul.o",
):
    """Single-token post-attention RMSNorm + MLP gate/up checkpoint."""
    dtype = bfloat16
    weights_size = hidden_size + 2 * intermediate_size * hidden_size
    outputs_size = hidden_size + 4 * intermediate_size
    tensor_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    weights_ty = np.ndarray[(weights_size,), np.dtype[dtype]]
    outputs_ty = np.ndarray[(outputs_size,), np.dtype[dtype]]
    hidden_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    hidden_weight_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    ffn_tile_ty = np.ndarray[(tile_size_output,), np.dtype[dtype]]
    gemv_a_ty = np.ndarray[(tile_size_input, hidden_size), np.dtype[dtype]]

    if hidden_size != 1024:
        raise ValueError("post-attn MLP checkpoint expects hidden_size=1024")
    if intermediate_size != 3072:
        raise ValueError("post-attn MLP checkpoint expects intermediate_size=3072")
    if intermediate_size % num_columns != 0:
        raise ValueError("intermediate_size must be divisible by num_columns")
    if intermediate_size % tile_size_output != 0:
        raise ValueError("intermediate_size must be divisible by tile_size_output")
    if tile_size_output % tile_size_input != 0:
        raise ValueError("tile_size_output must be a multiple of tile_size_input")
    if tile_size_output % 16 != 0:
        raise ValueError("tile_size_output must be a multiple of 16")

    in_residual = ObjectFifo(hidden_ty, name="qwen3_mlp_attn_residual", depth=2)
    post_norm_weight = ObjectFifo(
        hidden_weight_ty, name="qwen3_mlp_post_norm_weight", depth=2
    )
    mlp_xnorm = ObjectFifo(hidden_ty, name="qwen3_mlp_xnorm", depth=2)

    gate_weight_fifos = [
        ObjectFifo(gemv_a_ty, name=f"qwen3_mlp_gate_weight_{col}", depth=2)
        for col in range(num_columns)
    ]
    up_weight_fifos = [
        ObjectFifo(gemv_a_ty, name=f"qwen3_mlp_up_weight_{col}", depth=2)
        for col in range(num_columns)
    ]
    gate_fifos = [
        ObjectFifo(ffn_tile_ty, name=f"qwen3_mlp_gate_{col}", depth=2)
        for col in range(num_columns)
    ]
    up_fifos = [
        ObjectFifo(ffn_tile_ty, name=f"qwen3_mlp_up_{col}", depth=2)
        for col in range(num_columns)
    ]
    gate_silu_fifos = [
        ObjectFifo(ffn_tile_ty, name=f"qwen3_mlp_gate_silu_{col}", depth=2)
        for col in range(num_columns)
    ]
    hidden_fifos = [
        ObjectFifo(ffn_tile_ty, name=f"qwen3_mlp_hidden_{col}", depth=2)
        for col in range(num_columns)
    ]

    weighted_rms_norm = Kernel(
        f"{func_prefix}weighted_rms_norm",
        f"{func_prefix}{rms_kernel_object}",
        [hidden_ty, hidden_weight_ty, hidden_ty, np.int32],
    )
    matvec = Kernel(
        f"{func_prefix}matvec_vectorized_bf16_bf16",
        f"{func_prefix}{gemv_kernel_object}",
        [np.int32, np.int32, gemv_a_ty, hidden_ty, ffn_tile_ty],
    )
    silu = Kernel(
        f"{func_prefix}silu_bf16",
        f"{func_prefix}{silu_kernel_object}",
        [ffn_tile_ty, ffn_tile_ty, np.int32],
    )
    mul = Kernel(
        f"{func_prefix}eltwise_mul_bf16_vector",
        f"{func_prefix}{mul_kernel_object}",
        [ffn_tile_ty, ffn_tile_ty, ffn_tile_ty, np.int32],
    )

    def post_norm_worker(of_in, of_weight, of_out, rms_norm):
        residual = of_in.acquire(1)
        weight = of_weight.acquire(1)
        out = of_out.acquire(1)
        rms_norm(residual, weight, out, hidden_size)
        of_out.release(1)
        of_weight.release(1)
        of_in.release(1)

    def mlp_matvec_worker(weight_fifo, x_fifo, out_fifo, matvec_kernel):
        x = x_fifo.acquire(1)
        for _ in range_(intermediate_size // tile_size_output // num_columns):
            c = out_fifo.acquire(1)
            for j_idx in range_(tile_size_output // tile_size_input):
                j_i32 = index.casts(T.i32(), j_idx)
                output_row_offset = j_i32 * tile_size_input
                w = weight_fifo.acquire(1)
                matvec_kernel(tile_size_input, output_row_offset, w, x, c)
                weight_fifo.release(1)
            out_fifo.release(1)
        x_fifo.release(1)

    def silu_worker(of_in, of_out, silu_kernel):
        for _ in range_(intermediate_size // tile_size_output // num_columns):
            gate = of_in.acquire(1)
            out = of_out.acquire(1)
            silu_kernel(gate, out, tile_size_output)
            of_out.release(1)
            of_in.release(1)

    def mul_worker(of_gate, of_up, of_out, mul_kernel):
        for _ in range_(intermediate_size // tile_size_output // num_columns):
            gate = of_gate.acquire(1)
            up = of_up.acquire(1)
            out = of_out.acquire(1)
            mul_kernel(gate, up, out, tile_size_output)
            of_out.release(1)
            of_up.release(1)
            of_gate.release(1)

    workers = [
        Worker(
            post_norm_worker,
            [
                in_residual.cons(),
                post_norm_weight.cons(),
                mlp_xnorm.prod(),
                weighted_rms_norm,
            ],
        ),
    ]
    for col in range(num_columns):
        workers.extend(
            [
                Worker(
                    mlp_matvec_worker,
                    [
                        gate_weight_fifos[col].cons(),
                        mlp_xnorm.cons(),
                        gate_fifos[col].prod(),
                        matvec,
                    ],
                ),
                Worker(
                    mlp_matvec_worker,
                    [
                        up_weight_fifos[col].cons(),
                        mlp_xnorm.cons(),
                        up_fifos[col].prod(),
                        matvec,
                    ],
                ),
                Worker(
                    silu_worker,
                    [
                        gate_fifos[col].cons(),
                        gate_silu_fifos[col].prod(),
                        silu,
                    ],
                ),
                Worker(
                    mul_worker,
                    [
                        gate_silu_fifos[col].cons(),
                        up_fifos[col].cons(),
                        hidden_fifos[col].prod(),
                        mul,
                    ],
                ),
            ]
        )

    hidden_tap = TensorAccessPattern(
        (1, hidden_size),
        0,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )
    post_norm_weight_tap = TensorAccessPattern(
        (weights_size,),
        0,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )
    gate_weight_base = hidden_size
    up_weight_base = gate_weight_base + intermediate_size * hidden_size

    def weight_taps(total_rows, base_offset):
        return [
            TensorAccessPattern(
                (weights_size,),
                base_offset + col * (total_rows // num_columns) * hidden_size,
                [1, 1, 1, (total_rows // num_columns) * hidden_size],
                [0, 0, 0, 1],
            )
            for col in range(num_columns)
        ]

    mlp_xnorm_output_base = 0
    ffn_gate_output_base = hidden_size
    ffn_up_output_base = ffn_gate_output_base + intermediate_size
    ffn_gate_silu_output_base = ffn_up_output_base + intermediate_size
    ffn_hidden_output_base = ffn_gate_silu_output_base + intermediate_size

    mlp_xnorm_output_tap = TensorAccessPattern(
        (outputs_size,),
        mlp_xnorm_output_base,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )

    def out_taps(total_rows, base_offset):
        return [
            TensorAccessPattern(
                (outputs_size,),
                base_offset + col * (total_rows // num_columns),
                [1, 1, 1, total_rows // num_columns],
                [0, 0, 0, 1],
            )
            for col in range(num_columns)
        ]

    gate_weight_taps = weight_taps(intermediate_size, gate_weight_base)
    up_weight_taps = weight_taps(intermediate_size, up_weight_base)
    gate_output_taps = out_taps(intermediate_size, ffn_gate_output_base)
    up_output_taps = out_taps(intermediate_size, ffn_up_output_base)
    gate_silu_output_taps = out_taps(intermediate_size, ffn_gate_silu_output_base)
    hidden_output_taps = out_taps(intermediate_size, ffn_hidden_output_base)

    rt = Runtime()
    with rt.sequence(tensor_ty, weights_ty, outputs_ty) as (
        residual,
        weights,
        outputs,
    ):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(in_residual.prod(), residual, hidden_tap, task_group=tg)
        rt.fill(
            post_norm_weight.prod(),
            weights,
            post_norm_weight_tap,
            task_group=tg,
        )
        for col in range(num_columns):
            rt.fill(
                gate_weight_fifos[col].prod(),
                weights,
                gate_weight_taps[col],
                task_group=tg,
            )
            rt.fill(
                up_weight_fifos[col].prod(),
                weights,
                up_weight_taps[col],
                task_group=tg,
            )
        rt.drain(
            mlp_xnorm.cons(),
            outputs,
            mlp_xnorm_output_tap,
            wait=True,
            task_group=tg,
        )
        for col in range(num_columns):
            rt.drain(
                gate_fifos[col].cons(),
                outputs,
                gate_output_taps[col],
                wait=True,
                task_group=tg,
            )
            rt.drain(
                up_fifos[col].cons(),
                outputs,
                up_output_taps[col],
                wait=True,
                task_group=tg,
            )
            rt.drain(
                gate_silu_fifos[col].cons(),
                outputs,
                gate_silu_output_taps[col],
                wait=True,
                task_group=tg,
            )
            rt.drain(
                hidden_fifos[col].cons(),
                outputs,
                hidden_output_taps[col],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


def qwen3_persistent_post_attn_mlp_down_residual(
    dev,
    hidden_size,
    intermediate_size,
    num_columns,
    tile_size_input,
    tile_size_output,
    trace_size,
    func_prefix="",
    gemv_kernel_object="mv_down.o",
    add_kernel_object="add.o",
):
    """Single-token MLP down projection + layer residual checkpoint."""
    dtype = bfloat16
    weights_size = hidden_size * intermediate_size
    outputs_size = 2 * hidden_size
    ffn_ty = np.ndarray[(intermediate_size,), np.dtype[dtype]]
    residual_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    weights_ty = np.ndarray[(weights_size,), np.dtype[dtype]]
    outputs_ty = np.ndarray[(outputs_size,), np.dtype[dtype]]
    hidden_tile_ty = np.ndarray[(tile_size_output,), np.dtype[dtype]]
    gemv_a_ty = np.ndarray[(tile_size_input, intermediate_size), np.dtype[dtype]]

    if hidden_size != 1024:
        raise ValueError("MLP down checkpoint expects hidden_size=1024")
    if intermediate_size != 3072:
        raise ValueError("MLP down checkpoint expects intermediate_size=3072")
    if hidden_size % num_columns != 0:
        raise ValueError("hidden_size must be divisible by num_columns")
    if hidden_size % tile_size_output != 0:
        raise ValueError("hidden_size must be divisible by tile_size_output")
    if tile_size_output % tile_size_input != 0:
        raise ValueError("tile_size_output must be a multiple of tile_size_input")
    if tile_size_output % 16 != 0:
        raise ValueError("tile_size_output must be a multiple of 16")

    ffn_hidden = ObjectFifo(ffn_ty, name="qwen3_down_ffn_hidden", depth=2)
    down_weight_fifos = [
        ObjectFifo(gemv_a_ty, name=f"qwen3_down_weight_{col}", depth=2)
        for col in range(num_columns)
    ]
    residual_fifos = [
        ObjectFifo(hidden_tile_ty, name=f"qwen3_down_attn_residual_{col}", depth=2)
        for col in range(num_columns)
    ]
    ffn_out_fifos = [
        ObjectFifo(hidden_tile_ty, name=f"qwen3_down_ffn_out_{col}", depth=2)
        for col in range(num_columns)
    ]
    layer_residual_fifos = [
        ObjectFifo(hidden_tile_ty, name=f"qwen3_down_layer_residual_{col}", depth=2)
        for col in range(num_columns)
    ]

    matvec = Kernel(
        f"{func_prefix}qwen3_down_proj_matvec_vectorized_bf16_bf16",
        f"{func_prefix}{gemv_kernel_object}",
        [np.int32, np.int32, gemv_a_ty, ffn_ty, hidden_tile_ty],
    )
    add = Kernel(
        f"{func_prefix}eltwise_add_bf16_vector",
        f"{func_prefix}{add_kernel_object}",
        [hidden_tile_ty, hidden_tile_ty, hidden_tile_ty, np.int32],
    )

    def down_matvec_worker(weight_fifo, x_fifo, out_fifo, matvec_kernel):
        x = x_fifo.acquire(1)
        for _ in range_(hidden_size // tile_size_output // num_columns):
            c = out_fifo.acquire(1)
            for j_idx in range_(tile_size_output // tile_size_input):
                j_i32 = index.casts(T.i32(), j_idx)
                output_row_offset = j_i32 * tile_size_input
                w = weight_fifo.acquire(1)
                matvec_kernel(tile_size_input, output_row_offset, w, x, c)
                weight_fifo.release(1)
            out_fifo.release(1)
        x_fifo.release(1)

    def residual_add_worker(of_ffn, of_residual, of_out, add_kernel):
        for _ in range_(hidden_size // tile_size_output // num_columns):
            ffn = of_ffn.acquire(1)
            residual = of_residual.acquire(1)
            out = of_out.acquire(1)
            add_kernel(residual, ffn, out, tile_size_output)
            of_out.release(1)
            of_residual.release(1)
            of_ffn.release(1)

    workers = []
    for col in range(num_columns):
        workers.extend(
            [
                Worker(
                    down_matvec_worker,
                    [
                        down_weight_fifos[col].cons(),
                        ffn_hidden.cons(),
                        ffn_out_fifos[col].prod(),
                        matvec,
                    ],
                ),
                Worker(
                    residual_add_worker,
                    [
                        ffn_out_fifos[col].cons(),
                        residual_fifos[col].cons(),
                        layer_residual_fifos[col].prod(),
                        add,
                    ],
                ),
            ]
        )

    ffn_hidden_tap = TensorAccessPattern(
        (1, intermediate_size),
        0,
        [1, 1, 1, intermediate_size],
        [0, 0, 0, 1],
    )

    def weight_taps(total_rows, base_offset):
        return [
            TensorAccessPattern(
                (weights_size,),
                base_offset + col * (total_rows // num_columns) * intermediate_size,
                [1, 1, 1, (total_rows // num_columns) * intermediate_size],
                [0, 0, 0, 1],
            )
            for col in range(num_columns)
        ]

    def hidden_taps(tensor_shape, total_rows, base_offset):
        return [
            TensorAccessPattern(
                tensor_shape,
                base_offset + col * (total_rows // num_columns),
                [1, 1, 1, total_rows // num_columns],
                [0, 0, 0, 1],
            )
            for col in range(num_columns)
        ]

    ffn_out_output_base = 0
    layer_residual_output_base = hidden_size
    down_weight_taps = weight_taps(hidden_size, 0)
    residual_taps = hidden_taps((hidden_size,), hidden_size, 0)
    ffn_out_taps = hidden_taps((outputs_size,), hidden_size, ffn_out_output_base)
    layer_residual_taps = hidden_taps(
        (outputs_size,), hidden_size, layer_residual_output_base
    )

    rt = Runtime()
    with rt.sequence(ffn_ty, residual_ty, weights_ty, outputs_ty) as (
        hidden,
        residual,
        weights,
        outputs,
    ):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(ffn_hidden.prod(), hidden, ffn_hidden_tap, task_group=tg)
        for col in range(num_columns):
            rt.fill(
                down_weight_fifos[col].prod(),
                weights,
                down_weight_taps[col],
                task_group=tg,
            )
            rt.fill(
                residual_fifos[col].prod(),
                residual,
                residual_taps[col],
                task_group=tg,
            )
            rt.drain(
                ffn_out_fifos[col].cons(),
                outputs,
                ffn_out_taps[col],
                wait=True,
                task_group=tg,
            )
            rt.drain(
                layer_residual_fifos[col].cons(),
                outputs,
                layer_residual_taps[col],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


def qwen3_persistent_post_attn_rmsnorm_full_mlp(
    dev,
    hidden_size,
    intermediate_size,
    num_columns,
    tile_size_input,
    tile_size_output,
    trace_size,
    func_prefix="",
    rms_kernel_object="rms_norm.o",
    gemv_kernel_object="mv.o",
    silu_kernel_object="silu.o",
    mul_kernel_object="mul.o",
    down_gemv_kernel_object="mv_down.o",
    add_kernel_object="add.o",
):
    """Single-token post-attention RMSNorm + full MLP checkpoint."""
    dtype = bfloat16
    weights_size = (
        hidden_size
        + 2 * intermediate_size * hidden_size
        + hidden_size * intermediate_size
    )
    outputs_size = 3 * hidden_size + 4 * intermediate_size
    tensor_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    weights_ty = np.ndarray[(weights_size,), np.dtype[dtype]]
    outputs_ty = np.ndarray[(outputs_size,), np.dtype[dtype]]
    hidden_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    hidden_weight_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    ffn_ty = np.ndarray[(intermediate_size,), np.dtype[dtype]]
    gate_gemv_a_ty = np.ndarray[(tile_size_input, hidden_size), np.dtype[dtype]]
    down_gemv_a_ty = np.ndarray[(tile_size_input, intermediate_size), np.dtype[dtype]]
    hidden_tile_ty = np.ndarray[(tile_size_output,), np.dtype[dtype]]

    if hidden_size != 1024:
        raise ValueError("full MLP checkpoint expects hidden_size=1024")
    if intermediate_size != 3072:
        raise ValueError("full MLP checkpoint expects intermediate_size=3072")
    if num_columns != 1:
        raise ValueError("full MLP checkpoint is currently single-column only")
    if hidden_size % tile_size_output != 0:
        raise ValueError("hidden_size must be divisible by tile_size_output")
    if tile_size_output % tile_size_input != 0:
        raise ValueError("tile_size_output must be a multiple of tile_size_input")
    if tile_size_output % 16 != 0:
        raise ValueError("tile_size_output must be a multiple of 16")

    in_residual = ObjectFifo(hidden_ty, name="qwen3_full_mlp_attn_residual", depth=2)
    post_norm_weight = ObjectFifo(
        hidden_weight_ty, name="qwen3_full_mlp_post_norm_weight", depth=2
    )
    mlp_xnorm = ObjectFifo(hidden_ty, name="qwen3_full_mlp_xnorm", depth=2)
    gate_weight = ObjectFifo(gate_gemv_a_ty, name="qwen3_full_mlp_gate_weight", depth=2)
    up_weight = ObjectFifo(gate_gemv_a_ty, name="qwen3_full_mlp_up_weight", depth=2)
    down_weight = ObjectFifo(down_gemv_a_ty, name="qwen3_full_mlp_down_weight", depth=2)
    ffn_gate = ObjectFifo(ffn_ty, name="qwen3_full_mlp_gate", depth=2)
    ffn_up = ObjectFifo(ffn_ty, name="qwen3_full_mlp_up", depth=2)
    ffn_gate_silu = ObjectFifo(ffn_ty, name="qwen3_full_mlp_gate_silu", depth=2)
    ffn_hidden = ObjectFifo(ffn_ty, name="qwen3_full_mlp_hidden", depth=2)
    residual_fifos = [
        ObjectFifo(hidden_tile_ty, name=f"qwen3_full_mlp_residual_{col}", depth=2)
        for col in range(num_columns)
    ]
    ffn_out_fifos = [
        ObjectFifo(hidden_tile_ty, name=f"qwen3_full_mlp_ffn_out_{col}", depth=2)
        for col in range(num_columns)
    ]
    layer_residual_fifos = [
        ObjectFifo(hidden_tile_ty, name=f"qwen3_full_mlp_layer_residual_{col}", depth=2)
        for col in range(num_columns)
    ]

    weighted_rms_norm = Kernel(
        f"{func_prefix}weighted_rms_norm",
        f"{func_prefix}{rms_kernel_object}",
        [hidden_ty, hidden_weight_ty, hidden_ty, np.int32],
    )
    gate_matvec = Kernel(
        f"{func_prefix}matvec_vectorized_bf16_bf16",
        f"{func_prefix}{gemv_kernel_object}",
        [np.int32, np.int32, gate_gemv_a_ty, hidden_ty, ffn_ty],
    )
    silu = Kernel(
        f"{func_prefix}silu_bf16",
        f"{func_prefix}{silu_kernel_object}",
        [ffn_ty, ffn_ty, np.int32],
    )
    mul = Kernel(
        f"{func_prefix}eltwise_mul_bf16_vector",
        f"{func_prefix}{mul_kernel_object}",
        [ffn_ty, ffn_ty, ffn_ty, np.int32],
    )
    down_matvec = Kernel(
        f"{func_prefix}qwen3_down_proj_matvec_vectorized_bf16_bf16",
        f"{func_prefix}{down_gemv_kernel_object}",
        [np.int32, np.int32, down_gemv_a_ty, ffn_ty, hidden_tile_ty],
    )
    add = Kernel(
        f"{func_prefix}eltwise_add_bf16_vector",
        f"{func_prefix}{add_kernel_object}",
        [hidden_tile_ty, hidden_tile_ty, hidden_tile_ty, np.int32],
    )

    def post_norm_worker(of_in, of_weight, of_out, rms_norm):
        residual = of_in.acquire(1)
        weight = of_weight.acquire(1)
        out = of_out.acquire(1)
        rms_norm(residual, weight, out, hidden_size)
        of_out.release(1)
        of_weight.release(1)
        of_in.release(1)

    def gate_matvec_worker(weight_fifo, x_fifo, out_fifo, matvec_kernel):
        x = x_fifo.acquire(1)
        c = out_fifo.acquire(1)
        for j_idx in range_(intermediate_size // tile_size_input):
            j_i32 = index.casts(T.i32(), j_idx)
            output_row_offset = j_i32 * tile_size_input
            w = weight_fifo.acquire(1)
            matvec_kernel(tile_size_input, output_row_offset, w, x, c)
            weight_fifo.release(1)
        out_fifo.release(1)
        x_fifo.release(1)

    def silu_worker(of_in, of_out, silu_kernel):
        gate = of_in.acquire(1)
        out = of_out.acquire(1)
        silu_kernel(gate, out, intermediate_size)
        of_out.release(1)
        of_in.release(1)

    def mul_worker(of_gate, of_up, of_out, mul_kernel):
        gate = of_gate.acquire(1)
        up = of_up.acquire(1)
        out = of_out.acquire(1)
        mul_kernel(gate, up, out, intermediate_size)
        of_out.release(1)
        of_up.release(1)
        of_gate.release(1)

    def down_matvec_worker(weight_fifo, x_fifo, out_fifo, matvec_kernel):
        x = x_fifo.acquire(1)
        for _ in range_(hidden_size // tile_size_output // num_columns):
            c = out_fifo.acquire(1)
            for j_idx in range_(tile_size_output // tile_size_input):
                j_i32 = index.casts(T.i32(), j_idx)
                output_row_offset = j_i32 * tile_size_input
                w = weight_fifo.acquire(1)
                matvec_kernel(tile_size_input, output_row_offset, w, x, c)
                weight_fifo.release(1)
            out_fifo.release(1)
        x_fifo.release(1)

    def residual_add_worker(of_ffn, of_residual, of_out, add_kernel):
        for _ in range_(hidden_size // tile_size_output // num_columns):
            ffn = of_ffn.acquire(1)
            residual = of_residual.acquire(1)
            out = of_out.acquire(1)
            add_kernel(residual, ffn, out, tile_size_output)
            of_out.release(1)
            of_residual.release(1)
            of_ffn.release(1)

    workers = [
        Worker(
            post_norm_worker,
            [
                in_residual.cons(),
                post_norm_weight.cons(),
                mlp_xnorm.prod(),
                weighted_rms_norm,
            ],
        ),
        Worker(
            gate_matvec_worker,
            [
                gate_weight.cons(),
                mlp_xnorm.cons(),
                ffn_gate.prod(),
                gate_matvec,
            ],
        ),
        Worker(
            gate_matvec_worker,
            [
                up_weight.cons(),
                mlp_xnorm.cons(),
                ffn_up.prod(),
                gate_matvec,
            ],
        ),
        Worker(
            silu_worker,
            [
                ffn_gate.cons(),
                ffn_gate_silu.prod(),
                silu,
            ],
        ),
        Worker(
            mul_worker,
            [
                ffn_gate_silu.cons(),
                ffn_up.cons(),
                ffn_hidden.prod(),
                mul,
            ],
        ),
    ]
    for col in range(num_columns):
        workers.extend(
            [
                Worker(
                    down_matvec_worker,
                    [
                        down_weight.cons(),
                        ffn_hidden.cons(),
                        ffn_out_fifos[col].prod(),
                        down_matvec,
                    ],
                ),
                Worker(
                    residual_add_worker,
                    [
                        ffn_out_fifos[col].cons(),
                        residual_fifos[col].cons(),
                        layer_residual_fifos[col].prod(),
                        add,
                    ],
                ),
            ]
        )

    hidden_tap = TensorAccessPattern(
        (1, hidden_size),
        0,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )
    post_norm_weight_tap = TensorAccessPattern(
        (weights_size,),
        0,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )
    gate_weight_base = hidden_size
    up_weight_base = gate_weight_base + intermediate_size * hidden_size
    down_weight_base = up_weight_base + intermediate_size * hidden_size
    gate_weight_tap = TensorAccessPattern(
        (weights_size,),
        gate_weight_base,
        [1, 1, 1, intermediate_size * hidden_size],
        [0, 0, 0, 1],
    )
    up_weight_tap = TensorAccessPattern(
        (weights_size,),
        up_weight_base,
        [1, 1, 1, intermediate_size * hidden_size],
        [0, 0, 0, 1],
    )
    down_weight_tap = TensorAccessPattern(
        (weights_size,),
        down_weight_base,
        [1, 1, 1, hidden_size * intermediate_size],
        [0, 0, 0, 1],
    )

    mlp_xnorm_output_base = 0
    ffn_gate_output_base = hidden_size
    ffn_up_output_base = ffn_gate_output_base + intermediate_size
    ffn_gate_silu_output_base = ffn_up_output_base + intermediate_size
    ffn_hidden_output_base = ffn_gate_silu_output_base + intermediate_size
    ffn_out_output_base = ffn_hidden_output_base + intermediate_size
    layer_residual_output_base = ffn_out_output_base + hidden_size

    def output_tap(base_offset, size):
        return TensorAccessPattern(
            (outputs_size,),
            base_offset,
            [1, 1, 1, size],
            [0, 0, 0, 1],
        )

    def hidden_taps(tensor_shape, total_rows, base_offset):
        return [
            TensorAccessPattern(
                tensor_shape,
                base_offset + col * (total_rows // num_columns),
                [1, 1, 1, total_rows // num_columns],
                [0, 0, 0, 1],
            )
            for col in range(num_columns)
        ]

    residual_taps = hidden_taps((hidden_size,), hidden_size, 0)
    ffn_out_taps = hidden_taps((outputs_size,), hidden_size, ffn_out_output_base)
    layer_residual_taps = hidden_taps(
        (outputs_size,), hidden_size, layer_residual_output_base
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, weights_ty, outputs_ty) as (
        residual,
        weights,
        outputs,
    ):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(in_residual.prod(), residual, hidden_tap, task_group=tg)
        rt.fill(
            post_norm_weight.prod(),
            weights,
            post_norm_weight_tap,
            task_group=tg,
        )
        rt.fill(gate_weight.prod(), weights, gate_weight_tap, task_group=tg)
        rt.fill(up_weight.prod(), weights, up_weight_tap, task_group=tg)
        rt.fill(down_weight.prod(), weights, down_weight_tap, task_group=tg)
        rt.drain(
            mlp_xnorm.cons(),
            outputs,
            output_tap(mlp_xnorm_output_base, hidden_size),
            wait=True,
            task_group=tg,
        )
        rt.drain(
            ffn_gate.cons(),
            outputs,
            output_tap(ffn_gate_output_base, intermediate_size),
            wait=True,
            task_group=tg,
        )
        rt.drain(
            ffn_up.cons(),
            outputs,
            output_tap(ffn_up_output_base, intermediate_size),
            wait=True,
            task_group=tg,
        )
        rt.drain(
            ffn_gate_silu.cons(),
            outputs,
            output_tap(ffn_gate_silu_output_base, intermediate_size),
            wait=True,
            task_group=tg,
        )
        rt.drain(
            ffn_hidden.cons(),
            outputs,
            output_tap(ffn_hidden_output_base, intermediate_size),
            wait=True,
            task_group=tg,
        )
        for col in range(num_columns):
            rt.fill(
                residual_fifos[col].prod(),
                residual,
                residual_taps[col],
                task_group=tg,
            )
            rt.drain(
                ffn_out_fifos[col].cons(),
                outputs,
                ffn_out_taps[col],
                wait=True,
                task_group=tg,
            )
            rt.drain(
                layer_residual_fifos[col].cons(),
                outputs,
                layer_residual_taps[col],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
