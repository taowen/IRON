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


def _qwen3_persistent_input_rmsnorm_qkv_rope_cache_impl(
    dev,
    hidden_size,
    q_size,
    kv_size,
    head_dim,
    max_seq_len,
    position,
    num_columns,
    tile_size_input,
    tile_size_output,
    trace_size,
    func_prefix="",
    rms_kernel_object="rms_norm.o",
    gemv_kernel_object="mv.o",
    rope_kernel_object="rope.o",
    include_scores_softmax=False,
    attention_kernel_object="qwen3_attention.o",
    softmax_kernel_object="softmax.o",
):
    """Single-token Qwen3 input RMSNorm, QKV, Q/K norm, RoPE, KV write, and optional softmax."""
    dtype = bfloat16
    q_heads = q_size // head_dim
    kv_heads = kv_size // head_dim
    cache_block_seq = 64
    score_size = q_heads * max_seq_len if include_scores_softmax else 0
    weights_size = (
        hidden_size + q_size * hidden_size + 2 * kv_size * hidden_size + 2 * head_dim
    )
    outputs_size = (
        hidden_size + q_size + kv_size + q_size + kv_size + q_size + 2 * score_size
    )
    cache_size = 2 * kv_size * max_seq_len

    tensor_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    weights_ty = np.ndarray[(weights_size,), np.dtype[dtype]]
    angles_ty = np.ndarray[(head_dim,), np.dtype[dtype]]
    outputs_ty = np.ndarray[(outputs_size,), np.dtype[dtype]]
    cache_ty = np.ndarray[(cache_size,), np.dtype[dtype]]
    tile_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    hidden_weight_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    head_ty = np.ndarray[(head_dim,), np.dtype[dtype]]
    qk_pair_ty = np.ndarray[(3 * head_dim,), np.dtype[dtype]]
    score_ty = np.ndarray[(max_seq_len,), np.dtype[dtype]]
    k_cache_block_ty = np.ndarray[(cache_block_seq, head_dim), np.dtype[dtype]]
    gemv_a_ty = np.ndarray[(tile_size_input, hidden_size), np.dtype[dtype]]

    if q_size % head_dim != 0 or kv_size % head_dim != 0:
        raise ValueError("Q/KV sizes must be divisible by head_dim")
    if q_size % num_columns != 0 or kv_size % num_columns != 0:
        raise ValueError("Q/KV output sizes must be divisible by num_columns")
    if tile_size_output != head_dim:
        raise ValueError("tile_size_output must equal head_dim for rope-cache stage")
    if tile_size_output % tile_size_input != 0:
        raise ValueError("tile_size_output must be a multiple of tile_size_input")
    if not (0 <= position < max_seq_len):
        raise ValueError("position must be inside max_seq_len")
    if include_scores_softmax and num_columns != 1:
        raise ValueError(
            "scores+softmax checkpoint is currently NPU2 single-column only"
        )
    if include_scores_softmax and q_heads % kv_heads != 0:
        raise ValueError("q_heads must be a multiple of kv_heads for GQA")
    if include_scores_softmax and q_heads // kv_heads != 2:
        raise ValueError("scores+softmax checkpoint expects Qwen3-0.6B GQA repeat=2")
    if include_scores_softmax and max_seq_len % cache_block_seq != 0:
        raise ValueError("max_seq_len must be divisible by cache_block_seq")

    in_hidden = ObjectFifo(tile_ty, name="qwen3_rc_hidden_in", depth=2)
    in_weight = ObjectFifo(hidden_weight_ty, name="qwen3_rc_input_norm_weight", depth=2)
    normed = ObjectFifo(tile_ty, name="qwen3_rc_input_norm_unweighted", depth=2)
    xnorm = ObjectFifo(tile_ty, name="qwen3_rc_xnorm", depth=2)

    q_norm_weight = ObjectFifo(head_ty, name="qwen3_rc_q_norm_weight", depth=2)
    k_norm_weight = ObjectFifo(head_ty, name="qwen3_rc_k_norm_weight", depth=2)
    rope_angles = ObjectFifo(head_ty, name="qwen3_rc_rope_angles", depth=2)

    q_weight_fifos = [
        ObjectFifo(gemv_a_ty, name=f"qwen3_rc_q_weight_{col}", depth=2)
        for col in range(num_columns)
    ]
    k_weight_fifos = [
        ObjectFifo(gemv_a_ty, name=f"qwen3_rc_k_weight_{col}", depth=2)
        for col in range(num_columns)
    ]
    v_weight_fifos = [
        ObjectFifo(gemv_a_ty, name=f"qwen3_rc_v_weight_{col}", depth=2)
        for col in range(num_columns)
    ]
    q_raw_fifos = [
        ObjectFifo(head_ty, name=f"qwen3_rc_q_raw_{col}", depth=2)
        for col in range(num_columns)
    ]
    k_raw_fifos = [
        ObjectFifo(head_ty, name=f"qwen3_rc_k_raw_{col}", depth=2)
        for col in range(num_columns)
    ]
    v_fifos = [
        ObjectFifo(head_ty, name=f"qwen3_rc_v_{col}", depth=2)
        for col in range(num_columns)
    ]
    q_norm_fifos = [
        ObjectFifo(head_ty, name=f"qwen3_rc_q_norm_{col}", depth=2)
        for col in range(num_columns)
    ]
    k_norm_fifos = [
        ObjectFifo(head_ty, name=f"qwen3_rc_k_norm_{col}", depth=2)
        for col in range(num_columns)
    ]
    q_rope_fifos = [
        ObjectFifo(head_ty, name=f"qwen3_rc_q_rope_{col}", depth=2)
        for col in range(num_columns)
    ]
    k_rope_fifos = [
        ObjectFifo(head_ty, name=f"qwen3_rc_k_rope_{col}", depth=2)
        for col in range(num_columns)
    ]
    k_cache_fifos = []
    qk_pair_fifos = []
    attn_score_debug_fifos = []
    attn_score_softmax_fifos = []
    attn_weight_fifos = []
    if include_scores_softmax:
        qk_pair_fifos = [
            ObjectFifo(qk_pair_ty, name=f"qwen3_rc_qk_pair_{col}", depth=2)
            for col in range(num_columns)
        ]
        k_cache_fifos = [
            ObjectFifo(k_cache_block_ty, name=f"qwen3_rc_k_cache_{col}", depth=2)
            for col in range(num_columns)
        ]
        attn_score_debug_fifos = [
            ObjectFifo(score_ty, name=f"qwen3_rc_attn_scores_{col}", depth=2)
            for col in range(num_columns)
        ]
        attn_score_softmax_fifos = [
            ObjectFifo(
                score_ty,
                name=f"qwen3_rc_attn_scores_for_softmax_{col}",
                depth=2,
            )
            for col in range(num_columns)
        ]
        attn_weight_fifos = [
            ObjectFifo(score_ty, name=f"qwen3_rc_attn_weights_{col}", depth=2)
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
        [tile_ty, hidden_weight_ty, tile_ty, np.int32],
    )
    matvec = Kernel(
        f"{func_prefix}matvec_vectorized_bf16_bf16",
        f"{func_prefix}{gemv_kernel_object}",
        [np.int32, np.int32, gemv_a_ty, tile_ty, head_ty],
    )
    weighted_rms_norm = Kernel(
        f"{func_prefix}weighted_rms_norm",
        f"{func_prefix}{rms_kernel_object}",
        [head_ty, head_ty, head_ty, np.int32],
    )
    rope = Kernel(
        f"{func_prefix}rope",
        f"{func_prefix}{rope_kernel_object}",
        [head_ty, head_ty, head_ty, np.int32],
    )
    attention_scores = None
    pack_qk_pair = None
    mask = None
    softmax = None
    if include_scores_softmax:
        pack_qk_pair = Kernel(
            f"{func_prefix}qwen3_pack_qk_pair_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [head_ty, head_ty, qk_pair_ty, np.int32],
        )
        attention_scores = Kernel(
            f"{func_prefix}qwen3_attention_scores_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [
                qk_pair_ty,
                k_cache_block_ty,
                score_ty,
                score_ty,
                np.int32,
                np.int32,
                np.int32,
            ],
        )
        mask = Kernel(
            f"{func_prefix}mask_bf16",
            f"{func_prefix}{softmax_kernel_object}",
            [score_ty, np.int32, np.int32],
        )
        softmax = Kernel(
            f"{func_prefix}softmax_bf16",
            f"{func_prefix}{softmax_kernel_object}",
            [score_ty, score_ty, np.int32],
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
        for _ in range_(q_size // tile_size_output // num_columns):
            c = out_fifo.acquire(1)
            for j_idx in range_(tile_size_output // tile_size_input):
                j_i32 = index.casts(T.i32(), j_idx)
                output_row_offset = j_i32 * tile_size_input
                w = weight_fifo.acquire(1)
                matvec_kernel(tile_size_input, output_row_offset, w, x, c)
                weight_fifo.release(1)
            out_fifo.release(1)
        x_fifo.release(1)

    def q_matvec_weight_gated_worker(weight_fifo, x_fifo, out_fifo, matvec_kernel):
        first_w = weight_fifo.acquire(1)
        x = x_fifo.acquire(1)
        first_c = out_fifo.acquire(1)
        matvec_kernel(tile_size_input, 0, first_w, x, first_c)
        weight_fifo.release(1)
        for j_idx in range_(1, tile_size_output // tile_size_input):
            j_i32 = index.casts(T.i32(), j_idx)
            output_row_offset = j_i32 * tile_size_input
            w = weight_fifo.acquire(1)
            matvec_kernel(tile_size_input, output_row_offset, w, x, first_c)
            weight_fifo.release(1)
        out_fifo.release(1)
        for _ in range_(1, q_size // tile_size_output // num_columns):
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
        for _ in range_(kv_size // tile_size_output // num_columns):
            c = out_fifo.acquire(1)
            for j_idx in range_(tile_size_output // tile_size_input):
                j_i32 = index.casts(T.i32(), j_idx)
                output_row_offset = j_i32 * tile_size_input
                w = weight_fifo.acquire(1)
                matvec_kernel(tile_size_input, output_row_offset, w, x, c)
                weight_fifo.release(1)
            out_fifo.release(1)
        x_fifo.release(1)

    def q_head_norm_worker(raw_fifo, weight_fifo, out_fifo, norm_kernel):
        weight = weight_fifo.acquire(1)
        for _ in range_(q_heads // num_columns):
            raw = raw_fifo.acquire(1)
            out = out_fifo.acquire(1)
            norm_kernel(raw, weight, out, head_dim)
            raw_fifo.release(1)
            out_fifo.release(1)
        weight_fifo.release(1)

    def k_head_norm_worker(raw_fifo, weight_fifo, out_fifo, norm_kernel):
        weight = weight_fifo.acquire(1)
        for _ in range_(kv_heads // num_columns):
            raw = raw_fifo.acquire(1)
            out = out_fifo.acquire(1)
            norm_kernel(raw, weight, out, head_dim)
            raw_fifo.release(1)
            out_fifo.release(1)
        weight_fifo.release(1)

    def q_rope_worker(in_fifo, angles_fifo, out_fifo, rope_kernel):
        angles = angles_fifo.acquire(1)
        for _ in range_(q_heads // num_columns):
            elem_in = in_fifo.acquire(1)
            elem_out = out_fifo.acquire(1)
            rope_kernel(elem_in, angles, elem_out, head_dim)
            in_fifo.release(1)
            out_fifo.release(1)
        angles_fifo.release(1)

    def k_rope_worker(in_fifo, angles_fifo, out_fifo, rope_kernel):
        angles = angles_fifo.acquire(1)
        for _ in range_(kv_heads // num_columns):
            elem_in = in_fifo.acquire(1)
            elem_out = out_fifo.acquire(1)
            rope_kernel(elem_in, angles, elem_out, head_dim)
            in_fifo.release(1)
            out_fifo.release(1)
        angles_fifo.release(1)

    def qk_pair_worker(q_fifo, current_k_fifo, pair_fifo, pack_kernel):
        for _ in range_(kv_heads // num_columns):
            current_k = current_k_fifo.acquire(1)
            pair = pair_fifo.acquire(1)
            for q_select in range_(q_heads // kv_heads):
                q_select_i32 = index.casts(T.i32(), q_select)
                q = q_fifo.acquire(1)
                pack_kernel(q, current_k, pair, q_select_i32)
                q_fifo.release(1)
            pair_fifo.release(1)
            current_k_fifo.release(1)

    def attention_score_worker(
        pair_fifo,
        k_cache_fifo,
        score_debug_fifo,
        score_softmax_fifo,
        score_kernel,
    ):
        for _ in range_(kv_heads // num_columns):
            pair = pair_fifo.acquire(1)
            score_debug0 = score_debug_fifo.acquire(1)
            score_softmax0 = score_softmax_fifo.acquire(1)
            score_debug1 = score_debug_fifo.acquire(1)
            score_softmax1 = score_softmax_fifo.acquire(1)
            for block_idx in range_(max_seq_len // cache_block_seq):
                block_i32 = index.casts(T.i32(), block_idx)
                row_base = block_i32 * cache_block_seq
                k_cache = k_cache_fifo.acquire(1)
                score_kernel(
                    pair,
                    k_cache,
                    score_debug0,
                    score_softmax0,
                    position,
                    row_base,
                    0,
                )
                score_kernel(
                    pair,
                    k_cache,
                    score_debug1,
                    score_softmax1,
                    position,
                    row_base,
                    1,
                )
                k_cache_fifo.release(1)
            pair_fifo.release(1)
            score_debug_fifo.release(1)
            score_softmax_fifo.release(1)
            score_debug_fifo.release(1)
            score_softmax_fifo.release(1)

    def attention_softmax_worker(score_fifo, weight_fifo, mask_kernel, softmax_kernel):
        for _ in range_(q_heads // num_columns):
            scores = score_fifo.acquire(1)
            weights = weight_fifo.acquire(1)
            mask_kernel(scores, position + 1, max_seq_len)
            softmax_kernel(scores, weights, max_seq_len)
            score_fifo.release(1)
            weight_fifo.release(1)

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
        q_workers = [
            Worker(
                q_matvec_worker,
                [
                    q_weight_fifos[col].cons(),
                    xnorm.cons(),
                    q_raw_fifos[col].prod(),
                    matvec,
                ],
            ),
            Worker(
                q_head_norm_worker,
                [
                    q_raw_fifos[col].cons(),
                    q_norm_weight.cons(),
                    q_norm_fifos[col].prod(),
                    weighted_rms_norm,
                ],
            ),
            Worker(
                q_rope_worker,
                [
                    q_norm_fifos[col].cons(),
                    rope_angles.cons(),
                    q_rope_fifos[col].prod(),
                    rope,
                ],
            ),
        ]
        kv_workers = [
            Worker(
                kv_matvec_worker,
                [
                    k_weight_fifos[col].cons(),
                    xnorm.cons(),
                    k_raw_fifos[col].prod(),
                    matvec,
                ],
            ),
            Worker(
                kv_matvec_worker,
                [
                    v_weight_fifos[col].cons(),
                    xnorm.cons(),
                    v_fifos[col].prod(),
                    matvec,
                ],
            ),
            Worker(
                k_head_norm_worker,
                [
                    k_raw_fifos[col].cons(),
                    k_norm_weight.cons(),
                    k_norm_fifos[col].prod(),
                    weighted_rms_norm,
                ],
            ),
            Worker(
                k_rope_worker,
                [
                    k_norm_fifos[col].cons(),
                    rope_angles.cons(),
                    k_rope_fifos[col].prod(),
                    rope,
                ],
            ),
        ]
        workers.extend(q_workers + kv_workers)
        if include_scores_softmax:
            workers.extend(
                [
                    Worker(
                        qk_pair_worker,
                        [
                            q_rope_fifos[col].cons(),
                            k_rope_fifos[col].cons(),
                            qk_pair_fifos[col].prod(),
                            pack_qk_pair,
                        ],
                    ),
                    Worker(
                        attention_score_worker,
                        [
                            qk_pair_fifos[col].cons(),
                            k_cache_fifos[col].cons(),
                            attn_score_debug_fifos[col].prod(),
                            attn_score_softmax_fifos[col].prod(),
                            attention_scores,
                        ],
                    ),
                    Worker(
                        attention_softmax_worker,
                        [
                            attn_score_softmax_fifos[col].cons(),
                            attn_weight_fifos[col].prod(),
                            mask,
                            softmax,
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
    norm_weight_tap = TensorAccessPattern(
        (weights_size,),
        0,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )
    q_weight_base = hidden_size
    k_weight_base = q_weight_base + q_size * hidden_size
    v_weight_base = k_weight_base + kv_size * hidden_size
    q_norm_weight_base = v_weight_base + kv_size * hidden_size
    k_norm_weight_base = q_norm_weight_base + head_dim

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

    q_norm_weight_tap = TensorAccessPattern(
        (weights_size,),
        q_norm_weight_base,
        [1, 1, 1, head_dim],
        [0, 0, 0, 1],
    )
    k_norm_weight_tap = TensorAccessPattern(
        (weights_size,),
        k_norm_weight_base,
        [1, 1, 1, head_dim],
        [0, 0, 0, 1],
    )
    rope_angles_tap = TensorAccessPattern(
        (1, head_dim),
        0,
        [1, 1, 1, head_dim],
        [0, 0, 0, 1],
    )

    xnorm_output_base = 0
    q_raw_output_base = xnorm_output_base + hidden_size
    k_raw_output_base = q_raw_output_base + q_size
    q_norm_output_base = k_raw_output_base + kv_size
    k_norm_output_base = q_norm_output_base + q_size
    q_rope_output_base = k_norm_output_base + kv_size
    attn_scores_output_base = q_rope_output_base + q_size
    attn_weights_output_base = attn_scores_output_base + score_size

    xnorm_output_tap = TensorAccessPattern(
        (outputs_size,),
        xnorm_output_base,
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

    cache_key_tap = TensorAccessPattern(
        (cache_size,),
        position * head_dim,
        [1, 1, kv_heads, head_dim],
        [0, 0, max_seq_len * head_dim, 1],
    )
    cache_value_tap = TensorAccessPattern(
        (cache_size,),
        kv_size * max_seq_len + position * head_dim,
        [1, 1, kv_heads, head_dim],
        [0, 0, max_seq_len * head_dim, 1],
    )

    cache_key_blocks_tap = TensorAccessPattern(
        (cache_size,),
        0,
        [
            kv_heads,
            max_seq_len // cache_block_seq,
            cache_block_seq,
            head_dim,
        ],
        [max_seq_len * head_dim, cache_block_seq * head_dim, head_dim, 1],
    )

    q_weight_taps = weight_taps(q_size, q_weight_base)
    k_weight_taps = weight_taps(kv_size, k_weight_base)
    v_weight_taps = weight_taps(kv_size, v_weight_base)
    q_raw_output_taps = out_taps(q_size, q_raw_output_base)
    k_raw_output_taps = out_taps(kv_size, k_raw_output_base)
    q_norm_output_taps = out_taps(q_size, q_norm_output_base)
    k_norm_output_taps = out_taps(kv_size, k_norm_output_base)
    q_rope_output_taps = out_taps(q_size, q_rope_output_base)
    attn_score_output_taps = (
        out_taps(score_size, attn_scores_output_base) if include_scores_softmax else []
    )
    attn_weight_output_taps = (
        out_taps(score_size, attn_weights_output_base) if include_scores_softmax else []
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, weights_ty, angles_ty, outputs_ty, cache_ty) as (
        hidden,
        weights,
        angles,
        outputs,
        cache,
    ):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(in_hidden.prod(), hidden, hidden_tap, task_group=tg)
        rt.fill(in_weight.prod(), weights, norm_weight_tap, task_group=tg)
        rt.fill(q_norm_weight.prod(), weights, q_norm_weight_tap, task_group=tg)
        rt.fill(k_norm_weight.prod(), weights, k_norm_weight_tap, task_group=tg)
        rt.fill(rope_angles.prod(), angles, rope_angles_tap, task_group=tg)
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
        if include_scores_softmax:
            for col in range(num_columns):
                rt.fill(
                    k_cache_fifos[col].prod(),
                    cache,
                    cache_key_blocks_tap,
                    task_group=tg,
                )
            rt.drain(
                xnorm.cons(),
                outputs,
                xnorm_output_tap,
                wait=True,
                task_group=tg,
            )
            for col in range(num_columns):
                rt.drain(
                    q_raw_fifos[col].cons(),
                    outputs,
                    q_raw_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    k_raw_fifos[col].cons(),
                    outputs,
                    k_raw_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    q_norm_fifos[col].cons(),
                    outputs,
                    q_norm_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    k_norm_fifos[col].cons(),
                    outputs,
                    k_norm_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    q_rope_fifos[col].cons(),
                    outputs,
                    q_rope_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    attn_score_debug_fifos[col].cons(),
                    outputs,
                    attn_score_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    attn_weight_fifos[col].cons(),
                    outputs,
                    attn_weight_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    k_rope_fifos[col].cons(),
                    cache,
                    cache_key_tap,
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    v_fifos[col].cons(),
                    cache,
                    cache_value_tap,
                    wait=True,
                    task_group=tg,
                )
            rt.finish_task_group(tg)
        else:
            rt.drain(
                xnorm.cons(),
                outputs,
                xnorm_output_tap,
                wait=True,
                task_group=tg,
            )
            for col in range(num_columns):
                rt.drain(
                    q_raw_fifos[col].cons(),
                    outputs,
                    q_raw_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    k_raw_fifos[col].cons(),
                    outputs,
                    k_raw_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    q_norm_fifos[col].cons(),
                    outputs,
                    q_norm_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    k_norm_fifos[col].cons(),
                    outputs,
                    k_norm_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    q_rope_fifos[col].cons(),
                    outputs,
                    q_rope_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    k_rope_fifos[col].cons(),
                    cache,
                    cache_key_tap,
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    v_fifos[col].cons(),
                    cache,
                    cache_value_tap,
                    wait=True,
                    task_group=tg,
                )
            rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


def qwen3_persistent_input_rmsnorm_qkv_rope_cache(
    dev,
    hidden_size,
    q_size,
    kv_size,
    head_dim,
    max_seq_len,
    position,
    num_columns,
    tile_size_input,
    tile_size_output,
    trace_size,
    func_prefix="",
    rms_kernel_object="rms_norm.o",
    gemv_kernel_object="mv.o",
    rope_kernel_object="rope.o",
):
    """Single-token Qwen3 input RMSNorm, QKV, Q/K norm, RoPE, and KV write."""
    return _qwen3_persistent_input_rmsnorm_qkv_rope_cache_impl(
        dev,
        hidden_size,
        q_size,
        kv_size,
        head_dim,
        max_seq_len,
        position,
        num_columns,
        tile_size_input,
        tile_size_output,
        trace_size,
        func_prefix=func_prefix,
        rms_kernel_object=rms_kernel_object,
        gemv_kernel_object=gemv_kernel_object,
        rope_kernel_object=rope_kernel_object,
    )


def qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax(
    dev,
    hidden_size,
    q_size,
    kv_size,
    head_dim,
    max_seq_len,
    position,
    num_columns,
    tile_size_input,
    tile_size_output,
    trace_size,
    func_prefix="",
    rms_kernel_object="rms_norm.o",
    gemv_kernel_object="mv.o",
    rope_kernel_object="rope.o",
    attention_kernel_object="qwen3_attention.o",
    softmax_kernel_object="softmax.o",
):
    """Single-token Qwen3 persistent stage through attention scores and softmax."""
    return _qwen3_persistent_input_rmsnorm_qkv_rope_cache_impl(
        dev,
        hidden_size,
        q_size,
        kv_size,
        head_dim,
        max_seq_len,
        position,
        num_columns,
        tile_size_input,
        tile_size_output,
        trace_size,
        func_prefix=func_prefix,
        rms_kernel_object=rms_kernel_object,
        gemv_kernel_object=gemv_kernel_object,
        rope_kernel_object=rope_kernel_object,
        include_scores_softmax=True,
        attention_kernel_object=attention_kernel_object,
        softmax_kernel_object=softmax_kernel_object,
    )
