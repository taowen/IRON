#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer


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
    intermediate_size=3072,
    func_prefix="",
    rms_kernel_object="rms_norm.o",
    gemv_kernel_object="mv.o",
    rope_kernel_object="rope.o",
    include_scores_softmax=False,
    include_context=False,
    include_o_proj=False,
    include_full_mlp=False,
    attention_kernel_object="qwen3_attention.o",
    passthrough_kernel_object="passThrough.o",
    softmax_kernel_object="softmax.o",
    o_gemv_kernel_object="mv_o_proj.o",
    add_kernel_object="add.o",
    mlp_gemv_kernel_object="mv_mlp.o",
    silu_kernel_object="silu.o",
    mul_kernel_object="mul.o",
    down_gemv_kernel_object="mv_down.o",
    layer_iterations=1,
    final_output_only=False,
):
    """Single-token Qwen3 input RMSNorm, QKV, Q/K norm, RoPE, KV write, and optional softmax."""
    dtype = bfloat16
    q_heads = q_size // head_dim
    kv_heads = kv_size // head_dim
    cache_block_seq = 64
    if include_context and not include_scores_softmax:
        raise ValueError("context checkpoint requires scores+softmax")
    if include_o_proj and not include_context:
        raise ValueError("O projection checkpoint requires attention context")
    if include_full_mlp and not include_o_proj:
        raise ValueError("full MLP checkpoint requires attention O projection")
    include_k_cache_debug = include_scores_softmax and not include_o_proj
    score_size = q_heads * max_seq_len if include_scores_softmax else 0
    qk_pair_debug_size = kv_heads * 3 * head_dim if include_scores_softmax else 0
    k_cache_debug_size = (
        kv_heads * max_seq_len * head_dim if include_k_cache_debug else 0
    )
    v_cache_debug_size = kv_heads * max_seq_len * head_dim if include_context else 0
    context_size = q_size if include_context else 0
    context_flat_size = q_size if include_o_proj else 0
    o_proj_size = hidden_size if include_o_proj else 0
    residual_size = hidden_size if include_o_proj else 0
    mlp_xnorm_size = hidden_size if include_full_mlp else 0
    mlp_ffn_size = intermediate_size if include_full_mlp else 0
    mlp_ffn_out_size = hidden_size if include_full_mlp else 0
    mlp_layer_residual_size = hidden_size if include_full_mlp else 0
    mlp_weights_size = (
        hidden_size + 3 * intermediate_size * hidden_size if include_full_mlp else 0
    )
    weights_size = (
        hidden_size
        + q_size * hidden_size
        + 2 * kv_size * hidden_size
        + 2 * head_dim
        + (hidden_size * q_size if include_o_proj else 0)
        + mlp_weights_size
    )
    stage_outputs_size = (
        hidden_size
        + q_size
        + kv_size
        + q_size
        + kv_size
        + q_size
        + qk_pair_debug_size
        + k_cache_debug_size
        + 2 * score_size
        + v_cache_debug_size
        + context_size
        + context_flat_size
        + o_proj_size
        + residual_size
        + mlp_xnorm_size
        + 4 * mlp_ffn_size
        + mlp_ffn_out_size
        + mlp_layer_residual_size
    )
    outputs_size = hidden_size if final_output_only else stage_outputs_size
    cache_size = 2 * kv_size * max_seq_len

    tensor_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    weights_ty = np.ndarray[(weights_size,), np.dtype[dtype]]
    weights_pair_ty = np.ndarray[(2 * weights_size,), np.dtype[dtype]]
    angles_ty = np.ndarray[(head_dim,), np.dtype[dtype]]
    outputs_ty = np.ndarray[(outputs_size,), np.dtype[dtype]]
    cache_ty = np.ndarray[(cache_size,), np.dtype[dtype]]
    cache_pair_ty = np.ndarray[(2 * cache_size,), np.dtype[dtype]]
    tile_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    hidden_weight_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    head_ty = np.ndarray[(head_dim,), np.dtype[dtype]]
    qk_norm_weight_ty = np.ndarray[(2 * head_dim,), np.dtype[dtype]]
    ffn_ty = np.ndarray[(intermediate_size,), np.dtype[dtype]]
    qk_pair_ty = np.ndarray[(3 * head_dim,), np.dtype[dtype]]
    score_ty = np.ndarray[(max_seq_len,), np.dtype[dtype]]
    k_cache_block_ty = np.ndarray[(cache_block_seq, head_dim), np.dtype[dtype]]
    v_cache_block_ty = np.ndarray[(cache_block_seq, head_dim), np.dtype[dtype]]
    gemv_a_ty = np.ndarray[(tile_size_input, hidden_size), np.dtype[dtype]]
    context_flat_ty = np.ndarray[(q_size,), np.dtype[dtype]]
    o_gemv_a_ty = np.ndarray[(tile_size_input, q_size), np.dtype[dtype]]
    mlp_gemv_a_ty = np.ndarray[(tile_size_input, hidden_size), np.dtype[dtype]]
    down_gemv_a_ty = np.ndarray[(tile_size_input, intermediate_size), np.dtype[dtype]]

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
    if include_full_mlp and num_columns != 1:
        raise ValueError("full-layer checkpoint is currently single-column only")
    if include_full_mlp and hidden_size != 1024:
        raise ValueError("full-layer checkpoint expects hidden_size=1024")
    if include_full_mlp and intermediate_size != 3072:
        raise ValueError("full-layer checkpoint expects intermediate_size=3072")
    if layer_iterations not in (1, 2):
        raise ValueError("layer_iterations currently supports only 1 or 2")
    if final_output_only and not include_full_mlp:
        raise ValueError("final_output_only requires the full MLP stage")

    runtime_hidden_in = ObjectFifo(tile_ty, name="qwen3_rc_hidden_in", depth=2)
    in_hidden = runtime_hidden_in
    hidden_feedback = None
    final_layer_residual = None
    if layer_iterations > 1:
        in_hidden = ObjectFifo(tile_ty, name="qwen3_two_layer_hidden", depth=2)
        hidden_feedback = ObjectFifo(
            tile_ty, name="qwen3_two_layer_hidden_feedback", depth=2
        )
        final_layer_residual = ObjectFifo(
            tile_ty, name="qwen3_two_layer_final_residual", depth=2
        )
    in_weight = ObjectFifo(hidden_weight_ty, name="qwen3_rc_input_norm_weight", depth=2)
    normed = ObjectFifo(tile_ty, name="qwen3_rc_input_norm_unweighted", depth=2)
    xnorm = ObjectFifo(tile_ty, name="qwen3_rc_xnorm", depth=2)

    q_norm_weight = ObjectFifo(head_ty, name="qwen3_rc_q_norm_weight", depth=2)
    k_norm_weight = ObjectFifo(head_ty, name="qwen3_rc_k_norm_weight", depth=2)
    qk_norm_weight = (
        ObjectFifo(qk_norm_weight_ty, name="qwen3_rc_qk_norm_weight", depth=2)
        if include_full_mlp
        else None
    )
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
    q_norm_fifos = (
        []
        if include_full_mlp
        else [
            ObjectFifo(head_ty, name=f"qwen3_rc_q_norm_{col}", depth=2)
            for col in range(num_columns)
        ]
    )
    k_norm_fifos = (
        []
        if include_full_mlp
        else [
            ObjectFifo(head_ty, name=f"qwen3_rc_k_norm_{col}", depth=2)
            for col in range(num_columns)
        ]
    )
    q_rope_fifos = [
        ObjectFifo(head_ty, name=f"qwen3_rc_q_rope_{col}", depth=2)
        for col in range(num_columns)
    ]
    k_rope_fifos = [
        ObjectFifo(head_ty, name=f"qwen3_rc_k_rope_{col}", depth=2)
        for col in range(num_columns)
    ]
    k_cache_fifos = []
    k_cache_debug_in_fifos = []
    k_cache_debug_out_fifos = []
    qk_pair_fifos = []
    qk_rope_meta = None
    qk_pair_debug_fifos = []
    attn_score_debug_fifos = []
    attn_score_softmax_fifos = []
    attn_weight_fifos = []
    v_cache_raw_fifos = []
    v_context_fifos = []
    v_context_debug_fifos = []
    attn_context_fifos = []
    attn_context_flat_fifos = []
    o_weight_fifos = []
    o_proj_fifos = []
    residual_hidden_fifos = []
    residual_out_fifos = []
    mlp_gate_up_weight_rows = None
    mlp_down_weight = None
    mlp_xnorm = None
    ffn_gate = None
    ffn_up = None
    ffn_hidden = None
    ffn_out = None
    layer_residual = None
    if include_scores_softmax:
        if include_full_mlp:
            qk_rope_meta = ObjectFifo(
                qk_pair_ty, name="qwen3_rc_qk_rope_metadata", depth=2
            )
        qk_pair_fifos = [
            ObjectFifo(qk_pair_ty, name=f"qwen3_rc_qk_pair_{col}", depth=2)
            for col in range(num_columns)
        ]
        qk_pair_debug_fifos = [
            ObjectFifo(qk_pair_ty, name=f"qwen3_rc_qk_pair_debug_{col}", depth=2)
            for col in range(num_columns)
        ]
        k_cache_fifos = [
            ObjectFifo(k_cache_block_ty, name=f"qwen3_rc_k_cache_{col}", depth=2)
            for col in range(num_columns)
        ]
        if include_k_cache_debug:
            k_cache_debug_in_fifos = [
                ObjectFifo(
                    k_cache_block_ty,
                    name=f"qwen3_rc_k_cache_debug_in_{col}",
                    depth=1,
                )
                for col in range(num_columns)
            ]
            k_cache_debug_out_fifos = [
                ObjectFifo(
                    k_cache_block_ty,
                    name=f"qwen3_rc_k_cache_debug_out_{col}",
                    depth=1,
                )
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
    if include_context:
        v_cache_raw_fifos = [
            ObjectFifo(v_cache_block_ty, name=f"qwen3_rc_v_cache_{col}", depth=1)
            for col in range(num_columns)
        ]
        v_context_fifos = [
            ObjectFifo(
                v_cache_block_ty,
                name=f"qwen3_rc_v_context_block_{col}",
                depth=1,
            )
            for col in range(num_columns)
        ]
        v_context_debug_fifos = [
            ObjectFifo(
                v_cache_block_ty,
                name=f"qwen3_rc_v_context_debug_{col}",
                depth=1,
            )
            for col in range(num_columns)
        ]
        attn_context_fifos = [
            ObjectFifo(head_ty, name=f"qwen3_rc_attn_context_{col}", depth=2)
            for col in range(num_columns)
        ]
    if include_o_proj:
        attn_context_flat_fifos = [
            ObjectFifo(
                context_flat_ty,
                name=f"qwen3_rc_attn_context_flat_{col}",
                depth=1,
            )
            for col in range(num_columns)
        ]
        o_weight_fifos = [
            ObjectFifo(o_gemv_a_ty, name=f"qwen3_rc_o_weight_{col}", depth=2)
            for col in range(num_columns)
        ]
        o_proj_ty = tile_ty if include_full_mlp else head_ty
        o_proj_depth = 1 if include_full_mlp else 2
        o_proj_fifos = [
            ObjectFifo(
                o_proj_ty, name=f"qwen3_rc_attn_o_proj_{col}", depth=o_proj_depth
            )
            for col in range(num_columns)
        ]
        if not include_full_mlp:
            residual_hidden_fifos = [
                ObjectFifo(head_ty, name=f"qwen3_rc_residual_hidden_{col}", depth=2)
                for col in range(num_columns)
            ]
        residual_ty = tile_ty if include_full_mlp else head_ty
        residual_out_fifos = [
            ObjectFifo(residual_ty, name=f"qwen3_rc_attn_residual_{col}", depth=2)
            for col in range(num_columns)
        ]
    if include_full_mlp:
        mlp_gate_up_weight_rows = ObjectFifo(
            hidden_weight_ty, name="qwen3_full_layer_mlp_gate_up_weight_rows", depth=4
        )
        mlp_down_weight = ObjectFifo(
            down_gemv_a_ty, name="qwen3_full_layer_mlp_down_weight", depth=1
        )
        mlp_xnorm = ObjectFifo(tile_ty, name="qwen3_full_layer_mlp_xnorm", depth=2)
        ffn_gate = ObjectFifo(ffn_ty, name="qwen3_full_layer_ffn_gate", depth=2)
        ffn_up = ObjectFifo(ffn_ty, name="qwen3_full_layer_ffn_up", depth=2)
        ffn_hidden = ObjectFifo(ffn_ty, name="qwen3_full_layer_ffn_hidden", depth=2)
        ffn_out = ObjectFifo(tile_ty, name="qwen3_full_layer_ffn_out", depth=2)
        layer_residual = ObjectFifo(tile_ty, name="qwen3_full_layer_residual", depth=2)

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
    pass_through_tile = None
    mask = None
    softmax = None
    merge_current_v = None
    attention_context = None
    pack_context_head = None
    o_matvec = None
    add_kernel = None
    add_hidden_tile_to_full = None
    add_full_slice = None
    hidden_copy = None
    norm_rope_with_weight_offset = None
    pack_qk_rope_metadata = None
    norm_rope_with_metadata = None
    mlp_weighted_rms_norm = None
    mlp_matvec = None
    mlp_matvec4_rows = None
    silu = None
    eltwise_mul = None
    silu_mul = None
    down_matvec = None
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
        pass_through_tile = Kernel(
            f"{func_prefix}passThroughTile",
            f"{func_prefix}{passthrough_kernel_object}",
            [k_cache_block_ty, k_cache_block_ty, np.int32, np.int32],
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
    if include_context:
        merge_current_v = Kernel(
            f"{func_prefix}qwen3_merge_current_v_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [v_cache_block_ty, head_ty, v_cache_block_ty, np.int32, np.int32],
        )
        attention_context = Kernel(
            f"{func_prefix}qwen3_attention_context_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [score_ty, v_cache_block_ty, head_ty, np.int32, np.int32],
        )
    if include_o_proj:
        pack_context_head = Kernel(
            f"{func_prefix}qwen3_pack_context_head_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [head_ty, context_flat_ty, np.int32],
        )
        o_matvec_out_ty = tile_ty if include_full_mlp else head_ty
        o_matvec = Kernel(
            f"{func_prefix}qwen3_o_proj_matvec_vectorized_bf16_bf16",
            f"{func_prefix}{o_gemv_kernel_object}",
            [np.int32, np.int32, o_gemv_a_ty, context_flat_ty, o_matvec_out_ty],
        )
        if include_full_mlp:
            add_hidden_tile_to_full = Kernel(
                f"{func_prefix}qwen3_add_hidden_tile_to_full_bf16",
                f"{func_prefix}{attention_kernel_object}",
                [head_ty, tile_ty, tile_ty, np.int32, np.int32],
            )
        else:
            add_kernel = Kernel(
                f"{func_prefix}eltwise_add_bf16_vector",
                f"{func_prefix}{add_kernel_object}",
                [head_ty, head_ty, head_ty, np.int32],
            )
    if include_full_mlp:
        norm_rope_with_weight_offset = Kernel(
            f"{func_prefix}qwen3_norm_rope_with_weight_offset_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [
                head_ty,
                qk_norm_weight_ty,
                head_ty,
                head_ty,
                head_ty,
                np.int32,
                np.int32,
            ],
        )
        pack_qk_rope_metadata = Kernel(
            f"{func_prefix}qwen3_pack_qk_rope_metadata_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [qk_norm_weight_ty, head_ty, qk_pair_ty, np.int32],
        )
        norm_rope_with_metadata = Kernel(
            f"{func_prefix}qwen3_norm_rope_with_metadata_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [
                head_ty,
                qk_pair_ty,
                head_ty,
                head_ty,
                np.int32,
                np.int32,
            ],
        )
        mlp_weighted_rms_norm = Kernel(
            f"{func_prefix}qwen3_weighted_rms_norm_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [tile_ty, hidden_weight_ty, tile_ty, np.int32],
        )
        mlp_matvec = Kernel(
            f"{func_prefix}qwen3_mlp_matvec_vectorized_bf16_bf16",
            f"{func_prefix}{mlp_gemv_kernel_object}",
            [np.int32, np.int32, mlp_gemv_a_ty, tile_ty, ffn_ty],
        )
        mlp_matvec4_rows = Kernel(
            f"{func_prefix}qwen3_mlp_matvec4_rows_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [
                np.int32,
                np.int32,
                hidden_weight_ty,
                hidden_weight_ty,
                hidden_weight_ty,
                hidden_weight_ty,
                tile_ty,
                ffn_ty,
            ],
        )
        silu = Kernel(
            f"{func_prefix}silu_bf16",
            f"{func_prefix}{silu_kernel_object}",
            [ffn_ty, ffn_ty, np.int32],
        )
        eltwise_mul = Kernel(
            f"{func_prefix}eltwise_mul_bf16_vector",
            f"{func_prefix}{mul_kernel_object}",
            [ffn_ty, ffn_ty, ffn_ty, np.int32],
        )
        silu_mul = Kernel(
            f"{func_prefix}qwen3_silu_mul_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [ffn_ty, ffn_ty, ffn_ty, np.int32],
        )
        down_matvec = Kernel(
            f"{func_prefix}qwen3_down_proj_matvec_vectorized_bf16_bf16",
            f"{func_prefix}{down_gemv_kernel_object}",
            [np.int32, np.int32, down_gemv_a_ty, ffn_ty, tile_ty],
        )
        add_full_slice = Kernel(
            f"{func_prefix}qwen3_add_full_slice_bf16",
            f"{func_prefix}{attention_kernel_object}",
            [tile_ty, tile_ty, tile_ty, np.int32, np.int32],
        )
        if layer_iterations > 1:
            hidden_copy = Kernel(
                f"{func_prefix}qwen3_copy_bf16",
                f"{func_prefix}{attention_kernel_object}",
                [tile_ty, tile_ty, np.int32],
            )

    def rmsnorm_worker(of_in, of_out, rms_norm):
        for _ in range_(layer_iterations):
            hidden = of_in.acquire(1)
            tmp = of_out.acquire(1)
            rms_norm(hidden, tmp, hidden_size)
            of_in.release(1)
            of_out.release(1)

    def weight_worker(of_in, of_weight, of_out, mul):
        for _ in range_(layer_iterations):
            weight = of_weight.acquire(1)
            tmp = of_in.acquire(1)
            out = of_out.acquire(1)
            mul(tmp, weight, out, hidden_size)
            of_out.release(1)
            of_in.release(1)
            of_weight.release(1)

    def q_matvec_worker(weight_fifo, x_fifo, out_fifo, matvec_kernel):
        for _ in range_(layer_iterations):
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
        for _ in range_(layer_iterations):
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
        for _ in range_(layer_iterations):
            weight = weight_fifo.acquire(1)
            for _ in range_(q_heads // num_columns):
                raw = raw_fifo.acquire(1)
                out = out_fifo.acquire(1)
                norm_kernel(raw, weight, out, head_dim)
                raw_fifo.release(1)
                out_fifo.release(1)
            weight_fifo.release(1)

    def k_head_norm_worker(raw_fifo, weight_fifo, out_fifo, norm_kernel):
        for _ in range_(layer_iterations):
            weight = weight_fifo.acquire(1)
            for _ in range_(kv_heads // num_columns):
                raw = raw_fifo.acquire(1)
                out = out_fifo.acquire(1)
                norm_kernel(raw, weight, out, head_dim)
                raw_fifo.release(1)
                out_fifo.release(1)
            weight_fifo.release(1)

    def q_rope_worker(in_fifo, angles_fifo, out_fifo, rope_kernel):
        for _ in range_(layer_iterations):
            angles = angles_fifo.acquire(1)
            for _ in range_(q_heads // num_columns):
                elem_in = in_fifo.acquire(1)
                elem_out = out_fifo.acquire(1)
                rope_kernel(elem_in, angles, elem_out, head_dim)
                in_fifo.release(1)
                out_fifo.release(1)
            angles_fifo.release(1)

    def k_rope_worker(in_fifo, angles_fifo, out_fifo, rope_kernel):
        for _ in range_(layer_iterations):
            angles = angles_fifo.acquire(1)
            for _ in range_(kv_heads // num_columns):
                elem_in = in_fifo.acquire(1)
                elem_out = out_fifo.acquire(1)
                rope_kernel(elem_in, angles, elem_out, head_dim)
                in_fifo.release(1)
                out_fifo.release(1)
            angles_fifo.release(1)

    def q_project_norm_rope_worker(
        weight_fifo,
        x_fifo,
        norm_weight_fifo,
        angles_fifo,
        raw_fifo,
        norm_fifo,
        rope_fifo,
        matvec_kernel,
        norm_rope_kernel,
    ):
        for _ in range_(layer_iterations):
            x = x_fifo.acquire(1)
            norm_weights = norm_weight_fifo.acquire(1)
            angles = angles_fifo.acquire(1)
            for _ in range_(q_heads // num_columns):
                raw = raw_fifo.acquire(1)
                norm = norm_fifo.acquire(1)
                rope_out = rope_fifo.acquire(1)
                for j_idx in range_(tile_size_output // tile_size_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    output_row_offset = j_i32 * tile_size_input
                    w = weight_fifo.acquire(1)
                    matvec_kernel(tile_size_input, output_row_offset, w, x, raw)
                    weight_fifo.release(1)
                norm_rope_kernel(raw, norm_weights, angles, norm, rope_out, 0, head_dim)
                raw_fifo.release(1)
                norm_fifo.release(1)
                rope_fifo.release(1)
            angles_fifo.release(1)
            norm_weight_fifo.release(1)
            x_fifo.release(1)

    def k_project_norm_rope_worker(
        weight_fifo,
        x_fifo,
        norm_weight_fifo,
        angles_fifo,
        raw_fifo,
        norm_fifo,
        rope_fifo,
        matvec_kernel,
        norm_rope_kernel,
    ):
        for _ in range_(layer_iterations):
            x = x_fifo.acquire(1)
            norm_weights = norm_weight_fifo.acquire(1)
            angles = angles_fifo.acquire(1)
            for _ in range_(kv_heads // num_columns):
                raw = raw_fifo.acquire(1)
                norm = norm_fifo.acquire(1)
                rope_out = rope_fifo.acquire(1)
                for j_idx in range_(tile_size_output // tile_size_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    output_row_offset = j_i32 * tile_size_input
                    w = weight_fifo.acquire(1)
                    matvec_kernel(tile_size_input, output_row_offset, w, x, raw)
                    weight_fifo.release(1)
                norm_rope_kernel(
                    raw, norm_weights, angles, norm, rope_out, head_dim, head_dim
                )
                raw_fifo.release(1)
                norm_fifo.release(1)
                rope_fifo.release(1)
            angles_fifo.release(1)
            norm_weight_fifo.release(1)
            x_fifo.release(1)

    def q_project_norm_rope_buffered_worker(
        weight_fifo,
        x_fifo,
        norm_weight_fifo,
        angles_fifo,
        rope_fifo,
        matvec_kernel,
        norm_rope_kernel,
        raw_buffer,
        norm_buffer,
    ):
        for _ in range_(layer_iterations):
            x = x_fifo.acquire(1)
            norm_weights = norm_weight_fifo.acquire(1)
            angles = angles_fifo.acquire(1)
            for _ in range_(q_heads // num_columns):
                rope_out = rope_fifo.acquire(1)
                for j_idx in range_(tile_size_output // tile_size_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    output_row_offset = j_i32 * tile_size_input
                    w = weight_fifo.acquire(1)
                    matvec_kernel(tile_size_input, output_row_offset, w, x, raw_buffer)
                    weight_fifo.release(1)
                norm_rope_kernel(
                    raw_buffer, norm_weights, angles, norm_buffer, rope_out, 0, head_dim
                )
                rope_fifo.release(1)
            angles_fifo.release(1)
            norm_weight_fifo.release(1)
            x_fifo.release(1)

    def k_project_norm_rope_buffered_worker(
        weight_fifo,
        x_fifo,
        norm_weight_fifo,
        angles_fifo,
        rope_fifo,
        matvec_kernel,
        norm_rope_kernel,
        raw_buffer,
        norm_buffer,
    ):
        for _ in range_(layer_iterations):
            x = x_fifo.acquire(1)
            norm_weights = norm_weight_fifo.acquire(1)
            angles = angles_fifo.acquire(1)
            for _ in range_(kv_heads // num_columns):
                rope_out = rope_fifo.acquire(1)
                for j_idx in range_(tile_size_output // tile_size_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    output_row_offset = j_i32 * tile_size_input
                    w = weight_fifo.acquire(1)
                    matvec_kernel(tile_size_input, output_row_offset, w, x, raw_buffer)
                    weight_fifo.release(1)
                norm_rope_kernel(
                    raw_buffer,
                    norm_weights,
                    angles,
                    norm_buffer,
                    rope_out,
                    head_dim,
                    head_dim,
                )
                rope_fifo.release(1)
            angles_fifo.release(1)
            norm_weight_fifo.release(1)
            x_fifo.release(1)

    def qk_rope_metadata_worker(norm_weight_fifo, angles_fifo, meta_fifo, pack_kernel):
        for _ in range_(layer_iterations):
            norm_weights = norm_weight_fifo.acquire(1)
            angles = angles_fifo.acquire(1)
            meta = meta_fifo.acquire(1)
            pack_kernel(norm_weights, angles, meta, head_dim)
            meta_fifo.release(1)
            angles_fifo.release(1)
            norm_weight_fifo.release(1)

    def q_norm_rope_metadata_worker(
        raw_fifo,
        meta_fifo,
        rope_fifo,
        norm_rope_kernel,
    ):
        for _ in range_(layer_iterations):
            meta = meta_fifo.acquire(1)
            for _ in range_(q_heads // num_columns):
                raw = raw_fifo.acquire(1)
                rope_out = rope_fifo.acquire(1)
                norm_rope_kernel(raw, meta, raw, rope_out, 0, head_dim)
                rope_fifo.release(1)
                raw_fifo.release(1)
            meta_fifo.release(1)

    def k_norm_rope_metadata_worker(
        raw_fifo,
        meta_fifo,
        rope_fifo,
        norm_rope_kernel,
    ):
        for _ in range_(layer_iterations):
            meta = meta_fifo.acquire(1)
            for _ in range_(kv_heads // num_columns):
                raw = raw_fifo.acquire(1)
                rope_out = rope_fifo.acquire(1)
                norm_rope_kernel(raw, meta, raw, rope_out, head_dim, head_dim)
                rope_fifo.release(1)
                raw_fifo.release(1)
            meta_fifo.release(1)

    def qk_pair_worker(q_fifo, current_k_fifo, pair_fifo, pair_debug_fifo, pack_kernel):
        for _ in range_(layer_iterations):
            for _ in range_(kv_heads // num_columns):
                current_k = current_k_fifo.acquire(1)
                pair = pair_fifo.acquire(1)
                pair_debug = pair_debug_fifo.acquire(1)
                for q_select in range_(q_heads // kv_heads):
                    q_select_i32 = index.casts(T.i32(), q_select)
                    q = q_fifo.acquire(1)
                    pack_kernel(q, current_k, pair, q_select_i32)
                    pack_kernel(q, current_k, pair_debug, q_select_i32)
                    q_fifo.release(1)
                pair_debug_fifo.release(1)
                pair_fifo.release(1)
                current_k_fifo.release(1)

    def qk_pair_final_worker(q_fifo, current_k_fifo, pair_fifo, pack_kernel):
        for _ in range_(layer_iterations):
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
        for _ in range_(layer_iterations):
            for _ in range_(kv_heads // num_columns):
                pair = pair_fifo.acquire(1)
                score_debug_pair = score_debug_fifo.acquire(2)
                score_softmax_pair = score_softmax_fifo.acquire(2)
                score_debug0 = score_debug_pair[0]
                score_softmax0 = score_softmax_pair[0]
                score_debug1 = score_debug_pair[1]
                score_softmax1 = score_softmax_pair[1]
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
                score_debug_fifo.release(2)
                score_softmax_fifo.release(2)

    def attention_score_final_worker(
        pair_fifo,
        k_cache_fifo,
        score_softmax_fifo,
        score_kernel,
    ):
        for _ in range_(layer_iterations):
            for _ in range_(kv_heads // num_columns):
                pair = pair_fifo.acquire(1)
                score_softmax_pair = score_softmax_fifo.acquire(2)
                score_softmax0 = score_softmax_pair[0]
                score_softmax1 = score_softmax_pair[1]
                for block_idx in range_(max_seq_len // cache_block_seq):
                    block_i32 = index.casts(T.i32(), block_idx)
                    row_base = block_i32 * cache_block_seq
                    k_cache = k_cache_fifo.acquire(1)
                    score_kernel(
                        pair,
                        k_cache,
                        score_softmax0,
                        score_softmax0,
                        position,
                        row_base,
                        0,
                    )
                    score_kernel(
                        pair,
                        k_cache,
                        score_softmax1,
                        score_softmax1,
                        position,
                        row_base,
                        1,
                    )
                    k_cache_fifo.release(1)
                pair_fifo.release(1)
                score_softmax_fifo.release(2)

    def attention_softmax_worker(score_fifo, weight_fifo, mask_kernel, softmax_kernel):
        for _ in range_(layer_iterations):
            for _ in range_(q_heads // num_columns):
                scores = score_fifo.acquire(1)
                weights = weight_fifo.acquire(1)
                mask_kernel(scores, position + 1, max_seq_len)
                softmax_kernel(scores, weights, max_seq_len)
                score_fifo.release(1)
                weight_fifo.release(1)

    def k_cache_debug_worker(in_fifo, out_fifo, copy_kernel):
        for _ in range_(layer_iterations):
            for _ in range_(kv_heads // num_columns):
                for _ in range_(max_seq_len // cache_block_seq):
                    block = in_fifo.acquire(1)
                    out = out_fifo.acquire(1)
                    copy_kernel(block, out, cache_block_seq, head_dim)
                    in_fifo.release(1)
                    out_fifo.release(1)

    def v_context_merge_worker(
        current_v_fifo,
        v_cache_fifo,
        v_context_fifo,
        v_debug_fifo,
        merge_kernel,
        copy_kernel,
    ):
        for _ in range_(layer_iterations):
            for _ in range_(kv_heads // num_columns):
                current_v = current_v_fifo.acquire(1)
                for block_idx in range_(max_seq_len // cache_block_seq):
                    block_i32 = index.casts(T.i32(), block_idx)
                    row_base = block_i32 * cache_block_seq
                    cached_v = v_cache_fifo.acquire(1)
                    merged_v = v_context_fifo.acquire(1)
                    debug_v = v_debug_fifo.acquire(1)
                    merge_kernel(cached_v, current_v, merged_v, position, row_base)
                    copy_kernel(merged_v, debug_v, cache_block_seq, head_dim)
                    v_cache_fifo.release(1)
                    v_context_fifo.release(1)
                    v_debug_fifo.release(1)
                current_v_fifo.release(1)

    def v_context_merge_final_worker(
        current_v_fifo,
        v_cache_fifo,
        v_context_fifo,
        merge_kernel,
    ):
        for _ in range_(layer_iterations):
            for _ in range_(kv_heads // num_columns):
                current_v = current_v_fifo.acquire(1)
                for block_idx in range_(max_seq_len // cache_block_seq):
                    block_i32 = index.casts(T.i32(), block_idx)
                    row_base = block_i32 * cache_block_seq
                    cached_v = v_cache_fifo.acquire(1)
                    merged_v = v_context_fifo.acquire(1)
                    merge_kernel(cached_v, current_v, merged_v, position, row_base)
                    v_cache_fifo.release(1)
                    v_context_fifo.release(1)
                current_v_fifo.release(1)

    def attention_context_worker(
        weight_fifo, v_context_fifo, context_fifo, context_kernel
    ):
        for _ in range_(layer_iterations):
            for _ in range_(kv_heads // num_columns):
                context_pair = context_fifo.acquire(2)
                context0 = context_pair[0]
                context1 = context_pair[1]
                weight_pair = weight_fifo.acquire(2)
                weights0 = weight_pair[0]
                weights1 = weight_pair[1]
                for block_idx in range_(max_seq_len // cache_block_seq):
                    block_i32 = index.casts(T.i32(), block_idx)
                    row_base = block_i32 * cache_block_seq
                    v_block = v_context_fifo.acquire(1)
                    context_kernel(weights0, v_block, context0, position, row_base)
                    context_kernel(weights1, v_block, context1, position, row_base)
                    v_context_fifo.release(1)
                weight_fifo.release(2)
                context_fifo.release(2)

    def attention_context_o_proj_worker(
        weight_fifo,
        v_context_fifo,
        context_fifo,
        flat_fifo,
        context_kernel,
        pack_kernel,
    ):
        for _ in range_(layer_iterations):
            flat = flat_fifo.acquire(1)
            for kv_head in range_(kv_heads // num_columns):
                kv_head_i32 = index.casts(T.i32(), kv_head)
                context_pair = context_fifo.acquire(2)
                context0 = context_pair[0]
                context1 = context_pair[1]
                weight_pair = weight_fifo.acquire(2)
                weights0 = weight_pair[0]
                weights1 = weight_pair[1]
                for block_idx in range_(max_seq_len // cache_block_seq):
                    block_i32 = index.casts(T.i32(), block_idx)
                    row_base = block_i32 * cache_block_seq
                    v_block = v_context_fifo.acquire(1)
                    context_kernel(weights0, v_block, context0, position, row_base)
                    context_kernel(weights1, v_block, context1, position, row_base)
                    v_context_fifo.release(1)
                pack_kernel(context0, flat, kv_head_i32 * 2)
                pack_kernel(context1, flat, kv_head_i32 * 2 + 1)
                weight_fifo.release(2)
                context_fifo.release(2)
            flat_fifo.release(1)

    def attention_context_o_proj_final_worker(
        weight_fifo,
        v_context_fifo,
        flat_fifo,
        context_kernel,
        pack_kernel,
        context0_buffer,
        context1_buffer,
    ):
        for _ in range_(layer_iterations):
            flat = flat_fifo.acquire(1)
            for kv_head in range_(kv_heads // num_columns):
                kv_head_i32 = index.casts(T.i32(), kv_head)
                weight_pair = weight_fifo.acquire(2)
                weights0 = weight_pair[0]
                weights1 = weight_pair[1]
                for block_idx in range_(max_seq_len // cache_block_seq):
                    block_i32 = index.casts(T.i32(), block_idx)
                    row_base = block_i32 * cache_block_seq
                    v_block = v_context_fifo.acquire(1)
                    context_kernel(
                        weights0, v_block, context0_buffer, position, row_base
                    )
                    context_kernel(
                        weights1, v_block, context1_buffer, position, row_base
                    )
                    v_context_fifo.release(1)
                pack_kernel(context0_buffer, flat, kv_head_i32 * 2)
                pack_kernel(context1_buffer, flat, kv_head_i32 * 2 + 1)
                weight_fifo.release(2)
            flat_fifo.release(1)

    def o_matvec_worker(weight_fifo, context_fifo, out_fifo, matvec_kernel):
        for _ in range_(layer_iterations):
            context = context_fifo.acquire(1)
            for _ in range_(hidden_size // tile_size_output // num_columns):
                c = out_fifo.acquire(1)
                for j_idx in range_(tile_size_output // tile_size_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    output_row_offset = j_i32 * tile_size_input
                    w = weight_fifo.acquire(1)
                    matvec_kernel(tile_size_input, output_row_offset, w, context, c)
                    weight_fifo.release(1)
                out_fifo.release(1)
            context_fifo.release(1)

    def o_matvec_full_worker(weight_fifo, context_fifo, out_fifo, matvec_kernel):
        for _ in range_(layer_iterations):
            context = context_fifo.acquire(1)
            out = out_fifo.acquire(1)
            for tile_idx in range_(hidden_size // tile_size_output // num_columns):
                tile_i32 = index.casts(T.i32(), tile_idx)
                for j_idx in range_(tile_size_output // tile_size_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    output_row_offset = (
                        tile_i32 * tile_size_output + j_i32 * tile_size_input
                    )
                    w = weight_fifo.acquire(1)
                    matvec_kernel(tile_size_input, output_row_offset, w, context, out)
                    weight_fifo.release(1)
            out_fifo.release(1)
            context_fifo.release(1)

    def residual_add_worker(hidden_fifo, o_proj_fifo, out_fifo, add):
        for _ in range_(layer_iterations):
            for _ in range_(hidden_size // tile_size_output // num_columns):
                hidden = hidden_fifo.acquire(1)
                o_proj = o_proj_fifo.acquire(1)
                out = out_fifo.acquire(1)
                add(hidden, o_proj, out, tile_size_output)
                hidden_fifo.release(1)
                o_proj_fifo.release(1)
                out_fifo.release(1)

    def residual_add_full_worker(hidden_fifo, o_proj_fifo, out_fifo, add):
        for _ in range_(layer_iterations):
            hidden = hidden_fifo.acquire(1)
            o_proj = o_proj_fifo.acquire(1)
            out = out_fifo.acquire(1)
            for tile_idx in range_(hidden_size // tile_size_output // num_columns):
                tile_i32 = index.casts(T.i32(), tile_idx)
                row_offset = tile_i32 * tile_size_output
                add(hidden, o_proj, out, row_offset, tile_size_output)
            out_fifo.release(1)
            o_proj_fifo.release(1)
            hidden_fifo.release(1)

    def mlp_postnorm_gate_up_worker(
        residual_fifo,
        gate_up_weight_rows_fifo,
        gate_fifo,
        up_fifo,
        norm_kernel,
        matvec4_rows_kernel,
        xnorm_buffer,
    ):
        for _ in range_(layer_iterations):
            residual = residual_fifo.acquire(1)
            post_norm = gate_up_weight_rows_fifo.acquire(1)
            gate_out = gate_fifo.acquire(1)
            up_out = up_fifo.acquire(1)
            norm_kernel(residual, post_norm, xnorm_buffer, hidden_size)
            gate_up_weight_rows_fifo.release(1)
            for j_idx in range_(intermediate_size // tile_size_input):
                j_i32 = index.casts(T.i32(), j_idx)
                output_row_offset = j_i32 * tile_size_input
                rows = gate_up_weight_rows_fifo.acquire(tile_size_input)
                matvec4_rows_kernel(
                    tile_size_input,
                    output_row_offset,
                    rows[0],
                    rows[1],
                    rows[2],
                    rows[3],
                    xnorm_buffer,
                    gate_out,
                )
                gate_up_weight_rows_fifo.release(tile_size_input)
            for j_idx in range_(intermediate_size // tile_size_input):
                j_i32 = index.casts(T.i32(), j_idx)
                output_row_offset = j_i32 * tile_size_input
                rows = gate_up_weight_rows_fifo.acquire(tile_size_input)
                matvec4_rows_kernel(
                    tile_size_input,
                    output_row_offset,
                    rows[0],
                    rows[1],
                    rows[2],
                    rows[3],
                    xnorm_buffer,
                    up_out,
                )
                gate_up_weight_rows_fifo.release(tile_size_input)
            up_fifo.release(1)
            gate_fifo.release(1)
            residual_fifo.release(1)

    def mlp_silu_mul_worker(gate_fifo, up_fifo, hidden_fifo, silu_mul_kernel):
        for _ in range_(layer_iterations):
            gate = gate_fifo.acquire(1)
            up = up_fifo.acquire(1)
            hidden_out = hidden_fifo.acquire(1)
            silu_mul_kernel(gate, up, hidden_out, intermediate_size)
            hidden_fifo.release(1)
            up_fifo.release(1)
            gate_fifo.release(1)

    def mlp_down_worker(
        hidden_fifo,
        down_weight_fifo,
        ffn_out_fifo,
        matvec_kernel,
    ):
        for _ in range_(layer_iterations):
            hidden = hidden_fifo.acquire(1)
            ffn = ffn_out_fifo.acquire(1)
            for tile_idx in range_(hidden_size // tile_size_output // num_columns):
                tile_i32 = index.casts(T.i32(), tile_idx)
                row_offset = tile_i32 * tile_size_output
                for j_idx in range_(tile_size_output // tile_size_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    output_row_offset = row_offset + j_i32 * tile_size_input
                    w = down_weight_fifo.acquire(1)
                    matvec_kernel(tile_size_input, output_row_offset, w, hidden, ffn)
                    down_weight_fifo.release(1)
            ffn_out_fifo.release(1)
            hidden_fifo.release(1)

    def mlp_layer_residual_worker(
        ffn_out_fifo,
        residual_fifo,
        layer_residual_fifo,
        add_kernel,
    ):
        for _ in range_(layer_iterations):
            ffn = ffn_out_fifo.acquire(1)
            residual = residual_fifo.acquire(1)
            out = layer_residual_fifo.acquire(1)
            for tile_idx in range_(hidden_size // tile_size_output // num_columns):
                tile_i32 = index.casts(T.i32(), tile_idx)
                row_offset = tile_i32 * tile_size_output
                add_kernel(residual, ffn, out, row_offset, tile_size_output)
            layer_residual_fifo.release(1)
            residual_fifo.release(1)
            ffn_out_fifo.release(1)

    def two_layer_initial_hidden_worker(initial_fifo, feedback_fifo, out_fifo, copy):
        initial = initial_fifo.acquire(1)
        out = out_fifo.acquire(1)
        copy(initial, out, hidden_size)
        out_fifo.release(1)
        initial_fifo.release(1)

        feedback = feedback_fifo.acquire(1)
        out = out_fifo.acquire(1)
        copy(feedback, out, hidden_size)
        out_fifo.release(1)
        feedback_fifo.release(1)

    def two_layer_residual_router_worker(
        layer_residual_fifo,
        feedback_fifo,
        final_fifo,
        copy,
    ):
        first = layer_residual_fifo.acquire(1)
        feedback = feedback_fifo.acquire(1)
        copy(first, feedback, hidden_size)
        feedback_fifo.release(1)
        layer_residual_fifo.release(1)

        second = layer_residual_fifo.acquire(1)
        final = final_fifo.acquire(1)
        copy(second, final, hidden_size)
        final_fifo.release(1)
        layer_residual_fifo.release(1)

    workers = []
    if layer_iterations > 1:
        workers.append(
            Worker(
                two_layer_initial_hidden_worker,
                [
                    runtime_hidden_in.cons(),
                    hidden_feedback.cons(),
                    in_hidden.prod(),
                    hidden_copy,
                ],
            )
        )
    workers.extend(
        [
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
    )
    if include_full_mlp:
        workers.append(
            Worker(
                qk_rope_metadata_worker,
                [
                    qk_norm_weight.cons(),
                    rope_angles.cons(),
                    qk_rope_meta.prod(),
                    pack_qk_rope_metadata,
                ],
            )
        )
    for col in range(num_columns):
        if include_full_mlp:
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
                    q_norm_rope_metadata_worker,
                    [
                        q_raw_fifos[col].cons(),
                        qk_rope_meta.cons(),
                        q_rope_fifos[col].prod(),
                        norm_rope_with_metadata,
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
                    k_norm_rope_metadata_worker,
                    [
                        k_raw_fifos[col].cons(),
                        qk_rope_meta.cons(),
                        k_rope_fifos[col].prod(),
                        norm_rope_with_metadata,
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
            ]
        else:
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
            if final_output_only:
                score_workers = [
                    Worker(
                        qk_pair_final_worker,
                        [
                            q_rope_fifos[col].cons(),
                            k_rope_fifos[col].cons(),
                            qk_pair_fifos[col].prod(),
                            pack_qk_pair,
                        ],
                    ),
                    Worker(
                        attention_score_final_worker,
                        [
                            qk_pair_fifos[col].cons(),
                            k_cache_fifos[col].cons(),
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
            else:
                score_workers = [
                    Worker(
                        qk_pair_worker,
                        [
                            q_rope_fifos[col].cons(),
                            k_rope_fifos[col].cons(),
                            qk_pair_fifos[col].prod(),
                            qk_pair_debug_fifos[col].prod(),
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
            if include_k_cache_debug:
                score_workers.append(
                    Worker(
                        k_cache_debug_worker,
                        [
                            k_cache_debug_in_fifos[col].cons(),
                            k_cache_debug_out_fifos[col].prod(),
                            pass_through_tile,
                        ],
                    )
                )
            workers.extend(score_workers)
        if include_context:
            if final_output_only:
                context_workers = [
                    Worker(
                        v_context_merge_final_worker,
                        [
                            v_fifos[col].cons(),
                            v_cache_raw_fifos[col].cons(),
                            v_context_fifos[col].prod(),
                            merge_current_v,
                        ],
                    )
                ]
            else:
                context_workers = [
                    Worker(
                        v_context_merge_worker,
                        [
                            v_fifos[col].cons(),
                            v_cache_raw_fifos[col].cons(),
                            v_context_fifos[col].prod(),
                            v_context_debug_fifos[col].prod(),
                            merge_current_v,
                            pass_through_tile,
                        ],
                    )
                ]
            if include_o_proj:
                if final_output_only:
                    context_workers.append(
                        Worker(
                            attention_context_o_proj_final_worker,
                            [
                                attn_weight_fifos[col].cons(),
                                v_context_fifos[col].cons(),
                                attn_context_flat_fifos[col].prod(),
                                attention_context,
                                pack_context_head,
                                Buffer(
                                    type=head_ty,
                                    name=f"qwen3_final_context0_buf_{col}",
                                ),
                                Buffer(
                                    type=head_ty,
                                    name=f"qwen3_final_context1_buf_{col}",
                                ),
                            ],
                        )
                    )
                else:
                    context_workers.append(
                        Worker(
                            attention_context_o_proj_worker,
                            [
                                attn_weight_fifos[col].cons(),
                                v_context_fifos[col].cons(),
                                attn_context_fifos[col].prod(),
                                attn_context_flat_fifos[col].prod(),
                                attention_context,
                                pack_context_head,
                            ],
                        )
                    )
            else:
                context_workers.append(
                    Worker(
                        attention_context_worker,
                        [
                            attn_weight_fifos[col].cons(),
                            v_context_fifos[col].cons(),
                            attn_context_fifos[col].prod(),
                            attention_context,
                        ],
                    )
                )
            workers.extend(context_workers)
        if include_o_proj:
            if include_full_mlp:
                workers.extend(
                    [
                        Worker(
                            o_matvec_full_worker,
                            [
                                o_weight_fifos[col].cons(),
                                attn_context_flat_fifos[col].cons(),
                                o_proj_fifos[col].prod(),
                                o_matvec,
                            ],
                        ),
                        Worker(
                            residual_add_full_worker,
                            [
                                in_hidden.cons(),
                                o_proj_fifos[col].cons(),
                                residual_out_fifos[col].prod(),
                                add_full_slice,
                            ],
                        ),
                    ]
                )
            else:
                workers.extend(
                    [
                        Worker(
                            o_matvec_worker,
                            [
                                o_weight_fifos[col].cons(),
                                attn_context_flat_fifos[col].cons(),
                                o_proj_fifos[col].prod(),
                                o_matvec,
                            ],
                        ),
                        Worker(
                            residual_add_worker,
                            [
                                residual_hidden_fifos[col].cons(),
                                o_proj_fifos[col].cons(),
                                residual_out_fifos[col].prod(),
                                add_kernel,
                            ],
                        ),
                    ]
                )

    if include_full_mlp:
        workers.extend(
            [
                Worker(
                    mlp_postnorm_gate_up_worker,
                    [
                        residual_out_fifos[0].cons(),
                        mlp_gate_up_weight_rows.cons(),
                        ffn_gate.prod(),
                        ffn_up.prod(),
                        mlp_weighted_rms_norm,
                        mlp_matvec4_rows,
                        Buffer(type=tile_ty, name="qwen3_full_layer_mlp_xnorm_buf"),
                    ],
                ),
                Worker(
                    mlp_silu_mul_worker,
                    [
                        ffn_gate.cons(),
                        ffn_up.cons(),
                        ffn_hidden.prod(),
                        silu_mul,
                    ],
                ),
                Worker(
                    mlp_down_worker,
                    [
                        ffn_hidden.cons(),
                        mlp_down_weight.cons(),
                        ffn_out.prod(),
                        down_matvec,
                    ],
                ),
                Worker(
                    mlp_layer_residual_worker,
                    [
                        ffn_out.cons(),
                        residual_out_fifos[0].cons(),
                        layer_residual.prod(),
                        add_full_slice,
                    ],
                ),
            ]
        )
        if layer_iterations > 1:
            workers.append(
                Worker(
                    two_layer_residual_router_worker,
                    [
                        layer_residual.cons(),
                        hidden_feedback.prod(),
                        final_layer_residual.prod(),
                        hidden_copy,
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
    q_norm_weight_base = v_weight_base + kv_size * hidden_size
    k_norm_weight_base = q_norm_weight_base + head_dim
    o_weight_base = k_norm_weight_base + head_dim
    mlp_post_norm_weight_base = o_weight_base + (
        hidden_size * q_size if include_o_proj else 0
    )
    mlp_gate_weight_base = mlp_post_norm_weight_base + hidden_size
    mlp_up_weight_base = mlp_gate_weight_base + intermediate_size * hidden_size
    mlp_down_weight_base = mlp_up_weight_base + intermediate_size * hidden_size

    def weight_taps_for_k(total_rows, k_size, base_offset):
        return [
            TensorAccessPattern(
                (weights_size,),
                base_offset + col * (total_rows // num_columns) * k_size,
                [1, 1, 1, (total_rows // num_columns) * k_size],
                [0, 0, 0, 1],
            )
            for col in range(num_columns)
        ]

    def weight_taps(total_rows, base_offset):
        return weight_taps_for_k(total_rows, hidden_size, base_offset)

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
    qk_norm_weight_tap = (
        TensorAccessPattern(
            (weights_size,),
            q_norm_weight_base,
            [1, 1, 1, 2 * head_dim],
            [0, 0, 0, 1],
        )
        if include_full_mlp
        else None
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
    qk_pair_output_base = q_rope_output_base + q_size
    k_cache_stream_output_base = qk_pair_output_base + qk_pair_debug_size
    attn_scores_output_base = k_cache_stream_output_base + k_cache_debug_size
    attn_weights_output_base = attn_scores_output_base + score_size
    v_context_stream_output_base = attn_weights_output_base + score_size
    attn_context_output_base = v_context_stream_output_base + v_cache_debug_size
    attn_context_flat_output_base = attn_context_output_base + context_size
    attn_o_proj_output_base = attn_context_flat_output_base + context_flat_size
    attn_residual_output_base = attn_o_proj_output_base + o_proj_size
    mlp_xnorm_output_base = attn_residual_output_base + residual_size
    ffn_gate_output_base = mlp_xnorm_output_base + mlp_xnorm_size
    ffn_up_output_base = ffn_gate_output_base + mlp_ffn_size
    ffn_gate_silu_output_base = ffn_up_output_base + mlp_ffn_size
    ffn_hidden_output_base = ffn_gate_silu_output_base + mlp_ffn_size
    ffn_out_output_base = ffn_hidden_output_base + mlp_ffn_size
    layer_residual_output_base = ffn_out_output_base + mlp_ffn_out_size
    output_tap_size = stage_outputs_size if final_output_only else outputs_size

    xnorm_output_tap = TensorAccessPattern(
        (output_tap_size,),
        xnorm_output_base,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )

    def out_taps(total_rows, base_offset):
        return [
            TensorAccessPattern(
                (output_tap_size,),
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
    cache_value_blocks_tap = TensorAccessPattern(
        (cache_size,),
        kv_size * max_seq_len,
        [
            kv_heads,
            max_seq_len // cache_block_seq,
            cache_block_seq,
            head_dim,
        ],
        [max_seq_len * head_dim, cache_block_seq * head_dim, head_dim, 1],
    )

    def layer_pair_weight_tap(layer_idx, base_offset, length):
        return TensorAccessPattern(
            (2 * weights_size,),
            layer_idx * weights_size + base_offset,
            [1, 1, 1, length],
            [0, 0, 0, 1],
        )

    def layer_pair_weight_taps_for_k(layer_idx, total_rows, k_size, base_offset):
        return [
            layer_pair_weight_tap(
                layer_idx,
                base_offset + col * (total_rows // num_columns) * k_size,
                (total_rows // num_columns) * k_size,
            )
            for col in range(num_columns)
        ]

    def layer_pair_weight_taps(layer_idx, total_rows, base_offset):
        return layer_pair_weight_taps_for_k(
            layer_idx, total_rows, hidden_size, base_offset
        )

    def layer_pair_cache_key_blocks_tap(layer_idx):
        return TensorAccessPattern(
            (2 * cache_size,),
            layer_idx * cache_size,
            [
                kv_heads,
                max_seq_len // cache_block_seq,
                cache_block_seq,
                head_dim,
            ],
            [max_seq_len * head_dim, cache_block_seq * head_dim, head_dim, 1],
        )

    def layer_pair_cache_value_blocks_tap(layer_idx):
        return TensorAccessPattern(
            (2 * cache_size,),
            layer_idx * cache_size + kv_size * max_seq_len,
            [
                kv_heads,
                max_seq_len // cache_block_seq,
                cache_block_seq,
                head_dim,
            ],
            [max_seq_len * head_dim, cache_block_seq * head_dim, head_dim, 1],
        )

    def layer_pair_cache_key_tap(layer_idx):
        return TensorAccessPattern(
            (2 * cache_size,),
            layer_idx * cache_size + position * head_dim,
            [1, 1, kv_heads, head_dim],
            [0, 0, max_seq_len * head_dim, 1],
        )

    def layer_pair_cache_value_tap(layer_idx):
        return TensorAccessPattern(
            (2 * cache_size,),
            layer_idx * cache_size + kv_size * max_seq_len + position * head_dim,
            [1, 1, kv_heads, head_dim],
            [0, 0, max_seq_len * head_dim, 1],
        )

    q_weight_taps = weight_taps(q_size, q_weight_base)
    k_weight_taps = weight_taps(kv_size, k_weight_base)
    v_weight_taps = weight_taps(kv_size, v_weight_base)
    q_raw_output_taps = out_taps(q_size, q_raw_output_base)
    k_raw_output_taps = out_taps(kv_size, k_raw_output_base)
    q_norm_output_taps = out_taps(q_size, q_norm_output_base)
    k_norm_output_taps = out_taps(kv_size, k_norm_output_base)
    q_rope_output_taps = out_taps(q_size, q_rope_output_base)
    qk_pair_output_taps = (
        out_taps(qk_pair_debug_size, qk_pair_output_base)
        if include_scores_softmax
        else []
    )
    k_cache_stream_output_taps = (
        out_taps(k_cache_debug_size, k_cache_stream_output_base)
        if include_k_cache_debug
        else []
    )
    attn_score_output_taps = (
        out_taps(score_size, attn_scores_output_base) if include_scores_softmax else []
    )
    attn_weight_output_taps = (
        out_taps(score_size, attn_weights_output_base) if include_scores_softmax else []
    )
    v_context_output_taps = (
        out_taps(v_cache_debug_size, v_context_stream_output_base)
        if include_context
        else []
    )
    attn_context_output_taps = (
        out_taps(context_size, attn_context_output_base) if include_context else []
    )
    attn_context_flat_output_taps = (
        out_taps(context_flat_size, attn_context_flat_output_base)
        if include_o_proj
        else []
    )
    o_weight_taps = (
        weight_taps_for_k(hidden_size, q_size, o_weight_base) if include_o_proj else []
    )
    residual_hidden_taps = (
        [
            TensorAccessPattern(
                (1, hidden_size),
                col * (hidden_size // num_columns),
                [1, 1, 1, hidden_size // num_columns],
                [0, 0, 0, 1],
            )
            for col in range(num_columns)
        ]
        if include_o_proj and not include_full_mlp
        else []
    )
    o_proj_output_taps = (
        out_taps(o_proj_size, attn_o_proj_output_base) if include_o_proj else []
    )
    residual_output_taps = (
        out_taps(residual_size, attn_residual_output_base) if include_o_proj else []
    )
    mlp_gate_up_weight_rows_tap = (
        TensorAccessPattern(
            (weights_size,),
            mlp_post_norm_weight_base,
            [1, 1, 1, hidden_size + 2 * intermediate_size * hidden_size],
            [0, 0, 0, 1],
        )
        if include_full_mlp
        else None
    )
    mlp_down_weight_tap = (
        TensorAccessPattern(
            (weights_size,),
            mlp_down_weight_base,
            [1, 1, 1, hidden_size * intermediate_size],
            [0, 0, 0, 1],
        )
        if include_full_mlp
        else None
    )
    mlp_xnorm_output_tap = (
        TensorAccessPattern(
            (output_tap_size,),
            mlp_xnorm_output_base,
            [1, 1, 1, hidden_size],
            [0, 0, 0, 1],
        )
        if include_full_mlp
        else None
    )
    ffn_gate_output_tap = (
        TensorAccessPattern(
            (output_tap_size,),
            ffn_gate_output_base,
            [1, 1, 1, intermediate_size],
            [0, 0, 0, 1],
        )
        if include_full_mlp
        else None
    )
    ffn_up_output_tap = (
        TensorAccessPattern(
            (output_tap_size,),
            ffn_up_output_base,
            [1, 1, 1, intermediate_size],
            [0, 0, 0, 1],
        )
        if include_full_mlp
        else None
    )
    ffn_gate_silu_output_tap = (
        TensorAccessPattern(
            (output_tap_size,),
            ffn_gate_silu_output_base,
            [1, 1, 1, intermediate_size],
            [0, 0, 0, 1],
        )
        if include_full_mlp
        else None
    )
    ffn_hidden_output_tap = (
        TensorAccessPattern(
            (output_tap_size,),
            ffn_hidden_output_base,
            [1, 1, 1, intermediate_size],
            [0, 0, 0, 1],
        )
        if include_full_mlp
        else None
    )
    ffn_out_output_tap = (
        TensorAccessPattern(
            (output_tap_size,),
            ffn_out_output_base,
            [1, 1, 1, hidden_size],
            [0, 0, 0, 1],
        )
        if include_full_mlp
        else None
    )
    layer_residual_output_tap = (
        TensorAccessPattern(
            (output_tap_size,),
            layer_residual_output_base,
            [1, 1, 1, hidden_size],
            [0, 0, 0, 1],
        )
        if include_full_mlp
        else None
    )
    final_layer_residual_output_tap = (
        TensorAccessPattern(
            (outputs_size,),
            0,
            [1, 1, 1, hidden_size],
            [0, 0, 0, 1],
        )
        if final_output_only
        else None
    )

    rt = Runtime()
    if final_output_only:

        def fill_single_layer_inputs(weights, angles, cache, tg):
            rt.fill(in_weight.prod(), weights, norm_weight_tap, task_group=tg)
            rt.fill(
                qk_norm_weight.prod(),
                weights,
                qk_norm_weight_tap,
                task_group=tg,
            )
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
                rt.fill(
                    o_weight_fifos[col].prod(),
                    weights,
                    o_weight_taps[col],
                    task_group=tg,
                )
            rt.fill(
                mlp_gate_up_weight_rows.prod(),
                weights,
                mlp_gate_up_weight_rows_tap,
                task_group=tg,
            )
            rt.fill(
                mlp_down_weight.prod(),
                weights,
                mlp_down_weight_tap,
                task_group=tg,
            )
            for col in range(num_columns):
                rt.fill(
                    k_cache_fifos[col].prod(),
                    cache,
                    cache_key_blocks_tap,
                    task_group=tg,
                )
                rt.fill(
                    v_cache_raw_fifos[col].prod(),
                    cache,
                    cache_value_blocks_tap,
                    task_group=tg,
                )

        def drain_single_layer_cache(cache, tg):
            for col in range(num_columns):
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

        def fill_layer_pair_inputs(layer_idx, weights, angles, cache, tg):
            rt.fill(
                in_weight.prod(),
                weights,
                layer_pair_weight_tap(layer_idx, 0, hidden_size),
                task_group=tg,
            )
            rt.fill(
                qk_norm_weight.prod(),
                weights,
                layer_pair_weight_tap(layer_idx, q_norm_weight_base, 2 * head_dim),
                task_group=tg,
            )
            rt.fill(rope_angles.prod(), angles, rope_angles_tap, task_group=tg)
            layer_q_weight_taps = layer_pair_weight_taps(
                layer_idx, q_size, q_weight_base
            )
            layer_k_weight_taps = layer_pair_weight_taps(
                layer_idx, kv_size, k_weight_base
            )
            layer_v_weight_taps = layer_pair_weight_taps(
                layer_idx, kv_size, v_weight_base
            )
            layer_o_weight_taps = layer_pair_weight_taps_for_k(
                layer_idx, hidden_size, q_size, o_weight_base
            )
            for col in range(num_columns):
                rt.fill(
                    q_weight_fifos[col].prod(),
                    weights,
                    layer_q_weight_taps[col],
                    task_group=tg,
                )
                rt.fill(
                    k_weight_fifos[col].prod(),
                    weights,
                    layer_k_weight_taps[col],
                    task_group=tg,
                )
                rt.fill(
                    v_weight_fifos[col].prod(),
                    weights,
                    layer_v_weight_taps[col],
                    task_group=tg,
                )
                rt.fill(
                    o_weight_fifos[col].prod(),
                    weights,
                    layer_o_weight_taps[col],
                    task_group=tg,
                )
            rt.fill(
                mlp_gate_up_weight_rows.prod(),
                weights,
                layer_pair_weight_tap(
                    layer_idx,
                    mlp_post_norm_weight_base,
                    hidden_size + 2 * intermediate_size * hidden_size,
                ),
                task_group=tg,
            )
            rt.fill(
                mlp_down_weight.prod(),
                weights,
                layer_pair_weight_tap(
                    layer_idx,
                    mlp_down_weight_base,
                    hidden_size * intermediate_size,
                ),
                task_group=tg,
            )
            for col in range(num_columns):
                rt.fill(
                    k_cache_fifos[col].prod(),
                    cache,
                    layer_pair_cache_key_blocks_tap(layer_idx),
                    task_group=tg,
                )
                rt.fill(
                    v_cache_raw_fifos[col].prod(),
                    cache,
                    layer_pair_cache_value_blocks_tap(layer_idx),
                    task_group=tg,
                )

        def drain_layer_pair_cache(layer_idx, cache, tg):
            for col in range(num_columns):
                rt.drain(
                    k_rope_fifos[col].cons(),
                    cache,
                    layer_pair_cache_key_tap(layer_idx),
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    v_fifos[col].cons(),
                    cache,
                    layer_pair_cache_value_tap(layer_idx),
                    wait=True,
                    task_group=tg,
                )

        if layer_iterations == 1:
            with rt.sequence(
                tensor_ty, weights_ty, angles_ty, outputs_ty, cache_ty
            ) as (
                hidden,
                weights,
                angles,
                outputs,
                cache,
            ):
                rt.start(*workers)
                tg = rt.task_group()
                rt.fill(runtime_hidden_in.prod(), hidden, hidden_tap, task_group=tg)
                fill_single_layer_inputs(weights, angles, cache, tg)
                drain_single_layer_cache(cache, tg)
                rt.drain(
                    layer_residual.cons(),
                    outputs,
                    final_layer_residual_output_tap,
                    wait=True,
                    task_group=tg,
                )
                rt.finish_task_group(tg)
        else:
            with rt.sequence(
                tensor_ty,
                weights_pair_ty,
                angles_ty,
                outputs_ty,
                cache_pair_ty,
            ) as (
                hidden,
                weights,
                angles,
                outputs,
                cache,
            ):
                rt.start(*workers)
                tg = rt.task_group()
                rt.fill(runtime_hidden_in.prod(), hidden, hidden_tap, task_group=tg)
                fill_layer_pair_inputs(0, weights, angles, cache, tg)
                fill_layer_pair_inputs(1, weights, angles, cache, tg)
                drain_layer_pair_cache(0, cache, tg)
                drain_layer_pair_cache(1, cache, tg)
                rt.drain(
                    final_layer_residual.cons(),
                    outputs,
                    final_layer_residual_output_tap,
                    wait=True,
                    task_group=tg,
                )
                rt.finish_task_group(tg)
        return Program(dev, rt).resolve_program(SequentialPlacer())

    with rt.sequence(tensor_ty, weights_ty, angles_ty, outputs_ty, cache_ty) as (
        hidden,
        weights,
        angles,
        outputs,
        cache,
    ):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(runtime_hidden_in.prod(), hidden, hidden_tap, task_group=tg)
        rt.fill(in_weight.prod(), weights, norm_weight_tap, task_group=tg)
        if include_full_mlp:
            rt.fill(
                qk_norm_weight.prod(),
                weights,
                qk_norm_weight_tap,
                task_group=tg,
            )
        else:
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
            if include_o_proj:
                rt.fill(
                    o_weight_fifos[col].prod(),
                    weights,
                    o_weight_taps[col],
                    task_group=tg,
                )
                if not include_full_mlp:
                    rt.fill(
                        residual_hidden_fifos[col].prod(),
                        hidden,
                        residual_hidden_taps[col],
                        task_group=tg,
                    )
        if include_full_mlp:
            rt.fill(
                mlp_gate_up_weight_rows.prod(),
                weights,
                mlp_gate_up_weight_rows_tap,
                task_group=tg,
            )
            rt.fill(
                mlp_down_weight.prod(),
                weights,
                mlp_down_weight_tap,
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
                if include_k_cache_debug:
                    rt.fill(
                        k_cache_debug_in_fifos[col].prod(),
                        cache,
                        cache_key_blocks_tap,
                        task_group=tg,
                    )
                if include_context:
                    rt.fill(
                        v_cache_raw_fifos[col].prod(),
                        cache,
                        cache_value_blocks_tap,
                        task_group=tg,
                    )
            if not include_full_mlp:
                rt.drain(
                    xnorm.cons(),
                    outputs,
                    xnorm_output_tap,
                    wait=True,
                    task_group=tg,
                )
            for col in range(num_columns):
                if not include_full_mlp:
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
                    qk_pair_debug_fifos[col].cons(),
                    outputs,
                    qk_pair_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
                if include_k_cache_debug:
                    rt.drain(
                        k_cache_debug_out_fifos[col].cons(),
                        outputs,
                        k_cache_stream_output_taps[col],
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
                if not include_full_mlp:
                    rt.drain(
                        attn_weight_fifos[col].cons(),
                        outputs,
                        attn_weight_output_taps[col],
                        wait=True,
                        task_group=tg,
                    )
                if include_context:
                    rt.drain(
                        v_context_debug_fifos[col].cons(),
                        outputs,
                        v_context_output_taps[col],
                        wait=True,
                        task_group=tg,
                    )
                    rt.drain(
                        attn_context_fifos[col].cons(),
                        outputs,
                        attn_context_output_taps[col],
                        wait=True,
                        task_group=tg,
                    )
                    if include_o_proj:
                        if not include_full_mlp:
                            rt.drain(
                                attn_context_flat_fifos[col].cons(),
                                outputs,
                                attn_context_flat_output_taps[col],
                                wait=True,
                                task_group=tg,
                            )
                            rt.drain(
                                o_proj_fifos[col].cons(),
                                outputs,
                                o_proj_output_taps[col],
                                wait=True,
                                task_group=tg,
                            )
                        rt.drain(
                            residual_out_fifos[col].cons(),
                            outputs,
                            residual_output_taps[col],
                            wait=True,
                            task_group=tg,
                        )
                        if include_full_mlp:
                            rt.drain(
                                ffn_hidden.cons(),
                                outputs,
                                ffn_hidden_output_tap,
                                wait=True,
                                task_group=tg,
                            )
                            rt.drain(
                                ffn_out.cons(),
                                outputs,
                                ffn_out_output_tap,
                                wait=True,
                                task_group=tg,
                            )
                            rt.drain(
                                layer_residual.cons(),
                                outputs,
                                layer_residual_output_tap,
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
    passthrough_kernel_object="passThrough.o",
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
        passthrough_kernel_object=passthrough_kernel_object,
        softmax_kernel_object=softmax_kernel_object,
    )


def qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax_context(
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
    passthrough_kernel_object="passThrough.o",
    softmax_kernel_object="softmax.o",
):
    """Single-token Qwen3 persistent stage through attention context."""
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
        include_context=True,
        attention_kernel_object=attention_kernel_object,
        passthrough_kernel_object=passthrough_kernel_object,
        softmax_kernel_object=softmax_kernel_object,
    )


def qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax_context_o_proj(
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
    passthrough_kernel_object="passThrough.o",
    softmax_kernel_object="softmax.o",
    o_gemv_kernel_object="mv_o_proj.o",
    add_kernel_object="add.o",
):
    """Single-token Qwen3 persistent stage through attention O projection and residual add."""
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
        include_context=True,
        include_o_proj=True,
        attention_kernel_object=attention_kernel_object,
        passthrough_kernel_object=passthrough_kernel_object,
        softmax_kernel_object=softmax_kernel_object,
        o_gemv_kernel_object=o_gemv_kernel_object,
        add_kernel_object=add_kernel_object,
    )


def qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax_context_o_proj_full_mlp(
    dev,
    hidden_size,
    q_size,
    kv_size,
    head_dim,
    max_seq_len,
    position,
    intermediate_size,
    num_columns,
    tile_size_input,
    tile_size_output,
    trace_size,
    func_prefix="",
    rms_kernel_object="rms_norm.o",
    gemv_kernel_object="mv.o",
    rope_kernel_object="rope.o",
    attention_kernel_object="qwen3_attention.o",
    passthrough_kernel_object="passThrough.o",
    softmax_kernel_object="softmax.o",
    o_gemv_kernel_object="mv_o_proj.o",
    add_kernel_object="add.o",
    mlp_gemv_kernel_object="mv_mlp.o",
    silu_kernel_object="silu.o",
    mul_kernel_object="mul.o",
    down_gemv_kernel_object="mv_down.o",
):
    """Single-token Qwen3 persistent stage through attention and full MLP."""
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
        intermediate_size=intermediate_size,
        func_prefix=func_prefix,
        rms_kernel_object=rms_kernel_object,
        gemv_kernel_object=gemv_kernel_object,
        rope_kernel_object=rope_kernel_object,
        include_scores_softmax=True,
        include_context=True,
        include_o_proj=True,
        include_full_mlp=True,
        attention_kernel_object=attention_kernel_object,
        passthrough_kernel_object=passthrough_kernel_object,
        softmax_kernel_object=softmax_kernel_object,
        o_gemv_kernel_object=o_gemv_kernel_object,
        add_kernel_object=add_kernel_object,
        mlp_gemv_kernel_object=mlp_gemv_kernel_object,
        silu_kernel_object=silu_kernel_object,
        mul_kernel_object=mul_kernel_object,
        down_gemv_kernel_object=down_gemv_kernel_object,
    )


def qwen3_persistent_single_layer_final_only(
    dev,
    hidden_size,
    q_size,
    kv_size,
    head_dim,
    max_seq_len,
    position,
    intermediate_size,
    num_columns,
    tile_size_input,
    tile_size_output,
    trace_size,
    func_prefix="",
    rms_kernel_object="rms_norm.o",
    gemv_kernel_object="mv.o",
    rope_kernel_object="rope.o",
    attention_kernel_object="qwen3_attention.o",
    passthrough_kernel_object="passThrough.o",
    softmax_kernel_object="softmax.o",
    o_gemv_kernel_object="mv_o_proj.o",
    add_kernel_object="add.o",
    mlp_gemv_kernel_object="mv_mlp.o",
    silu_kernel_object="silu.o",
    mul_kernel_object="mul.o",
    down_gemv_kernel_object="mv_down.o",
):
    """Single Qwen3 full layer that drains only the final hidden state."""
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
        intermediate_size=intermediate_size,
        func_prefix=func_prefix,
        rms_kernel_object=rms_kernel_object,
        gemv_kernel_object=gemv_kernel_object,
        rope_kernel_object=rope_kernel_object,
        include_scores_softmax=True,
        include_context=True,
        include_o_proj=True,
        include_full_mlp=True,
        attention_kernel_object=attention_kernel_object,
        passthrough_kernel_object=passthrough_kernel_object,
        softmax_kernel_object=softmax_kernel_object,
        o_gemv_kernel_object=o_gemv_kernel_object,
        add_kernel_object=add_kernel_object,
        mlp_gemv_kernel_object=mlp_gemv_kernel_object,
        silu_kernel_object=silu_kernel_object,
        mul_kernel_object=mul_kernel_object,
        down_gemv_kernel_object=down_gemv_kernel_object,
        final_output_only=True,
    )


def qwen3_persistent_two_layer_full_layer(
    dev,
    hidden_size,
    q_size,
    kv_size,
    head_dim,
    max_seq_len,
    position,
    intermediate_size,
    num_columns,
    tile_size_input,
    tile_size_output,
    trace_size,
    func_prefix="",
    rms_kernel_object="rms_norm.o",
    gemv_kernel_object="mv.o",
    rope_kernel_object="rope.o",
    attention_kernel_object="qwen3_attention.o",
    passthrough_kernel_object="passThrough.o",
    softmax_kernel_object="softmax.o",
    o_gemv_kernel_object="mv_o_proj.o",
    add_kernel_object="add.o",
    mlp_gemv_kernel_object="mv_mlp.o",
    silu_kernel_object="silu.o",
    mul_kernel_object="mul.o",
    down_gemv_kernel_object="mv_down.o",
):
    """Two sequential Qwen3 full layers using one persistent worker graph."""
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
        intermediate_size=intermediate_size,
        func_prefix=func_prefix,
        rms_kernel_object=rms_kernel_object,
        gemv_kernel_object=gemv_kernel_object,
        rope_kernel_object=rope_kernel_object,
        include_scores_softmax=True,
        include_context=True,
        include_o_proj=True,
        include_full_mlp=True,
        attention_kernel_object=attention_kernel_object,
        passthrough_kernel_object=passthrough_kernel_object,
        softmax_kernel_object=softmax_kernel_object,
        o_gemv_kernel_object=o_gemv_kernel_object,
        add_kernel_object=add_kernel_object,
        mlp_gemv_kernel_object=mlp_gemv_kernel_object,
        silu_kernel_object=silu_kernel_object,
        mul_kernel_object=mul_kernel_object,
        down_gemv_kernel_object=down_gemv_kernel_object,
        layer_iterations=2,
        final_output_only=True,
    )
