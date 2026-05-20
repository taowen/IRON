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


def _qwen3_persistent_n_layer_final_only_impl(
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
    mlp_down_columns=1,
):
    """Single-token Qwen3 input RMSNorm, QKV, Q/K norm, RoPE, KV write, and optional softmax."""
    dtype = bfloat16
    q_heads = q_size // head_dim
    kv_heads = kv_size // head_dim
    cache_block_seq = 64
    if layer_iterations < 1:
        raise ValueError("layer_iterations must be positive")
    score_size = q_heads * max_seq_len
    qk_pair_debug_size = kv_heads * 3 * head_dim
    k_cache_debug_size = 0
    v_cache_debug_size = kv_heads * max_seq_len * head_dim
    context_size = q_size
    context_flat_size = q_size
    o_proj_size = hidden_size
    residual_size = hidden_size
    mlp_xnorm_size = hidden_size
    mlp_ffn_size = intermediate_size
    mlp_ffn_out_size = hidden_size
    mlp_layer_residual_size = hidden_size
    mlp_weights_size = hidden_size + 3 * intermediate_size * hidden_size
    weights_size = (
        hidden_size
        + q_size * hidden_size
        + 2 * kv_size * hidden_size
        + 2 * head_dim
        + hidden_size * q_size
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
    outputs_size = hidden_size
    cache_size = 2 * kv_size * max_seq_len
    tensor_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    weights_ty = np.ndarray[(weights_size,), np.dtype[dtype]]
    weights_chunk_ty = np.ndarray[(layer_iterations * weights_size,), np.dtype[dtype]]
    angles_ty = np.ndarray[(head_dim,), np.dtype[dtype]]
    outputs_ty = np.ndarray[(outputs_size,), np.dtype[dtype]]
    cache_ty = np.ndarray[(cache_size,), np.dtype[dtype]]
    cache_chunk_ty = np.ndarray[(layer_iterations * cache_size,), np.dtype[dtype]]
    tile_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    hidden_weight_ty = np.ndarray[(hidden_size,), np.dtype[dtype]]
    head_ty = np.ndarray[(head_dim,), np.dtype[dtype]]
    qk_norm_weight_ty = np.ndarray[(2 * head_dim,), np.dtype[dtype]]
    ffn_ty = np.ndarray[(intermediate_size,), np.dtype[dtype]]
    ffn_shard_ty = np.ndarray[(intermediate_size // 2,), np.dtype[dtype]]
    qk_pair_ty = np.ndarray[(3 * head_dim,), np.dtype[dtype]]
    score_ty = np.ndarray[(max_seq_len,), np.dtype[dtype]]
    k_cache_block_ty = np.ndarray[(cache_block_seq, head_dim), np.dtype[dtype]]
    v_cache_block_ty = np.ndarray[(cache_block_seq, head_dim), np.dtype[dtype]]
    gemv_a_ty = np.ndarray[(tile_size_input, hidden_size), np.dtype[dtype]]
    context_flat_ty = np.ndarray[(q_size,), np.dtype[dtype]]
    o_gemv_a_ty = np.ndarray[(tile_size_input, q_size), np.dtype[dtype]]
    mlp_gemv_a_ty = np.ndarray[(tile_size_input, hidden_size), np.dtype[dtype]]
    down_gemv_a_ty = np.ndarray[(tile_size_input, intermediate_size), np.dtype[dtype]]
    hidden_tile_ty = np.ndarray[(tile_size_output,), np.dtype[dtype]]
    if q_size % head_dim != 0 or kv_size % head_dim != 0:
        raise ValueError("Q/KV sizes must be divisible by head_dim")
    if q_size % num_columns != 0 or kv_size % num_columns != 0:
        raise ValueError("Q/KV output sizes must be divisible by num_columns")
    if tile_size_output != head_dim:
        raise ValueError("tile_size_output must equal head_dim for rope-cache stage")
    if tile_size_output % tile_size_input != 0:
        raise ValueError("tile_size_output must be a multiple of tile_size_input")
    if not 0 <= position < max_seq_len:
        raise ValueError("position must be inside max_seq_len")
    if num_columns != 1:
        raise ValueError(
            "n-layer final-only attention path is currently NPU2 single-column only"
        )
    if mlp_down_columns < 1:
        raise ValueError("mlp_down_columns must be positive")
    if mlp_down_columns not in {1, 2, 4}:
        raise ValueError("mlp_down_columns must be one of 1, 2, or 4")
    if mlp_down_columns > 2 and layer_iterations != 1:
        raise ValueError(
            "n-layer MLP down column scaling above 2 columns is currently "
            "limited to one layer"
        )
    mlp_gate_up_columns = 2 if mlp_down_columns == 2 else 1
    if hidden_size % (tile_size_output * mlp_down_columns) != 0:
        raise ValueError(
            "hidden_size must be divisible by tile_size_output * mlp_down_columns"
        )
    if intermediate_size % mlp_gate_up_columns != 0:
        raise ValueError(
            "intermediate_size must be divisible by the MLP gate/up column count"
        )
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be a multiple of kv_heads for GQA")
    if q_heads // kv_heads != 2:
        raise ValueError("scores+softmax checkpoint expects Qwen3-0.6B GQA repeat=2")
    if max_seq_len % cache_block_seq != 0:
        raise ValueError("max_seq_len must be divisible by cache_block_seq")
    if hidden_size != 1024:
        raise ValueError("n-layer final-only path expects hidden_size=1024")
    if intermediate_size != 3072:
        raise ValueError("n-layer final-only path expects intermediate_size=3072")
    full_cache_blocks = max_seq_len // cache_block_seq
    active_cache_blocks = (position + cache_block_seq) // cache_block_seq
    cache_blocks_to_process = active_cache_blocks
    runtime_hidden_in = ObjectFifo(tile_ty, name="qwen3_rc_hidden_in", depth=2)
    in_hidden = runtime_hidden_in
    hidden_feedback = None
    final_layer_residual = None
    if layer_iterations > 1:
        in_hidden = ObjectFifo(tile_ty, name="qwen3_chunk_hidden", depth=2)
        hidden_feedback = ObjectFifo(
            tile_ty, name="qwen3_chunk_hidden_feedback", depth=2
        )
        final_layer_residual = ObjectFifo(
            tile_ty, name="qwen3_chunk_final_residual", depth=2
        )
    in_weight = ObjectFifo(hidden_weight_ty, name="qwen3_rc_input_norm_weight", depth=2)
    normed = ObjectFifo(tile_ty, name="qwen3_rc_input_norm_unweighted", depth=2)
    xnorm = ObjectFifo(tile_ty, name="qwen3_rc_xnorm", depth=2)
    q_norm_weight = ObjectFifo(head_ty, name="qwen3_rc_q_norm_weight", depth=2)
    k_norm_weight = ObjectFifo(head_ty, name="qwen3_rc_k_norm_weight", depth=2)
    qk_norm_weight = ObjectFifo(
        qk_norm_weight_ty, name="qwen3_rc_qk_norm_weight", depth=2
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
    q_norm_fifos = []
    k_norm_fifos = []
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
    mlp_post_norm_weight = None
    mlp_gate_up_shard_weight_fifos = []
    mlp_down_weight_fifos = []
    mlp_xnorm = None
    ffn_gate = None
    ffn_up = None
    ffn_hidden = None
    ffn_hidden_shard_fifos = []
    ffn_hidden_joined = None
    ffn_out = None
    ffn_out_fifos = []
    layer_residual = None
    layer_residual_fifos = []
    layer_residual_joined = None
    qk_rope_meta = ObjectFifo(qk_pair_ty, name="qwen3_rc_qk_rope_metadata", depth=2)
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
    attn_score_debug_fifos = [
        ObjectFifo(score_ty, name=f"qwen3_rc_attn_scores_{col}", depth=2)
        for col in range(num_columns)
    ]
    attn_score_softmax_fifos = [
        ObjectFifo(score_ty, name=f"qwen3_rc_attn_scores_for_softmax_{col}", depth=2)
        for col in range(num_columns)
    ]
    attn_weight_fifos = [
        ObjectFifo(score_ty, name=f"qwen3_rc_attn_weights_{col}", depth=2)
        for col in range(num_columns)
    ]
    v_cache_raw_fifos = [
        ObjectFifo(v_cache_block_ty, name=f"qwen3_rc_v_cache_{col}", depth=1)
        for col in range(num_columns)
    ]
    v_context_fifos = [
        ObjectFifo(v_cache_block_ty, name=f"qwen3_rc_v_context_block_{col}", depth=1)
        for col in range(num_columns)
    ]
    v_context_debug_fifos = [
        ObjectFifo(v_cache_block_ty, name=f"qwen3_rc_v_context_debug_{col}", depth=1)
        for col in range(num_columns)
    ]
    attn_context_fifos = [
        ObjectFifo(head_ty, name=f"qwen3_rc_attn_context_{col}", depth=2)
        for col in range(num_columns)
    ]
    attn_context_flat_fifos = [
        ObjectFifo(context_flat_ty, name=f"qwen3_rc_attn_context_flat_{col}", depth=1)
        for col in range(num_columns)
    ]
    o_weight_fifos = [
        ObjectFifo(o_gemv_a_ty, name=f"qwen3_rc_o_weight_{col}", depth=2)
        for col in range(num_columns)
    ]
    o_proj_ty = tile_ty
    o_proj_depth = 1
    o_proj_fifos = [
        ObjectFifo(o_proj_ty, name=f"qwen3_rc_attn_o_proj_{col}", depth=o_proj_depth)
        for col in range(num_columns)
    ]
    residual_ty = tile_ty
    residual_out_fifos = [
        ObjectFifo(residual_ty, name=f"qwen3_rc_attn_residual_{col}", depth=2)
        for col in range(num_columns)
    ]
    if mlp_gate_up_columns == 1:
        mlp_gate_up_weight_rows = ObjectFifo(
            hidden_weight_ty, name="qwen3_full_layer_mlp_gate_up_weight_rows", depth=4
        )
    else:
        mlp_post_norm_weight = ObjectFifo(
            hidden_weight_ty, name="qwen3_full_layer_mlp_post_norm_weight", depth=2
        )
        mlp_gate_up_shard_weight_fifos = [
            ObjectFifo(
                hidden_weight_ty,
                name=f"qwen3_full_layer_mlp_gate_up_weight_{col}",
                depth=4,
            )
            for col in range(mlp_gate_up_columns)
        ]
    mlp_down_weight_fifos = [
        ObjectFifo(
            down_gemv_a_ty,
            name=f"qwen3_full_layer_mlp_down_weight_{col}",
            depth=1,
        )
        for col in range(mlp_down_columns)
    ]
    mlp_xnorm = ObjectFifo(tile_ty, name="qwen3_full_layer_mlp_xnorm", depth=2)
    if mlp_gate_up_columns == 1:
        ffn_gate = ObjectFifo(ffn_ty, name="qwen3_full_layer_ffn_gate", depth=2)
        ffn_up = ObjectFifo(ffn_ty, name="qwen3_full_layer_ffn_up", depth=2)
        ffn_hidden = ObjectFifo(ffn_ty, name="qwen3_full_layer_ffn_hidden", depth=2)
    else:
        ffn_hidden_shard_fifos = [
            ObjectFifo(
                ffn_shard_ty,
                name=f"qwen3_full_layer_ffn_hidden_shard_{col}",
                depth=2,
            )
            for col in range(mlp_gate_up_columns)
        ]
        ffn_hidden_joined = ObjectFifo(
            ffn_ty, name="qwen3_full_layer_ffn_hidden_joined", depth=2
        )
        ffn_hidden = ffn_hidden_joined
    ffn_out_ty = hidden_tile_ty if mlp_down_columns != 1 else tile_ty
    ffn_out_fifos = [
        ObjectFifo(ffn_out_ty, name=f"qwen3_full_layer_ffn_out_{col}", depth=2)
        for col in range(mlp_down_columns)
    ]
    layer_residual_ty = hidden_tile_ty if mlp_down_columns != 1 else tile_ty
    layer_residual_fifos = [
        ObjectFifo(
            layer_residual_ty,
            name=f"qwen3_full_layer_residual_{col}",
            depth=2,
        )
        for col in range(mlp_down_columns)
    ]
    ffn_out = ffn_out_fifos[0]
    layer_residual = layer_residual_fifos[0]
    if mlp_down_columns != 1 and layer_iterations > 1:
        layer_residual_joined = ObjectFifo(
            tile_ty, name="qwen3_full_layer_residual_joined", depth=2
        )
        layer_residual = layer_residual_joined
    rms_norm_kernel = Kernel(
        f"{func_prefix}rms_norm_bf16_vector",
        f"{func_prefix}{rms_kernel_object}",
        [tile_ty, tile_ty, np.int32],
    )
    mul_kernel = Kernel(
        f"{func_prefix}eltwise_mul_bf16_vector",
        f"{func_prefix}{mul_kernel_object}",
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
    add_full_slice_to_tile = None
    copy_tile_to_full = None
    copy_ffn_shard_to_full = None
    hidden_copy = None
    norm_rope_with_weight_offset = None
    pack_qk_rope_metadata = None
    norm_rope_with_metadata = None
    mlp_weighted_rms_norm = None
    mlp_matvec = None
    mlp_matvec4_rows = None
    mlp_matvec4_rows_shard = None
    silu = None
    eltwise_mul = None
    silu_mul = None
    silu_mul_shard = None
    down_matvec = None
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
    pack_context_head = Kernel(
        f"{func_prefix}qwen3_pack_context_head_bf16",
        f"{func_prefix}{attention_kernel_object}",
        [head_ty, context_flat_ty, np.int32],
    )
    o_matvec_out_ty = tile_ty
    o_matvec = Kernel(
        f"{func_prefix}qwen3_o_proj_matvec_vectorized_bf16_bf16",
        f"{func_prefix}{o_gemv_kernel_object}",
        [np.int32, np.int32, o_gemv_a_ty, context_flat_ty, o_matvec_out_ty],
    )
    add_hidden_tile_to_full = Kernel(
        f"{func_prefix}qwen3_add_hidden_tile_to_full_bf16",
        f"{func_prefix}{attention_kernel_object}",
        [head_ty, tile_ty, tile_ty, np.int32, np.int32],
    )
    norm_rope_with_weight_offset = Kernel(
        f"{func_prefix}qwen3_norm_rope_with_weight_offset_bf16",
        f"{func_prefix}{attention_kernel_object}",
        [head_ty, qk_norm_weight_ty, head_ty, head_ty, head_ty, np.int32, np.int32],
    )
    pack_qk_rope_metadata = Kernel(
        f"{func_prefix}qwen3_pack_qk_rope_metadata_bf16",
        f"{func_prefix}{attention_kernel_object}",
        [qk_norm_weight_ty, head_ty, qk_pair_ty, np.int32],
    )
    norm_rope_with_metadata = Kernel(
        f"{func_prefix}qwen3_norm_rope_with_metadata_bf16",
        f"{func_prefix}{attention_kernel_object}",
        [head_ty, qk_pair_ty, head_ty, head_ty, np.int32, np.int32],
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
    mlp_matvec4_rows_shard = Kernel(
        f"{func_prefix}qwen3_mlp_matvec4_rows_shard_bf16",
        f"{func_prefix}{attention_kernel_object}",
        [
            np.int32,
            np.int32,
            hidden_weight_ty,
            hidden_weight_ty,
            hidden_weight_ty,
            hidden_weight_ty,
            tile_ty,
            ffn_shard_ty,
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
    silu_mul_shard = Kernel(
        f"{func_prefix}qwen3_silu_mul_shard_bf16",
        f"{func_prefix}{attention_kernel_object}",
        [ffn_shard_ty, ffn_shard_ty, ffn_shard_ty, np.int32],
    )
    down_output_ty = hidden_tile_ty if mlp_down_columns != 1 else tile_ty
    down_matvec = Kernel(
        f"{func_prefix}qwen3_down_proj_matvec_vectorized_bf16_bf16",
        f"{func_prefix}{down_gemv_kernel_object}",
        [np.int32, np.int32, down_gemv_a_ty, ffn_ty, down_output_ty],
    )
    add_full_slice = Kernel(
        f"{func_prefix}qwen3_add_full_slice_bf16",
        f"{func_prefix}{attention_kernel_object}",
        [tile_ty, tile_ty, tile_ty, np.int32, np.int32],
    )
    add_full_slice_to_tile = Kernel(
        f"{func_prefix}qwen3_add_full_slice_to_tile_bf16",
        f"{func_prefix}{attention_kernel_object}",
        [tile_ty, hidden_tile_ty, hidden_tile_ty, np.int32, np.int32],
    )
    copy_tile_to_full = Kernel(
        f"{func_prefix}qwen3_copy_tile_to_full_bf16",
        f"{func_prefix}{attention_kernel_object}",
        [hidden_tile_ty, tile_ty, np.int32, np.int32],
    )
    copy_ffn_shard_to_full = Kernel(
        f"{func_prefix}qwen3_copy_ffn_shard_to_full_bf16",
        f"{func_prefix}{attention_kernel_object}",
        [ffn_shard_ty, ffn_ty, np.int32, np.int32],
    )
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

    def qk_rope_metadata_worker(norm_weight_fifo, angles_fifo, meta_fifo, pack_kernel):
        angles = angles_fifo.acquire(1)
        for _ in range_(layer_iterations):
            norm_weights = norm_weight_fifo.acquire(1)
            meta = meta_fifo.acquire(1)
            pack_kernel(norm_weights, angles, meta, head_dim)
            meta_fifo.release(1)
            norm_weight_fifo.release(1)
        angles_fifo.release(1)

    def q_norm_rope_metadata_worker(raw_fifo, meta_fifo, rope_fifo, norm_rope_kernel):
        for _ in range_(layer_iterations):
            meta = meta_fifo.acquire(1)
            for _ in range_(q_heads // num_columns):
                raw = raw_fifo.acquire(1)
                rope_out = rope_fifo.acquire(1)
                norm_rope_kernel(raw, meta, raw, rope_out, 0, head_dim)
                rope_fifo.release(1)
                raw_fifo.release(1)
            meta_fifo.release(1)

    def k_norm_rope_metadata_worker(raw_fifo, meta_fifo, rope_fifo, norm_rope_kernel):
        for _ in range_(layer_iterations):
            meta = meta_fifo.acquire(1)
            for _ in range_(kv_heads // num_columns):
                raw = raw_fifo.acquire(1)
                rope_out = rope_fifo.acquire(1)
                norm_rope_kernel(raw, meta, raw, rope_out, head_dim, head_dim)
                rope_fifo.release(1)
                raw_fifo.release(1)
            meta_fifo.release(1)

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

    def attention_score_final_worker(
        pair_fifo, k_cache_fifo, score_softmax_fifo, score_kernel
    ):
        for _ in range_(layer_iterations):
            for _ in range_(kv_heads // num_columns):
                pair = pair_fifo.acquire(1)
                score_softmax_pair = score_softmax_fifo.acquire(2)
                score_softmax0 = score_softmax_pair[0]
                score_softmax1 = score_softmax_pair[1]
                for block_idx in range_(cache_blocks_to_process):
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

    def v_context_merge_final_worker(
        current_v_fifo, v_cache_fifo, v_context_fifo, merge_kernel
    ):
        for _ in range_(layer_iterations):
            for _ in range_(kv_heads // num_columns):
                current_v = current_v_fifo.acquire(1)
                for block_idx in range_(cache_blocks_to_process):
                    block_i32 = index.casts(T.i32(), block_idx)
                    row_base = block_i32 * cache_block_seq
                    cached_v = v_cache_fifo.acquire(1)
                    merged_v = v_context_fifo.acquire(1)
                    merge_kernel(cached_v, current_v, merged_v, position, row_base)
                    v_cache_fifo.release(1)
                    v_context_fifo.release(1)
                current_v_fifo.release(1)

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
                for block_idx in range_(cache_blocks_to_process):
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

    def mlp_postnorm_worker(
        residual_fifo, post_norm_weight_fifo, xnorm_fifo, norm_kernel
    ):
        for _ in range_(layer_iterations):
            residual = residual_fifo.acquire(1)
            post_norm = post_norm_weight_fifo.acquire(1)
            xnorm_out = xnorm_fifo.acquire(1)
            norm_kernel(residual, post_norm, xnorm_out, hidden_size)
            xnorm_fifo.release(1)
            post_norm_weight_fifo.release(1)
            residual_fifo.release(1)

    def make_mlp_gate_up_silu_shard_worker():
        def mlp_gate_up_silu_shard_worker(
            gate_up_weight_fifo,
            xnorm_fifo,
            hidden_fifo,
            matvec4_rows_kernel,
            silu_mul_kernel,
            gate_buffer,
            up_buffer,
        ):
            for _ in range_(layer_iterations):
                xnorm_in = xnorm_fifo.acquire(1)
                hidden_out = hidden_fifo.acquire(1)
                for j_idx in range_(
                    intermediate_size // tile_size_input // mlp_gate_up_columns
                ):
                    j_i32 = index.casts(T.i32(), j_idx)
                    output_row_offset = j_i32 * tile_size_input
                    rows = gate_up_weight_fifo.acquire(tile_size_input)
                    matvec4_rows_kernel(
                        tile_size_input,
                        output_row_offset,
                        rows[0],
                        rows[1],
                        rows[2],
                        rows[3],
                        xnorm_in,
                        gate_buffer,
                    )
                    gate_up_weight_fifo.release(tile_size_input)
                for j_idx in range_(
                    intermediate_size // tile_size_input // mlp_gate_up_columns
                ):
                    j_i32 = index.casts(T.i32(), j_idx)
                    output_row_offset = j_i32 * tile_size_input
                    rows = gate_up_weight_fifo.acquire(tile_size_input)
                    matvec4_rows_kernel(
                        tile_size_input,
                        output_row_offset,
                        rows[0],
                        rows[1],
                        rows[2],
                        rows[3],
                        xnorm_in,
                        up_buffer,
                    )
                    gate_up_weight_fifo.release(tile_size_input)
                silu_mul_kernel(
                    gate_buffer,
                    up_buffer,
                    hidden_out,
                    intermediate_size // mlp_gate_up_columns,
                )
                hidden_fifo.release(1)
                xnorm_fifo.release(1)

        return mlp_gate_up_silu_shard_worker

    def mlp_silu_mul_worker(gate_fifo, up_fifo, hidden_fifo, silu_mul_kernel):
        for _ in range_(layer_iterations):
            gate = gate_fifo.acquire(1)
            up = up_fifo.acquire(1)
            hidden_out = hidden_fifo.acquire(1)
            silu_mul_kernel(gate, up, hidden_out, intermediate_size)
            hidden_fifo.release(1)
            up_fifo.release(1)
            gate_fifo.release(1)

    def mlp_ffn_hidden_join2_worker(left_fifo, right_fifo, out_fifo, copy_shard_kernel):
        for _ in range_(layer_iterations):
            left = left_fifo.acquire(1)
            right = right_fifo.acquire(1)
            out = out_fifo.acquire(1)
            copy_shard_kernel(left, out, 0, intermediate_size // 2)
            copy_shard_kernel(
                right,
                out,
                intermediate_size // 2,
                intermediate_size // 2,
            )
            out_fifo.release(1)
            right_fifo.release(1)
            left_fifo.release(1)

    def mlp_down_worker(hidden_fifo, down_weight_fifo, ffn_out_fifo, matvec_kernel):
        for _ in range_(layer_iterations):
            hidden = hidden_fifo.acquire(1)
            ffn = ffn_out_fifo.acquire(1)
            for tile_idx in range_(hidden_size // tile_size_output):
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

    def make_mlp_down_tile_worker(column_index):
        def mlp_down_tile_worker(
            hidden_fifo, down_weight_fifo, ffn_out_fifo, matvec_kernel
        ):
            for _ in range_(layer_iterations):
                hidden = hidden_fifo.acquire(1)
                for _ in range_(hidden_size // tile_size_output // mlp_down_columns):
                    ffn = ffn_out_fifo.acquire(1)
                    for j_idx in range_(tile_size_output // tile_size_input):
                        j_i32 = index.casts(T.i32(), j_idx)
                        output_row_offset = j_i32 * tile_size_input
                        w = down_weight_fifo.acquire(1)
                        matvec_kernel(
                            tile_size_input, output_row_offset, w, hidden, ffn
                        )
                        down_weight_fifo.release(1)
                    ffn_out_fifo.release(1)
                hidden_fifo.release(1)

        return mlp_down_tile_worker

    def mlp_layer_residual_worker(
        ffn_out_fifo, residual_fifo, layer_residual_fifo, add_kernel
    ):
        for _ in range_(layer_iterations):
            ffn = ffn_out_fifo.acquire(1)
            residual = residual_fifo.acquire(1)
            out = layer_residual_fifo.acquire(1)
            for tile_idx in range_(hidden_size // tile_size_output):
                tile_i32 = index.casts(T.i32(), tile_idx)
                row_offset = tile_i32 * tile_size_output
                add_kernel(residual, ffn, out, row_offset, tile_size_output)
            layer_residual_fifo.release(1)
            residual_fifo.release(1)
            ffn_out_fifo.release(1)

    def make_mlp_layer_residual_tile_worker(column_index):
        def mlp_layer_residual_tile_worker(
            ffn_out_fifo, residual_fifo, layer_residual_fifo, add_kernel
        ):
            for _ in range_(layer_iterations):
                residual = residual_fifo.acquire(1)
                for tile_idx in range_(
                    hidden_size // tile_size_output // mlp_down_columns
                ):
                    tile_i32 = index.casts(T.i32(), tile_idx)
                    row_offset = (
                        column_index * (hidden_size // mlp_down_columns)
                        + tile_i32 * tile_size_output
                    )
                    ffn = ffn_out_fifo.acquire(1)
                    out = layer_residual_fifo.acquire(1)
                    add_kernel(residual, ffn, out, row_offset, tile_size_output)
                    layer_residual_fifo.release(1)
                    ffn_out_fifo.release(1)
                residual_fifo.release(1)

        return mlp_layer_residual_tile_worker

    def mlp_layer_residual_join2_worker(
        left_fifo, right_fifo, out_fifo, copy_tile_kernel
    ):
        for _ in range_(layer_iterations):
            out = out_fifo.acquire(1)
            for tile_idx in range_(hidden_size // tile_size_output // 2):
                tile_i32 = index.casts(T.i32(), tile_idx)
                row_offset = tile_i32 * tile_size_output
                left = left_fifo.acquire(1)
                right = right_fifo.acquire(1)
                copy_tile_kernel(left, out, row_offset, tile_size_output)
                copy_tile_kernel(
                    right,
                    out,
                    hidden_size // 2 + row_offset,
                    tile_size_output,
                )
                right_fifo.release(1)
                left_fifo.release(1)
            out_fifo.release(1)

    def chunk_initial_hidden_worker(initial_fifo, feedback_fifo, out_fifo, copy):
        initial = initial_fifo.acquire(1)
        out = out_fifo.acquire(1)
        copy(initial, out, hidden_size)
        out_fifo.release(1)
        initial_fifo.release(1)
        for _ in range_(layer_iterations - 1):
            feedback = feedback_fifo.acquire(1)
            out = out_fifo.acquire(1)
            copy(feedback, out, hidden_size)
            out_fifo.release(1)
            feedback_fifo.release(1)

    def chunk_residual_router_worker(
        layer_residual_fifo, feedback_fifo, final_fifo, copy
    ):
        for _ in range_(layer_iterations - 1):
            residual = layer_residual_fifo.acquire(1)
            feedback = feedback_fifo.acquire(1)
            copy(residual, feedback, hidden_size)
            feedback_fifo.release(1)
            layer_residual_fifo.release(1)
        final_residual = layer_residual_fifo.acquire(1)
        final = final_fifo.acquire(1)
        copy(final_residual, final, hidden_size)
        final_fifo.release(1)
        layer_residual_fifo.release(1)

    workers = []
    if layer_iterations > 1:
        workers.append(
            Worker(
                chunk_initial_hidden_worker,
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
            Worker(rmsnorm_worker, [in_hidden.cons(), normed.prod(), rms_norm_kernel]),
            Worker(
                weight_worker,
                [normed.cons(), in_weight.cons(), xnorm.prod(), mul_kernel],
            ),
        ]
    )
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
                [v_weight_fifos[col].cons(), xnorm.cons(), v_fifos[col].prod(), matvec],
            ),
        ]
        workers.extend(q_workers + kv_workers)
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
        workers.extend(score_workers)
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
        context_workers.append(
            Worker(
                attention_context_o_proj_final_worker,
                [
                    attn_weight_fifos[col].cons(),
                    v_context_fifos[col].cons(),
                    attn_context_flat_fifos[col].prod(),
                    attention_context,
                    pack_context_head,
                    Buffer(type=head_ty, name=f"qwen3_final_context0_buf_{col}"),
                    Buffer(type=head_ty, name=f"qwen3_final_context1_buf_{col}"),
                ],
            )
        )
        workers.extend(context_workers)
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
    if mlp_gate_up_columns == 1:
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
                    [ffn_gate.cons(), ffn_up.cons(), ffn_hidden.prod(), silu_mul],
                ),
            ]
        )
    else:
        workers.append(
            Worker(
                mlp_postnorm_worker,
                [
                    residual_out_fifos[0].cons(),
                    mlp_post_norm_weight.cons(),
                    mlp_xnorm.prod(),
                    mlp_weighted_rms_norm,
                ],
            )
        )
        for col in range(mlp_gate_up_columns):
            workers.append(
                Worker(
                    make_mlp_gate_up_silu_shard_worker(),
                    [
                        mlp_gate_up_shard_weight_fifos[col].cons(),
                        mlp_xnorm.cons(),
                        ffn_hidden_shard_fifos[col].prod(),
                        mlp_matvec4_rows_shard,
                        silu_mul_shard,
                        Buffer(
                            type=ffn_shard_ty,
                            name=f"qwen3_full_layer_ffn_gate_buf_{col}",
                        ),
                        Buffer(
                            type=ffn_shard_ty,
                            name=f"qwen3_full_layer_ffn_up_buf_{col}",
                        ),
                    ],
                )
            )
        workers.append(
            Worker(
                mlp_ffn_hidden_join2_worker,
                [
                    ffn_hidden_shard_fifos[0].cons(),
                    ffn_hidden_shard_fifos[1].cons(),
                    ffn_hidden.prod(),
                    copy_ffn_shard_to_full,
                ],
            )
        )
    if mlp_down_columns == 1:
        workers.extend(
            [
                Worker(
                    mlp_down_worker,
                    [
                        ffn_hidden.cons(),
                        mlp_down_weight_fifos[0].cons(),
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
    else:
        for col in range(mlp_down_columns):
            workers.extend(
                [
                    Worker(
                        make_mlp_down_tile_worker(col),
                        [
                            ffn_hidden.cons(),
                            mlp_down_weight_fifos[col].cons(),
                            ffn_out_fifos[col].prod(),
                            down_matvec,
                        ],
                    ),
                    Worker(
                        make_mlp_layer_residual_tile_worker(col),
                        [
                            ffn_out_fifos[col].cons(),
                            residual_out_fifos[0].cons(),
                            layer_residual_fifos[col].prod(),
                            add_full_slice_to_tile,
                        ],
                    ),
                ]
            )
        if layer_iterations > 1:
            workers.append(
                Worker(
                    mlp_layer_residual_join2_worker,
                    [
                        layer_residual_fifos[0].cons(),
                        layer_residual_fifos[1].cons(),
                        layer_residual.prod(),
                        copy_tile_to_full,
                    ],
                )
            )
    if layer_iterations > 1:
        workers.append(
            Worker(
                chunk_residual_router_worker,
                [
                    layer_residual.cons(),
                    hidden_feedback.prod(),
                    final_layer_residual.prod(),
                    hidden_copy,
                ],
            )
        )
    hidden_tap = TensorAccessPattern(
        (1, hidden_size), 0, [1, 1, 1, hidden_size], [0, 0, 0, 1]
    )
    norm_weight_tap = TensorAccessPattern(
        (weights_size,), 0, [1, 1, 1, hidden_size], [0, 0, 0, 1]
    )
    q_weight_base = hidden_size
    k_weight_base = q_weight_base + q_size * hidden_size
    v_weight_base = k_weight_base + kv_size * hidden_size
    q_norm_weight_base = v_weight_base + kv_size * hidden_size
    k_norm_weight_base = q_norm_weight_base + head_dim
    o_weight_base = k_norm_weight_base + head_dim

    segment_input_norm_size = hidden_size
    segment_q_weight_size = q_size * hidden_size
    segment_k_weight_size = kv_size * hidden_size
    segment_v_weight_size = kv_size * hidden_size
    segment_qk_norm_size = 2 * head_dim
    segment_o_weight_size = hidden_size * q_size
    segment_mlp_gate_up_size = hidden_size + 2 * intermediate_size * hidden_size
    segment_mlp_down_size = hidden_size * intermediate_size
    segment_mlp_post_norm_size = hidden_size
    segment_mlp_gate_up_shard_size = (
        2 * (intermediate_size // mlp_gate_up_columns) * hidden_size
    )
    segment_mlp_down_shard_size = (hidden_size // mlp_down_columns) * intermediate_size
    segment_input_norm_base = 0
    segment_q_weight_base = segment_input_norm_base + (
        layer_iterations * segment_input_norm_size
    )
    segment_k_weight_base = segment_q_weight_base + (
        layer_iterations * segment_q_weight_size
    )
    segment_v_weight_base = segment_k_weight_base + (
        layer_iterations * segment_k_weight_size
    )
    segment_qk_norm_base = segment_v_weight_base + (
        layer_iterations * segment_v_weight_size
    )
    segment_o_weight_base = segment_qk_norm_base + (
        layer_iterations * segment_qk_norm_size
    )
    segment_mlp_base = segment_o_weight_base + (
        layer_iterations * segment_o_weight_size
    )
    if mlp_gate_up_columns == 1:
        segment_mlp_gate_up_base = segment_mlp_base
        segment_mlp_down_base = segment_mlp_gate_up_base + (
            layer_iterations * segment_mlp_gate_up_size
        )
        segment_mlp_end = segment_mlp_down_base + (
            layer_iterations * segment_mlp_down_size
        )
    else:
        segment_mlp_post_norm_base = segment_mlp_base
        segment_mlp_gate_up_base = segment_mlp_post_norm_base + (
            layer_iterations * segment_mlp_post_norm_size
        )
        segment_mlp_down_base = segment_mlp_gate_up_base + (
            mlp_gate_up_columns * layer_iterations * segment_mlp_gate_up_shard_size
        )
        segment_mlp_end = segment_mlp_down_base + (
            mlp_down_columns * layer_iterations * segment_mlp_down_shard_size
        )
    if segment_mlp_end != (layer_iterations * weights_size):
        raise ValueError("segment-major n-layer weight layout size mismatch")

    def weight_taps_for_k(total_rows, k_size, base_offset):
        return [
            TensorAccessPattern(
                (weights_size,),
                base_offset + col * (total_rows // num_columns) * k_size,
                [1, 1, 1, total_rows // num_columns * k_size],
                [0, 0, 0, 1],
            )
            for col in range(num_columns)
        ]

    def weight_taps(total_rows, base_offset):
        return weight_taps_for_k(total_rows, hidden_size, base_offset)

    rope_angles_tap = TensorAccessPattern(
        (1, head_dim), 0, [1, 1, 1, head_dim], [0, 0, 0, 1]
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
    output_tap_size = stage_outputs_size
    xnorm_output_tap = TensorAccessPattern(
        (output_tap_size,), xnorm_output_base, [1, 1, 1, hidden_size], [0, 0, 0, 1]
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

    def segment_major_weight_tap(base_offset, length):
        return TensorAccessPattern(
            (layer_iterations * weights_size,),
            base_offset,
            [1, 1, 1, layer_iterations * length],
            [0, 0, 0, 1],
        )

    def segment_major_mlp_down_weight_taps():
        shard_rows = hidden_size // mlp_down_columns
        return [
            TensorAccessPattern(
                (layer_iterations * weights_size,),
                segment_mlp_down_base + col * shard_rows * intermediate_size,
                [layer_iterations, 1, 1, shard_rows * intermediate_size],
                [segment_mlp_down_size, 0, 0, 1],
            )
            for col in range(mlp_down_columns)
        ]

    def segment_major_mlp_post_norm_weight_tap():
        return TensorAccessPattern(
            (layer_iterations * weights_size,),
            segment_mlp_post_norm_base,
            [1, 1, 1, layer_iterations * segment_mlp_post_norm_size],
            [0, 0, 0, 1],
        )

    def segment_major_mlp_gate_up_shard_weight_taps():
        return [
            TensorAccessPattern(
                (layer_iterations * weights_size,),
                segment_mlp_gate_up_base
                + col * layer_iterations * segment_mlp_gate_up_shard_size,
                [1, 1, 1, layer_iterations * segment_mlp_gate_up_shard_size],
                [0, 0, 0, 1],
            )
            for col in range(mlp_gate_up_columns)
        ]

    def segment_major_mlp_down_shard_weight_taps():
        return [
            TensorAccessPattern(
                (layer_iterations * weights_size,),
                segment_mlp_down_base
                + col * layer_iterations * segment_mlp_down_shard_size,
                [1, 1, 1, layer_iterations * segment_mlp_down_shard_size],
                [0, 0, 0, 1],
            )
            for col in range(mlp_down_columns)
        ]

    def per_layer_mlp_down_weight_tap(layer_idx, col):
        shard_rows = hidden_size // mlp_down_columns
        return TensorAccessPattern(
            (layer_iterations * weights_size,),
            segment_mlp_down_base
            + layer_idx * segment_mlp_down_size
            + col * shard_rows * intermediate_size,
            [1, 1, 1, shard_rows * intermediate_size],
            [0, 0, 0, 1],
        )

    def layer_group_cache_key_blocks_tap(layer_start, layer_count):
        return TensorAccessPattern(
            (layer_iterations * cache_size,),
            layer_start * cache_size,
            [
                layer_count,
                kv_heads,
                cache_blocks_to_process * cache_block_seq,
                head_dim,
            ],
            [cache_size, max_seq_len * head_dim, head_dim, 1],
        )

    def layer_group_cache_value_blocks_tap(layer_start, layer_count):
        return TensorAccessPattern(
            (layer_iterations * cache_size,),
            layer_start * cache_size + kv_size * max_seq_len,
            [
                layer_count,
                kv_heads,
                cache_blocks_to_process * cache_block_seq,
                head_dim,
            ],
            [cache_size, max_seq_len * head_dim, head_dim, 1],
        )

    def layer_group_cache_key_current_tap(layer_start, layer_count):
        return TensorAccessPattern(
            (layer_iterations * cache_size,),
            layer_start * cache_size + position * head_dim,
            [layer_count, 1, kv_heads, head_dim],
            [cache_size, 0, max_seq_len * head_dim, 1],
        )

    def layer_group_cache_value_current_tap(layer_start, layer_count):
        return TensorAccessPattern(
            (layer_iterations * cache_size,),
            layer_start * cache_size + kv_size * max_seq_len + position * head_dim,
            [layer_count, 1, kv_heads, head_dim],
            [cache_size, 0, max_seq_len * head_dim, 1],
        )

    q_weight_taps = weight_taps(q_size, q_weight_base)
    k_weight_taps = weight_taps(kv_size, k_weight_base)
    v_weight_taps = weight_taps(kv_size, v_weight_base)
    q_raw_output_taps = out_taps(q_size, q_raw_output_base)
    k_raw_output_taps = out_taps(kv_size, k_raw_output_base)
    q_norm_output_taps = out_taps(q_size, q_norm_output_base)
    k_norm_output_taps = out_taps(kv_size, k_norm_output_base)
    q_rope_output_taps = out_taps(q_size, q_rope_output_base)
    qk_pair_output_taps = out_taps(qk_pair_debug_size, qk_pair_output_base)
    k_cache_stream_output_taps = []
    attn_score_output_taps = out_taps(score_size, attn_scores_output_base)
    attn_weight_output_taps = out_taps(score_size, attn_weights_output_base)
    v_context_output_taps = out_taps(v_cache_debug_size, v_context_stream_output_base)
    attn_context_output_taps = out_taps(context_size, attn_context_output_base)
    attn_context_flat_output_taps = out_taps(
        context_flat_size, attn_context_flat_output_base
    )
    o_weight_taps = weight_taps_for_k(hidden_size, q_size, o_weight_base)
    residual_hidden_taps = []
    o_proj_output_taps = out_taps(o_proj_size, attn_o_proj_output_base)
    residual_output_taps = out_taps(residual_size, attn_residual_output_base)
    if mlp_down_columns == 1:
        mlp_down_weight_taps = [
            segment_major_weight_tap(segment_mlp_down_base, segment_mlp_down_size)
        ]
    elif mlp_gate_up_columns != 1:
        mlp_down_weight_taps = segment_major_mlp_down_shard_weight_taps()
    else:
        mlp_down_weight_taps = segment_major_mlp_down_weight_taps()
    mlp_xnorm_output_tap = TensorAccessPattern(
        (output_tap_size,), mlp_xnorm_output_base, [1, 1, 1, hidden_size], [0, 0, 0, 1]
    )
    ffn_gate_output_tap = TensorAccessPattern(
        (output_tap_size,),
        ffn_gate_output_base,
        [1, 1, 1, intermediate_size],
        [0, 0, 0, 1],
    )
    ffn_up_output_tap = TensorAccessPattern(
        (output_tap_size,),
        ffn_up_output_base,
        [1, 1, 1, intermediate_size],
        [0, 0, 0, 1],
    )
    ffn_gate_silu_output_tap = TensorAccessPattern(
        (output_tap_size,),
        ffn_gate_silu_output_base,
        [1, 1, 1, intermediate_size],
        [0, 0, 0, 1],
    )
    ffn_hidden_output_tap = TensorAccessPattern(
        (output_tap_size,),
        ffn_hidden_output_base,
        [1, 1, 1, intermediate_size],
        [0, 0, 0, 1],
    )
    ffn_out_output_tap = TensorAccessPattern(
        (output_tap_size,), ffn_out_output_base, [1, 1, 1, hidden_size], [0, 0, 0, 1]
    )
    layer_residual_output_tap = TensorAccessPattern(
        (output_tap_size,),
        layer_residual_output_base,
        [1, 1, 1, hidden_size],
        [0, 0, 0, 1],
    )
    final_layer_residual_output_tap = TensorAccessPattern(
        (outputs_size,), 0, [1, 1, 1, hidden_size], [0, 0, 0, 1]
    )
    final_layer_residual_output_taps = [
        TensorAccessPattern(
            (outputs_size,),
            col * (hidden_size // mlp_down_columns),
            [1, 1, 1, hidden_size // mlp_down_columns],
            [0, 0, 0, 1],
        )
        for col in range(mlp_down_columns)
    ]
    rt = Runtime()
    # Debug chunks use the proven group-of-4 schedule. The full-depth generate
    # path needs one cache task per FIFO; otherwise chunk=28 exhausts BD IDs.
    layer_writeback_dma_group_size = 4 if layer_iterations <= 8 else layer_iterations
    layer_writeback_dma_groups = [
        (
            layer_start,
            min(layer_writeback_dma_group_size, layer_iterations - layer_start),
        )
        for layer_start in range(0, layer_iterations, layer_writeback_dma_group_size)
    ]
    layer_cache_dma_group_size = 4 if layer_iterations <= 8 else layer_iterations
    layer_cache_dma_groups = [
        (
            layer_start,
            min(layer_cache_dma_group_size, layer_iterations - layer_start),
        )
        for layer_start in range(0, layer_iterations, layer_cache_dma_group_size)
    ]

    def fill_chunk_weight_segments(weights, tg):
        rt.fill(
            in_weight.prod(),
            weights,
            segment_major_weight_tap(segment_input_norm_base, segment_input_norm_size),
            task_group=tg,
        )
        rt.fill(
            qk_norm_weight.prod(),
            weights,
            segment_major_weight_tap(segment_qk_norm_base, segment_qk_norm_size),
            task_group=tg,
        )
        for col in range(num_columns):
            rt.fill(
                q_weight_fifos[col].prod(),
                weights,
                segment_major_weight_tap(segment_q_weight_base, segment_q_weight_size),
                task_group=tg,
            )
            rt.fill(
                k_weight_fifos[col].prod(),
                weights,
                segment_major_weight_tap(segment_k_weight_base, segment_k_weight_size),
                task_group=tg,
            )
            rt.fill(
                v_weight_fifos[col].prod(),
                weights,
                segment_major_weight_tap(segment_v_weight_base, segment_v_weight_size),
                task_group=tg,
            )
            rt.fill(
                o_weight_fifos[col].prod(),
                weights,
                segment_major_weight_tap(segment_o_weight_base, segment_o_weight_size),
                task_group=tg,
            )
        if mlp_gate_up_columns == 1:
            rt.fill(
                mlp_gate_up_weight_rows.prod(),
                weights,
                segment_major_weight_tap(
                    segment_mlp_gate_up_base,
                    segment_mlp_gate_up_size,
                ),
                task_group=tg,
            )
        else:
            rt.fill(
                mlp_post_norm_weight.prod(),
                weights,
                segment_major_mlp_post_norm_weight_tap(),
                task_group=tg,
            )
            for col in range(mlp_gate_up_columns):
                rt.fill(
                    mlp_gate_up_shard_weight_fifos[col].prod(),
                    weights,
                    segment_major_mlp_gate_up_shard_weight_taps()[col],
                    task_group=tg,
                )
        if mlp_down_columns == 1:
            rt.fill(
                mlp_down_weight_fifos[0].prod(),
                weights,
                mlp_down_weight_taps[0],
                task_group=tg,
            )
        elif mlp_gate_up_columns != 1:
            mlp_down_shard_weight_taps = segment_major_mlp_down_shard_weight_taps()
            for col in range(mlp_down_columns):
                rt.fill(
                    mlp_down_weight_fifos[col].prod(),
                    weights,
                    mlp_down_shard_weight_taps[col],
                    task_group=tg,
                )
        else:
            for layer_idx in range(layer_iterations):
                for col in range(mlp_down_columns):
                    rt.fill(
                        mlp_down_weight_fifos[col].prod(),
                        weights,
                        per_layer_mlp_down_weight_tap(layer_idx, col),
                        task_group=tg,
                    )

    def fill_chunk_layer_group_cache_inputs(layer_start, layer_count, cache, tg):
        for col in range(num_columns):
            rt.fill(
                k_cache_fifos[col].prod(),
                cache,
                layer_group_cache_key_blocks_tap(layer_start, layer_count),
                task_group=tg,
            )
            rt.fill(
                v_cache_raw_fifos[col].prod(),
                cache,
                layer_group_cache_value_blocks_tap(layer_start, layer_count),
                task_group=tg,
            )

    def drain_chunk_cache(cache, tg):
        for layer_start, layer_count in layer_writeback_dma_groups:
            for col in range(num_columns):
                rt.drain(
                    k_rope_fifos[col].cons(),
                    cache,
                    layer_group_cache_key_current_tap(layer_start, layer_count),
                    wait=True,
                    task_group=tg,
                )
                rt.drain(
                    v_fifos[col].cons(),
                    cache,
                    layer_group_cache_value_current_tap(layer_start, layer_count),
                    wait=True,
                    task_group=tg,
                )

    with rt.sequence(
        tensor_ty, weights_chunk_ty, angles_ty, outputs_ty, cache_chunk_ty
    ) as (hidden, weights, angles, outputs, cache):
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(runtime_hidden_in.prod(), hidden, hidden_tap, task_group=tg)
        rt.fill(rope_angles.prod(), angles, rope_angles_tap, task_group=tg)
        fill_chunk_weight_segments(weights, tg)
        for layer_start, layer_count in layer_cache_dma_groups:
            fill_chunk_layer_group_cache_inputs(layer_start, layer_count, cache, tg)
        drain_chunk_cache(cache, tg)
        if mlp_down_columns == 1 or layer_iterations > 1:
            rt.drain(
                (
                    layer_residual.cons()
                    if layer_iterations == 1
                    else final_layer_residual.cons()
                ),
                outputs,
                final_layer_residual_output_tap,
                wait=True,
                task_group=tg,
            )
        else:
            for col in range(mlp_down_columns):
                rt.drain(
                    layer_residual_fifos[col].cons(),
                    outputs,
                    final_layer_residual_output_taps[col],
                    wait=True,
                    task_group=tg,
                )
        rt.finish_task_group(tg)
    return Program(dev, rt).resolve_program(SequentialPlacer())


def qwen3_persistent_n_layer_final_only(
    dev,
    hidden_size,
    q_size,
    kv_size,
    head_dim,
    max_seq_len,
    position,
    intermediate_size,
    layer_iterations,
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
    """One or more sequential Qwen3 full layers that drain only final hidden."""
    return _qwen3_persistent_n_layer_final_only_impl(
        dev,
        hidden_size,
        q_size,
        kv_size,
        head_dim,
        max_seq_len,
        position,
        1,
        tile_size_input,
        tile_size_output,
        trace_size,
        intermediate_size=intermediate_size,
        func_prefix=func_prefix,
        rms_kernel_object=rms_kernel_object,
        gemv_kernel_object=gemv_kernel_object,
        rope_kernel_object=rope_kernel_object,
        attention_kernel_object=attention_kernel_object,
        passthrough_kernel_object=passthrough_kernel_object,
        softmax_kernel_object=softmax_kernel_object,
        o_gemv_kernel_object=o_gemv_kernel_object,
        add_kernel_object=add_kernel_object,
        mlp_gemv_kernel_object=mlp_gemv_kernel_object,
        silu_kernel_object=silu_kernel_object,
        mul_kernel_object=mul_kernel_object,
        down_gemv_kernel_object=down_gemv_kernel_object,
        layer_iterations=layer_iterations,
        mlp_down_columns=num_columns,
    )
