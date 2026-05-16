# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import torch


def _pad_to_multiple_of_64(tensor, seq_dim):
    seq_len = tensor.shape[seq_dim]
    padded_seq_len = ((seq_len + 63) // 64) * 64
    if padded_seq_len == seq_len:
        return tensor

    pad_size = padded_seq_len - seq_len
    pad_dims = [0] * (2 * tensor.ndim)
    pad_dims[2 * (tensor.ndim - 1 - seq_dim) + 1] = pad_size
    return torch.nn.functional.pad(tensor, pad_dims)


def _quantize_symmetric(x):
    x_f32 = x.to(torch.float32)
    max_abs = torch.max(torch.abs(x_f32)).item()
    scale = max(max_abs / 127.0, 1.0e-8)
    q = torch.clamp(torch.round(x_f32 / scale), -128, 127).to(torch.int8)
    return q, scale


def _quantize_blocks(x, block_size=64):
    q = torch.empty_like(x, dtype=torch.int8)
    scales = []
    for offset in range(0, x.shape[0], block_size):
        q_block, scale = _quantize_symmetric(x[offset : offset + block_size])
        q[offset : offset + block_size] = q_block
        scales.append(scale)
    return q, torch.tensor(scales, dtype=torch.float32)


def generate_golden_reference(S_q=128, S_kv=128, d=64, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    val_range = 4
    Q = torch.rand(S_q, d, dtype=torch.bfloat16) * val_range
    K = torch.rand(S_kv, d, dtype=torch.bfloat16) * val_range
    V = torch.rand(S_kv, d, dtype=torch.bfloat16) * val_range

    causal_mask = torch.ones(S_q, S_kv, dtype=torch.bool).tril()
    full_logits = (Q.to(torch.float32) @ K.to(torch.float32).transpose(0, 1)) * (
        1.0 / math.sqrt(d)
    )
    full_logits = full_logits.masked_fill(~causal_mask, float("-inf"))
    O_full = torch.softmax(full_logits, dim=-1).to(torch.bfloat16) @ V

    # SAGEAttn-B smoothing: subtract mean over token dimension.
    K_smooth = K - K.to(torch.float32).mean(dim=0, keepdim=True).to(torch.bfloat16)

    smooth_logits = (
        Q.to(torch.float32) @ K_smooth.to(torch.float32).transpose(0, 1)
    ) * (1.0 / math.sqrt(d))
    smooth_logits = smooth_logits.masked_fill(~causal_mask, float("-inf"))
    O_smooth = torch.softmax(smooth_logits, dim=-1).to(torch.bfloat16) @ V

    Q_pad = _pad_to_multiple_of_64(Q, seq_dim=0)
    K_pad = _pad_to_multiple_of_64(K_smooth, seq_dim=0)
    V_pad = _pad_to_multiple_of_64(V, seq_dim=0)

    Q_i8, q_scales = _quantize_blocks(Q_pad)
    K_i8, k_scales = _quantize_blocks(K_pad)
    dequant_scales = (q_scales[:, None] * k_scales[None, :]).contiguous().reshape(-1)

    Q_deq_blocks = []
    for block_idx, offset in enumerate(range(0, Q_i8.shape[0], 64)):
        Q_deq_blocks.append(
            Q_i8[offset : offset + 64].to(torch.float32) * q_scales[block_idx]
        )
    K_deq_blocks = []
    for block_idx, offset in enumerate(range(0, K_i8.shape[0], 64)):
        K_deq_blocks.append(
            K_i8[offset : offset + 64].to(torch.float32) * k_scales[block_idx]
        )
    Q_deq = torch.cat(Q_deq_blocks, dim=0)[:S_q]
    K_deq = torch.cat(K_deq_blocks, dim=0)[:S_kv]
    logits = (Q_deq @ K_deq.transpose(0, 1)) * (1.0 / math.sqrt(d))

    logits = logits.masked_fill(~causal_mask, float("-inf"))
    P = torch.softmax(logits, dim=-1).to(torch.bfloat16)
    O = P @ V

    return {
        "Q": _pad_to_multiple_of_64(Q, seq_dim=0),
        "K": _pad_to_multiple_of_64(K, seq_dim=0),
        "Q_i8": _pad_to_multiple_of_64(Q_i8, seq_dim=0),
        "K_i8": _pad_to_multiple_of_64(K_i8, seq_dim=0),
        "V": V_pad,
        "O_full": _pad_to_multiple_of_64(O_full, seq_dim=0),
        "O_smooth": _pad_to_multiple_of_64(O_smooth, seq_dim=0),
        "O": _pad_to_multiple_of_64(O, seq_dim=0),
        "q_scales": q_scales,
        "k_scales": k_scales,
        "dequant_scales": dequant_scales,
    }
