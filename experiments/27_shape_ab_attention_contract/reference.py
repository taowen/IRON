"""CPU reference for exp27 shape-A/shape-B attention contract."""

from __future__ import annotations

from math import ceil

import numpy as np

CURRENT_DWORDS = 512
HISTORY_SOURCE_DWORDS = 4096
HISTORY_DWORDS = 2048
SIDEBAND_DWORDS = 17
SIDEBAND_DEBUG_DWORDS = 4
ATTENTION_OUT_DWORDS = 512
TOTAL_OUT_DWORDS = SIDEBAND_DEBUG_DWORDS + ATTENTION_OUT_DWORDS

HEAD_DIM = 128
TOKENS_PER_TILE = 16


def last_valid_for_context(context_len: int) -> int:
    valid = context_len % TOKENS_PER_TILE
    return TOKENS_PER_TILE if valid == 0 else valid


def approx_exp(x: np.ndarray | np.float32) -> np.ndarray | np.float32:
    clipped = np.minimum(np.maximum(x, np.float32(-16.0)), np.float32(0.0)).astype(np.float32)
    y = (np.float32(1.0) + clipped * np.float32(1.0 / 64.0)).astype(np.float32)
    for _ in range(6):
        y = (y * y).astype(np.float32)
    return y


def make_current_payload(context_len: int) -> np.ndarray:
    current = np.zeros(CURRENT_DWORDS, dtype=np.float32)
    for d in range(HEAD_DIM):
        lane = np.float32((d % 11) - 5)
        current[d] = lane * np.float32(0.03125) + np.float32((d % 3) * 0.0078125)
    for i in range(HEAD_DIM, CURRENT_DWORDS):
        current[i] = np.float32(((i * 3 + context_len) % 19) - 9) * np.float32(0.015625)
    current[CURRENT_DWORDS - 1] = np.float32(last_valid_for_context(context_len))
    return current


def make_history_payload(context_len: int, plane: str) -> np.ndarray:
    history = np.zeros(HISTORY_SOURCE_DWORDS, dtype=np.float32)
    plane_bias = np.float32(0.125 if plane == "v" else -0.0625)
    for t in range(TOKENS_PER_TILE):
        for d in range(HEAD_DIM):
            idx = t * HEAD_DIM + d
            raw = ((context_len + t * 7 + d * 5 + (13 if plane == "v" else 0)) % 23) - 11
            history[idx] = np.float32(raw) * np.float32(0.015625) + plane_bias
    for i in range(HISTORY_DWORDS, HISTORY_SOURCE_DWORDS):
        history[i] = np.float32(((i + context_len) % 29) - 14) * np.float32(0.03125)
    return history


def shape_a_sideband(current: np.ndarray, k_history: np.ndarray) -> np.ndarray:
    valid = int(current[CURRENT_DWORDS - 1])
    valid = min(max(valid, 1), TOKENS_PER_TILE)
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    scores = np.zeros(TOKENS_PER_TILE, dtype=np.float32)
    for t in range(valid):
        q = current[:HEAD_DIM].astype(np.float32)
        k = k_history[t * HEAD_DIM:(t + 1) * HEAD_DIM].astype(np.float32)
        scores[t] = np.float32(np.sum(q * k) * scale)

    max_score = np.max(scores[:valid])
    sideband = np.zeros(SIDEBAND_DWORDS, dtype=np.float32)
    weights = approx_exp(scores[:valid] - max_score).astype(np.float32)
    sideband[:valid] = weights
    sideband[16] = np.sum(weights, dtype=np.float32)
    return sideband


def shape_b_attention(sideband: np.ndarray, v_history: np.ndarray) -> np.ndarray:
    out = np.zeros(ATTENTION_OUT_DWORDS, dtype=np.float32)
    denom = sideband[16]
    for d in range(HEAD_DIM):
        acc = np.float32(0.0)
        for t in range(TOKENS_PER_TILE):
            acc = np.float32(acc + sideband[t] * v_history[t * HEAD_DIM + d])
        out[d] = np.float32(0.0) if denom == 0 else np.float32(acc / denom)
    out[128] = denom
    out[129] = sideband[0]
    out[130] = sideband[15]
    out[131] = v_history[0]
    return out


def expected_output(context_len: int) -> np.ndarray:
    current = make_current_payload(context_len)
    k_history = make_history_payload(context_len, "k")
    v_history = make_history_payload(context_len, "v")
    sideband = shape_a_sideband(current, k_history)
    sideband_debug = np.asarray(
        [np.sum(sideband[:16], dtype=np.float32), sideband[0], sideband[15], sideband[16]],
        dtype=np.float32,
    )
    attention = shape_b_attention(sideband, v_history)
    return np.concatenate([sideband_debug, attention]).astype(np.float32)


if __name__ == "__main__":
    for length in [17, 31, 32, 128]:
        ref = expected_output(length)
        print(f"L={length} valid={last_valid_for_context(length)} out[0:8]={ref[:8]}")
