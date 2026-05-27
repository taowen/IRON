"""CPU reference for exp28 multi-tile shape-A/shape-B online attention."""

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


def num_tiles_for_context(context_len: int) -> int:
    return ceil(context_len / TOKENS_PER_TILE)


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
        lane = np.float32((d % 13) - 6)
        current[d] = lane * np.float32(0.0234375) + np.float32((d % 5) * 0.00390625)
    for i in range(HEAD_DIM, CURRENT_DWORDS):
        current[i] = np.float32(((i * 3 + context_len) % 19) - 9) * np.float32(0.015625)
    return current


def _history_tile(context_len: int, tile_idx: int, plane: str) -> np.ndarray:
    history = np.zeros(HISTORY_SOURCE_DWORDS, dtype=np.float32)
    plane_bias = np.float32(0.109375 if plane == "v" else -0.046875)
    for t in range(TOKENS_PER_TILE):
        global_t = tile_idx * TOKENS_PER_TILE + t
        for d in range(HEAD_DIM):
            idx = t * HEAD_DIM + d
            raw = ((context_len + global_t * 7 + d * 5 + (17 if plane == "v" else 0)) % 29) - 14
            history[idx] = np.float32(raw) * np.float32(0.01171875) + plane_bias
    for i in range(HISTORY_DWORDS, HISTORY_SOURCE_DWORDS):
        history[i] = np.float32(((i + tile_idx + context_len) % 31) - 15) * np.float32(0.015625)
    return history


def make_history_payload(context_len: int, plane: str) -> np.ndarray:
    return np.concatenate(
        [_history_tile(context_len, tile_idx, plane) for tile_idx in range(num_tiles_for_context(context_len))]
    ).astype(np.float32)


def shape_a_sideband(
    current: np.ndarray,
    k_tile: np.ndarray,
    tile_idx: int,
    num_tiles: int,
    last_valid: int,
) -> np.ndarray:
    valid = TOKENS_PER_TILE if tile_idx < num_tiles - 1 else last_valid
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    scores = np.zeros(TOKENS_PER_TILE, dtype=np.float32)
    for t in range(valid):
        q = current[:HEAD_DIM].astype(np.float32)
        k = k_tile[t * HEAD_DIM:(t + 1) * HEAD_DIM].astype(np.float32)
        scores[t] = np.float32(np.sum(q * k) * scale)
    local_max = np.max(scores[:valid])
    sideband = np.zeros(SIDEBAND_DWORDS, dtype=np.float32)
    sideband[:valid] = approx_exp(scores[:valid] - local_max).astype(np.float32)
    sideband[16] = local_max
    return sideband


def shape_b_online_reference(context_len: int) -> tuple[np.ndarray, np.ndarray]:
    current = make_current_payload(context_len)
    k_history = make_history_payload(context_len, "k")
    v_history = make_history_payload(context_len, "v")
    num_tiles = num_tiles_for_context(context_len)
    last_valid = last_valid_for_context(context_len)

    output = np.zeros(ATTENTION_OUT_DWORDS, dtype=np.float32)
    running_sum = np.float32(0.0)
    running_max = np.float32(-np.inf)
    debug = np.zeros(SIDEBAND_DEBUG_DWORDS, dtype=np.float32)

    for tile_idx in range(num_tiles):
        offset = tile_idx * HISTORY_SOURCE_DWORDS
        k_tile = k_history[offset:offset + HISTORY_DWORDS]
        v_tile = v_history[offset:offset + HISTORY_DWORDS]
        sideband = shape_a_sideband(current, k_tile, tile_idx, num_tiles, last_valid)
        local_sum = np.sum(sideband[:16], dtype=np.float32)
        local_max = sideband[16]
        new_max = max(running_max, local_max)
        old_scale = np.float32(0.0) if running_sum == 0 else approx_exp(np.float32(running_max - new_max))
        tile_scale = np.float32(0.0) if local_sum == 0 else approx_exp(np.float32(local_max - new_max))

        for d in range(HEAD_DIM):
            tile_acc = np.float32(0.0)
            for t in range(TOKENS_PER_TILE):
                tile_acc = np.float32(tile_acc + sideband[t] * v_tile[t * HEAD_DIM + d])
            output[d] = np.float32(output[d] * old_scale + tile_scale * tile_acc)

        running_sum = np.float32(running_sum * old_scale + tile_scale * local_sum)
        running_max = np.float32(new_max)
        output[128] = running_sum
        output[129] = running_max
        output[130] = local_max
        output[131] = local_sum
        debug[0] = np.float32(debug[0] + local_sum)
        debug[1] = sideband[0]
        debug[2] = sideband[15]
        debug[3] = np.float32(tile_idx)

    output[:HEAD_DIM] = output[:HEAD_DIM] / running_sum
    return debug, output


def direct_attention_reference(context_len: int) -> np.ndarray:
    current = make_current_payload(context_len)
    k_history = make_history_payload(context_len, "k")
    v_history = make_history_payload(context_len, "v")
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    scores = np.zeros(context_len, dtype=np.float32)
    values = np.zeros((context_len, HEAD_DIM), dtype=np.float32)
    for token in range(context_len):
        tile_idx = token // TOKENS_PER_TILE
        local_t = token % TOKENS_PER_TILE
        offset = tile_idx * HISTORY_SOURCE_DWORDS + local_t * HEAD_DIM
        k = k_history[offset:offset + HEAD_DIM]
        v = v_history[offset:offset + HEAD_DIM]
        scores[token] = np.float32(np.sum(current[:HEAD_DIM] * k) * scale)
        values[token] = v
    weights = approx_exp(scores - np.max(scores)).astype(np.float32)
    weights = weights / np.sum(weights, dtype=np.float32)
    out = np.zeros(ATTENTION_OUT_DWORDS, dtype=np.float32)
    out[:HEAD_DIM] = weights @ values
    return out


def expected_output(context_len: int) -> np.ndarray:
    debug, output = shape_b_online_reference(context_len)
    direct = direct_attention_reference(context_len)
    if not np.allclose(output[:HEAD_DIM], direct[:HEAD_DIM], rtol=1e-5, atol=1e-6):
        raise RuntimeError("tiled online attention reference diverged from direct attention")
    return np.concatenate([debug, output]).astype(np.float32)


if __name__ == "__main__":
    for length in [17, 31, 32, 64, 128, 129]:
        ref = expected_output(length)
        print(
            f"L={length:3d} tiles={num_tiles_for_context(length):2d} "
            f"last={last_valid_for_context(length):2d} out[0:8]={ref[:8]}"
        )
