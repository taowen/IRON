"""CPU reference for exp36 current-write one-read KV fanout attention."""

from math import ceil

import numpy as np

NUM_Q_HEADS = 4
HEAD_DIM = 128
TOKENS_PER_TILE = 16
DIM_GROUPS = 32
GROUP_DWORDS = 4

QUERY_DWORDS = NUM_Q_HEADS * HEAD_DIM
CURRENT_DWORDS = 2 * HEAD_DIM
PLANE_TILE_DWORDS = TOKENS_PER_TILE * HEAD_DIM
RESHAPED_TILE_DWORDS = DIM_GROUPS * TOKENS_PER_TILE * GROUP_DWORDS
OUTPUT_DWORDS = QUERY_DWORDS


def num_tiles_for_context(context_len: int) -> int:
    return ceil(context_len / TOKENS_PER_TILE)


def current_offsets(context_len: int) -> tuple[int, int]:
    tile = (context_len - 1) // TOKENS_PER_TILE
    token = (context_len - 1) % TOKENS_PER_TILE
    k_plane_dwords = num_tiles_for_context(context_len) * PLANE_TILE_DWORDS
    k_offset = tile * PLANE_TILE_DWORDS + token * HEAD_DIM
    v_offset = k_plane_dwords + tile * PLANE_TILE_DWORDS + token * HEAD_DIM
    return k_offset, v_offset


def approx_exp(x: np.ndarray | np.float32) -> np.ndarray | np.float32:
    clipped = np.minimum(np.maximum(x, np.float32(-16.0)), np.float32(0.0)).astype(np.float32)
    y = (np.float32(1.0) + clipped * np.float32(1.0 / 64.0)).astype(np.float32)
    for _ in range(6):
        y = (y * y).astype(np.float32)
    return y


def reshaped_index(dim: int, token: int) -> int:
    dim_group = dim // GROUP_DWORDS
    lane = dim - dim_group * GROUP_DWORDS
    return (dim_group * TOKENS_PER_TILE + token) * GROUP_DWORDS + lane


def make_query(seed: int = 36) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.08, 0.08, QUERY_DWORDS).astype(np.float32)


def make_current_kv(context_len: int) -> np.ndarray:
    current = np.zeros(CURRENT_DWORDS, dtype=np.float32)
    for dim in range(HEAD_DIM):
        current[dim] = np.float32((((context_len + dim * 3) % 31) - 15) * 0.01171875)
        current[HEAD_DIM + dim] = np.float32((((context_len * 2 + dim * 5) % 37) - 18) * 0.009375)
    return current


def _plane_value(global_t: int, dim: int, plane: str) -> np.float32:
    if plane == "v":
        raw = ((global_t * 5 + dim * 3) % 23) - 11
        return np.float32(raw * 0.01875)
    raw = ((global_t * 7 + dim * 2) % 29) - 14
    return np.float32(raw * 0.0125)


def make_kv_cache_without_current(context_len: int) -> np.ndarray:
    num_tiles = num_tiles_for_context(context_len)
    k_plane_dwords = num_tiles * PLANE_TILE_DWORDS
    current_token = context_len - 1
    data = np.zeros(2 * k_plane_dwords, dtype=np.float32)
    for tile in range(num_tiles):
        for token in range(TOKENS_PER_TILE):
            global_t = tile * TOKENS_PER_TILE + token
            for dim in range(HEAD_DIM):
                k_offset = tile * PLANE_TILE_DWORDS + token * HEAD_DIM + dim
                v_offset = k_plane_dwords + tile * PLANE_TILE_DWORDS + token * HEAD_DIM + dim
                if global_t < context_len and global_t != current_token:
                    data[k_offset] = _plane_value(global_t, dim, "k")
                    data[v_offset] = _plane_value(global_t, dim, "v")
                elif global_t >= context_len:
                    data[k_offset] = np.float32(9.0 + dim * 0.01)
                    data[v_offset] = np.float32(-7.0 - dim * 0.01)
    return data


def apply_current_to_cache(context_len: int, kv_cache: np.ndarray, current_kv: np.ndarray) -> np.ndarray:
    out = kv_cache.copy()
    k_offset, v_offset = current_offsets(context_len)
    out[k_offset:k_offset + HEAD_DIM] = current_kv[:HEAD_DIM]
    out[v_offset:v_offset + HEAD_DIM] = current_kv[HEAD_DIM:]
    return out


def reshape_tile_token_to_dim_group(tile: np.ndarray) -> np.ndarray:
    out = np.zeros(RESHAPED_TILE_DWORDS, dtype=np.float32)
    for dim_group in range(DIM_GROUPS):
        for token in range(TOKENS_PER_TILE):
            for lane in range(GROUP_DWORDS):
                dim = dim_group * GROUP_DWORDS + lane
                out[(dim_group * TOKENS_PER_TILE + token) * GROUP_DWORDS + lane] = tile[token * HEAD_DIM + dim]
    return out


def _online_attention_head(context_len: int, query: np.ndarray, kv_cache: np.ndarray) -> np.ndarray:
    num_tiles = num_tiles_for_context(context_len)
    k_plane_dwords = num_tiles * PLANE_TILE_DWORDS
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    out = np.zeros(HEAD_DIM, dtype=np.float32)
    running_max = np.float32(-np.inf)
    running_sum = np.float32(0.0)

    for tile in range(num_tiles):
        valid = TOKENS_PER_TILE if tile < num_tiles - 1 else context_len - tile * TOKENS_PER_TILE
        k_raw = kv_cache[tile * PLANE_TILE_DWORDS:(tile + 1) * PLANE_TILE_DWORDS]
        v_raw = kv_cache[k_plane_dwords + tile * PLANE_TILE_DWORDS:k_plane_dwords + (tile + 1) * PLANE_TILE_DWORDS]
        k_tile = reshape_tile_token_to_dim_group(k_raw)
        v_tile = reshape_tile_token_to_dim_group(v_raw)
        scores = np.zeros(valid, dtype=np.float32)
        for token in range(valid):
            score = np.float32(0.0)
            for dim in range(HEAD_DIM):
                score += query[dim] * k_tile[reshaped_index(dim, token)]
            scores[token] = score * scale

        local_max = np.max(scores)
        new_max = max(running_max, local_max)
        old_scale = np.float32(0.0) if running_sum == 0 else approx_exp(running_max - new_max)
        weights = approx_exp(scores - new_max).astype(np.float32)
        running_sum = running_sum * old_scale + np.sum(weights)
        out *= old_scale
        for token in range(valid):
            for dim in range(HEAD_DIM):
                out[dim] += weights[token] * v_tile[reshaped_index(dim, token)]
        running_max = new_max

    out /= running_sum
    return out.astype(np.float32)


def online_attention_from_reshaped(context_len: int, query: np.ndarray, kv_cache: np.ndarray) -> np.ndarray:
    out = np.zeros(OUTPUT_DWORDS, dtype=np.float32)
    for head in range(NUM_Q_HEADS):
        start = head * HEAD_DIM
        out[start:start + HEAD_DIM] = _online_attention_head(
            context_len,
            query[start:start + HEAD_DIM],
            kv_cache,
        )
    return out


def make_case(context_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query = make_query()
    current_kv = make_current_kv(context_len)
    cache_before = make_kv_cache_without_current(context_len)
    cache_after = apply_current_to_cache(context_len, cache_before, current_kv)
    expected = online_attention_from_reshaped(context_len, query, cache_after)
    return query, current_kv, cache_before, cache_after, expected


if __name__ == "__main__":
    q, current, before, after, out = make_case(31)
    k_offset, v_offset = current_offsets(31)
    print(f"q0={q[:4]} current_k0={current[:4]} cache_k0={after[k_offset:k_offset + 4]} out0={out[:4]}")
