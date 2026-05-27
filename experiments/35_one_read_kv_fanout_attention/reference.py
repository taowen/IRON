"""CPU reference for exp35 one-read KV fanout attention."""

from math import ceil

import numpy as np

NUM_Q_HEADS = 4
HEAD_DIM = 128
TOKENS_PER_TILE = 16
DIM_GROUPS = 32
GROUP_DWORDS = 4

QUERY_DWORDS = NUM_Q_HEADS * HEAD_DIM
PLANE_TILE_DWORDS = TOKENS_PER_TILE * HEAD_DIM
RESHAPED_TILE_DWORDS = DIM_GROUPS * TOKENS_PER_TILE * GROUP_DWORDS
OUTPUT_DWORDS = QUERY_DWORDS


def num_tiles_for_context(context_len: int) -> int:
    return ceil(context_len / TOKENS_PER_TILE)


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


def make_query(seed: int = 34) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.08, 0.08, QUERY_DWORDS).astype(np.float32)


def _plane_value(global_t: int, dim: int, plane: str) -> np.float32:
    if plane == "v":
        raw = ((global_t * 5 + dim * 3) % 23) - 11
        return np.float32(raw * 0.01875)
    raw = ((global_t * 7 + dim * 2) % 29) - 14
    return np.float32(raw * 0.0125)


def make_kv_cache(context_len: int) -> np.ndarray:
    num_tiles = num_tiles_for_context(context_len)
    k_plane_dwords = num_tiles * PLANE_TILE_DWORDS
    data = np.zeros(2 * k_plane_dwords, dtype=np.float32)
    for tile in range(num_tiles):
        for token in range(TOKENS_PER_TILE):
            global_t = tile * TOKENS_PER_TILE + token
            for dim in range(HEAD_DIM):
                k_offset = tile * PLANE_TILE_DWORDS + token * HEAD_DIM + dim
                v_offset = k_plane_dwords + tile * PLANE_TILE_DWORDS + token * HEAD_DIM + dim
                if global_t < context_len:
                    data[k_offset] = _plane_value(global_t, dim, "k")
                    data[v_offset] = _plane_value(global_t, dim, "v")
                else:
                    data[k_offset] = np.float32(9.0 + dim * 0.01)
                    data[v_offset] = np.float32(-7.0 - dim * 0.01)
    return data


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


def make_case(context_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = make_query()
    kv_cache = make_kv_cache(context_len)
    expected = online_attention_from_reshaped(context_len, query, kv_cache)
    return query, kv_cache, expected


if __name__ == "__main__":
    q, kv, out = make_case(31)
    first_tile = reshape_tile_token_to_dim_group(kv[:PLANE_TILE_DWORDS])
    print(f"query0={q[:4]} reshaped_k0={first_tile[:8]} out0={out[:4]}")
