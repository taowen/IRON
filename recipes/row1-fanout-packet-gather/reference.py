"""CPU reference for row1 fanout and packet-gather routing."""

from math import ceil

import numpy as np

NUM_KV_GROUPS = 8
Q_HEADS_PER_GROUP = 4
NUM_Q_HEADS = NUM_KV_GROUPS * Q_HEADS_PER_GROUP
HEAD_DIM = 128
TOKENS_PER_TILE = 16
DIM_GROUPS = 32
GROUP_DWORDS = 4

QUERY_GROUP_DWORDS = Q_HEADS_PER_GROUP * HEAD_DIM
QUERY_DWORDS = NUM_Q_HEADS * HEAD_DIM
PLANE_TILE_DWORDS = TOKENS_PER_TILE * HEAD_DIM
RESHAPED_TILE_DWORDS = DIM_GROUPS * TOKENS_PER_TILE * GROUP_DWORDS
OUTPUT_DWORDS = QUERY_DWORDS


def num_tiles_for_context(context_len: int) -> int:
    return ceil(context_len / TOKENS_PER_TILE)


def kv_group_stride_dwords(context_len: int) -> int:
    return 2 * num_tiles_for_context(context_len) * PLANE_TILE_DWORDS


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


def make_query(seed: int = 38) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.08, 0.08, QUERY_DWORDS).astype(np.float32)


def _plane_value(group: int, global_t: int, dim: int, plane: str) -> np.float32:
    if plane == "v":
        raw = ((group * 11 + global_t * 5 + dim * 3) % 31) - 15
        return np.float32(raw * 0.0125)
    raw = ((group * 13 + global_t * 7 + dim * 2) % 37) - 18
    return np.float32(raw * 0.00875)


def make_kv_cache(context_len: int) -> np.ndarray:
    num_tiles = num_tiles_for_context(context_len)
    k_plane_dwords = num_tiles * PLANE_TILE_DWORDS
    group_stride = 2 * k_plane_dwords
    data = np.zeros(NUM_KV_GROUPS * group_stride, dtype=np.float32)
    for group in range(NUM_KV_GROUPS):
        group_base = group * group_stride
        for tile in range(num_tiles):
            for token in range(TOKENS_PER_TILE):
                global_t = tile * TOKENS_PER_TILE + token
                for dim in range(HEAD_DIM):
                    k_offset = group_base + tile * PLANE_TILE_DWORDS + token * HEAD_DIM + dim
                    v_offset = group_base + k_plane_dwords + tile * PLANE_TILE_DWORDS + token * HEAD_DIM + dim
                    if global_t < context_len:
                        data[k_offset] = _plane_value(group, global_t, dim, "k")
                        data[v_offset] = _plane_value(group, global_t, dim, "v")
                    else:
                        data[k_offset] = np.float32(17.0 + group + dim * 0.01)
                        data[v_offset] = np.float32(-19.0 - group - dim * 0.01)
    return data


def reshape_tile_token_to_dim_group(tile: np.ndarray) -> np.ndarray:
    out = np.zeros(RESHAPED_TILE_DWORDS, dtype=np.float32)
    for dim_group in range(DIM_GROUPS):
        for token in range(TOKENS_PER_TILE):
            for lane in range(GROUP_DWORDS):
                dim = dim_group * GROUP_DWORDS + lane
                out[(dim_group * TOKENS_PER_TILE + token) * GROUP_DWORDS + lane] = tile[token * HEAD_DIM + dim]
    return out


def _online_attention_head(context_len: int, query: np.ndarray, group_cache: np.ndarray) -> np.ndarray:
    num_tiles = num_tiles_for_context(context_len)
    k_plane_dwords = num_tiles * PLANE_TILE_DWORDS
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    out = np.zeros(HEAD_DIM, dtype=np.float32)
    running_max = np.float32(-np.inf)
    running_sum = np.float32(0.0)

    for tile in range(num_tiles):
        valid = TOKENS_PER_TILE if tile < num_tiles - 1 else context_len - tile * TOKENS_PER_TILE
        k_raw = group_cache[tile * PLANE_TILE_DWORDS:(tile + 1) * PLANE_TILE_DWORDS]
        v_raw = group_cache[k_plane_dwords + tile * PLANE_TILE_DWORDS:k_plane_dwords + (tile + 1) * PLANE_TILE_DWORDS]
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


def attention_reference(context_len: int, query: np.ndarray, kv_cache: np.ndarray) -> np.ndarray:
    group_stride = kv_group_stride_dwords(context_len)
    out = np.zeros(OUTPUT_DWORDS, dtype=np.float32)
    for group in range(NUM_KV_GROUPS):
        group_cache = kv_cache[group * group_stride:(group + 1) * group_stride]
        for row in range(Q_HEADS_PER_GROUP):
            head = group * Q_HEADS_PER_GROUP + row
            start = head * HEAD_DIM
            out[start:start + HEAD_DIM] = _online_attention_head(
                context_len,
                query[start:start + HEAD_DIM],
                group_cache,
            )
    return out


def make_case(context_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = make_query()
    kv_cache = make_kv_cache(context_len)
    expected = attention_reference(context_len, query, kv_cache)
    return query, kv_cache, expected


if __name__ == "__main__":
    q, cache, out = make_case(31)
    print(f"query={q.shape} cache={cache.shape} out0={out[:4]}")
