"""CPU reference for exp39 projected current-write full attention."""

from math import ceil

import numpy as np
from ml_dtypes import bfloat16

NUM_KV_GROUPS = 8
Q_HEADS_PER_GROUP = 4
NUM_Q_HEADS = NUM_KV_GROUPS * Q_HEADS_PER_GROUP
HEAD_DIM = 128
HIDDEN_DIM = 4096
TOKENS_PER_TILE = 16
DIM_GROUPS = 32
GROUP_DWORDS = 4
PROJECTION_TAPS = 32

QUERY_GROUP_DWORDS = Q_HEADS_PER_GROUP * HEAD_DIM
QUERY_DWORDS = NUM_Q_HEADS * HEAD_DIM
PLANE_TILE_DWORDS = TOKENS_PER_TILE * HEAD_DIM
RESHAPED_TILE_DWORDS = DIM_GROUPS * TOKENS_PER_TILE * GROUP_DWORDS
OUTPUT_DWORDS = QUERY_DWORDS


def num_tiles_for_context(context_len: int) -> int:
    return ceil(context_len / TOKENS_PER_TILE)


def kv_group_stride_dwords(context_len: int) -> int:
    return 2 * num_tiles_for_context(context_len) * PLANE_TILE_DWORDS


def current_offsets(context_len: int, group: int) -> tuple[int, int]:
    tile = (context_len - 1) // TOKENS_PER_TILE
    token = (context_len - 1) % TOKENS_PER_TILE
    group_base = group * kv_group_stride_dwords(context_len)
    k_plane_dwords = num_tiles_for_context(context_len) * PLANE_TILE_DWORDS
    k_offset = group_base + tile * PLANE_TILE_DWORDS + token * HEAD_DIM
    v_offset = group_base + k_plane_dwords + tile * PLANE_TILE_DWORDS + token * HEAD_DIM
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


def make_hidden(seed: int = 39) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.08, 0.08, HIDDEN_DIM).astype(bfloat16)


def _projected_value(hidden: np.ndarray, group: int, row: int, dim: int, phase: int) -> np.float32:
    hidden_f32 = hidden.astype(np.float32)
    base = (group * 197 + row * 53 + dim * 17 + phase * 31) & (HIDDEN_DIM - 1)
    acc = np.float32(0.0)
    for tap in range(PROJECTION_TAPS):
        idx = (base + tap * 113) & (HIDDEN_DIM - 1)
        raw = (phase * 7 + group * 5 + row * 3 + dim + tap * 11) % 17
        coeff = np.float32((raw - 8) * 0.001953125)
        acc = np.float32(acc + hidden_f32[idx] * coeff)
    return acc


def project_query_from_hidden(hidden: np.ndarray) -> np.ndarray:
    query = np.zeros(QUERY_DWORDS, dtype=np.float32)
    for group in range(NUM_KV_GROUPS):
        for row in range(Q_HEADS_PER_GROUP):
            head = group * Q_HEADS_PER_GROUP + row
            for dim in range(HEAD_DIM):
                query[head * HEAD_DIM + dim] = _projected_value(hidden, group, row, dim, 0)
    return query


def project_current_from_hidden(hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current_k = np.zeros((NUM_KV_GROUPS, HEAD_DIM), dtype=np.float32)
    current_v = np.zeros((NUM_KV_GROUPS, HEAD_DIM), dtype=np.float32)
    for group in range(NUM_KV_GROUPS):
        for dim in range(HEAD_DIM):
            current_k[group, dim] = _projected_value(hidden, group, 0, dim, 1)
            current_v[group, dim] = _projected_value(hidden, group, 0, dim, 2)
    return current_k, current_v


def _plane_value(group: int, global_t: int, dim: int, plane: str) -> np.float32:
    if plane == "v":
        raw = ((group * 11 + global_t * 5 + dim * 3) % 31) - 15
        return np.float32(raw * 0.0125)
    raw = ((group * 13 + global_t * 7 + dim * 2) % 37) - 18
    return np.float32(raw * 0.00875)


def make_kv_cache_without_current(context_len: int) -> np.ndarray:
    num_tiles = num_tiles_for_context(context_len)
    k_plane_dwords = num_tiles * PLANE_TILE_DWORDS
    group_stride = 2 * k_plane_dwords
    current_token = context_len - 1
    data = np.zeros(NUM_KV_GROUPS * group_stride, dtype=np.float32)
    for group in range(NUM_KV_GROUPS):
        group_base = group * group_stride
        for tile in range(num_tiles):
            for token in range(TOKENS_PER_TILE):
                global_t = tile * TOKENS_PER_TILE + token
                for dim in range(HEAD_DIM):
                    k_offset = group_base + tile * PLANE_TILE_DWORDS + token * HEAD_DIM + dim
                    v_offset = group_base + k_plane_dwords + tile * PLANE_TILE_DWORDS + token * HEAD_DIM + dim
                    if global_t < context_len and global_t != current_token:
                        data[k_offset] = _plane_value(group, global_t, dim, "k")
                        data[v_offset] = _plane_value(group, global_t, dim, "v")
                    elif global_t >= context_len:
                        data[k_offset] = np.float32(17.0 + group + dim * 0.01)
                        data[v_offset] = np.float32(-19.0 - group - dim * 0.01)
    return data


def apply_current_to_cache(
    context_len: int,
    kv_cache: np.ndarray,
    current_k: np.ndarray,
    current_v: np.ndarray,
) -> np.ndarray:
    out = kv_cache.copy()
    for group in range(NUM_KV_GROUPS):
        k_offset, v_offset = current_offsets(context_len, group)
        out[k_offset:k_offset + HEAD_DIM] = current_k[group]
        out[v_offset:v_offset + HEAD_DIM] = current_v[group]
    return out


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


def make_case(context_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hidden = make_hidden()
    query = project_query_from_hidden(hidden)
    current_k, current_v = project_current_from_hidden(hidden)
    cache_before = make_kv_cache_without_current(context_len)
    cache_after = apply_current_to_cache(context_len, cache_before, current_k, current_v)
    expected = attention_reference(context_len, query, cache_after)
    return hidden, query, current_k, current_v, cache_before, cache_after, expected


if __name__ == "__main__":
    hidden_data, q, k, v, before, after, out = make_case(31)
    print(f"hidden={hidden_data.shape} q0={q[:4]} k0={k[0, :4]} v0={v[0, :4]} out0={out[:4]}")
