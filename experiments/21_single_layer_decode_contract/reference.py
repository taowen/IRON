"""CPU reference for exp21 single-layer decode contract."""

from math import ceil

import numpy as np

NUM_KV_HEADS_PER_GROUP = 4
HEAD_DIM = 32
TOKENS_PER_TILE = 16
TOKEN_DWORDS = NUM_KV_HEADS_PER_GROUP * HEAD_DIM
PLANE_TILE_DWORDS = TOKEN_DWORDS * TOKENS_PER_TILE


def approx_exp(x: np.ndarray | np.float32) -> np.ndarray | np.float32:
    clipped = np.minimum(np.maximum(x, np.float32(-16.0)), np.float32(0.0)).astype(np.float32)
    y = (np.float32(1.0) + clipped * np.float32(1.0 / 64.0)).astype(np.float32)
    for _ in range(6):
        y = (y * y).astype(np.float32)
    return y


def approx_sigmoid(x: np.ndarray | np.float32) -> np.ndarray | np.float32:
    arr = np.asarray(x, dtype=np.float32)
    positive = arr >= np.float32(0.0)
    out = np.empty_like(arr, dtype=np.float32)
    neg_exp = approx_exp(-arr[positive]).astype(np.float32)
    out[positive] = np.float32(1.0) / (np.float32(1.0) + neg_exp)
    pos_exp = approx_exp(arr[~positive]).astype(np.float32)
    out[~positive] = pos_exp / (np.float32(1.0) + pos_exp)
    if np.isscalar(x):
        return np.float32(out.item())
    return out


def _plane_value(global_t: int, local_head: int, dim: int, plane: str) -> np.float32:
    group_offset = 0 if plane in ("k03", "v03") else 4
    if plane in ("v03", "v47"):
        raw = ((global_t * 4 + local_head + group_offset + dim) % 9) - 4
        return np.float32(raw * 0.125)
    raw = ((global_t * 4 + local_head + group_offset + dim * 2) % 11) - 5
    return np.float32(raw * 0.0625)


def make_history_plane_without_current(L: int, plane: str) -> np.ndarray:
    num_tiles = ceil(L / TOKENS_PER_TILE)
    data = np.zeros(num_tiles * PLANE_TILE_DWORDS, dtype=np.float32)
    current_token = L - 1
    for tile_idx in range(num_tiles):
        tile_base = tile_idx * PLANE_TILE_DWORDS
        for t in range(TOKENS_PER_TILE):
            global_t = tile_idx * TOKENS_PER_TILE + t
            if global_t >= L or global_t == current_token:
                continue
            for h in range(NUM_KV_HEADS_PER_GROUP):
                for d in range(HEAD_DIM):
                    idx = tile_base + t * TOKEN_DWORDS + h * HEAD_DIM + d
                    data[idx] = _plane_value(global_t, h, d, plane)
    return data


def make_current_input(L: int) -> np.ndarray:
    token = L - 1
    parts = []
    for plane in ("k03", "v03", "k47", "v47"):
        data = np.zeros(TOKEN_DWORDS, dtype=np.float32)
        for h in range(NUM_KV_HEADS_PER_GROUP):
            for d in range(HEAD_DIM):
                data[h * HEAD_DIM + d] = _plane_value(token, h, d, plane)
        parts.append(data)
    return np.concatenate(parts)


def make_kv_cache_without_current(L: int) -> np.ndarray:
    return np.concatenate(
        [
            make_history_plane_without_current(L, "k03"),
            make_history_plane_without_current(L, "v03"),
            make_history_plane_without_current(L, "k47"),
            make_history_plane_without_current(L, "v47"),
        ]
    )


def apply_current_to_cache(L: int, kv_cache: np.ndarray, current: np.ndarray) -> np.ndarray:
    out = kv_cache.copy()
    num_tiles = ceil(L / TOKENS_PER_TILE)
    plane_dwords = num_tiles * PLANE_TILE_DWORDS
    current_token_offset = (L - 1) * TOKEN_DWORDS
    for plane_idx in range(4):
        src = plane_idx * TOKEN_DWORDS
        dst = plane_idx * plane_dwords + current_token_offset
        out[dst:dst + TOKEN_DWORDS] = current[src:src + TOKEN_DWORDS]
    return out


def derive_query(current_k: np.ndarray, current_v: np.ndarray) -> np.ndarray:
    query = np.zeros(TOKEN_DWORDS, dtype=np.float32)
    for h in range(NUM_KV_HEADS_PER_GROUP):
        base = h * HEAD_DIM
        for d in range(HEAD_DIM):
            k = current_k[base + d]
            v = current_v[base + ((d + 3) & 31)]
            neighbor = current_k[base + ((d + 1) & 31)]
            lane = np.float32((d % 7) - 3)
            query[base + d] = (
                k * np.float32(0.625)
                + v * np.float32(0.1875)
                - neighbor * np.float32(0.0625)
                + lane * np.float32(0.00390625) * np.float32(h + 1)
            )
    return query


def online_attention_reference_from_cache(L: int, query: np.ndarray, k_plane: np.ndarray, v_plane: np.ndarray) -> np.ndarray:
    num_tiles = ceil(L / TOKENS_PER_TILE)
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    out = np.zeros(TOKEN_DWORDS, dtype=np.float32)
    running_max = np.full((NUM_KV_HEADS_PER_GROUP,), -np.inf, dtype=np.float32)
    running_sum = np.zeros((NUM_KV_HEADS_PER_GROUP,), dtype=np.float32)

    for tile_idx in range(num_tiles):
        valid = TOKENS_PER_TILE if tile_idx < num_tiles - 1 else L - tile_idx * TOKENS_PER_TILE
        tile_base = tile_idx * PLANE_TILE_DWORDS
        for h in range(NUM_KV_HEADS_PER_GROUP):
            scores = np.zeros(valid, dtype=np.float32)
            q_head = query[h * HEAD_DIM:(h + 1) * HEAD_DIM]
            for t in range(valid):
                head_base = tile_base + t * TOKEN_DWORDS + h * HEAD_DIM
                scores[t] = np.sum(q_head * k_plane[head_base:head_base + HEAD_DIM]) * scale

            local_max = np.max(scores)
            new_max = max(running_max[h], local_max)
            old_scale = np.float32(0.0) if running_sum[h] == 0 else approx_exp(
                np.float32(running_max[h] - new_max)
            )
            weights = approx_exp(scores - new_max).astype(np.float32)
            running_sum[h] = running_sum[h] * old_scale + np.sum(weights)
            dst = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            out[dst] *= old_scale
            for t in range(valid):
                head_base = tile_base + t * TOKEN_DWORDS + h * HEAD_DIM
                out[dst] += weights[t] * v_plane[head_base:head_base + HEAD_DIM]
            running_max[h] = new_max

    for h in range(NUM_KV_HEADS_PER_GROUP):
        dst = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        out[dst] /= running_sum[h]
    return out


def layer_epilogue_reference(query: np.ndarray, attention: np.ndarray) -> np.ndarray:
    out = np.zeros_like(attention)
    for h in range(NUM_KV_HEADS_PER_GROUP):
        base = h * HEAD_DIM
        for d in range(HEAD_DIM):
            mixed = (
                attention[base + d] * np.float32(0.75)
                + attention[base + ((d + 1) & 31)] * np.float32(0.125)
                - attention[base + ((d + 7) & 31)] * np.float32(0.0625)
                + query[base + d] * np.float32(0.25)
            )
            gate = mixed * approx_sigmoid(mixed * np.float32(0.5))
            out[base + d] = mixed + gate * np.float32(0.1)
    return out.astype(np.float32)


def layer_reference_after_current_write(L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = make_current_input(L)
    before = make_kv_cache_without_current(L)
    after = apply_current_to_cache(L, before, current)
    num_tiles = ceil(L / TOKENS_PER_TILE)
    plane_dwords = num_tiles * PLANE_TILE_DWORDS
    k03 = after[0:plane_dwords]
    v03 = after[plane_dwords:2 * plane_dwords]
    k47 = after[2 * plane_dwords:3 * plane_dwords]
    v47 = after[3 * plane_dwords:4 * plane_dwords]

    query03 = derive_query(current[0:TOKEN_DWORDS], current[TOKEN_DWORDS:2 * TOKEN_DWORDS])
    query47 = derive_query(current[2 * TOKEN_DWORDS:3 * TOKEN_DWORDS], current[3 * TOKEN_DWORDS:4 * TOKEN_DWORDS])
    attention03 = online_attention_reference_from_cache(L, query03, k03, v03)
    attention47 = online_attention_reference_from_cache(L, query47, k47, v47)
    out03 = layer_epilogue_reference(query03, attention03)
    out47 = layer_epilogue_reference(query47, attention47)
    return current, before, np.concatenate([out03, out47]).astype(np.float32)


if __name__ == "__main__":
    for length in [1, 15, 16, 17, 31, 32, 79]:
        _, _, ref = layer_reference_after_current_write(length)
        print(f"L={length:3d} out[0:4]={ref[:4]} out[128:132]={ref[128:132]}")
