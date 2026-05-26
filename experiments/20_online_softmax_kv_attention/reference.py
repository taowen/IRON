"""CPU reference for exp20 online-softmax attention."""

from math import ceil

import numpy as np

NUM_KV_HEADS_PER_GROUP = 4
HEAD_DIM = 32
TOKENS_PER_TILE = 16
TOKEN_DWORDS = NUM_KV_HEADS_PER_GROUP * HEAD_DIM
PLANE_TILE_DWORDS = TOKEN_DWORDS * TOKENS_PER_TILE


def query_value(head_offset: int, local_head: int, dim: int) -> np.float32:
    lane = dim % 7 - 3
    return np.float32(0.02 * (head_offset + local_head + 1) * lane)


def approx_exp(x: np.ndarray | np.float32) -> np.ndarray | np.float32:
    clipped = np.minimum(np.maximum(x, np.float32(-16.0)), np.float32(0.0)).astype(np.float32)
    y = (np.float32(1.0) + clipped * np.float32(1.0 / 64.0)).astype(np.float32)
    for _ in range(6):
        y = (y * y).astype(np.float32)
    return y


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


def attention_reference_from_cache(L: int, kv_cache: np.ndarray) -> np.ndarray:
    num_tiles = ceil(L / TOKENS_PER_TILE)
    plane_dwords = num_tiles * PLANE_TILE_DWORDS
    planes = {
        "k03": kv_cache[0:plane_dwords],
        "v03": kv_cache[plane_dwords:2 * plane_dwords],
        "k47": kv_cache[2 * plane_dwords:3 * plane_dwords],
        "v47": kv_cache[3 * plane_dwords:4 * plane_dwords],
    }
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    out = np.zeros(2 * TOKEN_DWORDS, dtype=np.float32)

    for group in range(2):
        group_offset = group * 4
        k_plane = planes["k03" if group == 0 else "k47"]
        v_plane = planes["v03" if group == 0 else "v47"]
        out_base = group * TOKEN_DWORDS
        for h in range(NUM_KV_HEADS_PER_GROUP):
            scores = []
            values = []
            for tile_idx in range(num_tiles):
                valid = TOKENS_PER_TILE if tile_idx < num_tiles - 1 else L - tile_idx * TOKENS_PER_TILE
                tile_base = tile_idx * PLANE_TILE_DWORDS
                for t in range(valid):
                    token_base = tile_base + t * TOKEN_DWORDS
                    head_base = token_base + h * HEAD_DIM
                    score = np.float32(0.0)
                    for d in range(HEAD_DIM):
                        score += query_value(group_offset, h, d) * k_plane[head_base + d]
                    scores.append(score * scale)
                    values.append(v_plane[head_base:head_base + HEAD_DIM].copy())

            score_arr = np.asarray(scores, dtype=np.float32)
            max_score = np.max(score_arr)
            weights = approx_exp(score_arr - max_score).astype(np.float32)
            weights /= np.sum(weights)
            value_arr = np.stack(values).astype(np.float32)
            out[out_base + h * HEAD_DIM:out_base + (h + 1) * HEAD_DIM] = weights @ value_arr
    return out


def online_softmax_reference_from_cache(L: int, kv_cache: np.ndarray) -> np.ndarray:
    """Reference using the same tiled online update as the NPU kernel."""
    num_tiles = ceil(L / TOKENS_PER_TILE)
    plane_dwords = num_tiles * PLANE_TILE_DWORDS
    planes = {
        "k03": kv_cache[0:plane_dwords],
        "v03": kv_cache[plane_dwords:2 * plane_dwords],
        "k47": kv_cache[2 * plane_dwords:3 * plane_dwords],
        "v47": kv_cache[3 * plane_dwords:4 * plane_dwords],
    }
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    out = np.zeros(2 * TOKEN_DWORDS, dtype=np.float32)
    running_max = np.full((2, NUM_KV_HEADS_PER_GROUP), -np.inf, dtype=np.float32)
    running_sum = np.zeros((2, NUM_KV_HEADS_PER_GROUP), dtype=np.float32)

    for tile_idx in range(num_tiles):
        valid = TOKENS_PER_TILE if tile_idx < num_tiles - 1 else L - tile_idx * TOKENS_PER_TILE
        tile_base = tile_idx * PLANE_TILE_DWORDS
        for group in range(2):
            group_offset = group * 4
            k_plane = planes["k03" if group == 0 else "k47"]
            v_plane = planes["v03" if group == 0 else "v47"]
            out_base = group * TOKEN_DWORDS
            for h in range(NUM_KV_HEADS_PER_GROUP):
                scores = np.zeros(valid, dtype=np.float32)
                for t in range(valid):
                    head_base = tile_base + t * TOKEN_DWORDS + h * HEAD_DIM
                    score = np.float32(0.0)
                    for d in range(HEAD_DIM):
                        score += query_value(group_offset, h, d) * k_plane[head_base + d]
                    scores[t] = score * scale

                local_max = np.max(scores)
                new_max = max(running_max[group, h], local_max)
                old_scale = np.float32(0.0) if running_sum[group, h] == 0 else approx_exp(
                    np.float32(running_max[group, h] - new_max)
                )
                weights = approx_exp(scores - new_max).astype(np.float32)
                running_sum[group, h] = running_sum[group, h] * old_scale + np.sum(weights)
                dst = slice(out_base + h * HEAD_DIM, out_base + (h + 1) * HEAD_DIM)
                out[dst] *= old_scale
                for t in range(valid):
                    head_base = tile_base + t * TOKEN_DWORDS + h * HEAD_DIM
                    out[dst] += weights[t] * v_plane[head_base:head_base + HEAD_DIM]
                running_max[group, h] = new_max

    for group in range(2):
        out_base = group * TOKEN_DWORDS
        for h in range(NUM_KV_HEADS_PER_GROUP):
            dst = slice(out_base + h * HEAD_DIM, out_base + (h + 1) * HEAD_DIM)
            out[dst] /= running_sum[group, h]
    return out


def attention_reference_after_current_write(L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = make_current_input(L)
    before = make_kv_cache_without_current(L)
    after = apply_current_to_cache(L, before, current)
    expected = online_softmax_reference_from_cache(L, after)
    direct = attention_reference_from_cache(L, after)
    if not np.allclose(expected, direct, rtol=1e-5, atol=1e-6):
        raise RuntimeError("online softmax reference diverged from direct softmax")
    return current, before, expected


if __name__ == "__main__":
    for length in [1, 15, 16, 17, 31, 32, 79]:
        _, _, ref = attention_reference_after_current_write(length)
        print(f"L={length:3d} out[0:4]={ref[:4]} out[128:132]={ref[128:132]}")
