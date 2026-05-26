"""CPU reference for current-token KV update followed by four-plane scan."""

from math import ceil

import numpy as np

NUM_KV_HEADS_PER_GROUP = 4
HEAD_DIM = 32
TOKENS_PER_TILE = 16
TOKEN_DWORDS = NUM_KV_HEADS_PER_GROUP * HEAD_DIM
PLANE_TILE_DWORDS = TOKEN_DWORDS * TOKENS_PER_TILE


def _plane_value(global_t: int, local_head: int, dim: int, plane: str) -> int:
    group_offset = 0 if plane in ("k03", "v03") else 4
    if plane in ("v03", "v47"):
        return ((global_t * 4 + local_head + group_offset + dim) % 5) - 2
    return ((global_t * 4 + local_head + group_offset) % 7) - 3


def make_history_plane_without_current(L: int, plane: str) -> np.ndarray:
    """Create a rounded cache plane with token L-1 intentionally empty.

    The NPU sequence must write the current token before scanning history.  This
    makes a missing current-write visible in the final attention output.
    """
    num_tiles = ceil(L / TOKENS_PER_TILE)
    data = np.zeros(num_tiles * PLANE_TILE_DWORDS, dtype=np.int32)
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
    """Pack current token as k03 | v03 | k47 | v47."""
    token = L - 1
    parts = []
    for plane in ("k03", "v03", "k47", "v47"):
        data = np.zeros(TOKEN_DWORDS, dtype=np.int32)
        for h in range(NUM_KV_HEADS_PER_GROUP):
            for d in range(HEAD_DIM):
                data[h * HEAD_DIM + d] = _plane_value(token, h, d, plane)
        parts.append(data)
    return np.concatenate(parts)


def make_kv_cache_without_current(L: int) -> np.ndarray:
    """Pack cache as k03 | v03 | k47 | v47, matching generate.py."""
    return np.concatenate(
        [
            make_history_plane_without_current(L, "k03"),
            make_history_plane_without_current(L, "v03"),
            make_history_plane_without_current(L, "k47"),
            make_history_plane_without_current(L, "v47"),
        ]
    )


def apply_current_to_cache(L: int, kv_cache: np.ndarray, current: np.ndarray) -> np.ndarray:
    """CPU equivalent of the NPU current-token cache write."""
    out = kv_cache.copy()
    num_tiles = ceil(L / TOKENS_PER_TILE)
    plane_dwords = num_tiles * PLANE_TILE_DWORDS
    current_token_offset = (L - 1) * TOKEN_DWORDS
    plane_order = [0, 1, 2, 3]
    for current_plane_idx, cache_plane_idx in enumerate(plane_order):
        src = current_plane_idx * TOKEN_DWORDS
        dst = cache_plane_idx * plane_dwords + current_token_offset
        out[dst:dst + TOKEN_DWORDS] = current[src:src + TOKEN_DWORDS]
    return out


def attention_reference_from_cache(L: int, kv_cache: np.ndarray) -> np.ndarray:
    """Compute expected attention output after current has been written."""
    num_tiles = ceil(L / TOKENS_PER_TILE)
    plane_dwords = num_tiles * PLANE_TILE_DWORDS
    planes = {
        "k03": kv_cache[0:plane_dwords],
        "v03": kv_cache[plane_dwords:2 * plane_dwords],
        "k47": kv_cache[2 * plane_dwords:3 * plane_dwords],
        "v47": kv_cache[3 * plane_dwords:4 * plane_dwords],
    }

    out = np.zeros(2 * TOKEN_DWORDS, dtype=np.int64)
    for group in range(2):
        group_offset = group * 4
        k_plane = planes["k03" if group == 0 else "k47"]
        v_plane = planes["v03" if group == 0 else "v47"]
        out_base = group * TOKEN_DWORDS
        for tile_idx in range(num_tiles):
            valid = TOKENS_PER_TILE if tile_idx < num_tiles - 1 else L - tile_idx * TOKENS_PER_TILE
            tile_base = tile_idx * PLANE_TILE_DWORDS
            for t in range(valid):
                token_base = tile_base + t * TOKEN_DWORDS
                for h in range(NUM_KV_HEADS_PER_GROUP):
                    q_val = group_offset + h + 1
                    score = np.int64(0)
                    head_base = token_base + h * HEAD_DIM
                    for d in range(HEAD_DIM):
                        score += np.int64(q_val) * np.int64(k_plane[head_base + d])
                    for d in range(HEAD_DIM):
                        out[out_base + h * HEAD_DIM + d] += score * np.int64(v_plane[head_base + d])
    return out.astype(np.int32)


def attention_reference_after_current_write(L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = make_current_input(L)
    before = make_kv_cache_without_current(L)
    after = apply_current_to_cache(L, before, current)
    expected = attention_reference_from_cache(L, after)
    return current, before, expected


if __name__ == "__main__":
    for L in [1, 15, 16, 17, 31, 32, 79]:
        current, cache, expected = attention_reference_after_current_write(L)
        print(
            f"L={L:3d} current_sum={int(current.sum()):5d} "
            f"cache_before_sum={int(cache.sum()):5d} "
            f"w0[0:4]={expected[:4]} w1[0:4]={expected[128:132]}"
        )
