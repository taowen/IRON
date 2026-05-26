"""CPU reference for KV scan attention demo."""

import numpy as np
from math import ceil

NUM_HEADS = 4
HEAD_DIM = 32
TOKENS_PER_TILE = 16
HALF_TILE = NUM_HEADS * HEAD_DIM * TOKENS_PER_TILE  # 2048 int32 per half


def make_q() -> np.ndarray:
    """Q vector: Q[h*HEAD_DIM + d] = h + 1. Shape: [128] int32."""
    q = np.zeros(NUM_HEADS * HEAD_DIM, dtype=np.int32)
    for h in range(NUM_HEADS):
        for d in range(HEAD_DIM):
            q[h * HEAD_DIM + d] = h + 1
    return q


def make_kv_input(L: int) -> np.ndarray:
    """Generate KV tiles for effective length L.

    Returns flat array of num_tiles * 4096 int32.
    Each tile: [K half: 2048 int32][V half: 2048 int32]
    K[t][h*HEAD_DIM+d] = ((global_t * 4 + h) % 7) - 3
    V[t][h*HEAD_DIM+d] = ((global_t * 4 + h + d) % 5) - 2
    """
    num_tiles = ceil(L / TOKENS_PER_TILE)
    total_elems = num_tiles * 4096
    data = np.zeros(total_elems, dtype=np.int32)

    for tile_idx in range(num_tiles):
        tile_offset = tile_idx * 4096
        for t in range(TOKENS_PER_TILE):
            global_t = tile_idx * TOKENS_PER_TILE + t
            for h in range(NUM_HEADS):
                for d in range(HEAD_DIM):
                    idx = t * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d
                    # K half (first 2048)
                    data[tile_offset + idx] = ((global_t * 4 + h) % 7) - 3
                    # V half (second 2048)
                    data[tile_offset + HALF_TILE + idx] = ((global_t * 4 + h + d) % 5) - 2
    return data


def attention_reference(L: int) -> np.ndarray:
    """Compute expected attention output for effective length L.

    out[h*HEAD_DIM+d] = sum over valid tokens t of:
        dot(Q[h], K[t,h]) * V[t,h,d]

    Returns shape [128] int32.
    """
    num_tiles = ceil(L / TOKENS_PER_TILE)
    out = np.zeros(NUM_HEADS * HEAD_DIM, dtype=np.int64)

    for tile_idx in range(num_tiles):
        if tile_idx < num_tiles - 1:
            valid = TOKENS_PER_TILE
        else:
            valid = L - tile_idx * TOKENS_PER_TILE

        for t in range(valid):
            global_t = tile_idx * TOKENS_PER_TILE + t
            for h in range(NUM_HEADS):
                # Compute score = dot(Q[h], K[t,h])
                score = np.int64(0)
                q_val = h + 1
                for d in range(HEAD_DIM):
                    k_val = ((global_t * 4 + h) % 7) - 3
                    score += np.int64(q_val) * np.int64(k_val)

                # Accumulate: out[h,d] += score * V[t,h,d]
                for d in range(HEAD_DIM):
                    v_val = ((global_t * 4 + h + d) % 5) - 2
                    out[h * HEAD_DIM + d] += score * np.int64(v_val)

    return out.astype(np.int32)


if __name__ == "__main__":
    for L in [1, 15, 16, 17, 31, 32, 79]:
        num_tiles = ceil(L / TOKENS_PER_TILE)
        last_valid = L - (num_tiles - 1) * TOKENS_PER_TILE
        out = attention_reference(L)
        print(f"L={L:3d}  tiles={num_tiles}  last_valid={last_valid:2d}  "
              f"out[0:4]={out[:4]}  sum={out.sum()}")
