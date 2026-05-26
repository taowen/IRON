"""CPU reference for 4-plane GQA attention demo.

4 planes: k03, v03, k47, v47
2 head groups: group 03 (heads 0-3), group 47 (heads 4-7)
Each plane tile: [16 tokens][4 KV heads][HEAD_DIM] int32
"""

import numpy as np
from math import ceil

NUM_KV_HEADS_PER_GROUP = 4
HEAD_DIM = 32
TOKENS_PER_TILE = 16
PLANE_TILE_DWORDS = NUM_KV_HEADS_PER_GROUP * HEAD_DIM * TOKENS_PER_TILE  # 2048


def make_plane_input(L: int, plane: str) -> np.ndarray:
    """Generate one plane's tile data for effective length L.

    plane: 'k03', 'v03', 'k47', 'v47'
    Returns flat array of num_tiles * 2048 int32.
    Layout per tile: [16 tokens][4 heads][32 dim]
    """
    num_tiles = ceil(L / TOKENS_PER_TILE)
    total_elems = num_tiles * PLANE_TILE_DWORDS
    data = np.zeros(total_elems, dtype=np.int32)

    group_offset = 0 if plane in ('k03', 'v03') else 4
    is_v = plane in ('v03', 'v47')

    for tile_idx in range(num_tiles):
        tile_offset = tile_idx * PLANE_TILE_DWORDS
        for t in range(TOKENS_PER_TILE):
            global_t = tile_idx * TOKENS_PER_TILE + t
            for h in range(NUM_KV_HEADS_PER_GROUP):
                for d in range(HEAD_DIM):
                    idx = t * NUM_KV_HEADS_PER_GROUP * HEAD_DIM + h * HEAD_DIM + d
                    if is_v:
                        data[tile_offset + idx] = ((global_t * 4 + h + group_offset + d) % 5) - 2
                    else:
                        data[tile_offset + idx] = ((global_t * 4 + h + group_offset) % 7) - 3
    return data


def attention_reference(L: int) -> np.ndarray:
    """Compute expected attention output for effective length L.

    Returns shape [256] int32: first 128 = worker0 (heads 0-3), last 128 = worker1 (heads 4-7).
    out[group*128 + h*HEAD_DIM + d] = sum over valid tokens t of:
        dot(Q[h], K[t,h]) * V[t,h,d]
    where Q[h] = group_offset + h + 1
    """
    num_tiles = ceil(L / TOKENS_PER_TILE)
    out = np.zeros(2 * NUM_KV_HEADS_PER_GROUP * HEAD_DIM, dtype=np.int64)

    for group in range(2):
        group_offset = group * 4
        out_base = group * NUM_KV_HEADS_PER_GROUP * HEAD_DIM

        for tile_idx in range(num_tiles):
            if tile_idx < num_tiles - 1:
                valid = TOKENS_PER_TILE
            else:
                valid = L - tile_idx * TOKENS_PER_TILE

            for t in range(valid):
                global_t = tile_idx * TOKENS_PER_TILE + t
                for h in range(NUM_KV_HEADS_PER_GROUP):
                    q_val = group_offset + h + 1
                    score = np.int64(0)
                    for d in range(HEAD_DIM):
                        k_val = ((global_t * 4 + h + group_offset) % 7) - 3
                        score += np.int64(q_val) * np.int64(k_val)

                    for d in range(HEAD_DIM):
                        v_val = ((global_t * 4 + h + group_offset + d) % 5) - 2
                        out[out_base + h * HEAD_DIM + d] += score * np.int64(v_val)

    return out.astype(np.int32)


if __name__ == "__main__":
    for L in [1, 15, 16, 17, 31, 32, 79]:
        num_tiles = ceil(L / TOKENS_PER_TILE)
        last_valid = L - (num_tiles - 1) * TOKENS_PER_TILE
        out = attention_reference(L)
        print(f"L={L:3d}  tiles={num_tiles}  last_valid={last_valid:2d}  "
              f"w0[0:4]={out[:4]}  w1[0:4]={out[128:132]}  sum={out.sum()}")
