"""CPU reference for exp25 edge/KV BD-ring checksum skeleton."""

from math import ceil

import numpy as np

TOKENS_PER_TILE = 16
PLANE_TILE_DWORDS = 0x1000
HALF_TILE_DWORDS = PLANE_TILE_DWORDS // 2
NUM_PLANES = 4
CHECKSUM_DWORDS = 4


def make_plane(context_len: int, plane_idx: int) -> np.ndarray:
    num_tiles = ceil(context_len / TOKENS_PER_TILE)
    data = np.zeros(num_tiles * PLANE_TILE_DWORDS, dtype=np.int32)
    for tile in range(num_tiles):
        for i in range(PLANE_TILE_DWORDS):
            global_idx = tile * PLANE_TILE_DWORDS + i
            token = i // 256
            valid_token = tile * TOKENS_PER_TILE + token
            if valid_token >= context_len:
                continue
            data[global_idx] = ((plane_idx + 1) * 17 + valid_token * 3 + i) % 97 - 48
    return data


def make_kv_cache(context_len: int) -> np.ndarray:
    return np.concatenate(
        [make_plane(context_len, plane) for plane in range(NUM_PLANES)]
    )


def checksum_plane(plane: np.ndarray) -> np.ndarray:
    out = np.zeros(CHECKSUM_DWORDS, dtype=np.int32)
    first_written = False
    chunks = len(plane) // HALF_TILE_DWORDS
    for chunk_idx in range(chunks):
        start = chunk_idx * HALF_TILE_DWORDS
        end = start + HALF_TILE_DWORDS
        half = plane[start:end]
        out[0] += int(half.sum(dtype=np.int64)) + chunk_idx
        if not first_written:
            out[2] = int(half[0])
            first_written = True
        out[3] = int(half[-1])
        out[1] += HALF_TILE_DWORDS
    return out


def expected_output(context_len: int) -> np.ndarray:
    cache = make_kv_cache(context_len)
    plane_dwords = ceil(context_len / TOKENS_PER_TILE) * PLANE_TILE_DWORDS
    planes = [
        cache[0:plane_dwords],
        cache[2 * plane_dwords : 3 * plane_dwords],
        cache[plane_dwords : 2 * plane_dwords],
        cache[3 * plane_dwords : 4 * plane_dwords],
    ]
    return np.concatenate([checksum_plane(plane) for plane in planes])


if __name__ == "__main__":
    for length in [17, 31, 32, 128]:
        print(length, expected_output(length).reshape(NUM_PLANES, CHECKSUM_DWORDS))
