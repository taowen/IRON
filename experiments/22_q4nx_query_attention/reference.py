"""CPU reference for exp22 Q4NX query projection + KV attention."""

from math import ceil

import numpy as np
from ml_dtypes import bfloat16

NUM_HEADS = 1
HEAD_DIM = 32
TOKENS_PER_TILE = 16
TOKEN_DWORDS = NUM_HEADS * HEAD_DIM
PLANE_TILE_DWORDS = TOKEN_DWORDS * TOKENS_PER_TILE
HIDDEN_DIM = 1024
Q_CHUNK = 256
Q_CHUNKS = HIDDEN_DIM // Q_CHUNK
GROUP_SIZE = 32
Q_ROWS = TOKEN_DWORDS
CHUNK_BYTES = Q_ROWS * (Q_CHUNK // GROUP_SIZE) * 2 * 2 + Q_ROWS * Q_CHUNK // 2
CHUNK_BF16 = CHUNK_BYTES // 2


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


def _to_bf16_f32(values: np.ndarray) -> np.ndarray:
    return values.astype(bfloat16).astype(np.float32)


def pack_q4nx_chunk(
    scales: np.ndarray,
    zeros: np.ndarray,
    int4_data: np.ndarray,
) -> np.ndarray:
    groups_per_row = Q_CHUNK // GROUP_SIZE
    packed = bytearray()
    packed += scales.astype(bfloat16).view(np.uint8).tobytes()
    packed += zeros.astype(bfloat16).view(np.uint8).tobytes()
    for row in range(Q_ROWS):
        for col in range(0, Q_CHUNK, 2):
            lo = int(int4_data[row, col]) & 0xF
            hi = int(int4_data[row, col + 1]) & 0xF
            packed.append(lo | (hi << 4))
    return np.frombuffer(bytes(packed), dtype=np.uint8)


def pack_q4nx_weight(scales: np.ndarray, zeros: np.ndarray, int4_data: np.ndarray) -> np.ndarray:
    groups_per_chunk = Q_CHUNK // GROUP_SIZE
    chunks = []
    for chunk in range(Q_CHUNKS):
        g0 = chunk * groups_per_chunk
        g1 = g0 + groups_per_chunk
        k0 = chunk * Q_CHUNK
        k1 = k0 + Q_CHUNK
        chunks.append(pack_q4nx_chunk(scales[:, g0:g1], zeros[:, g0:g1], int4_data[:, k0:k1]))
    return np.concatenate(chunks)


def q4nx_query_reference(packed_weight: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    groups_per_row = Q_CHUNK // GROUP_SIZE
    scale_bytes = Q_ROWS * groups_per_row * 2
    zero_bytes = scale_bytes
    data_offset = scale_bytes + zero_bytes
    query = np.zeros(Q_ROWS, dtype=np.float32)
    hidden_f32 = hidden.astype(np.float32)

    for chunk in range(Q_CHUNKS):
        chunk_data = packed_weight[chunk * CHUNK_BYTES:(chunk + 1) * CHUNK_BYTES]
        scales = np.frombuffer(chunk_data[:scale_bytes], dtype=bfloat16).reshape(Q_ROWS, groups_per_row)
        zeros = np.frombuffer(chunk_data[scale_bytes:data_offset], dtype=bfloat16).reshape(Q_ROWS, groups_per_row)
        int4_raw = chunk_data[data_offset:]
        act = hidden_f32[chunk * Q_CHUNK:(chunk + 1) * Q_CHUNK]

        for row in range(Q_ROWS):
            row_acc = np.float32(0.0)
            for group in range(groups_per_row):
                s = np.float32(scales[row, group])
                z = np.float32(zeros[row, group])
                start = group * GROUP_SIZE
                for lane in range(GROUP_SIZE):
                    col = start + lane
                    packed_byte = int(int4_raw[row * (Q_CHUNK // 2) + col // 2])
                    q = np.float32((packed_byte >> 4) & 0xF) if col & 1 else np.float32(packed_byte & 0xF)
                    dequant = _to_bf16_f32(np.array([(q - z) * s], dtype=np.float32))[0]
                    row_acc = np.float32(row_acc + dequant * act[col])
            query[row] = np.float32(query[row] + row_acc)
    return query


def make_case_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hidden = rng.uniform(-0.4, 0.4, HIDDEN_DIM).astype(bfloat16)
    scales = rng.uniform(0.001, 0.015, (Q_ROWS, HIDDEN_DIM // GROUP_SIZE)).astype(bfloat16)
    zeros = rng.uniform(6.0, 9.0, (Q_ROWS, HIDDEN_DIM // GROUP_SIZE)).astype(bfloat16)
    int4_data = rng.integers(0, 16, (Q_ROWS, HIDDEN_DIM), dtype=np.uint8)
    packed_weight = pack_q4nx_weight(scales, zeros, int4_data)
    query = q4nx_query_reference(packed_weight, hidden)
    return hidden, packed_weight, scales, zeros, query


def _plane_value(global_t: int, dim: int, plane: str) -> np.float32:
    if plane == "v":
        raw = ((global_t * 3 + dim) % 9) - 4
        return np.float32(raw * 0.125)
    raw = ((global_t * 5 + dim * 2) % 11) - 5
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
            for d in range(HEAD_DIM):
                data[tile_base + t * TOKEN_DWORDS + d] = _plane_value(global_t, d, plane)
    return data


def make_current_input(L: int) -> np.ndarray:
    token = L - 1
    k = np.zeros(TOKEN_DWORDS, dtype=np.float32)
    v = np.zeros(TOKEN_DWORDS, dtype=np.float32)
    for d in range(HEAD_DIM):
        k[d] = _plane_value(token, d, "k")
        v[d] = _plane_value(token, d, "v")
    return np.concatenate([k, v])


def make_kv_cache_without_current(L: int) -> np.ndarray:
    return np.concatenate(
        [
            make_history_plane_without_current(L, "k"),
            make_history_plane_without_current(L, "v"),
        ]
    )


def apply_current_to_cache(L: int, kv_cache: np.ndarray, current: np.ndarray) -> np.ndarray:
    out = kv_cache.copy()
    num_tiles = ceil(L / TOKENS_PER_TILE)
    plane_dwords = num_tiles * PLANE_TILE_DWORDS
    token_offset = (L - 1) * TOKEN_DWORDS
    out[token_offset:token_offset + TOKEN_DWORDS] = current[:TOKEN_DWORDS]
    dst = plane_dwords + token_offset
    out[dst:dst + TOKEN_DWORDS] = current[TOKEN_DWORDS:2 * TOKEN_DWORDS]
    return out


def online_attention_reference_from_cache(L: int, query: np.ndarray, k_plane: np.ndarray, v_plane: np.ndarray) -> np.ndarray:
    num_tiles = ceil(L / TOKENS_PER_TILE)
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    out = np.zeros(TOKEN_DWORDS, dtype=np.float32)
    running_max = np.float32(-np.inf)
    running_sum = np.float32(0.0)

    for tile_idx in range(num_tiles):
        valid = TOKENS_PER_TILE if tile_idx < num_tiles - 1 else L - tile_idx * TOKENS_PER_TILE
        tile_base = tile_idx * PLANE_TILE_DWORDS
        scores = np.zeros(valid, dtype=np.float32)
        for t in range(valid):
            head_base = tile_base + t * TOKEN_DWORDS
            scores[t] = np.sum(query * k_plane[head_base:head_base + HEAD_DIM]) * scale

        local_max = np.max(scores)
        new_max = max(running_max, local_max)
        old_scale = np.float32(0.0) if running_sum == 0 else approx_exp(np.float32(running_max - new_max))
        weights = approx_exp(scores - new_max).astype(np.float32)
        running_sum = running_sum * old_scale + np.sum(weights)
        out *= old_scale
        for t in range(valid):
            head_base = tile_base + t * TOKEN_DWORDS
            out += weights[t] * v_plane[head_base:head_base + HEAD_DIM]
        running_max = new_max

    out /= running_sum
    return out.astype(np.float32)


def layer_epilogue_reference(query: np.ndarray, attention: np.ndarray) -> np.ndarray:
    out = np.zeros_like(attention)
    for d in range(HEAD_DIM):
        mixed = (
            attention[d] * np.float32(0.75)
            + attention[(d + 1) & 31] * np.float32(0.125)
            - attention[(d + 7) & 31] * np.float32(0.0625)
            + query[d] * np.float32(0.25)
        )
        gate = mixed * approx_sigmoid(mixed * np.float32(0.5))
        out[d] = mixed + gate * np.float32(0.1)
    return out.astype(np.float32)


def layer_reference_after_current_write(
    L: int,
    hidden: np.ndarray,
    packed_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query = q4nx_query_reference(packed_weight, hidden)
    current = make_current_input(L)
    before = make_kv_cache_without_current(L)
    after = apply_current_to_cache(L, before, current)
    num_tiles = ceil(L / TOKENS_PER_TILE)
    plane_dwords = num_tiles * PLANE_TILE_DWORDS
    k_plane = after[:plane_dwords]
    v_plane = after[plane_dwords:2 * plane_dwords]
    attention = online_attention_reference_from_cache(L, query, k_plane, v_plane)
    out = layer_epilogue_reference(query, attention)
    return current, before, query, out


if __name__ == "__main__":
    hidden_data, weight_data, _, _, projected_query = make_case_data()
    print(f"hidden={hidden_data.shape} weight_bytes={weight_data.shape[0]} query[0:4]={projected_query[:4]}")
    for length in [1, 15, 16, 17, 31, 32, 79]:
        _, _, _, ref = layer_reference_after_current_write(length, hidden_data, weight_data)
        print(f"L={length:3d} out[0:4]={ref[:4]}")
