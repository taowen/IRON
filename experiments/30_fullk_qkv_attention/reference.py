"""CPU reference for exp30 full-K Q/K/V projection feeding attention."""

from math import ceil

import numpy as np
from ml_dtypes import bfloat16

NUM_HEADS = 1
HEAD_DIM = 32
TOKENS_PER_TILE = 16
TOKEN_DWORDS = NUM_HEADS * HEAD_DIM
PLANE_TILE_DWORDS = TOKEN_DWORDS * TOKENS_PER_TILE
KV_TILE_DWORDS = PLANE_TILE_DWORDS * 2

HIDDEN_DIM = 4096
Q4_K_CHUNK = 256
Q4_CHUNKS = HIDDEN_DIM // Q4_K_CHUNK
GROUP_SIZE = 32
Q4_ROWS = 32
PROJECTIONS = 3
GROUPS_PER_CHUNK = Q4_K_CHUNK // GROUP_SIZE
CHUNK_BYTES = Q4_ROWS * GROUPS_PER_CHUNK * 2 * 2 + Q4_ROWS * Q4_K_CHUNK // 2
ROWBLOCK_BYTES = Q4_CHUNKS * CHUNK_BYTES


def approx_exp(x: np.ndarray | np.float32) -> np.ndarray | np.float32:
    clipped = np.minimum(np.maximum(x, np.float32(-16.0)), np.float32(0.0)).astype(np.float32)
    y = (np.float32(1.0) + clipped * np.float32(1.0 / 64.0)).astype(np.float32)
    for _ in range(6):
        y = (y * y).astype(np.float32)
    return y


def _to_bf16_f32(values: np.ndarray) -> np.ndarray:
    return values.astype(bfloat16).astype(np.float32)


def pack_q4nx_chunk(scales: np.ndarray, zeros: np.ndarray, int4_data: np.ndarray) -> np.ndarray:
    packed = bytearray()
    packed += scales.astype(bfloat16).view(np.uint8).tobytes()
    packed += zeros.astype(bfloat16).view(np.uint8).tobytes()
    for row in range(Q4_ROWS):
        for col in range(0, Q4_K_CHUNK, 2):
            lo = int(int4_data[row, col]) & 0xF
            hi = int(int4_data[row, col + 1]) & 0xF
            packed.append(lo | (hi << 4))
    return np.frombuffer(bytes(packed), dtype=np.uint8)


def _pack_projection(scales: np.ndarray, zeros: np.ndarray, int4_data: np.ndarray) -> np.ndarray:
    chunks = []
    for chunk in range(Q4_CHUNKS):
        group0 = chunk * GROUPS_PER_CHUNK
        group1 = group0 + GROUPS_PER_CHUNK
        k0 = chunk * Q4_K_CHUNK
        k1 = k0 + Q4_K_CHUNK
        chunks.append(
            pack_q4nx_chunk(
                scales[:, group0:group1],
                zeros[:, group0:group1],
                int4_data[:, k0:k1],
            )
        )
    return np.concatenate(chunks)


def pack_qkv_weights(
    q_scales: np.ndarray,
    q_zeros: np.ndarray,
    q_int4: np.ndarray,
    k_scales: np.ndarray,
    k_zeros: np.ndarray,
    k_int4: np.ndarray,
    v_scales: np.ndarray,
    v_zeros: np.ndarray,
    v_int4: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            _pack_projection(q_scales, q_zeros, q_int4),
            _pack_projection(k_scales, k_zeros, k_int4),
            _pack_projection(v_scales, v_zeros, v_int4),
        ]
    )


def _project_rowblock(packed_rowblock: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    scale_bytes = Q4_ROWS * GROUPS_PER_CHUNK * 2
    zero_bytes = scale_bytes
    data_offset = scale_bytes + zero_bytes
    hidden_f32 = hidden.astype(np.float32)
    out = np.zeros(Q4_ROWS, dtype=np.float32)

    for chunk in range(Q4_CHUNKS):
        chunk_data = packed_rowblock[chunk * CHUNK_BYTES:(chunk + 1) * CHUNK_BYTES]
        scales = np.frombuffer(chunk_data[:scale_bytes], dtype=bfloat16).reshape(Q4_ROWS, GROUPS_PER_CHUNK)
        zeros = np.frombuffer(chunk_data[scale_bytes:data_offset], dtype=bfloat16).reshape(Q4_ROWS, GROUPS_PER_CHUNK)
        int4_raw = chunk_data[data_offset:]
        act = hidden_f32[chunk * Q4_K_CHUNK:(chunk + 1) * Q4_K_CHUNK]

        for row in range(Q4_ROWS):
            row_acc = np.float32(0.0)
            for group in range(GROUPS_PER_CHUNK):
                s = np.float32(scales[row, group])
                z = np.float32(zeros[row, group])
                values = np.zeros(GROUP_SIZE, dtype=np.float32)
                for lane in range(GROUP_SIZE):
                    col = group * GROUP_SIZE + lane
                    packed_byte = int(int4_raw[row * (Q4_K_CHUNK // 2) + col // 2])
                    values[lane] = (
                        np.float32((packed_byte >> 4) & 0xF)
                        if col & 1
                        else np.float32(packed_byte & 0xF)
                    )
                dequant = _to_bf16_f32((values - z) * s)
                row_acc = np.float32(row_acc + np.sum(dequant * act[group * GROUP_SIZE:(group + 1) * GROUP_SIZE]))
            out[row] = np.float32(out[row] + row_acc)
    return out


def project_qkv_reference(packed_weight: np.ndarray, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = []
    for projection in range(PROJECTIONS):
        start = projection * ROWBLOCK_BYTES
        end = start + ROWBLOCK_BYTES
        outputs.append(_project_rowblock(packed_weight[start:end], hidden))
    return outputs[0], outputs[1], outputs[2]


def make_case_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hidden = rng.uniform(-0.08, 0.08, HIDDEN_DIM).astype(bfloat16)

    params = []
    for _ in range(PROJECTIONS):
        scales = rng.uniform(0.0005, 0.004, (TOKEN_DWORDS, HIDDEN_DIM // GROUP_SIZE)).astype(bfloat16)
        zeros = rng.uniform(6.0, 9.0, (TOKEN_DWORDS, HIDDEN_DIM // GROUP_SIZE)).astype(bfloat16)
        int4_data = rng.integers(0, 16, (TOKEN_DWORDS, HIDDEN_DIM), dtype=np.uint8)
        params.append((scales, zeros, int4_data))

    packed_weight = pack_qkv_weights(*params[0], *params[1], *params[2])
    query, current_k, current_v = project_qkv_reference(packed_weight, hidden)
    return hidden, packed_weight, query, current_k, current_v


def _plane_value(global_t: int, dim: int, plane: str) -> np.float32:
    if plane == "v":
        raw = ((global_t * 5 + dim) % 13) - 6
        return np.float32(raw * 0.03125)
    raw = ((global_t * 7 + dim * 2) % 17) - 8
    return np.float32(raw * 0.025)


def make_kv_pair_cache_without_current(L: int) -> np.ndarray:
    num_tiles = ceil(L / TOKENS_PER_TILE)
    data = np.zeros(num_tiles * KV_TILE_DWORDS, dtype=np.float32)
    current_token = L - 1
    for tile_idx in range(num_tiles):
        tile_base = tile_idx * KV_TILE_DWORDS
        for t in range(TOKENS_PER_TILE):
            global_t = tile_idx * TOKENS_PER_TILE + t
            if global_t >= L or global_t == current_token:
                continue
            for dim in range(HEAD_DIM):
                data[tile_base + t * TOKEN_DWORDS + dim] = _plane_value(global_t, dim, "k")
                data[tile_base + PLANE_TILE_DWORDS + t * TOKEN_DWORDS + dim] = _plane_value(global_t, dim, "v")
    return data


def current_offsets(L: int) -> tuple[int, int]:
    tile_idx = (L - 1) // TOKENS_PER_TILE
    local_t = (L - 1) % TOKENS_PER_TILE
    tile_base = tile_idx * KV_TILE_DWORDS
    return tile_base + local_t * TOKEN_DWORDS, tile_base + PLANE_TILE_DWORDS + local_t * TOKEN_DWORDS


def apply_current_to_cache(L: int, kv_cache: np.ndarray, current_k: np.ndarray, current_v: np.ndarray) -> np.ndarray:
    out = kv_cache.copy()
    k_offset, v_offset = current_offsets(L)
    out[k_offset:k_offset + TOKEN_DWORDS] = current_k
    out[v_offset:v_offset + TOKEN_DWORDS] = current_v
    return out


def online_attention_reference_from_cache(L: int, query: np.ndarray, kv_cache: np.ndarray) -> np.ndarray:
    num_tiles = ceil(L / TOKENS_PER_TILE)
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    out = np.zeros(TOKEN_DWORDS, dtype=np.float32)
    running_max = np.float32(-np.inf)
    running_sum = np.float32(0.0)

    for tile_idx in range(num_tiles):
        valid = TOKENS_PER_TILE if tile_idx < num_tiles - 1 else L - tile_idx * TOKENS_PER_TILE
        tile_base = tile_idx * KV_TILE_DWORDS
        scores = np.zeros(valid, dtype=np.float32)
        for t in range(valid):
            token_base = tile_base + t * TOKEN_DWORDS
            scores[t] = np.sum(query * kv_cache[token_base:token_base + HEAD_DIM]) * scale

        local_max = np.max(scores)
        new_max = max(running_max, local_max)
        old_scale = np.float32(0.0) if running_sum == 0 else approx_exp(np.float32(running_max - new_max))
        weights = approx_exp(scores - new_max).astype(np.float32)
        running_sum = running_sum * old_scale + np.sum(weights)
        out *= old_scale
        for t in range(valid):
            token_base = tile_base + PLANE_TILE_DWORDS + t * TOKEN_DWORDS
            out += weights[t] * kv_cache[token_base:token_base + HEAD_DIM]
        running_max = new_max

    return (out / running_sum).astype(np.float32)


def attention_reference_after_current_write(
    L: int,
    hidden: np.ndarray,
    packed_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query, current_k, current_v = project_qkv_reference(packed_weight, hidden)
    cache_before = make_kv_pair_cache_without_current(L)
    cache_after = apply_current_to_cache(L, cache_before, current_k, current_v)
    expected = online_attention_reference_from_cache(L, query, cache_after)
    return query, current_k, current_v, cache_before, expected


if __name__ == "__main__":
    hidden_data, weight_data, q, k, v = make_case_data()
    _, _, _, _, out = attention_reference_after_current_write(33, hidden_data, weight_data)
    print(f"packed={weight_data.size} bytes query0={q[:4]} k0={k[:4]} v0={v[:4]} out0={out[:4]}")
