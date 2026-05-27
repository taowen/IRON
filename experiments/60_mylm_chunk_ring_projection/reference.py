"""CPU reference for exp59 exact MyLM N-block projection."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from generate import (
    CHUNK_BF16,
    COLUMN_OUTPUT_BF16,
    COLUMN_WEIGHT_BF16,
    GROUP_SIZE,
    GROUPS_PER_CHUNK,
    K,
    K_CHUNK,
    K_CHUNKS,
    M_PER_TILE,
    MAIN_COLUMNS,
    OUT_RECORD_BF16,
    OUT_TOTAL_I32,
    PATCHES_PER_COLUMN,
    PATCH_BF16,
    ROWS_PER_COLUMN,
    TOTAL_WEIGHT_I32,
)


def pack_q4nx_chunk(scales: np.ndarray, zeros: np.ndarray, int4_data: np.ndarray) -> np.ndarray:
    packed = bytearray()
    packed += scales.astype(bfloat16).view(np.uint8).tobytes()
    packed += zeros.astype(bfloat16).view(np.uint8).tobytes()
    for row in range(M_PER_TILE):
        for col in range(0, K_CHUNK, 2):
            lo = int(int4_data[row, col]) & 0x0F
            hi = int(int4_data[row, col + 1]) & 0x0F
            packed.append(lo | (hi << 4))
    out = np.frombuffer(bytes(packed), dtype=np.uint8)
    if out.shape[0] != CHUNK_BF16 * 2:
        raise RuntimeError(f"chunk size mismatch: {out.shape[0]} != {CHUNK_BF16 * 2}")
    return out


def make_packed_weights(seed: int = 59) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for group in range(len(MAIN_COLUMNS)):
        for patch in range(PATCHES_PER_COLUMN):
            for chunk in range(K_CHUNKS):
                for row_in_patch in range(2):
                    global_row = patch * 2 + row_in_patch
                    scales = rng.uniform(0.001, 0.012, (M_PER_TILE, GROUPS_PER_CHUNK)).astype(bfloat16)
                    zeros = rng.uniform(6.0, 9.0, (M_PER_TILE, GROUPS_PER_CHUNK)).astype(bfloat16)
                    int4 = rng.integers(0, 16, (M_PER_TILE, K_CHUNK), dtype=np.uint8)
                    # Fold coordinates into the RNG stream deterministically through data.
                    int4[:, 0] = (int4[:, 0].astype(np.uint16) + group + global_row + chunk) & 0x0F
                    parts.append(pack_q4nx_chunk(scales, zeros, int4))
    packed = np.concatenate(parts)
    expected_bytes = TOTAL_WEIGHT_I32 * 4
    if packed.shape[0] != expected_bytes:
        raise RuntimeError(f"packed size mismatch: {packed.shape[0]} != {expected_bytes}")
    return packed


def activation_slice(group: int, row: int, chunk: int) -> np.ndarray:
    values = []
    for idx in range(K_CHUNK):
        raw = (group * 17 + row * 19 + chunk * 23 + idx * 7 + 5) % 127
        values.append(bfloat16((raw - 63) / 64.0))
    return np.asarray(values, dtype=bfloat16)


def q4nx_chunk_matvec(packed_chunk: np.ndarray, act: np.ndarray) -> np.ndarray:
    scale_bytes = M_PER_TILE * GROUPS_PER_CHUNK * 2
    zero_bytes = M_PER_TILE * GROUPS_PER_CHUNK * 2
    data_offset = scale_bytes + zero_bytes
    scales = np.frombuffer(packed_chunk[:scale_bytes], dtype=bfloat16).reshape(M_PER_TILE, GROUPS_PER_CHUNK)
    zeros = np.frombuffer(packed_chunk[scale_bytes:data_offset], dtype=bfloat16).reshape(M_PER_TILE, GROUPS_PER_CHUNK)
    raw = packed_chunk[data_offset:]

    weights_u4 = np.zeros((M_PER_TILE, K_CHUNK), dtype=np.float32)
    for row in range(M_PER_TILE):
        for col in range(0, K_CHUNK, 2):
            byte = int(raw[row * (K_CHUNK // 2) + col // 2])
            weights_u4[row, col] = byte & 0x0F
            weights_u4[row, col + 1] = (byte >> 4) & 0x0F

    weights_f32 = np.zeros((M_PER_TILE, K_CHUNK), dtype=np.float32)
    for row in range(M_PER_TILE):
        for group in range(GROUPS_PER_CHUNK):
            start = group * GROUP_SIZE
            end = start + GROUP_SIZE
            weights_f32[row, start:end] = (
                weights_u4[row, start:end] - float(zeros[row, group])
            ) * float(scales[row, group])

    return weights_f32 @ act.astype(np.float32)


def chunk_for_tile(packed: np.ndarray, group: int, row: int, chunk: int) -> np.ndarray:
    patch = row // 2
    row_in_patch = row % 2
    byte_offset = (
        group * COLUMN_WEIGHT_BF16 * 2
        + patch * PATCH_BF16 * 2
        + chunk * 2 * CHUNK_BF16 * 2
        + row_in_patch * CHUNK_BF16 * 2
    )
    return packed[byte_offset:byte_offset + CHUNK_BF16 * 2]


def expected_tile_record(packed: np.ndarray, group: int, row: int) -> np.ndarray:
    accum = np.zeros(M_PER_TILE, dtype=np.float32)
    for chunk in range(K_CHUNKS):
        accum += q4nx_chunk_matvec(chunk_for_tile(packed, group, row, chunk), activation_slice(group, row, chunk))

    record = np.empty(OUT_RECORD_BF16, dtype=bfloat16)
    record[0] = bfloat16(group)
    record[1] = bfloat16(row)
    record[2:] = accum.astype(bfloat16)
    return record


def expected_output(packed: np.ndarray) -> np.ndarray:
    output = np.empty(OUT_TOTAL_I32 * 2, dtype=bfloat16)
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            start = group * COLUMN_OUTPUT_BF16 + row * OUT_RECORD_BF16
            output[start:start + OUT_RECORD_BF16] = expected_tile_record(packed, group, row)
    return output


def packed_as_i32(packed: np.ndarray) -> np.ndarray:
    return np.frombuffer(packed.tobytes(), dtype=np.int32)


if __name__ == "__main__":
    packed_weights = make_packed_weights()
    expected = expected_output(packed_weights)
    print(f"K={K}, K_chunks={K_CHUNKS}")
    print(f"chunk_bf16={CHUNK_BF16}, patch_bf16={PATCH_BF16}, patch_bytes=0x{PATCH_BF16 * 2:x}")
    print(f"weight_i32={packed_as_i32(packed_weights).shape[0]}")
    print(f"expected[0:10]={expected[:10].tolist()}")
