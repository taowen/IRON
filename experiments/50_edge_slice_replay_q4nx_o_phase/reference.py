"""CPU reference for exp50 edge shard slice replay feeding Q4NX O phase."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

MAIN_COLUMNS = (2, 3, 4, 5)
EDGE_COLUMNS = (0, 1, 6, 7)
ROWS_PER_COLUMN = 4
M_PER_TILE = 32
O_INPUT_DIM = 1024
ACT_SLICE_BF16 = 256
K_CHUNK = 256
NUM_CHUNKS = O_INPUT_DIM // K_CHUNK
GROUP_SIZE = 32
RECORD_DWORDS = 17
RECORD_PAYLOAD_DWORDS = RECORD_DWORDS - 1

GROUPS_PER_CHUNK = K_CHUNK // GROUP_SIZE
CHUNK_BYTES = M_PER_TILE * GROUPS_PER_CHUNK * 2 * 2 + M_PER_TILE * K_CHUNK // 2
CHUNK_BF16 = CHUNK_BYTES // 2
FAT_CHUNK_BF16 = CHUNK_BF16 * ROWS_PER_COLUMN
COLUMN_WEIGHT_BF16 = NUM_CHUNKS * FAT_CHUNK_BF16
TOTAL_WEIGHT_BF16 = len(MAIN_COLUMNS) * COLUMN_WEIGHT_BF16
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2

COLUMN_OUTPUT_BF16 = ROWS_PER_COLUMN * M_PER_TILE
TOTAL_OUTPUT_BF16 = len(MAIN_COLUMNS) * COLUMN_OUTPUT_BF16
OUT_TOTAL_I32 = TOTAL_OUTPUT_BF16 // 2


def record_header(group: int, row: int) -> np.int32:
    return np.int32(0x50000000 | (group << 12) | (row << 4) | 0xA)


def record_payload(group: int, row: int, lane: int) -> np.int32:
    return np.int32(group * 32 + row * 8 + lane)


def make_record(group: int, row: int) -> np.ndarray:
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(group, row)
    for lane in range(RECORD_PAYLOAD_DWORDS):
        record[1 + lane] = record_payload(group, row, lane)
    return record


def attention_value(group: int, row: int, record: np.ndarray, global_idx: int) -> bfloat16:
    lane = global_idx & (RECORD_PAYLOAD_DWORDS - 1)
    a = int(record[1 + lane])
    b = int(record[1 + ((lane + 5) & (RECORD_PAYLOAD_DWORDS - 1))])
    raw = (a * 3 + b * 2 + global_idx * 7 + group * 11 + row * 13) % 97
    return bfloat16((raw - 48) / 64.0)


def attention_slice_from_record(group: int, row: int, record: np.ndarray, slice_idx: int) -> np.ndarray:
    base = slice_idx * ACT_SLICE_BF16
    values = [attention_value(group, row, record, base + idx) for idx in range(ACT_SLICE_BF16)]
    return np.asarray(values, dtype=bfloat16)


def attention_shard_from_record(group: int, row: int, record: np.ndarray) -> np.ndarray:
    parts = [attention_slice_from_record(group, row, record, slice_idx) for slice_idx in range(NUM_CHUNKS)]
    return np.concatenate(parts).astype(bfloat16)


def pack_q4nx_chunk(scales: np.ndarray, zeros: np.ndarray, int4_data: np.ndarray) -> np.ndarray:
    packed = bytearray()
    packed += scales.astype(bfloat16).view(np.uint8).tobytes()
    packed += zeros.astype(bfloat16).view(np.uint8).tobytes()
    for row in range(M_PER_TILE):
        for col in range(0, K_CHUNK, 2):
            lo = int(int4_data[row, col]) & 0xF
            hi = int(int4_data[row, col + 1]) & 0xF
            packed.append(lo | (hi << 4))
    return np.frombuffer(bytes(packed), dtype=np.uint8)


def make_quantized_weights(seed: int = 50):
    rng = np.random.default_rng(seed)
    n_groups = O_INPUT_DIM // GROUP_SIZE
    scales = [[None] * ROWS_PER_COLUMN for _ in MAIN_COLUMNS]
    zeros = [[None] * ROWS_PER_COLUMN for _ in MAIN_COLUMNS]
    int4 = [[None] * ROWS_PER_COLUMN for _ in MAIN_COLUMNS]

    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            scales[group][row] = rng.uniform(0.001, 0.02, (M_PER_TILE, n_groups)).astype(bfloat16)
            zeros[group][row] = rng.uniform(6.0, 9.0, (M_PER_TILE, n_groups)).astype(bfloat16)
            int4[group][row] = rng.integers(0, 16, (M_PER_TILE, O_INPUT_DIM), dtype=np.uint8)
    return scales, zeros, int4


def pack_all_weights_fat(scales, zeros, int4) -> np.ndarray:
    parts: list[np.ndarray] = []
    for group in range(len(MAIN_COLUMNS)):
        for chunk in range(NUM_CHUNKS):
            group_start = chunk * GROUPS_PER_CHUNK
            group_end = group_start + GROUPS_PER_CHUNK
            col_start = chunk * K_CHUNK
            col_end = col_start + K_CHUNK
            for row in range(ROWS_PER_COLUMN):
                parts.append(
                    pack_q4nx_chunk(
                        scales[group][row][:, group_start:group_end],
                        zeros[group][row][:, group_start:group_end],
                        int4[group][row][:, col_start:col_end],
                    )
                )
    return np.concatenate(parts)


def q4nx_matvec_from_chunk(packed_chunk: np.ndarray, activation_slice: np.ndarray) -> np.ndarray:
    scale_bytes = M_PER_TILE * GROUPS_PER_CHUNK * 2
    zero_bytes = M_PER_TILE * GROUPS_PER_CHUNK * 2
    data_offset = scale_bytes + zero_bytes
    chunk_scales = np.frombuffer(packed_chunk[:scale_bytes], dtype=bfloat16).reshape(M_PER_TILE, GROUPS_PER_CHUNK)
    chunk_zeros = np.frombuffer(packed_chunk[scale_bytes:data_offset], dtype=bfloat16).reshape(M_PER_TILE, GROUPS_PER_CHUNK)
    int4_raw = packed_chunk[data_offset:]

    weights_u4 = np.zeros((M_PER_TILE, K_CHUNK), dtype=np.float32)
    for row in range(M_PER_TILE):
        for col in range(0, K_CHUNK, 2):
            byte = int(int4_raw[row * (K_CHUNK // 2) + col // 2])
            weights_u4[row, col] = byte & 0x0F
            weights_u4[row, col + 1] = (byte >> 4) & 0x0F

    weights_f32 = np.zeros((M_PER_TILE, K_CHUNK), dtype=np.float32)
    for row in range(M_PER_TILE):
        for group in range(GROUPS_PER_CHUNK):
            c0 = group * GROUP_SIZE
            c1 = c0 + GROUP_SIZE
            scale = float(chunk_scales[row, group])
            zero = float(chunk_zeros[row, group])
            weights_f32[row, c0:c1] = (weights_u4[row, c0:c1] - zero) * scale

    return weights_f32 @ activation_slice.astype(np.float32)


def chunks_for_tile(packed: np.ndarray, group: int, row: int) -> list[np.ndarray]:
    per_column_bytes = COLUMN_WEIGHT_BF16 * 2
    fat_chunk_bytes = FAT_CHUNK_BF16 * 2
    chunks: list[np.ndarray] = []
    for chunk in range(NUM_CHUNKS):
        fat_offset = group * per_column_bytes + chunk * fat_chunk_bytes
        row_offset = fat_offset + row * CHUNK_BYTES
        chunks.append(packed[row_offset:row_offset + CHUNK_BYTES])
    return chunks


def expected_tile_output(packed: np.ndarray, group: int, row: int) -> np.ndarray:
    record = make_record(group, row)
    accum = np.zeros(M_PER_TILE, dtype=np.float32)
    for slice_idx, chunk in enumerate(chunks_for_tile(packed, group, row)):
        activation_slice = attention_slice_from_record(group, row, record, slice_idx)
        accum += q4nx_matvec_from_chunk(chunk, activation_slice)
    return accum.astype(bfloat16)


def expected_output(packed: np.ndarray) -> np.ndarray:
    output = np.empty(TOTAL_OUTPUT_BF16, dtype=bfloat16)
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            start = group * COLUMN_OUTPUT_BF16 + row * M_PER_TILE
            output[start:start + M_PER_TILE] = expected_tile_output(packed, group, row)
    return output


def make_packed_weights(seed: int = 50) -> np.ndarray:
    scales, zeros, int4 = make_quantized_weights(seed)
    packed = pack_all_weights_fat(scales, zeros, int4)
    assert packed.shape[0] == TOTAL_WEIGHT_BF16 * 2
    return packed


if __name__ == "__main__":
    packed_data = make_packed_weights()
    expected = expected_output(packed_data)
    print(f"weight_bytes={packed_data.shape[0]}")
    print(f"slice0[0:8]={attention_slice_from_record(0, 0, make_record(0, 0), 0)[:8].tolist()}")
    print(f"slice3[0:8]={attention_slice_from_record(0, 0, make_record(0, 0), 3)[:8].tolist()}")
    print(f"output[0:8]={expected[:8].tolist()}")
