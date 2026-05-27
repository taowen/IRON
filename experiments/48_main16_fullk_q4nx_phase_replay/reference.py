"""CPU reference for exp48 full-K Q4NX phase replay on main16."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

MAIN_COLUMNS = (2, 3, 4, 5)
ROWS_PER_COLUMN = 4
M_PER_TILE = 32
HIDDEN_DIM = 4096
K_CHUNK = 256
NUM_CHUNKS = HIDDEN_DIM // K_CHUNK
GROUP_SIZE = 32
PHASE_NAMES = ("O", "GATE", "UP", "DOWN")
NUM_PHASES = len(PHASE_NAMES)

GROUPS_PER_CHUNK = K_CHUNK // GROUP_SIZE
CHUNK_BYTES = M_PER_TILE * GROUPS_PER_CHUNK * 2 * 2 + M_PER_TILE * K_CHUNK // 2
CHUNK_BF16 = CHUNK_BYTES // 2
FAT_CHUNK_BF16 = CHUNK_BF16 * ROWS_PER_COLUMN
PHASE_WEIGHT_BF16 = NUM_CHUNKS * FAT_CHUNK_BF16
COLUMN_WEIGHT_BF16 = NUM_PHASES * PHASE_WEIGHT_BF16
TOTAL_WEIGHT_BF16 = len(MAIN_COLUMNS) * COLUMN_WEIGHT_BF16
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2

COLUMN_OUTPUT_BF16 = ROWS_PER_COLUMN * M_PER_TILE
TOTAL_OUTPUT_BF16 = len(MAIN_COLUMNS) * COLUMN_OUTPUT_BF16
OUT_TOTAL_I32 = TOTAL_OUTPUT_BF16 // 2


def activation_for_tile(group: int, row: int) -> np.ndarray:
    values = [(((group * 17 + row * 13 + idx * 5) % 31) - 15) / 32.0 for idx in range(HIDDEN_DIM)]
    return np.asarray(values, dtype=bfloat16)


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


def make_quantized_weights(seed: int = 48):
    rng = np.random.default_rng(seed)
    n_groups = HIDDEN_DIM // GROUP_SIZE
    scales = [[None] * ROWS_PER_COLUMN for _ in MAIN_COLUMNS]
    zeros = [[None] * ROWS_PER_COLUMN for _ in MAIN_COLUMNS]
    int4 = [[None] * ROWS_PER_COLUMN for _ in MAIN_COLUMNS]

    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            scales[group][row] = [
                rng.uniform(0.001, 0.02, (M_PER_TILE, n_groups)).astype(bfloat16)
                for _ in range(NUM_PHASES)
            ]
            zeros[group][row] = [
                rng.uniform(6.0, 9.0, (M_PER_TILE, n_groups)).astype(bfloat16)
                for _ in range(NUM_PHASES)
            ]
            int4[group][row] = [
                rng.integers(0, 16, (M_PER_TILE, HIDDEN_DIM), dtype=np.uint8)
                for _ in range(NUM_PHASES)
            ]
    return scales, zeros, int4


def pack_all_weights_fat(scales, zeros, int4) -> np.ndarray:
    parts: list[np.ndarray] = []
    for group in range(len(MAIN_COLUMNS)):
        for phase in range(NUM_PHASES):
            for chunk in range(NUM_CHUNKS):
                group_start = chunk * GROUPS_PER_CHUNK
                group_end = group_start + GROUPS_PER_CHUNK
                col_start = chunk * K_CHUNK
                col_end = col_start + K_CHUNK
                for row in range(ROWS_PER_COLUMN):
                    parts.append(
                        pack_q4nx_chunk(
                            scales[group][row][phase][:, group_start:group_end],
                            zeros[group][row][phase][:, group_start:group_end],
                            int4[group][row][phase][:, col_start:col_end],
                        )
                    )
    return np.concatenate(parts)


def q4nx_matvec_from_chunks(packed: np.ndarray, activation: np.ndarray) -> np.ndarray:
    accum = np.zeros(M_PER_TILE, dtype=np.float32)
    scale_bytes = M_PER_TILE * GROUPS_PER_CHUNK * 2
    zero_bytes = M_PER_TILE * GROUPS_PER_CHUNK * 2
    data_offset = scale_bytes + zero_bytes

    for chunk in range(NUM_CHUNKS):
        chunk_data = packed[chunk * CHUNK_BYTES:(chunk + 1) * CHUNK_BYTES]
        chunk_scales = np.frombuffer(chunk_data[:scale_bytes], dtype=bfloat16).reshape(M_PER_TILE, GROUPS_PER_CHUNK)
        chunk_zeros = np.frombuffer(chunk_data[scale_bytes:data_offset], dtype=bfloat16).reshape(M_PER_TILE, GROUPS_PER_CHUNK)
        int4_raw = chunk_data[data_offset:]

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

        act = activation[chunk * K_CHUNK:(chunk + 1) * K_CHUNK].astype(np.float32)
        accum += weights_f32 @ act

    return accum.astype(bfloat16)


def phase_chunks_for_tile(packed: np.ndarray, group: int, row: int, phase: int) -> np.ndarray:
    per_column_bytes = COLUMN_WEIGHT_BF16 * 2
    per_phase_bytes = PHASE_WEIGHT_BF16 * 2
    fat_chunk_bytes = FAT_CHUNK_BF16 * 2
    tile_chunks = bytearray()
    for chunk in range(NUM_CHUNKS):
        fat_offset = group * per_column_bytes + phase * per_phase_bytes + chunk * fat_chunk_bytes
        row_offset = fat_offset + row * CHUNK_BYTES
        tile_chunks.extend(packed[row_offset:row_offset + CHUNK_BYTES])
    return np.frombuffer(bytes(tile_chunks), dtype=np.uint8)


def combine_phases(o_phase: np.ndarray, gate_phase: np.ndarray, up_phase: np.ndarray,
                   down_phase: np.ndarray, group: int, row: int) -> np.ndarray:
    o = o_phase.astype(np.float32)
    gate = gate_phase.astype(np.float32)
    up = up_phase.astype(np.float32)
    down = down_phase.astype(np.float32)
    swiglu = (gate * up) / 128.0
    return (o + swiglu + down * 0.25 + group * 0.5 + row * 0.125).astype(bfloat16)


def expected_output(packed: np.ndarray) -> np.ndarray:
    output = np.empty(TOTAL_OUTPUT_BF16, dtype=bfloat16)
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            activation = activation_for_tile(group, row)
            phase_outputs = [
                q4nx_matvec_from_chunks(phase_chunks_for_tile(packed, group, row, phase), activation)
                for phase in range(NUM_PHASES)
            ]
            start = group * COLUMN_OUTPUT_BF16 + row * M_PER_TILE
            output[start:start + M_PER_TILE] = combine_phases(*phase_outputs, group, row)
    return output


def make_packed_weights(seed: int = 48) -> np.ndarray:
    scales, zeros, int4 = make_quantized_weights(seed)
    packed = pack_all_weights_fat(scales, zeros, int4)
    assert packed.shape[0] == TOTAL_WEIGHT_BF16 * 2
    return packed


if __name__ == "__main__":
    packed_data = make_packed_weights()
    expected = expected_output(packed_data)
    print(f"weight_bytes={packed_data.shape[0]}")
    print(f"output[0:8]={expected[:8].tolist()}")
