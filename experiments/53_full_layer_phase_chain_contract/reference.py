"""CPU reference for exp53 full fused-layer phase-chain contract."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

MAIN_COLUMNS = (2, 3, 4, 5)
EDGE_COLUMNS = (0, 1, 6, 7)
ROWS_PER_COLUMN = 4
M_PER_TILE = 32
HIDDEN_DIM = 4096
ACT_SLICE_BF16 = 256
K_CHUNK = 256
NUM_CHUNKS = HIDDEN_DIM // K_CHUNK
GROUP_SIZE = 32
RECORD_DWORDS = 17
RECORD_PAYLOAD_DWORDS = RECORD_DWORDS - 1
PHASE_NAMES = ("Q", "K", "V", "O", "GATE", "UP", "DOWN")
NUM_PHASES = len(PHASE_NAMES)

GROUPS_PER_CHUNK = K_CHUNK // GROUP_SIZE
CHUNK_BYTES = M_PER_TILE * GROUPS_PER_CHUNK * 2 * 2 + M_PER_TILE * K_CHUNK // 2
CHUNK_BF16 = CHUNK_BYTES // 2
FAT_CHUNK_BF16 = CHUNK_BF16 * ROWS_PER_COLUMN
OUT_RECORD_BF16 = M_PER_TILE + 2
PHASE_WEIGHT_BF16 = NUM_CHUNKS * FAT_CHUNK_BF16
COLUMN_WEIGHT_BF16 = NUM_PHASES * PHASE_WEIGHT_BF16
TOTAL_WEIGHT_BF16 = len(MAIN_COLUMNS) * COLUMN_WEIGHT_BF16
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2
COLUMN_OUTPUT_BF16 = ROWS_PER_COLUMN * OUT_RECORD_BF16
TOTAL_OUTPUT_BF16 = len(MAIN_COLUMNS) * COLUMN_OUTPUT_BF16
OUT_TOTAL_I32 = TOTAL_OUTPUT_BF16 // 2


def record_header(tag: int, group: int, row: int) -> np.int32:
    return np.int32(0x53000000 | (tag << 16) | (group << 12) | (row << 4) | 0xA)


def bf16_to_bucket(value: bfloat16, scale: int) -> int:
    return int(float(value) * float(scale))


def emit_seed_sideband(group: int, row: int) -> np.ndarray:
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(0, group, row)
    for lane in range(RECORD_PAYLOAD_DWORDS):
        record[1 + lane] = group * 37 + row * 11 + lane * 3 + 1
    return record


def main_emit_phase_record(
    phase_out: np.ndarray,
    o: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    swiglu: np.ndarray,
    phase: int,
    group: int,
    row: int,
) -> np.ndarray:
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(phase + 1, group, row)
    if phase == 3:
        o[:] = phase_out
    elif phase == 4:
        gate[:] = phase_out
    elif phase == 5:
        up[:] = phase_out
        for idx in range(M_PER_TILE):
            gate_f = max(min(float(gate[idx]), 8.0), -8.0)
            up_f = max(min(float(up[idx]), 8.0), -8.0)
            sigmoidish = gate_f / (1.0 + abs(gate_f))
            swiglu[idx] = bfloat16(sigmoidish * up_f)

    for lane in range(RECORD_PAYLOAD_DWORDS):
        record[1 + lane] = group * 37 + row * 11 + (phase + 1) * 101 + lane * 3 + 1
    return record


def edge_phase_slice(record: np.ndarray, phase: int, group: int, row: int, chunk_idx: int) -> np.ndarray:
    values: list[bfloat16] = []
    base = chunk_idx * ACT_SLICE_BF16
    for idx in range(ACT_SLICE_BF16):
        global_idx = base + idx
        lane = global_idx & (RECORD_PAYLOAD_DWORDS - 1)
        a = int(record[1 + lane])
        b = int(record[1 + ((lane + phase + 3) & (RECORD_PAYLOAD_DWORDS - 1))])
        raw = (a * (phase + 3) + b * 2 + global_idx * 7 + group * 11 + row * 13 + phase * 17) % 127
        values.append(bfloat16((raw - 63) / 64.0))
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


def make_quantized_weights(seed: int = 53):
    rng = np.random.default_rng(seed)
    n_groups = HIDDEN_DIM // GROUP_SIZE
    scales = [[None] * ROWS_PER_COLUMN for _ in MAIN_COLUMNS]
    zeros = [[None] * ROWS_PER_COLUMN for _ in MAIN_COLUMNS]
    int4 = [[None] * ROWS_PER_COLUMN for _ in MAIN_COLUMNS]

    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            scales[group][row] = [
                rng.uniform(0.001, 0.018, (M_PER_TILE, n_groups)).astype(bfloat16)
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


def chunk_for_tile(packed: np.ndarray, group: int, row: int, phase: int, chunk: int) -> np.ndarray:
    per_column_bytes = COLUMN_WEIGHT_BF16 * 2
    per_phase_bytes = PHASE_WEIGHT_BF16 * 2
    fat_chunk_bytes = FAT_CHUNK_BF16 * 2
    fat_offset = group * per_column_bytes + phase * per_phase_bytes + chunk * fat_chunk_bytes
    row_offset = fat_offset + row * CHUNK_BYTES
    return packed[row_offset:row_offset + CHUNK_BYTES]


def run_phase(packed: np.ndarray, group: int, row: int, phase: int, record: np.ndarray) -> np.ndarray:
    accum = np.zeros(M_PER_TILE, dtype=np.float32)
    for chunk in range(NUM_CHUNKS):
        activation_slice = edge_phase_slice(record, phase, group, row, chunk)
        accum += q4nx_matvec_from_chunk(chunk_for_tile(packed, group, row, phase, chunk), activation_slice)
    return accum.astype(bfloat16)


def final_record(o: np.ndarray, gate: np.ndarray, up: np.ndarray, swiglu: np.ndarray,
                 down: np.ndarray, group: int, row: int) -> np.ndarray:
    output = np.empty(OUT_RECORD_BF16, dtype=bfloat16)
    output[0] = bfloat16(group)
    output[1] = bfloat16(row)
    for idx in range(M_PER_TILE):
        value = float(o[idx])
        value += float(swiglu[idx]) * 0.25
        value += float(down[idx]) * 0.5
        value += float(gate[idx]) * 0.03125
        value += float(up[idx]) * 0.015625
        output[2 + idx] = bfloat16(value)
    return output


def expected_tile_record(packed: np.ndarray, group: int, row: int) -> np.ndarray:
    record = emit_seed_sideband(group, row)
    o = np.zeros(M_PER_TILE, dtype=bfloat16)
    gate = np.zeros(M_PER_TILE, dtype=bfloat16)
    up = np.zeros(M_PER_TILE, dtype=bfloat16)
    swiglu = np.zeros(M_PER_TILE, dtype=bfloat16)
    down = np.zeros(M_PER_TILE, dtype=bfloat16)
    for phase in range(NUM_PHASES):
        phase_out = run_phase(packed, group, row, phase, record)
        if phase == NUM_PHASES - 1:
            down = phase_out
        else:
            record = main_emit_phase_record(phase_out, o, gate, up, swiglu, phase, group, row)
    return final_record(o, gate, up, swiglu, down, group, row)


def expected_output(packed: np.ndarray) -> np.ndarray:
    output = np.empty(TOTAL_OUTPUT_BF16, dtype=bfloat16)
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            start = group * COLUMN_OUTPUT_BF16 + row * OUT_RECORD_BF16
            output[start:start + OUT_RECORD_BF16] = expected_tile_record(packed, group, row)
    return output


def make_packed_weights(seed: int = 53) -> np.ndarray:
    scales, zeros, int4 = make_quantized_weights(seed)
    packed = pack_all_weights_fat(scales, zeros, int4)
    assert packed.shape[0] == TOTAL_WEIGHT_BF16 * 2
    return packed


if __name__ == "__main__":
    packed_data = make_packed_weights()
    expected = expected_output(packed_data)
    print(f"weight_bytes={packed_data.shape[0]}")
    print(f"phases={PHASE_NAMES}")
    print(f"record0[0:10]={expected[:10].tolist()}")
