"""CPU reference for exp65 MyLM fused layer engine v0."""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

MAIN_COLUMNS = (2, 3, 4, 5)
EDGE_COLUMNS = (0, 1, 6, 7)
ROWS_PER_COLUMN = 4
M_PER_TILE = 32
OUTPUT_BLOCK_ROWS = 512
HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 12288
ACT_SLICE_BF16 = 256
K_CHUNK = 256
GROUP_SIZE = 32
RECORD_DWORDS = 17
RECORD_PAYLOAD_DWORDS = RECORD_DWORDS - 1
PHASE_NAMES = ("Q", "K", "V", "O", "UP", "GATE", "DOWN")
PHASE_INPUT_DIMS = (4096, 4096, 4096, 4096, 4096, 4096, 12288)
PHASE_OUTPUT_DIMS = (4096, 1024, 1024, 4096, 12288, 12288, 4096)
PHASE_BLOCKS = tuple(
    output_dim // OUTPUT_BLOCK_ROWS for output_dim in PHASE_OUTPUT_DIMS
)
PHASE_CHUNKS = tuple(input_dim // K_CHUNK for input_dim in PHASE_INPUT_DIMS)
NUM_PHASES = len(PHASE_NAMES)
TOTAL_LOGICAL_BLOCKS = sum(PHASE_BLOCKS)
PATCHES_PER_COLUMN = 2
ROWS_PER_PATCH = 2
O_PHASE = PHASE_NAMES.index("O")

GROUPS_PER_CHUNK = K_CHUNK // GROUP_SIZE
CHUNK_BYTES = M_PER_TILE * GROUPS_PER_CHUNK * 2 * 2 + M_PER_TILE * K_CHUNK // 2
CHUNK_BF16 = CHUNK_BYTES // 2
OUT_RECORD_BF16 = M_PER_TILE + 2
PATCH_BF16_BY_PHASE = tuple(
    ROWS_PER_PATCH * PHASE_CHUNKS[phase] * CHUNK_BF16 for phase in range(NUM_PHASES)
)
PHASE_PATCH_COUNTS = tuple(
    PHASE_BLOCKS[phase] * len(MAIN_COLUMNS) * PATCHES_PER_COLUMN
    for phase in range(NUM_PHASES)
)
PHASE_WEIGHT_BF16 = tuple(
    PHASE_PATCH_COUNTS[phase] * PATCH_BF16_BY_PHASE[phase]
    for phase in range(NUM_PHASES)
)
TOTAL_PATCHES = sum(PHASE_PATCH_COUNTS)
TOTAL_WEIGHT_BF16 = sum(PHASE_WEIGHT_BF16)
TOTAL_WEIGHT_I32 = TOTAL_WEIGHT_BF16 // 2
COLUMN_OUTPUT_BF16 = ROWS_PER_COLUMN * OUT_RECORD_BF16
TOTAL_OUTPUT_BF16 = len(MAIN_COLUMNS) * COLUMN_OUTPUT_BF16
OUT_TOTAL_I32 = TOTAL_OUTPUT_BF16 // 2


def record_header(tag: int, group: int, row: int) -> np.int32:
    return np.int32(0x54000000 | (tag << 16) | (group << 12) | (row << 4) | 0xA)


def summary_scale(phase: int, block: int) -> float:
    return 0.0078125 * float((phase + 1) * ((block & 3) + 1))


def emit_seed_sideband(group: int, row: int) -> np.ndarray:
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(0, group, row)
    for lane in range(RECORD_PAYLOAD_DWORDS):
        record[1 + lane] = group * 37 + row * 11 + lane * 3 + 1
    return record


def emit_next_block_sideband(
    phase_out: np.ndarray,
    summary: np.ndarray,
    phase: int,
    block: int,
    group: int,
    row: int,
) -> np.ndarray:
    for idx in range(M_PER_TILE):
        next_value = float(summary[idx]) + float(phase_out[idx]) * summary_scale(
            phase, block
        )
        summary[idx] = bfloat16(next_value)

    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(phase * 32 + block + 1, group, row)
    for lane in range(RECORD_PAYLOAD_DWORDS):
        record[1 + lane] = (
            group * 37 + row * 11 + (phase + 1) * 101 + (block + 1) * 17 + lane * 3 + 1
        )
    return record


def edge_block_slice(
    record: np.ndarray, phase: int, block: int, group: int, row: int, chunk_idx: int
) -> np.ndarray:
    values: list[bfloat16] = []
    base = chunk_idx * ACT_SLICE_BF16
    for idx in range(ACT_SLICE_BF16):
        global_idx = base + idx
        if phase == O_PHASE:
            values.append(attention_output_value(global_idx))
        else:
            lane = global_idx & (RECORD_PAYLOAD_DWORDS - 1)
            a = int(record[1 + lane])
            b = int(
                record[1 + ((lane + phase + block + 5) & (RECORD_PAYLOAD_DWORDS - 1))]
            )
            raw = (
                a * (phase + 3)
                + b * 2
                + global_idx * 7
                + chunk_idx * 11
                + block * 13
                + group * 17
                + row * 19
                + phase * 23
            ) % 127
            values.append(bfloat16((raw - 63) / 64.0))
    return np.asarray(values, dtype=bfloat16)


def attention_output_value(global_idx: int) -> bfloat16:
    head = global_idx // 128
    dim = global_idx - head * 128
    raw = (head * 11 + dim * 7 + global_idx * 3 + 19) % 127
    return bfloat16((raw - 63) / 64.0)


def make_q4nx_chunk(rng: np.random.Generator) -> np.ndarray:
    scales = rng.uniform(0.001, 0.014, (M_PER_TILE, GROUPS_PER_CHUNK)).astype(bfloat16)
    zeros = rng.uniform(6.0, 9.0, (M_PER_TILE, GROUPS_PER_CHUNK)).astype(bfloat16)
    packed_data = rng.integers(0, 256, M_PER_TILE * (K_CHUNK // 2), dtype=np.uint8)
    packed = bytearray()
    packed += scales.view(np.uint8).tobytes()
    packed += zeros.view(np.uint8).tobytes()
    packed += packed_data.tobytes()
    return np.frombuffer(bytes(packed), dtype=np.uint8)


def make_packed_weights(seed: int = 54) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for phase in range(NUM_PHASES):
        for block in range(PHASE_BLOCKS[phase]):
            for group in range(len(MAIN_COLUMNS)):
                for pair in range(PATCHES_PER_COLUMN):
                    for chunk in range(PHASE_CHUNKS[phase]):
                        for _row_in_pair in range(ROWS_PER_PATCH):
                            parts.append(make_q4nx_chunk(rng))
    packed = np.concatenate(parts)
    assert packed.shape[0] == TOTAL_WEIGHT_BF16 * 2
    return packed


def q4nx_matvec_from_chunk(
    packed_chunk: np.ndarray, activation_slice: np.ndarray
) -> np.ndarray:
    scale_bytes = M_PER_TILE * GROUPS_PER_CHUNK * 2
    zero_bytes = M_PER_TILE * GROUPS_PER_CHUNK * 2
    data_offset = scale_bytes + zero_bytes
    chunk_scales = np.frombuffer(packed_chunk[:scale_bytes], dtype=bfloat16).reshape(
        M_PER_TILE, GROUPS_PER_CHUNK
    )
    chunk_zeros = np.frombuffer(
        packed_chunk[scale_bytes:data_offset], dtype=bfloat16
    ).reshape(M_PER_TILE, GROUPS_PER_CHUNK)
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


def phase_offset_bf16(phase: int) -> int:
    return sum(PHASE_WEIGHT_BF16[:phase])


def chunk_for_tile(
    packed: np.ndarray, group: int, row: int, phase: int, block: int, chunk: int
) -> np.ndarray:
    pair = row // ROWS_PER_PATCH
    row_in_pair = row % ROWS_PER_PATCH
    patch_bf16 = PATCH_BF16_BY_PHASE[phase]
    patches_per_block = len(MAIN_COLUMNS) * PATCHES_PER_COLUMN
    patch_in_phase = block * patches_per_block + group * PATCHES_PER_COLUMN + pair
    patch_offset_bf16 = phase_offset_bf16(phase) + patch_in_phase * patch_bf16
    chunk_offset_bf16 = chunk * ROWS_PER_PATCH * CHUNK_BF16
    row_offset_bf16 = row_in_pair * CHUNK_BF16
    offset_bytes = (patch_offset_bf16 + chunk_offset_bf16 + row_offset_bf16) * 2
    return packed[offset_bytes : offset_bytes + CHUNK_BYTES]


def run_block(
    packed: np.ndarray, group: int, row: int, phase: int, block: int, record: np.ndarray
) -> np.ndarray:
    accum = np.zeros(M_PER_TILE, dtype=np.float32)
    for chunk in range(PHASE_CHUNKS[phase]):
        activation_slice = edge_block_slice(record, phase, block, group, row, chunk)
        accum += q4nx_matvec_from_chunk(
            chunk_for_tile(packed, group, row, phase, block, chunk), activation_slice
        )
    return accum.astype(bfloat16)


def expected_tile_record(packed: np.ndarray, group: int, row: int) -> np.ndarray:
    record = emit_seed_sideband(group, row)
    summary = np.zeros(M_PER_TILE, dtype=bfloat16)
    for phase in range(NUM_PHASES):
        for block in range(PHASE_BLOCKS[phase]):
            phase_out = run_block(packed, group, row, phase, block, record)
            if phase == NUM_PHASES - 1 and block == PHASE_BLOCKS[phase] - 1:
                for idx in range(M_PER_TILE):
                    next_value = float(summary[idx]) + float(
                        phase_out[idx]
                    ) * summary_scale(phase, block)
                    summary[idx] = bfloat16(next_value)
            else:
                record = emit_next_block_sideband(
                    phase_out, summary, phase, block, group, row
                )

    output = np.empty(OUT_RECORD_BF16, dtype=bfloat16)
    output[0] = bfloat16(group)
    output[1] = bfloat16(row)
    output[2:] = summary
    return output


def expected_output(packed: np.ndarray) -> np.ndarray:
    output = np.empty(TOTAL_OUTPUT_BF16, dtype=bfloat16)
    for group in range(len(MAIN_COLUMNS)):
        for row in range(ROWS_PER_COLUMN):
            start = group * COLUMN_OUTPUT_BF16 + row * OUT_RECORD_BF16
            output[start : start + OUT_RECORD_BF16] = expected_tile_record(
                packed, group, row
            )
    return output


if __name__ == "__main__":
    packed_data = make_packed_weights()
    expected = expected_output(packed_data)
    print(f"weight_bytes={packed_data.shape[0]}")
    print(f"patches={TOTAL_PATCHES}")
    print(f"phase_blocks={dict(zip(PHASE_NAMES, PHASE_BLOCKS, strict=True))}")
    print(f"phase_chunks={dict(zip(PHASE_NAMES, PHASE_CHUNKS, strict=True))}")
    print(f"record0[0:10]={expected[:10].tolist()}")
