"""CPU reference for exp66 MyLM real attention-to-O producer."""

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
RECORD_PAYLOAD_BF16 = RECORD_PAYLOAD_DWORDS * 2
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
Q_PHASE = PHASE_NAMES.index("Q")
K_PHASE = PHASE_NAMES.index("K")
V_PHASE = PHASE_NAMES.index("V")
O_PHASE = PHASE_NAMES.index("O")
CONTEXT_LEN = 31
TOKENS_PER_TILE = 16
LOCAL_Q_VALUES = PHASE_BLOCKS[Q_PHASE] * M_PER_TILE
LOCAL_KV_VALUES = PHASE_BLOCKS[K_PHASE] * M_PER_TILE

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


def record_tag(record: np.ndarray) -> int:
    return (int(record[0]) >> 16) & 0xFF


def record_payload(record: np.ndarray) -> np.ndarray:
    return record[1:].view(bfloat16)


def summary_scale(phase: int, block: int) -> float:
    return 0.0078125 * float((phase + 1) * ((block & 3) + 1))


def emit_seed_sideband(group: int, row: int) -> np.ndarray:
    record = np.zeros(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(0, group, row)
    payload = record_payload(record)
    for lane in range(RECORD_PAYLOAD_BF16):
        raw = (group * 37 + row * 11 + lane * 3 + 1) % 127
        payload[lane] = bfloat16((raw - 63) / 64.0)
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

    record = np.zeros(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(phase * 32 + block + 1, group, row)
    record_payload(record)[:] = payload_values_from_phase(phase_out, phase, block)
    return record


def payload_values_from_phase(
    phase_out: np.ndarray, phase: int, block: int
) -> np.ndarray:
    scaled = phase_out[:RECORD_PAYLOAD_BF16].astype(np.float32) * np.float32(0.00390625)
    scaled = scaled + np.float32((phase + 1) * ((block & 3) + 1)) * np.float32(
        0.0009765625
    )
    return np.clip(scaled, -2.0, 2.0).astype(bfloat16)


def edge_block_slice(
    record: np.ndarray,
    phase: int,
    block: int,
    group: int,
    row: int,
    chunk_idx: int,
    state: "TileEdgeState",
) -> np.ndarray:
    state.remember(record)
    values: list[bfloat16] = []
    base = chunk_idx * ACT_SLICE_BF16
    payload = record_payload(record)
    for idx in range(ACT_SLICE_BF16):
        global_idx = base + idx
        if phase == O_PHASE:
            values.append(state.attention_output_value(global_idx))
        else:
            lane = global_idx & (RECORD_PAYLOAD_BF16 - 1)
            other = (lane + phase + block + 5) & (RECORD_PAYLOAD_BF16 - 1)
            a = np.float32(payload[lane])
            b = np.float32(payload[other])
            raw = (
                global_idx * 7
                + chunk_idx * 11
                + block * 13
                + group * 17
                + row * 19
                + phase * 23
            ) % 127
            mix = np.float32(a * np.float32(0.5 + 0.03125 * (phase + 1)))
            mix = np.float32(mix + b * np.float32(0.25 + 0.015625 * ((block & 7) + 1)))
            mix = np.float32(mix + np.float32(raw - 63) * np.float32(0.00390625))
            values.append(bfloat16(mix))
    return np.asarray(values, dtype=bfloat16)


def approx_exp_scalar(x: np.float32) -> np.float32:
    if x <= np.float32(-16.0):
        return np.float32(0.0)
    if x > np.float32(0.0):
        x = np.float32(0.0)
    y = np.float32(1.0) + x * np.float32(1.0 / 64.0)
    for _ in range(6):
        y = np.float32(y * y)
    return y


def history_k_value(group: int, row: int, token: int, dim: int) -> np.float32:
    raw = (group * 13 + row * 5 + token * 7 + dim * 3) % 37
    return np.float32((raw - 18) * 0.00875)


def history_v_value(group: int, row: int, token: int, dim: int) -> np.float32:
    raw = (group * 11 + row * 3 + token * 5 + dim * 2) % 31
    return np.float32((raw - 15) * 0.0125)


class TileEdgeState:
    def __init__(self, group: int, row: int) -> None:
        self.group = group
        self.row = row
        self.q = np.zeros(LOCAL_Q_VALUES, dtype=np.float32)
        self.k = np.zeros(LOCAL_KV_VALUES, dtype=np.float32)
        self.v = np.zeros(LOCAL_KV_VALUES, dtype=np.float32)
        self.attn_cache = np.zeros(HIDDEN_DIM, dtype=bfloat16)
        self.attn_valid = np.zeros(HIDDEN_DIM, dtype=bool)

    def remember(self, record: np.ndarray) -> None:
        tag = record_tag(record)
        if tag <= 0:
            return
        produced_phase = (tag - 1) // 32
        produced_block = (tag - 1) - produced_phase * 32
        payload = record_payload(record).astype(np.float32)
        if produced_phase == Q_PHASE and produced_block < PHASE_BLOCKS[Q_PHASE]:
            base = produced_block * M_PER_TILE
            self.q[base : base + M_PER_TILE] = payload[:M_PER_TILE]
        elif produced_phase == K_PHASE and produced_block < PHASE_BLOCKS[K_PHASE]:
            base = produced_block * M_PER_TILE
            self.k[base : base + M_PER_TILE] = payload[:M_PER_TILE]
        elif produced_phase == V_PHASE and produced_block < PHASE_BLOCKS[V_PHASE]:
            base = produced_block * M_PER_TILE
            self.v[base : base + M_PER_TILE] = payload[:M_PER_TILE]

    def attention_output_value(self, global_idx: int) -> bfloat16:
        if self.attn_valid[global_idx]:
            return self.attn_cache[global_idx]

        q_idx = global_idx & (LOCAL_Q_VALUES - 1)
        kv_idx = (global_idx + self.row * 7 + self.group * 11) & (LOCAL_KV_VALUES - 1)
        query = np.float32(self.q[q_idx])

        running_max = np.float32(-np.inf)
        running_sum = np.float32(0.0)
        out = np.float32(0.0)
        for tile in range(2):
            valid = TOKENS_PER_TILE if tile == 0 else CONTEXT_LEN - TOKENS_PER_TILE
            local_max = np.float32(-np.inf)
            for token in range(valid):
                global_token = tile * TOKENS_PER_TILE + token
                if global_token == CONTEXT_LEN - 1:
                    key = np.float32(self.k[kv_idx])
                else:
                    key = history_k_value(self.group, self.row, global_token, kv_idx)
                score = np.float32(query * key * np.float32(0.08838834764831845))
                score = np.float32(
                    score
                    + np.float32((global_idx + global_token * 3) & 7)
                    * np.float32(0.0009765625)
                )
                local_max = max(local_max, score)

            new_max = max(running_max, local_max)
            old_scale = (
                np.float32(0.0)
                if running_sum == np.float32(0.0)
                else approx_exp_scalar(np.float32(running_max - new_max))
            )
            out = np.float32(out * old_scale)

            local_sum = np.float32(0.0)
            for token in range(valid):
                global_token = tile * TOKENS_PER_TILE + token
                if global_token == CONTEXT_LEN - 1:
                    key = np.float32(self.k[kv_idx])
                    value = np.float32(self.v[kv_idx])
                else:
                    key = history_k_value(self.group, self.row, global_token, kv_idx)
                    value = history_v_value(self.group, self.row, global_token, kv_idx)
                score = np.float32(query * key * np.float32(0.08838834764831845))
                score = np.float32(
                    score
                    + np.float32((global_idx + global_token * 3) & 7)
                    * np.float32(0.0009765625)
                )
                weight = approx_exp_scalar(np.float32(score - new_max))
                local_sum = np.float32(local_sum + weight)
                out = np.float32(out + weight * value)

            running_max = np.float32(new_max)
            running_sum = np.float32(running_sum * old_scale + local_sum)

        result = np.float32(0.0) if running_sum == 0 else np.float32(out / running_sum)
        self.attn_cache[global_idx] = bfloat16(result)
        self.attn_valid[global_idx] = True
        return self.attn_cache[global_idx]


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

    packed_u8 = int4_raw.reshape(M_PER_TILE, K_CHUNK // 2)
    weights_u4 = np.empty((M_PER_TILE, K_CHUNK), dtype=np.float32)
    weights_u4[:, 0::2] = (packed_u8 & 0x0F).astype(np.float32)
    weights_u4[:, 1::2] = (packed_u8 >> 4).astype(np.float32)

    grouped_weights = weights_u4.reshape(M_PER_TILE, GROUPS_PER_CHUNK, GROUP_SIZE)
    scales = chunk_scales.astype(np.float32)[:, :, None]
    zeros = chunk_zeros.astype(np.float32)[:, :, None]
    grouped_act = activation_slice.astype(np.float32).reshape(
        GROUPS_PER_CHUNK, GROUP_SIZE
    )
    return np.sum(
        (grouped_weights - zeros) * scales * grouped_act[None, :, :], axis=(1, 2)
    )


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
    packed: np.ndarray,
    group: int,
    row: int,
    phase: int,
    block: int,
    record: np.ndarray,
    state: TileEdgeState,
) -> np.ndarray:
    accum = np.zeros(M_PER_TILE, dtype=np.float32)
    for chunk in range(PHASE_CHUNKS[phase]):
        activation_slice = edge_block_slice(
            record, phase, block, group, row, chunk, state
        )
        accum += q4nx_matvec_from_chunk(
            chunk_for_tile(packed, group, row, phase, block, chunk), activation_slice
        )
    return accum.astype(bfloat16)


def expected_tile_record(packed: np.ndarray, group: int, row: int) -> np.ndarray:
    record = emit_seed_sideband(group, row)
    state = TileEdgeState(group, row)
    summary = np.zeros(M_PER_TILE, dtype=bfloat16)
    for phase in range(NUM_PHASES):
        for block in range(PHASE_BLOCKS[phase]):
            phase_out = run_block(packed, group, row, phase, block, record, state)
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
