"""CPU reference for the full deterministic qwen3-layer bridge contract."""

from __future__ import annotations

import numpy as np

from c1r2_reference import (
    MAIN_ACCUM_DWORDS,
    MAIN_CHUNKS_PER_REPLAY,
    TOTAL_MAIN_CHUNKS,
    main_accum,
    make_upgate_records,
)
from contract import (
    C6R2_HALF_DWORDS,
    C6R2_INPUT_DWORDS,
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    RECORD_DWORDS,
    RECORD_PAYLOAD_DWORDS,
    ROWS_PER_COLUMN,
)
from cases.qkv_shape_o_c1r2_reference import (
    COLUMN_COMPACT_DWORDS,
    K_GLOBAL_PACKET_ID,
    KV_SIDE_DWORDS,
    MAIN_CHUNK_DWORDS,
    O_GLOBAL_PACKET_ID,
    PACKET_ID_ATTENTION,
    Q_DWORDS,
    Q_GLOBAL_PACKET_ID,
    SUMMARY_DWORDS,
    V_GLOBAL_PACKET_ID,
    WINDOW_DWORDS,
    column_compact_from_records,
    column_packet,
    global_compact_from_columns,
    main_packet,
    o_global_compact,
)

CASE_NAME = "full-layer-contract-bridge"
Q_PHASE = 0
K_PHASE = 1
V_PHASE = 2
O_PHASE = 3
UP_PHASE = 4
GATE_PHASE = 5
DOWN_PHASE = 6
STAGES = ("q", "k", "v", "o", "up", "gate", "down")
MAIN_RECORD_DWORDS = RECORD_DWORDS * len(STAGES)
OUTPUT_DWORDS = 8
FFN_GLOBAL_PACKET_ID = 14
DOWN_GLOBAL_PACKET_ID = 15
FULL_REPLAY_PACKET_ID = 0
DOWN_ACT_PACKET_ID = 1
DOWN_CHUNKS = C6R2_HALF_DWORDS // MAIN_CHUNK_DWORDS


def ffn_global_compact(phase: int) -> np.ndarray:
    compact = o_global_compact()
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        records = []
        for row in range(ROWS_PER_COLUMN):
            up, gate = make_upgate_records(group, row, compact)
            records.append(up if phase == UP_PHASE else gate)
        columns.append(column_compact_from_records(records))
    return global_compact_from_columns(columns)


def swiglu_input() -> np.ndarray:
    up = ffn_global_compact(UP_PHASE)[1:]
    gate = ffn_global_compact(GATE_PHASE)[1:]
    if up.shape[0] != C6R2_HALF_DWORDS or gate.shape[0] != C6R2_HALF_DWORDS:
        raise RuntimeError(f"bad swiglu halves: {up.shape[0]}/{gate.shape[0]}")
    return np.concatenate((up, gate)).astype(np.int32)


def swiglu_output() -> np.ndarray:
    values = swiglu_input()
    if values.shape[0] != C6R2_INPUT_DWORDS:
        raise RuntimeError(f"bad swiglu input: {values.shape[0]}")
    output = np.empty(C6R2_HALF_DWORDS, dtype=np.int32)
    for idx in range(C6R2_HALF_DWORDS):
        low = int(values[idx]) & 0xFFFF_FFFF
        high = int(values[C6R2_HALF_DWORDS + idx]) & 0xFFFF
        output[idx] = np.array(((low << 16) | high) & 0xFFFF_FFFF, dtype=np.uint32).view(np.int32)
    return output


def _u32_to_i32(value: int) -> int:
    return int(np.array(value & 0xFFFF_FFFF, dtype=np.uint32).view(np.int32))


def down_summary() -> tuple[int, int, int, int, int, int]:
    payload = swiglu_output()
    sum_u32 = 0
    hash_u32 = 0
    for idx, value in enumerate(payload):
        value_u32 = int(value) & 0xFFFF_FFFF
        sum_u32 = (sum_u32 + value_u32) & 0xFFFF_FFFF
        hash_u32 = ((hash_u32 * 16_777_619) ^ ((value_u32 + idx) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
    return (
        payload.shape[0] // MAIN_CHUNK_DWORDS,
        int(payload[0]),
        int(payload[-1]),
        _u32_to_i32(sum_u32),
        _u32_to_i32(hash_u32),
        payload.shape[0],
    )


def down_record_header(group: int, row: int) -> int:
    return (DOWN_PHASE << 24) | (group << 16) | (row << 8) | 0xD0


def make_down_record(group: int, row: int) -> np.ndarray:
    chunks, first, last, payload_sum, payload_hash, total = down_summary()
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = down_record_header(group, row)
    values = (
        chunks,
        first,
        last,
        payload_sum,
        payload_hash,
        total,
        group,
        row,
        chunks ^ group,
        first ^ row,
        last ^ group,
        payload_sum ^ row,
        payload_hash ^ group,
        total ^ row,
        0x444F574E,
        0x46494E,
    )
    record[1:] = values
    return record


def down_global_compact() -> np.ndarray:
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        records = [make_down_record(group, row) for row in range(ROWS_PER_COLUMN)]
        columns.append(column_compact_from_records(records))
    return global_compact_from_columns(columns)


def expected_output() -> np.ndarray:
    compact = down_global_compact()
    sum_u32 = 0
    hash_u32 = 0
    for idx, value in enumerate(compact):
        value_u32 = int(value) & 0xFFFF_FFFF
        sum_u32 = (sum_u32 + value_u32) & 0xFFFF_FFFF
        hash_u32 = ((hash_u32 * 16_777_619) ^ ((value_u32 + idx) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
    return np.array(
        (
            0x464C4C43,
            compact.shape[0],
            int(compact[0]),
            int(compact[-1]),
            _u32_to_i32(sum_u32),
            _u32_to_i32(hash_u32),
            int(compact[1]),
            int(compact[-2]),
        ),
        dtype=np.int32,
    )


def validate_output(got: np.ndarray) -> list[str]:
    expected = expected_output()
    if got.shape != expected.shape:
        return [f"shape mismatch: {got.shape} != {expected.shape}"]
    mismatch = np.where(got != expected)[0]
    errors = [
        f"out[{idx}]: expected={int(expected[idx])} got={int(got[idx])}"
        for idx in mismatch[:32]
    ]
    if mismatch.size > 32:
        errors.append(f"{mismatch.size - 32} additional mismatches")
    return errors


def route_summary() -> list[str]:
    return [
        f"case={CASE_NAME}",
        "closed_loop_1=main16 Q/K/V -> Shape-A/B -> packet2 -> main16 O",
        f"closed_loop_2=c1r2 O replay -> {TOTAL_MAIN_CHUNKS} main chunks -> up/gate compact",
        "closed_loop_3=c6r2 SwiGLU -> c6r1 shared bridge -> main16 down compact",
        f"down_chunks={DOWN_CHUNKS}, final_output={OUTPUT_DWORDS} dwords",
    ]
