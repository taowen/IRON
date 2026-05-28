"""CPU reference for main16 Q/K/V -> Shape-A/B -> O compact -> c1r2."""

from __future__ import annotations

import numpy as np

from contract import (
    ATTENTION_PACKET_DWORDS,
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    RECORD_DWORDS,
    RECORD_PAYLOAD_DWORDS,
    ROWS_PER_COLUMN,
    SHAPE_CARRIER_DWORDS,
)

CASE_NAME = "qkv-shape-o-c1r2-bridge"
Q_PHASE = 0
K_PHASE = 1
V_PHASE = 2
O_PHASE = 3
STAGES = ("q", "k", "v", "o")
MAIN_RECORD_DWORDS = RECORD_DWORDS * len(STAGES)
COLUMN_COMPACT_DWORDS = RECORD_DWORDS + (ROWS_PER_COLUMN - 1) * RECORD_PAYLOAD_DWORDS
WINDOW_DWORDS = 512
Q_DWORDS = ATTENTION_PACKET_DWORDS
KV_SIDE_DWORDS = WINDOW_DWORDS * 4
MAIN_CHUNK_DWORDS = 128
SUMMARY_DWORDS = 8
OUTPUT_DWORDS = 8
PACKET_ID_ATTENTION = 2
Q_GLOBAL_PACKET_ID = 10
K_GLOBAL_PACKET_ID = 11
V_GLOBAL_PACKET_ID = 12
O_GLOBAL_PACKET_ID = 13
MAIN_PACKET_BASE = 16
COLUMN_PACKET_BASE = 4


def main_packet(group: int, row: int) -> int:
    return MAIN_PACKET_BASE + group * ROWS_PER_COLUMN + row


def column_packet(group: int) -> int:
    return COLUMN_PACKET_BASE + group


def record_header(phase: int, group: int, row: int) -> int:
    return (phase << 24) | (group << 16) | (row << 8) | 0xC3


def qkv_payload_value(phase: int, group: int, row: int, lane: int) -> int:
    return phase * 4096 + group * 1024 + row * 256 + lane * 3 + 7


def make_qkv_record(phase: int, group: int, row: int) -> np.ndarray:
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(phase, group, row)
    for lane in range(RECORD_PAYLOAD_DWORDS):
        record[1 + lane] = qkv_payload_value(phase, group, row, lane)
    return record


def column_compact_from_records(records: list[np.ndarray]) -> np.ndarray:
    compact = np.empty(COLUMN_COMPACT_DWORDS, dtype=np.int32)
    offset = 0
    for row, record in enumerate(records):
        segment = record if row == 0 else record[1:]
        compact[offset : offset + segment.shape[0]] = segment
        offset += segment.shape[0]
    return compact


def global_compact_from_columns(columns: list[np.ndarray]) -> np.ndarray:
    compact = np.empty(COMPACT_PACKET_DWORDS, dtype=np.int32)
    offset = 0
    for group, column in enumerate(columns):
        segment = column if group == 0 else column[1:]
        compact[offset : offset + segment.shape[0]] = segment
        offset += segment.shape[0]
    return compact


def qkv_global_compact(phase: int) -> np.ndarray:
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        records = [make_qkv_record(phase, group, row) for row in range(ROWS_PER_COLUMN)]
        columns.append(column_compact_from_records(records))
    return global_compact_from_columns(columns)


def q_payload() -> np.ndarray:
    q_compact = qkv_global_compact(Q_PHASE)
    k_compact = qkv_global_compact(K_PHASE)
    output = np.empty(Q_DWORDS, dtype=np.int32)
    for idx in range(Q_DWORDS):
        q_seed = int(q_compact[1 + (idx & 255)]) & 31
        k_seed = int(k_compact[1 + ((idx * 3) & 255)]) & 7
        output[idx] = 10_000 + idx * 3 + q_seed + k_seed
    return output


def kv_payload(side: int) -> np.ndarray:
    k_compact = qkv_global_compact(K_PHASE)
    v_compact = qkv_global_compact(V_PHASE)
    output = np.empty(KV_SIDE_DWORDS, dtype=np.int32)
    base = 20_000 + side * 10_000
    for idx in range(KV_SIDE_DWORDS):
        is_v = (idx // WINDOW_DWORDS) & 1
        if is_v:
            seed = int(v_compact[1 + ((idx * 7 + side * 13) & 255)]) & 31
        else:
            seed = int(k_compact[1 + ((idx * 5 + side * 11) & 255)]) & 31
        output[idx] = base + idx * 5 + seed
    return output


def q_window(window: int) -> np.ndarray:
    start = window * WINDOW_DWORDS
    return q_payload()[start : start + WINDOW_DWORDS]


def _kv_side_and_slot(window: int) -> tuple[int, int]:
    side = 0 if window < 2 else 1
    slot = window if side == 0 else window - 2
    return side, slot


def k_window(window: int) -> np.ndarray:
    side, slot = _kv_side_and_slot(window)
    start = slot * WINDOW_DWORDS * 2
    return kv_payload(side)[start : start + WINDOW_DWORDS]


def v_window(window: int) -> np.ndarray:
    side, slot = _kv_side_and_slot(window)
    start = slot * WINDOW_DWORDS * 2 + WINDOW_DWORDS
    return kv_payload(side)[start : start + WINDOW_DWORDS]


def make_carrier(window: int) -> np.ndarray:
    q = q_window(window)
    k = k_window(window)
    carrier = np.empty(SHAPE_CARRIER_DWORDS, dtype=np.int32)
    for idx in range(SHAPE_CARRIER_DWORDS):
        q_idx = (idx * 7 + window) & (WINDOW_DWORDS - 1)
        k_idx = (idx * 13 + window * 3) & (WINDOW_DWORDS - 1)
        carrier[idx] = (
            (window + 1) * 100_000
            + int(q[q_idx] & 0xFFFF)
            + int((k[k_idx] & 0xFFFF) << 1)
            + idx
        )
    return carrier


def return_window(window: int) -> np.ndarray:
    carrier = make_carrier(window)
    v = v_window(window)
    output = np.empty(WINDOW_DWORDS, dtype=np.int32)
    for idx in range(WINDOW_DWORDS):
        c_idx = (idx * 3 + window) % SHAPE_CARRIER_DWORDS
        v_idx = (idx * 5 + window * 11) & (WINDOW_DWORDS - 1)
        output[idx] = (
            (window + 1) * 1_000_000
            + int(carrier[c_idx])
            + int(v[v_idx] & 0xFFFF)
            + idx * 17
        )
    return output


def attention_payload() -> np.ndarray:
    return np.concatenate([return_window(window) for window in range(4)]).astype(np.int32)


def _u32_to_i32(value: int) -> int:
    return int(np.array(value & 0xFFFF_FFFF, dtype=np.uint32).view(np.int32))


def main_summary_from_attention() -> tuple[int, int, int, int, int, int]:
    payload = attention_payload()
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


def make_o_record(group: int, row: int) -> np.ndarray:
    chunks, first, last, payload_sum, payload_hash, total = main_summary_from_attention()
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(O_PHASE, group, row)
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
        0x51564F,
        0x4F434D50,
    )
    record[1:] = values
    return record


def o_global_compact() -> np.ndarray:
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        records = [make_o_record(group, row) for row in range(ROWS_PER_COLUMN)]
        columns.append(column_compact_from_records(records))
    return global_compact_from_columns(columns)


def expected_output() -> np.ndarray:
    compact = o_global_compact()
    sum_u32 = 0
    hash_u32 = 0
    for idx, value in enumerate(compact):
        value_u32 = int(value) & 0xFFFF_FFFF
        sum_u32 = (sum_u32 + value_u32) & 0xFFFF_FFFF
        hash_u32 = ((hash_u32 * 16_777_619) ^ ((value_u32 + idx) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
    return np.array(
        (
            0x51564F43,
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
        "main16 emits Q/K/V compact records into row1/c1r1",
        "c1r3 expands Q/K/V compacts into Q[2048] and split KV payloads",
        "Shape-A/B returns 4x512 dwords to c6r1 packet2",
        "main16 consumes packet2 as O chunks and emits O compact to c1r2",
        f"c1r2_output={OUTPUT_DWORDS} dwords",
    ]
