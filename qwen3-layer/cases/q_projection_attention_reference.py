"""CPU reference for Q projection payloads entering KV-scan attention."""

from __future__ import annotations

import numpy as np

from contract import MAIN_COLUMNS, MAIN_ROWS, RECORD_DWORDS, RECORD_PAYLOAD_DWORDS, ROWS_PER_COLUMN
from cases import attention_block_reference as attention
from cases import kv_scan_reference as kvscan
from qkv_compact_reference import (
    Q_PHASE,
    column_compact_from_records,
    global_compact_from_columns,
    make_qkv_record,
    record_header,
)

CASE_NAME = "q-projection-attention-reference"
O_PHASE = 3
STAGES = ("q", "o")
MAIN_RECORD_DWORDS = RECORD_DWORDS * len(STAGES)
HOST_OUTPUT_DWORDS = 8
Q_GLOBAL_PACKET_ID = 10
O_GLOBAL_PACKET_ID = 13
MAIN_PACKET_BASE = 16
COLUMN_PACKET_BASE = 4
COLUMN_COMPACT_DWORDS = RECORD_DWORDS + (ROWS_PER_COLUMN - 1) * RECORD_PAYLOAD_DWORDS

CONTEXT = attention.CONTEXT
HEAD_DIM = attention.HEAD_DIM
HEADS_PER_WINDOW = attention.HEADS_PER_WINDOW
K_CACHE_SIDE_DWORDS = kvscan.K_CACHE_SIDE_DWORDS
K_WINDOW_DWORDS = kvscan.K_WINDOW_DWORDS
KV_CACHE_SIDE_DWORDS = kvscan.KV_CACHE_SIDE_DWORDS
KV_HEADS_PER_WINDOW = kvscan.KV_HEADS_PER_WINDOW
KV_SIDE_DWORDS = kvscan.KV_SIDE_DWORDS
OUTPUT_DWORDS = kvscan.OUTPUT_DWORDS
Q_DWORDS = kvscan.Q_DWORDS
SCALAR_DWORDS = kvscan.SCALAR_DWORDS
V_CACHE_SIDE_DWORDS = kvscan.V_CACHE_SIDE_DWORDS
V_WINDOW_DWORDS = kvscan.V_WINDOW_DWORDS
WEIGHT_DWORDS = kvscan.WEIGHT_DWORDS
WINDOW_DWORDS = kvscan.WINDOW_DWORDS


def main_packet(group: int, row: int) -> int:
    return MAIN_PACKET_BASE + group * ROWS_PER_COLUMN + row


def column_packet(group: int) -> int:
    return COLUMN_PACKET_BASE + group


def q_global_compact() -> np.ndarray:
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        records = [make_qkv_record(Q_PHASE, group, row) for row in range(ROWS_PER_COLUMN)]
        columns.append(column_compact_from_records(records))
    return global_compact_from_columns(columns)


def _pack_s16_pair(low: int, high: int) -> np.int32:
    low_u16 = low & 0xFFFF
    high_u16 = high & 0xFFFF
    return np.array(low_u16 | (high_u16 << 16), dtype=np.uint32).view(np.int32)


def _q_lane(q_compact: np.ndarray, lane: int) -> int:
    seed = int(q_compact[1 + (lane & 255)]) & 31
    return ((lane * 5 + seed) % 31) - 15


def make_q_payload() -> np.ndarray:
    q_compact = q_global_compact()
    payload = np.empty(Q_DWORDS, dtype=np.int32)
    for idx in range(Q_DWORDS):
        low_lane = idx * 2
        high_lane = low_lane + 1
        payload[idx] = _pack_s16_pair(
            _q_lane(q_compact, low_lane),
            _q_lane(q_compact, high_lane),
        )
    return payload


def q_window(window: int) -> np.ndarray:
    start = window * WINDOW_DWORDS
    return make_q_payload()[start : start + WINDOW_DWORDS]


def _side_and_slot(window: int) -> tuple[int, int]:
    side = 0 if window < 2 else 1
    slot = window if side == 0 else window - 2
    return side, slot


def k_window(window: int) -> np.ndarray:
    side, slot = _side_and_slot(window)
    start = slot * K_WINDOW_DWORDS
    return kvscan.make_kv_cache_payload(side)[start : start + K_WINDOW_DWORDS]


def v_window(window: int) -> np.ndarray:
    side, slot = _side_and_slot(window)
    start = K_CACHE_SIDE_DWORDS + slot * V_WINDOW_DWORDS
    return kvscan.make_kv_cache_payload(side)[start : start + V_WINDOW_DWORDS]


def _score(q: np.ndarray, k: np.ndarray, q_head: int, token: int) -> int:
    kv_head = q_head // attention.GQA_RATIO
    total = 0
    q_base = q_head * HEAD_DIM
    k_base = kv_head * CONTEXT * HEAD_DIM + token * HEAD_DIM
    for dim in range(HEAD_DIM):
        total += attention._unpack_s16(q, q_base + dim) * attention._unpack_s16(k, k_base + dim)
    return attention._trunc_div(total, HEAD_DIM)


def make_carrier(window: int) -> np.ndarray:
    q = q_window(window)
    k = k_window(window)
    carrier = np.zeros(attention.WEIGHT_DWORDS + attention.SCALAR_DWORDS, dtype=np.int32)
    for q_head in range(HEADS_PER_WINDOW):
        scores = [_score(q, k, q_head, token) for token in range(CONTEXT)]
        running_max = max(scores)
        weights = [attention._softmax_weight(running_max - score) for score in scores]
        for token in range(0, CONTEXT, 2):
            word = (q_head * CONTEXT + token) // 2
            carrier[word] = attention._pack_weight_pair(weights[token], weights[token + 1])
        scalar = WEIGHT_DWORDS + q_head * 2
        carrier[scalar] = running_max
        carrier[scalar + 1] = sum(weights)
    return carrier


def return_window(window: int) -> np.ndarray:
    carrier = make_carrier(window)
    v = v_window(window)
    lanes = np.zeros(OUTPUT_DWORDS * 2, dtype=np.int32)
    for q_head in range(HEADS_PER_WINDOW):
        kv_head = q_head // attention.GQA_RATIO
        weight_sum = int(carrier[WEIGHT_DWORDS + q_head * 2 + 1])
        for dim in range(HEAD_DIM):
            total = 0
            for token in range(CONTEXT):
                weight = attention._unpack_weight(carrier, q_head, token)
                v_lane = kv_head * CONTEXT * HEAD_DIM + token * HEAD_DIM + dim
                total += weight * attention._unpack_s16(v, v_lane)
            lanes[q_head * HEAD_DIM + dim] = attention._trunc_div(total, weight_sum)
    return attention._pack_lanes(lanes)


def attention_payload() -> np.ndarray:
    return np.concatenate([return_window(window) for window in range(4)]).astype(np.int32)


def _u32_to_i32(value: int) -> int:
    return int(np.array(value & 0xFFFF_FFFF, dtype=np.uint32).view(np.int32))


def _summary_hash(payload: np.ndarray) -> tuple[int, int]:
    sum_u32 = 0
    hash_u32 = 0
    for idx, value in enumerate(payload):
        value_u32 = int(value) & 0xFFFF_FFFF
        sum_u32 = (sum_u32 + value_u32) & 0xFFFF_FFFF
        hash_u32 = ((hash_u32 * 16_777_619) ^ ((value_u32 + idx) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
    return _u32_to_i32(sum_u32), _u32_to_i32(hash_u32)


def main_summary_from_payload(payload: np.ndarray) -> tuple[int, int, int, int, int, int]:
    payload_sum, payload_hash = _summary_hash(payload)
    return (
        payload.shape[0] // attention.MAIN_CHUNK_DWORDS,
        int(payload[0]),
        int(payload[-1]),
        payload_sum,
        payload_hash,
        payload.shape[0],
    )


def main_summary_from_attention() -> tuple[int, int, int, int, int, int]:
    return main_summary_from_payload(attention_payload())


def make_o_record_from_summary(
    group: int,
    row: int,
    summary: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    chunks, first, last, payload_sum, payload_hash, total = summary
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(O_PHASE, group, row)
    record[1:] = (
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
    return record


def make_o_record(group: int, row: int) -> np.ndarray:
    return make_o_record_from_summary(group, row, main_summary_from_attention())


def o_global_compact_from_summary(summary: tuple[int, int, int, int, int, int]) -> np.ndarray:
    columns = []
    for group in range(len(MAIN_COLUMNS)):
        records = [make_o_record_from_summary(group, row, summary) for row in range(ROWS_PER_COLUMN)]
        columns.append(column_compact_from_records(records))
    return global_compact_from_columns(columns)


def o_global_compact() -> np.ndarray:
    return o_global_compact_from_summary(main_summary_from_attention())


def expected_output_from_attention(payload: np.ndarray) -> np.ndarray:
    compact = o_global_compact_from_summary(main_summary_from_payload(payload))
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


def expected_output() -> np.ndarray:
    return expected_output_from_attention(attention_payload())


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
        "main16 emits Q compact records; c1r3 expands them into packed Q[2048]",
        f"KV cache sides stay external: {KV_CACHE_SIDE_DWORDS} dwords each as K0,K1,V0,V1",
        "shim BD slices rebuild K,V,K,V row1 scan layout with per-slot locks",
        "Shape-A/B returns packet2 to main16 O chunks, then O compact reaches c1r2",
        f"host_output={HOST_OUTPUT_DWORDS} dwords",
    ]
