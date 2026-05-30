"""CPU reference helpers for one rounded 16-token attention block."""

from __future__ import annotations

import numpy as np

from contract import MAIN_COLUMNS, MAIN_ROWS, SHAPE_CARRIER_DWORDS
from qkv_compact_reference import (
    MAIN_CHUNK_DWORDS,
    PACKET_ID_ATTENTION,
    Q_DWORDS,
    SUMMARY_DWORDS,
    WINDOW_DWORDS,
)

CASE_NAME = "attention-block-reference"
PACKET_ID = PACKET_ID_ATTENTION
HEADS_PER_WINDOW = 8
KV_HEADS_PER_WINDOW = 2
GQA_RATIO = HEADS_PER_WINDOW // KV_HEADS_PER_WINDOW
CONTEXT = 16
HEAD_DIM = 128
K_WINDOW_DWORDS = KV_HEADS_PER_WINDOW * CONTEXT * HEAD_DIM // 2
V_WINDOW_DWORDS = K_WINDOW_DWORDS
KV_SLOT_DWORDS = K_WINDOW_DWORDS + V_WINDOW_DWORDS
KV_SIDE_DWORDS = KV_SLOT_DWORDS * 2
OUTPUT_DWORDS = HEADS_PER_WINDOW * HEAD_DIM // 2
WEIGHT_DWORDS = HEADS_PER_WINDOW * CONTEXT // 2
SCALAR_DWORDS = HEADS_PER_WINDOW * 2
COLUMN_SUMMARY_DWORDS = len(MAIN_ROWS) * SUMMARY_DWORDS
TOTAL_SUMMARY_DWORDS = len(MAIN_COLUMNS) * COLUMN_SUMMARY_DWORDS
SOFTMAX_SCALE = 4096

# Q12 lookup for round(4096 * exp(-delta / 8)); score delta is integer dot/head_dim.
SOFTMAX_EXP_DELTA_Q12 = (
    4096, 3615, 3190, 2815, 2484, 2192, 1935, 1707,
    1507, 1330, 1174, 1036, 914, 807, 712, 628,
    554, 489, 432, 381, 336, 297, 262, 231,
    204, 180, 159, 140, 124, 109, 96, 85,
    75, 66, 58, 52, 46, 40, 35, 31,
    28, 24, 21, 19, 17, 15, 13, 12,
    10, 9, 8, 7, 6, 5, 5, 4,
    4, 3, 3, 3, 2, 2, 2, 2,
    1,
)


def _u32_to_i32(value: int) -> int:
    return int(np.array(value & 0xFFFF_FFFF, dtype=np.uint32).view(np.int32))


def _clamp_s16(value: int) -> int:
    if value < -32768:
        return -32768
    if value > 32767:
        return 32767
    return value


def _pack_s16_pair(low: int, high: int) -> np.int32:
    low_u16 = _clamp_s16(low) & 0xFFFF
    high_u16 = _clamp_s16(high) & 0xFFFF
    return np.array(low_u16 | (high_u16 << 16), dtype=np.uint32).view(np.int32)


def _unpack_s16(payload: np.ndarray, lane: int) -> int:
    word = int(payload[lane // 2]) & 0xFFFF_FFFF
    raw = (word >> 16) & 0xFFFF if lane & 1 else word & 0xFFFF
    return raw - 0x10000 if raw & 0x8000 else raw


def _pack_lanes(lanes: np.ndarray) -> np.ndarray:
    output = np.empty(lanes.shape[0] // 2, dtype=np.int32)
    for idx in range(output.shape[0]):
        output[idx] = _pack_s16_pair(int(lanes[idx * 2]), int(lanes[idx * 2 + 1]))
    return output


def _trunc_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        return 0
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


def _softmax_weight(delta: int) -> int:
    if delta <= 0:
        return SOFTMAX_SCALE
    if delta < len(SOFTMAX_EXP_DELTA_Q12):
        return SOFTMAX_EXP_DELTA_Q12[delta]
    return 1


def make_q_payload() -> np.ndarray:
    lanes = np.zeros(Q_DWORDS * 2, dtype=np.int32)
    for window in range(4):
        for head in range(HEADS_PER_WINDOW):
            global_head = window * HEADS_PER_WINDOW + head
            for dim in range(HEAD_DIM):
                lane = window * WINDOW_DWORDS * 2 + head * HEAD_DIM + dim
                lanes[lane] = ((global_head + 1) * 7 + dim * 5 + window * 3) % 31 - 15
    return _pack_lanes(lanes)


def make_kv_payload(side: int) -> np.ndarray:
    lanes = np.zeros(KV_SIDE_DWORDS * 2, dtype=np.int32)
    for slot in range(2):
        window = side * 2 + slot
        k_lane_base = slot * KV_SLOT_DWORDS * 2
        v_lane_base = k_lane_base + K_WINDOW_DWORDS * 2
        for kv_head in range(KV_HEADS_PER_WINDOW):
            global_kv_head = window * KV_HEADS_PER_WINDOW + kv_head
            head_base = kv_head * CONTEXT * HEAD_DIM
            for token in range(CONTEXT):
                token_base = head_base + token * HEAD_DIM
                for dim in range(HEAD_DIM):
                    lane = token_base + dim
                    lanes[k_lane_base + lane] = (
                        (global_kv_head + 3) * 9 + token * 5 + dim * 3
                    ) % 31 - 15
                    lanes[v_lane_base + lane] = (
                        (global_kv_head + 5) * 7 + token * 11 + dim * 2
                    ) % 63 - 31
    return _pack_lanes(lanes)


def q_window(window: int) -> np.ndarray:
    start = window * WINDOW_DWORDS
    return make_q_payload()[start : start + WINDOW_DWORDS]


def _side_and_slot(window: int) -> tuple[int, int]:
    side = 0 if window < 2 else 1
    slot = window if side == 0 else window - 2
    return side, slot


def k_window(window: int) -> np.ndarray:
    side, slot = _side_and_slot(window)
    start = slot * KV_SLOT_DWORDS
    return make_kv_payload(side)[start : start + K_WINDOW_DWORDS]


def v_window(window: int) -> np.ndarray:
    side, slot = _side_and_slot(window)
    start = slot * KV_SLOT_DWORDS + K_WINDOW_DWORDS
    return make_kv_payload(side)[start : start + V_WINDOW_DWORDS]


def _score(q: np.ndarray, k: np.ndarray, q_head: int, token: int) -> int:
    kv_head = q_head // GQA_RATIO
    total = 0
    q_base = q_head * HEAD_DIM
    k_base = kv_head * CONTEXT * HEAD_DIM + token * HEAD_DIM
    for dim in range(HEAD_DIM):
        total += _unpack_s16(q, q_base + dim) * _unpack_s16(k, k_base + dim)
    return _trunc_div(total, HEAD_DIM)


def _pack_weight_pair(low: int, high: int) -> np.int32:
    return np.array((low & 0xFFFF) | ((high & 0xFFFF) << 16), dtype=np.uint32).view(np.int32)


def _unpack_weight(carrier: np.ndarray, head: int, token: int) -> int:
    lane = head * CONTEXT + token
    word = int(carrier[lane // 2]) & 0xFFFF_FFFF
    return (word >> 16) & 0xFFFF if lane & 1 else word & 0xFFFF


def make_carrier(window: int) -> np.ndarray:
    q = q_window(window)
    k = k_window(window)
    carrier = np.zeros(SHAPE_CARRIER_DWORDS, dtype=np.int32)
    for q_head in range(HEADS_PER_WINDOW):
        scores = [_score(q, k, q_head, token) for token in range(CONTEXT)]
        running_max = max(scores)
        weights = [_softmax_weight(running_max - score) for score in scores]
        for token in range(0, CONTEXT, 2):
            word = (q_head * CONTEXT + token) // 2
            carrier[word] = _pack_weight_pair(weights[token], weights[token + 1])
        scalar = WEIGHT_DWORDS + q_head * 2
        carrier[scalar] = running_max
        carrier[scalar + 1] = sum(weights)
    return carrier


def return_window(window: int) -> np.ndarray:
    carrier = make_carrier(window)
    v = v_window(window)
    lanes = np.zeros(OUTPUT_DWORDS * 2, dtype=np.int32)
    for q_head in range(HEADS_PER_WINDOW):
        kv_head = q_head // GQA_RATIO
        weight_sum = int(carrier[WEIGHT_DWORDS + q_head * 2 + 1])
        for dim in range(HEAD_DIM):
            total = 0
            for token in range(CONTEXT):
                weight = _unpack_weight(carrier, q_head, token)
                v_lane = kv_head * CONTEXT * HEAD_DIM + token * HEAD_DIM + dim
                total += weight * _unpack_s16(v, v_lane)
            lanes[q_head * HEAD_DIM + dim] = _trunc_div(total, weight_sum)
    return _pack_lanes(lanes)


def attention_payload() -> np.ndarray:
    return np.concatenate([return_window(window) for window in range(4)]).astype(np.int32)


def _summary_hash(payload: np.ndarray) -> tuple[int, int]:
    sum_u32 = 0
    hash_u32 = 0
    for idx, value in enumerate(payload):
        value_u32 = int(value) & 0xFFFF_FFFF
        sum_u32 = (sum_u32 + value_u32) & 0xFFFF_FFFF
        hash_u32 = ((hash_u32 * 16_777_619) ^ ((value_u32 + idx) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
    return _u32_to_i32(sum_u32), _u32_to_i32(hash_u32)


def expected_output() -> np.ndarray:
    payload = attention_payload()
    payload_sum, payload_hash = _summary_hash(payload)
    output = np.empty(TOTAL_SUMMARY_DWORDS, dtype=np.int32)
    for group in range(len(MAIN_COLUMNS)):
        for row in range(len(MAIN_ROWS)):
            start = group * COLUMN_SUMMARY_DWORDS + row * SUMMARY_DWORDS
            output[start : start + SUMMARY_DWORDS] = (
                group,
                row,
                payload.shape[0] // MAIN_CHUNK_DWORDS,
                int(payload[0]),
                int(payload[-1]),
                payload_sum,
                payload_hash,
                payload.shape[0],
            )
    return output


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
        f"Shape-A: {HEADS_PER_WINDOW} Q heads x {HEAD_DIM} dim, "
        f"{KV_HEADS_PER_WINDOW} KV heads x {CONTEXT} tokens",
        f"carrier={WEIGHT_DWORDS} Q12 exp weight dwords + {SCALAR_DWORDS} int32 scalar dwords",
        f"q={Q_DWORDS} dwords, kv_left/right={KV_SIDE_DWORDS} dwords each",
        f"return=4x{OUTPUT_DWORDS} dwords -> packet{PACKET_ID} -> main16 summaries",
    ]
