"""CPU reference for the Shape-A/B attention-to-O bridge case."""

from __future__ import annotations

import numpy as np

from contract import ATTENTION_PACKET_DWORDS, MAIN_COLUMNS, MAIN_ROWS, SHAPE_CARRIER_DWORDS

CASE_NAME = "shape-attention-o-bridge"
WINDOW_DWORDS = 512
Q_DWORDS = ATTENTION_PACKET_DWORDS
KV_SIDE_DWORDS = WINDOW_DWORDS * 4
MAIN_CHUNK_DWORDS = 128
SUMMARY_DWORDS = 8
COLUMN_SUMMARY_DWORDS = len(MAIN_ROWS) * SUMMARY_DWORDS
TOTAL_SUMMARY_DWORDS = len(MAIN_COLUMNS) * COLUMN_SUMMARY_DWORDS
PACKET_ID = 2


def make_q_payload() -> np.ndarray:
    return (10_000 + np.arange(Q_DWORDS, dtype=np.int32) * 3).astype(np.int32)


def make_kv_payload(side: int) -> np.ndarray:
    base = 20_000 + side * 10_000
    return (base + np.arange(KV_SIDE_DWORDS, dtype=np.int32) * 5).astype(np.int32)


def q_window(window: int) -> np.ndarray:
    start = window * WINDOW_DWORDS
    return make_q_payload()[start : start + WINDOW_DWORDS]


def _kv_side_and_slot(window: int) -> tuple[int, int]:
    side = 0 if window < 2 else 1
    slot = window if side == 0 else window - 2
    return side, slot


def k_window(window: int) -> np.ndarray:
    side, slot = _kv_side_and_slot(window)
    start = slot * WINDOW_DWORDS * 2
    return make_kv_payload(side)[start : start + WINDOW_DWORDS]


def v_window(window: int) -> np.ndarray:
    side, slot = _kv_side_and_slot(window)
    start = slot * WINDOW_DWORDS * 2 + WINDOW_DWORDS
    return make_kv_payload(side)[start : start + WINDOW_DWORDS]


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


def _summary_hash(payload: np.ndarray) -> tuple[int, int]:
    sum_u32 = 0
    hash_u32 = 0
    for idx, value in enumerate(payload):
        value_u32 = int(value) & 0xFFFF_FFFF
        sum_u32 = (sum_u32 + value_u32) & 0xFFFF_FFFF
        hash_u32 = ((hash_u32 * 16_777_619) ^ ((value_u32 + idx) & 0xFFFF_FFFF)) & 0xFFFF_FFFF
    return int(np.array(sum_u32, dtype=np.uint32).view(np.int32)), int(
        np.array(hash_u32, dtype=np.uint32).view(np.int32)
    )


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
    errors: list[str] = []
    if got.shape != expected.shape:
        return [f"shape mismatch: {got.shape} != {expected.shape}"]
    mismatch = np.where(got != expected)[0]
    for idx in mismatch[:32]:
        errors.append(f"out[{idx}]: expected={int(expected[idx])} got={int(got[idx])}")
    if mismatch.size > 32:
        errors.append(f"{mismatch.size - 32} additional mismatches")
    return errors


def route_summary() -> list[str]:
    return [
        f"case={CASE_NAME}",
        f"q={Q_DWORDS} dwords -> c6r1 -> 4x{WINDOW_DWORDS} Shape-A windows",
        f"kv_left/right={KV_SIDE_DWORDS} dwords each -> c0r1/c7r1 split",
        f"carrier=4x{SHAPE_CARRIER_DWORDS} dwords Shape-A -> Shape-B",
        f"return=4x{WINDOW_DWORDS} dwords Shape-B -> c6r1 -> packet{PACKET_ID}",
        f"main_chunks={ATTENTION_PACKET_DWORDS // MAIN_CHUNK_DWORDS}",
    ]
