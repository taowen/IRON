"""CPU reference for the main16 -> row1/c1r1 -> c6r2 FFN bridge."""

from __future__ import annotations

import numpy as np

from contract import (
    C6R2_HALF_DWORDS,
    C6R2_INPUT_DWORDS,
    COMPACT_PACKET_DWORDS,
    MAIN_COLUMNS,
    MAIN_ROWS,
    RECORD_DWORDS,
    RECORD_PAYLOAD_DWORDS,
    ROWS_PER_COLUMN,
)

CASE_NAME = "ffn-upgate-c6r2-bridge"
UP_PHASE = 4
GATE_PHASE = 5
MAIN_RECORD_DWORDS = RECORD_DWORDS * 2
SWIGLU_OUTPUT_DWORDS = C6R2_HALF_DWORDS
MAIN_PACKET_BASE = 16
COLUMN_PACKET_BASE = 4
GLOBAL_PACKET_ID = 8
SIGMOID_Q15_ONE = 32768
SIGMOID_Q15_HALF = SIGMOID_Q15_ONE // 2
SIGMOID_LINEAR_LIMIT = 2048
SIGMOID_LINEAR_SLOPE = 8
SWIGLU_PRODUCT_SCALE = 16384


def main_packet(group: int, row: int) -> int:
    return MAIN_PACKET_BASE + group * ROWS_PER_COLUMN + row


def column_packet(group: int) -> int:
    return COLUMN_PACKET_BASE + group


def record_header(phase: int, group: int, row: int) -> int:
    return (phase << 24) | (group << 16) | (row << 8) | 0x5A


def payload_value(phase: int, group: int, row: int, lane: int) -> int:
    return phase * 4096 + group * 1024 + row * 256 + lane


def make_record(phase: int, group: int, row: int) -> np.ndarray:
    record = np.empty(RECORD_DWORDS, dtype=np.int32)
    record[0] = record_header(phase, group, row)
    for lane in range(RECORD_PAYLOAD_DWORDS):
        record[1 + lane] = payload_value(phase, group, row, lane)
    return record


def make_main_records(group: int, row: int) -> np.ndarray:
    records = np.empty(MAIN_RECORD_DWORDS, dtype=np.int32)
    records[:RECORD_DWORDS] = make_record(UP_PHASE, group, row)
    records[RECORD_DWORDS:] = make_record(GATE_PHASE, group, row)
    return records


def column_compact(phase: int, group: int) -> np.ndarray:
    compact = np.empty(
        RECORD_DWORDS + (ROWS_PER_COLUMN - 1) * RECORD_PAYLOAD_DWORDS,
        dtype=np.int32,
    )
    offset = 0
    for row in range(ROWS_PER_COLUMN):
        record = make_record(phase, group, row)
        if row == 0:
            segment = record
        else:
            segment = record[1:]
        compact[offset : offset + segment.shape[0]] = segment
        offset += segment.shape[0]
    return compact


def global_compact(phase: int) -> np.ndarray:
    compact = np.empty(COMPACT_PACKET_DWORDS, dtype=np.int32)
    offset = 0
    for group in range(len(MAIN_COLUMNS)):
        column = column_compact(phase, group)
        if group == 0:
            segment = column
        else:
            segment = column[1:]
        compact[offset : offset + segment.shape[0]] = segment
        offset += segment.shape[0]
    return compact


def expected_c6r2_input() -> np.ndarray:
    up = global_compact(UP_PHASE)[1:]
    gate = global_compact(GATE_PHASE)[1:]
    if up.shape[0] != C6R2_HALF_DWORDS or gate.shape[0] != C6R2_HALF_DWORDS:
        raise RuntimeError(f"bad c6r2 halves: {up.shape[0]}/{gate.shape[0]}")
    return np.concatenate((up, gate)).astype(np.int32)


def _trunc_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        return 0
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


def _clamp_s16(value: int) -> int:
    return max(-32768, min(32767, value))


def _unpack_s16_word(value: int, lane: int) -> int:
    word = int(value) & 0xFFFF_FFFF
    raw = (word >> 16) & 0xFFFF if lane else word & 0xFFFF
    return raw - 0x10000 if raw & 0x8000 else raw


def _pack_s16_pair(low: int, high: int) -> np.int32:
    low_u16 = _clamp_s16(low) & 0xFFFF
    high_u16 = _clamp_s16(high) & 0xFFFF
    return np.array(low_u16 | (high_u16 << 16), dtype=np.uint32).view(np.int32)


def _sigmoid_q15(gate: int) -> int:
    if gate <= -SIGMOID_LINEAR_LIMIT:
        return 0
    if gate >= SIGMOID_LINEAR_LIMIT:
        return SIGMOID_Q15_ONE - 1
    return SIGMOID_Q15_HALF + gate * SIGMOID_LINEAR_SLOPE


def fixed_swiglu_lane(up: int, gate: int) -> int:
    silu_gate = _trunc_div(gate * _sigmoid_q15(gate), SIGMOID_Q15_ONE)
    return _clamp_s16(_trunc_div(up * silu_gate, SWIGLU_PRODUCT_SCALE))


def _slice_adjust(value: int, slice_index: int, lane: int) -> int:
    if slice_index == 0:
        return value
    scale = 256 + slice_index
    delta = ((slice_index * 13 + lane * 3) % 17) - 8
    return _clamp_s16(_trunc_div(value * scale, 256) + delta)


def fixed_swiglu_slice_output(values: np.ndarray, slice_index: int) -> np.ndarray:
    if values.shape[0] != C6R2_INPUT_DWORDS:
        raise RuntimeError(f"bad swiglu input: {values.shape[0]}")
    output = np.empty(SWIGLU_OUTPUT_DWORDS, dtype=np.int32)
    for idx in range(SWIGLU_OUTPUT_DWORDS):
        up_word = int(values[idx])
        gate_word = int(values[C6R2_HALF_DWORDS + idx])
        low = fixed_swiglu_lane(_unpack_s16_word(up_word, 0), _unpack_s16_word(gate_word, 0))
        high = fixed_swiglu_lane(_unpack_s16_word(up_word, 1), _unpack_s16_word(gate_word, 1))
        output[idx] = _pack_s16_pair(
            _slice_adjust(low, slice_index, idx * 2),
            _slice_adjust(high, slice_index, idx * 2 + 1),
        )
    return output


def fixed_swiglu_output(values: np.ndarray) -> np.ndarray:
    return fixed_swiglu_slice_output(values, 0)


def expected_output() -> np.ndarray:
    return fixed_swiglu_output(expected_c6r2_input())


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
        f"main_records={len(MAIN_COLUMNS) * len(MAIN_ROWS)}x{MAIN_RECORD_DWORDS} dwords",
        "column_compact=4x65 dwords per phase",
        f"global_compact={COMPACT_PACKET_DWORDS} dwords per phase",
        f"c6r2_input={C6R2_INPUT_DWORDS} dwords, fixed-point SiLU(gate)*up",
        f"c6r2_output={SWIGLU_OUTPUT_DWORDS} dwords",
    ]
