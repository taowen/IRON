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


def expected_output() -> np.ndarray:
    c6r2_input = expected_c6r2_input()
    if c6r2_input.shape[0] != C6R2_INPUT_DWORDS:
        raise RuntimeError(f"bad c6r2 input: {c6r2_input.shape[0]}")
    output = np.empty(SWIGLU_OUTPUT_DWORDS, dtype=np.int32)
    for idx in range(SWIGLU_OUTPUT_DWORDS):
        output[idx] = (int(c6r2_input[idx]) << 16) | int(c6r2_input[C6R2_HALF_DWORDS + idx])
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
        f"main_records={len(MAIN_COLUMNS) * len(MAIN_ROWS)}x{MAIN_RECORD_DWORDS} dwords",
        "column_compact=4x65 dwords per phase",
        f"global_compact={COMPACT_PACKET_DWORDS} dwords per phase",
        f"c6r2_input={C6R2_INPUT_DWORDS} dwords, low=up high=gate",
        f"c6r2_output={SWIGLU_OUTPUT_DWORDS} dwords",
    ]
